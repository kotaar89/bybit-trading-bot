"""
Модуль бэктестинга v2.3 для Bybit USDT Futures.
Включает:
- Единый MTF-фильтр с live-ботом (сравнение закрытой 4H свечи HTF_CLOSE с HTF_EMA)
- Единый MarketRegimeFilter по закрытым 4H свечам бенчмарка (без NaN-пропусков)
- Обязательный прогрев индикаторов (>= 250 свечей на длинной истории)
- Раздельные сигналы выхода (exit_signal) и входа (signal)
- Реальный исторический Funding Rate (с фоллбэком на FUNDING_RATE_DEFAULT) и Maker fee на лимитном TP1
- Внутрисвечное обновление BE/Trailing без внутрибарового Lookahead Bias
- Переоценку (MTM) открытых позиций на конце истории
- Группировку ног TP1 + TP2/SL в единые позиции (честный Position Win Rate)
- Честный Daily Sharpe Ratio по дневной кривой капитала и ограничение Profit Factor
- Синхронный портфельный симулятор (MAX_OPEN_POSITIONS, RISK_PER_TRADE_PCT, POSITION_SIZE_PCT, компаундинг)
"""

import sys
import pandas as pd
import numpy as np
from datetime import datetime, timezone

import config
from config import (
    SYMBOLS, SYMBOL, INTERVAL, HIGHER_INTERVAL,
    ST_PERIOD, ST_MULTIPLIER, ST_CONFIRM_EMA,
    FAST_MA, SLOW_MA, RSI_PERIOD,
    RSI_LONG_MIN, RSI_LONG_MAX, RSI_SHORT_MIN, RSI_SHORT_MAX,
    VOLUME_FILTER, USE_MTF_FILTER, HTF_EMA_PERIOD,
    USE_ATR_STOPS, ATR_PERIOD, ATR_SL_MULT, ATR_TP_MULT,
    STOP_LOSS_PCT, TAKE_PROFIT_PCT,
    USE_BREAKEVEN, BREAKEVEN_TRIGGER, BREAKEVEN_R_MULT, BREAKEVEN_OFFSET,
    USE_TRAILING_STOP, TRAILING_STOP_TRIGGER, TRAILING_STOP_R_MULT, TRAILING_STOP_ATR_MULT, TRAILING_STOP_FALLBACK,
    USE_PARTIAL_TP, USE_EXCHANGE_TP1, PARTIAL_TP_R_MULT, PARTIAL_TP_RATIO, PARTIAL_TP_MOVE_BE,
    USE_MARKET_REGIME_FILTER, BENCHMARK_SYMBOL, BENCHMARK_INTERVAL,
    BENCHMARK_EMA_PERIOD, BENCHMARK_RSI_THRESHOLD, BENCHMARK_DUMP_PCT, BENCHMARK_DUMP_CANDLES,
    LEVERAGE, ACTIVE_STRATEGY, MAX_OPEN_POSITIONS, POSITION_SIZE_PCT,
    USE_RISK_BASED_SIZING, RISK_PER_TRADE_PCT,
    SIGNAL_LOOKBACK_CANDLES, BENCHMARK_BULL_RSI_THRESHOLD, BENCHMARK_PUMP_PCT,
    USE_ADX_FILTER, ADX_PERIOD, ADX_THRESHOLD
)
from data_loader import load_or_download, load_or_download_funding, INTERVAL_MINUTES
from strategies import get_strategy
from risk_manager import PositionRiskState, MarketRegimeFilter

# Базовые комиссии Bybit VIP0
TAKER_FEE = 0.00055   # 0.055%
MAKER_FEE = 0.00020   # 0.020% (лимитный reduceOnly TP1)
BASE_SLIPPAGE = 0.0002  # 0.02% базовое проскальзывание
FUNDING_RATE_DEFAULT = 0.0001  # 0.01% за 8-часовой интервал (fallback)
MAX_PROFIT_FACTOR = 10.0


def calculate_dynamic_slippage(atr: float, price: float) -> float:
    """Рассчитывает динамическое проскальзывание, пропорциональное волатильности (ATR)."""
    if price <= 0 or atr <= 0:
        return BASE_SLIPPAGE
    volatility_slippage = (atr / price) * 0.05
    return float(max(BASE_SLIPPAGE, min(volatility_slippage, 0.003)))


def count_funding_events(entry_time: pd.Timestamp, exit_time: pd.Timestamp) -> int:
    """Подсчитывает количество 8-часовых клирингов Funding Rate (00:00, 08:00, 16:00 UTC)."""
    if pd.isna(entry_time) or pd.isna(exit_time) or exit_time <= entry_time:
        return 0
    try:
        t0 = entry_time.floor("8h")
        t1 = exit_time.ceil("8h")
        grid = pd.date_range(start=t0, end=t1, freq="8h")
        events = [t for t in grid if entry_time < t <= exit_time]
        return len(events)
    except Exception:
        hours = (exit_time - entry_time).total_seconds() / 3600.0
        return int(max(0.0, hours) // 8)


def compute_funding_cost_pct(
    entry_time: pd.Timestamp,
    exit_time: pd.Timestamp,
    is_long: bool,
    leverage: float,
    funding_df: pd.DataFrame | None = None
) -> float:
    """
    Рассчитывает фандинг в % к марже.
    Если передан funding_df и покрывает период сделки, суммирует реальные ставки 8h клирингов;
    иначе использует консервативный FUNDING_RATE_DEFAULT.
    """
    if pd.isna(entry_time) or pd.isna(exit_time) or exit_time <= entry_time:
        return 0.0
    direction = 1.0 if is_long else -1.0
    if funding_df is not None and not funding_df.empty and "timestamp" in funding_df.columns:
        try:
            f_min = funding_df["timestamp"].min()
            f_max = funding_df["timestamp"].max()
            if f_min <= exit_time and f_max >= entry_time:
                mask = (funding_df["timestamp"] > entry_time) & (funding_df["timestamp"] <= exit_time)
                total_rate = float(funding_df.loc[mask, "fundingRate"].sum())
                return total_rate * leverage * 100.0 * direction
        except Exception:
            pass
    cnt = count_funding_events(entry_time, exit_time)
    return cnt * FUNDING_RATE_DEFAULT * leverage * 100.0 * direction


def align_htf_to_ltf(
    df: pd.DataFrame,
    df_htf: pd.DataFrame,
    interval: str,
    higher_interval: str,
    htf_ema_period: int
) -> pd.DataFrame:
    """
    Привязывает закрытую свечу старшего ТФ (HTF_CLOSE, HTF_EMA) к рабочему ТФ строго по времени закрытия
    без Lookahead Bias и полностью идентично live get_mtf_trend(df_htf).
    """
    res = df.copy()
    if df_htf is None or df_htf.empty or len(df_htf) < 30:
        res["HTF_EMA"] = np.nan
        res["HTF_CLOSE"] = np.nan
        return res

    htf = df_htf.copy().sort_values("timestamp").reset_index(drop=True)
    htf["timestamp"] = pd.to_datetime(htf["timestamp"]).astype("datetime64[ms]")
    span = min(htf_ema_period, len(htf)) if len(htf) < htf_ema_period else htf_ema_period
    htf["HTF_EMA"] = htf["close"].ewm(span=span, adjust=False).mean()

    ltf_min = INTERVAL_MINUTES.get(str(interval), 60)
    htf_min = INTERVAL_MINUTES.get(str(higher_interval), 240)

    res["timestamp"] = pd.to_datetime(res["timestamp"]).astype("datetime64[ms]")
    res["_close_ts"] = res["timestamp"] + pd.Timedelta(minutes=ltf_min)

    htf_sub = htf[["timestamp", "close", "HTF_EMA"]].copy()
    htf_sub["_close_ts"] = htf_sub["timestamp"] + pd.Timedelta(minutes=htf_min)
    htf_sub = htf_sub.rename(columns={"close": "HTF_CLOSE"}).drop(columns=["timestamp"]).sort_values("_close_ts")

    merged = pd.merge_asof(
        res.sort_values("_close_ts"),
        htf_sub,
        on="_close_ts",
        direction="backward"
    )
    return merged.drop(columns=["_close_ts"]).sort_values("timestamp").reset_index(drop=True)


def align_benchmark_to_ltf(
    df: pd.DataFrame,
    benchmark_df: pd.DataFrame,
    interval: str,
    benchmark_interval: str
) -> pd.DataFrame:
    """
    Привязывает закрытые 4H бары бенчмарка BTC к 1H свечам строго по времени закрытия.
    Исключает Lookahead Bias и NaN -> True утечку.
    """
    res = df.copy()
    if benchmark_df is None or benchmark_df.empty:
        return res

    bench = benchmark_df.copy().sort_values("timestamp").reset_index(drop=True)
    bench["timestamp"] = pd.to_datetime(bench["timestamp"]).astype("datetime64[ms]")
    if "BENCH_LONG_ALLOWED" not in bench.columns:
        bench = MarketRegimeFilter.add_indicators(
            bench,
            ema_period=BENCHMARK_EMA_PERIOD,
            dump_lookback_candles=BENCHMARK_DUMP_CANDLES,
            dump_pct_threshold=BENCHMARK_DUMP_PCT,
            pump_pct_threshold=BENCHMARK_PUMP_PCT,
            rsi_threshold=BENCHMARK_RSI_THRESHOLD,
            bull_rsi_threshold=BENCHMARK_BULL_RSI_THRESHOLD
        )

    ltf_min = INTERVAL_MINUTES.get(str(interval), 60)
    bench_min = INTERVAL_MINUTES.get(str(benchmark_interval), 240)

    res["timestamp"] = pd.to_datetime(res["timestamp"]).astype("datetime64[ms]")
    res["_close_ts"] = res["timestamp"] + pd.Timedelta(minutes=ltf_min)

    bench_cols = [
        "BENCH_LONG_ALLOWED", "BENCH_SHORT_ALLOWED",
        "BENCH_SELF_LONG_ALLOWED", "BENCH_SELF_SHORT_ALLOWED"
    ]
    avail_cols = [c for c in bench_cols if c in bench.columns]
    bench_sub = bench[["timestamp"] + avail_cols].copy()
    bench_sub["_close_ts"] = bench_sub["timestamp"] + pd.Timedelta(minutes=bench_min)
    bench_sub = bench_sub.drop(columns=["timestamp"]).sort_values("_close_ts")

    merged = pd.merge_asof(
        res.sort_values("_close_ts"),
        bench_sub,
        on="_close_ts",
        direction="backward"
    )
    return merged.drop(columns=["_close_ts"]).sort_values("timestamp").reset_index(drop=True)


def compute_warmup_bars(strategy, df_len: int) -> int:
    """Определяет число стартовых баров для корректного прогрева индикаторов."""
    ind_warmup = max(
        getattr(strategy, "slow_ma", 0),
        getattr(strategy, "confirm_ema_period", 0),
        getattr(strategy, "st_period", 0) * 2,
        getattr(strategy, "lookback_period", 0),
        getattr(strategy, "bb_period", 0),
        getattr(strategy, "atr_period", 14) * 2,
        getattr(strategy, "adx_period", 14) * 2,
        50,
    )
    if df_len >= 350:
        return min(max(ind_warmup, 250), df_len - 50)
    return min(ind_warmup, max(0, df_len - 20))


def _calc_position_margin_frac(
    entry_price: float,
    sl_dist: float,
    use_risk_based: bool = USE_RISK_BASED_SIZING,
    risk_pct: float = RISK_PER_TRADE_PCT,
    cap_pct: float = POSITION_SIZE_PCT,
    leverage: float = LEVERAGE,
) -> float:
    """Доля текущего equity, выделяемая под маржу одной позиции."""
    if not use_risk_based or sl_dist <= 0 or entry_price <= 0 or leverage <= 0:
        return cap_pct
    sl_pct = sl_dist / entry_price
    if sl_pct <= 0:
        return cap_pct
    notional_frac = risk_pct / sl_pct
    margin_frac = notional_frac / leverage
    return float(min(cap_pct, max(0.01, margin_frac)))


def _compute_daily_metrics(
    equity_timeline: list[tuple[pd.Timestamp, float]],
    initial_equity: float
) -> tuple[float, float, float]:
    """
    Вычисляет итоговый возврат (%), MaxDD (%) и годовой Daily Sharpe Ratio
    по дневной кривой капитала (учитывает дни без сделок как 0% доходности).
    """
    if not equity_timeline or initial_equity <= 0:
        return 0.0, 0.0, 0.0

    edf = pd.DataFrame(equity_timeline, columns=["timestamp", "equity"])
    edf["timestamp"] = pd.to_datetime(edf["timestamp"])
    edf = edf.sort_values("timestamp")
    edf["date"] = edf["timestamp"].dt.floor("D")
    daily = edf.groupby("date")["equity"].last()

    full_idx = pd.date_range(daily.index.min(), daily.index.max(), freq="D")
    daily = daily.reindex(full_idx).ffill().fillna(initial_equity)

    # Расчет MaxDD по полной (внутридневной/часовой) кривой или дневной — берем полную кривую для точности
    eq_series = edf["equity"].astype(float)
    roll_max = eq_series.cummax()
    dd_series = (eq_series - roll_max) / roll_max.replace(0, np.nan) * 100.0
    max_dd = float(dd_series.min()) if not dd_series.empty and not dd_series.isna().all() else 0.0

    daily_ret = daily.pct_change().dropna()
    if len(daily_ret) > 1 and daily_ret.std(ddof=1) > 0:
        daily_sharpe = float(daily_ret.mean() / daily_ret.std(ddof=1) * np.sqrt(365))
    else:
        daily_sharpe = 0.0

    total_ret = ((float(eq_series.iloc[-1]) / initial_equity) - 1.0) * 100.0
    return total_ret, max_dd, daily_sharpe


def _aggregate_position_stats(trades: list[dict]) -> tuple[int, float, float, float, float]:
    """
    Группирует ноги сделок (TP1 + TP2/SL) по pos_id.
    Возвращает: (positions_count, position_win_rate, profit_factor, avg_win, avg_loss).
    """
    if not trades:
        return 0, 0.0, 0.0, 0.0, 0.0
    tdf = pd.DataFrame(trades)
    if "pos_id" in tdf.columns:
        pos_pnl = tdf.groupby("pos_id")["pnl_pct"].sum()
    else:
        pos_pnl = tdf["pnl_pct"]

    total = len(pos_pnl)
    wins = int((pos_pnl > 0).sum())
    losses = total - wins
    win_rate = (wins / total * 100.0) if total > 0 else 0.0
    avg_win = float(pos_pnl[pos_pnl > 0].mean()) if wins > 0 else 0.0
    avg_loss = float(pos_pnl[pos_pnl <= 0].mean()) if losses > 0 else 0.0

    gross_profit = float(pos_pnl[pos_pnl > 0].sum())
    gross_loss = float(abs(pos_pnl[pos_pnl <= 0].sum()))
    if gross_loss > 0:
        pf = min(MAX_PROFIT_FACTOR, gross_profit / gross_loss)
    else:
        pf = MAX_PROFIT_FACTOR if gross_profit > 0 else 0.0
    return total, win_rate, pf, avg_win, avg_loss


def prepare_symbol_dataframe(
    symbol: str,
    days: int = 90,
    force_download: bool = False,
    strategy_params: dict = None,
    benchmark_df: pd.DataFrame = None,
    use_mtf: bool | None = None,
    use_regime: bool | None = None,
) -> tuple[pd.DataFrame, object, int]:
    """Загружает, выравнивает и прогревает данные для бэктеста одной пары."""
    params = strategy_params or {}
    strat_name = params.get("strategy_name", ACTIVE_STRATEGY).strip().upper()
    if use_mtf is None:
        use_mtf = USE_MTF_FILTER if strat_name != "BB_RSI" else bool(params.get("use_mtf_filter", False))

    strategy = get_strategy(
        strat_name,
        st_period=params.get("st_period", ST_PERIOD),
        st_multiplier=params.get("st_multiplier", ST_MULTIPLIER),
        confirm_ema_period=params.get("confirm_ema_period", ST_CONFIRM_EMA),
        confirm_lookback=params.get("confirm_lookback", getattr(config, "ST_CONFIRM_LOOKBACK", 1)),
        use_confirm_ema=params.get("use_confirm_ema", True),
        fast_ma=params.get("fast_ma", FAST_MA),
        slow_ma=params.get("slow_ma", SLOW_MA),
        rsi_period=params.get("rsi_period", RSI_PERIOD),
        rsi_long_min=params.get("rsi_long_min", RSI_LONG_MIN),
        rsi_long_max=params.get("rsi_long_max", RSI_LONG_MAX),
        rsi_short_min=params.get("rsi_short_min", RSI_SHORT_MIN),
        rsi_short_max=params.get("rsi_short_max", RSI_SHORT_MAX),
        volume_filter=params.get("volume_filter", VOLUME_FILTER),
        use_mtf_filter=use_mtf,
        htf_ema_period=params.get("htf_ema_period", HTF_EMA_PERIOD),
        atr_period=params.get("atr_period", ATR_PERIOD),
        cross_lookback=params.get("cross_lookback", SIGNAL_LOOKBACK_CANDLES),
        use_adx_filter=params.get("use_adx_filter", USE_ADX_FILTER),
        adx_period=params.get("adx_period", ADX_PERIOD),
        adx_threshold=params.get("adx_threshold", ADX_THRESHOLD),
    )

    df = load_or_download(symbol, INTERVAL, days, force_download)
    if df.empty or len(df) < 50:
        return pd.DataFrame(), strategy, 0

    if use_mtf:
        htf_days = max(days, 60)
        df_htf = load_or_download(symbol, HIGHER_INTERVAL, htf_days, force_download)
        df = align_htf_to_ltf(
            df, df_htf, INTERVAL, HIGHER_INTERVAL,
            params.get("htf_ema_period", HTF_EMA_PERIOD)
        )
    else:
        df["HTF_EMA"] = np.nan
        df["HTF_CLOSE"] = np.nan

    df = strategy.calculate_indicators(df)

    if use_regime is None:
        use_regime = USE_MARKET_REGIME_FILTER
    if use_regime:
        if benchmark_df is None:
            try:
                benchmark_df = load_or_download(BENCHMARK_SYMBOL, BENCHMARK_INTERVAL, max(days, 60), force_download)
            except Exception:
                benchmark_df = None
        if benchmark_df is not None and not benchmark_df.empty:
            df = align_benchmark_to_ltf(df, benchmark_df, INTERVAL, BENCHMARK_INTERVAL)

    warmup = compute_warmup_bars(strategy, len(df))
    return df, strategy, warmup


def run_backtest_symbol(
    symbol: str,
    days: int = 90,
    force_download: bool = False,
    verbose: bool = True,
    strategy_params: dict = None,
    benchmark_df: pd.DataFrame = None,
    funding_df: pd.DataFrame = None,
    use_real_funding: bool = True,
) -> dict | None:
    """
    Запускает честный бэктест по одной торговой паре с учетом:
    - прогрева индикаторов,
    - live-эквивалентного MTF и MarketRegimeFilter,
    - позиционной группировки (TP1+TP2),
    - MTM закрытия в конце истории,
    - риск-сайзинга и Daily Sharpe Ratio.
    """
    params = strategy_params or {}
    atr_sl_mult = params.get("atr_sl_mult", ATR_SL_MULT)
    atr_tp_mult = params.get("atr_tp_mult", ATR_TP_MULT)
    trailing_atr_mult = params.get("trailing_atr_mult", TRAILING_STOP_ATR_MULT)
    be_r_mult = params.get("breakeven_r_mult", BREAKEVEN_R_MULT)
    tp1_r_mult = params.get("partial_tp_r_mult", PARTIAL_TP_R_MULT)
    trail_r_mult = params.get("trailing_r_mult", TRAILING_STOP_R_MULT)
    use_regime = params.get("use_market_regime_filter", USE_MARKET_REGIME_FILTER)

    df, strategy, warmup = prepare_symbol_dataframe(
        symbol=symbol,
        days=days,
        force_download=force_download,
        strategy_params=params,
        benchmark_df=benchmark_df,
        use_regime=use_regime,
    )
    if df.empty or len(df) <= warmup + 10:
        if verbose:
            print(f"Недостаточно данных для {symbol}")
        return None

    if funding_df is None and use_real_funding:
        try:
            funding_df = load_or_download_funding(symbol, days=days, allow_network=False)
        except Exception:
            funding_df = None

    if verbose:
        print("\n" + "=" * 62)
        print(f"  БЭКТЕСТ v2.3 — {symbol} | TF: {INTERVAL}m | {days} дней (warmup {warmup} баров)")
        print(f"  Стратегия: {strategy.name} | Плечо: x{LEVERAGE} | Risk/Trade: {RISK_PER_TRADE_PCT*100:.1f}%")
        print(f"  Трейлинг: {USE_TRAILING_STOP} | Частичный TP: {USE_PARTIAL_TP} ({tp1_r_mult}R) | Режим рынка: {use_regime}")
        print("=" * 62)

    trades = []
    pos_state: PositionRiskState | None = None
    entry_idx = None
    pos_id = 0
    pos_margin_frac = POSITION_SIZE_PCT
    pos_allocated_margin = 0.0
    equity = 100.0
    initial_equity = 100.0
    equity_timeline = [(df["timestamp"].iloc[warmup], equity)]
    is_benchmark_symbol = (symbol.upper() == BENCHMARK_SYMBOL.upper())
    tp1_fee_rate = MAKER_FEE if USE_EXCHANGE_TP1 else TAKER_FEE

    start_idx = max(1, warmup)
    for i in range(start_idx, len(df)):
        candle = df.iloc[i]
        open_price = float(candle["open"])
        high_price = float(candle["high"])
        low_price = float(candle["low"])
        close_price = float(candle["close"])
        candle_time = candle["timestamp"]

        prev_candle = df.iloc[i - 1]
        raw_signal = int(prev_candle.get("signal", 0))
        exit_sig = int(prev_candle.get("exit_signal", raw_signal))
        atr = float(prev_candle.get("ATR", 0.0))
        if np.isnan(atr) or atr <= 0:
            atr = open_price * 0.015

        # 1. Приоритетный выход / разворот по exit_signal ДО фильтра сентимента
        if exit_sig == 1 and pos_state is not None and not pos_state.is_long:
            exit_slip = calculate_dynamic_slippage(atr, open_price)
            actual_exit = open_price * (1 + exit_slip)
            fee_pct = (pos_state.entry_price + actual_exit) * TAKER_FEE * LEVERAGE / pos_state.entry_price * 100
            pnl_raw = (pos_state.entry_price - actual_exit) / pos_state.entry_price * LEVERAGE * 100
            funding_cost_pct = compute_funding_cost_pct(df["timestamp"].iloc[entry_idx], candle_time, False, LEVERAGE, funding_df)
            weight = pos_state.size / pos_state.initial_size
            net_pnl = (pnl_raw - fee_pct - funding_cost_pct) * weight
            usd_delta = pos_allocated_margin * (net_pnl / 100.0)
            equity = max(0.01, equity + usd_delta)
            trades.append({
                "pos_id": pos_id, "symbol": symbol, "type": "short",
                "entry": pos_state.entry_price, "exit": actual_exit,
                "pnl_pct": net_pnl, "equity_impact_pct": (usd_delta / initial_equity) * 100.0,
                "reason": "Сигнал разворота", "bars": i - entry_idx,
                "entry_time": df["timestamp"].iloc[entry_idx], "exit_time": candle_time,
            })
            pos_state = None
            entry_idx = None

        elif exit_sig == -1 and pos_state is not None and pos_state.is_long:
            exit_slip = calculate_dynamic_slippage(atr, open_price)
            actual_exit = open_price * (1 - exit_slip)
            fee_pct = (pos_state.entry_price + actual_exit) * TAKER_FEE * LEVERAGE / pos_state.entry_price * 100
            pnl_raw = (actual_exit - pos_state.entry_price) / pos_state.entry_price * LEVERAGE * 100
            funding_cost_pct = compute_funding_cost_pct(df["timestamp"].iloc[entry_idx], candle_time, True, LEVERAGE, funding_df)
            weight = pos_state.size / pos_state.initial_size
            net_pnl = (pnl_raw - fee_pct - funding_cost_pct) * weight
            usd_delta = pos_allocated_margin * (net_pnl / 100.0)
            equity = max(0.01, equity + usd_delta)
            trades.append({
                "pos_id": pos_id, "symbol": symbol, "type": "long",
                "entry": pos_state.entry_price, "exit": actual_exit,
                "pnl_pct": net_pnl, "equity_impact_pct": (usd_delta / initial_equity) * 100.0,
                "reason": "Сигнал разворота", "bars": i - entry_idx,
                "entry_time": df["timestamp"].iloc[entry_idx], "exit_time": candle_time,
            })
            pos_state = None
            entry_idx = None

        # 2. Фильтрация входа по бенчмарку (fail-safe: NaN -> False)
        if raw_signal != 0 and use_regime:
            col_name = (
                ("BENCH_SELF_LONG_ALLOWED" if is_benchmark_symbol else "BENCH_LONG_ALLOWED")
                if raw_signal == 1
                else ("BENCH_SELF_SHORT_ALLOWED" if is_benchmark_symbol else "BENCH_SHORT_ALLOWED")
            )
            if col_name not in df.columns:
                raw_signal = 0
            else:
                raw_allowed = prev_candle.get(col_name)
                if pd.isna(raw_allowed) or not bool(raw_allowed):
                    raw_signal = 0

        # 3. Открытие новой позиции на открытии свечи i
        if pos_state is None and raw_signal != 0:
            entry_slip = calculate_dynamic_slippage(atr, open_price)
            side_str = "buy" if raw_signal == 1 else "sell"
            entry_p = open_price * (1 + entry_slip if raw_signal == 1 else 1 - entry_slip)
            stops = strategy.calculate_stops(
                entry_p, side_str, atr_value=atr,
                stop_loss_pct=STOP_LOSS_PCT, take_profit_pct=TAKE_PROFIT_PCT,
                atr_sl_mult=atr_sl_mult, atr_tp_mult=atr_tp_mult,
                use_atr=USE_ATR_STOPS
            )
            pos_id += 1
            pos_margin_frac = _calc_position_margin_frac(entry_p, stops["sl_dist"])
            pos_allocated_margin = equity * pos_margin_frac
            pos_state = PositionRiskState(
                symbol=symbol,
                side="Buy" if raw_signal == 1 else "Sell",
                entry_price=entry_p,
                size=1.0,
                initial_size=1.0,
                stop_loss=stops["sl"],
                take_profit=stops["tp"],
                atr_value=atr,
                r_mult=tp1_r_mult
            )
            entry_idx = i

        # 4. Внутрисвечевая симуляция (приоритет SL -> TP1 -> TP2 -> обновление BE/Trailing на следующий бар)
        if pos_state is not None:
            closed_reason = None
            exit_price = None

            if pos_state.is_long:
                if low_price <= pos_state.stop_loss:
                    exit_price = pos_state.stop_loss
                    closed_reason = "Trailing SL" if pos_state.trailing_active else ("Безубыток" if pos_state.breakeven_applied else "SL")
            else:
                if high_price >= pos_state.stop_loss:
                    exit_price = pos_state.stop_loss
                    closed_reason = "Trailing SL" if pos_state.trailing_active else ("Безубыток" if pos_state.breakeven_applied else "SL")

            if not closed_reason:
                if USE_PARTIAL_TP and not pos_state.tp1_executed:
                    triggered, qty_close, be_sl = pos_state.check_partial_tp(
                        high_price if pos_state.is_long else low_price,
                        ratio=PARTIAL_TP_RATIO,
                        offset_pct=BREAKEVEN_OFFSET
                    )
                    if triggered:
                        tp1_target = pos_state.tp1_price
                        fee_pct = ((pos_state.entry_price * TAKER_FEE) + (tp1_target * tp1_fee_rate)) * LEVERAGE / pos_state.entry_price * 100
                        pnl_raw = (
                            (tp1_target - pos_state.entry_price) / pos_state.entry_price
                            if pos_state.is_long
                            else (pos_state.entry_price - tp1_target) / pos_state.entry_price
                        ) * LEVERAGE * 100
                        funding_cost_pct = compute_funding_cost_pct(df["timestamp"].iloc[entry_idx], candle_time, pos_state.is_long, LEVERAGE, funding_df)
                        net_pnl = (pnl_raw - fee_pct - funding_cost_pct) * PARTIAL_TP_RATIO
                        usd_delta = pos_allocated_margin * (net_pnl / 100.0)
                        equity = max(0.01, equity + usd_delta)
                        trades.append({
                            "pos_id": pos_id, "symbol": symbol,
                            "type": "long" if pos_state.is_long else "short",
                            "entry": pos_state.entry_price, "exit": tp1_target,
                            "pnl_pct": net_pnl, "equity_impact_pct": (usd_delta / initial_equity) * 100.0,
                            "reason": f"TP1 ({tp1_r_mult:g}R)", "bars": i - entry_idx,
                            "entry_time": df["timestamp"].iloc[entry_idx], "exit_time": candle_time,
                        })
                        pos_state.tp1_executed = True
                        pos_state.size -= qty_close
                        if PARTIAL_TP_MOVE_BE:
                            if (pos_state.is_long and be_sl > pos_state.stop_loss) or (not pos_state.is_long and be_sl < pos_state.stop_loss):
                                pos_state.stop_loss = be_sl
                            pos_state.breakeven_applied = True

                if pos_state.is_long and high_price >= pos_state.take_profit:
                    exit_price = pos_state.take_profit
                    closed_reason = "TP2" if pos_state.tp1_executed else "TP"
                elif not pos_state.is_long and low_price <= pos_state.take_profit:
                    exit_price = pos_state.take_profit
                    closed_reason = "TP2" if pos_state.tp1_executed else "TP"

            if closed_reason and exit_price is not None:
                is_tp_limit = closed_reason in ("TP", "TP2")
                exit_slip = 0.0 if is_tp_limit else calculate_dynamic_slippage(atr, exit_price)
                if pos_state.is_long:
                    actual_exit = exit_price * (1 - exit_slip)
                    pnl_raw = (actual_exit - pos_state.entry_price) / pos_state.entry_price * LEVERAGE * 100
                else:
                    actual_exit = exit_price * (1 + exit_slip)
                    pnl_raw = (pos_state.entry_price - actual_exit) / pos_state.entry_price * LEVERAGE * 100

                fee_pct = (pos_state.entry_price + actual_exit) * TAKER_FEE * LEVERAGE / pos_state.entry_price * 100
                funding_cost_pct = compute_funding_cost_pct(df["timestamp"].iloc[entry_idx], candle_time, pos_state.is_long, LEVERAGE, funding_df)
                weight = pos_state.size / pos_state.initial_size
                net_pnl = (pnl_raw - fee_pct - funding_cost_pct) * weight
                usd_delta = pos_allocated_margin * (net_pnl / 100.0)
                equity = max(0.01, equity + usd_delta)
                trades.append({
                    "pos_id": pos_id, "symbol": symbol,
                    "type": "long" if pos_state.is_long else "short",
                    "entry": pos_state.entry_price, "exit": actual_exit,
                    "pnl_pct": net_pnl, "equity_impact_pct": (usd_delta / initial_equity) * 100.0,
                    "reason": closed_reason, "bars": i - entry_idx,
                    "entry_time": df["timestamp"].iloc[entry_idx], "exit_time": candle_time,
                })
                pos_state = None
                entry_idx = None
            else:
                # Внутрисвечной check для следующего бара (без lookahead внутри текущего бара)
                pos_state.update_price_extremes(high_price, low_price)
                intra_fav = high_price if pos_state.is_long else low_price

                if USE_BREAKEVEN and not pos_state.breakeven_applied:
                    be_updated, new_sl = pos_state.check_breakeven(
                        current_price=intra_fav,
                        trigger_pct=BREAKEVEN_TRIGGER,
                        offset_pct=BREAKEVEN_OFFSET,
                        r_mult=be_r_mult
                    )
                    if be_updated:
                        pos_state.stop_loss = new_sl
                        pos_state.breakeven_applied = True

                if USE_TRAILING_STOP:
                    trail_updated, new_sl = pos_state.check_trailing_stop(
                        current_price=intra_fav,
                        high_price=high_price,
                        low_price=low_price,
                        trigger_pct=TRAILING_STOP_TRIGGER,
                        atr_mult=trailing_atr_mult,
                        fallback_pct=TRAILING_STOP_FALLBACK,
                        r_mult=trail_r_mult
                    )
                    if trail_updated:
                        pos_state.stop_loss = new_sl

        # MTM кривая капитала на конец бара
        mtm_equity = equity
        if pos_state is not None:
            mtm_dir = (close_price - pos_state.entry_price) if pos_state.is_long else (pos_state.entry_price - close_price)
            mtm_roi = (mtm_dir / pos_state.entry_price) * LEVERAGE * (pos_state.size / pos_state.initial_size)
            mtm_equity = max(0.01, equity + pos_allocated_margin * mtm_roi)
        equity_timeline.append((candle_time, mtm_equity))

    # Принудительное MTM-закрытие незакрытой позиции на конце истории
    if pos_state is not None and entry_idx is not None:
        last_candle = df.iloc[-1]
        last_close = float(last_candle["close"])
        last_time = last_candle["timestamp"]
        last_atr = float(last_candle.get("ATR", 0.0)) or (last_close * 0.015)
        exit_slip = calculate_dynamic_slippage(last_atr, last_close)
        actual_exit = last_close * (1 - exit_slip if pos_state.is_long else 1 + exit_slip)
        pnl_raw = (
            (actual_exit - pos_state.entry_price) / pos_state.entry_price
            if pos_state.is_long
            else (pos_state.entry_price - actual_exit) / pos_state.entry_price
        ) * LEVERAGE * 100
        fee_pct = (pos_state.entry_price + actual_exit) * TAKER_FEE * LEVERAGE / pos_state.entry_price * 100
        funding_cost_pct = compute_funding_cost_pct(df["timestamp"].iloc[entry_idx], last_time, pos_state.is_long, LEVERAGE, funding_df)
        weight = pos_state.size / pos_state.initial_size
        net_pnl = (pnl_raw - fee_pct - funding_cost_pct) * weight
        usd_delta = pos_allocated_margin * (net_pnl / 100.0)
        equity = max(0.01, equity + usd_delta)
        trades.append({
            "pos_id": pos_id, "symbol": symbol,
            "type": "long" if pos_state.is_long else "short",
            "entry": pos_state.entry_price, "exit": actual_exit,
            "pnl_pct": net_pnl, "equity_impact_pct": (usd_delta / initial_equity) * 100.0,
            "reason": "MTM (Конец периода)", "bars": (len(df) - 1) - entry_idx,
            "entry_time": df["timestamp"].iloc[entry_idx], "exit_time": last_time,
        })
        equity_timeline[-1] = (last_time, equity)

    if not trades:
        if verbose:
            print("Сделок не найдено.")
        return None

    tdf = pd.DataFrame(trades)
    pos_count, win_rate, profit_factor, avg_win, avg_loss = _aggregate_position_stats(trades)
    total_pnl_pct, max_drawdown, sharpe = _compute_daily_metrics(equity_timeline, initial_equity)
    raw_margin_pnl_sum = float(tdf["pnl_pct"].sum())

    if verbose:
        wins_cnt = int(round(win_rate * pos_count / 100.0))
        print(f"Позиций (ног):     {pos_count} ({len(tdf)} ног) | Win: {wins_cnt} | Loss: {pos_count - wins_cnt}")
        print(f"Винрейт (Pos WR):  {win_rate:.1f}%")
        print(f"Profit Factor:     {profit_factor:.2f}")
        print(f"Доходность счёта:  {total_pnl_pct:+.2f}% (сумма ROI маржи: {raw_margin_pnl_sum:+.2f}%)")
        print(f"Макс. просадка:    {max_drawdown:.2f}%")
        print(f"Daily Sharpe:      {sharpe:.2f}")
        print(f"Средняя прибыль:   {avg_win:+.2f}% | Средний убыток: {avg_loss:+.2f}%")
        print("=" * 62 + "\n")

    return {
        "symbol": symbol,
        "trades": pos_count,
        "legs": len(tdf),
        "win_rate": win_rate,
        "profit_factor": profit_factor,
        "pnl_pct": total_pnl_pct,
        "raw_margin_pnl_pct": raw_margin_pnl_sum,
        "max_drawdown": max_drawdown,
        "sharpe_ratio": sharpe,
        "tdf": tdf
    }


def run_portfolio_backtest(
    symbols: list[str] = None,
    days: int = 90,
    force_download: bool = False,
    strategy_params: dict = None,
    max_open_positions: int = None,
    max_same_dir: int | None = None,
    initial_capital: float = 1000.0,
    use_real_funding: bool = True,
    verbose: bool = True,
) -> dict:
    """
    Синхронный портфельный симулятор по корзине пар с общим капиталом,
    лимитом открытых позиций (MAX_OPEN_POSITIONS), риск-сайзингом,
    лимитом однонаправленного риска и компаундингом.
    """
    symbols = symbols or SYMBOLS
    max_pos = max_open_positions if max_open_positions is not None else MAX_OPEN_POSITIONS
    if max_same_dir is None:
        max_same_dir = getattr(config, "MAX_SAME_DIRECTION_POSITIONS", max_pos)
    params = strategy_params or {}
    atr_sl_mult = params.get("atr_sl_mult", ATR_SL_MULT)
    atr_tp_mult = params.get("atr_tp_mult", ATR_TP_MULT)
    trailing_atr_mult = params.get("trailing_atr_mult", TRAILING_STOP_ATR_MULT)
    be_r_mult = params.get("breakeven_r_mult", BREAKEVEN_R_MULT)
    tp1_r_mult = params.get("partial_tp_r_mult", PARTIAL_TP_R_MULT)
    trail_r_mult = params.get("trailing_r_mult", TRAILING_STOP_R_MULT)
    use_regime = params.get("use_market_regime_filter", USE_MARKET_REGIME_FILTER)
    tp1_fee_rate = MAKER_FEE if USE_EXCHANGE_TP1 else TAKER_FEE

    bench_df = None
    if use_regime:
        try:
            bench_df = load_or_download(BENCHMARK_SYMBOL, BENCHMARK_INTERVAL, max(days, 60), force_download)
            if bench_df is not None and not bench_df.empty:
                bench_df = MarketRegimeFilter.add_indicators(
                    bench_df,
                    ema_period=BENCHMARK_EMA_PERIOD,
                    dump_lookback_candles=BENCHMARK_DUMP_CANDLES,
                    dump_pct_threshold=BENCHMARK_DUMP_PCT,
                    pump_pct_threshold=BENCHMARK_PUMP_PCT,
                    rsi_threshold=BENCHMARK_RSI_THRESHOLD,
                    bull_rsi_threshold=BENCHMARK_BULL_RSI_THRESHOLD
                )
        except Exception:
            bench_df = None

    prepared = {}
    funding_map = {}
    bar_maps = {}
    all_timestamps = set()
    common_start = None

    for sym in symbols:
        df, strategy, warmup = prepare_symbol_dataframe(
            symbol=sym, days=days, force_download=force_download,
            strategy_params=params, benchmark_df=bench_df, use_regime=use_regime
        )
        if df.empty or len(df) <= warmup + 5:
            continue
        prepared[sym] = (df, strategy, warmup)
        start_ts = df["timestamp"].iloc[warmup]
        if common_start is None or start_ts > common_start:
            common_start = start_ts
        bar_maps[sym] = {ts: idx for idx, ts in enumerate(df["timestamp"])}
        all_timestamps.update(df["timestamp"].iloc[warmup:].tolist())
        if use_real_funding:
            try:
                funding_map[sym] = load_or_download_funding(sym, days=days, allow_network=False)
            except Exception:
                funding_map[sym] = None

    if not prepared or not all_timestamps:
        return {"symbols": [], "trades": 0, "pnl_pct": 0.0, "max_drawdown": 0.0, "sharpe_ratio": 0.0, "profit_factor": 0.0, "win_rate": 0.0, "tdf": pd.DataFrame()}

    timeline = sorted(ts for ts in all_timestamps if common_start is None or ts >= common_start)
    equity = float(initial_capital)
    open_pos: dict[str, dict] = {}  # sym -> {"risk": PositionRiskState, "margin": float, "entry_time": ts, "pos_id": int}
    trades = []
    pos_id = 0
    equity_timeline = [(timeline[0], equity)]

    for ts in timeline:
        # 1. Выходы по сигналам разворота на открытии бара
        for sym in list(open_pos.keys()):
            idx = bar_maps[sym].get(ts)
            if idx is None or idx < 1:
                continue
            df, _, _ = prepared[sym]
            candle = df.iloc[idx]
            prev = df.iloc[idx - 1]
            open_p = float(candle["open"])
            atr = float(prev.get("ATR", 0.0))
            if np.isnan(atr) or atr <= 0:
                atr = open_p * 0.015
            raw_sig = int(prev.get("signal", 0))
            exit_sig = int(prev.get("exit_signal", raw_sig))

            p_info = open_pos[sym]
            st: PositionRiskState = p_info["risk"]
            if (exit_sig == 1 and not st.is_long) or (exit_sig == -1 and st.is_long):
                slip = calculate_dynamic_slippage(atr, open_p)
                actual_exit = open_p * (1 + slip if not st.is_long else 1 - slip)
                pnl_raw = ((st.entry_price - actual_exit) if not st.is_long else (actual_exit - st.entry_price)) / st.entry_price * LEVERAGE * 100
                fee_pct = (st.entry_price + actual_exit) * TAKER_FEE * LEVERAGE / st.entry_price * 100
                fund_pct = compute_funding_cost_pct(p_info["entry_time"], ts, st.is_long, LEVERAGE, funding_map.get(sym))
                weight = st.size / st.initial_size
                net_pnl = (pnl_raw - fee_pct - fund_pct) * weight
                equity = max(1.0, equity + p_info["margin"] * (net_pnl / 100.0))
                trades.append({
                    "pos_id": p_info["pos_id"], "symbol": sym,
                    "type": "long" if st.is_long else "short",
                    "entry": st.entry_price, "exit": actual_exit,
                    "pnl_pct": net_pnl, "reason": "Сигнал разворота",
                    "entry_time": p_info["entry_time"], "exit_time": ts,
                })
                del open_pos[sym]

        # 2. Открытие новых позиций на открытии бара
        used_margin = sum(p["margin"] * (p["risk"].size / p["risk"].initial_size) for p in open_pos.values())
        free_margin = max(0.0, equity - used_margin)
        for sym, (df, strategy, warmup) in prepared.items():
            if sym in open_pos:
                continue
            if len(open_pos) >= max_pos:
                break
            idx = bar_maps[sym].get(ts)
            if idx is None or idx < max(1, warmup):
                continue
            candle = df.iloc[idx]
            prev = df.iloc[idx - 1]
            raw_sig = int(prev.get("signal", 0))
            if raw_sig == 0:
                continue

            if use_regime:
                is_bench = (sym.upper() == BENCHMARK_SYMBOL.upper())
                col = (
                    ("BENCH_SELF_LONG_ALLOWED" if is_bench else "BENCH_LONG_ALLOWED")
                    if raw_sig == 1
                    else ("BENCH_SELF_SHORT_ALLOWED" if is_bench else "BENCH_SHORT_ALLOWED")
                )
                if col not in df.columns or pd.isna(prev.get(col)) or not bool(prev.get(col)):
                    continue

            # Лимит однонаправленного коррелированного риска
            same_dir_cnt = sum(
                1 for p in open_pos.values()
                if (raw_sig == 1 and p["risk"].is_long) or (raw_sig == -1 and not p["risk"].is_long)
            )
            if same_dir_cnt >= max_same_dir:
                continue

            open_p = float(candle["open"])
            atr = float(prev.get("ATR", 0.0))
            if np.isnan(atr) or atr <= 0:
                atr = open_p * 0.015
            slip = calculate_dynamic_slippage(atr, open_p)
            entry_p = open_p * (1 + slip if raw_sig == 1 else 1 - slip)
            stops = strategy.calculate_stops(
                entry_p, "buy" if raw_sig == 1 else "sell", atr_value=atr,
                stop_loss_pct=STOP_LOSS_PCT, take_profit_pct=TAKE_PROFIT_PCT,
                atr_sl_mult=atr_sl_mult, atr_tp_mult=atr_tp_mult, use_atr=USE_ATR_STOPS
            )
            margin_frac = _calc_position_margin_frac(entry_p, stops["sl_dist"])
            margin = min(free_margin * 0.95, equity * margin_frac)
            if margin < 5.0:
                continue

            pos_id += 1
            open_pos[sym] = {
                "pos_id": pos_id,
                "margin": margin,
                "entry_time": ts,
                "risk": PositionRiskState(
                    symbol=sym, side="Buy" if raw_sig == 1 else "Sell",
                    entry_price=entry_p, size=1.0, initial_size=1.0,
                    stop_loss=stops["sl"], take_profit=stops["tp"],
                    atr_value=atr, r_mult=tp1_r_mult
                )
            }
            free_margin = max(0.0, free_margin - margin)

        # 3. Внутрисвечевая симуляция SL -> TP1 -> TP2 -> BE/Trailing
        for sym in list(open_pos.keys()):
            idx = bar_maps[sym].get(ts)
            if idx is None:
                continue
            df, _, _ = prepared[sym]
            candle = df.iloc[idx]
            high_p = float(candle["high"])
            low_p = float(candle["low"])
            atr = float(candle.get("ATR", 0.0)) or (float(candle["close"]) * 0.015)

            p_info = open_pos[sym]
            st: PositionRiskState = p_info["risk"]
            closed_reason = None
            exit_price = None

            if st.is_long and low_p <= st.stop_loss:
                exit_price = st.stop_loss
                closed_reason = "Trailing SL" if st.trailing_active else ("Безубыток" if st.breakeven_applied else "SL")
            elif not st.is_long and high_p >= st.stop_loss:
                exit_price = st.stop_loss
                closed_reason = "Trailing SL" if st.trailing_active else ("Безубыток" if st.breakeven_applied else "SL")

            if not closed_reason:
                if USE_PARTIAL_TP and not st.tp1_executed:
                    triggered, qty_close, be_sl = st.check_partial_tp(
                        high_p if st.is_long else low_p, ratio=PARTIAL_TP_RATIO, offset_pct=BREAKEVEN_OFFSET
                    )
                    if triggered:
                        tp1_p = st.tp1_price
                        fee_pct = ((st.entry_price * TAKER_FEE) + (tp1_p * tp1_fee_rate)) * LEVERAGE / st.entry_price * 100
                        pnl_raw = ((tp1_p - st.entry_price) if st.is_long else (st.entry_price - tp1_p)) / st.entry_price * LEVERAGE * 100
                        fund_pct = compute_funding_cost_pct(p_info["entry_time"], ts, st.is_long, LEVERAGE, funding_map.get(sym))
                        net_pnl = (pnl_raw - fee_pct - fund_pct) * PARTIAL_TP_RATIO
                        equity = max(1.0, equity + p_info["margin"] * (net_pnl / 100.0))
                        trades.append({
                            "pos_id": p_info["pos_id"], "symbol": sym,
                            "type": "long" if st.is_long else "short",
                            "entry": st.entry_price, "exit": tp1_p,
                            "pnl_pct": net_pnl, "reason": f"TP1 ({tp1_r_mult:g}R)",
                            "entry_time": p_info["entry_time"], "exit_time": ts,
                        })
                        st.tp1_executed = True
                        st.size -= qty_close
                        if PARTIAL_TP_MOVE_BE:
                            if (st.is_long and be_sl > st.stop_loss) or (not st.is_long and be_sl < st.stop_loss):
                                st.stop_loss = be_sl
                            st.breakeven_applied = True

                if st.is_long and high_p >= st.take_profit:
                    exit_price = st.take_profit
                    closed_reason = "TP2" if st.tp1_executed else "TP"
                elif not st.is_long and low_p <= st.take_profit:
                    exit_price = st.take_profit
                    closed_reason = "TP2" if st.tp1_executed else "TP"

            if closed_reason and exit_price is not None:
                is_tp_lim = closed_reason in ("TP", "TP2")
                slip = 0.0 if is_tp_lim else calculate_dynamic_slippage(atr, exit_price)
                actual_exit = exit_price * (1 - slip if st.is_long else 1 + slip)
                pnl_raw = ((actual_exit - st.entry_price) if st.is_long else (st.entry_price - actual_exit)) / st.entry_price * LEVERAGE * 100
                fee_pct = (st.entry_price + actual_exit) * TAKER_FEE * LEVERAGE / st.entry_price * 100
                fund_pct = compute_funding_cost_pct(p_info["entry_time"], ts, st.is_long, LEVERAGE, funding_map.get(sym))
                weight = st.size / st.initial_size
                net_pnl = (pnl_raw - fee_pct - fund_pct) * weight
                equity = max(1.0, equity + p_info["margin"] * (net_pnl / 100.0))
                trades.append({
                    "pos_id": p_info["pos_id"], "symbol": sym,
                    "type": "long" if st.is_long else "short",
                    "entry": st.entry_price, "exit": actual_exit,
                    "pnl_pct": net_pnl, "reason": closed_reason,
                    "entry_time": p_info["entry_time"], "exit_time": ts,
                })
                del open_pos[sym]
            else:
                st.update_price_extremes(high_p, low_p)
                intra_fav = high_p if st.is_long else low_p
                if USE_BREAKEVEN and not st.breakeven_applied:
                    be_upd, new_sl = st.check_breakeven(intra_fav, BREAKEVEN_TRIGGER, BREAKEVEN_OFFSET, be_r_mult)
                    if be_upd:
                        st.stop_loss = new_sl
                        st.breakeven_applied = True
                if USE_TRAILING_STOP:
                    tr_upd, new_sl = st.check_trailing_stop(
                        intra_fav, high_p, low_p, TRAILING_STOP_TRIGGER, trailing_atr_mult, TRAILING_STOP_FALLBACK, trail_r_mult
                    )
                    if tr_upd:
                        st.stop_loss = new_sl

        # MTM портфеля на конце бара ts
        mtm_eq = equity
        for sym, p_info in open_pos.items():
            idx = bar_maps[sym].get(ts)
            if idx is None:
                continue
            df, _, _ = prepared[sym]
            c_p = float(df.iloc[idx]["close"])
            st = p_info["risk"]
            diff = (c_p - st.entry_price) if st.is_long else (st.entry_price - c_p)
            mtm_eq += p_info["margin"] * (diff / st.entry_price) * LEVERAGE * (st.size / st.initial_size)
        equity_timeline.append((ts, max(1.0, mtm_eq)))

    # MTM-закрытие оставшихся открытых позиций на конце истории
    last_ts = timeline[-1]
    open_mtm_count = len(open_pos)
    for sym, p_info in list(open_pos.items()):
        df, _, _ = prepared[sym]
        last_c = df.iloc[-1]
        last_close = float(last_c["close"])
        last_atr = float(last_c.get("ATR", 0.0)) or (last_close * 0.015)
        st = p_info["risk"]
        slip = calculate_dynamic_slippage(last_atr, last_close)
        actual_exit = last_close * (1 - slip if st.is_long else 1 + slip)
        pnl_raw = ((actual_exit - st.entry_price) if st.is_long else (st.entry_price - actual_exit)) / st.entry_price * LEVERAGE * 100
        fee_pct = (st.entry_price + actual_exit) * TAKER_FEE * LEVERAGE / st.entry_price * 100
        fund_pct = compute_funding_cost_pct(p_info["entry_time"], last_c["timestamp"], st.is_long, LEVERAGE, funding_map.get(sym))
        weight = st.size / st.initial_size
        net_pnl = (pnl_raw - fee_pct - fund_pct) * weight
        equity = max(1.0, equity + p_info["margin"] * (net_pnl / 100.0))
        trades.append({
            "pos_id": p_info["pos_id"], "symbol": sym,
            "type": "long" if st.is_long else "short",
            "entry": st.entry_price, "exit": actual_exit,
            "pnl_pct": net_pnl, "reason": "MTM (Конец периода)",
            "entry_time": p_info["entry_time"], "exit_time": last_c["timestamp"],
        })
    equity_timeline[-1] = (last_ts, equity)

    pos_count, win_rate, pf, _, _ = _aggregate_position_stats(trades)
    total_ret, max_dd, daily_sharpe = _compute_daily_metrics(equity_timeline, initial_capital)
    tdf = pd.DataFrame(trades) if trades else pd.DataFrame()

    if verbose and trades:
        print(f"  [ПОРТФЕЛЬНАЯ СИМУЛЯЦИЯ] Капитал: {initial_capital:.0f}$ -> {equity:.2f}$ ({total_ret:+.2f}%)")
        print(f"  Позиций: {pos_count} (откр. на финише: {open_mtm_count}) | Pos WR: {win_rate:.1f}% | "
              f"PF: {pf:.2f} | MaxDD: {max_dd:.2f}% | Daily Sharpe: {daily_sharpe:.2f}")

    return {
        "symbols": list(prepared.keys()),
        "trades": pos_count,
        "closed_positions": max(0, pos_count - open_mtm_count),
        "open_mtm_positions": open_mtm_count,
        "win_rate": win_rate,
        "profit_factor": pf,
        "pnl_pct": total_ret,
        "max_drawdown": max_dd,
        "sharpe_ratio": daily_sharpe,
        "final_equity": equity,
        "tdf": tdf,
    }


def run_backtest(days: int = 90, force_download: bool = False, target_symbol: str = None):
    symbols_to_test = [target_symbol] if target_symbol and target_symbol != "all" else SYMBOLS

    results = []
    all_trades = []

    bench_df = None
    if USE_MARKET_REGIME_FILTER:
        try:
            bench_df = load_or_download(BENCHMARK_SYMBOL, BENCHMARK_INTERVAL, max(days, 60), force_download)
            if bench_df is not None and not bench_df.empty:
                bench_df = MarketRegimeFilter.add_indicators(
                    bench_df,
                    ema_period=BENCHMARK_EMA_PERIOD,
                    dump_lookback_candles=BENCHMARK_DUMP_CANDLES,
                    dump_pct_threshold=BENCHMARK_DUMP_PCT,
                    pump_pct_threshold=BENCHMARK_PUMP_PCT,
                    rsi_threshold=BENCHMARK_RSI_THRESHOLD,
                    bull_rsi_threshold=BENCHMARK_BULL_RSI_THRESHOLD
                )
        except Exception:
            bench_df = None

    for sym in symbols_to_test:
        res = run_backtest_symbol(
            sym,
            days=days,
            force_download=force_download,
            verbose=(len(symbols_to_test) == 1),
            benchmark_df=bench_df
        )
        if res:
            results.append(res)
            all_trades.append(res["tdf"])

    if len(symbols_to_test) > 1 and results:
        print("\n" + "=" * 78)
        print(f"  СВОДНЫЙ РЕЗУЛЬТАТ ПО КОРЗИНЕ ИЗ {len(results)} ПАР ({days} ДНЕЙ) [v2.3]")
        print("=" * 78)
        print(f"{'Пара':<10}{'Позиций':<9}{'PosWR':<9}{'PF':<8}{'Счет%':<11}{'MaxDD%':<10}{'D.Sharpe':<9}")
        print("-" * 78)
        for r in results:
            wr_s = f"{r['win_rate']:.1f}%"
            pnl_s = f"{r['pnl_pct']:+.2f}%"
            dd_s = f"{r['max_drawdown']:.2f}%"
            print(f"{r['symbol']:<10}{r['trades']:<9}{wr_s:<9}"
                  f"{r['profit_factor']:<8.2f}{pnl_s:<11}{dd_s:<10}{r['sharpe_ratio']:<9.2f}")
        print("-" * 78)
        run_portfolio_backtest(symbols_to_test, days=days, force_download=False, verbose=True)
        print("=" * 78 + "\n")

    return results


if __name__ == "__main__":
    target = sys.argv[1] if len(sys.argv) > 1 and not sys.argv[1].isdigit() else "all"
    days = 90
    for arg in sys.argv[1:]:
        if arg.isdigit():
            days = int(arg)
            break
    force = "--force" in sys.argv
    run_backtest(days=days, force_download=force, target_symbol=target)
