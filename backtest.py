"""
Модуль бэктестинга v2.2 для Bybit USDT Futures.
Включает:
- Модульную стратегию (BaseStrategy / SupertrendProStrategy / MaRsiStrategy)
- Моделирование Funding Rate (каждые 8 часов: 00:00, 08:00, 16:00 UTC)
- Динамическое проскальзывание (Slippage) пропорционально ATR
- Устранение Lookahead Bias: вход и разворот по candle['open'] * (1 ± slip)
- Частичную фиксацию прибыли (Partial TP на 1.5R) и перевод в безубыток
- Трейлинг-стоп (Trailing Stop)
- Векторно ускоренный фильтр рыночного сентимента по бенчмарку BTC (Market Regime Filter)

Использование:
    python backtest.py                  # бэктест по всем парам из config.py
    python backtest.py SOLUSDT 90       # пара, количество дней
    python backtest.py all 90           # бэктест по ВСЕМ парам из корзины
    python backtest.py ETHUSDT 60 --force # принудительно перекачать свечи
"""

import sys
import pandas as pd
import numpy as np
from datetime import datetime, timezone

from config import (
    SYMBOLS, SYMBOL, INTERVAL, HIGHER_INTERVAL,
    ST_PERIOD, ST_MULTIPLIER, ST_CONFIRM_EMA,
    FAST_MA, SLOW_MA, RSI_PERIOD,
    RSI_LONG_MIN, RSI_LONG_MAX, RSI_SHORT_MIN, RSI_SHORT_MAX,
    VOLUME_FILTER, USE_MTF_FILTER, HTF_EMA_PERIOD,
    USE_ATR_STOPS, ATR_PERIOD, ATR_SL_MULT, ATR_TP_MULT,
    STOP_LOSS_PCT, TAKE_PROFIT_PCT,
    USE_BREAKEVEN, BREAKEVEN_TRIGGER, BREAKEVEN_OFFSET,
    USE_TRAILING_STOP, TRAILING_STOP_TRIGGER, TRAILING_STOP_ATR_MULT, TRAILING_STOP_FALLBACK,
    USE_PARTIAL_TP, PARTIAL_TP_R_MULT, PARTIAL_TP_RATIO, PARTIAL_TP_MOVE_BE,
    USE_MARKET_REGIME_FILTER, BENCHMARK_SYMBOL, BENCHMARK_INTERVAL,
    BENCHMARK_EMA_PERIOD, BENCHMARK_RSI_THRESHOLD, BENCHMARK_DUMP_PCT, BENCHMARK_DUMP_CANDLES,
    LEVERAGE, ACTIVE_STRATEGY,
    SIGNAL_LOOKBACK_CANDLES, BENCHMARK_BULL_RSI_THRESHOLD, BENCHMARK_PUMP_PCT,
    USE_ADX_FILTER, ADX_PERIOD, ADX_THRESHOLD
)
from data_loader import load_or_download
from strategies import get_strategy
from risk_manager import PositionRiskState, MarketRegimeFilter

# Базовые комиссии
TAKER_FEE = 0.00055   # 0.055% Bybit VIP0 taker fee
BASE_SLIPPAGE = 0.0002 # 0.02% базовое проскальзывание
FUNDING_RATE_DEFAULT = 0.0001 # 0.01% за 8-часовой интервал (Bybit standard)


def calculate_dynamic_slippage(atr: float, price: float) -> float:
    """
    Рассчитывает динамическое проскальзывание, пропорциональное волатильности (ATR).
    """
    if price <= 0 or atr <= 0:
        return BASE_SLIPPAGE
    volatility_slippage = (atr / price) * 0.05
    return float(max(BASE_SLIPPAGE, min(volatility_slippage, 0.003))) # от 0.02% до 0.3%


def count_funding_events(entry_time: pd.Timestamp, exit_time: pd.Timestamp) -> int:
    """
    Подсчитывает количество 8-часовых клирингов Funding Rate (00:00, 08:00, 16:00 UTC)
    между входом и выходом из позиции.
    """
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
        return int(hours // 8)


def run_backtest_symbol(
    symbol: str,
    days: int = 90,
    force_download: bool = False,
    verbose: bool = True,
    strategy_params: dict = None,
    benchmark_df: pd.DataFrame = None
) -> dict | None:
    """
    Запускает бэктест по одной торговой паре.
    """
    params = strategy_params or {}
    fast_ma = params.get("fast_ma", FAST_MA)
    slow_ma = params.get("slow_ma", SLOW_MA)
    rsi_period = params.get("rsi_period", RSI_PERIOD)
    atr_sl_mult = params.get("atr_sl_mult", ATR_SL_MULT)
    atr_tp_mult = params.get("atr_tp_mult", ATR_TP_MULT)
    trailing_atr_mult = params.get("trailing_atr_mult", TRAILING_STOP_ATR_MULT)

    strategy = get_strategy(
        params.get("strategy_name", ACTIVE_STRATEGY),
        st_period=params.get("st_period", ST_PERIOD),
        st_multiplier=params.get("st_multiplier", ST_MULTIPLIER),
        confirm_ema_period=params.get("confirm_ema_period", ST_CONFIRM_EMA),
        fast_ma=fast_ma,
        slow_ma=slow_ma,
        rsi_period=rsi_period,
        rsi_long_min=params.get("rsi_long_min", RSI_LONG_MIN),
        rsi_long_max=params.get("rsi_long_max", RSI_LONG_MAX),
        rsi_short_min=params.get("rsi_short_min", RSI_SHORT_MIN),
        rsi_short_max=params.get("rsi_short_max", RSI_SHORT_MAX),
        volume_filter=params.get("volume_filter", VOLUME_FILTER),
        use_mtf_filter=USE_MTF_FILTER,
        htf_ema_period=HTF_EMA_PERIOD,
        atr_period=ATR_PERIOD,
        cross_lookback=params.get("cross_lookback", SIGNAL_LOOKBACK_CANDLES),
        use_adx_filter=params.get("use_adx_filter", USE_ADX_FILTER),
        adx_period=params.get("adx_period", ADX_PERIOD),
        adx_threshold=params.get("adx_threshold", ADX_THRESHOLD)
    )

    if verbose:
        print("\n" + "="*62)
        print(f"  БЭКТЕСТ v2.2 — {symbol} | TF: {INTERVAL}m | {days} дней")
        print(f"  Стратегия: {strategy.name} | Плечо: x{LEVERAGE} | Funding & Slippage: ON")
        print(f"  Трейлинг: {USE_TRAILING_STOP} | Частичный TP: {USE_PARTIAL_TP} | Режим рынка: {USE_MARKET_REGIME_FILTER}")
        print("="*62)

    df = load_or_download(symbol, INTERVAL, days, force_download)
    if df.empty or len(df) < slow_ma + 15:
        if verbose:
            print(f"Недостаточно данных для {symbol}")
        return None

    # Загрузка и привязка старшего таймфрейма (MTF)
    if USE_MTF_FILTER:
        htf_days = max(days, 60)
        df_htf = load_or_download(symbol, HIGHER_INTERVAL, htf_days, force_download)
        if not df_htf.empty and len(df_htf) >= 50:
            span = min(HTF_EMA_PERIOD, len(df_htf))
            df_htf["HTF_EMA"] = df_htf["close"].ewm(span=span, adjust=False).mean().shift(1)
            df["timestamp"] = pd.to_datetime(df["timestamp"]).astype("datetime64[ms]")
            df_htf["timestamp"] = pd.to_datetime(df_htf["timestamp"]).astype("datetime64[ms]")
            df_htf_sub = df_htf[["timestamp", "HTF_EMA"]].sort_values("timestamp")
            df = pd.merge_asof(
                df.sort_values("timestamp"),
                df_htf_sub,
                on="timestamp",
                direction="backward"
            )
        else:
            df["HTF_EMA"] = np.nan
    else:
        df["HTF_EMA"] = np.nan

    # Расчет индикаторов и сигналов стратегии
    df = strategy.calculate_indicators(df)
    # Удаляем строки только если нет ключевых колонок
    req_cols = [c for c in ["MA_fast", "MA_slow", "RSI", "signal"] if c in df.columns]
    if req_cols:
        df = df.dropna(subset=req_cols).reset_index(drop=True)

    # Векторное объединение с бенчмарком (O(1) доступ вместо O(N*M) срезов в цикле)
    if USE_MARKET_REGIME_FILTER:
        if benchmark_df is None:
            try:
                benchmark_df = load_or_download(BENCHMARK_SYMBOL, BENCHMARK_INTERVAL, max(days, 60), force_download)
            except Exception:
                benchmark_df = None
        if benchmark_df is not None and not benchmark_df.empty:
            if "BENCH_LONG_ALLOWED" not in benchmark_df.columns:
                benchmark_df = MarketRegimeFilter.add_indicators(
                    benchmark_df,
                    ema_period=BENCHMARK_EMA_PERIOD,
                    dump_lookback_candles=BENCHMARK_DUMP_CANDLES,
                    dump_pct_threshold=BENCHMARK_DUMP_PCT,
                    pump_pct_threshold=BENCHMARK_PUMP_PCT,
                    rsi_threshold=BENCHMARK_RSI_THRESHOLD,
                    bull_rsi_threshold=BENCHMARK_BULL_RSI_THRESHOLD
                )
            benchmark_df["timestamp"] = pd.to_datetime(benchmark_df["timestamp"]).astype("datetime64[ms]")
            df["timestamp"] = pd.to_datetime(df["timestamp"]).astype("datetime64[ms]")

            bench_cols = [
                "timestamp", "BENCH_LONG_ALLOWED", "BENCH_SHORT_ALLOWED",
                "BENCH_SELF_LONG_ALLOWED", "BENCH_SELF_SHORT_ALLOWED"
            ]
            bench_sub = benchmark_df[[c for c in bench_cols if c in benchmark_df.columns]].sort_values("timestamp").copy()
            # Сдвиг на 1 свечу назад исключает Lookahead Bias по незакрытой 4H свече бенчмарка
            for b_col in ["BENCH_LONG_ALLOWED", "BENCH_SHORT_ALLOWED", "BENCH_SELF_LONG_ALLOWED", "BENCH_SELF_SHORT_ALLOWED"]:
                if b_col in bench_sub.columns:
                    bench_sub[b_col] = bench_sub[b_col].shift(1, fill_value=True)
            df = pd.merge_asof(
                df.sort_values("timestamp"),
                bench_sub,
                on="timestamp",
                direction="backward"
            )

    # Симуляция исполнения сделок с риск-менеджментом v2.1
    trades = []
    pos_state: PositionRiskState | None = None
    entry_idx = None
    is_benchmark_symbol = (symbol.upper() == BENCHMARK_SYMBOL.upper())

    for i in range(1, len(df)):
        candle = df.iloc[i]
        open_price = float(candle["open"])
        high_price = float(candle["high"])
        low_price = float(candle["low"])
        close_price = float(candle["close"])
        candle_time = candle["timestamp"]

        prev_candle = df.iloc[i - 1]
        raw_signal = int(prev_candle.get("signal", 0))
        atr = float(prev_candle.get("ATR", 0.0))
        if atr <= 0:
            atr = open_price * 0.015

        # 1. Фильтрация сентимента рынка по бенчмарку (O(1) из предварительно объединенной строки)
        if raw_signal != 0 and USE_MARKET_REGIME_FILTER:
            if raw_signal == 1:
                allowed = bool(candle.get(
                    "BENCH_SELF_LONG_ALLOWED" if is_benchmark_symbol else "BENCH_LONG_ALLOWED",
                    True
                ))
                if not allowed:
                    raw_signal = 0
            elif raw_signal == -1:
                allowed = bool(candle.get(
                    "BENCH_SELF_SHORT_ALLOWED" if is_benchmark_symbol else "BENCH_SHORT_ALLOWED",
                    True
                ))
                if not allowed:
                    raw_signal = 0

        # 2. Симуляция исполнения на ОТКРЫТИИ свечи i (устранение Lookahead Bias)
        # Если есть открытая позиция и получен сигнал разворота:
        if raw_signal == 1 and pos_state is not None and not pos_state.is_long:
            # Закрываем short по open свечи с учетом динамического проскальзывания
            exit_slip = calculate_dynamic_slippage(atr, open_price)
            actual_exit = open_price * (1 + exit_slip)
            fee_pct = (pos_state.entry_price + actual_exit) * TAKER_FEE * LEVERAGE / pos_state.entry_price * 100
            pnl_raw = (pos_state.entry_price - actual_exit) / pos_state.entry_price * LEVERAGE * 100
            funding_cnt = count_funding_events(df["timestamp"].iloc[entry_idx], candle_time)
            funding_cost_pct = funding_cnt * FUNDING_RATE_DEFAULT * LEVERAGE * 100 * (-1)
            weight = pos_state.size / pos_state.initial_size

            trades.append({
                "symbol": symbol,
                "type": "short",
                "entry": pos_state.entry_price,
                "exit": actual_exit,
                "pnl_pct": (pnl_raw - fee_pct - funding_cost_pct) * weight,
                "reason": "Сигнал разворота",
                "bars": i - entry_idx,
                "entry_time": df["timestamp"].iloc[entry_idx],
                "exit_time": candle_time,
            })
            pos_state = None
            entry_idx = None

        elif raw_signal == -1 and pos_state is not None and pos_state.is_long:
            # Закрываем long по open свечи с учетом динамического проскальзывания
            exit_slip = calculate_dynamic_slippage(atr, open_price)
            actual_exit = open_price * (1 - exit_slip)
            fee_pct = (pos_state.entry_price + actual_exit) * TAKER_FEE * LEVERAGE / pos_state.entry_price * 100
            pnl_raw = (actual_exit - pos_state.entry_price) / pos_state.entry_price * LEVERAGE * 100
            funding_cnt = count_funding_events(df["timestamp"].iloc[entry_idx], candle_time)
            funding_cost_pct = funding_cnt * FUNDING_RATE_DEFAULT * LEVERAGE * 100 * (1)
            weight = pos_state.size / pos_state.initial_size

            trades.append({
                "symbol": symbol,
                "type": "long",
                "entry": pos_state.entry_price,
                "exit": actual_exit,
                "pnl_pct": (pnl_raw - fee_pct - funding_cost_pct) * weight,
                "reason": "Сигнал разворота",
                "bars": i - entry_idx,
                "entry_time": df["timestamp"].iloc[entry_idx],
                "exit_time": candle_time,
            })
            pos_state = None
            entry_idx = None

        # Открытие новой позиции на ОТКРЫТИИ свечи i
        if pos_state is None and raw_signal != 0:
            entry_slip = calculate_dynamic_slippage(atr, open_price)
            if raw_signal == 1:
                entry_p = open_price * (1 + entry_slip)
                stops = strategy.calculate_stops(
                    entry_p, "buy", atr_value=atr,
                    stop_loss_pct=STOP_LOSS_PCT, take_profit_pct=TAKE_PROFIT_PCT,
                    atr_sl_mult=atr_sl_mult, atr_tp_mult=atr_tp_mult,
                    use_atr=USE_ATR_STOPS
                )
                pos_state = PositionRiskState(
                    symbol=symbol,
                    side="Buy",
                    entry_price=entry_p,
                    size=1.0,
                    initial_size=1.0,
                    stop_loss=stops["sl"],
                    take_profit=stops["tp"],
                    atr_value=atr,
                    r_mult=PARTIAL_TP_R_MULT
                )
                entry_idx = i
            elif raw_signal == -1:
                entry_p = open_price * (1 - entry_slip)
                stops = strategy.calculate_stops(
                    entry_p, "sell", atr_value=atr,
                    stop_loss_pct=STOP_LOSS_PCT, take_profit_pct=TAKE_PROFIT_PCT,
                    atr_sl_mult=atr_sl_mult, atr_tp_mult=atr_tp_mult,
                    use_atr=USE_ATR_STOPS
                )
                pos_state = PositionRiskState(
                    symbol=symbol,
                    side="Sell",
                    entry_price=entry_p,
                    size=1.0,
                    initial_size=1.0,
                    stop_loss=stops["sl"],
                    take_profit=stops["tp"],
                    atr_value=atr,
                    r_mult=PARTIAL_TP_R_MULT
                )
                entry_idx = i

        # 3. Внутрисвечевая симуляция цены (High, Low, Close) для открытой позиции
        # Порядок без внутрисвечевого Lookahead Bias:
        # 3.1. Сначала проверяется активный на начало свечи Stop-Loss (консервативный приоритет SL).
        # 3.2. Затем частичный TP1 (1.5R) и полный Take-Profit (TP/TP2).
        # 3.3. В конце свечи обновляются экстремумы, Breakeven и Trailing Stop для следующих свечей.
        if pos_state is not None:
            closed_reason = None
            exit_price = None

            # 3.1. Проверка срабатывания текущего Stop-Loss
            if pos_state.is_long:
                if low_price <= pos_state.stop_loss:
                    exit_price = pos_state.stop_loss
                    closed_reason = "Trailing SL" if pos_state.trailing_active else ("Безубыток" if pos_state.breakeven_applied else "SL")
            else:
                if high_price >= pos_state.stop_loss:
                    exit_price = pos_state.stop_loss
                    closed_reason = "Trailing SL" if pos_state.trailing_active else ("Безубыток" if pos_state.breakeven_applied else "SL")

            # 3.2. Если SL не выбит — проверяем Partial TP (1.5R) и основной Take-Profit
            if not closed_reason:
                if USE_PARTIAL_TP and not pos_state.tp1_executed:
                    triggered, qty_close, be_sl = pos_state.check_partial_tp(
                        high_price if pos_state.is_long else low_price,
                        ratio=PARTIAL_TP_RATIO,
                        offset_pct=BREAKEVEN_OFFSET
                    )
                    if triggered:
                        tp1_target = pos_state.tp1_price
                        fee_pct = (pos_state.entry_price + tp1_target) * TAKER_FEE * LEVERAGE / pos_state.entry_price * 100
                        pnl_raw = ((tp1_target - pos_state.entry_price) / pos_state.entry_price if pos_state.is_long else (pos_state.entry_price - tp1_target) / pos_state.entry_price) * LEVERAGE * 100
                        funding_cnt = count_funding_events(df["timestamp"].iloc[entry_idx], candle_time)
                        funding_cost_pct = funding_cnt * FUNDING_RATE_DEFAULT * LEVERAGE * 100 * (1 if pos_state.is_long else -1)
                        net_pnl = (pnl_raw - fee_pct - funding_cost_pct) * PARTIAL_TP_RATIO

                        trades.append({
                            "symbol": symbol,
                            "type": "long" if pos_state.is_long else "short",
                            "entry": pos_state.entry_price,
                            "exit": tp1_target,
                            "pnl_pct": net_pnl,
                            "reason": "TP1 (1.5R)",
                            "bars": i - entry_idx,
                            "entry_time": df["timestamp"].iloc[entry_idx],
                            "exit_time": candle_time,
                        })

                        pos_state.tp1_executed = True
                        pos_state.size -= qty_close
                        if PARTIAL_TP_MOVE_BE:
                            pos_state.stop_loss = be_sl
                            pos_state.breakeven_applied = True

                if pos_state.is_long and high_price >= pos_state.take_profit:
                    exit_price = pos_state.take_profit
                    closed_reason = "TP2" if pos_state.tp1_executed else "TP"
                elif not pos_state.is_long and low_price <= pos_state.take_profit:
                    exit_price = pos_state.take_profit
                    closed_reason = "TP2" if pos_state.tp1_executed else "TP"

            if closed_reason and exit_price is not None:
                exit_slip = calculate_dynamic_slippage(atr, exit_price)
                if pos_state.is_long:
                    actual_exit = exit_price * (1 - exit_slip)
                    pnl_raw = (actual_exit - pos_state.entry_price) / pos_state.entry_price * LEVERAGE * 100
                else:
                    actual_exit = exit_price * (1 + exit_slip)
                    pnl_raw = (pos_state.entry_price - actual_exit) / pos_state.entry_price * LEVERAGE * 100

                fee_pct = (pos_state.entry_price + actual_exit) * TAKER_FEE * LEVERAGE / pos_state.entry_price * 100
                funding_cnt = count_funding_events(df["timestamp"].iloc[entry_idx], candle_time)
                funding_cost_pct = funding_cnt * FUNDING_RATE_DEFAULT * LEVERAGE * 100 * (1 if pos_state.is_long else -1)

                weight = (pos_state.size / pos_state.initial_size)
                net_pnl = (pnl_raw - fee_pct - funding_cost_pct) * weight

                trades.append({
                    "symbol": symbol,
                    "type": "long" if pos_state.is_long else "short",
                    "entry": pos_state.entry_price,
                    "exit": actual_exit,
                    "pnl_pct": net_pnl,
                    "reason": closed_reason,
                    "bars": i - entry_idx,
                    "entry_time": df["timestamp"].iloc[entry_idx],
                    "exit_time": candle_time,
                })
                pos_state = None
                entry_idx = None
            else:
                # 3.3. Обновление экстремумов, безубытка и трейлинг-стопа по закрытию свечи (для свечи i+1)
                pos_state.update_price_extremes(high_price, low_price)

                if USE_BREAKEVEN and not pos_state.breakeven_applied:
                    be_updated, new_sl = pos_state.check_breakeven(
                        current_price=close_price,
                        trigger_pct=BREAKEVEN_TRIGGER,
                        offset_pct=BREAKEVEN_OFFSET
                    )
                    if be_updated:
                        pos_state.stop_loss = new_sl
                        pos_state.breakeven_applied = True

                if USE_TRAILING_STOP:
                    trail_updated, new_sl = pos_state.check_trailing_stop(
                        current_price=close_price,
                        high_price=high_price,
                        low_price=low_price,
                        trigger_pct=TRAILING_STOP_TRIGGER,
                        atr_mult=trailing_atr_mult,
                        fallback_pct=TRAILING_STOP_FALLBACK
                    )
                    if trail_updated:
                        pos_state.stop_loss = new_sl

    if not trades:
        if verbose:
            print("Сделок не найдено.")
        return None

    tdf = pd.DataFrame(trades)
    total = len(tdf)
    wins = len(tdf[tdf["pnl_pct"] > 0])
    losses = total - wins
    win_rate = (wins / total * 100) if total > 0 else 0.0

    avg_win = tdf[tdf["pnl_pct"] > 0]["pnl_pct"].mean() if wins else 0.0
    avg_loss = tdf[tdf["pnl_pct"] <= 0]["pnl_pct"].mean() if losses else 0.0
    total_pnl_pct = tdf["pnl_pct"].sum()

    cumulative = tdf["pnl_pct"].cumsum()
    max_drawdown = float((cumulative - cumulative.cummax()).min()) if not cumulative.empty else 0.0

    gross_profit = tdf[tdf["pnl_pct"] > 0]["pnl_pct"].sum()
    gross_loss = abs(tdf[tdf["pnl_pct"] <= 0]["pnl_pct"].sum())
    profit_factor = (gross_profit / gross_loss) if gross_loss > 0 else 999.0

    daily_returns = tdf["pnl_pct"]
    sharpe = float(daily_returns.mean() / daily_returns.std() * np.sqrt(365)) if daily_returns.std() > 0 else 0.0

    if verbose:
        print(f"Всего сделок:      {total} (Win: {wins} | Loss: {losses})")
        print(f"Винрейт (Win Rate): {win_rate:.1f}%")
        print(f"Profit Factor:     {profit_factor:.2f}")
        print(f"Итоговый PnL:      {total_pnl_pct:+.2f}% (с учетом комиссий и фандинга)")
        print(f"Макс. просадка:    {max_drawdown:.2f}%")
        print(f"Коэфф. Шарпа:      {sharpe:.2f}")
        print(f"Средняя прибыль:   {avg_win:+.2f}% | Средний убыток: {avg_loss:+.2f}%")
        print("="*62 + "\n")

    return {
        "symbol": symbol,
        "trades": total,
        "win_rate": win_rate,
        "profit_factor": profit_factor,
        "pnl_pct": total_pnl_pct,
        "max_drawdown": max_drawdown,
        "sharpe_ratio": sharpe,
        "tdf": tdf
    }


def run_backtest(days: int = 90, force_download: bool = False, target_symbol: str = None):
    symbols_to_test = [target_symbol] if target_symbol and target_symbol != "all" else SYMBOLS

    results = []
    all_trades = []

    # Предзагрузка и расчет бенчмарка один раз для всех пар
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
        print("\n" + "="*76)
        print(f"  СВОДНЫЙ РЕЗУЛЬТАТ ПО КОРЗИНЕ ИЗ {len(results)} ПАР ({days} ДНЕЙ) [v2.2]")
        print("="*76)
        print(f"{'Пара':<10}{'Сделок':<8}{'WinRate':<10}{'PF':<8}{'PnL%':<12}{'MaxDD%':<10}{'Sharpe':<8}")
        print("-" * 76)
        for r in results:
            print(f"{r['symbol']:<10}{r['trades']:<8}{r['win_rate']:<9.1f}%"
                  f"{r['profit_factor']:<8.2f}{r['pnl_pct']:<+12.2f}%{r['max_drawdown']:<10.2f}%{r['sharpe_ratio']:<8.2f}")
        print("-" * 76)

        combined_tdf = pd.concat(all_trades, ignore_index=True)
        tot_trades = len(combined_tdf)
        tot_wins = len(combined_tdf[combined_tdf["pnl_pct"] > 0])
        tot_wr = tot_wins / tot_trades * 100 if tot_trades else 0
        tot_pnl = combined_tdf["pnl_pct"].sum() / len(symbols_to_test)
        print(f"{'ИТОГО':<10}{tot_trades:<8}{tot_wr:<9.1f}%{'--':<8}{tot_pnl:<+12.2f}%")
        print("="*76 + "\n")

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
