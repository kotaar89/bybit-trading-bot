"""
Модуль параметрической оптимизации торговой стратегии (v2.3)
Поддерживает:
- Оптимизацию гиперпараметров для ВСЕХ стратегий (SUPERTREND_PRO, SUPERTREND, MA_RSI, BB_RSI, DONCHIAN)
- Интеграцию MTF-фильтра (4H HTF_CLOSE vs HTF_EMA) и MarketRegimeFilter (BTC)
- Позиционную группировку, MTM на конце окна, Daily Sharpe Ratio и ограничение PF
- Скользящий Walk-Forward Analysis (Rolling WFA) с буфером прогрева перед каждым OOS-окном

Использование:
    python optimizer.py ETHUSDT --wfa --trials 40 --days 180
    python optimizer.py SOLUSDT --trials 30 --days 90 --windows 3
"""

import argparse
import numpy as np
import pandas as pd

from config import (
    INTERVAL, HIGHER_INTERVAL, USE_MTF_FILTER, HTF_EMA_PERIOD,
    ST_PERIOD, ST_MULTIPLIER, ST_CONFIRM_EMA, ST_CONFIRM_LOOKBACK,
    ATR_PERIOD, ATR_SL_MULT, ATR_TP_MULT, TRAILING_STOP_ATR_MULT,
    LEVERAGE, ACTIVE_STRATEGY,
    RSI_LONG_MIN, RSI_LONG_MAX, RSI_SHORT_MIN, RSI_SHORT_MAX, VOLUME_FILTER,
    USE_TRAILING_STOP, TRAILING_STOP_TRIGGER, TRAILING_STOP_R_MULT, TRAILING_STOP_FALLBACK,
    USE_PARTIAL_TP, USE_EXCHANGE_TP1, PARTIAL_TP_R_MULT, PARTIAL_TP_RATIO, PARTIAL_TP_MOVE_BE,
    STOP_LOSS_PCT, TAKE_PROFIT_PCT, USE_ATR_STOPS,
    SIGNAL_LOOKBACK_CANDLES, USE_BREAKEVEN, BREAKEVEN_TRIGGER, BREAKEVEN_R_MULT, BREAKEVEN_OFFSET,
    USE_ADX_FILTER, ADX_PERIOD, ADX_THRESHOLD,
    USE_MARKET_REGIME_FILTER, BENCHMARK_SYMBOL, BENCHMARK_INTERVAL
)
from data_loader import load_or_download
from strategies import get_strategy
from risk_manager import PositionRiskState
from backtest import (
    TAKER_FEE, MAKER_FEE,
    align_htf_to_ltf, align_benchmark_to_ltf,
    calculate_dynamic_slippage, compute_funding_cost_pct,
    compute_warmup_bars, _calc_position_margin_frac,
    _compute_daily_metrics, _aggregate_position_stats,
)

try:
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False


def simulate_strategy_slice(
    df_slice: pd.DataFrame,
    params: dict,
    symbol: str = "BENCH",
    strategy_name: str | None = None,
    use_regime_filter: bool | None = None,
    eval_start_idx: int | None = None,
    funding_df: pd.DataFrame | None = None,
) -> dict:
    """
    Симуляция стратегии на срезе данных (In-Sample или Out-of-Sample)
    с прогревом индикаторов, MTM-закрытием и честным Daily Sharpe Ratio.
    """
    strat_name = (strategy_name or params.get("strategy_name") or ACTIVE_STRATEGY).strip().upper()
    min_len = max(params.get("slow_ma", 50), params.get("confirm_ema_period", 50), 30)
    if df_slice.empty or len(df_slice) < min_len:
        return {"trades": 0, "win_rate": 0.0, "pnl_pct": 0.0, "sharpe_ratio": 0.0, "profit_factor": 0.0, "max_drawdown": 0.0}

    use_mtf = USE_MTF_FILTER if strat_name != "BB_RSI" else bool(params.get("use_mtf_filter", False))
    strategy = get_strategy(
        strat_name,
        st_period=params.get("st_period", ST_PERIOD),
        st_multiplier=params.get("st_multiplier", ST_MULTIPLIER),
        confirm_ema_period=params.get("confirm_ema_period", ST_CONFIRM_EMA),
        confirm_lookback=params.get("confirm_lookback", ST_CONFIRM_LOOKBACK),
        use_confirm_ema=params.get("use_confirm_ema", True),
        fast_ma=params.get("fast_ma", 20),
        slow_ma=params.get("slow_ma", 50),
        rsi_period=params.get("rsi_period", 14),
        rsi_long_min=params.get("rsi_long_min", RSI_LONG_MIN),
        rsi_long_max=params.get("rsi_long_max", RSI_LONG_MAX),
        rsi_short_min=params.get("rsi_short_min", RSI_SHORT_MIN),
        rsi_short_max=params.get("rsi_short_max", RSI_SHORT_MAX),
        bb_period=params.get("bb_period", 20),
        bb_std=params.get("bb_std", 2.0),
        lookback_period=params.get("lookback_period", 20),
        volume_filter=params.get("volume_filter", VOLUME_FILTER),
        use_mtf_filter=use_mtf,
        htf_ema_period=HTF_EMA_PERIOD,
        atr_period=ATR_PERIOD,
        cross_lookback=params.get("cross_lookback", SIGNAL_LOOKBACK_CANDLES),
        use_adx_filter=params.get("use_adx_filter", USE_ADX_FILTER),
        adx_period=params.get("adx_period", ADX_PERIOD),
        adx_threshold=params.get("adx_threshold", ADX_THRESHOLD),
    )

    df = strategy.calculate_indicators(df_slice.copy()).reset_index(drop=True)
    warmup = eval_start_idx if eval_start_idx is not None else compute_warmup_bars(strategy, len(df))
    warmup = max(1, min(warmup, len(df) - 5))
    if len(df) - warmup < 5:
        return {"trades": 0, "win_rate": 0.0, "pnl_pct": 0.0, "sharpe_ratio": 0.0, "profit_factor": 0.0, "max_drawdown": 0.0}

    atr_sl_mult = params.get("atr_sl_mult", ATR_SL_MULT)
    atr_tp_mult = params.get("atr_tp_mult", ATR_TP_MULT)
    trailing_mult = params.get("trailing_atr_mult", TRAILING_STOP_ATR_MULT)
    be_r_mult = params.get("breakeven_r_mult", BREAKEVEN_R_MULT)
    tp1_r_mult = params.get("partial_tp_r_mult", PARTIAL_TP_R_MULT)
    trail_r_mult = params.get("trailing_r_mult", TRAILING_STOP_R_MULT)
    tp1_fee_rate = MAKER_FEE if USE_EXCHANGE_TP1 else TAKER_FEE

    has_bench_cols = ("BENCH_LONG_ALLOWED" in df.columns) and ("BENCH_SHORT_ALLOWED" in df.columns)
    apply_regime = (USE_MARKET_REGIME_FILTER if use_regime_filter is None else use_regime_filter) and has_bench_cols
    is_bench_sym = (symbol.upper() == BENCHMARK_SYMBOL.upper())

    trades = []
    pos_state: PositionRiskState | None = None
    entry_idx = None
    pos_id = 0
    pos_margin = 0.0
    equity = 100.0
    initial_equity = 100.0
    equity_timeline = [(df["timestamp"].iloc[warmup], equity)]

    for i in range(warmup, len(df)):
        candle = df.iloc[i]
        open_p = float(candle["open"])
        close_p = float(candle["close"])
        low_p = float(candle["low"])
        high_p = float(candle["high"])
        candle_time = candle["timestamp"]

        prev_candle = df.iloc[i - 1]
        raw_signal = int(prev_candle.get("signal", 0))
        exit_sig = int(prev_candle.get("exit_signal", raw_signal))
        atr = float(prev_candle.get("ATR", 0.0))
        if np.isnan(atr) or atr <= 0:
            atr = open_p * 0.015

        # 1. Выход по разворотному exit_signal на открытии бара
        if exit_sig == 1 and pos_state is not None and not pos_state.is_long:
            slip = calculate_dynamic_slippage(atr, open_p)
            actual_exit = open_p * (1 + slip)
            fee_pct = (pos_state.entry_price + actual_exit) * TAKER_FEE * LEVERAGE / pos_state.entry_price * 100
            pnl_raw = (pos_state.entry_price - actual_exit) / pos_state.entry_price * LEVERAGE * 100
            fund_pct = compute_funding_cost_pct(df["timestamp"].iloc[entry_idx], candle_time, False, LEVERAGE, funding_df)
            net_pnl = (pnl_raw - fee_pct - fund_pct) * (pos_state.size / pos_state.initial_size)
            equity = max(0.01, equity + pos_margin * (net_pnl / 100.0))
            trades.append({"pos_id": pos_id, "pnl_pct": net_pnl})
            pos_state = None
            entry_idx = None
        elif exit_sig == -1 and pos_state is not None and pos_state.is_long:
            slip = calculate_dynamic_slippage(atr, open_p)
            actual_exit = open_p * (1 - slip)
            fee_pct = (pos_state.entry_price + actual_exit) * TAKER_FEE * LEVERAGE / pos_state.entry_price * 100
            pnl_raw = (actual_exit - pos_state.entry_price) / pos_state.entry_price * LEVERAGE * 100
            fund_pct = compute_funding_cost_pct(df["timestamp"].iloc[entry_idx], candle_time, True, LEVERAGE, funding_df)
            net_pnl = (pnl_raw - fee_pct - fund_pct) * (pos_state.size / pos_state.initial_size)
            equity = max(0.01, equity + pos_margin * (net_pnl / 100.0))
            trades.append({"pos_id": pos_id, "pnl_pct": net_pnl})
            pos_state = None
            entry_idx = None

        # 2. Фильтр режима рынка (fail-safe)
        if raw_signal != 0 and apply_regime:
            col = (
                ("BENCH_SELF_LONG_ALLOWED" if is_bench_sym else "BENCH_LONG_ALLOWED")
                if raw_signal == 1
                else ("BENCH_SELF_SHORT_ALLOWED" if is_bench_sym else "BENCH_SHORT_ALLOWED")
            )
            raw_allowed = prev_candle.get(col)
            if pd.isna(raw_allowed) or not bool(raw_allowed):
                raw_signal = 0

        # 3. Новый вход
        if pos_state is None and raw_signal != 0:
            slip = calculate_dynamic_slippage(atr, open_p)
            entry_p = open_p * (1 + slip if raw_signal == 1 else 1 - slip)
            stops = strategy.calculate_stops(
                entry_p, "buy" if raw_signal == 1 else "sell", atr_value=atr,
                stop_loss_pct=STOP_LOSS_PCT, take_profit_pct=TAKE_PROFIT_PCT,
                atr_sl_mult=atr_sl_mult, atr_tp_mult=atr_tp_mult, use_atr=USE_ATR_STOPS
            )
            pos_id += 1
            pos_margin = equity * _calc_position_margin_frac(entry_p, stops["sl_dist"])
            pos_state = PositionRiskState(
                symbol=symbol, side="Buy" if raw_signal == 1 else "Sell",
                entry_price=entry_p, size=1.0, initial_size=1.0,
                stop_loss=stops["sl"], take_profit=stops["tp"],
                atr_value=atr, r_mult=tp1_r_mult
            )
            entry_idx = i

        # 4. Внутрисвечевая симуляция
        if pos_state is not None:
            exit_p = None
            is_tp_limit = False
            if pos_state.is_long and low_p <= pos_state.stop_loss:
                exit_p = pos_state.stop_loss
            elif not pos_state.is_long and high_p >= pos_state.stop_loss:
                exit_p = pos_state.stop_loss

            if exit_p is None:
                if USE_PARTIAL_TP and not pos_state.tp1_executed:
                    triggered, qty_c, be_sl = pos_state.check_partial_tp(
                        high_p if pos_state.is_long else low_p,
                        ratio=PARTIAL_TP_RATIO, offset_pct=BREAKEVEN_OFFSET
                    )
                    if triggered:
                        tp1_target = pos_state.tp1_price
                        fee_pct = ((pos_state.entry_price * TAKER_FEE) + (tp1_target * tp1_fee_rate)) * LEVERAGE / pos_state.entry_price * 100
                        pnl_raw = (
                            (tp1_target - pos_state.entry_price) / pos_state.entry_price
                            if pos_state.is_long
                            else (pos_state.entry_price - tp1_target) / pos_state.entry_price
                        ) * LEVERAGE * 100
                        fund_pct = compute_funding_cost_pct(df["timestamp"].iloc[entry_idx], candle_time, pos_state.is_long, LEVERAGE, funding_df)
                        net_pnl = (pnl_raw - fee_pct - fund_pct) * PARTIAL_TP_RATIO
                        equity = max(0.01, equity + pos_margin * (net_pnl / 100.0))
                        trades.append({"pos_id": pos_id, "pnl_pct": net_pnl})
                        pos_state.tp1_executed = True
                        pos_state.size -= qty_c
                        if PARTIAL_TP_MOVE_BE:
                            if (pos_state.is_long and be_sl > pos_state.stop_loss) or (not pos_state.is_long and be_sl < pos_state.stop_loss):
                                pos_state.stop_loss = be_sl
                            pos_state.breakeven_applied = True

                if pos_state.is_long and high_p >= pos_state.take_profit:
                    exit_p = pos_state.take_profit
                    is_tp_limit = True
                elif not pos_state.is_long and low_p <= pos_state.take_profit:
                    exit_p = pos_state.take_profit
                    is_tp_limit = True

            if exit_p is not None:
                slip = 0.0 if is_tp_limit else calculate_dynamic_slippage(atr, exit_p)
                actual_exit = exit_p * (1 - slip if pos_state.is_long else 1 + slip)
                pnl_raw = (
                    (actual_exit - pos_state.entry_price) / pos_state.entry_price
                    if pos_state.is_long
                    else (pos_state.entry_price - actual_exit) / pos_state.entry_price
                ) * LEVERAGE * 100
                fee_pct = (pos_state.entry_price + actual_exit) * TAKER_FEE * LEVERAGE / pos_state.entry_price * 100
                fund_pct = compute_funding_cost_pct(df["timestamp"].iloc[entry_idx], candle_time, pos_state.is_long, LEVERAGE, funding_df)
                net_pnl = (pnl_raw - fee_pct - fund_pct) * (pos_state.size / pos_state.initial_size)
                equity = max(0.01, equity + pos_margin * (net_pnl / 100.0))
                trades.append({"pos_id": pos_id, "pnl_pct": net_pnl})
                pos_state = None
                entry_idx = None
            else:
                pos_state.update_price_extremes(high_p, low_p)
                intra_fav = high_p if pos_state.is_long else low_p
                if USE_BREAKEVEN and not pos_state.breakeven_applied:
                    be_upd, new_sl = pos_state.check_breakeven(intra_fav, BREAKEVEN_TRIGGER, BREAKEVEN_OFFSET, be_r_mult)
                    if be_upd:
                        pos_state.stop_loss = new_sl
                        pos_state.breakeven_applied = True
                if USE_TRAILING_STOP:
                    tr_upd, new_sl = pos_state.check_trailing_stop(
                        intra_fav, high_p, low_p, TRAILING_STOP_TRIGGER, trailing_mult, TRAILING_STOP_FALLBACK, trail_r_mult
                    )
                    if tr_upd:
                        pos_state.stop_loss = new_sl

        mtm_eq = equity
        if pos_state is not None:
            mtm_dir = (close_p - pos_state.entry_price) if pos_state.is_long else (pos_state.entry_price - close_p)
            mtm_roi = (mtm_dir / pos_state.entry_price) * LEVERAGE * (pos_state.size / pos_state.initial_size)
            mtm_eq = max(0.01, equity + pos_margin * mtm_roi)
        equity_timeline.append((candle_time, mtm_eq))

    # MTM-переоценка незакрытой позиции на конце среза
    if pos_state is not None and entry_idx is not None:
        last_c = df.iloc[-1]
        last_close = float(last_c["close"])
        last_atr = float(last_c.get("ATR", 0.0)) or (last_close * 0.015)
        slip = calculate_dynamic_slippage(last_atr, last_close)
        actual_exit = last_close * (1 - slip if pos_state.is_long else 1 + slip)
        pnl_raw = (
            (actual_exit - pos_state.entry_price) / pos_state.entry_price
            if pos_state.is_long
            else (pos_state.entry_price - actual_exit) / pos_state.entry_price
        ) * LEVERAGE * 100
        fee_pct = (pos_state.entry_price + actual_exit) * TAKER_FEE * LEVERAGE / pos_state.entry_price * 100
        fund_pct = compute_funding_cost_pct(df["timestamp"].iloc[entry_idx], last_c["timestamp"], pos_state.is_long, LEVERAGE, funding_df)
        net_pnl = (pnl_raw - fee_pct - fund_pct) * (pos_state.size / pos_state.initial_size)
        equity = max(0.01, equity + pos_margin * (net_pnl / 100.0))
        trades.append({"pos_id": pos_id, "pnl_pct": net_pnl})
        equity_timeline[-1] = (last_c["timestamp"], equity)

    if not trades:
        return {"trades": 0, "win_rate": 0.0, "pnl_pct": 0.0, "sharpe_ratio": 0.0, "profit_factor": 0.0, "max_drawdown": 0.0}

    pos_count, win_rate, pf, _, _ = _aggregate_position_stats(trades)
    total_pnl, max_dd, daily_sharpe = _compute_daily_metrics(equity_timeline, initial_equity)

    return {
        "trades": pos_count,
        "win_rate": win_rate,
        "pnl_pct": total_pnl,
        "profit_factor": pf,
        "max_drawdown": max_dd,
        "sharpe_ratio": daily_sharpe,
    }


def _suggest_params_for_strategy(trial, strat_name: str) -> dict:
    """Формирует пространство поиска Optuna под конкретную активную стратегию."""
    params = {
        "atr_sl_mult": trial.suggest_float("atr_sl_mult", 1.5, 3.0, step=0.25),
        "atr_tp_mult": trial.suggest_float("atr_tp_mult", 3.0, 6.0, step=0.5),
        "trailing_atr_mult": trial.suggest_float("trailing_atr_mult", 1.25, 2.5, step=0.25),
    }
    if strat_name == "SUPERTREND_PRO":
        params.update({
            "st_period": trial.suggest_int("st_period", 7, 16, step=1),
            "st_multiplier": trial.suggest_float("st_multiplier", 2.0, 4.0, step=0.25),
            "confirm_ema_period": trial.suggest_int("confirm_ema_period", 20, 60, step=10),
            "adx_threshold": trial.suggest_float("adx_threshold", 12.0, 24.0, step=2.0),
            "confirm_lookback": trial.suggest_int("confirm_lookback", 1, 5),
        })
    elif strat_name == "SUPERTREND":
        params.update({
            "st_period": trial.suggest_int("st_period", 7, 16, step=1),
            "st_multiplier": trial.suggest_float("st_multiplier", 2.0, 4.0, step=0.25),
        })
    elif strat_name == "MA_RSI":
        fast_ma = trial.suggest_int("fast_ma", 10, 30, step=2)
        slow_ma = trial.suggest_int("slow_ma", 35, 90, step=5)
        if fast_ma >= slow_ma:
            raise optuna.TrialPruned()
        params.update({
            "fast_ma": fast_ma,
            "slow_ma": slow_ma,
            "rsi_period": trial.suggest_int("rsi_period", 9, 21, step=2),
        })
    elif strat_name == "BB_RSI":
        params.update({
            "bb_period": trial.suggest_int("bb_period", 14, 28, step=2),
            "bb_std": trial.suggest_float("bb_std", 1.5, 2.5, step=0.25),
            "rsi_period": trial.suggest_int("rsi_period", 9, 21, step=2),
        })
    elif strat_name == "DONCHIAN":
        params.update({
            "lookback_period": trial.suggest_int("lookback_period", 12, 36, step=4),
        })
    return params


def optimize_optuna(
    df_train: pd.DataFrame,
    df_test: pd.DataFrame | None,
    n_trials: int = 40,
    symbol: str = "BENCH",
    strategy_name: str | None = None,
    test_eval_start_idx: int | None = None,
) -> dict:
    """Оптимизация параметров с помощью Optuna с учетом активной стратегии."""
    strat_name = (strategy_name or ACTIVE_STRATEGY).strip().upper()
    min_trades = max(5, min(12, len(df_train) // 180))

    def objective(trial):
        params = _suggest_params_for_strategy(trial, strat_name)
        res = simulate_strategy_slice(df_train, params, symbol=symbol, strategy_name=strat_name)
        if res["trades"] < min_trades:
            return -99.0 + (res["trades"] / max(1, min_trades))
        score = res["sharpe_ratio"] * 0.5 + min(res["profit_factor"], 4.0) * 0.5 + (res["max_drawdown"] * 0.05)
        return score

    sampler = optuna.samplers.TPESampler(seed=42)
    study = optuna.create_study(direction="maximize", sampler=sampler)
    study.optimize(objective, n_trials=n_trials)

    best_params = study.best_params if study.best_trials else {}
    best_train_metrics = simulate_strategy_slice(df_train, best_params, symbol=symbol, strategy_name=strat_name)

    test_metrics = None
    if df_test is not None and not df_test.empty:
        test_metrics = simulate_strategy_slice(
            df_test, best_params, symbol=symbol, strategy_name=strat_name,
            eval_start_idx=test_eval_start_idx
        )

    return {
        "best_params": best_params,
        "train_metrics": best_train_metrics,
        "test_metrics": test_metrics,
        "best_score": study.best_value if study.best_trials else -99.0
    }


def optimize_grid(
    df_train: pd.DataFrame,
    df_test: pd.DataFrame | None,
    symbol: str = "BENCH",
    strategy_name: str | None = None,
    test_eval_start_idx: int | None = None,
) -> dict:
    """Fallback Grid Search с поддержкой всех стратегий."""
    strat_name = (strategy_name or ACTIVE_STRATEGY).strip().upper()
    candidates = []
    if strat_name in ("SUPERTREND_PRO", "SUPERTREND"):
        for sp in (8, 10, 14):
            for sm in (2.5, 3.0, 3.5):
                for sl in (1.8, 2.2):
                    for tp in (4.0, 4.5):
                        p = {"st_period": sp, "st_multiplier": sm, "atr_sl_mult": sl, "atr_tp_mult": tp, "trailing_atr_mult": 1.75}
                        if strat_name == "SUPERTREND_PRO":
                            for cl in (1, 3):
                                p2 = dict(p, confirm_lookback=cl, adx_threshold=18.0, confirm_ema_period=50)
                                candidates.append(p2)
                        else:
                            candidates.append(p)
    else:
        for f in (15, 20):
            for s in (40, 55):
                for sl in (1.5, 2.0):
                    for tp in (3.5, 4.5):
                        candidates.append({"fast_ma": f, "slow_ma": s, "rsi_period": 14, "atr_sl_mult": sl, "atr_tp_mult": tp, "trailing_atr_mult": 1.5})

    best_score = -999.0
    best_params = {}
    min_trades = max(5, min(12, len(df_train) // 180))

    for p in candidates:
        res = simulate_strategy_slice(df_train, p, symbol=symbol, strategy_name=strat_name)
        if res["trades"] < min_trades:
            continue
        score = res["sharpe_ratio"] * 0.5 + min(res["profit_factor"], 4.0) * 0.5 + (res["max_drawdown"] * 0.05)
        if score > best_score:
            best_score = score
            best_params = p

    train_metrics = simulate_strategy_slice(df_train, best_params, symbol=symbol, strategy_name=strat_name) if best_params else {}
    test_metrics = (
        simulate_strategy_slice(df_test, best_params, symbol=symbol, strategy_name=strat_name, eval_start_idx=test_eval_start_idx)
        if df_test is not None and not df_test.empty and best_params else None
    )
    return {
        "best_params": best_params,
        "train_metrics": train_metrics,
        "test_metrics": test_metrics,
        "best_score": best_score
    }


def run_rolling_wfa(
    df: pd.DataFrame,
    symbol: str,
    trials: int = 30,
    n_windows: int = 3,
    split_ratio: float = 0.7,
    strategy_name: str | None = None,
) -> dict:
    """
    Скользящий Walk-Forward Analysis по нескольким окнам с буфером прогрева перед каждым OOS-окном.
    """
    strat_name = (strategy_name or ACTIVE_STRATEGY).strip().upper()
    n = len(df)
    warmup_buf = min(250, max(50, n // 8))

    if n_windows <= 1 or n < warmup_buf + 400:
        split_idx = int(n * split_ratio)
        df_train = df.iloc[:split_idx].reset_index(drop=True)
        buf_start = max(0, split_idx - warmup_buf)
        df_test = df.iloc[buf_start:].reset_index(drop=True)
        eval_start = split_idx - buf_start
        res = (
            optimize_optuna(df_train, df_test, n_trials=trials, symbol=symbol, strategy_name=strat_name, test_eval_start_idx=eval_start)
            if OPTUNA_AVAILABLE else
            optimize_grid(df_train, df_test, symbol=symbol, strategy_name=strat_name, test_eval_start_idx=eval_start)
        )
        return {"windows": [res], "best_params": res["best_params"], "train_metrics": res["train_metrics"], "test_metrics": res["test_metrics"]}

    train_len = int(n * 0.55)
    test_len = max(120, (n - train_len) // n_windows)

    window_results = []
    for w in range(n_windows):
        tr_start = w * test_len
        tr_end = tr_start + train_len
        if tr_end >= n - 60:
            break
        te_end = n if (w == n_windows - 1) else min(n, tr_end + test_len)
        if te_end - tr_end < 60:
            break

        df_train = df.iloc[tr_start:tr_end].reset_index(drop=True)
        buf_start = max(0, tr_end - warmup_buf)
        df_test = df.iloc[buf_start:te_end].reset_index(drop=True)
        eval_start = tr_end - buf_start

        res = (
            optimize_optuna(df_train, df_test, n_trials=trials, symbol=symbol, strategy_name=strat_name, test_eval_start_idx=eval_start)
            if OPTUNA_AVAILABLE else
            optimize_grid(df_train, df_test, symbol=symbol, strategy_name=strat_name, test_eval_start_idx=eval_start)
        )
        res["train_range"] = (df_train["timestamp"].iloc[0].date(), df_train["timestamp"].iloc[-1].date())
        res["test_range"] = (df.iloc[tr_end]["timestamp"].date(), df.iloc[te_end - 1]["timestamp"].date())
        window_results.append(res)

    if not window_results:
        return run_rolling_wfa(df, symbol, trials=trials, n_windows=1, split_ratio=split_ratio, strategy_name=strat_name)

    last_res = window_results[-1]
    oos_list = [w["test_metrics"] for w in window_results if w.get("test_metrics")]
    oos_active = [m for m in oos_list if m.get("trades", 0) > 0]
    is_list = [w["train_metrics"] for w in window_results if w.get("train_metrics")]
    agg_oos = {
        "trades": sum(m["trades"] for m in oos_list),
        "win_rate": float(np.mean([m["win_rate"] for m in oos_active])) if oos_active else 0.0,
        "pnl_pct": float(sum(m["pnl_pct"] for m in oos_list)),
        "profit_factor": float(np.mean([m["profit_factor"] for m in oos_active])) if oos_active else 0.0,
        "max_drawdown": float(min(m["max_drawdown"] for m in oos_list)) if oos_list else 0.0,
        "sharpe_ratio": float(np.mean([m["sharpe_ratio"] for m in oos_list])) if oos_list else 0.0,
    }
    avg_is_sharpe = float(np.mean([m.get("sharpe_ratio", 0.0) for m in is_list])) if is_list else 0.0
    return {
        "windows": window_results,
        "best_params": last_res["best_params"],
        "train_metrics": {**(last_res.get("train_metrics") or {}), "sharpe_ratio": avg_is_sharpe},
        "test_metrics": agg_oos,
    }


def run_optimization(
    symbol: str = "BTCUSDT",
    days: int = 90,
    trials: int = 30,
    wfa: bool = True,
    split_ratio: float = 0.7,
    windows: int = 3,
    strategy_name: str | None = None,
):
    strat_name = (strategy_name or ACTIVE_STRATEGY).strip().upper()
    print("=" * 68)
    print(f"  ОПТИМИЗАЦИЯ СТРАТЕГИИ v2.3 ({strat_name}) — {symbol} ({days} дней)")
    print(f"  Движок: {'Optuna' if OPTUNA_AVAILABLE else 'Grid Search'} | WFA: {wfa} (Windows: {windows if wfa else 1})")
    print("=" * 68)

    df = load_or_download(symbol, INTERVAL, days)
    if df.empty or len(df) < 120:
        print(f"Недостаточно данных для оптимизации {symbol}")
        return None

    if USE_MTF_FILTER and strat_name != "BB_RSI":
        df_htf = load_or_download(symbol, HIGHER_INTERVAL, max(days, 60))
        df = align_htf_to_ltf(df, df_htf, INTERVAL, HIGHER_INTERVAL, HTF_EMA_PERIOD)

    if USE_MARKET_REGIME_FILTER:
        try:
            bench_df = load_or_download(BENCHMARK_SYMBOL, BENCHMARK_INTERVAL, max(days, 60))
            if bench_df is not None and not bench_df.empty:
                df = align_benchmark_to_ltf(df, bench_df, INTERVAL, BENCHMARK_INTERVAL)
        except Exception:
            pass

    if wfa:
        result = run_rolling_wfa(df, symbol=symbol, trials=trials, n_windows=windows, split_ratio=split_ratio, strategy_name=strat_name)
        print(f"\nВыполнено WFA-окон: {len(result['windows'])}")
        for idx, w in enumerate(result["windows"], 1):
            tr_m = w.get("train_metrics") or {}
            te_m = w.get("test_metrics") or {}
            rng = f" | {w.get('test_range', ('', ''))[0]}..{w.get('test_range', ('', ''))[1]}" if "test_range" in w else ""
            print(
                f"  Окно {idx}{rng}: IS Sharpe={tr_m.get('sharpe_ratio', 0):.2f} -> "
                f"OOS PnL={te_m.get('pnl_pct', 0):+.2f}%, WR={te_m.get('win_rate', 0):.1f}%, "
                f"PF={te_m.get('profit_factor', 0):.2f}, Sharpe={te_m.get('sharpe_ratio', 0):.2f}"
            )
    else:
        res = (
            optimize_optuna(df, None, n_trials=trials, symbol=symbol, strategy_name=strat_name)
            if OPTUNA_AVAILABLE else
            optimize_grid(df, None, symbol=symbol, strategy_name=strat_name)
        )
        result = {"windows": [res], **res}

    best = result["best_params"]
    tr = result["train_metrics"]
    te = result.get("test_metrics")

    print("\n" + "=" * 68)
    print("  РЕЗУЛЬТАТЫ ОПТИМИЗАЦИИ")
    print("=" * 68)
    print("Лучшие параметры (последнее окно):")
    for k, v in best.items():
        print(f"  {k:<22}: {v}")

    print("\nМетрики на обучающей выборке (In-Sample):")
    print(f"  Позиций:       {tr.get('trades')} (Pos WR: {tr.get('win_rate', 0):.1f}%)")
    print(f"  PnL:           {tr.get('pnl_pct', 0):+.2f}%")
    print(f"  Profit Factor: {tr.get('profit_factor', 0):.2f}")
    print(f"  Max Drawdown:  {tr.get('max_drawdown', 0):.2f}%")
    print(f"  Daily Sharpe:  {tr.get('sharpe_ratio', 0):.2f}")

    if te:
        print("\nАгрегированные метрики на валидации (Out-of-Sample / Rolling WFA):")
        print(f"  Позиций:       {te.get('trades')} (Pos WR: {te.get('win_rate', 0):.1f}%)")
        print(f"  PnL (сумма):   {te.get('pnl_pct', 0):+.2f}%")
        print(f"  Profit Factor: {te.get('profit_factor', 0):.2f}")
        print(f"  Max Drawdown:  {te.get('max_drawdown', 0):.2f}%")
        print(f"  Daily Sharpe:  {te.get('sharpe_ratio', 0):.2f}")

        is_sh = tr.get("sharpe_ratio", 0.0)
        wfa_eff = (te.get("sharpe_ratio", 0.0) / is_sh) if is_sh > 0 else 0.0
        print(f"\n  WFA Efficiency (OOS Sharpe / IS Sharpe): {wfa_eff:.2f}")
        if wfa_eff >= 0.5 and te.get("profit_factor", 0.0) > 1.3:
            print("  [OK] Стратегия устойчива к переобучению (WFA >= 50%, PF > 1.3).")
        else:
            print("  [ВНИМАНИЕ] Низкая устойчивость OOS или риск переобучения.")

    print("=" * 68 + "\n")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Параметрическая оптимизация торговой стратегии Bybit")
    parser.add_argument("symbol", nargs="?", default="BTCUSDT", help="Торговый символ (по умолчанию BTCUSDT)")
    parser.add_argument("--days", type=int, default=90, help="Количество дней истории (по умолчанию 90)")
    parser.add_argument("--trials", type=int, default=30, help="Количество итераций Optuna (по умолчанию 30)")
    parser.add_argument("--wfa", dest="wfa", action="store_true", default=True, help="Включить Walk-Forward Analysis")
    parser.add_argument("--no-wfa", dest="wfa", action="store_false", help="Отключить Walk-Forward Analysis")
    parser.add_argument("--split", type=float, default=0.7, help="Доля выборки In-Sample для одиночного WFA")
    parser.add_argument("--windows", type=int, default=3, help="Число скользящих окон WFA (по умолчанию 3)")
    parser.add_argument("--strategy", type=str, default=None, help="Стратегия для оптимизации (по умолчанию из config)")

    args = parser.parse_args()
    run_optimization(
        symbol=args.symbol,
        days=args.days,
        trials=args.trials,
        wfa=args.wfa,
        split_ratio=args.split,
        windows=args.windows,
        strategy_name=args.strategy,
    )
