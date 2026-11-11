"""
Модуль параметрической оптимизации торговой стратегии (v0.3.0)
Поддерживает:
- Оптимизацию гиперпараметров для ВСЕХ стратегий (SUPERTREND_PRO, SUPERTREND, MA_RSI, BB_RSI, DONCHIAN, REGIME_SWITCHER)
- Интеграцию MTF-фильтра (4H HTF_CLOSE vs HTF_EMA) и MarketRegimeFilter (BTC + Funding Z-score + OI)
- Позиционную группировку, Time-Stop, MTM на конце окна, Daily Sharpe Ratio и ограничение PF
- Скользящий Walk-Forward Analysis (Rolling WFA) по одной паре И по всему портфелю (Portfolio-Level Optuna + WFA)
- Monte Carlo bootstrap (500+ симуляций) и Deflated Sharpe Ratio (DSR)
- Версионируемые профили параметров и детектор деградации live-метрик

Использование:
    python optimizer.py ETHUSDT --wfa --trials 40 --days 180
    python optimizer.py --portfolio --trials 25 --days 90 --windows 3 --save-profile
"""

import argparse
import json
import math
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

import config
from config import (
    SYMBOLS, INTERVAL, HIGHER_INTERVAL, USE_MTF_FILTER, HTF_EMA_PERIOD,
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
from data_loader import (
    load_or_download, load_or_download_funding,
    load_or_download_open_interest, INTERVAL_MINUTES
)
from strategies import get_strategy
from risk_manager import PositionRiskState, MarketRegimeFilter
from backtest import (
    TAKER_FEE, MAKER_FEE,
    align_htf_to_ltf, align_benchmark_to_ltf,
    calculate_dynamic_slippage, compute_funding_cost_pct,
    compute_warmup_bars, _calc_position_margin_frac,
    _compute_daily_metrics, _aggregate_position_stats,
    run_portfolio_backtest,
)

try:
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False


def _default_strategy_name() -> str:
    """Возвращает эффективную стратегию с учетом переключателя USE_REGIME_SWITCHER."""
    active = str(getattr(config, "ACTIVE_STRATEGY", ACTIVE_STRATEGY)).strip().upper()
    if getattr(config, "USE_REGIME_SWITCHER", False) and active not in ("REGIME_SWITCHER", "ADAPTIVE"):
        return "REGIME_SWITCHER"
    return active


def _build_strategy(strat_name: str, params: dict):
    """Фабрика стратегии для одиночной и портфельной оптимизации."""
    use_mtf = USE_MTF_FILTER if strat_name != "BB_RSI" else bool(params.get("use_mtf_filter", False))
    return get_strategy(
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
        rsi_oversold=params.get("rsi_oversold", 35.0),
        rsi_overbought=params.get("rsi_overbought", 65.0),
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
        adx_trend_min=params.get("adx_trend_min", getattr(config, "SWITCHER_ADX_TREND_MIN", 22.0)),
        adx_range_max=params.get("adx_range_max", getattr(config, "SWITCHER_ADX_RANGE_MAX", 18.0)),
        er_period=params.get("er_period", getattr(config, "SWITCHER_ER_PERIOD", 14)),
        er_trend_min=params.get("er_trend_min", getattr(config, "SWITCHER_ER_TREND_MIN", 0.35)),
        er_range_max=params.get("er_range_max", getattr(config, "SWITCHER_ER_RANGE_MAX", 0.25)),
        bb_pctl_window=params.get("bb_pctl_window", getattr(config, "SWITCHER_BB_PCTL_WINDOW", 100)),
        mr_sl_mult=params.get("mr_sl_mult", getattr(config, "SWITCHER_MR_SL_MULT", 1.8)),
        mr_tp_mult=params.get("mr_tp_mult", getattr(config, "SWITCHER_MR_TP_MULT", 2.5)),
    )


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
    с прогревом индикаторов, Time-Stop, RegimeSwitcher MR-выходом, MTM-закрытием и Daily Sharpe Ratio.
    """
    strat_name = (strategy_name or params.get("strategy_name") or _default_strategy_name()).strip().upper()
    min_len = max(params.get("slow_ma", 50), params.get("confirm_ema_period", 50), 30)
    if df_slice.empty or len(df_slice) < min_len:
        return {"trades": 0, "win_rate": 0.0, "pnl_pct": 0.0, "sharpe_ratio": 0.0, "profit_factor": 0.0, "max_drawdown": 0.0}

    strategy = _build_strategy(strat_name, params)

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

        # 1. Выход по разворотному exit_signal или BB_middle (MR) на открытии бара
        mr_close = False
        if pos_state is not None and getattr(pos_state, "entry_regime", "") == "RANGE":
            if (pos_state.is_long and bool(prev_candle.get("mr_exit_long", False))) or (
                not pos_state.is_long and bool(prev_candle.get("mr_exit_short", False))
            ):
                mr_close = True
        if pos_state is not None and ((exit_sig == 1 and not pos_state.is_long) or (exit_sig == -1 and pos_state.is_long) or mr_close):
            slip = calculate_dynamic_slippage(atr, open_p)
            actual_exit = open_p * (1 + slip if not pos_state.is_long else 1 - slip)
            fee_pct = (pos_state.entry_price + actual_exit) * TAKER_FEE * LEVERAGE / pos_state.entry_price * 100
            pnl_raw = (
                (pos_state.entry_price - actual_exit) if not pos_state.is_long else (actual_exit - pos_state.entry_price)
            ) / pos_state.entry_price * LEVERAGE * 100
            fund_pct = compute_funding_cost_pct(df["timestamp"].iloc[entry_idx], candle_time, pos_state.is_long, LEVERAGE, funding_df)
            net_pnl = (pnl_raw - fee_pct - fund_pct) * (pos_state.size / pos_state.initial_size)
            usd_delta = pos_margin * (net_pnl / 100.0)
            equity = max(0.01, equity + usd_delta)
            trades.append({"pos_id": pos_id, "pnl_pct": net_pnl, "usd_delta": usd_delta})
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
            entry_regime = str(prev_candle.get("SIGNAL_REGIME", prev_candle.get("REGIME_MODE", "TREND"))).upper()
            stop_kwargs = dict(
                atr_value=atr, stop_loss_pct=STOP_LOSS_PCT, take_profit_pct=TAKE_PROFIT_PCT,
                atr_sl_mult=atr_sl_mult, atr_tp_mult=atr_tp_mult, use_atr=USE_ATR_STOPS,
            )
            if strat_name in ("REGIME_SWITCHER", "ADAPTIVE"):
                stop_kwargs["regime"] = entry_regime
            stops = strategy.calculate_stops(entry_p, "buy" if raw_signal == 1 else "sell", **stop_kwargs)
            pos_id += 1
            pos_margin = equity * _calc_position_margin_frac(entry_p, stops["sl_dist"])
            pos_state = PositionRiskState(
                symbol=symbol, side="Buy" if raw_signal == 1 else "Sell",
                entry_price=entry_p, size=1.0, initial_size=1.0,
                stop_loss=stops["sl"], take_profit=stops["tp"],
                atr_value=atr, r_mult=tp1_r_mult,
                strategy_tag=strat_name, entry_regime=entry_regime,
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
                        usd_delta = pos_margin * (net_pnl / 100.0)
                        equity = max(0.01, equity + usd_delta)
                        trades.append({"pos_id": pos_id, "pnl_pct": net_pnl, "usd_delta": usd_delta})
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

            if exit_p is None and params.get("use_time_stop", getattr(config, "USE_TIME_STOP", True)):
                bar_h = INTERVAL_MINUTES.get(str(INTERVAL), 60) / 60.0
                ts_act, ts_sl, _ = pos_state.check_time_stop(
                    current_price=close_p,
                    bars_held=int(round((i - entry_idx) * bar_h)),
                    max_hours=params.get("time_stop_hours", getattr(config, "TIME_STOP_HOURS", 30.0)),
                    hard_hours=params.get("time_stop_hard_hours", getattr(config, "TIME_STOP_HARD_HOURS", 36.0)),
                    min_r_progress=params.get("time_stop_min_r", getattr(config, "TIME_STOP_MIN_R", 0.3)),
                    offset_pct=BREAKEVEN_OFFSET,
                )
                if ts_act == "close":
                    exit_p = close_p
                elif ts_act == "tighten_be":
                    pos_state.stop_loss = ts_sl
                    pos_state.breakeven_applied = True
                    pos_state.time_stop_tightened = True

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
                usd_delta = pos_margin * (net_pnl / 100.0)
                equity = max(0.01, equity + usd_delta)
                trades.append({"pos_id": pos_id, "pnl_pct": net_pnl, "usd_delta": usd_delta})
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
                if USE_TRAILING_STOP and getattr(pos_state, "entry_regime", "") != "RANGE":
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
        usd_delta = pos_margin * (net_pnl / 100.0)
        equity = max(0.01, equity + usd_delta)
        trades.append({"pos_id": pos_id, "pnl_pct": net_pnl, "usd_delta": usd_delta})
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
        "tdf": pd.DataFrame(trades),
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
            "rsi_oversold": trial.suggest_float("rsi_oversold", 28.0, 38.0, step=2.0),
            "rsi_overbought": trial.suggest_float("rsi_overbought", 62.0, 72.0, step=2.0),
        })
    elif strat_name == "DONCHIAN":
        params.update({
            "lookback_period": trial.suggest_int("lookback_period", 12, 36, step=4),
        })
    elif strat_name in ("REGIME_SWITCHER", "ADAPTIVE"):
        adx_trend_min = trial.suggest_float("adx_trend_min", 20.0, 26.0, step=2.0)
        adx_range_max = trial.suggest_float("adx_range_max", 14.0, 20.0, step=2.0)
        if adx_range_max >= adx_trend_min:
            raise optuna.TrialPruned()
        params.update({
            "st_period": trial.suggest_int("st_period", 8, 14, step=2),
            "st_multiplier": trial.suggest_float("st_multiplier", 2.25, 3.75, step=0.25),
            "confirm_ema_period": trial.suggest_int("confirm_ema_period", 30, 60, step=10),
            "adx_trend_min": adx_trend_min,
            "adx_range_max": adx_range_max,
            "rsi_oversold": trial.suggest_float("rsi_oversold", 28.0, 38.0, step=2.0),
            "rsi_overbought": trial.suggest_float("rsi_overbought", 62.0, 72.0, step=2.0),
            "mr_sl_mult": trial.suggest_float("mr_sl_mult", 1.4, 2.2, step=0.2),
            "mr_tp_mult": trial.suggest_float("mr_tp_mult", 2.0, 3.2, step=0.2),
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
    strat_name = (strategy_name or _default_strategy_name()).strip().upper()
    min_trades = max(5, min(12, len(df_train) // 180))
    trial_sharpes: list[float] = []

    def objective(trial):
        params = _suggest_params_for_strategy(trial, strat_name)
        res = simulate_strategy_slice(df_train, params, symbol=symbol, strategy_name=strat_name)
        if res["trades"] < min_trades:
            return -99.0 + (res["trades"] / max(1, min_trades))
        trial_sharpes.append(float(res.get("sharpe_ratio", 0.0)))
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
        "best_score": study.best_value if study.best_trials else -99.0,
        "trial_sharpes": trial_sharpes,
    }


def _grid_candidates_for_strategy(strat_name: str) -> list[dict]:
    """Компактная сетка гиперпараметров для fallback Grid Search."""
    candidates = []
    if strat_name in ("SUPERTREND_PRO", "SUPERTREND", "REGIME_SWITCHER", "ADAPTIVE"):
        for sp in (8, 10, 14):
            for sm in (2.5, 3.0, 3.5):
                for sl in (1.8, 2.2):
                    for tp in (4.0, 4.5):
                        p = {"st_period": sp, "st_multiplier": sm, "atr_sl_mult": sl, "atr_tp_mult": tp, "trailing_atr_mult": 1.75}
                        if strat_name == "SUPERTREND_PRO":
                            for cl in (1, 3):
                                candidates.append(dict(p, confirm_lookback=cl, adx_threshold=18.0, confirm_ema_period=50))
                        elif strat_name in ("REGIME_SWITCHER", "ADAPTIVE"):
                            candidates.append(dict(p, confirm_lookback=1, adx_trend_min=22.0, adx_range_max=18.0, mr_sl_mult=1.8, mr_tp_mult=2.5))
                        else:
                            candidates.append(p)
    elif strat_name == "BB_RSI":
        for bp in (18, 20):
            for bs in (1.8, 2.0):
                for sl in (1.5, 2.0):
                    for tp in (2.5, 3.5):
                        candidates.append({"bb_period": bp, "bb_std": bs, "rsi_period": 14, "atr_sl_mult": sl, "atr_tp_mult": tp, "trailing_atr_mult": 1.5})
    elif strat_name == "DONCHIAN":
        for lb in (16, 20, 24):
            for sl in (1.8, 2.2):
                for tp in (3.5, 4.5):
                    candidates.append({"lookback_period": lb, "atr_sl_mult": sl, "atr_tp_mult": tp, "trailing_atr_mult": 1.75})
    else:
        for f in (15, 20):
            for s in (40, 55):
                for sl in (1.5, 2.0):
                    for tp in (3.5, 4.5):
                        candidates.append({"fast_ma": f, "slow_ma": s, "rsi_period": 14, "atr_sl_mult": sl, "atr_tp_mult": tp, "trailing_atr_mult": 1.5})
    return candidates


def optimize_grid(
    df_train: pd.DataFrame,
    df_test: pd.DataFrame | None,
    symbol: str = "BENCH",
    strategy_name: str | None = None,
    test_eval_start_idx: int | None = None,
) -> dict:
    """Fallback Grid Search с поддержкой всех стратегий."""
    strat_name = (strategy_name or _default_strategy_name()).strip().upper()
    candidates = _grid_candidates_for_strategy(strat_name)

    best_score = -999.0
    best_params = {}
    min_trades = max(5, min(12, len(df_train) // 180))
    trial_sharpes: list[float] = []

    for p in candidates:
        res = simulate_strategy_slice(df_train, p, symbol=symbol, strategy_name=strat_name)
        if res["trades"] < min_trades:
            continue
        trial_sharpes.append(float(res.get("sharpe_ratio", 0.0)))
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
        "best_score": best_score,
        "trial_sharpes": trial_sharpes,
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
    strat_name = (strategy_name or _default_strategy_name()).strip().upper()
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
        return {
            "windows": [res],
            "best_params": res["best_params"],
            "train_metrics": res["train_metrics"],
            "test_metrics": res["test_metrics"],
            "trial_sharpes": list(res.get("trial_sharpes") or []),
        }

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
    all_trial_sharpes: list[float] = []
    for w in window_results:
        all_trial_sharpes.extend(w.get("trial_sharpes") or [])

    oos_tdfs = []
    for w_idx, m in enumerate(oos_list):
        tdf_w = m.get("tdf")
        if isinstance(tdf_w, pd.DataFrame) and not tdf_w.empty:
            tdf_copy = tdf_w.copy()
            if "pos_id" in tdf_copy.columns:
                tdf_copy["pos_id"] = tdf_copy["pos_id"].apply(lambda pid, wi=w_idx: f"w{wi}_{pid}")
            oos_tdfs.append(tdf_copy)
    combined_oos_tdf = pd.concat(oos_tdfs, ignore_index=True) if oos_tdfs else pd.DataFrame()

    agg_oos = {
        "trades": sum(m["trades"] for m in oos_list),
        "win_rate": float(np.mean([m["win_rate"] for m in oos_active])) if oos_active else 0.0,
        "pnl_pct": float(sum(m["pnl_pct"] for m in oos_list)),
        "profit_factor": float(np.mean([m["profit_factor"] for m in oos_active])) if oos_active else 0.0,
        "max_drawdown": float(min(m["max_drawdown"] for m in oos_list)) if oos_list else 0.0,
        "sharpe_ratio": float(np.mean([m["sharpe_ratio"] for m in oos_list])) if oos_list else 0.0,
        "tdf": combined_oos_tdf,
    }
    avg_is_sharpe = float(np.mean([m.get("sharpe_ratio", 0.0) for m in is_list])) if is_list else 0.0
    return {
        "windows": window_results,
        "best_params": last_res["best_params"],
        "train_metrics": {**(last_res.get("train_metrics") or {}), "sharpe_ratio": avg_is_sharpe},
        "test_metrics": agg_oos,
        "trial_sharpes": all_trial_sharpes,
    }


# ==============================================================================
# v0.3.0: Deflated Sharpe Ratio, Monte Carlo, Portfolio WFA & Strategy Profiles
# ==============================================================================

def _norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(float(x) / math.sqrt(2.0)))


def _norm_ppf(p: float) -> float:
    """Рациональная аппроксимация обратной функции нормального распределения (Acklam)."""
    p = min(1.0 - 1e-12, max(1e-12, float(p)))
    a = [-3.969683028665376e01, 2.209460984245205e02, -2.759285104469687e02, 1.383577518672690e02, -3.066479806614716e01, 2.506628277459239e00]
    b = [-5.447609879822406e01, 1.615858368580409e02, -1.556989798598866e02, 6.680131188771972e01, -1.328068155288572e01]
    c = [-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e00, -2.549732539343734e00, 4.374664141464968e00, 2.938163982698783e00]
    d = [7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e00, 3.754408661907416e00]
    plow = 0.02425
    phigh = 1.0 - plow
    if p < plow:
        q = math.sqrt(-2.0 * math.log(p))
        return (((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) / ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1.0)
    if p <= phigh:
        q = p - 0.5
        r = q * q
        return (((((a[0]*r+a[1])*r+a[2])*r+a[3])*r+a[4])*r+a[5])*q / (((((b[0]*r+b[1])*r+b[2])*r+b[3])*r+b[4])*r+1.0)
    q = math.sqrt(-2.0 * math.log(1.0 - p))
    return -(((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) / ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1.0)


def compute_deflated_sharpe_ratio(
    observed_sharpe: float,
    n_trials: int = 1,
    daily_returns: list[float] | np.ndarray | None = None,
    n_obs: int | None = None,
    trial_sharpes: list[float] | None = None,
    annual_factor: float = 365.0,
) -> dict:
    """
    Вычисляет Deflated Sharpe Ratio (Bailey & Lopez de Prado, 2014) без внешних зависимостей.
    Оценивает вероятность того, что найденный Sharpe Ratio не является следствием множественного перебора.
    """
    ann_sqrt = math.sqrt(max(1.0, float(annual_factor)))
    sr_hat = float(observed_sharpe) / ann_sqrt
    skew = 0.0
    kurt = 3.0

    if daily_returns is not None and len(daily_returns) >= 4:
        arr = np.asarray(daily_returns, dtype=float)
        arr = arr[np.isfinite(arr)]
        n = max(4, len(arr))
        mean = float(np.mean(arr))
        std = float(np.std(arr, ddof=1)) if n > 1 else 0.0
        if std > 1e-12:
            z = (arr - mean) / std
            skew = float(np.mean(z ** 3))
            kurt = max(1.0, float(np.mean(z ** 4)))
    else:
        n = max(10, int(n_obs or 90))

    n_trials_eff = max(1, int(n_trials))
    if trial_sharpes is not None and len(trial_sharpes) >= 2:
        ts_arr = np.asarray([float(x) for x in trial_sharpes if x is not None and np.isfinite(x)], dtype=float) / ann_sqrt
        sr_trial_std = float(np.std(ts_arr, ddof=1)) if len(ts_arr) >= 2 else 0.0
        sr_trial_std = max(0.01, sr_trial_std)
    else:
        sr_trial_std = max(0.03, abs(sr_hat) * 0.25) if n_trials_eff > 1 else 0.0

    if n_trials_eff > 1 and sr_trial_std > 0:
        euler = 0.5772156649015329
        z1 = _norm_ppf(1.0 - 1.0 / n_trials_eff)
        z2 = _norm_ppf(1.0 - 1.0 / (n_trials_eff * math.e))
        sr0 = sr_trial_std * ((1.0 - euler) * z1 + euler * z2)
    else:
        sr0 = 0.0

    var_factor = max(1e-6, 1.0 - skew * sr_hat + ((kurt - 1.0) / 4.0) * (sr_hat ** 2))
    se = math.sqrt(var_factor / max(2, n - 1))
    dsr_prob = _norm_cdf((sr_hat - sr0) / se)
    psr_prob = _norm_cdf(sr_hat / se)

    return {
        "dsr": round(float(dsr_prob), 4),
        "psr": round(float(psr_prob), 4),
        "benchmark_sr": round(float(sr0 * ann_sqrt), 4),
        "observed_sr": round(float(observed_sharpe), 4),
        "is_significant": bool(dsr_prob >= 0.95),
        "n_trials": n_trials_eff,
        "n_obs": int(n),
    }


def run_monte_carlo_bootstrap(
    trades_or_tdf,
    n_sims: int = 500,
    initial_capital: float = 1000.0,
    seed: int = 42,
    ruin_dd_pct: float = -30.0,
) -> dict:
    """
    Bootstrap Monte Carlo (N перестановок с возвращением по закрытым позициям).
    Возвращает распределение доходности, MaxDD и вероятность прибыли / глубокой просадки.
    """
    cap0 = max(1.0, float(initial_capital))
    rets = np.array([], dtype=float)

    if isinstance(trades_or_tdf, pd.DataFrame):
        tdf = trades_or_tdf
        if not tdf.empty:
            if "pos_id" in tdf.columns and "usd_delta" in tdf.columns:
                grp = tdf.groupby("pos_id")["usd_delta"].sum().to_numpy(dtype=float)
                rets = grp / cap0
            elif "usd_delta" in tdf.columns:
                rets = tdf["usd_delta"].to_numpy(dtype=float) / cap0
            elif "pos_id" in tdf.columns and "equity_impact_pct" in tdf.columns:
                rets = tdf.groupby("pos_id")["equity_impact_pct"].sum().to_numpy(dtype=float) / 100.0
            elif "equity_impact_pct" in tdf.columns:
                rets = tdf["equity_impact_pct"].to_numpy(dtype=float) / 100.0
            elif "pos_id" in tdf.columns and "pnl_pct" in tdf.columns:
                rets = (tdf.groupby("pos_id")["pnl_pct"].sum().to_numpy(dtype=float) / 100.0) * float(getattr(config, "POSITION_SIZE_PCT", 0.25))
            elif "pnl_pct" in tdf.columns:
                rets = (tdf["pnl_pct"].to_numpy(dtype=float) / 100.0) * float(getattr(config, "POSITION_SIZE_PCT", 0.25))
    elif trades_or_tdf is not None:
        arr = np.asarray(list(trades_or_tdf), dtype=float)
        arr = arr[np.isfinite(arr)]
        rets = arr / 100.0

    rets = rets[np.isfinite(rets)]
    n_trades = int(len(rets))
    n_sims_eff = max(10, int(n_sims))
    if n_trades == 0:
        return {
            "n_sims": n_sims_eff,
            "n_trades": 0,
            "initial_capital": round(cap0, 2),
            "median_final_equity": round(cap0, 2),
            "median_return_pct": 0.0,
            "p5_return_pct": 0.0,
            "p95_return_pct": 0.0,
            "median_max_dd_pct": 0.0,
            "p5_max_dd_pct": 0.0,
            "worst_5pct_max_dd_pct": 0.0,
            "prob_profit_pct": 0.0,
            "prob_ruin_pct": 0.0,
        }

    step_mults = np.clip(1.0 + rets, 0.05, 10.0)
    rng = np.random.default_rng(seed)
    sampled_idx = rng.integers(0, n_trades, size=(n_sims_eff, n_trades))
    sampled_mults = step_mults[sampled_idx]

    paths = np.empty((n_sims_eff, n_trades + 1), dtype=float)
    paths[:, 0] = cap0
    paths[:, 1:] = cap0 * np.cumprod(sampled_mults, axis=1)

    final_eqs = paths[:, -1]
    ret_pcts = (final_eqs / cap0 - 1.0) * 100.0

    peaks = np.maximum.accumulate(paths, axis=1)
    drawdowns = (paths - peaks) / np.maximum(peaks, 1e-9) * 100.0
    max_dds = drawdowns.min(axis=1)

    worst_5_dd = float(np.percentile(max_dds, 5))
    return {
        "n_sims": n_sims_eff,
        "n_trades": n_trades,
        "initial_capital": round(cap0, 2),
        "median_final_equity": round(float(np.percentile(final_eqs, 50)), 2),
        "median_return_pct": round(float(np.percentile(ret_pcts, 50)), 2),
        "p5_return_pct": round(float(np.percentile(ret_pcts, 5)), 2),
        "p95_return_pct": round(float(np.percentile(ret_pcts, 95)), 2),
        "median_max_dd_pct": round(float(np.percentile(max_dds, 50)), 2),
        "p5_max_dd_pct": round(worst_5_dd, 2),
        "worst_5pct_max_dd_pct": round(worst_5_dd, 2),
        "prob_profit_pct": round(float(np.mean(final_eqs > cap0) * 100.0), 2),
        "prob_ruin_pct": round(float(np.mean(max_dds <= float(ruin_dd_pct)) * 100.0), 2),
    }


def prepare_portfolio_base_data(
    symbols: list[str] | None = None,
    days: int = 90,
    strategy_name: str | None = None,
    force_download: bool = False,
    use_regime: bool | None = None,
    use_mtf: bool | None = None,
) -> tuple[dict[str, pd.DataFrame], dict[str, pd.DataFrame | None]]:
    """
    Единоразовая загрузка и выравнивание OHLCV + 4H MTF + Benchmark Regime (с Funding Z-score и OI)
    для быстрого портфельного Optuna / Rolling WFA без повторных дисковых чтений.
    """
    syms = list(symbols or SYMBOLS)
    strat_name = (strategy_name or _default_strategy_name()).strip().upper()
    if use_mtf is None:
        use_mtf = USE_MTF_FILTER if strat_name != "BB_RSI" else False
    if use_regime is None:
        use_regime = USE_MARKET_REGIME_FILTER

    bench_df = None
    bench_f_df = None
    bench_o_df = None
    if use_regime:
        try:
            bench_df = load_or_download(BENCHMARK_SYMBOL, BENCHMARK_INTERVAL, max(days, 60), force_download)
            if getattr(config, "USE_FUNDING_ZSCORE_FILTER", True):
                bench_f_df = load_or_download_funding(BENCHMARK_SYMBOL, days=max(days, 90), allow_network=False)
            if getattr(config, "USE_OI_FILTER", True):
                bench_o_df = load_or_download_open_interest(BENCHMARK_SYMBOL, interval_time=BENCHMARK_INTERVAL, days=min(days, 60), allow_network=False)
            if bench_df is not None and not bench_df.empty:
                bench_df = MarketRegimeFilter.add_indicators(
                    bench_df,
                    ema_period=getattr(config, "BENCHMARK_EMA_PERIOD", 200),
                    dump_lookback_candles=getattr(config, "BENCHMARK_DUMP_CANDLES", 4),
                    dump_pct_threshold=getattr(config, "BENCHMARK_DUMP_PCT", 0.03),
                    pump_pct_threshold=getattr(config, "BENCHMARK_PUMP_PCT", 0.04),
                    rsi_threshold=getattr(config, "BENCHMARK_RSI_THRESHOLD", 45.0),
                    bull_rsi_threshold=getattr(config, "BENCHMARK_BULL_RSI_THRESHOLD", 55.0),
                    fast_ema_period=getattr(config, "REGIME_FAST_EMA_PERIOD", 50),
                    use_hybrid_trend=getattr(config, "REGIME_USE_HYBRID_TREND", True),
                    funding_df=bench_f_df,
                    oi_df=bench_o_df,
                    funding_zscore_window=getattr(config, "FUNDING_ZSCORE_WINDOW", 30),
                    funding_z_long_max=getattr(config, "FUNDING_ZSCORE_LONG_MAX", 2.0),
                    funding_z_short_min=getattr(config, "FUNDING_ZSCORE_SHORT_MIN", -2.0),
                    oi_lookback=getattr(config, "OI_LOOKBACK_PERIODS", 4),
                    oi_div_pct=getattr(config, "OI_DIVERGENCE_PCT", 0.03),
                    oi_price_move_pct=getattr(config, "OI_PRICE_MOVE_PCT", 0.015),
                    vol_window=getattr(config, "REGIME_VOL_WINDOW", 20),
                    extreme_vol_pct=getattr(config, "REGIME_EXTREME_VOL_PCT", 2.50),
                )
        except Exception:
            bench_df = None

    base_dfs: dict[str, pd.DataFrame] = {}
    funding_map: dict[str, pd.DataFrame | None] = {}
    for sym in syms:
        df = load_or_download(sym, INTERVAL, days, force_download)
        if df is None or df.empty or len(df) < 80:
            continue
        if use_mtf:
            df_htf = load_or_download(sym, HIGHER_INTERVAL, max(days, 60), force_download)
            if df_htf is not None and not df_htf.empty:
                df = align_htf_to_ltf(df, df_htf, INTERVAL, HIGHER_INTERVAL, HTF_EMA_PERIOD)
        if bench_df is not None and not bench_df.empty:
            df = align_benchmark_to_ltf(df, bench_df, INTERVAL, BENCHMARK_INTERVAL, funding_df=bench_f_df, oi_df=bench_o_df)
        base_dfs[sym] = df.sort_values("timestamp").reset_index(drop=True)
        try:
            funding_map[sym] = load_or_download_funding(sym, days=days, allow_network=False)
        except Exception:
            funding_map[sym] = None

    return base_dfs, funding_map


def _eval_portfolio_params(
    base_dfs: dict[str, pd.DataFrame],
    funding_map: dict[str, pd.DataFrame | None],
    strat_name: str,
    params: dict,
    eval_start_ts: pd.Timestamp | None = None,
    end_ts: pd.Timestamp | None = None,
    initial_capital: float = 1000.0,
) -> dict:
    strategy = _build_strategy(strat_name, params)
    prepared_override = {}
    for sym, raw_df in base_dfs.items():
        sub = raw_df[raw_df["timestamp"] <= end_ts] if end_ts is not None else raw_df
        if len(sub) < 60:
            continue
        ind_df = strategy.calculate_indicators(sub.copy()).reset_index(drop=True)
        warmup = compute_warmup_bars(strategy, len(ind_df))
        if len(ind_df) > warmup + 2:
            prepared_override[sym] = (ind_df, strategy, warmup)
    if not prepared_override:
        return {"symbols": [], "trades": 0, "pnl_pct": 0.0, "max_drawdown": 0.0, "sharpe_ratio": 0.0, "profit_factor": 0.0, "win_rate": 0.0, "tdf": pd.DataFrame()}

    eff_params = dict(params, strategy_name=strat_name)
    return run_portfolio_backtest(
        symbols=list(prepared_override.keys()),
        strategy_params=eff_params,
        initial_capital=initial_capital,
        verbose=False,
        prepared_override=prepared_override,
        funding_map_override=funding_map,
        eval_start_ts=eval_start_ts,
        end_ts=end_ts,
    )


def optimize_portfolio_optuna(
    base_dfs: dict[str, pd.DataFrame],
    funding_map: dict[str, pd.DataFrame | None],
    strategy_name: str | None = None,
    n_trials: int = 25,
    train_start_ts: pd.Timestamp | None = None,
    train_end_ts: pd.Timestamp | None = None,
    test_eval_start_ts: pd.Timestamp | None = None,
    test_end_ts: pd.Timestamp | None = None,
    initial_capital: float = 1000.0,
) -> dict:
    """Портфельная оптимизация параметров с помощью Optuna на общем капитале корзины пар."""
    strat_name = (strategy_name or _default_strategy_name()).strip().upper()
    min_trades = max(4, len(base_dfs) * 2)
    trial_sharpes: list[float] = []

    def objective(trial):
        params = _suggest_params_for_strategy(trial, strat_name)
        res = _eval_portfolio_params(
            base_dfs, funding_map, strat_name, params,
            eval_start_ts=train_start_ts, end_ts=train_end_ts, initial_capital=initial_capital,
        )
        if res["trades"] < min_trades:
            return -99.0 + (res["trades"] / max(1, min_trades))
        sh = float(res.get("sharpe_ratio", 0.0))
        trial_sharpes.append(sh)
        pf = min(float(res.get("profit_factor", 0.0)), 4.0)
        dd = float(res.get("max_drawdown", 0.0))
        ret = float(res.get("pnl_pct", 0.0))
        return sh * 0.50 + pf * 0.45 + (ret * 0.01) + (dd * 0.08)

    sampler = optuna.samplers.TPESampler(seed=42)
    study = optuna.create_study(direction="maximize", sampler=sampler)
    study.optimize(objective, n_trials=n_trials)

    best_params = dict(study.best_params) if study.best_trials else {}
    best_train = _eval_portfolio_params(
        base_dfs, funding_map, strat_name, best_params,
        eval_start_ts=train_start_ts, end_ts=train_end_ts, initial_capital=initial_capital,
    )
    best_test = None
    if test_eval_start_ts is not None and test_end_ts is not None:
        best_test = _eval_portfolio_params(
            base_dfs, funding_map, strat_name, best_params,
            eval_start_ts=test_eval_start_ts, end_ts=test_end_ts, initial_capital=initial_capital,
        )

    return {
        "best_params": best_params,
        "train_metrics": best_train,
        "test_metrics": best_test,
        "best_score": study.best_value if study.best_trials else -99.0,
        "trial_sharpes": trial_sharpes,
    }


def optimize_portfolio_grid(
    base_dfs: dict[str, pd.DataFrame],
    funding_map: dict[str, pd.DataFrame | None],
    strategy_name: str | None = None,
    train_start_ts: pd.Timestamp | None = None,
    train_end_ts: pd.Timestamp | None = None,
    test_eval_start_ts: pd.Timestamp | None = None,
    test_end_ts: pd.Timestamp | None = None,
    initial_capital: float = 1000.0,
) -> dict:
    """Fallback Grid Search для портфельной оптимизации без Optuna."""
    strat_name = (strategy_name or _default_strategy_name()).strip().upper()
    candidates = _grid_candidates_for_strategy(strat_name)
    min_trades = max(4, len(base_dfs) * 2)
    best_score = -999.0
    best_params: dict = {}
    trial_sharpes: list[float] = []

    for p in candidates:
        res = _eval_portfolio_params(
            base_dfs, funding_map, strat_name, p,
            eval_start_ts=train_start_ts, end_ts=train_end_ts, initial_capital=initial_capital,
        )
        if res["trades"] < min_trades:
            continue
        sh = float(res.get("sharpe_ratio", 0.0))
        trial_sharpes.append(sh)
        ret = float(res.get("pnl_pct", 0.0))
        score = sh * 0.50 + min(float(res.get("profit_factor", 0.0)), 4.0) * 0.45 + (ret * 0.01) + float(res.get("max_drawdown", 0.0)) * 0.08
        if score > best_score:
            best_score = score
            best_params = p

    best_train = _eval_portfolio_params(
        base_dfs, funding_map, strat_name, best_params,
        eval_start_ts=train_start_ts, end_ts=train_end_ts, initial_capital=initial_capital,
    ) if best_params else {}
    best_test = _eval_portfolio_params(
        base_dfs, funding_map, strat_name, best_params,
        eval_start_ts=test_eval_start_ts, end_ts=test_end_ts, initial_capital=initial_capital,
    ) if (best_params and test_eval_start_ts is not None and test_end_ts is not None) else None

    return {
        "best_params": best_params,
        "train_metrics": best_train,
        "test_metrics": best_test,
        "best_score": best_score,
        "trial_sharpes": trial_sharpes,
    }


def run_portfolio_rolling_wfa(
    symbols: list[str] | None = None,
    days: int = 90,
    trials: int = 25,
    n_windows: int = 3,
    split_ratio: float = 0.7,
    strategy_name: str | None = None,
    initial_capital: float = 1000.0,
    mc_sims: int = 500,
    force_download: bool = False,
    base_dfs_override: dict[str, pd.DataFrame] | None = None,
    funding_map_override: dict[str, pd.DataFrame | None] | None = None,
) -> dict:
    """
    Портфельный Rolling Walk-Forward Analysis + Monte Carlo bootstrap + Deflated Sharpe Ratio.
    """
    strat_name = (strategy_name or _default_strategy_name()).strip().upper()
    if base_dfs_override is not None:
        base_dfs = base_dfs_override
        funding_map = dict(funding_map_override or {})
    else:
        base_dfs, funding_map = prepare_portfolio_base_data(
            symbols=symbols, days=days, strategy_name=strat_name, force_download=force_download,
        )

    if not base_dfs:
        return {"windows": [], "best_params": {}, "train_metrics": {}, "test_metrics": None, "monte_carlo": None, "dsr": None}

    all_ts = sorted({ts for df in base_dfs.values() for ts in df["timestamp"].tolist()})
    n_ts = len(all_ts)
    if n_ts < 120:
        return {"windows": [], "best_params": {}, "train_metrics": {}, "test_metrics": None, "monte_carlo": None, "dsr": None}

    window_results = []
    all_trial_sharpes: list[float] = []

    if n_windows <= 1 or n_ts < 360:
        split_idx = max(80, min(n_ts - 40, int(n_ts * split_ratio)))
        train_end_ts = all_ts[split_idx - 1]
        oos_start_ts = all_ts[split_idx]
        oos_end_ts = all_ts[-1]
        res = (
            optimize_portfolio_optuna(
                base_dfs, funding_map, strategy_name=strat_name, n_trials=trials,
                train_start_ts=None, train_end_ts=train_end_ts,
                test_eval_start_ts=oos_start_ts, test_end_ts=oos_end_ts, initial_capital=initial_capital,
            )
            if OPTUNA_AVAILABLE else
            optimize_portfolio_grid(
                base_dfs, funding_map, strategy_name=strat_name,
                train_start_ts=None, train_end_ts=train_end_ts,
                test_eval_start_ts=oos_start_ts, test_end_ts=oos_end_ts, initial_capital=initial_capital,
            )
        )
        res["train_range"] = (pd.Timestamp(all_ts[0]).date(), pd.Timestamp(train_end_ts).date())
        res["test_range"] = (pd.Timestamp(oos_start_ts).date(), pd.Timestamp(oos_end_ts).date())
        all_trial_sharpes.extend(res.get("trial_sharpes") or [])
        window_results.append(res)
    else:
        train_len = int(n_ts * 0.55)
        test_len = max(72, (n_ts - train_len) // n_windows)
        for w in range(n_windows):
            tr_start_idx = w * test_len
            tr_end_idx = tr_start_idx + train_len
            if tr_end_idx >= n_ts - 40:
                break
            te_end_idx = n_ts if (w == n_windows - 1) else min(n_ts, tr_end_idx + test_len)
            if te_end_idx - tr_end_idx < 40:
                break

            train_start_ts = all_ts[tr_start_idx] if tr_start_idx > 0 else None
            train_end_ts = all_ts[tr_end_idx - 1]
            oos_start_ts = all_ts[tr_end_idx]
            oos_end_ts = all_ts[te_end_idx - 1]

            res = (
                optimize_portfolio_optuna(
                    base_dfs, funding_map, strategy_name=strat_name, n_trials=trials,
                    train_start_ts=train_start_ts, train_end_ts=train_end_ts,
                    test_eval_start_ts=oos_start_ts, test_end_ts=oos_end_ts, initial_capital=initial_capital,
                )
                if OPTUNA_AVAILABLE else
                optimize_portfolio_grid(
                    base_dfs, funding_map, strategy_name=strat_name,
                    train_start_ts=train_start_ts, train_end_ts=train_end_ts,
                    test_eval_start_ts=oos_start_ts, test_end_ts=oos_end_ts, initial_capital=initial_capital,
                )
            )
            res["train_range"] = (pd.Timestamp(all_ts[tr_start_idx]).date(), pd.Timestamp(train_end_ts).date())
            res["test_range"] = (pd.Timestamp(oos_start_ts).date(), pd.Timestamp(oos_end_ts).date())
            all_trial_sharpes.extend(res.get("trial_sharpes") or [])
            window_results.append(res)

        if not window_results:
            return run_portfolio_rolling_wfa(
                symbols=symbols, days=days, trials=trials, n_windows=1, split_ratio=split_ratio,
                strategy_name=strat_name, initial_capital=initial_capital, mc_sims=mc_sims,
                base_dfs_override=base_dfs, funding_map_override=funding_map,
            )

    last_res = window_results[-1]
    oos_list = [w["test_metrics"] for w in window_results if w.get("test_metrics")]
    oos_active = [m for m in oos_list if m.get("trades", 0) > 0]
    is_list = [w["train_metrics"] for w in window_results if w.get("train_metrics")]

    oos_tdfs = []
    for w_idx, m in enumerate(oos_list):
        tdf_w = m.get("tdf")
        if isinstance(tdf_w, pd.DataFrame) and not tdf_w.empty:
            tdf_copy = tdf_w.copy()
            if "pos_id" in tdf_copy.columns:
                tdf_copy["pos_id"] = tdf_copy["pos_id"].apply(lambda pid, wi=w_idx: f"w{wi}_{pid}")
            oos_tdfs.append(tdf_copy)
    combined_oos_tdf = pd.concat(oos_tdfs, ignore_index=True) if oos_tdfs else pd.DataFrame()
    mc_result = run_monte_carlo_bootstrap(combined_oos_tdf, n_sims=mc_sims, initial_capital=initial_capital) if not combined_oos_tdf.empty else None

    agg_oos = {
        "trades": sum(m.get("trades", 0) for m in oos_list),
        "win_rate": float(np.mean([m["win_rate"] for m in oos_active])) if oos_active else 0.0,
        "pnl_pct": float(sum(m.get("pnl_pct", 0.0) for m in oos_list)),
        "profit_factor": float(np.mean([m["profit_factor"] for m in oos_active])) if oos_active else 0.0,
        "max_drawdown": float(min(m.get("max_drawdown", 0.0) for m in oos_list)) if oos_list else 0.0,
        "sharpe_ratio": float(np.mean([m.get("sharpe_ratio", 0.0) for m in oos_list])) if oos_list else 0.0,
        "tdf": combined_oos_tdf,
    }
    avg_is_sharpe = float(np.mean([m.get("sharpe_ratio", 0.0) for m in is_list])) if is_list else 0.0
    total_trials_eval = max(1, len(all_trial_sharpes) or (trials * len(window_results)))
    dsr_result = compute_deflated_sharpe_ratio(
        observed_sharpe=agg_oos["sharpe_ratio"],
        n_trials=total_trials_eval,
        n_obs=max(30, n_ts // 24),
        trial_sharpes=all_trial_sharpes or None,
    )

    return {
        "windows": window_results,
        "symbols": list(base_dfs.keys()),
        "strategy": strat_name,
        "best_params": last_res["best_params"],
        "train_metrics": {**(last_res.get("train_metrics") or {}), "sharpe_ratio": avg_is_sharpe},
        "test_metrics": agg_oos,
        "monte_carlo": mc_result,
        "dsr": dsr_result,
    }


def save_strategy_profile(
    strategy_name: str,
    params: dict,
    oos_metrics: dict | None = None,
    dsr_result: dict | None = None,
    mc_result: dict | None = None,
    symbols: list[str] | None = None,
    profile_path: str | Path | None = None,
) -> dict:
    """Атомарно сохраняет версионированный профиль параметров стратегии (OOS baseline для live-мониторинга)."""
    path = Path(profile_path or getattr(config, "PROFILES_FILE", os.path.join("data", "strategy_profiles.json")))
    path.parent.mkdir(parents=True, exist_ok=True)
    strat_key = str(strategy_name).strip().upper()

    existing: dict = {"profiles": {}}
    if path.exists():
        try:
            with open(path, "r", encoding="utf-8") as f:
                loaded = json.load(f)
                if isinstance(loaded, dict):
                    existing = loaded
        except Exception:
            existing = {"profiles": {}}
    profiles = existing.setdefault("profiles", {})
    prev_ver = int((profiles.get(strat_key) or {}).get("version", 0) or 0)

    clean_oos = {}
    if isinstance(oos_metrics, dict):
        for k in ("trades", "win_rate", "pnl_pct", "profit_factor", "max_drawdown", "sharpe_ratio"):
            if k in oos_metrics and oos_metrics[k] is not None:
                clean_oos[k] = round(float(oos_metrics[k]), 4) if k != "trades" else int(oos_metrics[k])

    entry = {
        "version": prev_ver + 1,
        "strategy": strat_key,
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "symbols": list(symbols or []),
        "best_params": dict(params or {}),
        "oos_metrics": clean_oos,
        "dsr": dict(dsr_result) if dsr_result else None,
        "monte_carlo": dict(mc_result) if mc_result else None,
    }
    profiles[strat_key] = entry
    existing["active_strategy"] = strat_key
    existing["updated_at"] = entry["updated_at"]

    tmp_fd, tmp_path = tempfile.mkstemp(prefix=".profile_", suffix=".tmp", dir=str(path.parent))
    try:
        with os.fdopen(tmp_fd, "w", encoding="utf-8") as f:
            json.dump(existing, f, ensure_ascii=False, indent=2, default=str)
        os.replace(tmp_path, str(path))
    finally:
        if os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except Exception:
                pass
    return entry


def load_strategy_profile(strategy_name: str | None = None, profile_path: str | Path | None = None) -> dict | None:
    path = Path(profile_path or getattr(config, "PROFILES_FILE", os.path.join("data", "strategy_profiles.json")))
    if not path.exists():
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, dict):
            return None
        profiles = data.get("profiles") or {}
        if strategy_name:
            return profiles.get(str(strategy_name).strip().upper())
        active = data.get("active_strategy") or ACTIVE_STRATEGY
        return profiles.get(str(active).strip().upper())
    except Exception:
        return None


def detect_live_degradation(
    live_stats: dict,
    profile: dict | None = None,
    min_live_trades: int = 12,
) -> dict:
    """
    Детектор деградации live-метрик относительно OOS-бейзлайна стратегии.
    Возвращает status ('OK', 'WARNING', 'DEGRADED', 'INSUFFICIENT_DATA') и конкретные причины.
    """
    stats = live_stats or {}
    live_trades = int(
        stats.get("total_trades")
        if stats.get("total_trades") is not None
        else stats.get("closed_trades", stats.get("trades", 0)) or 0
    )
    oos = (profile or {}).get("oos_metrics") or {}
    oos_wr = float(oos.get("win_rate", 45.0) or 45.0)
    oos_pf = float(oos.get("profit_factor", 1.3) or 1.3)
    oos_dd = float(oos.get("max_drawdown", -12.0) or -12.0)

    baseline = {
        "version": (profile or {}).get("version"),
        "win_rate": round(oos_wr, 2),
        "profit_factor": round(oos_pf, 2),
        "max_drawdown": round(oos_dd, 2),
    }
    if live_trades < min_live_trades:
        return {
            "status": "INSUFFICIENT_DATA",
            "degraded": False,
            "reasons": [],
            "live_trades": live_trades,
            "min_trades_required": int(min_live_trades),
            "oos_baseline": baseline,
        }

    live_wr = float(stats.get("win_rate", 0.0) or 0.0)
    live_pf = float(stats.get("profit_factor", 0.0) or 0.0)
    live_dd = float(stats.get("max_drawdown_pct", stats.get("max_drawdown", 0.0)) or 0.0)

    reasons: list[str] = []
    pf_floor = max(0.85, oos_pf * 0.65)
    if live_pf < pf_floor:
        reasons.append(f"Profit Factor {live_pf:.2f} < {pf_floor:.2f} (OOS={oos_pf:.2f})")
    if live_wr < (oos_wr - 15.0):
        reasons.append(f"WinRate {live_wr:.1f}% ниже OOS {oos_wr:.1f}% более чем на 15 п.п.")
    dd_limit = min(-15.0, oos_dd * 1.5)
    if live_dd <= dd_limit:
        reasons.append(f"Live MaxDD {live_dd:.1f}% хуже порога {dd_limit:.1f}%")

    if len(reasons) >= 2 or (live_pf < 0.75 and live_trades >= min_live_trades):
        status = "DEGRADED"
        degraded = True
    elif reasons:
        status = "WARNING"
        degraded = True
    else:
        status = "OK"
        degraded = False

    return {
        "status": status,
        "degraded": degraded,
        "reasons": reasons,
        "live_trades": live_trades,
        "oos_baseline": baseline,
    }


def run_optimization(
    symbol: str = "BTCUSDT",
    days: int = 90,
    trials: int = 30,
    wfa: bool = True,
    split_ratio: float = 0.7,
    windows: int = 3,
    strategy_name: str | None = None,
    portfolio: bool = False,
    symbols: list[str] | None = None,
    save_profile: bool = False,
    mc_sims: int = 500,
):
    strat_name = (strategy_name or _default_strategy_name()).strip().upper()
    print("=" * 68)
    if portfolio:
        target_syms = list(symbols or SYMBOLS)
        print(f"  ПОРТФЕЛЬНАЯ ОПТИМИЗАЦИЯ v0.3.0 ({strat_name}) — {len(target_syms)} пар ({days} дней)")
    else:
        target_syms = [symbol]
        print(f"  ОПТИМИЗАЦИЯ СТРАТЕГИИ v0.3.0 ({strat_name}) — {symbol} ({days} дней)")
    print(f"  Движок: {'Optuna' if OPTUNA_AVAILABLE else 'Grid Search'} | WFA: {wfa} (Windows: {windows if wfa else 1})")
    print("=" * 68)

    if portfolio:
        result = run_portfolio_rolling_wfa(
            symbols=target_syms,
            days=days,
            trials=trials,
            n_windows=(windows if wfa else 1),
            split_ratio=split_ratio,
            strategy_name=strat_name,
            mc_sims=mc_sims,
        )
        if not result.get("windows"):
            print("Недостаточно данных для портфельной оптимизации")
            return None
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
                    bench_f_df = (
                        load_or_download_funding(BENCHMARK_SYMBOL, days=max(days, 90), allow_network=False)
                        if getattr(config, "USE_FUNDING_ZSCORE_FILTER", True) else None
                    )
                    bench_o_df = (
                        load_or_download_open_interest(BENCHMARK_SYMBOL, interval_time=BENCHMARK_INTERVAL, days=min(days, 60), allow_network=False)
                        if getattr(config, "USE_OI_FILTER", True) else None
                    )
                    df = align_benchmark_to_ltf(
                        df, bench_df, INTERVAL, BENCHMARK_INTERVAL,
                        funding_df=bench_f_df, oi_df=bench_o_df,
                    )
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

        if result.get("test_metrics"):
            te_tdf = (result["test_metrics"] or {}).get("tdf")
            if not isinstance(te_tdf, pd.DataFrame) or te_tdf.empty:
                win_tdfs = []
                for w_idx, w in enumerate(result.get("windows", [])):
                    wt = (w.get("test_metrics") or {}).get("tdf")
                    if isinstance(wt, pd.DataFrame) and not wt.empty:
                        wc = wt.copy()
                        if "pos_id" in wc.columns:
                            wc["pos_id"] = wc["pos_id"].apply(lambda pid, wi=w_idx: f"w{wi}_{pid}")
                        win_tdfs.append(wc)
                te_tdf = pd.concat(win_tdfs, ignore_index=True) if win_tdfs else pd.DataFrame()
            if isinstance(te_tdf, pd.DataFrame) and not te_tdf.empty and mc_sims > 0:
                result["monte_carlo"] = run_monte_carlo_bootstrap(te_tdf, n_sims=mc_sims)
            trial_sharpes_all = list(result.get("trial_sharpes") or [])
            if not trial_sharpes_all:
                for w in result.get("windows", []):
                    trial_sharpes_all.extend(w.get("trial_sharpes") or [])
            result["dsr"] = compute_deflated_sharpe_ratio(
                observed_sharpe=float(result["test_metrics"].get("sharpe_ratio", 0.0)),
                n_trials=max(1, len(trial_sharpes_all) or (trials * max(1, len(result.get("windows", []))))),
                n_obs=max(30, len(df) // 24),
                trial_sharpes=trial_sharpes_all or None,
            )

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

    dsr = result.get("dsr")
    if dsr:
        print(f"  Deflated Sharpe (DSR): {dsr['dsr']:.3f} | Sig>=0.95: {dsr['is_significant']} (Benchmark SR={dsr['benchmark_sr']:.2f})")
    mc = result.get("monte_carlo")
    if mc and mc.get("n_trades", 0) > 0:
        print(
            f"  Monte Carlo ({mc['n_sims']}x): Median Ret={mc['median_return_pct']:+.1f}% "
            f"(P5={mc['p5_return_pct']:+.1f}%..P95={mc['p95_return_pct']:+.1f}%), "
            f"Worst5% MaxDD={mc['worst_5pct_max_dd_pct']:.1f}%, P(Profit)={mc['prob_profit_pct']:.1f}%"
        )

    if save_profile and best:
        saved = save_strategy_profile(
            strategy_name=strat_name,
            params=best,
            oos_metrics=te or tr,
            dsr_result=dsr,
            mc_result=mc,
            symbols=target_syms,
        )
        print(f"  Профиль v{saved['version']} сохранен в {getattr(config, 'PROFILES_FILE', 'data/strategy_profiles.json')}")

    print("=" * 68 + "\n")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Параметрическая оптимизация торговой стратегии Bybit v0.3.0")
    parser.add_argument("symbol", nargs="?", default="BTCUSDT", help="Торговый символ (по умолчанию BTCUSDT)")
    parser.add_argument("--days", type=int, default=90, help="Количество дней истории (по умолчанию 90)")
    parser.add_argument("--trials", type=int, default=30, help="Количество итераций Optuna (по умолчанию 30)")
    parser.add_argument("--wfa", dest="wfa", action="store_true", default=True, help="Включить Walk-Forward Analysis")
    parser.add_argument("--no-wfa", dest="wfa", action="store_false", help="Отключить Walk-Forward Analysis")
    parser.add_argument("--split", type=float, default=0.7, help="Доля выборки In-Sample для одиночного WFA")
    parser.add_argument("--windows", type=int, default=3, help="Число скользящих окон WFA (по умолчанию 3)")
    parser.add_argument("--strategy", type=str, default=None, help="Стратегия для оптимизации (по умолчанию из config)")
    parser.add_argument("--portfolio", action="store_true", default=False, help="Портфельная оптимизация сразу по корзине пар")
    parser.add_argument("--symbols", type=str, default=None, help="Список пар через запятую для --portfolio")
    parser.add_argument("--save-profile", action="store_true", default=False, help="Сохранить версионированный профиль стратегии")
    parser.add_argument("--mc-sims", type=int, default=500, help="Число Monte Carlo симуляций (по умолчанию 500)")

    args = parser.parse_args()
    sym_list = [s.strip().upper() for s in args.symbols.split(",") if s.strip()] if args.symbols else None
    run_optimization(
        symbol=args.symbol,
        days=args.days,
        trials=args.trials,
        wfa=args.wfa,
        split_ratio=args.split,
        windows=args.windows,
        strategy_name=args.strategy,
        portfolio=args.portfolio,
        symbols=sym_list,
        save_profile=args.save_profile,
        mc_sims=args.mc_sims,
    )
