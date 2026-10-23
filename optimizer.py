"""
Модуль параметрической оптимизации торговой стратегии (v2.2)
Поддерживает:
- Оптимизацию гиперпараметров через Optuna (или Grid Search)
- Целевые функции: Sharpe Ratio, Profit Factor, минимизация Max Drawdown
- Walk-Forward Analysis (WFA): разбиение данных на In-Sample (обучение) и Out-of-Sample (валидация)

Использование:
    python optimizer.py SOLUSDT --trials 30 --days 90
    python optimizer.py ETHUSDT --wfa --split 0.7
"""

import sys
import argparse
import numpy as np
import pandas as pd
from datetime import datetime

from config import (
    INTERVAL, HIGHER_INTERVAL, USE_MTF_FILTER, HTF_EMA_PERIOD,
    ST_PERIOD, ST_MULTIPLIER, ST_CONFIRM_EMA,
    ATR_PERIOD, ATR_SL_MULT, ATR_TP_MULT, TRAILING_STOP_ATR_MULT,
    LEVERAGE, ACTIVE_STRATEGY,
    RSI_LONG_MIN, RSI_LONG_MAX, RSI_SHORT_MIN, RSI_SHORT_MAX, VOLUME_FILTER,
    USE_TRAILING_STOP, TRAILING_STOP_TRIGGER, TRAILING_STOP_FALLBACK,
    USE_PARTIAL_TP, PARTIAL_TP_R_MULT, PARTIAL_TP_RATIO, PARTIAL_TP_MOVE_BE,
    STOP_LOSS_PCT, TAKE_PROFIT_PCT, USE_ATR_STOPS,
    SIGNAL_LOOKBACK_CANDLES, USE_BREAKEVEN, BREAKEVEN_TRIGGER, BREAKEVEN_OFFSET,
    USE_ADX_FILTER, ADX_PERIOD, ADX_THRESHOLD
)
from data_loader import load_or_download
from strategies import get_strategy
from risk_manager import PositionRiskState

try:
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False


TAKER_FEE = 0.00055
BASE_SLIPPAGE = 0.0002
FUNDING_RATE = 0.0001


def simulate_strategy_slice(
    df_slice: pd.DataFrame,
    params: dict,
    symbol: str = "BENCH"
) -> dict:
    """
    Быстрая симуляция стратегии на срезе данных (In-Sample или Out-of-Sample).
    """
    if df_slice.empty or len(df_slice) < params.get("slow_ma", 50) + 10:
        return {"trades": 0, "pnl_pct": 0.0, "sharpe_ratio": -99.0, "profit_factor": 0.0, "max_drawdown": -99.0}

    strategy = get_strategy(
        ACTIVE_STRATEGY,
        st_period=params.get("st_period", ST_PERIOD),
        st_multiplier=params.get("st_multiplier", ST_MULTIPLIER),
        confirm_ema_period=params.get("confirm_ema_period", ST_CONFIRM_EMA),
        fast_ma=params.get("fast_ma", 20),
        slow_ma=params.get("slow_ma", 50),
        rsi_period=params.get("rsi_period", 14),
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

    df = strategy.calculate_indicators(df_slice.copy())
    req_cols = [c for c in ["MA_fast", "MA_slow", "RSI", "signal"] if c in df.columns]
    if req_cols:
        df = df.dropna(subset=req_cols).reset_index(drop=True)

    if len(df) < 10:
        return {"trades": 0, "pnl_pct": 0.0, "sharpe_ratio": -99.0, "profit_factor": 0.0, "max_drawdown": -99.0}

    atr_sl_mult = params.get("atr_sl_mult", ATR_SL_MULT)
    atr_tp_mult = params.get("atr_tp_mult", ATR_TP_MULT)
    trailing_mult = params.get("trailing_atr_mult", TRAILING_STOP_ATR_MULT)

    trades = []
    pos_state: PositionRiskState | None = None
    entry_idx = None

    for i in range(1, len(df)):
        candle = df.iloc[i]
        open_p = float(candle["open"])
        price = float(candle["close"])
        low_p = float(candle["low"])
        high_p = float(candle["high"])
        candle_time = candle["timestamp"]

        prev_candle = df.iloc[i - 1]
        raw_signal = int(prev_candle.get("signal", 0))
        atr = float(prev_candle.get("ATR", 0.0))
        if atr <= 0:
            atr = open_p * 0.015

        # 1. Разворот или открытие позиции на ОТКРЫТИИ свечи i (без Lookahead Bias)
        if raw_signal == 1 and pos_state is not None and not pos_state.is_long:
            slip = max(BASE_SLIPPAGE, min((atr / open_p) * 0.05, 0.003)) if open_p > 0 else BASE_SLIPPAGE
            actual_exit = open_p * (1 + slip)
            fee_pct = (pos_state.entry_price + actual_exit) * TAKER_FEE * LEVERAGE / pos_state.entry_price * 100
            pnl_raw = (pos_state.entry_price - actual_exit) / pos_state.entry_price * LEVERAGE * 100
            hours = (candle_time - df["timestamp"].iloc[entry_idx]).total_seconds() / 3600.0 if entry_idx is not None else 0.0
            funding_cost_pct = int(max(0.0, hours) // 8) * FUNDING_RATE * LEVERAGE * 100 * (-1)
            trades.append((pnl_raw - fee_pct - funding_cost_pct) * (pos_state.size / pos_state.initial_size))
            pos_state = None
            entry_idx = None

        elif raw_signal == -1 and pos_state is not None and pos_state.is_long:
            slip = max(BASE_SLIPPAGE, min((atr / open_p) * 0.05, 0.003)) if open_p > 0 else BASE_SLIPPAGE
            actual_exit = open_p * (1 - slip)
            fee_pct = (pos_state.entry_price + actual_exit) * TAKER_FEE * LEVERAGE / pos_state.entry_price * 100
            pnl_raw = (actual_exit - pos_state.entry_price) / pos_state.entry_price * LEVERAGE * 100
            hours = (candle_time - df["timestamp"].iloc[entry_idx]).total_seconds() / 3600.0 if entry_idx is not None else 0.0
            funding_cost_pct = int(max(0.0, hours) // 8) * FUNDING_RATE * LEVERAGE * 100 * (1)
            trades.append((pnl_raw - fee_pct - funding_cost_pct) * (pos_state.size / pos_state.initial_size))
            pos_state = None
            entry_idx = None

        if pos_state is None and raw_signal != 0:
            slip = max(BASE_SLIPPAGE, min((atr / open_p) * 0.05, 0.003)) if open_p > 0 else BASE_SLIPPAGE
            if raw_signal == 1:
                entry_p = open_p * (1 + slip)
                stops = strategy.calculate_stops(
                    entry_p, "buy", atr_value=atr,
                    stop_loss_pct=STOP_LOSS_PCT, take_profit_pct=TAKE_PROFIT_PCT,
                    atr_sl_mult=atr_sl_mult, atr_tp_mult=atr_tp_mult,
                    use_atr=USE_ATR_STOPS
                )
                pos_state = PositionRiskState(
                    symbol=symbol, side="Buy", entry_price=entry_p,
                    size=1.0, initial_size=1.0,
                    stop_loss=stops["sl"], take_profit=stops["tp"],
                    atr_value=atr, r_mult=PARTIAL_TP_R_MULT
                )
                entry_idx = i
            elif raw_signal == -1:
                entry_p = open_p * (1 - slip)
                stops = strategy.calculate_stops(
                    entry_p, "sell", atr_value=atr,
                    stop_loss_pct=STOP_LOSS_PCT, take_profit_pct=TAKE_PROFIT_PCT,
                    atr_sl_mult=atr_sl_mult, atr_tp_mult=atr_tp_mult,
                    use_atr=USE_ATR_STOPS
                )
                pos_state = PositionRiskState(
                    symbol=symbol, side="Sell", entry_price=entry_p,
                    size=1.0, initial_size=1.0,
                    stop_loss=stops["sl"], take_profit=stops["tp"],
                    atr_value=atr, r_mult=PARTIAL_TP_R_MULT
                )
                entry_idx = i

        # 2. Внутрисвечевая симуляция без заглядывания вперед
        if pos_state is not None:
            exit_p = None
            if pos_state.is_long:
                if low_p <= pos_state.stop_loss:
                    exit_p = pos_state.stop_loss
            else:
                if high_p >= pos_state.stop_loss:
                    exit_p = pos_state.stop_loss

            if exit_p is None:
                # Partial TP
                if USE_PARTIAL_TP and not pos_state.tp1_executed:
                    triggered, qty_c, be_sl = pos_state.check_partial_tp(
                        high_p if pos_state.is_long else low_p,
                        ratio=PARTIAL_TP_RATIO,
                        offset_pct=BREAKEVEN_OFFSET
                    )
                    if triggered:
                        tp1_target = pos_state.tp1_price
                        fee_pct = (pos_state.entry_price + tp1_target) * TAKER_FEE * LEVERAGE / pos_state.entry_price * 100
                        pnl_raw = ((tp1_target - pos_state.entry_price) / pos_state.entry_price if pos_state.is_long else (pos_state.entry_price - tp1_target) / pos_state.entry_price) * LEVERAGE * 100
                        hours = (candle_time - df["timestamp"].iloc[entry_idx]).total_seconds() / 3600.0 if entry_idx is not None else 0.0
                        funding_cost_pct = int(max(0.0, hours) // 8) * FUNDING_RATE * LEVERAGE * 100 * (1 if pos_state.is_long else -1)
                        net_pnl = (pnl_raw - fee_pct - funding_cost_pct) * PARTIAL_TP_RATIO
                        trades.append(net_pnl)
                        pos_state.tp1_executed = True
                        pos_state.size -= qty_c
                        if PARTIAL_TP_MOVE_BE:
                            pos_state.stop_loss = be_sl
                            pos_state.breakeven_applied = True

                if pos_state.is_long and high_p >= pos_state.take_profit:
                    exit_p = pos_state.take_profit
                elif not pos_state.is_long and low_p <= pos_state.take_profit:
                    exit_p = pos_state.take_profit

            if exit_p is not None:
                slip = max(BASE_SLIPPAGE, min((atr / exit_p) * 0.05, 0.003)) if exit_p > 0 and atr > 0 else BASE_SLIPPAGE
                actual_exit = exit_p * (1 - slip if pos_state.is_long else 1 + slip)
                pnl_raw = ((actual_exit - pos_state.entry_price) / pos_state.entry_price if pos_state.is_long else (pos_state.entry_price - actual_exit) / pos_state.entry_price) * LEVERAGE * 100
                fee_pct = (pos_state.entry_price + actual_exit) * TAKER_FEE * LEVERAGE / pos_state.entry_price * 100
                hours = (candle_time - df["timestamp"].iloc[entry_idx]).total_seconds() / 3600.0 if entry_idx is not None else 0.0
                funding_cost_pct = int(max(0.0, hours) // 8) * FUNDING_RATE * LEVERAGE * 100 * (1 if pos_state.is_long else -1)
                weight = pos_state.size / pos_state.initial_size
                trades.append((pnl_raw - fee_pct - funding_cost_pct) * weight)
                pos_state = None
                entry_idx = None
            else:
                pos_state.update_price_extremes(high_p, low_p)

                # Breakeven
                if USE_BREAKEVEN and not pos_state.breakeven_applied:
                    be_updated, new_sl = pos_state.check_breakeven(
                        current_price=price,
                        trigger_pct=BREAKEVEN_TRIGGER,
                        offset_pct=BREAKEVEN_OFFSET
                    )
                    if be_updated:
                        pos_state.stop_loss = new_sl
                        pos_state.breakeven_applied = True

                # Trailing stop
                if USE_TRAILING_STOP:
                    trail_updated, new_sl = pos_state.check_trailing_stop(
                        current_price=price,
                        high_price=high_p,
                        low_price=low_p,
                        trigger_pct=TRAILING_STOP_TRIGGER,
                        atr_mult=trailing_mult,
                        fallback_pct=TRAILING_STOP_FALLBACK
                    )
                    if trail_updated:
                        pos_state.stop_loss = new_sl

    if not trades:
        return {"trades": 0, "pnl_pct": 0.0, "sharpe_ratio": -99.0, "profit_factor": 0.0, "max_drawdown": -99.0}

    ts = np.array(trades)
    total_trades = len(ts)
    wins = np.sum(ts > 0)
    win_rate = (wins / total_trades * 100) if total_trades else 0.0
    total_pnl = float(np.sum(ts))
    cum = np.cumsum(ts)
    max_dd = float(np.min(cum - np.maximum.accumulate(cum)))

    pos_sum = np.sum(ts[ts > 0])
    neg_sum = abs(np.sum(ts[ts < 0]))
    pf = float(pos_sum / neg_sum) if neg_sum > 0 else (50.0 if pos_sum > 0 else 0.0)

    std = np.std(ts)
    sharpe = float((np.mean(ts) / std * np.sqrt(total_trades)) if std > 0 else 0.0)

    return {
        "trades": total_trades,
        "win_rate": win_rate,
        "pnl_pct": total_pnl,
        "profit_factor": pf,
        "max_drawdown": max_dd,
        "sharpe_ratio": sharpe
    }


def optimize_optuna(df_train: pd.DataFrame, df_test: pd.DataFrame | None, n_trials: int = 40) -> dict:
    """
    Оптимизация параметров с помощью Optuna.
    """
    def objective(trial):
        fast_ma = trial.suggest_int("fast_ma", 10, 30, step=2)
        slow_ma = trial.suggest_int("slow_ma", 35, 90, step=5)
        if fast_ma >= slow_ma:
            raise optuna.TrialPruned()

        rsi_period = trial.suggest_int("rsi_period", 9, 21, step=2)
        atr_sl_mult = trial.suggest_float("atr_sl_mult", 1.0, 3.0, step=0.25)
        atr_tp_mult = trial.suggest_float("atr_tp_mult", 2.0, 5.5, step=0.5)
        trailing_atr_mult = trial.suggest_float("trailing_atr_mult", 1.0, 2.5, step=0.25)

        params = {
            "fast_ma": fast_ma,
            "slow_ma": slow_ma,
            "rsi_period": rsi_period,
            "atr_sl_mult": atr_sl_mult,
            "atr_tp_mult": atr_tp_mult,
            "trailing_atr_mult": trailing_atr_mult
        }

        res = simulate_strategy_slice(df_train, params)
        if res["trades"] < 5:
            return -99.0

        # Композитная целевая функция: максимизация Шарпа и PF, штраф за просадку
        score = res["sharpe_ratio"] * 0.6 + min(res["profit_factor"], 4.0) * 0.3 + (res["max_drawdown"] * 0.01)
        return score

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials)

    best_params = study.best_params
    best_train_metrics = simulate_strategy_slice(df_train, best_params)

    test_metrics = None
    if df_test is not None and not df_test.empty:
        test_metrics = simulate_strategy_slice(df_test, best_params)

    return {
        "best_params": best_params,
        "train_metrics": best_train_metrics,
        "test_metrics": test_metrics,
        "best_score": study.best_value
    }


def optimize_grid(df_train: pd.DataFrame, df_test: pd.DataFrame | None) -> dict:
    """Fallback Grid Search когда Optuna недоступна."""
    fast_mas = [15, 20, 25]
    slow_mas = [40, 50, 60]
    rsi_periods = [10, 14]
    atr_sls = [1.25, 1.5, 2.0]
    atr_tps = [3.0, 3.5, 4.0]

    best_score = -999.0
    best_params = {}

    for f in fast_mas:
        for s in slow_mas:
            if f >= s:
                continue
            for r in rsi_periods:
                for sl in atr_sls:
                    for tp in atr_tps:
                        p = {
                            "fast_ma": f, "slow_ma": s, "rsi_period": r,
                            "atr_sl_mult": sl, "atr_tp_mult": tp,
                            "trailing_atr_mult": 1.5
                        }
                        res = simulate_strategy_slice(df_train, p)
                        if res["trades"] < 5:
                            continue
                        score = res["sharpe_ratio"] * 0.6 + min(res["profit_factor"], 4.0) * 0.3 + (res["max_drawdown"] * 0.01)
                        if score > best_score:
                            best_score = score
                            best_params = p

    train_metrics = simulate_strategy_slice(df_train, best_params) if best_params else {}
    test_metrics = simulate_strategy_slice(df_test, best_params) if df_test is not None and best_params else None

    return {
        "best_params": best_params,
        "train_metrics": train_metrics,
        "test_metrics": test_metrics,
        "best_score": best_score
    }


def run_optimization(
    symbol: str = "BTCUSDT",
    days: int = 90,
    trials: int = 30,
    wfa: bool = True,
    split_ratio: float = 0.7
):
    print("=" * 66)
    print(f"  ОПТИМИЗАЦИЯ СТРАТЕГИИ v2.0 — {symbol} ({days} дней)")
    print(f"  Движок: {'Optuna' if OPTUNA_AVAILABLE else 'Grid Search'} | WFA: {wfa} (Split: {split_ratio:.0%})")
    print("=" * 66)

    df = load_or_download(symbol, INTERVAL, days)
    if df.empty or len(df) < 100:
        print(f"Недостаточно данных для оптимизации {symbol}")
        return

    # Загрузка и привязка старшего таймфрейма (MTF)
    if USE_MTF_FILTER:
        df_htf = load_or_download(symbol, HIGHER_INTERVAL, max(days, 60))
        if not df_htf.empty and len(df_htf) >= 50:
            span = min(HTF_EMA_PERIOD, len(df_htf))
            df_htf["HTF_EMA"] = df_htf["close"].ewm(span=span, adjust=False).mean().shift(1)
            df["timestamp"] = pd.to_datetime(df["timestamp"]).astype("datetime64[ms]")
            df_htf["timestamp"] = pd.to_datetime(df_htf["timestamp"]).astype("datetime64[ms]")
            df = pd.merge_asof(
                df.sort_values("timestamp"),
                df_htf[["timestamp", "HTF_EMA"]].sort_values("timestamp"),
                on="timestamp",
                direction="backward"
            )

    if wfa:
        split_idx = int(len(df) * split_ratio)
        df_train = df.iloc[:split_idx].reset_index(drop=True)
        df_test = df.iloc[split_idx:].reset_index(drop=True)
        print(f"Разбиение данных WFA:")
        print(f"  In-Sample (Обучение):  {df_train['timestamp'].iloc[0].date()} -> {df_train['timestamp'].iloc[-1].date()} ({len(df_train)} свечей)")
        print(f"  Out-of-Sample (Тест):  {df_test['timestamp'].iloc[0].date()} -> {df_test['timestamp'].iloc[-1].date()} ({len(df_test)} свечей)")
    else:
        df_train = df
        df_test = None

    print(f"\nПоиск оптимальных параметров...")
    if OPTUNA_AVAILABLE:
        result = optimize_optuna(df_train, df_test, n_trials=trials)
    else:
        result = optimize_grid(df_train, df_test)

    best = result["best_params"]
    tr = result["train_metrics"]
    te = result.get("test_metrics")

    print("\n" + "=" * 66)
    print("  РЕЗУЛЬТАТЫ ОПТИМИЗАЦИИ")
    print("=" * 66)
    print(f"Лучшие параметры:")
    for k, v in best.items():
        print(f"  {k:<20}: {v}")

    print("\nМетрики на обучающей выборке (In-Sample):")
    print(f"  Сделок:        {tr.get('trades')} (Winrate: {tr.get('win_rate', 0):.1f}%)")
    print(f"  PnL:           {tr.get('pnl_pct', 0):+.2f}%")
    print(f"  Profit Factor: {tr.get('profit_factor', 0):.2f}")
    print(f"  Max Drawdown:  {tr.get('max_drawdown', 0):.2f}%")
    print(f"  Sharpe Ratio:  {tr.get('sharpe_ratio', 0):.2f}")

    if te:
        print("\nМетрики на валидационной выборке (Out-of-Sample / WFA):")
        print(f"  Сделок:        {te.get('trades')} (Winrate: {te.get('win_rate', 0):.1f}%)")
        print(f"  PnL:           {te.get('pnl_pct', 0):+.2f}%")
        print(f"  Profit Factor: {te.get('profit_factor', 0):.2f}")
        print(f"  Max Drawdown:  {te.get('max_drawdown', 0):.2f}%")
        print(f"  Sharpe Ratio:  {te.get('sharpe_ratio', 0):.2f}")

        # WFA Efficiency
        wfa_eff = (te.get("sharpe_ratio", 0) / tr.get("sharpe_ratio", 1)) if tr.get("sharpe_ratio", 0) > 0 else 0.0
        print(f"\n  WFA Efficiency (OOS Sharpe / IS Sharpe): {wfa_eff:.2f}")
        if wfa_eff >= 0.5:
            print("  [OK] Стратегия устойчива к переобучению (WFA > 50%).")
        else:
            print("  [ВНИМАНИЕ] Возможен оверфиттинг параметров.")

    print("=" * 66 + "\n")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Параметрическая оптимизация торговой стратегии Bybit")
    parser.add_argument("symbol", nargs="?", default="BTCUSDT", help="Торговый символ (по умолчанию BTCUSDT)")
    parser.add_argument("--days", type=int, default=90, help="Количество дней истории (по умолчанию 90)")
    parser.add_argument("--trials", type=int, default=30, help="Количество итераций Optuna (по умолчанию 30)")
    parser.add_argument("--wfa", dest="wfa", action="store_true", default=True, help="Включить Walk-Forward Analysis (по умолчанию включено)")
    parser.add_argument("--no-wfa", dest="wfa", action="store_false", help="Отключить Walk-Forward Analysis")
    parser.add_argument("--split", type=float, default=0.7, help="Доля выборки In-Sample для WFA (по умолчанию 0.7)")

    args = parser.parse_args()
    run_optimization(
        symbol=args.symbol,
        days=args.days,
        trials=args.trials,
        wfa=args.wfa,
        split_ratio=args.split
    )
