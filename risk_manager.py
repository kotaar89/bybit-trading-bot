"""
Модуль расширенного риск-менеджмента (v0.3.0):
- Состояние позиции PositionRiskState: Partial TP, Breakeven, Trailing Stop, Time-Stop
- Деривативный MarketRegimeFilter: Funding Rate Z-score, Open Interest (OI) + Price Divergence,
  Realized Volatility и гибридный трендовый фильтр без 33-дневного лага EMA(200)
- Портфельный риск-менеджер PortfolioRiskManager: Aggregate Portfolio Heat Cap,
  корреляционный скейлинг (14d), многоуровневый MaxDD-троттлинг (HWM) и Cooldown после серии стопов
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd

log = logging.getLogger("bot")


@dataclass
class PositionRiskState:
    """Хранит состояние и параметры управления рисками для открытой позиции."""
    symbol: str
    side: str                            # "Buy" / "Sell" или "long" / "short"
    entry_price: float
    size: float                          # текущий размер
    initial_size: float                  # исходный размер
    stop_loss: float
    take_profit: float
    atr_value: float = 0.0
    risk_r: float = 0.0                  # |entry_price - initial_sl|
    r_mult: float = 1.5                  # Множитель R для TP1
    tp1_price: float = 0.0               # Цель TP1 (r_mult * R)
    tp1_executed: bool = False           # Выполнена ли частичная фиксация
    tp1_order_id: str = ""               # ID биржевого лимитного ордера TP1
    tp1_fill_price: float = 0.0          # Фактическая цена исполнения TP1
    tp1_closed_qty: float = 0.0          # Фактически закрытый объем по TP1
    breakeven_applied: bool = False      # Перенесен ли стоп в безубыток
    trailing_active: bool = False        # Активирован ли трейлинг-стоп
    native_trailing_active: bool = False # Активирован ли нативный трейлинг на бирже
    highest_price: float = 0.0           # Максимум цены с момента входа (для лонга)
    lowest_price: float = float("inf")   # Минимум цены с момента входа (для шорта)
    position_idx: int = 0                # 0: One-Way, 1: Hedge Buy, 2: Hedge Sell
    opened_at: float = 0.0               # Unix timestamp открытия позиции
    strategy_tag: str = ""               # Имя стратегии, открывшей позицию
    entry_regime: str = ""               # "TREND" / "RANGE" / ""
    time_stop_tightened: bool = False    # Сработал ли Time-Stop перенос в БУ
    bars_held: int = 0                   # Счетчик баров (для бэктестера)
    entry_order_id: str = ""             # ID ордера входа
    margin_used: float = 0.0             # Задействованная маржа (USDT)
    pos_ref: str = ""                    # Уникальный ключ позиции для группировки в SQLite

    def __post_init__(self):
        if self.highest_price <= 0:
            self.highest_price = self.entry_price
        if self.lowest_price == float("inf"):
            self.lowest_price = self.entry_price
        if self.risk_r <= 0 and self.stop_loss > 0:
            self.risk_r = abs(self.entry_price - self.stop_loss)

        # Расчет TP1
        is_long = self.side.lower() in ("buy", "long")
        if self.tp1_price <= 0 and self.risk_r > 0:
            if is_long:
                self.tp1_price = self.entry_price + (self.r_mult * self.risk_r)
            else:
                self.tp1_price = self.entry_price - (self.r_mult * self.risk_r)

        if not self.pos_ref and self.opened_at > 0:
            self.pos_ref = f"{self.symbol}_{int(self.opened_at)}"

    @property
    def is_long(self) -> bool:
        return self.side.lower() in ("buy", "long")

    def is_risk_free(self) -> bool:
        """True, если позиция уже переведена в безубыток или стоп защищает прибыль (риск <= 0)."""
        if self.breakeven_applied:
            return True
        if self.entry_price <= 0 or self.stop_loss <= 0:
            return False
        if self.is_long:
            return self.stop_loss >= self.entry_price
        return self.stop_loss <= self.entry_price

    def open_risk_usdt(self) -> float:
        """Текущий открытый риск до Stop-Loss в USDT (0.0 если уже в безубытке)."""
        if self.is_risk_free() or self.size <= 0 or self.entry_price <= 0:
            return 0.0
        if self.stop_loss > 0:
            dist = (self.entry_price - self.stop_loss) if self.is_long else (self.stop_loss - self.entry_price)
            if dist <= 0:
                return 0.0
            return float(self.size * dist)
        if self.risk_r > 0:
            return float(self.size * self.risk_r)
        return 0.0

    def update_price_extremes(self, high: float, low: float):
        """Обновляет экстремумы цены с момента открытия позиции."""
        if high > self.highest_price:
            self.highest_price = high
        if low < self.lowest_price:
            self.lowest_price = low

    def check_breakeven(
        self,
        current_price: float,
        trigger_pct: float = 0.015,
        offset_pct: float = 0.002,
        r_mult: float = 0.6
    ) -> tuple[bool, float]:
        """
        Проверяет условие переноса Stop-Loss в безубыток.
        Возвращает (активировать_ли, новый_уровень_SL).
        """
        if self.breakeven_applied:
            return False, self.stop_loss

        if self.entry_price <= 0:
            return False, self.stop_loss

        effective_trigger = trigger_pct
        if self.risk_r > 0 and r_mult > 0:
            r_pct = (self.risk_r * r_mult) / self.entry_price
            effective_trigger = max(trigger_pct, r_pct)

        if self.is_long:
            profit_pct = (current_price - self.entry_price) / self.entry_price
            if profit_pct >= effective_trigger:
                new_sl = self.entry_price * (1 + offset_pct)
                if new_sl > self.stop_loss:
                    return True, new_sl
        else:
            profit_pct = (self.entry_price - current_price) / self.entry_price
            if profit_pct >= effective_trigger:
                new_sl = self.entry_price * (1 - offset_pct)
                if self.stop_loss == 0.0 or new_sl < self.stop_loss:
                    return True, new_sl

        return False, self.stop_loss

    def check_time_stop(
        self,
        current_price: float,
        now_ts: Optional[float] = None,
        bars_held: Optional[int] = None,
        max_hours: float = 30.0,
        hard_hours: float = 36.0,
        min_r_progress: float = 0.3,
        offset_pct: float = 0.002,
    ) -> tuple[str, float, str]:
        """
        Умный Time-Stop: если позиция не достигла TP1 и не ушла в +min_r_progress R
        за max_hours / hard_hours, подтягивает стоп к БУ или закрывает застойную позицию.
        Возвращает (action, new_sl, reason), где action in ('none', 'tighten_be', 'close').
        """
        if self.tp1_executed or self.entry_price <= 0 or current_price <= 0:
            return "none", self.stop_loss, ""

        if bars_held is not None:
            elapsed_h = float(bars_held)
        elif now_ts is not None and self.opened_at > 0:
            elapsed_h = max(0.0, (float(now_ts) - float(self.opened_at)) / 3600.0)
        else:
            return "none", self.stop_loss, ""

        if elapsed_h < max_hours:
            return "none", self.stop_loss, ""

        r_denom = self.risk_r if self.risk_r > 0 else (self.entry_price * 0.02)
        if r_denom <= 0:
            return "none", self.stop_loss, ""

        move = (current_price - self.entry_price) if self.is_long else (self.entry_price - current_price)
        r_prog = move / r_denom

        if r_prog >= min_r_progress:
            return "none", self.stop_loss, ""

        if elapsed_h >= hard_hours or r_prog <= 0.05:
            return (
                "close",
                self.stop_loss,
                f"Time-Stop застой ({elapsed_h:.1f}ч, прогресс {r_prog:+.2f}R < {min_r_progress:.2f}R)",
            )

        if not self.breakeven_applied and not self.time_stop_tightened:
            be_sl = self.entry_price * (1.0 + offset_pct if self.is_long else 1.0 - offset_pct)
            improved = (be_sl > self.stop_loss) if self.is_long else (self.stop_loss == 0.0 or be_sl < self.stop_loss)
            if improved:
                return (
                    "tighten_be",
                    be_sl,
                    f"Time-Stop подтяжка в БУ ({elapsed_h:.1f}ч, прогресс {r_prog:+.2f}R)",
                )

        return "none", self.stop_loss, ""

    def to_dict(self) -> dict:
        """Сериализует состояние позиции в словарь для сохранения на диск."""
        return {
            "symbol": self.symbol,
            "side": self.side,
            "entry_price": self.entry_price,
            "size": self.size,
            "initial_size": self.initial_size,
            "stop_loss": self.stop_loss,
            "take_profit": self.take_profit,
            "atr_value": self.atr_value,
            "risk_r": self.risk_r,
            "r_mult": self.r_mult,
            "tp1_price": self.tp1_price,
            "tp1_executed": self.tp1_executed,
            "tp1_order_id": self.tp1_order_id,
            "tp1_fill_price": self.tp1_fill_price,
            "tp1_closed_qty": self.tp1_closed_qty,
            "breakeven_applied": self.breakeven_applied,
            "trailing_active": self.trailing_active,
            "native_trailing_active": self.native_trailing_active,
            "highest_price": self.highest_price,
            "position_idx": self.position_idx,
            "opened_at": self.opened_at,
            "lowest_price": self.lowest_price if self.lowest_price != float("inf") else self.entry_price,
            "strategy_tag": self.strategy_tag,
            "entry_regime": self.entry_regime,
            "time_stop_tightened": self.time_stop_tightened,
            "bars_held": self.bars_held,
            "entry_order_id": self.entry_order_id,
            "margin_used": self.margin_used,
            "pos_ref": self.pos_ref,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "PositionRiskState":
        """Восстанавливает объект PositionRiskState из словаря."""
        return cls(
            symbol=data["symbol"],
            side=data["side"],
            entry_price=float(data["entry_price"]),
            size=float(data["size"]),
            initial_size=float(data.get("initial_size", data["size"])),
            stop_loss=float(data.get("stop_loss", 0.0)),
            take_profit=float(data.get("take_profit", 0.0)),
            atr_value=float(data.get("atr_value", 0.0)),
            risk_r=float(data.get("risk_r", 0.0)),
            r_mult=float(data.get("r_mult", 1.5)),
            tp1_price=float(data.get("tp1_price", 0.0)),
            tp1_executed=bool(data.get("tp1_executed", False)),
            tp1_order_id=str(data.get("tp1_order_id", "")),
            tp1_fill_price=float(data.get("tp1_fill_price", 0.0)),
            tp1_closed_qty=float(data.get("tp1_closed_qty", 0.0)),
            breakeven_applied=bool(data.get("breakeven_applied", False)),
            trailing_active=bool(data.get("trailing_active", False)),
            native_trailing_active=bool(data.get("native_trailing_active", False)),
            highest_price=float(data.get("highest_price", 0.0)),
            position_idx=int(data.get("position_idx", 0)),
            opened_at=float(data.get("opened_at", 0.0)),
            lowest_price=float(data.get("lowest_price", float("inf"))),
            strategy_tag=str(data.get("strategy_tag", "")),
            entry_regime=str(data.get("entry_regime", "")),
            time_stop_tightened=bool(data.get("time_stop_tightened", False)),
            bars_held=int(data.get("bars_held", 0)),
            entry_order_id=str(data.get("entry_order_id", "")),
            margin_used=float(data.get("margin_used", 0.0)),
            pos_ref=str(data.get("pos_ref", "")),
        )

    def check_partial_tp(self, current_price: float, ratio: float = 0.5, offset_pct: float = 0.002) -> tuple[bool, float, float]:
        """
        Проверяет достижение TP1 = r_mult * R.
        Возвращает (сработал_ли, объем_к_закрытию, новый_уровень_SL_б/у).
        """
        if self.tp1_executed or self.tp1_price <= 0:
            return False, 0.0, self.stop_loss

        qty_to_close = self.size * ratio

        if self.is_long:
            if current_price >= self.tp1_price:
                be_sl = self.entry_price * (1.0 + offset_pct)
                return True, qty_to_close, be_sl
        else:
            if current_price <= self.tp1_price:
                be_sl = self.entry_price * (1.0 - offset_pct)
                return True, qty_to_close, be_sl

        return False, 0.0, self.stop_loss

    def check_trailing_stop(
        self,
        current_price: float,
        high_price: float,
        low_price: float,
        trigger_pct: float = 0.015,
        atr_mult: float = 1.5,
        fallback_pct: float = 0.01,
        r_mult: float = 0.7
    ) -> tuple[bool, float]:
        """
        Проверяет и подтягивает динамический трейлинг-стоп вслед за экстремумом.
        Стоп может двигаться ТОЛЬКО в сторону прибыли позиции.
        """
        self.update_price_extremes(high_price, low_price)

        if self.atr_value > 0:
            dist = self.atr_value * atr_mult
        else:
            dist = current_price * fallback_pct

        effective_trigger = trigger_pct
        if self.risk_r > 0 and r_mult > 0 and self.entry_price > 0:
            r_pct = (self.risk_r * r_mult) / self.entry_price
            effective_trigger = max(trigger_pct, r_pct)

        if self.is_long:
            profit_pct = (current_price - self.entry_price) / self.entry_price
            if profit_pct >= effective_trigger or self.trailing_active:
                self.trailing_active = True
                candidate_sl = self.highest_price - dist
                if candidate_sl > self.stop_loss:
                    return True, candidate_sl
        else:
            profit_pct = (self.entry_price - current_price) / self.entry_price
            if profit_pct >= effective_trigger or self.trailing_active:
                self.trailing_active = True
                candidate_sl = self.lowest_price + dist
                if self.stop_loss == 0.0 or candidate_sl < self.stop_loss:
                    return True, candidate_sl

        return False, self.stop_loss


class MarketRegimeFilter:
    """
    Усиленный фильтр рыночного режима (v0.3.0):
    1. Защита от рыночного дампа/пампа бенчмарка (BTCUSDT)
    2. Funding Rate Z-score: отсев перегретых лонгов и шорт-сквизов
    3. Open Interest (OI) + Price Divergence: детекция выносов без притока нового риска
    4. BTC Realized Volatility + гибридная быстрая/медленная EMA (снижает лаг 4H EMA200 на V-отскоках)
    """

    def __init__(
        self,
        benchmark_symbol: str = "BTCUSDT",
        ema_period: int = 200,
        rsi_threshold: float = 45.0,
        bull_rsi_threshold: float = 55.0,
        dump_pct_threshold: float = 0.03,
        pump_pct_threshold: float = 0.04,
        dump_lookback_candles: int = 4,
        fast_ema_period: int = 50,
        use_hybrid_trend: bool = True,
        funding_zscore_window: int = 30,
        funding_z_long_max: float = 2.0,
        funding_z_short_min: float = -2.0,
        oi_lookback: int = 4,
        oi_div_pct: float = 0.03,
        oi_price_move_pct: float = 0.015,
        vol_window: int = 20,
        extreme_vol_pct: float = 2.50,
    ):
        self.benchmark_symbol = benchmark_symbol
        self.ema_period = ema_period
        self.rsi_threshold = rsi_threshold
        self.bull_rsi_threshold = bull_rsi_threshold
        self.dump_pct_threshold = dump_pct_threshold
        self.pump_pct_threshold = pump_pct_threshold
        self.dump_lookback_candles = dump_lookback_candles
        self.fast_ema_period = fast_ema_period
        self.use_hybrid_trend = bool(use_hybrid_trend)
        self.funding_zscore_window = funding_zscore_window
        self.funding_z_long_max = funding_z_long_max
        self.funding_z_short_min = funding_z_short_min
        self.oi_lookback = oi_lookback
        self.oi_div_pct = oi_div_pct
        self.oi_price_move_pct = oi_price_move_pct
        self.vol_window = vol_window
        self.extreme_vol_pct = extreme_vol_pct

    @staticmethod
    def compute_funding_zscore_series(funding_df: pd.DataFrame, window: int = 30) -> pd.Series:
        """Векторный расчет скользящего Z-score ставки фандинга."""
        if funding_df is None or funding_df.empty or "fundingRate" not in funding_df.columns:
            return pd.Series(dtype=float)
        fr = funding_df["fundingRate"].astype(float)
        min_p = max(5, min(window, window // 3))
        mean = fr.rolling(window=window, min_periods=min_p).mean()
        std = fr.rolling(window=window, min_periods=min_p).std(ddof=1).replace(0.0, np.nan)
        z = (fr - mean) / std
        return z.fillna(0.0)

    @staticmethod
    def compute_latest_funding_zscore(funding_df: pd.DataFrame, window: int = 30) -> Optional[float]:
        """Возвращает последнее значение Funding Rate Z-score или None при нехватке данных."""
        if funding_df is None or funding_df.empty or len(funding_df) < 5 or "fundingRate" not in funding_df.columns:
            return None
        z_ser = MarketRegimeFilter.compute_funding_zscore_series(funding_df, window=window)
        if z_ser.empty or pd.isna(z_ser.iloc[-1]):
            return None
        return float(z_ser.iloc[-1])

    @staticmethod
    def compute_oi_metrics(
        oi_df: pd.DataFrame,
        price_df: Optional[pd.DataFrame] = None,
        lookback: int = 4,
    ) -> Optional[dict]:
        """
        Рассчитывает изменение Open Interest (%) и цены (%) за последние lookback баров.
        Выявляет дивергенции: шорт-сквиз (рост цены при падении OI) и вынос лонгов (падение цены при падении OI).
        """
        if oi_df is None or oi_df.empty or "openInterest" not in oi_df.columns or len(oi_df) <= lookback:
            return None
        oi_vals = oi_df["openInterest"].astype(float)
        oi_now = float(oi_vals.iloc[-1])
        oi_prev = float(oi_vals.iloc[-1 - lookback])
        if oi_prev <= 0 or np.isnan(oi_now) or np.isnan(oi_prev):
            return None
        oi_chg = (oi_now - oi_prev) / oi_prev

        price_chg = 0.0
        if price_df is not None and not price_df.empty and "close" in price_df.columns and len(price_df) > lookback:
            p_now = float(price_df["close"].iloc[-1])
            p_prev = float(price_df["close"].iloc[-1 - lookback])
            if p_prev > 0 and not np.isnan(p_now) and not np.isnan(p_prev):
                price_chg = (p_now - p_prev) / p_prev

        return {
            "oi_now": oi_now,
            "oi_change_pct": float(oi_chg),
            "price_change_pct": float(price_chg),
        }

    @staticmethod
    def compute_realized_volatility(close_series: pd.Series, window: int = 20, bars_per_year: float = 2190.0) -> pd.Series:
        """Годовая реализованная волатильность по лог-доходностям."""
        if close_series is None or len(close_series) < 3:
            return pd.Series(0.0, index=close_series.index if close_series is not None else None)
        log_ret = np.log(close_series.astype(float).replace(0.0, np.nan)).diff()
        vol = log_ret.rolling(window=window, min_periods=max(5, window // 2)).std(ddof=1) * np.sqrt(bars_per_year)
        return vol.fillna(0.0)

    @staticmethod
    def add_indicators(
        df: pd.DataFrame,
        ema_period: int = 200,
        rsi_period: int = 14,
        dump_lookback_candles: int = 4,
        dump_pct_threshold: float = 0.03,
        pump_pct_threshold: float = 0.04,
        rsi_threshold: float = 45.0,
        bull_rsi_threshold: float = 55.0,
        fast_ema_period: int = 50,
        use_hybrid_trend: bool = True,
        funding_df: Optional[pd.DataFrame] = None,
        oi_df: Optional[pd.DataFrame] = None,
        funding_zscore_window: int = 30,
        funding_z_long_max: float = 2.0,
        funding_z_short_min: float = -2.0,
        oi_lookback: int = 4,
        oi_div_pct: float = 0.03,
        oi_price_move_pct: float = 0.015,
        vol_window: int = 20,
        extreme_vol_pct: float = 2.50,
    ) -> pd.DataFrame:
        """
        Векторный предварительный расчет индикаторов бенчмарка (и деривативов) для O(1) проверки в бэктестере.
        """
        if df.empty or len(df) < 20:
            return df
        res = df.copy()
        res["BENCH_EMA"] = res["close"].ewm(span=ema_period, adjust=False).mean()
        eff_fast = min(fast_ema_period, ema_period) if (fast_ema_period and fast_ema_period > 0) else ema_period
        res["BENCH_FAST_EMA"] = res["close"].ewm(span=eff_fast, adjust=False).mean()
        res["BENCH_FAST_EMA_SLOPE"] = res["BENCH_FAST_EMA"].diff(3).fillna(0.0)

        delta = res["close"].diff()
        gain = delta.clip(lower=0).ewm(alpha=1 / rsi_period, adjust=False).mean()
        loss = -delta.clip(upper=0).ewm(alpha=1 / rsi_period, adjust=False).mean()
        rs = gain / loss.replace(0, np.nan)
        res["BENCH_RSI"] = 100 - (100 / (1 + rs))
        res["BENCH_RSI"] = res["BENCH_RSI"].where(~((loss == 0) & (gain > 0)), 100.0)
        res["BENCH_RSI"] = res["BENCH_RSI"].fillna(50.0)

        res["BENCH_REALIZED_VOL"] = MarketRegimeFilter.compute_realized_volatility(res["close"], window=vol_window)

        # Скользящие экстремумы для детекции Dump / Pump
        res["BENCH_HIGH_MAX"] = res["high"].rolling(window=dump_lookback_candles, min_periods=1).max()
        res["BENCH_LOW_MIN"] = res["low"].rolling(window=dump_lookback_candles, min_periods=1).min()
        res["BENCH_CLOSE"] = res["close"]

        bench_high = res["BENCH_HIGH_MAX"]
        dump_drop = (bench_high - res["close"]) / bench_high.replace(0, np.nan)
        res["BENCH_DUMP"] = (dump_drop >= dump_pct_threshold).fillna(False)

        bench_low = res["BENCH_LOW_MIN"]
        pump_gain = (res["close"] - bench_low) / bench_low.replace(0, np.nan)
        res["BENCH_PUMP"] = (pump_gain >= pump_pct_threshold).fillna(False)

        # Опциональная привязка Funding Z-score без lookahead
        res["BENCH_FUNDING_Z"] = 0.0
        if funding_df is not None and not funding_df.empty and "timestamp" in funding_df.columns and "timestamp" in res.columns:
            fdf = funding_df.copy().sort_values("timestamp").reset_index(drop=True)
            fdf["timestamp"] = pd.to_datetime(fdf["timestamp"]).astype("datetime64[ms]")
            fdf["BENCH_FUNDING_Z"] = MarketRegimeFilter.compute_funding_zscore_series(fdf, window=funding_zscore_window)
            res["_ts_tmp"] = pd.to_datetime(res["timestamp"]).astype("datetime64[ms]")
            merged_f = pd.merge_asof(
                res.sort_values("_ts_tmp"),
                fdf[["timestamp", "BENCH_FUNDING_Z"]].rename(columns={"timestamp": "_ts_tmp"}),
                on="_ts_tmp",
                direction="backward",
                suffixes=("_drop", ""),
            )
            if "BENCH_FUNDING_Z_drop" in merged_f.columns:
                merged_f = merged_f.drop(columns=["BENCH_FUNDING_Z_drop"])
            res = merged_f.drop(columns=["_ts_tmp"]).sort_values("timestamp").reset_index(drop=True)
            res["BENCH_FUNDING_Z"] = res["BENCH_FUNDING_Z"].fillna(0.0)

        # Опциональная привязка Open Interest без lookahead
        res["BENCH_OI_CHG"] = 0.0
        if oi_df is not None and not oi_df.empty and "timestamp" in oi_df.columns and "timestamp" in res.columns:
            odf = oi_df.copy().sort_values("timestamp").reset_index(drop=True)
            odf["timestamp"] = pd.to_datetime(odf["timestamp"]).astype("datetime64[ms]")
            odf["BENCH_OI_CHG"] = odf["openInterest"].astype(float).pct_change(oi_lookback).fillna(0.0)
            res["_ts_tmp"] = pd.to_datetime(res["timestamp"]).astype("datetime64[ms]")
            merged_o = pd.merge_asof(
                res.sort_values("_ts_tmp"),
                odf[["timestamp", "BENCH_OI_CHG"]].rename(columns={"timestamp": "_ts_tmp"}),
                on="_ts_tmp",
                direction="backward",
                suffixes=("_drop", ""),
            )
            if "BENCH_OI_CHG_drop" in merged_o.columns:
                merged_o = merged_o.drop(columns=["BENCH_OI_CHG_drop"])
            res = merged_o.drop(columns=["_ts_tmp"]).sort_values("timestamp").reset_index(drop=True)
            res["BENCH_OI_CHG"] = res["BENCH_OI_CHG"].fillna(0.0)

        price_chg = res["close"].pct_change(oi_lookback).fillna(0.0)
        squeeze_long_trap = (price_chg >= oi_price_move_pct) & (res["BENCH_OI_CHG"] <= -oi_div_pct)
        liq_short_trap = (price_chg <= -oi_price_move_pct) & (res["BENCH_OI_CHG"] <= -oi_div_pct)
        vol_ok = res["BENCH_REALIZED_VOL"] < extreme_vol_pct

        if use_hybrid_trend and eff_fast < ema_period:
            trend_long_ok = (res["close"] >= res["BENCH_EMA"]) | (
                (res["close"] >= res["BENCH_FAST_EMA"]) & (res["BENCH_FAST_EMA_SLOPE"] > 0)
            )
        else:
            trend_long_ok = res["close"] >= res["BENCH_EMA"]

        res["BENCH_LONG_ALLOWED"] = (
            (~res["BENCH_DUMP"])
            & trend_long_ok
            & (res["BENCH_RSI"] >= rsi_threshold)
            & (res["BENCH_FUNDING_Z"] < funding_z_long_max)
            & (~squeeze_long_trap)
            & vol_ok
        )
        res["BENCH_SHORT_ALLOWED"] = (
            (~res["BENCH_PUMP"])
            & ~((res["close"] > res["BENCH_EMA"]) & (res["BENCH_RSI"] > bull_rsi_threshold))
            & (res["BENCH_FUNDING_Z"] > funding_z_short_min)
            & (~liq_short_trap)
            & vol_ok
        )
        res["BENCH_SELF_LONG_ALLOWED"] = (~res["BENCH_DUMP"]) & (res["BENCH_FUNDING_Z"] < funding_z_long_max)
        res["BENCH_SELF_SHORT_ALLOWED"] = (~res["BENCH_PUMP"]) & (res["BENCH_FUNDING_Z"] > funding_z_short_min)
        return res

    def get_regime_snapshot(
        self,
        btc_df: Optional[pd.DataFrame],
        funding_zscore: Optional[float] = None,
        oi_metrics: Optional[dict] = None,
        closed_idx: int = -2,
    ) -> dict:
        """Возвращает компактную сводку текущего рыночного режима для дашборда и логов."""
        snap = {
            "trend": "unknown",
            "btc_close": None,
            "rsi": None,
            "realized_vol": None,
            "funding_zscore": round(funding_zscore, 2) if funding_zscore is not None else None,
            "oi_change_pct": round(oi_metrics["oi_change_pct"] * 100.0, 2) if oi_metrics else None,
            "long_allowed": None,
            "short_allowed": None,
        }
        if btc_df is None or btc_df.empty or len(btc_df) < 10:
            return snap
        close = btc_df["close"]
        idx = closed_idx if abs(closed_idx) <= len(btc_df) else (-2 if len(btc_df) >= 2 else -1)
        c_val = float(close.iloc[idx])
        slow_ema = float(close.ewm(span=self.ema_period, adjust=False).mean().iloc[idx])
        fast_span = min(self.fast_ema_period, self.ema_period) if self.fast_ema_period > 0 else self.ema_period
        fast_ema = float(close.ewm(span=fast_span, adjust=False).mean().iloc[idx])
        vol_ser = self.compute_realized_volatility(close, window=self.vol_window)
        snap["btc_close"] = round(c_val, 2) if not np.isnan(c_val) else None
        snap["realized_vol"] = round(float(vol_ser.iloc[idx]) * 100.0, 1) if not vol_ser.empty else None

        delta = close.diff()
        gain = delta.clip(lower=0).ewm(alpha=1 / 14, adjust=False).mean()
        loss = -delta.clip(upper=0).ewm(alpha=1 / 14, adjust=False).mean()
        last_gain = float(gain.iloc[idx])
        last_loss = float(loss.iloc[idx])
        if not np.isnan(last_gain) and not np.isnan(last_loss):
            if last_loss == 0.0:
                last_rsi = 100.0 if last_gain > 0 else 50.0
            else:
                last_rsi = float(100 - (100 / (1 + (last_gain / last_loss))))
            snap["rsi"] = round(last_rsi, 1)

        if c_val >= slow_ema:
            snap["trend"] = "bullish"
        elif c_val >= fast_ema:
            snap["trend"] = "recovery"
        else:
            snap["trend"] = "bearish"

        if len(btc_df) >= max(self.ema_period, 30):
            l_ok, _ = self.is_long_allowed(
                "__ALT__", btc_df, funding_zscore=funding_zscore, oi_metrics=oi_metrics, closed_idx=idx
            )
            s_ok, _ = self.is_short_allowed(
                "__ALT__", btc_df, funding_zscore=funding_zscore, oi_metrics=oi_metrics, closed_idx=idx
            )
            snap["long_allowed"] = bool(l_ok)
            snap["short_allowed"] = bool(s_ok)
        return snap

    def is_long_allowed(
        self,
        symbol: str,
        btc_df: pd.DataFrame,
        funding_zscore: Optional[float] = None,
        oi_metrics: Optional[dict] = None,
        symbol_funding_zscore: Optional[float] = None,
        symbol_oi_metrics: Optional[dict] = None,
        closed_idx: int = -2,
    ) -> tuple[bool, str]:
        """Проверяет, разрешен ли вход в LONG с учетом BTC-сентимента, Funding Z-score и OI."""
        if btc_df is None or btc_df.empty or len(btc_df) < max(self.ema_period, 30):
            return False, f"Нет достаточных данных {self.benchmark_symbol} для фильтра сентимента (fail-safe)"

        df = btc_df
        close = df["close"]

        # 1. Проверка на Market Dump
        if "BENCH_DUMP" in df.columns:
            raw_dump = df["BENCH_DUMP"].iloc[-1]
            is_dump = True if pd.isna(raw_dump) else bool(raw_dump)
        else:
            lookback = max(1, min(self.dump_lookback_candles, len(df)))
            recent_high = df["high"].iloc[-lookback:].max()
            last_close = close.iloc[-1]
            dump_drop = (recent_high - last_close) / recent_high if recent_high > 0 else 0.0
            is_dump = (dump_drop >= self.dump_pct_threshold)

        if is_dump:
            return False, f"Рыночный дамп бенчмарка {self.benchmark_symbol}. Все лонг-входы заблокированы."

        # 2. Funding Rate Z-score (перегрев лонгов)
        for label, fz in ((symbol, symbol_funding_zscore), (self.benchmark_symbol, funding_zscore)):
            if fz is not None and not np.isnan(fz) and fz >= self.funding_z_long_max:
                return (
                    False,
                    f"Деривативный фильтр: перегретый Funding Z-score {label} ({fz:+.2f} >= {self.funding_z_long_max:.1f}). Лонг отклонен.",
                )

        # 3. Open Interest дивергенция (шорт-сквиз без притока нового OI)
        for label, oim in ((symbol, symbol_oi_metrics), (self.benchmark_symbol, oi_metrics)):
            if isinstance(oim, dict):
                oi_chg = float(oim.get("oi_change_pct", 0.0))
                px_chg = float(oim.get("price_change_pct", 0.0))
                if px_chg >= self.oi_price_move_pct and oi_chg <= -self.oi_div_pct:
                    return (
                        False,
                        f"Деривативный фильтр: рост {label} ({px_chg*100:+.1f}%) при падении OI ({oi_chg*100:+.1f}% — шорт-сквиз). Лонг отклонен.",
                    )

        if symbol.upper() == self.benchmark_symbol.upper():
            return True, "OK (Бенчмарк)"

        # 4. Проверка старшей EMA / гибридного тренда (по закрытой свече)
        idx = closed_idx if abs(closed_idx) <= len(df) else (-2 if len(df) >= 2 else -1)
        last_close = float(close.iloc[idx])
        if "BENCH_EMA" in df.columns:
            last_ema = float(df["BENCH_EMA"].iloc[idx])
        else:
            last_ema = float(close.ewm(span=self.ema_period, adjust=False).mean().iloc[idx])

        if np.isnan(last_close) or np.isnan(last_ema):
            return False, f"Фильтр сентимента: NaN в данных {self.benchmark_symbol} (fail-safe)"

        # Экстремальная реализованная волатильность
        if self.extreme_vol_pct > 0:
            vol_ser = self.compute_realized_volatility(close, window=self.vol_window)
            if not vol_ser.empty:
                cur_vol = float(vol_ser.iloc[idx])
                if not np.isnan(cur_vol) and cur_vol >= self.extreme_vol_pct:
                    return False, f"Фильтр волатильности: экстремальный разброс {self.benchmark_symbol} ({cur_vol*100:.0f}% годовых)."

        trend_ok = last_close >= last_ema
        if not trend_ok and self.use_hybrid_trend and self.fast_ema_period > 0 and self.fast_ema_period < self.ema_period:
            fast_ser = close.ewm(span=self.fast_ema_period, adjust=False).mean()
            fast_now = float(fast_ser.iloc[idx])
            prev_idx = max(-len(df), idx - 3)
            fast_prev = float(fast_ser.iloc[prev_idx])
            if not np.isnan(fast_now) and not np.isnan(fast_prev) and last_close >= fast_now and fast_now > fast_prev:
                trend_ok = True

        if not trend_ok:
            return (
                False,
                f"Фильтр сентимента: {self.benchmark_symbol} ({last_close:.1f}) ниже EMA ({last_ema:.1f}). Лонг по {symbol} отклонен.",
            )

        # 5. Проверка RSI бенчмарка
        if "BENCH_RSI" in df.columns:
            last_rsi = float(df["BENCH_RSI"].iloc[idx])
        else:
            delta = close.diff()
            gain = delta.clip(lower=0).ewm(alpha=1 / 14, adjust=False).mean()
            loss = -delta.clip(upper=0).ewm(alpha=1 / 14, adjust=False).mean()
            last_gain = float(gain.iloc[idx])
            last_loss = float(loss.iloc[idx])
            if np.isnan(last_gain) or np.isnan(last_loss):
                return False, f"Фильтр сентимента: NaN RSI {self.benchmark_symbol} (fail-safe)"
            if last_loss == 0.0:
                last_rsi = 100.0 if last_gain > 0 else 50.0
            else:
                rs = last_gain / last_loss
                last_rsi = float(100 - (100 / (1 + rs)))

        if np.isnan(last_rsi) or last_rsi < self.rsi_threshold:
            return (
                False,
                f"Фильтр сентимента: RSI {self.benchmark_symbol} ({last_rsi:.1f}) ниже порога {self.rsi_threshold:.1f}. Лонг по {symbol} отклонен.",
            )

        return True, "OK"

    def is_short_allowed(
        self,
        symbol: str,
        btc_df: pd.DataFrame,
        funding_zscore: Optional[float] = None,
        oi_metrics: Optional[dict] = None,
        symbol_funding_zscore: Optional[float] = None,
        symbol_oi_metrics: Optional[dict] = None,
        closed_idx: int = -2,
    ) -> tuple[bool, str]:
        """Проверяет, разрешен ли вход в SHORT с учетом BTC-сентимента, Funding Z-score и OI."""
        if btc_df is None or btc_df.empty or len(btc_df) < max(self.ema_period, 30):
            return False, f"Нет достаточных данных {self.benchmark_symbol} для фильтра сентимента (fail-safe)"

        df = btc_df
        close = df["close"]

        if "BENCH_PUMP" in df.columns:
            raw_pump = df["BENCH_PUMP"].iloc[-1]
            is_pump = True if pd.isna(raw_pump) else bool(raw_pump)
        else:
            lookback = max(1, min(self.dump_lookback_candles, len(df)))
            recent_low = df["low"].iloc[-lookback:].min()
            last_close = close.iloc[-1]
            pump_gain = (last_close - recent_low) / recent_low if recent_low > 0 else 0.0
            is_pump = (pump_gain >= self.pump_pct_threshold)

        if is_pump:
            return False, f"Рыночный памп бенчмарка {self.benchmark_symbol}. Все шорт-входы заблокированы."

        for label, fz in ((symbol, symbol_funding_zscore), (self.benchmark_symbol, funding_zscore)):
            if fz is not None and not np.isnan(fz) and fz <= self.funding_z_short_min:
                return (
                    False,
                    f"Деривативный фильтр: перепроданный Funding Z-score {label} ({fz:+.2f} <= {self.funding_z_short_min:.1f}). Риск шорт-сквиза.",
                )

        for label, oim in ((symbol, symbol_oi_metrics), (self.benchmark_symbol, oi_metrics)):
            if isinstance(oim, dict):
                oi_chg = float(oim.get("oi_change_pct", 0.0))
                px_chg = float(oim.get("price_change_pct", 0.0))
                if px_chg <= -self.oi_price_move_pct and oi_chg <= -self.oi_div_pct:
                    return (
                        False,
                        f"Деривативный фильтр: падение {label} ({px_chg*100:+.1f}%) при сбросе OI ({oi_chg*100:+.1f}% — ликвидации лонгов). Шорт отклонен.",
                    )

        if symbol.upper() == self.benchmark_symbol.upper():
            return True, "OK (Бенчмарк)"

        idx = closed_idx if abs(closed_idx) <= len(df) else (-2 if len(df) >= 2 else -1)
        last_close = float(close.iloc[idx])
        if "BENCH_EMA" in df.columns:
            last_ema = float(df["BENCH_EMA"].iloc[idx])
        else:
            last_ema = float(close.ewm(span=self.ema_period, adjust=False).mean().iloc[idx])

        if np.isnan(last_close) or np.isnan(last_ema):
            return False, f"Фильтр сентимента: NaN в данных {self.benchmark_symbol} (fail-safe)"

        # Экстремальная реализованная волатильность (симметрично с is_long_allowed и add_indicators)
        if self.extreme_vol_pct > 0:
            vol_ser = self.compute_realized_volatility(close, window=self.vol_window)
            if not vol_ser.empty:
                cur_vol = float(vol_ser.iloc[idx])
                if not np.isnan(cur_vol) and cur_vol >= self.extreme_vol_pct:
                    return False, f"Фильтр волатильности: экстремальный разброс {self.benchmark_symbol} ({cur_vol*100:.0f}% годовых)."

        if "BENCH_RSI" in df.columns:
            last_rsi = float(df["BENCH_RSI"].iloc[idx])
        else:
            delta = close.diff()
            gain = delta.clip(lower=0).ewm(alpha=1 / 14, adjust=False).mean()
            loss = -delta.clip(upper=0).ewm(alpha=1 / 14, adjust=False).mean()
            last_gain = float(gain.iloc[idx])
            last_loss = float(loss.iloc[idx])
            if np.isnan(last_gain) or np.isnan(last_loss):
                return False, f"Фильтр сентимента: NaN RSI {self.benchmark_symbol} (fail-safe)"
            if last_loss == 0.0:
                last_rsi = 100.0 if last_gain > 0 else 50.0
            else:
                rs = last_gain / last_loss
                last_rsi = float(100 - (100 / (1 + rs)))

        if np.isnan(last_rsi) or (last_close > last_ema and last_rsi > self.bull_rsi_threshold):
            return (
                False,
                f"Фильтр сентимента: {self.benchmark_symbol} в сильном бычьем тренде (выше EMA и RSI {last_rsi:.1f} > {self.bull_rsi_threshold:.1f}). Шорт по {symbol} отклонен.",
            )

        return True, "OK"


@dataclass
class PortfolioRiskManager:
    """
    Портфельный риск-менеджер v3:
    1. Aggregate Portfolio Heat Cap: суммарный открытый риск до SL (исключая позиции в БУ) <= max_heat_pct
    2. Correlation Scaling: снижение риска при высокой 14d корреляции с открытой позицией в том же направлении
    3. Trailing Peak Equity Throttle (HWM): -6% -> 0.5x риск, -12% -> пауза новых входов
    4. Loss-Streak Cooldown: таймаут после серии стопов по паре или по портфелю
    """

    max_heat_pct: float = 0.035
    use_correlation_scaling: bool = True
    corr_lookback_bars: int = 336
    corr_threshold: float = 0.75
    corr_risk_scale: float = 0.6
    use_peak_dd_throttle: bool = True
    dd_tier1_pct: float = 0.06
    dd_tier1_mult: float = 0.5
    dd_tier2_pct: float = 0.12
    dd_recover_pct: float = 0.09
    use_loss_cooldown: bool = True
    symbol_loss_limit: int = 2
    portfolio_loss_limit: int = 3
    cooldown_seconds: float = 21600.0  # 6 часов по умолчанию

    peak_equity: float = 0.0
    drawdown_pause_active: bool = False
    portfolio_loss_streak: int = 0
    symbol_loss_streaks: dict[str, int] = field(default_factory=dict)
    symbol_cooldown_until: dict[str, float] = field(default_factory=dict)
    portfolio_cooldown_until: float = 0.0

    def update_equity(self, equity: float, valid: bool = True) -> dict:
        """Обновляет High-Water Mark и рассчитывает текущий уровень MaxDD-троттлинга."""
        eq = float(equity or 0.0)
        if valid and eq > 0:
            if eq > self.peak_equity:
                self.peak_equity = eq

        peak_dd = 0.0
        if self.peak_equity > 0 and eq > 0:
            peak_dd = max(0.0, (self.peak_equity - eq) / self.peak_equity)

        risk_scale = 1.0
        if self.use_peak_dd_throttle and self.peak_equity > 0:
            if valid:
                if peak_dd >= self.dd_tier2_pct:
                    self.drawdown_pause_active = True
                    risk_scale = 0.0
                elif self.drawdown_pause_active:
                    if peak_dd <= self.dd_recover_pct:
                        self.drawdown_pause_active = False
                        risk_scale = self.dd_tier1_mult if peak_dd >= self.dd_tier1_pct else 1.0
                    else:
                        risk_scale = 0.0
                elif peak_dd >= self.dd_tier1_pct:
                    risk_scale = self.dd_tier1_mult
            else:
                if self.drawdown_pause_active:
                    risk_scale = 0.0
                elif peak_dd >= self.dd_tier1_pct:
                    risk_scale = self.dd_tier1_mult

        return {
            "peak_equity": round(self.peak_equity, 2),
            "peak_dd_pct": round(peak_dd * 100.0, 2),
            "risk_scale": risk_scale,
            "drawdown_pause": self.drawdown_pause_active,
        }

    def compute_open_heat_pct(self, risk_states: dict, equity: float) -> float:
        """Суммарный риск до SL (в долях equity) по всем НЕ переведенным в БУ позициям."""
        if equity <= 0 or not risk_states:
            return 0.0
        total_risk_usdt = 0.0
        for st in risk_states.values():
            if st is None:
                continue
            total_risk_usdt += st.open_risk_usdt()
        return max(0.0, total_risk_usdt / float(equity))

    def compute_correlation_scale(
        self,
        symbol: str,
        target_is_long: bool,
        open_positions: dict,
        klines_map: Optional[dict[str, pd.DataFrame]] = None,
    ) -> tuple[float, float]:
        """
        Вычисляет максимальную парную корреляцию доходностей (за corr_lookback_bars)
        с уже открытыми позициями в том же направлении.
        Возвращает (risk_scale, max_corr).
        """
        if not self.use_correlation_scaling or not open_positions or not klines_map:
            return 1.0, 0.0

        df_cand = klines_map.get(symbol)
        if df_cand is None or df_cand.empty or len(df_cand) < 30 or "close" not in df_cand.columns:
            return 1.0, 0.0

        cand_ret = df_cand["close"].astype(float).pct_change().dropna().tail(self.corr_lookback_bars)
        if len(cand_ret) < 20:
            return 1.0, 0.0

        max_corr = 0.0
        for sym_open, pos_obj in open_positions.items():
            if not pos_obj or sym_open == symbol:
                continue
            if isinstance(pos_obj, dict):
                open_is_long = pos_obj.get("position") == "long" or str(pos_obj.get("side", "")).lower() in ("buy", "long")
            else:
                open_is_long = bool(getattr(pos_obj, "is_long", True))

            if open_is_long != target_is_long:
                continue

            df_open = klines_map.get(sym_open)
            if df_open is None or df_open.empty or len(df_open) < 30 or "close" not in df_open.columns:
                continue

            open_ret = df_open["close"].astype(float).pct_change().dropna().tail(self.corr_lookback_bars)
            min_len = min(len(cand_ret), len(open_ret))
            if min_len < 20:
                continue
            a = cand_ret.iloc[-min_len:].to_numpy(dtype=float)
            b = open_ret.iloc[-min_len:].to_numpy(dtype=float)
            if np.std(a) <= 1e-12 or np.std(b) <= 1e-12:
                continue
            corr_val = float(np.corrcoef(a, b)[0, 1])
            if not np.isnan(corr_val) and corr_val > max_corr:
                max_corr = corr_val

        if max_corr >= self.corr_threshold:
            return float(self.corr_risk_scale), float(max_corr)
        return 1.0, float(max_corr)

    def check_entry_budget(
        self,
        symbol: str,
        is_long: bool,
        proposed_risk_usdt: float,
        equity: float,
        risk_states: dict,
        open_positions: dict,
        klines_map: Optional[dict[str, pd.DataFrame]] = None,
        now_ts: Optional[float] = None,
    ) -> tuple[bool, float, str]:
        """
        Комплексная проверка перед открытием позиции:
        Cooldowns -> Peak DD Pause -> Peak DD Throttle -> Correlation Scaling -> Portfolio Heat Cap.
        Возвращает (allowed, adjusted_risk_usdt, reason).
        """
        now = float(now_ts if now_ts is not None else time.time())
        sym = symbol.upper()

        if self.use_loss_cooldown:
            if self.portfolio_cooldown_until > now:
                rem_min = (self.portfolio_cooldown_until - now) / 60.0
                return False, 0.0, f"Портфельный Cooldown после серии убытков ({rem_min:.0f} мин)"
            sym_cd = float(self.symbol_cooldown_until.get(sym, 0.0) or 0.0)
            if sym_cd > now:
                rem_min = (sym_cd - now) / 60.0
                return False, 0.0, f"[{sym}] Cooldown после серии стопов ({rem_min:.0f} мин)"

        dd_info = self.update_equity(equity, valid=equity > 0)
        if dd_info["drawdown_pause"]:
            return (
                False,
                0.0,
                f"Tier-2 MaxDD Pause: просадка от пика {dd_info['peak_dd_pct']:.1f}% >= {self.dd_tier2_pct*100:.0f}%",
            )

        adj_risk = float(proposed_risk_usdt) * float(dd_info["risk_scale"])

        corr_scale, max_corr = self.compute_correlation_scale(symbol, is_long, open_positions, klines_map)
        if corr_scale < 1.0:
            adj_risk *= corr_scale

        note_parts = []
        if dd_info["risk_scale"] < 1.0:
            note_parts.append(f"DDx{dd_info['risk_scale']:.2f}")
        if corr_scale < 1.0:
            note_parts.append(f"Corr({max_corr:.2f})x{corr_scale:.2f}")

        if equity > 0 and self.max_heat_pct > 0:
            open_heat = self.compute_open_heat_pct(risk_states, equity)
            rem_heat = max(0.0, self.max_heat_pct - open_heat)
            rem_risk_usdt = rem_heat * equity
            if rem_risk_usdt < proposed_risk_usdt * 0.35:
                return (
                    False,
                    0.0,
                    f"Portfolio Heat Cap: открытый риск {open_heat*100:.2f}% / {self.max_heat_pct*100:.1f}%",
                )
            if adj_risk > rem_risk_usdt:
                adj_risk = rem_risk_usdt
                note_parts.append(f"Heat Cap({rem_heat*100:.2f}%)")

        return True, max(0.0, adj_risk), ", ".join(note_parts) if note_parts else "OK"

    def record_closed_trade(
        self,
        symbol: str,
        pnl_val: float,
        now_ts: Optional[float] = None,
        cooldown_sec: Optional[float] = None,
    ):
        """Обновляет счетчики серий убытков и при необходимости включает Cooldown."""
        now = float(now_ts if now_ts is not None else time.time())
        cd_sec = float(cooldown_sec if cooldown_sec is not None else self.cooldown_seconds)
        sym = symbol.upper()

        if pnl_val < -1e-9:
            self.portfolio_loss_streak += 1
            self.symbol_loss_streaks[sym] = int(self.symbol_loss_streaks.get(sym, 0)) + 1
            if self.use_loss_cooldown:
                if self.symbol_loss_streaks[sym] >= self.symbol_loss_limit:
                    self.symbol_cooldown_until[sym] = now + cd_sec
                if self.portfolio_loss_streak >= self.portfolio_loss_limit:
                    self.portfolio_cooldown_until = now + cd_sec
        elif pnl_val > 1e-9:
            self.portfolio_loss_streak = 0
            self.symbol_loss_streaks[sym] = 0

    def active_cooldowns(self, now_ts: Optional[float] = None) -> dict:
        now = float(now_ts if now_ts is not None else time.time())
        syms = {
            s: round((ts - now) / 60.0, 1)
            for s, ts in self.symbol_cooldown_until.items()
            if ts > now
        }
        return {
            "portfolio_cooldown_min": round(max(0.0, (self.portfolio_cooldown_until - now) / 60.0), 1),
            "symbol_cooldowns_min": syms,
            "portfolio_loss_streak": self.portfolio_loss_streak,
            "symbol_loss_streaks": dict(self.symbol_loss_streaks),
        }

    def to_dict(self) -> dict:
        return {
            "peak_equity": self.peak_equity,
            "drawdown_pause_active": self.drawdown_pause_active,
            "portfolio_loss_streak": self.portfolio_loss_streak,
            "symbol_loss_streaks": dict(self.symbol_loss_streaks),
            "symbol_cooldown_until": dict(self.symbol_cooldown_until),
            "portfolio_cooldown_until": self.portfolio_cooldown_until,
        }

    @classmethod
    def from_dict(cls, data: dict, **kwargs) -> "PortfolioRiskManager":
        obj = cls(**kwargs)
        if isinstance(data, dict):
            obj.peak_equity = float(data.get("peak_equity") or 0.0)
            obj.drawdown_pause_active = bool(data.get("drawdown_pause_active", False))
            obj.portfolio_loss_streak = int(data.get("portfolio_loss_streak") or 0)
            obj.symbol_loss_streaks = {
                str(k): int(v) for k, v in (data.get("symbol_loss_streaks") or {}).items()
            }
            obj.symbol_cooldown_until = {
                str(k): float(v) for k, v in (data.get("symbol_cooldown_until") or {}).items()
            }
            obj.portfolio_cooldown_until = float(data.get("portfolio_cooldown_until") or 0.0)
        return obj
