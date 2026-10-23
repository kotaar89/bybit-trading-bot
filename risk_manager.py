"""
Модуль расширенного риск-менеджмента (v2.1):
- Трейлинг-стоп (Trailing Stop) по ATR / пикам цены (программный и нативный)
- Частичная фиксация прибыли (Partial TP на 1.5R + перевод в Б/У)
- Фильтр рыночного сентимента по BTC/ETH (Market Regime Filter) с векторной оптимизацией
"""

import logging
from dataclasses import dataclass, field
import pandas as pd
import numpy as np

log = logging.getLogger("bot")


@dataclass
class PositionRiskState:
    """Хранит состояние и параметры управления рисками для открытой позиции."""
    symbol: str
    side: str                          # "Buy" / "Sell" или "long" / "short"
    entry_price: float
    size: float                        # текущий размер
    initial_size: float                # исходный размер
    stop_loss: float
    take_profit: float
    atr_value: float = 0.0
    risk_r: float = 0.0                # |entry_price - initial_sl|
    r_mult: float = 1.5                # Множитель R для TP1
    tp1_price: float = 0.0             # Цель TP1 (1.5R)
    tp1_executed: bool = False         # Выполнена ли частичная фиксация
    tp1_order_id: str = ""             # ID биржевого лимитного ордера TP1
    breakeven_applied: bool = False    # Перенесен ли стоп в безубыток
    trailing_active: bool = False      # Активирован ли трейлинг-стоп
    native_trailing_active: bool = False # Активирован ли нативный трейлинг на бирже
    highest_price: float = 0.0         # Максимум цены с момента входа (для лонга)
    lowest_price: float = float("inf") # Минимум цены с момента входа (для шорта)
    position_idx: int = 0              # 0: One-Way, 1: Hedge Buy, 2: Hedge Sell

    def __post_init__(self):
        if self.highest_price <= 0:
            self.highest_price = self.entry_price
        if self.lowest_price == float("inf"):
            self.lowest_price = self.entry_price
        if self.risk_r <= 0 and self.stop_loss > 0:
            self.risk_r = abs(self.entry_price - self.stop_loss)

        # Расчет TP1 (1.5R)
        is_long = self.side.lower() in ("buy", "long")
        if self.tp1_price <= 0 and self.risk_r > 0:
            if is_long:
                self.tp1_price = self.entry_price + (self.r_mult * self.risk_r)
            else:
                self.tp1_price = self.entry_price - (self.r_mult * self.risk_r)

    @property
    def is_long(self) -> bool:
        return self.side.lower() in ("buy", "long")

    def update_price_extremes(self, high: float, low: float):
        """Обновляет экстремумы цены с момента открытия позиции."""
        if high > self.highest_price:
            self.highest_price = high
        if low < self.lowest_price:
            self.lowest_price = low

    def check_breakeven(self, current_price: float, trigger_pct: float = 0.015, offset_pct: float = 0.002) -> tuple[bool, float]:
        """
        Проверяет условие переноса Stop-Loss в безубыток.
        Возвращает (активировать_ли, новый_уровень_SL).
        """
        if self.breakeven_applied:
            return False, self.stop_loss

        if self.entry_price <= 0:
            return False, self.stop_loss

        if self.is_long:
            profit_pct = (current_price - self.entry_price) / self.entry_price
            if profit_pct >= trigger_pct:
                new_sl = self.entry_price * (1 + offset_pct)
                if new_sl > self.stop_loss:
                    return True, new_sl
        else:
            profit_pct = (self.entry_price - current_price) / self.entry_price
            if profit_pct >= trigger_pct:
                new_sl = self.entry_price * (1 - offset_pct)
                if self.stop_loss == 0.0 or new_sl < self.stop_loss:
                    return True, new_sl

        return False, self.stop_loss

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
            "breakeven_applied": self.breakeven_applied,
            "trailing_active": self.trailing_active,
            "native_trailing_active": self.native_trailing_active,
            "highest_price": self.highest_price,
            "position_idx": self.position_idx,
            "lowest_price": self.lowest_price if self.lowest_price != float("inf") else self.entry_price,
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
            breakeven_applied=bool(data.get("breakeven_applied", False)),
            trailing_active=bool(data.get("trailing_active", False)),
            native_trailing_active=bool(data.get("native_trailing_active", False)),
            highest_price=float(data.get("highest_price", 0.0)),
            position_idx=int(data.get("position_idx", 0)),
            lowest_price=float(data.get("lowest_price", float("inf"))),
        )

    def check_partial_tp(self, current_price: float, ratio: float = 0.5, offset_pct: float = 0.002) -> tuple[bool, float, float]:
        """
        Проверяет достижение TP1 = 1.5R.
        Возвращает (сработал_ли, объем_к_закрытию, новый_уровень_SL_б/у).
        """
        if self.tp1_executed or self.tp1_price <= 0:
            return False, 0.0, self.stop_loss

        qty_to_close = self.size * ratio

        if self.is_long:
            if current_price >= self.tp1_price:
                # Стоп в безубыток (+offset_pct на комиссию)
                be_sl = self.entry_price * (1.0 + offset_pct)
                return True, qty_to_close, be_sl
        else:
            if current_price <= self.tp1_price:
                # Стоп в безубыток (-offset_pct на комиссию)
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
        fallback_pct: float = 0.01
    ) -> tuple[bool, float]:
        """
        Проверяет и подтягивает динамический трейлинг-стоп вслед за экстремумом.
        Стоп может двигаться ТОЛЬКО в сторону прибыли позиции.
        Возвращает (обновлен_ли, новый_SL).
        """
        self.update_price_extremes(high_price, low_price)

        # Расчет дистанции трейлинга
        if self.atr_value > 0:
            dist = self.atr_value * atr_mult
        else:
            dist = current_price * fallback_pct

        if self.is_long:
            profit_pct = (current_price - self.entry_price) / self.entry_price
            # Активация трейлинга после достижения триггера
            if profit_pct >= trigger_pct or self.trailing_active:
                self.trailing_active = True
                candidate_sl = self.highest_price - dist
                # Трейлинг двигается только вверх
                if candidate_sl > self.stop_loss:
                    return True, candidate_sl
        else:
            profit_pct = (self.entry_price - current_price) / self.entry_price
            if profit_pct >= trigger_pct or self.trailing_active:
                self.trailing_active = True
                candidate_sl = self.lowest_price + dist
                # Трейлинг двигается только вниз
                if self.stop_loss == 0.0 or candidate_sl < self.stop_loss:
                    return True, candidate_sl

        return False, self.stop_loss


class MarketRegimeFilter:
    """
    Фильтр рыночного сентимента по бенчмарку (BTCUSDT).
    Блокирует открытие:
    - Long-позиций при медвежьем тренде или обвале (Market Dump).
    - Short-позиций при мощном бычьем тренде или резком пампе (Market Pump).
    """

    def __init__(
        self,
        benchmark_symbol: str = "BTCUSDT",
        ema_period: int = 200,
        rsi_threshold: float = 45.0,
        bull_rsi_threshold: float = 55.0,
        dump_pct_threshold: float = 0.03,
        pump_pct_threshold: float = 0.04,
        dump_lookback_candles: int = 4
    ):
        self.benchmark_symbol = benchmark_symbol
        self.ema_period = ema_period
        self.rsi_threshold = rsi_threshold
        self.bull_rsi_threshold = bull_rsi_threshold
        self.dump_pct_threshold = dump_pct_threshold
        self.pump_pct_threshold = pump_pct_threshold
        self.dump_lookback_candles = dump_lookback_candles

    @staticmethod
    def add_indicators(
        df: pd.DataFrame,
        ema_period: int = 200,
        rsi_period: int = 14,
        dump_lookback_candles: int = 4,
        dump_pct_threshold: float = 0.03,
        pump_pct_threshold: float = 0.04,
        rsi_threshold: float = 45.0,
        bull_rsi_threshold: float = 55.0
    ) -> pd.DataFrame:
        """
        Векторный предварительный расчет индикаторов бенчмарка для O(1) проверки в бэктестере.
        """
        if df.empty or len(df) < 20:
            return df
        res = df.copy()
        res["BENCH_EMA"] = res["close"].ewm(span=ema_period, adjust=False).mean()
        delta = res["close"].diff()
        gain = delta.clip(lower=0).ewm(alpha=1/rsi_period, adjust=False).mean()
        loss = -delta.clip(upper=0).ewm(alpha=1/rsi_period, adjust=False).mean()
        rs = gain / loss.replace(0, np.nan)
        res["BENCH_RSI"] = 100 - (100 / (1 + rs))
        res["BENCH_RSI"] = res["BENCH_RSI"].where(~((loss == 0) & (gain > 0)), 100.0)
        res["BENCH_RSI"] = res["BENCH_RSI"].fillna(50.0)

        # Скользящие экстремумы для детекции Dump / Pump
        res["BENCH_HIGH_MAX"] = res["high"].rolling(window=dump_lookback_candles, min_periods=1).max()
        res["BENCH_LOW_MIN"] = res["low"].rolling(window=dump_lookback_candles, min_periods=1).min()
        res["BENCH_CLOSE"] = res["close"]

        # Векторные проверки дампа и пампа
        bench_high = res["BENCH_HIGH_MAX"]
        dump_drop = (bench_high - res["close"]) / bench_high.replace(0, np.nan)
        res["BENCH_DUMP"] = (dump_drop >= dump_pct_threshold).fillna(False)

        bench_low = res["BENCH_LOW_MIN"]
        pump_gain = (res["close"] - bench_low) / bench_low.replace(0, np.nan)
        res["BENCH_PUMP"] = (pump_gain >= pump_pct_threshold).fillna(False)

        # Флаги разрешения лонга и шорта для альткоинов
        res["BENCH_LONG_ALLOWED"] = (
            (~res["BENCH_DUMP"]) &
            (res["close"] >= res["BENCH_EMA"]) &
            (res["BENCH_RSI"] >= rsi_threshold)
        )
        res["BENCH_SHORT_ALLOWED"] = (
            (~res["BENCH_PUMP"]) &
            ~((res["close"] > res["BENCH_EMA"]) & (res["BENCH_RSI"] > bull_rsi_threshold))
        )
        # Для самого бенчмарка (только защита от dump/pump)
        res["BENCH_SELF_LONG_ALLOWED"] = ~res["BENCH_DUMP"]
        res["BENCH_SELF_SHORT_ALLOWED"] = ~res["BENCH_PUMP"]

        return res

    def is_long_allowed(self, symbol: str, btc_df: pd.DataFrame) -> tuple[bool, str]:
        """
        Проверяет, разрешен ли вход в LONG для указанного символа.
        :param symbol: Торгуемая пара (например, "SOLUSDT")
        :param btc_df: Датафрейм свечей бенчмарка (BTCUSDT)
        :return: (разрешен_ли, причина_запрета_или_статус)
        """
        if btc_df is None or btc_df.empty or len(btc_df) < max(self.ema_period, 30):
            return True, "Нет достаточных данных BTC для фильтра сентимента"

        df = btc_df
        close = df["close"]

        # 1. Проверка на Market Dump
        if "BENCH_DUMP" in df.columns:
            is_dump = bool(df["BENCH_DUMP"].iloc[-1])
        else:
            lookback = min(self.dump_lookback_candles, len(df) - 1)
            recent_high = df["high"].iloc[-lookback:].max()
            last_close = close.iloc[-1]
            dump_drop = (recent_high - last_close) / recent_high if recent_high > 0 else 0.0
            is_dump = (dump_drop >= self.dump_pct_threshold)

        if is_dump:
            msg = (
                f"Рыночный дамп бенчмарка {self.benchmark_symbol}. "
                f"Все лонг-входы заблокированы."
            )
            return False, msg

        # Если проверяем сам BTC, дальнейшие фильтры альткоинов не применяются
        if symbol.upper() == self.benchmark_symbol.upper():
            return True, "OK (Бенчмарк)"

        # 2. Проверка старшей EMA
        last_close = close.iloc[-1]
        if "BENCH_EMA" in df.columns:
            last_ema = df["BENCH_EMA"].iloc[-1]
        else:
            last_ema = close.ewm(span=self.ema_period, adjust=False).mean().iloc[-1]

        if last_close < last_ema:
            msg = (
                f"Фильтр сентимента: {self.benchmark_symbol} ({last_close:.1f}) ниже EMA "
                f"({last_ema:.1f}). Лонг по {symbol} отклонен."
            )
            return False, msg

        # 3. Проверка RSI бенчмарка
        if "BENCH_RSI" in df.columns:
            last_rsi = df["BENCH_RSI"].iloc[-1]
        else:
            delta = close.diff()
            gain = delta.clip(lower=0).ewm(alpha=1/14, adjust=False).mean()
            loss = -delta.clip(upper=0).ewm(alpha=1/14, adjust=False).mean()
            last_gain = float(gain.iloc[-1])
            last_loss = float(loss.iloc[-1])
            if last_loss == 0.0:
                last_rsi = 100.0 if last_gain > 0 else 50.0
            else:
                rs = last_gain / last_loss
                last_rsi = float(100 - (100 / (1 + rs)))

        if last_rsi < self.rsi_threshold:
            msg = (
                f"Фильтр сентимента: RSI {self.benchmark_symbol} ({last_rsi:.1f}) ниже порога "
                f"{self.rsi_threshold:.1f}. Лонг по {symbol} отклонен."
            )
            return False, msg

        return True, "OK"

    def is_short_allowed(self, symbol: str, btc_df: pd.DataFrame) -> tuple[bool, str]:
        """
        Проверяет, разрешен ли вход в SHORT для указанного символа.
        Блокирует шорты при сильном бычьем тренде или резком пампе BTC.
        """
        if btc_df is None or btc_df.empty or len(btc_df) < max(self.ema_period, 30):
            return True, "Нет достаточных данных BTC для фильтра сентимента"

        df = btc_df
        close = df["close"]

        # 1. Проверка на Market Pump
        if "BENCH_PUMP" in df.columns:
            is_pump = bool(df["BENCH_PUMP"].iloc[-1])
        else:
            lookback = min(self.dump_lookback_candles, len(df) - 1)
            recent_low = df["low"].iloc[-lookback:].min()
            last_close = close.iloc[-1]
            pump_gain = (last_close - recent_low) / recent_low if recent_low > 0 else 0.0
            is_pump = (pump_gain >= self.pump_pct_threshold)

        if is_pump:
            msg = (
                f"Рыночный памп бенчмарка {self.benchmark_symbol}. "
                f"Все шорт-входы заблокированы."
            )
            return False, msg

        if symbol.upper() == self.benchmark_symbol.upper():
            return True, "OK (Бенчмарк)"

        # 2. Проверка старшей EMA
        last_close = close.iloc[-1]
        if "BENCH_EMA" in df.columns:
            last_ema = df["BENCH_EMA"].iloc[-1]
        else:
            last_ema = close.ewm(span=self.ema_period, adjust=False).mean().iloc[-1]

        # 3. Проверка RSI бенчмарка
        if "BENCH_RSI" in df.columns:
            last_rsi = df["BENCH_RSI"].iloc[-1]
        else:
            delta = close.diff()
            gain = delta.clip(lower=0).ewm(alpha=1/14, adjust=False).mean()
            loss = -delta.clip(upper=0).ewm(alpha=1/14, adjust=False).mean()
            last_gain = float(gain.iloc[-1])
            last_loss = float(loss.iloc[-1])
            if last_loss == 0.0:
                last_rsi = 100.0 if last_gain > 0 else 50.0
            else:
                rs = last_gain / last_loss
                last_rsi = float(100 - (100 / (1 + rs)))

        if last_close > last_ema and last_rsi > self.bull_rsi_threshold:
            msg = (
                f"Фильтр сентимента: {self.benchmark_symbol} в сильном бычьем тренде "
                f"(выше EMA и RSI {last_rsi:.1f} > {self.bull_rsi_threshold:.1f}). Шорт по {symbol} отклонен."
            )
            return False, msg

        return True, "OK"
