"""
Стратегия пробоя ценового канала: Donchian Breakout (Черепахи).
Классический трендовый подход Ричарда Денниса.
"""

import pandas as pd
import numpy as np
from strategies.base import BaseStrategy


class DonchianBreakoutStrategy(BaseStrategy):
    """
    Стратегия пробоя каналов Дончяна (Turtle Trading Strategy).
    Вход в Long: пробой максимума за последние N свечей (20-периодный High).
    Вход в Short: пробой минимума за последние N свечей (20-периодный Low).
    """

    def __init__(
        self,
        lookback_period: int = 20,
        atr_period: int = 14,
        volume_filter: bool = True,
        vol_sma_period: int = 20,
        vol_threshold_mult: float = 0.85,
        use_mtf_filter: bool = True,
        htf_ema_period: int = 200,
        **kwargs
    ):
        super().__init__(name="DONCHIAN", use_mtf_filter=use_mtf_filter, htf_ema_period=htf_ema_period, **kwargs)
        self.lookback_period = lookback_period
        self.atr_period = atr_period
        self.volume_filter = volume_filter
        self.vol_sma_period = vol_sma_period
        self.vol_threshold_mult = vol_threshold_mult
        self.use_mtf_filter = use_mtf_filter
        self.htf_ema_period = htf_ema_period

    def calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        if len(df) < max(self.lookback_period, self.vol_sma_period) + 5:
            df["signal"] = 0
            df["ATR"] = 0.0
            return df

        # Каналы Дончяна по предыдущим барам (shift 1 исключает lookahead)
        upper, middle, lower = self.calc_donchian_channels(df, self.lookback_period)
        df["Donchian_upper"] = upper.shift(1)
        df["Donchian_lower"] = lower.shift(1)
        df["Donchian_middle"] = middle.shift(1)

        df["ATR"] = self.calc_atr(df, self.atr_period)
        df["VOL_SMA"] = self.calc_sma(df["volume"], self.vol_sma_period)

        df["signal"] = 0

        long_breakout = df["close"] > df["Donchian_upper"]
        short_breakout = df["close"] < df["Donchian_lower"]

        if self.volume_filter:
            vol_ok = df["volume"] >= (df["VOL_SMA"] * self.vol_threshold_mult)
        else:
            vol_ok = pd.Series(True, index=df.index)

        df.loc[long_breakout & vol_ok, "signal"] = 1
        df.loc[short_breakout & vol_ok, "signal"] = -1
        return self.apply_mtf_filter_to_df(df)

    def generate_signal(self, df: pd.DataFrame, htf_df: pd.DataFrame | None = None, candle_idx: int = -2) -> int:
        if "signal" not in df.columns or "ATR" not in df.columns:
            df = self.calculate_indicators(df)

        if len(df) < abs(candle_idx):
            return 0

        raw_signal = int(df["signal"].iloc[candle_idx])
        if raw_signal == 0:
            return 0

        if self.use_mtf_filter and htf_df is not None and "HTF_EMA" not in df.columns:
            trend = self.get_mtf_trend(htf_df)
            if raw_signal == 1 and trend == "bearish":
                return 0
            if raw_signal == -1 and trend == "bullish":
                return 0

        return raw_signal
