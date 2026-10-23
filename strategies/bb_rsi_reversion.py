"""
Стратегия возврата к средней: Bollinger Bands + RSI Reversion.
Эффективна на флэтовых и консолидирующихся рынках.
"""

import pandas as pd
import numpy as np
from strategies.base import BaseStrategy


class BbRsiReversionStrategy(BaseStrategy):
    """
    Стратегия Mean-Reversion по полосам Боллинджера и RSI.
    Вход в Long: цена касается или пробивает нижнюю полосу Боллинджера при перепроданности RSI (<= 35).
    Вход в Short: цена касается или пробивает верхнюю полосу Боллинджера при перекупленности RSI (>= 65).
    """

    def __init__(
        self,
        bb_period: int = 20,
        bb_std: float = 2.0,
        rsi_period: int = 14,
        rsi_oversold: float = 35.0,
        rsi_overbought: float = 65.0,
        atr_period: int = 14,
        use_mtf_filter: bool = False,
        htf_ema_period: int = 200,
        **kwargs
    ):
        super().__init__(name="BB_RSI", use_mtf_filter=use_mtf_filter, htf_ema_period=htf_ema_period, **kwargs)
        self.bb_period = bb_period
        self.bb_std = bb_std
        self.rsi_period = rsi_period
        self.rsi_oversold = rsi_oversold
        self.rsi_overbought = rsi_overbought
        self.atr_period = atr_period
        self.use_mtf_filter = use_mtf_filter
        self.htf_ema_period = htf_ema_period

    def calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        if len(df) < self.bb_period + 5:
            df["signal"] = 0
            df["ATR"] = 0.0
            return df

        upper, middle, lower, width = self.calc_bollinger_bands(df["close"], self.bb_period, self.bb_std)
        df["BB_upper"] = upper
        df["BB_middle"] = middle
        df["BB_lower"] = lower
        df["BB_width"] = width
        df["RSI"] = self.calc_rsi(df["close"], self.rsi_period)
        df["ATR"] = self.calc_atr(df, self.atr_period)

        df["signal"] = 0

        # Условие отскока от нижней полосы Боллинджера (Long Reversion)
        long_cond = (df["low"] <= df["BB_lower"]) & (df["RSI"] <= self.rsi_oversold) & (df["close"] > df["open"])
        # Условие отскока от верхней полосы Боллинджера (Short Reversion)
        short_cond = (df["high"] >= df["BB_upper"]) & (df["RSI"] >= self.rsi_overbought) & (df["close"] < df["open"])

        df.loc[long_cond, "signal"] = 1
        df.loc[short_cond, "signal"] = -1
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
