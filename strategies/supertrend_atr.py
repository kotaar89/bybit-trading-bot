"""
Трендовая стратегия: SuperTrend + ATR Breakout.
Надежная альтернатива классическим пересечениям скользящих средних.
"""

import pandas as pd
import numpy as np
from strategies.base import BaseStrategy


class SupertrendAtrStrategy(BaseStrategy):
    """
    Трендовая стратегия на базе индикатора SuperTrend и ATR волатильности.
    Генерирует вход в Long при смене тренда на бычий (flip -1 -> 1)
    и вход в Short при смене тренда на медвежий (flip 1 -> -1).
    """

    def __init__(
        self,
        st_period: int = 10,
        st_multiplier: float = 3.0,
        atr_period: int = 14,
        volume_filter: bool = True,
        vol_sma_period: int = 20,
        vol_threshold_mult: float = 0.85,
        use_mtf_filter: bool = True,
        htf_ema_period: int = 200,
        **kwargs
    ):
        super().__init__(name="SUPERTREND", use_mtf_filter=use_mtf_filter, htf_ema_period=htf_ema_period, **kwargs)
        self.st_period = st_period
        self.st_multiplier = st_multiplier
        self.atr_period = atr_period
        self.volume_filter = volume_filter
        self.vol_sma_period = vol_sma_period
        self.vol_threshold_mult = vol_threshold_mult
        self.use_mtf_filter = use_mtf_filter
        self.htf_ema_period = htf_ema_period

    def calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        if len(df) < max(self.st_period, self.vol_sma_period) + 5:
            df["signal"] = 0
            df["ATR"] = 0.0
            return df

        supertrend, trend = self.calc_supertrend(df, period=self.st_period, multiplier=self.st_multiplier)
        df["SuperTrend"] = supertrend
        df["Trend"] = trend
        df["ATR"] = self.calc_atr(df, self.atr_period)
        df["VOL_SMA"] = self.calc_sma(df["volume"], self.vol_sma_period)

        df["signal"] = 0

        # Сигналы при смене направления SuperTrend
        long_flip = (df["Trend"] == 1) & (df["Trend"].shift(1) == -1)
        short_flip = (df["Trend"] == -1) & (df["Trend"].shift(1) == 1)

        if self.volume_filter:
            vol_ok = df["volume"] >= (df["VOL_SMA"] * self.vol_threshold_mult)
        else:
            vol_ok = pd.Series(True, index=df.index)

        df.loc[long_flip & vol_ok, "signal"] = 1
        df.loc[short_flip & vol_ok, "signal"] = -1
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


class SupertrendProStrategy(SupertrendAtrStrategy):
    """
    Улучшенная трендовая стратегия v2.2: SuperTrend + ADX фильтр флэта + EMA50 + защита от пиков RSI.
    Генерирует сигналы при развороте SuperTrend только при наличии импульса (ADX >= adx_threshold),
    подтверждении направления по EMA(50) и отсутствии экстремальной перекупленности/перепроданности RSI.
    """

    def __init__(
        self,
        st_period: int = 10,
        st_multiplier: float = 3.0,
        atr_period: int = 14,
        volume_filter: bool = True,
        vol_sma_period: int = 20,
        vol_threshold_mult: float = 0.85,
        use_mtf_filter: bool = True,
        htf_ema_period: int = 200,
        use_adx_filter: bool = True,
        adx_period: int = 14,
        adx_threshold: float = 18.0,
        confirm_ema_period: int = 50,
        rsi_period: int = 14,
        rsi_long_max: float = 72.0,
        rsi_short_min: float = 28.0,
        **kwargs
    ):
        super().__init__(
            st_period=st_period,
            st_multiplier=st_multiplier,
            atr_period=atr_period,
            volume_filter=volume_filter,
            vol_sma_period=vol_sma_period,
            vol_threshold_mult=vol_threshold_mult,
            use_mtf_filter=use_mtf_filter,
            htf_ema_period=htf_ema_period,
            **kwargs
        )
        self.name = "SUPERTREND_PRO"
        self.use_adx_filter = use_adx_filter
        self.adx_period = adx_period
        self.adx_threshold = adx_threshold
        self.confirm_ema_period = confirm_ema_period
        self.rsi_period = rsi_period
        self.rsi_long_max = rsi_long_max
        self.rsi_short_min = rsi_short_min

    def calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        df = super().calculate_indicators(df)
        if len(df) < max(self.confirm_ema_period, self.st_period, self.vol_sma_period) + 5:
            return df

        df["ADX"] = self.calc_adx(df, self.adx_period)
        df["RSI"] = self.calc_rsi(df["close"], self.rsi_period)
        df["EMA50"] = self.calc_ema(df["close"], self.confirm_ema_period)

        if self.use_adx_filter:
            adx_ok = df["ADX"] >= self.adx_threshold
        else:
            adx_ok = pd.Series(True, index=df.index)

        long_ok = adx_ok & (df["close"] >= df["EMA50"]) & (df["RSI"] <= self.rsi_long_max)
        short_ok = adx_ok & (df["close"] <= df["EMA50"]) & (df["RSI"] >= self.rsi_short_min)

        df.loc[(df["signal"] == 1) & (~long_ok), "signal"] = 0
        df.loc[(df["signal"] == -1) & (~short_ok), "signal"] = 0
        return df

