"""
Стратегия MA Cross + Wilder's RSI + Volume Filter + MTF Trend (4H EMA) + ADX Flat Filter.
Перенесена из bot.py / backtest.py в модульную архитектуру v2.1.
"""

import pandas as pd
import numpy as np
from strategies.base import BaseStrategy


class MaRsiStrategy(BaseStrategy):
    """
    Стратегия пересечения скользящих средних (Golden / Death Cross)
    с фильтрацией импульса по RSI, объему, старшему тренду и фильтром флэта (ADX).
    """

    def __init__(
        self,
        fast_ma: int = 20,
        slow_ma: int = 50,
        rsi_period: int = 14,
        rsi_long_min: float = 50.0,
        rsi_long_max: float = 70.0,
        rsi_short_min: float = 30.0,
        rsi_short_max: float = 50.0,
        volume_filter: bool = True,
        vol_sma_period: int = 20,
        vol_threshold_mult: float = 0.85,
        use_mtf_filter: bool = True,
        htf_ema_period: int = 200,
        atr_period: int = 14,
        cross_lookback: int = 3,
        use_adx_filter: bool = True,
        adx_period: int = 14,
        adx_threshold: float = 22.0,
        **kwargs
    ):
        super().__init__(name="MA_RSI", **kwargs)
        self.fast_ma = fast_ma
        self.slow_ma = slow_ma
        self.rsi_period = rsi_period
        self.rsi_long_min = rsi_long_min
        self.rsi_long_max = rsi_long_max
        self.rsi_short_min = rsi_short_min
        self.rsi_short_max = rsi_short_max
        self.volume_filter = volume_filter
        self.vol_sma_period = vol_sma_period
        self.vol_threshold_mult = vol_threshold_mult
        self.use_mtf_filter = use_mtf_filter
        self.htf_ema_period = htf_ema_period
        self.atr_period = atr_period
        self.cross_lookback = max(1, cross_lookback)
        self.use_adx_filter = use_adx_filter
        self.adx_period = adx_period
        self.adx_threshold = adx_threshold

    def calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Рассчитывает индикаторы рабочего таймфрейма и размечает вектор сигналов.
        """
        df = df.copy()

        if len(df) < self.slow_ma + 5:
            df["signal"] = 0
            df["ATR"] = 0.0
            return df

        df["MA_fast"] = self.calc_sma(df["close"], self.fast_ma)
        df["MA_slow"] = self.calc_sma(df["close"], self.slow_ma)
        df["RSI"] = self.calc_rsi(df["close"], self.rsi_period)
        df["VOL_SMA"] = self.calc_sma(df["volume"], self.vol_sma_period)
        df["ATR"] = self.calc_atr(df, self.atr_period)
        df["ADX"] = self.calc_adx(df, self.adx_period)

        df["signal"] = 0

        # Условия пересечений
        golden_cross = (
            (df["MA_fast"] > df["MA_slow"]) &
            (df["MA_fast"].shift(1) <= df["MA_slow"].shift(1))
        )
        death_cross = (
            (df["MA_fast"] < df["MA_slow"]) &
            (df["MA_fast"].shift(1) >= df["MA_slow"].shift(1))
        )

        # Окно подтверждения сигнала (пересечение в пределах последних N свечей)
        if self.cross_lookback > 1:
            recent_golden = golden_cross.rolling(window=self.cross_lookback, min_periods=1).max().astype(bool) & (df["MA_fast"] > df["MA_slow"])
            recent_death = death_cross.rolling(window=self.cross_lookback, min_periods=1).max().astype(bool) & (df["MA_fast"] < df["MA_slow"])
        else:
            recent_golden = golden_cross
            recent_death = death_cross

        rsi_long = (df["RSI"] >= self.rsi_long_min) & (df["RSI"] <= self.rsi_long_max)
        rsi_short = (df["RSI"] <= self.rsi_short_max) & (df["RSI"] >= self.rsi_short_min)

        if self.volume_filter:
            vol_ok = df["volume"] >= (df["VOL_SMA"] * self.vol_threshold_mult)
        else:
            vol_ok = pd.Series(True, index=df.index)

        # Фильтр флэта по ADX (блокирует вход при отсутствии направленного движения)
        if self.use_adx_filter:
            adx_ok = df["ADX"] >= self.adx_threshold
        else:
            adx_ok = pd.Series(True, index=df.index)

        # Старший таймфрейм, если колонка HTF_EMA уже объединена с df
        if self.use_mtf_filter and "HTF_EMA" in df.columns:
            mtf_long = df["close"] > df["HTF_EMA"]
            mtf_short = df["close"] < df["HTF_EMA"]
        else:
            mtf_long = pd.Series(True, index=df.index)
            mtf_short = pd.Series(True, index=df.index)

        long_cond = recent_golden & rsi_long & vol_ok & mtf_long & adx_ok
        short_cond = recent_death & rsi_short & vol_ok & mtf_short & adx_ok

        # Защита от дублирования сигнала на соседних свечах одного и того же окна пересечения
        if self.cross_lookback > 1:
            long_cond = long_cond & (~long_cond.shift(1, fill_value=False))
            short_cond = short_cond & (~short_cond.shift(1, fill_value=False))

        df.loc[long_cond, "signal"] = 1
        df.loc[short_cond, "signal"] = -1
        return df

    def get_mtf_trend(self, htf_df: pd.DataFrame) -> str:
        """
        Определяет тренд на старшем таймфрейме: 'bullish', 'bearish' или 'neutral'.
        """
        if not self.use_mtf_filter or htf_df is None or htf_df.empty:
            return "neutral"

        if len(htf_df) < self.htf_ema_period + 5:
            return "neutral"

        ema_htf = self.calc_ema(htf_df["close"], self.htf_ema_period)
        last_close = htf_df["close"].iloc[-2]  # по закрытой свече
        last_ema = ema_htf.iloc[-2]

        if last_close > last_ema:
            return "bullish"
        elif last_close < last_ema:
            return "bearish"
        return "neutral"

    def generate_signal(
        self,
        df: pd.DataFrame,
        htf_df: pd.DataFrame | None = None,
        candle_idx: int = -2
    ) -> int:
        """
        Генерирует торговый сигнал (1: LONG, -1: SHORT, 0: HOLD)
        для указанной свечи (по умолчанию -2: последняя закрытая свеча в потоке kline).
        """
        if "signal" not in df.columns or "ATR" not in df.columns:
            df = self.calculate_indicators(df)

        if len(df) < abs(candle_idx):
            return 0

        raw_signal = int(df["signal"].iloc[candle_idx])
        if raw_signal == 0:
            return 0

        # Дополнительная проверка MTF тренда, если передан htf_df и фильтр не был включен в колонки
        if self.use_mtf_filter and htf_df is not None and "HTF_EMA" not in df.columns:
            trend = self.get_mtf_trend(htf_df)
            if raw_signal == 1 and trend == "bearish":
                return 0
            if raw_signal == -1 and trend == "bullish":
                return 0

        return raw_signal
