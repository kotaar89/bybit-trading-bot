"""
Адаптивный переключатель режимов рынка (Regime Switcher: Trend <-> Mean-Reversion) v0.3.0.
Автоматически переключает торговую логику на основе:
- ADX(14) (сила направленного тренда)
- Bollinger Bandwidth Percentile (сжатие/расширение волатильности)
- Kaufman Efficiency Ratio (ER — отношение чистого смещения цены к суммарному шуму)
"""

import numpy as np
import pandas as pd

from strategies.base import BaseStrategy
from strategies.supertrend_atr import SupertrendProStrategy
from strategies.bb_rsi_reversion import BbRsiReversionStrategy


class RegimeSwitcherStrategy(BaseStrategy):
    """
    Мета-стратегия:
    - В трендовом режиме ('TREND') использует SUPERTREND_PRO (с MTF и ADX).
    - Во флэтовом/сжатом режиме ('RANGE') переключается на BB_RSI Mean-Reversion
      со своими стопами и выходом по средней линии Боллинджера (BB_middle).
    """

    def __init__(
        self,
        st_period: int = 10,
        st_multiplier: float = 3.0,
        confirm_ema_period: int = 50,
        confirm_lookback: int = 1,
        bb_period: int = 20,
        bb_std: float = 2.0,
        rsi_period: int = 14,
        rsi_oversold: float = 35.0,
        rsi_overbought: float = 65.0,
        atr_period: int = 14,
        adx_period: int = 14,
        adx_trend_min: float = 22.0,
        adx_range_max: float = 18.0,
        er_period: int = 14,
        er_trend_min: float = 0.35,
        er_range_max: float = 0.25,
        bb_pctl_window: int = 100,
        mr_sl_mult: float = 1.8,
        mr_tp_mult: float = 2.5,
        use_mtf_filter: bool = True,
        htf_ema_period: int = 200,
        volume_filter: bool = True,
        **kwargs,
    ):
        super().__init__(
            name="REGIME_SWITCHER",
            use_mtf_filter=use_mtf_filter,
            htf_ema_period=htf_ema_period,
            **kwargs,
        )
        self.atr_period = atr_period
        self.adx_period = adx_period
        self.bb_period = bb_period
        self.bb_std = bb_std
        self.adx_trend_min = float(adx_trend_min)
        self.adx_range_max = float(adx_range_max)
        self.er_period = int(er_period)
        self.er_trend_min = float(er_trend_min)
        self.er_range_max = float(er_range_max)
        self.bb_pctl_window = max(20, int(bb_pctl_window))
        self.mr_sl_mult = float(mr_sl_mult)
        self.mr_tp_mult = float(mr_tp_mult)
        self.confirm_ema_period = confirm_ema_period
        self.st_period = st_period

        self.trend_strategy = SupertrendProStrategy(
            st_period=st_period,
            st_multiplier=st_multiplier,
            atr_period=atr_period,
            volume_filter=volume_filter,
            use_mtf_filter=use_mtf_filter,
            htf_ema_period=htf_ema_period,
            use_adx_filter=False,  # Сила тренда уже контролируется переключателем режимов
            adx_period=adx_period,
            confirm_ema_period=confirm_ema_period,
            rsi_period=rsi_period,
            confirm_lookback=confirm_lookback,
        )
        self.range_strategy = BbRsiReversionStrategy(
            bb_period=bb_period,
            bb_std=bb_std,
            rsi_period=rsi_period,
            rsi_oversold=rsi_oversold,
            rsi_overbought=rsi_overbought,
            atr_period=atr_period,
            use_mtf_filter=False,
            htf_ema_period=htf_ema_period,
        )

    @staticmethod
    def calc_efficiency_ratio(close: pd.Series, period: int = 14) -> pd.Series:
        """Kaufman Efficiency Ratio (ER) in [0, 1]."""
        if len(close) < period + 2:
            return pd.Series(0.0, index=close.index)
        direction = (close - close.shift(period)).abs()
        volatility = close.diff().abs().rolling(window=period, min_periods=period).sum().replace(0.0, np.nan)
        er = (direction / volatility).fillna(0.0).clip(0.0, 1.0)
        return er

    def calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        min_req = max(self.confirm_ema_period, self.bb_period, self.adx_period, self.er_period, 30) + 5
        if len(df) < min_req:
            df["signal"] = 0
            df["exit_signal"] = 0
            df["ATR"] = 0.0
            df["REGIME_MODE"] = "TREND"
            df["SIGNAL_REGIME"] = "TREND"
            df["mr_exit_long"] = False
            df["mr_exit_short"] = False
            return df

        trend_df = self.trend_strategy.calculate_indicators(df.copy())
        range_df = self.range_strategy.calculate_indicators(df.copy())

        adx = self.calc_adx(df, self.adx_period)
        er = self.calc_efficiency_ratio(df["close"], self.er_period)
        bb_width = range_df["BB_width"]
        min_p = min(20, self.bb_pctl_window)
        bb_pctl = bb_width.rolling(window=self.bb_pctl_window, min_periods=min_p).rank(pct=True).fillna(0.5)

        df["ADX"] = adx
        df["ER"] = er
        df["BB_upper"] = range_df["BB_upper"]
        df["BB_middle"] = range_df["BB_middle"]
        df["BB_lower"] = range_df["BB_lower"]
        df["BB_width"] = bb_width
        df["BB_width_pctl"] = bb_pctl
        df["SuperTrend"] = trend_df.get("SuperTrend", np.nan)
        df["Trend"] = trend_df.get("Trend", 0)
        df["EMA50"] = trend_df.get("EMA50", np.nan)
        df["RSI"] = range_df["RSI"]
        df["ATR"] = trend_df["ATR"]

        # Определение режима с гистерезисом
        is_trend = (adx >= self.adx_trend_min) & ((er >= self.er_trend_min) | (bb_pctl >= 0.55))
        is_range = (adx <= self.adx_range_max) & ((er <= self.er_range_max) | (bb_pctl <= 0.45))

        regimes = []
        cur_regime = "TREND"
        t_arr = is_trend.to_numpy()
        r_arr = is_range.to_numpy()
        for i in range(len(df)):
            if t_arr[i] and not r_arr[i]:
                cur_regime = "TREND"
            elif r_arr[i] and not t_arr[i]:
                cur_regime = "RANGE"
            regimes.append(cur_regime)

        df["REGIME_MODE"] = regimes
        df["SIGNAL_REGIME"] = df["REGIME_MODE"]

        is_trend_mode = df["REGIME_MODE"] == "TREND"
        is_range_mode = df["REGIME_MODE"] == "RANGE"

        df["signal"] = 0
        df.loc[is_trend_mode, "signal"] = trend_df.loc[is_trend_mode, "signal"]
        df.loc[is_range_mode, "signal"] = range_df.loc[is_range_mode, "signal"]

        # Выход по тренду и отдельный Mean-Reversion выход на средней полосе Боллинджера
        df["exit_signal"] = trend_df["exit_signal"]
        df["mr_exit_long"] = (df["close"] >= df["BB_middle"]) & df["BB_middle"].notna()
        df["mr_exit_short"] = (df["close"] <= df["BB_middle"]) & df["BB_middle"].notna()

        return df

    def generate_signal(self, df: pd.DataFrame, htf_df: pd.DataFrame | None = None, candle_idx: int = -2) -> int:
        if "signal" not in df.columns or "REGIME_MODE" not in df.columns:
            df = self.calculate_indicators(df)
        if len(df) < abs(candle_idx):
            return 0
        raw_sig = int(df["signal"].iloc[candle_idx])
        if raw_sig == 0:
            return 0
        regime = str(df["REGIME_MODE"].iloc[candle_idx])
        if regime == "TREND" and self.use_mtf_filter and htf_df is not None and "HTF_EMA" not in df.columns:
            trend = self.get_mtf_trend(htf_df)
            if raw_sig == 1 and trend == "bearish":
                return 0
            if raw_sig == -1 and trend == "bullish":
                return 0
        return raw_sig

    def calculate_stops(
        self,
        entry_price: float,
        side: str,
        atr_value: float = 0.0,
        stop_loss_pct: float = 0.02,
        take_profit_pct: float = 0.045,
        atr_sl_mult: float = 1.5,
        atr_tp_mult: float = 3.5,
        use_atr: bool = True,
        regime: str = "TREND",
    ) -> dict:
        if str(regime).upper() == "RANGE":
            res = super().calculate_stops(
                entry_price=entry_price,
                side=side,
                atr_value=atr_value,
                stop_loss_pct=stop_loss_pct,
                take_profit_pct=take_profit_pct,
                atr_sl_mult=self.mr_sl_mult,
                atr_tp_mult=self.mr_tp_mult,
                use_atr=use_atr,
            )
            res["type"] = f"MR-{res['type']}"
            return res
        return super().calculate_stops(
            entry_price=entry_price,
            side=side,
            atr_value=atr_value,
            stop_loss_pct=stop_loss_pct,
            take_profit_pct=take_profit_pct,
            atr_sl_mult=atr_sl_mult,
            atr_tp_mult=atr_tp_mult,
            use_atr=use_atr,
        )
