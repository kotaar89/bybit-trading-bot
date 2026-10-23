"""
Базовый абстрактный класс для торговых стратегий.
Определяет единый интерфейс расчета индикаторов, генерации сигналов и управления стопами.
"""

from abc import ABC, abstractmethod
import pandas as pd
import numpy as np


class BaseStrategy(ABC):
    """
    Абстрактный класс торговой стратегии.
    Все стратегии наследуются от него и реализуют расчет индикаторов и генерацию сигналов.
    """

    def __init__(self, name: str = "BaseStrategy", **kwargs):
        self.name = name
        self.params = kwargs
        self.use_mtf_filter = bool(kwargs.get("use_mtf_filter", True))
        self.htf_ema_period = int(kwargs.get("htf_ema_period", 200))

    def get_mtf_trend(self, htf_df: pd.DataFrame) -> str:
        """
        Определяет тренд на старшем таймфрейме: 'bullish', 'bearish' или 'neutral'.
        Доступен для всех стратегий, наследуемых от BaseStrategy.
        """
        if not getattr(self, "use_mtf_filter", True) or htf_df is None or htf_df.empty:
            return "neutral"

        ema_period = getattr(self, "htf_ema_period", 200)
        if len(htf_df) < ema_period + 5:
            return "neutral"

        ema_htf = self.calc_ema(htf_df["close"], ema_period)
        last_close = htf_df["close"].iloc[-2]  # по закрытой свече
        last_ema = ema_htf.iloc[-2]

        if last_close > last_ema:
            return "bullish"
        elif last_close < last_ema:
            return "bearish"
        return "neutral"

    def apply_mtf_filter_to_df(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Фильтрует колонку 'signal' по старшему тренду, если в df присутствует колонка 'HTF_EMA'.
        """
        if getattr(self, "use_mtf_filter", False) and "HTF_EMA" in df.columns and "signal" in df.columns:
            valid_htf = df["HTF_EMA"].notna()
            df.loc[valid_htf & (df["signal"] == 1) & (df["close"] <= df["HTF_EMA"]), "signal"] = 0
            df.loc[valid_htf & (df["signal"] == -1) & (df["close"] >= df["HTF_EMA"]), "signal"] = 0
        return df

    @abstractmethod
    def calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Рассчитывает технические индикаторы и добавляет их колонками в датафрейм.
        :param df: DataFrame со свечами (open, high, low, close, volume)
        :return: DataFrame с добавленными индикаторами
        """
        pass

    @abstractmethod
    def generate_signal(self, df: pd.DataFrame, htf_df: pd.DataFrame | None = None) -> int:
        """
        Генерирует торговый сигнал по последней закрытой свече.
        :param df: DataFrame рабочего таймфрейма с рассчитанными индикаторами
        :param htf_df: Опциональный DataFrame старшего таймфрейма (MTF)
        :return: 1 (LONG), -1 (SHORT) или 0 (HOLD/NO SIGNAL)
        """
        pass

    def calculate_stops(
        self,
        entry_price: float,
        side: str,
        atr_value: float = 0.0,
        stop_loss_pct: float = 0.02,
        take_profit_pct: float = 0.045,
        atr_sl_mult: float = 1.5,
        atr_tp_mult: float = 3.5,
        use_atr: bool = True
    ) -> dict:
        """
        Динамический или фиксированный расчет уровней Stop-Loss и Take-Profit.
        :return: {"sl": float, "tp": float, "sl_dist": float, "tp_dist": float, "type": str}
        """
        is_long = side.lower() in ("buy", "long")

        if use_atr and atr_value > 0:
            sl_dist = atr_value * atr_sl_mult
            tp_dist = atr_value * atr_tp_mult
            stop_type = f"ATR({atr_sl_mult}x/{atr_tp_mult}x)"
        else:
            sl_dist = entry_price * stop_loss_pct
            tp_dist = entry_price * take_profit_pct
            stop_type = "Fixed"

        if is_long:
            sl = entry_price - sl_dist
            tp = entry_price + tp_dist
        else:
            sl = entry_price + sl_dist
            tp = entry_price - tp_dist

        return {
            "sl": sl,
            "tp": tp,
            "sl_dist": sl_dist,
            "tp_dist": tp_dist,
            "type": stop_type
        }

    @staticmethod
    def calc_rsi(series: pd.Series, period: int = 14) -> pd.Series:
        """Расчет Wilder's RSI"""
        delta = series.diff()
        gain = delta.clip(lower=0).ewm(alpha=1/period, adjust=False).mean()
        loss = -delta.clip(upper=0).ewm(alpha=1/period, adjust=False).mean()
        rs = gain / loss.replace(0, np.nan)
        rsi = 100 - (100 / (1 + rs))
        rsi = rsi.where(~((loss == 0) & (gain > 0)), 100.0)
        return rsi.fillna(50.0)

    @staticmethod
    def calc_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
        """Расчет Average True Range (ATR)"""
        high_low = df["high"] - df["low"]
        high_close = (df["high"] - df["close"].shift(1)).abs()
        low_close = (df["low"] - df["close"].shift(1)).abs()
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        return tr.ewm(alpha=1/period, adjust=False).mean()

    @staticmethod
    def calc_ema(series: pd.Series, period: int) -> pd.Series:
        """Экспоненциальная скользящая средняя (EMA)"""
        return series.ewm(span=period, adjust=False).mean()

    @staticmethod
    def calc_sma(series: pd.Series, period: int) -> pd.Series:
        """Простая скользящая средняя (SMA)"""
        return series.rolling(period).mean()

    @staticmethod
    def calc_adx(df: pd.DataFrame, period: int = 14) -> pd.Series:
        """
        Расчет Average Directional Index (ADX) для фильтрации флэта.
        ADX > 22-25: рынок в направленном тренде.
        ADX < 20: консолидация / боковик (флэт).
        """
        if len(df) < period + 2:
            return pd.Series(0.0, index=df.index)

        up_move = df["high"] - df["high"].shift(1)
        down_move = df["low"].shift(1) - df["low"]

        plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
        minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)

        high_low = df["high"] - df["low"]
        high_close = (df["high"] - df["close"].shift(1)).abs()
        low_close = (df["low"] - df["close"].shift(1)).abs()
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        tr_smooth = tr.ewm(alpha=1/period, adjust=False).mean().replace(0, np.nan)

        plus_di = 100 * (pd.Series(plus_dm, index=df.index).ewm(alpha=1/period, adjust=False).mean() / tr_smooth)
        minus_di = 100 * (pd.Series(minus_dm, index=df.index).ewm(alpha=1/period, adjust=False).mean() / tr_smooth)

        di_sum = (plus_di + minus_di).replace(0, np.nan)
        dx = 100 * (plus_di - minus_di).abs() / di_sum
        adx = dx.ewm(alpha=1/period, adjust=False).mean().fillna(20.0)
        return adx

    @staticmethod
    def calc_bollinger_bands(series: pd.Series, period: int = 20, std_dev: float = 2.0) -> tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
        """
        Расчет полос Боллинджера (Bollinger Bands).
        Возвращает (upper, middle, lower, bandwidth_pct).
        """
        middle = series.rolling(period).mean()
        std = series.rolling(period).std()
        upper = middle + (std_dev * std)
        lower = middle - (std_dev * std)
        width = (upper - lower) / middle.replace(0, np.nan)
        return upper, middle, lower, width

    @staticmethod
    def calc_donchian_channels(df: pd.DataFrame, period: int = 20) -> tuple[pd.Series, pd.Series, pd.Series]:
        """
        Каналы Дончяна (Donchian Channels).
        Возвращает (upper, middle, lower).
        """
        upper = df["high"].rolling(period).max()
        lower = df["low"].rolling(period).min()
        middle = (upper + lower) / 2.0
        return upper, middle, lower

    @staticmethod
    def calc_supertrend(df: pd.DataFrame, period: int = 10, multiplier: float = 3.0) -> tuple[pd.Series, pd.Series]:
        """
        Индикатор SuperTrend.
        Возвращает (supertrend_line, trend_direction), где trend_direction: 1 (Bull) или -1 (Bear).
        """
        high_low = df["high"] - df["low"]
        high_close = (df["high"] - df["close"].shift(1)).abs()
        low_close = (df["low"] - df["close"].shift(1)).abs()
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        atr = tr.ewm(alpha=1/period, adjust=False).mean()

        hl2 = (df["high"] + df["low"]) / 2.0
        basic_upper = hl2 + (multiplier * atr)
        basic_lower = hl2 - (multiplier * atr)

        final_upper = basic_upper.copy()
        final_lower = basic_lower.copy()
        trend = pd.Series(1, index=df.index, dtype=int)

        close = df["close"].values
        b_up = basic_upper.values
        b_low = basic_lower.values
        f_up = final_upper.to_numpy(copy=True)
        f_low = final_lower.to_numpy(copy=True)
        t_arr = trend.to_numpy(copy=True)

        for i in range(1, len(df)):
            if b_low[i] > f_low[i-1] or close[i-1] < f_low[i-1]:
                f_low[i] = b_low[i]
            else:
                f_low[i] = f_low[i-1]

            if b_up[i] < f_up[i-1] or close[i-1] > f_up[i-1]:
                f_up[i] = b_up[i]
            else:
                f_up[i] = f_up[i-1]

            if close[i] > f_up[i-1]:
                t_arr[i] = 1
            elif close[i] < f_low[i-1]:
                t_arr[i] = -1
            else:
                t_arr[i] = t_arr[i-1]

        supertrend = pd.Series(np.where(t_arr == 1, f_low, f_up), index=df.index)
        trend_ser = pd.Series(t_arr, index=df.index)
        return supertrend, trend_ser
