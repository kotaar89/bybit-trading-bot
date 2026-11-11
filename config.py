"""
Модуль конфигурации Bybit Futures торгового бота (v2.2)
Загружает чувствительные данные из .env и предоставляет типизированные настройки с fallback-значениями.
"""

import os
from pathlib import Path

# Базовая директория проекта для единых абсолютных путей (Termux / Windows / Linux)
BASE_DIR         = Path(__file__).resolve().parent
DATA_DIR         = BASE_DIR / "data"
BOT_LOG_FILE     = BASE_DIR / "bot.log"
TRADE_LOG_FILE   = BASE_DIR / "trade.log"
DB_FILE          = BASE_DIR / "bot_data.db"
STATE_FILE       = BASE_DIR / "bot_state.json"
CONTROL_FILE     = BASE_DIR / "bot_control.json"
LIVE_STATUS_FILE = BASE_DIR / "bot_live_status.json"
PROFILES_FILE    = BASE_DIR / "strategy_profiles.json"

# Попытка загрузить переменные из .env файла
try:
    from dotenv import load_dotenv
    env_path = BASE_DIR / ".env"
    load_dotenv(dotenv_path=env_path)
except ImportError:
    pass


def _get_bool(key: str, default: bool) -> bool:
    val = os.getenv(key)
    if val is None:
        return default
    return val.strip().lower() in ("1", "true", "yes", "y", "on")


def _get_float(key: str, default: float) -> float:
    val = os.getenv(key)
    if val is None:
        return default
    try:
        return float(val.strip())
    except ValueError:
        return default


def _get_int(key: str, default: int) -> int:
    val = os.getenv(key)
    if val is None:
        return default
    try:
        return int(val.strip())
    except ValueError:
        return default


# =====================================================================
# 1. РЕЖИМ ТОРГОВЛИ И КЛЮЧИ BYBIT
# =====================================================================
VALID_TRADING_MODES = ("DEMO", "TESTNET", "LIVE")


def is_placeholder_key(val: str) -> bool:
    v = (val or "").strip()
    if not v:
        return True
    low = v.lower()
    return low.startswith("your_") or low in ("changeme", "placeholder", "none", "null")


TRADING_MODE = os.getenv("TRADING_MODE", "DEMO").strip().upper()

# Ключи для ДЕМО-счета (Bybit Demo Trading)
DEMO_API_KEY    = os.getenv("DEMO_API_KEY", "your_demo_key").strip()
DEMO_API_SECRET = os.getenv("DEMO_API_SECRET", "your_demo_secret").strip()

# Ключи для БОЕВОГО счета (Bybit Live Mainnet)
LIVE_API_KEY    = os.getenv("LIVE_API_KEY", "your_live_key").strip()
LIVE_API_SECRET = os.getenv("LIVE_API_SECRET", "your_live_secret").strip()


def resolve_trading_mode(mode: str, validate_live_keys: bool = True) -> tuple[str, str, str, bool, bool]:
    """
    Строго валидирует торговый режим (fail-closed).
    Исключает молчаливый переход в LIVE при опечатках вроде DEM0 / PAPER.
    """
    m = (mode or "").strip().upper()
    if m not in VALID_TRADING_MODES:
        raise ValueError(
            f"Некорректный TRADING_MODE={mode!r}. Допустимые значения: {', '.join(VALID_TRADING_MODES)}"
        )
    if m == "DEMO":
        return m, DEMO_API_KEY, DEMO_API_SECRET, True, False
    if m == "TESTNET":
        return m, DEMO_API_KEY, DEMO_API_SECRET, False, True
    if validate_live_keys and (is_placeholder_key(LIVE_API_KEY) or is_placeholder_key(LIVE_API_SECRET)):
        raise ValueError(
            "TRADING_MODE=LIVE невозможен с плейсхолдерными или пустыми LIVE_API_KEY / LIVE_API_SECRET."
        )
    return m, LIVE_API_KEY, LIVE_API_SECRET, False, False


# Строгий выбор рабочего ключа (fail-closed)
TRADING_MODE, API_KEY, API_SECRET, DEMO, TESTNET = resolve_trading_mode(TRADING_MODE, validate_live_keys=True)


# =====================================================================
# 2. СЕТЬ И ВЕБ-ДАШБОРД (TERMUX / LOCAL NETWORK)
# =====================================================================
DASHBOARD_HOST = os.getenv("DASHBOARD_HOST", "0.0.0.0").strip()
DASHBOARD_PORT = _get_int("DASHBOARD_PORT", 8000)
DASHBOARD_USERNAME = os.getenv("DASHBOARD_USERNAME", "").strip()
DASHBOARD_PASSWORD = os.getenv("DASHBOARD_PASSWORD", "").strip()


# =====================================================================
# 3. АКТИВНАЯ СТРАТЕГИЯ И ПАРЫ
# =====================================================================
ACTIVE_STRATEGY = os.getenv("ACTIVE_STRATEGY", "SUPERTREND_PRO").strip()

# Список торгуемых пар (мультивалютность — без BTCUSDT, т.к. BTC используется как 4H бенчмарк сентимента)
raw_symbols = os.getenv("SYMBOLS", "ETHUSDT,SOLUSDT,AVAXUSDT,NEARUSDT")
SYMBOLS = [s.strip().upper() for s in raw_symbols.split(",") if s.strip()]
SYMBOL  = SYMBOLS[0] if SYMBOLS else "ETHUSDT"

# Таймфреймы
INTERVAL        = os.getenv("INTERVAL", "60").strip()         # Рабочий ТФ входа (60 мин / 1 час)
HIGHER_INTERVAL = os.getenv("HIGHER_INTERVAL", "240").strip()  # Старший ТФ тренда (240 мин / 4 часа)


# =====================================================================
# 4. ПАРАМЕТРЫ СТРАТЕГИЙ (SUPERTREND_PRO / MA_RSI) И ФИЛЬТРЫ
# =====================================================================
ST_PERIOD           = _get_int("ST_PERIOD", 10)
ST_MULTIPLIER       = _get_float("ST_MULTIPLIER", 3.0)
ST_CONFIRM_EMA      = _get_int("ST_CONFIRM_EMA", 50)
ST_CONFIRM_LOOKBACK = _get_int("ST_CONFIRM_LOOKBACK", 1)

FAST_MA        = _get_int("FAST_MA", 20)
SLOW_MA        = _get_int("SLOW_MA", 50)
RSI_PERIOD     = _get_int("RSI_PERIOD", 14)
RSI_LONG_MIN   = _get_float("RSI_LONG_MIN", 50.0)
RSI_LONG_MAX   = _get_float("RSI_LONG_MAX", 72.0)
RSI_SHORT_MAX  = _get_float("RSI_SHORT_MAX", 50.0)
RSI_SHORT_MIN  = _get_float("RSI_SHORT_MIN", 28.0)
VOLUME_FILTER  = _get_bool("VOLUME_FILTER", True)

# Фильтр флэта ADX (Average Directional Index)
USE_ADX_FILTER = _get_bool("USE_ADX_FILTER", True)
ADX_PERIOD     = _get_int("ADX_PERIOD", 14)
ADX_THRESHOLD  = _get_float("ADX_THRESHOLD", 18.0)  # Входы при ADX >= 18 (фильтрация мертвого флэта)

# Фильтр старшего тренда (MTF)
USE_MTF_FILTER = _get_bool("USE_MTF_FILTER", True)
HTF_EMA_PERIOD = _get_int("HTF_EMA_PERIOD", 200)

# Окно валидности сигнала и защита от проскальзывания
SIGNAL_LOOKBACK_CANDLES = _get_int("SIGNAL_LOOKBACK_CANDLES", 3)
MAX_SLIPPAGE_PCT        = _get_float("MAX_SLIPPAGE_PCT", 0.015)  # 1.5% макс. невыгодное проскальзывание


# =====================================================================
# 5. ДИНАМИЧЕСКИЙ РИСК-МЕНЕДЖМЕНТ
# =====================================================================
USE_ATR_STOPS       = _get_bool("USE_ATR_STOPS", True)
ATR_PERIOD          = _get_int("ATR_PERIOD", 14)
ATR_SL_MULT         = _get_float("ATR_SL_MULT", 2.2)
ATR_TP_MULT         = _get_float("ATR_TP_MULT", 4.5)

# Страховочные фиксированные проценты (если ATR недоступен)
STOP_LOSS_PCT       = _get_float("STOP_LOSS_PCT", 0.02)
TAKE_PROFIT_PCT     = _get_float("TAKE_PROFIT_PCT", 0.045)

# Безубыток (Breakeven)
USE_BREAKEVEN       = _get_bool("USE_BREAKEVEN", True)
BREAKEVEN_TRIGGER   = _get_float("BREAKEVEN_TRIGGER", 0.015)   # Мин. % хода цены в плюс (fallback)
BREAKEVEN_R_MULT    = _get_float("BREAKEVEN_R_MULT", 1.1)      # Перевод в БУ при достижении 1.1R профита
BREAKEVEN_OFFSET    = _get_float("BREAKEVEN_OFFSET", 0.002)    # +0.2% от цены входа (комиссии)


# =====================================================================
# 6. ТРЕЙЛИНГ-СТОП, ЧАСТИЧНЫЙ TP, ФИЛЬТР РЕЖИМА РЫНКА, ИСПОЛНЕНИЕ
# =====================================================================

# 6.1. Трейлинг-стоп (Trailing Stop)
USE_TRAILING_STOP        = _get_bool("USE_TRAILING_STOP", True)
USE_NATIVE_TRAILING_STOP = _get_bool("USE_NATIVE_TRAILING_STOP", True)  # Выставлять нативный trailingStop Bybit
TRAILING_STOP_TRIGGER    = _get_float("TRAILING_STOP_TRIGGER", 0.015)   # Мин. % хода (fallback)
TRAILING_STOP_R_MULT     = _get_float("TRAILING_STOP_R_MULT", 1.4)      # Активация трейлинга от 1.4R (после TP1 и БУ)
TRAILING_STOP_ATR_MULT   = _get_float("TRAILING_STOP_ATR_MULT", 1.5)    # дистанция трейлинга в ATR
TRAILING_STOP_FALLBACK   = _get_float("TRAILING_STOP_FALLBACK", 0.01)   # 1.0% если ATR недоступен

# 6.2. Частичная фиксация прибыли (Partial TP)
USE_PARTIAL_TP      = _get_bool("USE_PARTIAL_TP", True)
USE_EXCHANGE_TP1    = _get_bool("USE_EXCHANGE_TP1", True)          # Лимитный ордер reduceOnly на бирже
PARTIAL_TP_R_MULT   = _get_float("PARTIAL_TP_R_MULT", 1.2)         # TP1 на 1.2R (R = дистанция SL)
PARTIAL_TP_RATIO    = _get_float("PARTIAL_TP_RATIO", 0.5)          # Закрывать 50% объема
PARTIAL_TP_MOVE_BE  = _get_bool("PARTIAL_TP_MOVE_BE", True)        # Переносить стоп в безубыток при TP1

# 6.3. Фильтр рыночного сентимента по BTC/ETH (Market Regime Filter)
USE_MARKET_REGIME_FILTER     = _get_bool("USE_MARKET_REGIME_FILTER", True)
BENCHMARK_SYMBOL             = os.getenv("BENCHMARK_SYMBOL", "BTCUSDT").strip().upper()
BENCHMARK_INTERVAL           = os.getenv("BENCHMARK_INTERVAL", "240").strip() # 4H
BENCHMARK_EMA_PERIOD         = _get_int("BENCHMARK_EMA_PERIOD", 200)
BENCHMARK_RSI_THRESHOLD      = _get_float("BENCHMARK_RSI_THRESHOLD", 45.0)
BENCHMARK_BULL_RSI_THRESHOLD = _get_float("BENCHMARK_BULL_RSI_THRESHOLD", 55.0) # Порог для запрета шортов
BENCHMARK_DUMP_PCT           = _get_float("BENCHMARK_DUMP_PCT", 0.03)         # Падение >= 3% блокирует лонги
BENCHMARK_PUMP_PCT           = _get_float("BENCHMARK_PUMP_PCT", 0.04)         # Рост >= 4% блокирует шорты
BENCHMARK_DUMP_CANDLES       = _get_int("BENCHMARK_DUMP_CANDLES", 4)          # За последние N свечей

# 6.4. Умное исполнение ордеров (Maker vs Taker)
USE_SMART_EXECUTION     = _get_bool("USE_SMART_EXECUTION", True)      # Limit Post-Only с таймаутом
SMART_EXECUTION_TIMEOUT = _get_int("SMART_EXECUTION_TIMEOUT", 12)     # Секунды ожидания исполнения Maker
MIN_ORDER_NOTIONAL      = _get_float("MIN_ORDER_NOTIONAL", 5.0)       # Минимальная стоимость ордера (5 USDT)


# =====================================================================
# 7. УПРАВЛЕНИЕ КАПИТАЛОМ
# =====================================================================
LEVERAGE                     = _get_int("LEVERAGE", 5)
MAX_OPEN_POSITIONS           = _get_int("MAX_OPEN_POSITIONS", 3)
MAX_SAME_DIRECTION_POSITIONS = _get_int("MAX_SAME_DIRECTION_POSITIONS", 2)
POSITION_SIZE_PCT     = _get_float("POSITION_SIZE_PCT", 0.15)
USE_RISK_BASED_SIZING = _get_bool("USE_RISK_BASED_SIZING", True)      # Расчет объема по риску от стопа
RISK_PER_TRADE_PCT    = _get_float("RISK_PER_TRADE_PCT", 0.015)       # 1.5% риска от депозита при стопе
FALLBACK_USDT         = _get_float("FALLBACK_USDT", 50.0)
MAX_DAILY_LOSS_PCT    = _get_float("MAX_DAILY_LOSS_PCT", 0.05)


# =====================================================================
# 8. ИСПОЛНЕНИЕ, WEBSOCKET KLINE, RATE LIMITS И КЭШИРОВАНИЕ (v0.3.0)
# =====================================================================
USE_WS_KLINE_TRIGGER        = _get_bool("USE_WS_KLINE_TRIGGER", True)
KLINE_FALLBACK_POLL_SEC     = _get_int("KLINE_FALLBACK_POLL_SEC", 180)
INSTRUMENT_CACHE_TTL_SEC    = _get_int("INSTRUMENT_CACHE_TTL_SEC", 86400)
USE_MAKER_CHASE             = _get_bool("USE_MAKER_CHASE", True)
MAX_MAKER_CHASE_ATTEMPTS    = _get_int("MAX_MAKER_CHASE_ATTEMPTS", 2)
MAX_MAKER_CHASE_PCT         = _get_float("MAX_MAKER_CHASE_PCT", 0.004)
MAX_MAKER_CHASE_ATR_MULT    = _get_float("MAX_MAKER_CHASE_ATR_MULT", 0.4)
MAKER_CHASE_INTERVAL_SEC    = _get_float("MAKER_CHASE_INTERVAL_SEC", 3.0)
MAX_ENTRY_SPREAD_PCT        = _get_float("MAX_ENTRY_SPREAD_PCT", 0.005)
RATE_LIMIT_RESERVE_THRESHOLD = _get_int("RATE_LIMIT_RESERVE_THRESHOLD", 8)
CLOSED_PNL_SYNC_DELAY_SEC   = _get_float("CLOSED_PNL_SYNC_DELAY_SEC", 4.0)


# =====================================================================
# 9. ДЕРИВАТИВНЫЙ ФИЛЬТР РЕЖИМА И REGIME SWITCHER (v0.3.0)
# =====================================================================
USE_FUNDING_ZSCORE_FILTER  = _get_bool("USE_FUNDING_ZSCORE_FILTER", True)
FUNDING_ZSCORE_WINDOW      = _get_int("FUNDING_ZSCORE_WINDOW", 30)
FUNDING_ZSCORE_LONG_MAX    = _get_float("FUNDING_ZSCORE_LONG_MAX", 2.0)
FUNDING_ZSCORE_SHORT_MIN   = _get_float("FUNDING_ZSCORE_SHORT_MIN", -2.0)
USE_OI_FILTER              = _get_bool("USE_OI_FILTER", True)
OI_LOOKBACK_PERIODS        = _get_int("OI_LOOKBACK_PERIODS", 4)
OI_DIVERGENCE_PCT          = _get_float("OI_DIVERGENCE_PCT", 0.03)
OI_PRICE_MOVE_PCT          = _get_float("OI_PRICE_MOVE_PCT", 0.015)
REGIME_FAST_EMA_PERIOD     = _get_int("REGIME_FAST_EMA_PERIOD", 50)
REGIME_USE_HYBRID_TREND    = _get_bool("REGIME_USE_HYBRID_TREND", True)
REGIME_VOL_WINDOW          = _get_int("REGIME_VOL_WINDOW", 20)
REGIME_EXTREME_VOL_PCT     = _get_float("REGIME_EXTREME_VOL_PCT", 1.80)
DERIVATIVES_CACHE_TTL_SEC  = _get_int("DERIVATIVES_CACHE_TTL_SEC", 900)

USE_REGIME_SWITCHER        = _get_bool("USE_REGIME_SWITCHER", False)
SWITCHER_ADX_TREND_MIN     = _get_float("SWITCHER_ADX_TREND_MIN", 22.0)
SWITCHER_ADX_RANGE_MAX     = _get_float("SWITCHER_ADX_RANGE_MAX", 18.0)
SWITCHER_ER_PERIOD         = _get_int("SWITCHER_ER_PERIOD", 14)
SWITCHER_ER_TREND_MIN      = _get_float("SWITCHER_ER_TREND_MIN", 0.35)
SWITCHER_ER_RANGE_MAX      = _get_float("SWITCHER_ER_RANGE_MAX", 0.25)
SWITCHER_BB_PCTL_WINDOW    = _get_int("SWITCHER_BB_PCTL_WINDOW", 100)
SWITCHER_MR_SL_MULT        = _get_float("SWITCHER_MR_SL_MULT", 1.8)
SWITCHER_MR_TP_MULT        = _get_float("SWITCHER_MR_TP_MULT", 2.5)


# =====================================================================
# 10. ДИНАМИЧЕСКИЙ ОТБОР КОРЗИНЫ, ПОРТФЕЛЬНЫЙ РИСК v3 И SHADOW (v0.3.0)
# =====================================================================
USE_DYNAMIC_UNIVERSE         = _get_bool("USE_DYNAMIC_UNIVERSE", False)
DYNAMIC_UNIVERSE_TOP_N       = _get_int("DYNAMIC_UNIVERSE_TOP_N", 6)
DYNAMIC_UNIVERSE_MIN_TURNOVER = _get_float("DYNAMIC_UNIVERSE_MIN_TURNOVER", 50_000_000.0)
DYNAMIC_UNIVERSE_MAX_SPREAD  = _get_float("DYNAMIC_UNIVERSE_MAX_SPREAD", 0.0025)
DYNAMIC_UNIVERSE_MIN_VOL     = _get_float("DYNAMIC_UNIVERSE_MIN_VOL", 0.015)
DYNAMIC_UNIVERSE_MAX_VOL     = _get_float("DYNAMIC_UNIVERSE_MAX_VOL", 0.18)
UNIVERSE_REFRESH_HOURS       = _get_float("UNIVERSE_REFRESH_HOURS", 24.0)
UNIVERSE_BLACKLIST           = [
    s.strip().upper()
    for s in os.getenv("UNIVERSE_BLACKLIST", "BTCUSDT,USDCUSDT,FDUSDUSDT,USDEUSDT").split(",")
    if s.strip()
]

MAX_PORTFOLIO_HEAT_PCT   = _get_float("MAX_PORTFOLIO_HEAT_PCT", 0.035)
USE_CORRELATION_SCALING  = _get_bool("USE_CORRELATION_SCALING", True)
CORR_LOOKBACK_BARS       = _get_int("CORR_LOOKBACK_BARS", 336)
CORR_THRESHOLD           = _get_float("CORR_THRESHOLD", 0.75)
CORR_RISK_SCALE          = _get_float("CORR_RISK_SCALE", 0.6)

USE_PEAK_DD_THROTTLE     = _get_bool("USE_PEAK_DD_THROTTLE", True)
DD_THROTTLE_TIER1_PCT    = _get_float("DD_THROTTLE_TIER1_PCT", 0.06)
DD_THROTTLE_TIER1_MULT   = _get_float("DD_THROTTLE_TIER1_MULT", 0.5)
DD_THROTTLE_TIER2_PCT    = _get_float("DD_THROTTLE_TIER2_PCT", 0.12)
DD_THROTTLE_RECOVER_PCT  = _get_float("DD_THROTTLE_RECOVER_PCT", 0.09)

USE_LOSS_STREAK_COOLDOWN   = _get_bool("USE_LOSS_STREAK_COOLDOWN", True)
LOSS_STREAK_SYMBOL_LIMIT   = _get_int("LOSS_STREAK_SYMBOL_LIMIT", 2)
LOSS_STREAK_PORTFOLIO_LIMIT = _get_int("LOSS_STREAK_PORTFOLIO_LIMIT", 3)
LOSS_STREAK_COOLDOWN_BARS  = _get_int("LOSS_STREAK_COOLDOWN_BARS", 6)

USE_TIME_STOP       = _get_bool("USE_TIME_STOP", True)
TIME_STOP_HOURS     = _get_float("TIME_STOP_HOURS", 30.0)
TIME_STOP_HARD_HOURS = _get_float("TIME_STOP_HARD_HOURS", 36.0)
TIME_STOP_MIN_R     = _get_float("TIME_STOP_MIN_R", 0.3)

SHADOW_MODE            = _get_bool("SHADOW_MODE", False)
SHADOW_STRATEGY        = os.getenv("SHADOW_STRATEGY", "").strip().upper()
SHADOW_INITIAL_BALANCE = _get_float("SHADOW_INITIAL_BALANCE", 1000.0)
