"""
Модуль конфигурации Bybit Futures торгового бота (v2.2)
Загружает чувствительные данные из .env и предоставляет типизированные настройки с fallback-значениями.
"""

import os
from pathlib import Path

# Попытка загрузить переменные из .env файла
try:
    from dotenv import load_dotenv
    # Ищем .env в текущей директории или директории проекта
    env_path = Path(__file__).resolve().parent / ".env"
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
TRADING_MODE = os.getenv("TRADING_MODE", "DEMO").strip().upper()

# Ключи для ДЕМО-счета (Bybit Demo Trading)
DEMO_API_KEY    = os.getenv("DEMO_API_KEY", "your_demo_key").strip()
DEMO_API_SECRET = os.getenv("DEMO_API_SECRET", "your_demo_secret").strip()

# Ключи для БОЕВОГО счета (Bybit Live Mainnet)
LIVE_API_KEY    = os.getenv("LIVE_API_KEY", "your_live_key").strip()
LIVE_API_SECRET = os.getenv("LIVE_API_SECRET", "your_live_secret").strip()

# Автоматический выбор рабочего ключа
if TRADING_MODE == "DEMO":
    API_KEY    = DEMO_API_KEY
    API_SECRET = DEMO_API_SECRET
    DEMO       = True
    TESTNET    = False
elif TRADING_MODE == "TESTNET":
    API_KEY    = DEMO_API_KEY
    API_SECRET = DEMO_API_SECRET
    DEMO       = False
    TESTNET    = True
else:  # LIVE
    API_KEY    = LIVE_API_KEY
    API_SECRET = LIVE_API_SECRET
    DEMO       = False
    TESTNET    = False


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
ST_PERIOD      = _get_int("ST_PERIOD", 10)
ST_MULTIPLIER  = _get_float("ST_MULTIPLIER", 3.0)
ST_CONFIRM_EMA = _get_int("ST_CONFIRM_EMA", 50)

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
BREAKEVEN_TRIGGER   = _get_float("BREAKEVEN_TRIGGER", 0.015)   # +1.5% хода цены в плюс
BREAKEVEN_OFFSET    = _get_float("BREAKEVEN_OFFSET", 0.002)    # +0.2% от цены входа (комиссии)


# =====================================================================
# 6. ТРЕЙЛИНГ-СТОП, ЧАСТИЧНЫЙ TP, ФИЛЬТР РЕЖИМА РЫНКА, ИСПОЛНЕНИЕ
# =====================================================================

# 6.1. Трейлинг-стоп (Trailing Stop)
USE_TRAILING_STOP        = _get_bool("USE_TRAILING_STOP", True)
USE_NATIVE_TRAILING_STOP = _get_bool("USE_NATIVE_TRAILING_STOP", True)  # Выставлять нативный trailingStop Bybit
TRAILING_STOP_TRIGGER    = _get_float("TRAILING_STOP_TRIGGER", 0.015)   # +1.5% хода активирует трейлинг
TRAILING_STOP_ATR_MULT   = _get_float("TRAILING_STOP_ATR_MULT", 2.2)    # дистанция трейлинга в ATR
TRAILING_STOP_FALLBACK   = _get_float("TRAILING_STOP_FALLBACK", 0.01)   # 1.0% если ATR недоступен

# 6.2. Частичная фиксация прибыли (Partial TP)
USE_PARTIAL_TP      = _get_bool("USE_PARTIAL_TP", True)
USE_EXCHANGE_TP1    = _get_bool("USE_EXCHANGE_TP1", True)          # Лимитный ордер reduceOnly на бирже
PARTIAL_TP_R_MULT   = _get_float("PARTIAL_TP_R_MULT", 1.5)         # TP1 на 1.5R (R = дистанция SL)
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
LEVERAGE              = _get_int("LEVERAGE", 5)
MAX_OPEN_POSITIONS    = _get_int("MAX_OPEN_POSITIONS", 3)
POSITION_SIZE_PCT     = _get_float("POSITION_SIZE_PCT", 0.15)
USE_RISK_BASED_SIZING = _get_bool("USE_RISK_BASED_SIZING", True)      # Расчет объема по риску от стопа
RISK_PER_TRADE_PCT    = _get_float("RISK_PER_TRADE_PCT", 0.015)       # 1.5% риска от депозита при стопе
FALLBACK_USDT         = _get_float("FALLBACK_USDT", 50.0)
MAX_DAILY_LOSS_PCT    = _get_float("MAX_DAILY_LOSS_PCT", 0.05)
