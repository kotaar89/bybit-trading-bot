"""
Алготрейдинг-бот для Bybit Futures (USDT Linear Perpetuals) v2.2
Поддерживает:
- Мультипарность (корзина валют)
- Модульные стратегии (SUPERTREND_PRO / MA_RSI / BB_RSI / SUPERTREND / DONCHIAN)
- Нативный и программный Трейлинг-стоп (Trailing Stop)
- Частичную фиксацию прибыли (Лимитный TP1 на бирже на 1.5R) + перевод в Б/У
- Приватный WebSocket User Data Stream (позиции, исполнения, ордера) с задержкой <50ms
- Умное исполнение Maker vs Taker (Limit Post-Only с таймаутом и фоллбэком на Market)
- Проверку минимального нотионала ордеров (>= 5 USDT)
- Атомарное сохранение состояния на диск (bot_state.json)
- Фильтр рыночного сентимента по бенчмарку BTC (Market Regime Filter)
- Фильтр флэта ADX (блокировка входов в консолидации)
- Автоматический расчет динамических ATR-стопов
- Встроенный легковесный веб-дашборд на FastAPI (порт 8000) под Termux
"""

import time
import threading
import logging
from logging.handlers import RotatingFileHandler
from collections import deque
from typing import Optional
from datetime import datetime, timezone
from decimal import Decimal, ROUND_DOWN, ROUND_HALF_UP
import json
from pathlib import Path
import pandas as pd
import numpy as np

from pybit.unified_trading import HTTP, WebSocket
from pybit import _helpers

import config
from config import (
    API_KEY, API_SECRET, TRADING_MODE, DEMO, TESTNET,
    SYMBOLS, SYMBOL, INTERVAL, HIGHER_INTERVAL,
    ST_PERIOD, ST_MULTIPLIER, ST_CONFIRM_EMA,
    FAST_MA, SLOW_MA, RSI_PERIOD,
    RSI_LONG_MIN, RSI_LONG_MAX, RSI_SHORT_MIN, RSI_SHORT_MAX,
    VOLUME_FILTER, USE_MTF_FILTER, HTF_EMA_PERIOD,
    USE_ATR_STOPS, ATR_PERIOD, ATR_SL_MULT, ATR_TP_MULT,
    STOP_LOSS_PCT, TAKE_PROFIT_PCT,
    USE_BREAKEVEN, BREAKEVEN_TRIGGER, BREAKEVEN_R_MULT, BREAKEVEN_OFFSET,
    USE_TRAILING_STOP, TRAILING_STOP_TRIGGER, TRAILING_STOP_R_MULT, TRAILING_STOP_ATR_MULT, TRAILING_STOP_FALLBACK,
    USE_NATIVE_TRAILING_STOP,
    USE_PARTIAL_TP, PARTIAL_TP_R_MULT, PARTIAL_TP_RATIO, PARTIAL_TP_MOVE_BE,
    USE_EXCHANGE_TP1,
    USE_MARKET_REGIME_FILTER, BENCHMARK_SYMBOL, BENCHMARK_INTERVAL,
    BENCHMARK_EMA_PERIOD, BENCHMARK_RSI_THRESHOLD, BENCHMARK_DUMP_PCT, BENCHMARK_DUMP_CANDLES,
    LEVERAGE, MAX_OPEN_POSITIONS, POSITION_SIZE_PCT,
    FALLBACK_USDT, MAX_DAILY_LOSS_PCT,
    DASHBOARD_HOST, DASHBOARD_PORT, ACTIVE_STRATEGY,
    SIGNAL_LOOKBACK_CANDLES, MAX_SLIPPAGE_PCT,
    BENCHMARK_BULL_RSI_THRESHOLD, BENCHMARK_PUMP_PCT,
    USE_RISK_BASED_SIZING, RISK_PER_TRADE_PCT,
    USE_ADX_FILTER, ADX_PERIOD, ADX_THRESHOLD,
    USE_SMART_EXECUTION, SMART_EXECUTION_TIMEOUT, MIN_ORDER_NOTIONAL
)
from strategies import get_strategy
from risk_manager import PositionRiskState, MarketRegimeFilter, PortfolioRiskManager
from storage import BotDatabase
from universe import DynamicUniverseSelector
from shadow import ShadowEngine
from web.state import (
    set_bot_instance, is_bot_paused, set_bot_paused,
    write_control_state, read_control_state, write_live_status,
    consume_shadow_control_actions,
)
from backtest import run_backtest


# =====================================================================
# Настройка логирования:
# При импорте (в т.ч. в автотестах) файловые обработчики НЕ создаются.
# setup_logging() вызывается явно при старте бота.
# =====================================================================
log = logging.getLogger("bot")
log.setLevel(logging.INFO)
log.propagate = False
if not log.handlers:
    log.addHandler(logging.NullHandler())

trade_log = logging.getLogger("trade")
trade_log.setLevel(logging.INFO)
trade_log.propagate = False
if not trade_log.handlers:
    trade_log.addHandler(logging.NullHandler())


def setup_logging(
    bot_log_path: Optional[Path] = None,
    trade_log_path: Optional[Path] = None,
    console: bool = True
):
    """Инициализирует файловые RotatingFileHandler и консольный вывод для live-бота."""
    formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
    bot_path = Path(bot_log_path or config.BOT_LOG_FILE)
    trade_path = Path(trade_log_path or config.TRADE_LOG_FILE)

    def _configure(logger_obj: logging.Logger, file_path: Path):
        has_file = any(isinstance(h, RotatingFileHandler) for h in logger_obj.handlers)
        if has_file:
            return
        logger_obj.handlers = [h for h in logger_obj.handlers if not isinstance(h, logging.NullHandler)]
        fh = RotatingFileHandler(str(file_path), maxBytes=5*1024*1024, backupCount=3, encoding="utf-8")
        fh.setFormatter(formatter)
        logger_obj.addHandler(fh)
        if console and not any(isinstance(h, logging.StreamHandler) and not isinstance(h, logging.FileHandler) for h in logger_obj.handlers):
            sh = logging.StreamHandler()
            sh.setFormatter(formatter)
            logger_obj.addHandler(sh)

    _configure(log, bot_path)
    _configure(trade_log, trade_path)


def sync_exchange_time(testnet: bool, demo: bool):
    """
    Синхронизирует локальное время с сервером Bybit.
    Предотвращает ошибку Bybit ErrCode 10002 (time drift / recv_window).
    """
    try:
        unauth = HTTP(testnet=testnet, demo=demo)
        server_t = int(unauth.get_server_time()["time"])
        local_t = int(time.time() * 1000)
        offset = server_t - local_t
        _helpers.generate_timestamp = lambda: int(time.time() * 1000) + offset
        log.info(f"Синхронизация времени с Bybit: смещение {offset:+d} мс")
    except Exception as e:
        log.warning(f"Синхронизация времени не удалась: {e}")


class RateLimitController:
    """Адаптивный упреждающий контроль Rate Limits Bybit v5 по заголовкам X-Bapi-Limit-*."""

    def __init__(self, reserve_threshold: int = 8):
        self._lock = threading.Lock()
        self.reserve_threshold = max(1, int(reserve_threshold))
        self.remaining: Optional[int] = None
        self.limit: Optional[int] = None
        self.reset_ts_ms: int = 0
        self.last_updated: float = 0.0

    def update_from_headers(self, headers):
        if not headers:
            return
        try:
            lower_map = {str(k).lower(): str(v) for k, v in dict(headers).items()}
            rem_raw = lower_map.get("x-bapi-limit-status")
            lim_raw = lower_map.get("x-bapi-limit")
            rst_raw = lower_map.get("x-bapi-limit-reset-timestamp")
            with self._lock:
                if rem_raw is not None:
                    self.remaining = int(float(rem_raw))
                if lim_raw is not None:
                    self.limit = int(float(lim_raw))
                if rst_raw is not None:
                    self.reset_ts_ms = int(float(rst_raw))
                self.last_updated = time.time()
        except Exception:
            pass

    def wait_if_needed(self):
        delay = 0.0
        with self._lock:
            if self.remaining is None or (time.time() - self.last_updated) > 5.0:
                return
            thresh = self.reserve_threshold
            if self.limit and self.limit > 0:
                thresh = max(self.reserve_threshold, min(15, int(self.limit * 0.08)))
            now_ms = int(time.time() * 1000)
            if self.remaining <= thresh:
                if self.reset_ts_ms > now_ms:
                    delay = min(2.0, (self.reset_ts_ms - now_ms) / 1000.0 + 0.05)
                else:
                    delay = 0.35
            elif self.remaining <= thresh * 2:
                delay = 0.08
        if delay > 0:
            time.sleep(delay)

    def snapshot(self) -> dict:
        with self._lock:
            return {
                "remaining": self.remaining,
                "limit": self.limit,
                "reset_ts_ms": self.reset_ts_ms,
            }


class BybitMultiFuturesBot:
    def __init__(self):
        setup_logging()
        eff_strategy = (
            "REGIME_SWITCHER"
            if getattr(config, "USE_REGIME_SWITCHER", False)
            and config.ACTIVE_STRATEGY.upper() not in ("REGIME_SWITCHER", "ADAPTIVE")
            else config.ACTIVE_STRATEGY
        )
        if getattr(config, "SHADOW_MODE", False) and str(getattr(config, "SHADOW_STRATEGY", "") or "").strip():
            eff_strategy = str(config.SHADOW_STRATEGY).strip().upper()
        log.info(f"=== Инициализация Multi-Pair Bot v0.3.0 | Режим: {config.TRADING_MODE} | Shadow={config.SHADOW_MODE} ===")
        log.info(
            f"Торгуемые пары: {', '.join(SYMBOLS)} | TF: {INTERVAL}m (MTF: {HIGHER_INTERVAL}m) | Плечо: x{LEVERAGE}"
        )
        log.info(
            f"Стратегия: {eff_strategy} | Трейлинг: {USE_TRAILING_STOP} (Native: {USE_NATIVE_TRAILING_STOP}) | "
            f"Частичный TP: {USE_PARTIAL_TP} (Limit: {USE_EXCHANGE_TP1}) | Smart Exec: {USE_SMART_EXECUTION} (Chase: {config.USE_MAKER_CHASE})"
        )

        # Синхронизация системного времени
        sync_exchange_time(config.TESTNET, config.DEMO)

        self.rate_limiter = RateLimitController(
            reserve_threshold=getattr(config, "RATE_LIMIT_RESERVE_THRESHOLD", 8)
        )
        self.client = HTTP(
            testnet=config.TESTNET,
            api_key=config.API_KEY,
            api_secret=config.API_SECRET,
            demo=config.DEMO,
            recv_window=10000,
            timeout=25,
            max_retries=5,
            retry_delay=2
        )
        self._attach_rate_limit_hook(self.client)

        self.db = BotDatabase()
        self.db.init_db()

        # Модульная стратегия и деривативный фильтр рыночного режима
        strat_use_mtf = USE_MTF_FILTER if eff_strategy.upper() != "BB_RSI" else False
        self.strategy = get_strategy(
            eff_strategy,
            st_period=ST_PERIOD,
            st_multiplier=ST_MULTIPLIER,
            confirm_ema_period=ST_CONFIRM_EMA,
            confirm_lookback=getattr(config, "ST_CONFIRM_LOOKBACK", 1),
            fast_ma=FAST_MA,
            slow_ma=SLOW_MA,
            rsi_period=RSI_PERIOD,
            rsi_long_min=RSI_LONG_MIN,
            rsi_long_max=RSI_LONG_MAX,
            rsi_short_min=RSI_SHORT_MIN,
            rsi_short_max=RSI_SHORT_MAX,
            volume_filter=VOLUME_FILTER,
            use_mtf_filter=strat_use_mtf,
            htf_ema_period=HTF_EMA_PERIOD,
            atr_period=ATR_PERIOD,
            cross_lookback=SIGNAL_LOOKBACK_CANDLES,
            use_adx_filter=USE_ADX_FILTER,
            adx_period=ADX_PERIOD,
            adx_threshold=ADX_THRESHOLD,
            adx_trend_min=getattr(config, "SWITCHER_ADX_TREND_MIN", 22.0),
            adx_range_max=getattr(config, "SWITCHER_ADX_RANGE_MAX", 18.0),
            er_period=getattr(config, "SWITCHER_ER_PERIOD", 14),
            er_trend_min=getattr(config, "SWITCHER_ER_TREND_MIN", 0.35),
            er_range_max=getattr(config, "SWITCHER_ER_RANGE_MAX", 0.25),
            bb_pctl_window=getattr(config, "SWITCHER_BB_PCTL_WINDOW", 100),
            mr_sl_mult=getattr(config, "SWITCHER_MR_SL_MULT", 1.8),
            mr_tp_mult=getattr(config, "SWITCHER_MR_TP_MULT", 2.5),
        )

        self.regime_filter = MarketRegimeFilter(
            benchmark_symbol=BENCHMARK_SYMBOL,
            ema_period=BENCHMARK_EMA_PERIOD,
            rsi_threshold=BENCHMARK_RSI_THRESHOLD,
            bull_rsi_threshold=BENCHMARK_BULL_RSI_THRESHOLD,
            dump_pct_threshold=BENCHMARK_DUMP_PCT,
            pump_pct_threshold=BENCHMARK_PUMP_PCT,
            dump_lookback_candles=BENCHMARK_DUMP_CANDLES,
            fast_ema_period=getattr(config, "REGIME_FAST_EMA_PERIOD", 50),
            use_hybrid_trend=getattr(config, "REGIME_USE_HYBRID_TREND", True),
            funding_zscore_window=getattr(config, "FUNDING_ZSCORE_WINDOW", 30),
            funding_z_long_max=getattr(config, "FUNDING_ZSCORE_LONG_MAX", 2.0),
            funding_z_short_min=getattr(config, "FUNDING_ZSCORE_SHORT_MIN", -2.0),
            oi_lookback=getattr(config, "OI_LOOKBACK_PERIODS", 4),
            oi_div_pct=getattr(config, "OI_DIVERGENCE_PCT", 0.03),
            oi_price_move_pct=getattr(config, "OI_PRICE_MOVE_PCT", 0.015),
            vol_window=getattr(config, "REGIME_VOL_WINDOW", 20),
            extreme_vol_pct=getattr(config, "REGIME_EXTREME_VOL_PCT", 2.50),
        )

        self.portfolio_risk = PortfolioRiskManager(
            max_heat_pct=getattr(config, "MAX_PORTFOLIO_HEAT_PCT", 0.035),
            use_correlation_scaling=getattr(config, "USE_CORRELATION_SCALING", True),
            corr_lookback_bars=getattr(config, "CORR_LOOKBACK_BARS", 336),
            corr_threshold=getattr(config, "CORR_THRESHOLD", 0.75),
            corr_risk_scale=getattr(config, "CORR_RISK_SCALE", 0.6),
            use_peak_dd_throttle=getattr(config, "USE_PEAK_DD_THROTTLE", True),
            dd_tier1_pct=getattr(config, "DD_THROTTLE_TIER1_PCT", 0.06),
            dd_tier1_mult=getattr(config, "DD_THROTTLE_TIER1_MULT", 0.5),
            dd_tier2_pct=getattr(config, "DD_THROTTLE_TIER2_PCT", 0.12),
            dd_recover_pct=getattr(config, "DD_THROTTLE_RECOVER_PCT", 0.09),
            use_loss_cooldown=getattr(config, "USE_LOSS_STREAK_COOLDOWN", True),
            symbol_loss_limit=getattr(config, "LOSS_STREAK_SYMBOL_LIMIT", 2),
            portfolio_loss_limit=getattr(config, "LOSS_STREAK_PORTFOLIO_LIMIT", 3),
            cooldown_seconds=getattr(config, "LOSS_STREAK_COOLDOWN_BARS", 6) * 3600.0,
        )

        self.universe_selector = DynamicUniverseSelector(
            top_n=getattr(config, "DYNAMIC_UNIVERSE_TOP_N", 6),
            min_turnover_24h=getattr(config, "DYNAMIC_UNIVERSE_MIN_TURNOVER", 50_000_000.0),
            max_spread_pct=getattr(config, "DYNAMIC_UNIVERSE_MAX_SPREAD", 0.0025),
            min_vol_pct=getattr(config, "DYNAMIC_UNIVERSE_MIN_VOL", 0.015),
            max_vol_pct=getattr(config, "DYNAMIC_UNIVERSE_MAX_VOL", 0.18),
            blacklist=getattr(config, "UNIVERSE_BLACKLIST", None),
        )
        self.active_symbols = list(SYMBOLS)

        self.lock = threading.RLock()
        self._bot_start_ts = time.time()
        self.last_time_sync = time.time()
        self.state_file = config.STATE_FILE

        self.instruments = {}
        self.positions = {}
        self.hedge_positions = {}  # {(symbol, positionIdx): dict}
        self.risk_states = {}  # {symbol: PositionRiskState}
        self.last_handled_candles = {}
        self.breakeven_applied = {}
        self.position_modes = {}  # {symbol: "MergedSingle" | "BothSides"}

        self.ws = None           # Public ticker & kline WebSocket
        self.ws_private = None   # Private User Data WebSocket (position, order, execution)
        self._public_ws_gen = 0
        self._private_ws_gen = 0
        self.ws_prices = {}
        self.ws_last_update = {}
        self.ws_orders = {}      # {orderId: {"status": str, "cum_qty": float, "avg_price": float, "ts": float}}
        self.kline_cache = {}    # {(symbol, interval): pd.DataFrame}
        self.kline_cache_ts = {} # {(symbol, interval): float}
        self._last_confirmed_kline_ts = {}  # {(symbol, interval): pd.Timestamp}
        self._candle_event = threading.Event()
        self._confirmed_candles = deque(maxlen=200)
        self._pending_pnl_sync = []  # [(symbol, order_id, margin_used, due_ts)]
        self._derivatives_cache = {}
        self._last_instrument_refresh = time.time()
        self._last_universe_refresh = 0.0
        self._last_regime_snapshot = {}

        self.last_exec_prices = {}
        self.last_exec_info = {}
        self._seen_exec_ids = deque(maxlen=1000)
        self.last_risk_api_call = {}
        self.last_ws_public_msg = time.time()
        self.last_ws_private_msg = time.time()
        self.last_ws_reconnect_attempt = time.time()

        # Риск-менеджмент: учет баланса и дневных убытков
        self.last_known_equity = 0.0
        self.last_known_balance = 0.0
        self.last_equity_ok = False
        self.last_balance_ok = False
        self.start_day = datetime.now().date()
        self.start_day_balance = 0.0
        self.circuit_breaker_active = False
        self.is_paused = False
        self._last_control_ts = time.time()
        self._last_live_status_write = 0.0

        # Shadow / Paper режим (полный paper или параллельный A/B тест стратегии)
        self.shadow_engine: Optional[ShadowEngine] = None
        if getattr(config, "SHADOW_MODE", False) or getattr(config, "SHADOW_STRATEGY", ""):
            sh_strat = config.SHADOW_STRATEGY or eff_strategy
            self.shadow_engine = ShadowEngine(
                db=self.db,
                initial_equity=getattr(config, "SHADOW_INITIAL_BALANCE", 1000.0),
                strategy_name=sh_strat,
                leverage=LEVERAGE,
            )

        # 0. Опциональный динамический отбор ликвидной корзины при старте
        if getattr(config, "USE_DYNAMIC_UNIVERSE", False):
            self.active_symbols = self.universe_selector.select_symbols(
                self.client,
                fallback_symbols=list(SYMBOLS),
                api_call_fn=self._safe_api_call,
            )
            self._last_universe_refresh = time.time()

        is_shadow_only = bool(getattr(config, "SHADOW_MODE", False)) and self.shadow_engine is not None

        # 1. Загрузка параметров инструментов, настройка плеча и режима позиций (Hedge/One-Way)
        for sym in self.active_symbols:
            self._load_instrument_info(sym)
            if not is_shadow_only:
                self._init_leverage(sym)
                self._detect_position_mode(sym)
            else:
                self.position_modes[sym] = "MergedSingle"
            self.positions[sym] = None
            self.risk_states[sym] = None
            self.last_handled_candles[sym] = None
            self.breakeven_applied[sym] = False

        # 2. Первичная синхронизация открытых позиций
        if is_shadow_only:
            self._sync_from_shadow_engine()
        else:
            self.sync_positions()

        # 3. Фиксация/восстановление дневного и пикового баланса между рестартами
        saved_meta = self.load_runtime_meta()
        today_str = str(self.start_day)
        saved_day = str(saved_meta.get("start_day") or "")
        saved_day_bal = float(saved_meta.get("start_day_balance") or 0.0)

        if is_shadow_only:
            init_bal = float(self.shadow_engine.equity)
            self.last_known_equity = init_bal
            self.last_equity_ok = True
            self.start_day_balance = saved_day_bal if (saved_day == today_str and saved_day_bal > 0) else init_bal
            self.circuit_breaker_active = bool(saved_meta.get("circuit_breaker_active", False)) if saved_day == today_str else False
            self.shadow_engine.portfolio_risk.update_equity(init_bal, valid=True)
            self.db.record_equity_snapshot(
                equity=init_bal, balance=init_bal, unrealized_pnl=0.0, source="start", is_shadow=True,
            )
        else:
            init_bal = self.get_wallet_equity()
            if saved_day == today_str and saved_day_bal > 0:
                self.start_day_balance = saved_day_bal
                self.circuit_breaker_active = bool(saved_meta.get("circuit_breaker_active", False))
                log.info(
                    f"Восстановлен внутридневной базовый баланс: {self.start_day_balance:.2f} USDT "
                    f"(CB={self.circuit_breaker_active})"
                )
            else:
                self.start_day_balance = init_bal if getattr(self, "last_equity_ok", False) and init_bal > 0 else FALLBACK_USDT

            if isinstance(saved_meta.get("portfolio_risk"), dict):
                self.portfolio_risk = PortfolioRiskManager.from_dict(
                    saved_meta["portfolio_risk"],
                    max_heat_pct=self.portfolio_risk.max_heat_pct,
                    use_correlation_scaling=self.portfolio_risk.use_correlation_scaling,
                    corr_lookback_bars=self.portfolio_risk.corr_lookback_bars,
                    corr_threshold=self.portfolio_risk.corr_threshold,
                    corr_risk_scale=self.portfolio_risk.corr_risk_scale,
                    use_peak_dd_throttle=self.portfolio_risk.use_peak_dd_throttle,
                    dd_tier1_pct=self.portfolio_risk.dd_tier1_pct,
                    dd_tier1_mult=self.portfolio_risk.dd_tier1_mult,
                    dd_tier2_pct=self.portfolio_risk.dd_tier2_pct,
                    dd_recover_pct=self.portfolio_risk.dd_recover_pct,
                    use_loss_cooldown=self.portfolio_risk.use_loss_cooldown,
                    symbol_loss_limit=self.portfolio_risk.symbol_loss_limit,
                    portfolio_loss_limit=self.portfolio_risk.portfolio_loss_limit,
                    cooldown_seconds=self.portfolio_risk.cooldown_seconds,
                )
            if getattr(self, "last_equity_ok", False) and init_bal > 0:
                self.portfolio_risk.update_equity(init_bal, valid=True)
                self.db.record_equity_snapshot(
                    equity=init_bal,
                    balance=init_bal,
                    unrealized_pnl=0.0,
                    source="start",
                    is_shadow=False,
                )
            elif float(saved_meta.get("peak_equity") or 0.0) > self.portfolio_risk.peak_equity:
                self.portfolio_risk.peak_equity = float(saved_meta["peak_equity"])

        eff_pr = self.shadow_engine.portfolio_risk if is_shadow_only else self.portfolio_risk
        trade_log.info(f"Стартовый баланс: {self.start_day_balance:.2f} USDT | Peak Equity: {eff_pr.peak_equity:.2f} USDT")
        self.save_state()

        # 4. Регистрация глобального экземпляра для веб-дашборда и файла управления
        self._last_control_ts = write_control_state(is_paused=False, trading_mode=config.TRADING_MODE)
        set_bot_instance(self)

        # 5. Запуск веб-дашборда в фоновом демоническом потоке
        self._start_dashboard_thread()

        # 6. Запуск фонового WebSocket клиента (публичный + приватный)
        self._init_websocket()

    def _attach_rate_limit_hook(self, client):
        """Подключает перехватчик HTTP-заголовков X-Bapi-Limit-* без изменения формата ответов pybit."""
        try:
            sess = getattr(client, "client", None)
            hooks = getattr(sess, "hooks", None)
            if isinstance(hooks, dict) and "response" in hooks:
                def _on_resp(resp, *args, **kwargs):
                    rl = getattr(self, "rate_limiter", None)
                    if rl is not None and hasattr(resp, "headers"):
                        rl.update_from_headers(resp.headers)
                    return resp
                hooks["response"].append(_on_resp)
        except Exception:
            pass

    def _build_runtime_meta(self) -> dict:
        is_shadow_only = bool(getattr(config, "SHADOW_MODE", False)) and getattr(self, "shadow_engine", None) is not None
        pr = (
            getattr(self.shadow_engine, "portfolio_risk", None)
            if is_shadow_only
            else getattr(self, "portfolio_risk", None)
        )
        return {
            "start_day": str(getattr(self, "start_day", datetime.now().date())),
            "start_day_balance": float(getattr(self, "start_day_balance", 0.0) or 0.0),
            "peak_equity": float(pr.peak_equity if pr is not None else 0.0),
            "circuit_breaker_active": bool(getattr(self, "circuit_breaker_active", False)),
            "portfolio_risk": pr.to_dict() if pr is not None else {},
            "active_symbols": list(getattr(self, "active_symbols", SYMBOLS)),
        }

    def load_runtime_meta(self) -> dict:
        """Загружает сохраненный дневной/пиковый баланс и риск-счетчики из SQLite или bot_state.json."""
        db = getattr(self, "db", None)
        if db is not None:
            try:
                bundle = db.get_all_runtime_state()
                if bundle and ("start_day" in bundle or "peak_equity" in bundle or "portfolio_risk" in bundle):
                    return bundle
            except Exception:
                pass
        for candidate in (self.state_file, self.state_file.with_suffix(".tmp")):
            if candidate.exists():
                try:
                    with open(candidate, "r", encoding="utf-8") as f:
                        raw = json.load(f)
                    if isinstance(raw, dict) and isinstance(raw.get("__meta__"), dict):
                        return raw["__meta__"]
                except Exception:
                    pass
        return {}

    def save_state(self):
        """
        Сохраняет активные риск-состояния и дневной/пиковый баланс на диск АТОМАРНО
        (bot_state.json + SQLite runtime_state).
        """
        try:
            with self.lock:
                data = {}
                for sym, st in self.risk_states.items():
                    if st is not None:
                        data[sym] = st.to_dict()
                meta = self._build_runtime_meta()
                if meta.get("start_day_balance", 0.0) > 0 or meta.get("peak_equity", 0.0) > 0 or meta.get("circuit_breaker_active"):
                    data["__meta__"] = meta
                tmp_file = self.state_file.with_suffix(".tmp")
                with open(tmp_file, "w", encoding="utf-8") as f:
                    json.dump(data, f, indent=2)
                tmp_file.replace(self.state_file)
                db = getattr(self, "db", None)
                if db is not None:
                    db.save_runtime_bundle(meta)
        except Exception as e:
            log.warning(f"Ошибка атомарного сохранения bot_state.json: {e}")

    def load_state(self) -> dict:
        """Загружает сохраненные риск-состояния с диска с защитой от сбоев."""
        target_file = self.state_file
        if not target_file.exists():
            tmp = self.state_file.with_suffix(".tmp")
            if tmp.exists():
                target_file = tmp
            else:
                return {}
        try:
            with open(target_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            return {
                sym: PositionRiskState.from_dict(d)
                for sym, d in data.items()
                if sym != "__meta__" and isinstance(d, dict)
            }
        except Exception as e:
            log.warning(f"Ошибка загрузки {target_file}: {e}")
            tmp = self.state_file.with_suffix(".tmp")
            if target_file != tmp and tmp.exists():
                try:
                    with open(tmp, "r", encoding="utf-8") as f:
                        data = json.load(f)
                    return {
                        sym: PositionRiskState.from_dict(d)
                        for sym, d in data.items()
                        if sym != "__meta__" and isinstance(d, dict)
                    }
                except Exception:
                    pass
            return {}

    def _start_dashboard_thread(self):
        """Запускает веб-дашборд в отдельном потоке."""
        def run_srv():
            try:
                import uvicorn
                from web.app import app
                config_srv = uvicorn.Config(
                    app=app,
                    host=DASHBOARD_HOST,
                    port=DASHBOARD_PORT,
                    log_level="warning"
                )
                srv = uvicorn.Server(config_srv)
                srv.run()
            except Exception as e:
                log.warning(f"Ошибка запуска веб-сервера: {e}")

        t = threading.Thread(target=run_srv, daemon=True)
        t.start()
        log.info(f"Веб-дашборд запущен: http://{DASHBOARD_HOST}:{DASHBOARD_PORT}")

    def _detect_position_mode(self, symbol: str) -> str:
        """
        Определяет режим позиций для символа:
        - One-Way Mode (MergedSingle): positionIdx = 0
        - Hedge Mode (BothSides): positionIdx = 1 (Buy/Long), 2 (Sell/Short)
        """
        if not hasattr(self, "position_modes"):
            self.position_modes = {}
        try:
            resp = self._safe_api_call(self.client.get_positions, category="linear", symbol=symbol)
            plist = resp.get("result", {}).get("list", [])
            for p in plist:
                idx = int(p.get("positionIdx", 0))
                if idx in (1, 2):
                    self.position_modes[symbol] = "BothSides"
                    log.info(f"[{symbol}] Обнаружен режим позиций: HEDGE MODE (positionIdx: Long=1, Short=2)")
                    return "BothSides"
            self.position_modes[symbol] = "MergedSingle"
            log.info(f"[{symbol}] Обнаружен режим позиций: ONE-WAY MODE (positionIdx=0)")
            return "MergedSingle"
        except Exception as e:
            log.warning(f"[{symbol}] Ошибка определения режима позиций: {e}. Используем One-Way (0).")
            self.position_modes[symbol] = "MergedSingle"
            return "MergedSingle"

    def get_position_idx(self, symbol: str, side: str) -> int:
        """
        Возвращает правильный positionIdx согласно режиму аккаунта/символа:
        - One-Way Mode: всегда 0
        - Hedge Mode: 1 для Buy/Long, 2 для Sell/Short
        """
        mode = getattr(self, "position_modes", {}).get(symbol, "MergedSingle")
        if mode == "BothSides":
            return 1 if side.lower() in ("buy", "long") else 2
        return 0

    def _safe_api_call(self, fn, *args, max_attempts: int = 3, **kwargs):
        """Безопасный вызов API через _api_call (совместимо с mock-объектами и object.__new__)."""
        api_fn = getattr(self, "_api_call", None)
        if callable(api_fn):
            return api_fn(fn, *args, max_attempts=max_attempts, **kwargs)
        return fn(*args, **kwargs)

    def _sync_from_shadow_engine(self, symbol: Optional[str] = None):
        """Синхронизирует self.positions / self.risk_states из ShadowEngine в режиме SHADOW_MODE."""
        sh_eng = getattr(self, "shadow_engine", None)
        if sh_eng is None:
            return
        with self.lock:
            syms = [symbol.upper()] if symbol else list(
                dict.fromkeys(
                    list(getattr(self, "active_symbols", SYMBOLS) or SYMBOLS)
                    + list(self.positions.keys())
                    + list(sh_eng.positions.keys())
                )
            )
            for s in syms:
                sh_pos = sh_eng.positions.get(s)
                sh_risk = sh_eng.risk_states.get(s)
                if sh_pos and sh_risk:
                    self.positions[s] = dict(sh_pos)
                    self.risk_states[s] = sh_risk
                    self.breakeven_applied[s] = bool(sh_risk.breakeven_applied)
                else:
                    if s in self.positions:
                        self.positions[s] = None
                    if hasattr(self, "risk_states"):
                        self.risk_states[s] = None
                    if hasattr(self, "breakeven_applied"):
                        self.breakeven_applied[s] = False

    def _api_call(self, fn, *args, max_attempts: int = 3, **kwargs):
        """Обертка для критических REST-вызовов Bybit с упреждающим Rate Limit и backoff (10006)."""
        rl = getattr(self, "rate_limiter", None)
        last_exc = None
        for attempt in range(max_attempts):
            if rl is not None:
                rl.wait_if_needed()
            try:
                resp = fn(*args, **kwargs)
                if isinstance(resp, dict) and resp.get("retCode") == 10006:
                    if attempt < max_attempts - 1:
                        time.sleep(0.5 * (2 ** attempt))
                        continue
                return resp
            except Exception as e:
                last_exc = e
                msg = str(e).lower()
                if ("10006" in msg or "rate limit" in msg or "too many visits" in msg) and attempt < max_attempts - 1:
                    time.sleep(0.5 * (2 ** attempt))
                    continue
                raise
        if last_exc:
            raise last_exc

    @staticmethod
    def _is_stop_not_modified(resp_or_exc) -> bool:
        """Распознает ответ/исключение Bybit 34040 ('not modified'), когда SL/TP уже установлены такими же."""
        if isinstance(resp_or_exc, dict):
            if resp_or_exc.get("retCode") == 34040:
                return True
            msg = str(resp_or_exc.get("retMsg") or "").lower()
            return "not modified" in msg or "34040" in msg
        msg = str(resp_or_exc or "").lower()
        return "34040" in msg or "not modified" in msg

    @staticmethod
    def _safe_close_ws(ws_obj):
        """
        Гарантированно отключает старый WebSocket-менеджер pybit, предотвращая его
        самопроизвольное переподключение и дублирование коллбэков после .exit().
        """
        if ws_obj is None:
            return
        try:
            if hasattr(ws_obj, "handle_error"):
                ws_obj.handle_error = False
            if hasattr(ws_obj, "retries"):
                ws_obj.retries = 1
            if hasattr(ws_obj, "exited"):
                ws_obj.exited = True
            if hasattr(ws_obj, "callback"):
                ws_obj.callback = lambda *a, **kw: None
            if hasattr(ws_obj, "callback_directory") and isinstance(ws_obj.callback_directory, dict):
                ws_obj.callback_directory.clear()
            if hasattr(ws_obj, "subscriptions") and isinstance(ws_obj.subscriptions, dict):
                ws_obj.subscriptions.clear()
            if hasattr(ws_obj, "_handle_incoming_message"):
                ws_obj._handle_incoming_message = lambda *a, **kw: None
            if hasattr(ws_obj, "_on_error"):
                ws_obj._on_error = lambda *a, **kw: None
            if hasattr(ws_obj, "_on_pong"):
                ws_obj._on_pong = lambda *a, **kw: None
            if hasattr(ws_obj, "_is_custom_pong"):
                ws_obj._is_custom_pong = lambda *a, **kw: False
            inner_ws = getattr(ws_obj, "ws", None)
            if inner_ws is not None:
                if hasattr(inner_ws, "keep_running"):
                    inner_ws.keep_running = False
                for attr in ("on_message", "on_error", "on_close", "on_pong"):
                    if hasattr(inner_ws, attr):
                        setattr(inner_ws, attr, None)
        except Exception:
            pass
        try:
            ws_obj.exit()
        except Exception:
            pass

    @staticmethod
    def _normalize_kline_interval(tf) -> tuple[str, int | str]:
        """Нормализует интервал свечей для Bybit WS/REST ('60' -> ('60', 60), 'd' -> ('D', 'D'))."""
        raw = str(tf or "").strip()
        if not raw:
            return "60", 60
        if raw.isdigit():
            norm = str(int(raw))
            return norm, int(norm)
        up = raw.upper()
        return up, up

    def _get_watch_symbols(self) -> list[str]:
        """Возвращает актуальный список отслеживаемых пар (корзина + открытые позиции) без дублей."""
        base_raw = getattr(self, "active_symbols", None) or getattr(config, "SYMBOLS", SYMBOLS) or SYMBOLS
        open_syms = [s for s, p in getattr(self, "positions", {}).items() if p]
        sh_eng = getattr(self, "shadow_engine", None)
        if sh_eng is not None and isinstance(getattr(sh_eng, "positions", None), dict):
            open_syms.extend(s for s, p in sh_eng.positions.items() if p)
        cleaned: list[str] = []
        seen: set[str] = set()
        for s in list(base_raw) + list(open_syms):
            su = str(s or "").strip().upper()
            if su and su not in seen:
                seen.add(su)
                cleaned.append(su)
        return cleaned

    def _init_public_ws(self):
        """Подключает публичный WebSocket тикеров и event-driven свечей (kline_stream)."""
        with self.lock:
            old_ws = getattr(self, "ws", None)
            self.ws = None
            self._public_ws_gen = getattr(self, "_public_ws_gen", 0) + 1
            gen = self._public_ws_gen
            if hasattr(self, "kline_cache"):
                self.kline_cache.clear()
            if hasattr(self, "kline_cache_ts"):
                self.kline_cache_ts.clear()
            if hasattr(self, "_last_confirmed_kline_ts"):
                self._last_confirmed_kline_ts.clear()
            if hasattr(self, "_confirmed_candles") and hasattr(self._confirmed_candles, "clear"):
                self._confirmed_candles.clear()
            if hasattr(self, "_candle_event") and self._candle_event is not None:
                self._candle_event.clear()

        self._safe_close_ws(old_ws)

        watch_syms = self._get_watch_symbols()
        bm_sym = str(getattr(config, "BENCHMARK_SYMBOL", BENCHMARK_SYMBOL) or "").strip().upper()
        kline_syms = list(dict.fromkeys(watch_syms + ([bm_sym] if bm_sym else [])))

        new_ws = None
        try:
            new_ws = WebSocket(
                testnet=config.TESTNET,
                demo=False,
                channel_type="linear",
                restart_on_error=True
            )
            if hasattr(new_ws, "_on_pong") and callable(new_ws._on_pong):
                orig_pong = new_ws._on_pong
                def _wrapped_pub_pong(orig=orig_pong, g=gen):
                    if g == getattr(self, "_public_ws_gen", g):
                        self.last_ws_public_msg = time.time()
                    return orig()
                new_ws._on_pong = _wrapped_pub_pong
            if hasattr(new_ws, "_is_custom_pong") and callable(new_ws._is_custom_pong):
                orig_cp = new_ws._is_custom_pong
                def _wrapped_pub_custom_pong(msg, orig=orig_cp, g=gen):
                    is_pong = orig(msg)
                    if is_pong and g == getattr(self, "_public_ws_gen", g):
                        self.last_ws_public_msg = time.time()
                    return is_pong
                new_ws._is_custom_pong = _wrapped_pub_custom_pong

            cb_dir = getattr(new_ws, "callback_directory", None)
            existing_topics = set(cb_dir.keys()) if isinstance(cb_dir, dict) else set()

            ticker_targets = [s for s in watch_syms if f"tickers.{s}" not in existing_topics]
            ticker_cb = lambda msg, g=gen: self._handle_ws_ticker(msg, ws_gen=g)
            for i in range(0, len(ticker_targets), 10):
                batch = ticker_targets[i : i + 10]
                if batch:
                    new_ws.ticker_stream(symbol=batch, callback=ticker_cb)
                    for s in batch:
                        existing_topics.add(f"tickers.{s}")

            if getattr(config, "USE_WS_KLINE_TRIGGER", True) and hasattr(new_ws, "kline_stream"):
                try:
                    ltf_norm, ltf_iv = self._normalize_kline_interval(getattr(config, "INTERVAL", INTERVAL))
                    htf_norm, htf_iv = self._normalize_kline_interval(getattr(config, "HIGHER_INTERVAL", HIGHER_INTERVAL))
                    bm_norm, bm_iv = self._normalize_kline_interval(getattr(config, "BENCHMARK_INTERVAL", BENCHMARK_INTERVAL))

                    kline_requests: list[tuple[str, int | str, list[str]]] = []
                    if kline_syms:
                        kline_requests.append((ltf_norm, ltf_iv, kline_syms))
                        if htf_norm != ltf_norm:
                            kline_requests.append((htf_norm, htf_iv, kline_syms))
                        if bm_sym and bm_norm not in (ltf_norm, htf_norm):
                            kline_requests.append((bm_norm, bm_iv, [bm_sym]))

                    kline_cb = lambda msg, g=gen: self._handle_ws_kline(msg, ws_gen=g)
                    for tf_str, tf_iv, syms_for_tf in kline_requests:
                        fresh_syms = []
                        for s in syms_for_tf:
                            t_key = f"kline.{tf_str}.{s}"
                            if t_key not in existing_topics:
                                existing_topics.add(t_key)
                                fresh_syms.append(s)
                        for i in range(0, len(fresh_syms), 10):
                            batch = fresh_syms[i : i + 10]
                            if batch:
                                new_ws.kline_stream(interval=tf_iv, symbol=batch, callback=kline_cb)
                except Exception as ke:
                    log.warning(f"Не удалось подписаться на WebSocket kline_stream ({ke}); будет использован REST fallback.")
            with self.lock:
                if gen == getattr(self, "_public_ws_gen", gen):
                    self.ws = new_ws
                    self.last_ws_public_msg = time.time()
                else:
                    self._safe_close_ws(new_ws)
                    return
            log.info(f"Публичный WebSocket подключен (тикеры {len(watch_syms)} пар + kline_stream {len(kline_syms)} пар).")
        except Exception as e:
            log.warning(f"Публичный WebSocket не удалось подключить ({e}). Работаем через REST API fallback.")
            self._safe_close_ws(new_ws)
            with self.lock:
                if gen == getattr(self, "_public_ws_gen", gen):
                    self.ws = None

    def _init_private_ws(self):
        """Подключает приватный WebSocket User Data Stream (position, execution, order)."""
        if getattr(config, "SHADOW_MODE", False):
            return
        if config.is_placeholder_key(config.API_KEY) or config.is_placeholder_key(config.API_SECRET):
            return
        with self.lock:
            old_priv = getattr(self, "ws_private", None)
            self.ws_private = None
            self._private_ws_gen = getattr(self, "_private_ws_gen", 0) + 1
            gen = self._private_ws_gen

        self._safe_close_ws(old_priv)

        try:
            new_priv = WebSocket(
                testnet=config.TESTNET,
                demo=config.DEMO,
                api_key=config.API_KEY,
                api_secret=config.API_SECRET,
                channel_type="private",
                restart_on_error=True
            )
            if hasattr(new_priv, "_on_pong") and callable(new_priv._on_pong):
                orig_pong = new_priv._on_pong
                def _wrapped_pong(orig=orig_pong, g=gen):
                    if g == getattr(self, "_private_ws_gen", g):
                        self.last_ws_private_msg = time.time()
                    return orig()
                new_priv._on_pong = _wrapped_pong
            if hasattr(new_priv, "_is_custom_pong") and callable(new_priv._is_custom_pong):
                orig_cp = new_priv._is_custom_pong
                def _wrapped_custom_pong(msg, orig=orig_cp, g=gen):
                    is_pong = orig(msg)
                    if is_pong and g == getattr(self, "_private_ws_gen", g):
                        self.last_ws_private_msg = time.time()
                    return is_pong
                new_priv._is_custom_pong = _wrapped_custom_pong

            new_priv.position_stream(callback=lambda msg, g=gen: self._handle_ws_position(msg, ws_gen=g))
            new_priv.execution_stream(callback=lambda msg, g=gen: self._handle_ws_execution(msg, ws_gen=g))
            new_priv.order_stream(callback=lambda msg, g=gen: self._handle_ws_order(msg, ws_gen=g))
            with self.lock:
                if gen == getattr(self, "_private_ws_gen", gen):
                    self.ws_private = new_priv
                    self.last_ws_private_msg = time.time()
                else:
                    self._safe_close_ws(new_priv)
                    return
            log.info("Приватный WebSocket успешно подключен (position, execution, order stream).")
        except Exception as e:
            log.warning(f"Приватный WebSocket не удалось подключить ({e}). Фоллбэк на REST sync_positions.")
            with self.lock:
                if gen == getattr(self, "_private_ws_gen", gen):
                    self.ws_private = None

    def _init_websocket(self):
        """Инициализирует фоновые WebSocket клиенты Bybit."""
        self._init_public_ws()
        self._init_private_ws()
        self.last_ws_reconnect_attempt = time.time()

    def check_and_reconnect_websockets(self):
        """
        Проверяет живость публичного и приватного WebSocket-потоков и потокобезопасно
        переподключает их при разрыве связи или таймауте heartbeat (без удержания self.lock на I/O).
        """
        reconnect_pub = False
        reconnect_priv = False
        has_keys = (
            not getattr(config, "SHADOW_MODE", False)
            and not (config.is_placeholder_key(config.API_KEY) or config.is_placeholder_key(config.API_SECRET))
        )

        with self.lock:
            now = time.time()
            if now - getattr(self, "last_ws_reconnect_attempt", 0.0) < 60.0:
                return

            pub_dead = False
            if self.ws is None:
                pub_dead = True
            elif getattr(self.ws, "exited", False):
                pub_dead = True
            elif hasattr(self.ws, "is_connected") and not self.ws.is_connected():
                pub_dead = True
            elif now - getattr(self, "last_ws_public_msg", now) > 180.0:
                pub_dead = True

            priv_dead = False
            if has_keys:
                if self.ws_private is None:
                    priv_dead = True
                elif getattr(self.ws_private, "exited", False):
                    priv_dead = True
                elif hasattr(self.ws_private, "is_connected") and not self.ws_private.is_connected():
                    priv_dead = True
                elif now - getattr(self, "last_ws_private_msg", now) > 180.0:
                    priv_dead = True

            if pub_dead or priv_dead:
                self.last_ws_reconnect_attempt = now
                reconnect_pub = pub_dead
                reconnect_priv = has_keys and (
                    priv_dead or pub_dead
                    or self.ws_private is None
                    or (hasattr(self.ws_private, "is_connected") and not self.ws_private.is_connected())
                )

        if reconnect_pub:
            log.warning("Обнаружено отключение публичного WebSocket (тикеры не поступают). Переподключение...")
            self._init_public_ws()
        if reconnect_priv:
            log.warning("Обнаружено отключение приватного WebSocket. Переподключение...")
            self._init_private_ws()

    def _handle_ws_ticker(self, message: dict, ws_gen: Optional[int] = None):
        """Callback-обработчик входящих тикеров WebSocket."""
        if ws_gen is not None and ws_gen != getattr(self, "_public_ws_gen", ws_gen):
            return
        try:
            data = message.get("data")
            if not data:
                return
            now = time.time()
            self.last_ws_public_msg = now
            items = data if isinstance(data, list) else [data]
            topic = str(message.get("topic") or "")
            topic_sym = topic.split(".")[-1] if topic.startswith("tickers.") else ""
            if not hasattr(self, "ws_last_update"):
                self.ws_last_update = {}
            watch_set = set(getattr(self, "active_symbols", SYMBOLS) or SYMBOLS) | set(SYMBOLS) | set(self.positions.keys())
            sh_eng = getattr(self, "shadow_engine", None)
            if sh_eng is not None:
                watch_set |= set(sh_eng.positions.keys())

            for item in items:
                sym = item.get("symbol") or topic_sym
                if not sym or sym not in watch_set:
                    continue
                last_p = item.get("lastPrice")
                if last_p:
                    price = float(last_p)
                    self.ws_prices[sym] = price
                    self.ws_last_update[sym] = now

                    # Real-time реакция на стоп/трейлинг/TP если открыта позиция
                    if not getattr(config, "SHADOW_MODE", False) and self.positions.get(sym) and self.risk_states.get(sym):
                        self._check_ws_risk_update(sym, price)
                    if sh_eng is not None and sym in sh_eng.positions:
                        sh_eng.on_tick(sym, price)
                        if getattr(config, "SHADOW_MODE", False):
                            self._sync_from_shadow_engine(sym)
        except Exception as e:
            log.debug(f"Ошибка в WebSocket ticker handler: {e}")

    @staticmethod
    def _interval_seconds(tf: str) -> float:
        s = str(tf).strip().upper()
        if s.isdigit():
            return float(int(s) * 60)
        if s == "D":
            return 86400.0
        if s == "W":
            return 604800.0
        return 3600.0

    def _closed_candle_idx(self, symbol: str, interval: str, df: Optional[pd.DataFrame]) -> int:
        """
        Возвращает -1 если последняя свеча в df уже подтверждена закрытой (confirm=True из WS),
        иначе -2 (стандартный индекс закрытой свечи при наличии формирующейся свечи).
        """
        if df is None or df.empty:
            return -2
        if len(df) < 2:
            return -1
        tf_norm = self._normalize_kline_interval(interval)[0]
        key = (symbol.upper(), tf_norm)
        conf_ts = getattr(self, "_last_confirmed_kline_ts", {}).get(key)
        if conf_ts is not None and "timestamp" in df.columns:
            try:
                if pd.Timestamp(df["timestamp"].iloc[-1]) == pd.Timestamp(conf_ts):
                    return -1
            except Exception:
                pass
        return -2

    def _handle_ws_kline(self, message: dict, ws_gen: Optional[int] = None):
        """Event-driven обработчик потока свечей WebSocket (kline_stream)."""
        if ws_gen is not None and ws_gen != getattr(self, "_public_ws_gen", ws_gen):
            return
        try:
            topic = str(message.get("topic") or "")
            parts = topic.split(".")
            if len(parts) < 3 or parts[0] != "kline":
                return
            tf = self._normalize_kline_interval(parts[1])[0]
            sym = str(parts[2]).strip().upper()
            data = message.get("data")
            if not data:
                return
            items = data if isinstance(data, list) else [data]
            now = time.time()
            self.last_ws_public_msg = now
            tf_sec = self._interval_seconds(tf)

            confirmed_hit = False
            with self.lock:
                if not hasattr(self, "kline_cache"):
                    self.kline_cache = {}
                if not hasattr(self, "kline_cache_ts"):
                    self.kline_cache_ts = {}
                if not hasattr(self, "_last_confirmed_kline_ts"):
                    self._last_confirmed_kline_ts = {}
                key = (sym, tf)
                df = self.kline_cache.get(key)

                for item in items:
                    start_ms = int(item.get("start") or 0)
                    if start_ms <= 0:
                        continue
                    ts = pd.to_datetime(start_ms, unit="ms")
                    row_dict = {
                        "timestamp": ts,
                        "open": float(item.get("open") or 0.0),
                        "high": float(item.get("high") or 0.0),
                        "low": float(item.get("low") or 0.0),
                        "close": float(item.get("close") or 0.0),
                        "volume": float(item.get("volume") or 0.0),
                        "turnover": str(item.get("turnover") or "0"),
                    }
                    if df is not None and not df.empty:
                        last_ts = pd.Timestamp(df["timestamp"].iloc[-1])
                        if ts == last_ts:
                            for col, val in row_dict.items():
                                df.at[df.index[-1], col] = val
                        elif len(df) >= 2 and ts == pd.Timestamp(df["timestamp"].iloc[-2]):
                            for col, val in row_dict.items():
                                df.at[df.index[-2], col] = val
                        elif ts > last_ts:
                            gap_sec = (pd.Timestamp(ts) - last_ts).total_seconds()
                            if gap_sec > tf_sec * 1.5:
                                # Обнаружен пропуск свечей (например, после разрыва WS) -> принудительно обновим через REST
                                self.kline_cache.pop(key, None)
                                self.kline_cache_ts.pop(key, None)
                                df = None
                            else:
                                df = pd.concat([df, pd.DataFrame([row_dict])], ignore_index=True)
                                if len(df) > 700:
                                    df = df.iloc[-700:].reset_index(drop=True)
                    else:
                        df = pd.DataFrame([row_dict])

                    if df is not None:
                        self.kline_cache[key] = df
                        self.kline_cache_ts[key] = now

                    if bool(item.get("confirm", False)):
                        confirmed_hit = True
                        self._last_confirmed_kline_ts[key] = ts
                        if hasattr(self, "_confirmed_candles"):
                            self._confirmed_candles.append((sym, tf, start_ms))

            if confirmed_hit and hasattr(self, "_candle_event") and self._candle_event is not None:
                self._candle_event.set()
        except Exception as e:
            log.debug(f"Ошибка в WebSocket kline handler: {e}")

    def _db_record_trade(self, **kwargs):
        """Безопасно записывает торговое событие в SQLite (если БД инициализирована)."""
        db = getattr(self, "db", None)
        if db is None:
            return None
        try:
            return db.record_trade_event(**kwargs)
        except Exception as e:
            log.debug(f"Ошибка записи сделки в SQLite: {e}")
            return None

    def _db_record_equity(self, equity: float, balance: float = 0.0, unrealized_pnl: float = 0.0, source: str = "bot"):
        db = getattr(self, "db", None)
        if db is None:
            return
        try:
            db.record_equity_snapshot(
                equity=equity,
                balance=balance,
                unrealized_pnl=unrealized_pnl,
                source=source,
                is_shadow=False,
            )
        except Exception as e:
            log.debug(f"Ошибка записи снимка equity в SQLite: {e}")

    def schedule_closed_pnl_sync(self, symbol: str, order_id: str = "", margin_used: float = 0.0, delay_sec: Optional[float] = None):
        """Ставит в очередь неблокирующую сверку закрытия с Bybit get_closed_pnl через 3-5с."""
        delay = float(delay_sec if delay_sec is not None else getattr(config, "CLOSED_PNL_SYNC_DELAY_SEC", 4.0))
        if not hasattr(self, "_pending_pnl_sync"):
            self._pending_pnl_sync = []
        self._pending_pnl_sync.append((symbol.upper(), str(order_id or ""), float(margin_used or 0.0), time.time() + delay))

    def reconcile_closed_pnl(self, symbol: str, order_id: str = "", margin_used: float = 0.0, limit: int = 10) -> int:
        """Запрашивает Bybit v5 get_closed_pnl и обновляет точный Net PnL (USDT) в SQLite."""
        db = getattr(self, "db", None)
        client = getattr(self, "client", None)
        if db is None or client is None or not hasattr(client, "get_closed_pnl"):
            return 0
        try:
            resp = self._safe_api_call(client.get_closed_pnl, category="linear", symbol=symbol, limit=limit)
            if not isinstance(resp, dict) or resp.get("retCode", 0) != 0:
                return 0
            items = resp.get("result", {}).get("list", []) or []
            min_orphan = float(getattr(self, "_bot_start_ts", time.time()) or time.time()) - 300.0
            synced = 0
            for idx, item in enumerate(items):
                item_oid = str(item.get("orderId") or "")
                eff_margin = margin_used if ((order_id and item_oid == order_id) or (not order_id and idx == 0)) else 0.0
                if db.upsert_closed_pnl_record(
                    item,
                    margin_used=eff_margin,
                    allow_orphan=True,
                    min_orphan_ts=min_orphan,
                ):
                    synced += 1
            return synced
        except Exception as e:
            log.debug(f"[{symbol}] Ошибка сверки get_closed_pnl: {e}")
            return 0

    def process_pending_pnl_sync(self):
        """Обрабатывает отложенные запросы сверки get_closed_pnl без блокировки торгового цикла."""
        pending = getattr(self, "_pending_pnl_sync", None)
        if not pending:
            return
        now = time.time()
        still_waiting = []
        ready = []
        for entry in list(pending):
            sym, oid, m_used, due = entry
            if now >= due:
                ready.append((sym, oid, m_used))
            else:
                still_waiting.append(entry)
        self._pending_pnl_sync = still_waiting
        any_synced = 0
        for sym, oid, m_used in ready:
            any_synced += self.reconcile_closed_pnl(sym, order_id=oid, margin_used=m_used)
        if any_synced > 0 and not getattr(config, "SHADOW_MODE", False):
            try:
                eq = self.get_wallet_equity()
                if getattr(self, "last_equity_ok", False) and eq > 0:
                    self._db_record_equity(equity=eq, balance=eq, source="close")
            except Exception:
                pass

    def _log_exchange_close(self, sym: str, cur_pos: dict, cur_risk: Optional[PositionRiskState], source: str = "WebSocket"):
        """Формирует и записывает в trade.log и SQLite закрытие позиции на бирже с взвешенным PnL."""
        try:
            entry_price = float(cur_pos.get("entry_price") or (cur_risk.entry_price if cur_risk else 0.0) or 0.0)
            is_short = (
                str(cur_pos.get("position", "")).lower() == "short"
                or str(cur_pos.get("side", "")).lower() == "sell"
            )
            cur_pos_side = str(cur_pos.get("side") or ("Sell" if is_short else "Buy"))

            exec_p = None
            close_oid = ""
            exec_info = getattr(self, "last_exec_info", {}).pop(sym, None)
            if isinstance(exec_info, dict) and exec_info.get("price", 0.0) > 0:
                if not exec_info.get("side") or exec_info["side"].lower() != cur_pos_side.lower():
                    exec_p = float(exec_info["price"])
                    close_oid = str(exec_info.get("order_id") or "")

            raw_exec_p = getattr(self, "last_exec_prices", {}).pop(sym, None)
            if exec_p is None and raw_exec_p:
                exec_p = float(raw_exec_p)

            ws_p = getattr(self, "ws_prices", {}).get(sym)
            sl_p = float((cur_risk.stop_loss if cur_risk else 0.0) or cur_pos.get("stop_loss") or 0.0)
            tp_p = float((cur_risk.take_profit if cur_risk else 0.0) or cur_pos.get("take_profit") or 0.0)
            exit_price = float(exec_p or ws_p or sl_p or entry_price or 0.0)

            hit_tp = False
            if tp_p > 0 and exit_price > 0:
                if (not is_short and exit_price >= tp_p * 0.9985) or (is_short and exit_price <= tp_p * 1.0015):
                    hit_tp = True

            if hit_tp:
                reason = "TP2" if (cur_risk and cur_risk.tp1_executed) else "TP"
            elif cur_risk and cur_risk.trailing_active:
                reason = "Trailing SL"
            elif cur_risk and cur_risk.native_trailing_active:
                be_price = entry_price * (1.0 - BREAKEVEN_OFFSET if is_short else 1.0 + BREAKEVEN_OFFSET) if entry_price > 0 else 0.0
                moved_past_be = (
                    (is_short and exit_price < be_price * 0.9985)
                    or (not is_short and exit_price > be_price * 1.0015)
                ) if (cur_risk.breakeven_applied and be_price > 0) else False
                if not cur_risk.breakeven_applied or moved_past_be:
                    reason = "Trailing SL"
                else:
                    reason = "Безубыток"
            elif cur_risk and cur_risk.breakeven_applied:
                reason = "Безубыток"
            else:
                reason = "SL/TP на бирже"

            if entry_price > 0 and exit_price > 0:
                dir_sign = -1.0 if is_short else 1.0
                rem_pnl_pct = ((exit_price - entry_price) / entry_price) * dir_sign * 100.0 * LEVERAGE
                pnl_pct = rem_pnl_pct

                if cur_risk and cur_risk.tp1_executed and cur_risk.initial_size > 0:
                    init_sz = float(cur_risk.initial_size)
                    rem_sz = float(cur_pos.get("size") or cur_risk.size or 0.0)
                    tp1_sz = float(cur_risk.tp1_closed_qty or max(0.0, init_sz - rem_sz))
                    tp1_p = float(cur_risk.tp1_fill_price or cur_risk.tp1_price or 0.0)
                    if 0 < tp1_sz < init_sz and rem_sz > 0 and tp1_p > 0:
                        w1 = tp1_sz / init_sz
                        w2 = rem_sz / init_sz
                        tp1_pnl_pct = ((tp1_p - entry_price) / entry_price) * dir_sign * 100.0 * LEVERAGE
                        pnl_pct = (w1 * tp1_pnl_pct) + (w2 * rem_pnl_pct)

                sign = "+" if pnl_pct > 0 else ""
                trade_log.info(
                    f"[{sym}] Закрыто ({source}: {reason}) @ {exit_price} | PnL: {sign}{pnl_pct:.2f}%"
                )

                pr = getattr(self, "portfolio_risk", None)
                if pr is not None:
                    pr.record_closed_trade(sym, pnl_pct)

                rem_sz_db = float(cur_pos.get("size") or (cur_risk.size if cur_risk else 0.0) or 0.0)
                margin_used = float(getattr(cur_risk, "margin_used", 0.0) or (rem_sz_db * entry_price / LEVERAGE))
                est_usdt = margin_used * (pnl_pct / 100.0) if margin_used > 0 else None
                self._db_record_trade(
                    symbol=sym,
                    action="ЗАКРЫТИЕ",
                    side="Sell" if not is_short else "Buy",
                    strategy=getattr(cur_risk, "strategy_tag", "") or getattr(self.strategy, "name", "") if hasattr(self, "strategy") else "",
                    price=exit_price,
                    qty=rem_sz_db,
                    pnl_pct=pnl_pct,
                    net_pnl_usdt=est_usdt,
                    order_id=close_oid,
                    exec_type=source,
                    pos_ref=getattr(cur_risk, "pos_ref", "") if cur_risk else "",
                    is_profit=(pnl_pct > 0),
                    is_shadow=False,
                    details=f"{source}: {reason} @ {exit_price} | PnL: {sign}{pnl_pct:.2f}%",
                )
                self.schedule_closed_pnl_sync(sym, order_id=close_oid, margin_used=margin_used)
            else:
                trade_log.info(f"[{sym}] {source}: Позиция закрыта на бирже (size=0).")
                self._db_record_trade(
                    symbol=sym,
                    action="ЗАКРЫТИЕ",
                    side=cur_pos_side,
                    price=exit_price,
                    order_id=close_oid,
                    exec_type=source,
                    pos_ref=getattr(cur_risk, "pos_ref", "") if cur_risk else "",
                    details=f"{source}: Позиция закрыта на бирже (size=0)",
                )
                self.schedule_closed_pnl_sync(sym, order_id=close_oid)
        except Exception as e:
            log.debug(f"Ошибка логирования закрытия на бирже [{sym}]: {e}")
            trade_log.info(f"[{sym}] {source}: Позиция закрыта на бирже (size=0).")

    def _is_confirmed_tp1_reduction(self, sym: str, cur_risk: PositionRiskState, new_size: float) -> tuple[bool, float, float]:
        """
        Проверяет, является ли сокращение размера позиции подтвержденным срабатыванием TP1.
        Возвращает (confirmed, closed_qty, fill_price).
        """
        prev_size = float(cur_risk.size)
        closed_qty = max(0.0, prev_size - float(new_size))
        if closed_qty <= 0:
            return False, 0.0, 0.0

        tp1_oid = getattr(cur_risk, "tp1_order_id", "")
        if tp1_oid:
            ws_ord = getattr(self, "ws_orders", {}).get(tp1_oid)
            if ws_ord:
                st = ws_ord.get("status", "")
                if st == "Filled":
                    fill_p = float(ws_ord.get("avg_price") or cur_risk.tp1_price or 0.0)
                    fill_q = float(ws_ord.get("cum_qty") or closed_qty)
                    return True, fill_q, fill_p
                if st in ("Cancelled", "Rejected", "Deactivated"):
                    cur_risk.tp1_order_id = ""
                    return False, closed_qty, 0.0

            try:
                if hasattr(self, "client") and hasattr(self.client, "get_open_orders"):
                    chk = self.client.get_open_orders(category="linear", symbol=sym, orderId=tp1_oid)
                    if isinstance(chk, dict) and chk.get("result", {}).get("list"):
                        return False, closed_qty, 0.0
                if hasattr(self, "client") and hasattr(self.client, "get_order_history"):
                    h_resp = self.client.get_order_history(category="linear", symbol=sym, orderId=tp1_oid)
                    if isinstance(h_resp, dict):
                        h_list = h_resp.get("result", {}).get("list", [])
                        if h_list and isinstance(h_list, list):
                            st = str(h_list[0].get("orderStatus", ""))
                            cum_q = float(h_list[0].get("cumExecQty") or 0.0)
                            avg_p = float(h_list[0].get("avgPrice") or cur_risk.tp1_price or 0.0)
                            if st == "Filled" or cum_q > 0:
                                return True, (cum_q if cum_q > 0 else closed_qty), avg_p
                            if st in ("Cancelled", "Rejected", "Deactivated"):
                                cur_risk.tp1_order_id = ""
                                return False, closed_qty, 0.0
            except Exception:
                pass

        # Если tp1_order_id не задан (программный/внешний TP1): проверяем, что объем соответствует PARTIAL_TP_RATIO
        # Если tp1_order_id был задан, но API/WS не дали явного подтверждения — требуем, чтобы известная цена
        # двигалась к уровню TP1 (исключает ложное срабатывание при ручном частичном закрытии в убыток).
        init_sz = float(cur_risk.initial_size or prev_size)
        expected_tp1_qty = init_sz * PARTIAL_TP_RATIO
        if expected_tp1_qty > 0 and closed_qty >= expected_tp1_qty * 0.65:
            if tp1_oid:
                known_p = float(
                    getattr(self, "last_exec_prices", {}).get(sym)
                    or getattr(self, "ws_prices", {}).get(sym)
                    or 0.0
                )
                tp1_p = float(cur_risk.tp1_price or 0.0)
                if known_p > 0 and tp1_p > 0:
                    reached = (known_p >= tp1_p * 0.998) if cur_risk.is_long else (known_p <= tp1_p * 1.002)
                    if not reached:
                        return False, closed_qty, 0.0
            return True, closed_qty, float(cur_risk.tp1_price or 0.0)
        return False, closed_qty, 0.0

    def _handle_ws_position(self, message: dict, ws_gen: Optional[int] = None):
        """
        Real-time обработчик изменений позиций по приватному WebSocket Bybit.
        Обеспечивает реакцию <50ms при закрытии по SL/TP или подтвержденном частичном закрытии TP1.
        """
        if ws_gen is not None and ws_gen != getattr(self, "_private_ws_gen", ws_gen):
            return
        if getattr(config, "SHADOW_MODE", False):
            return
        self.last_ws_private_msg = time.time()
        try:
            data = message.get("data", [])
            items = data if isinstance(data, list) else [data]
            with self.lock:
                for item in items:
                    sym = item.get("symbol")
                    if not sym or (sym not in SYMBOLS and sym not in getattr(self, "active_symbols", SYMBOLS) and sym not in self.positions):
                        continue
                    size = float(item.get("size", 0.0))
                    item_idx = int(item.get("positionIdx", 0))
                    item_side = str(item.get("side", ""))
                    cur_pos = self.positions.get(sym)
                    cur_risk = self.risk_states.get(sym)

                    if cur_pos is not None:
                        cur_idx = int(cur_pos.get("position_idx", cur_risk.position_idx if cur_risk else 0))
                        cur_side = str(cur_pos.get("side", cur_risk.side if cur_risk else ""))
                        # В Hedge Mode (positionIdx 1 и 2) игнорируем противоположную ногу позиции
                        if cur_idx in (1, 2) and item_idx in (1, 2) and item_idx != cur_idx:
                            continue
                        if size == 0.0 and item_side and cur_side and item_side != cur_side:
                            continue

                    if size == 0.0:
                        if cur_pos is not None:
                            self._log_exchange_close(sym, cur_pos, cur_risk, source="WebSocket")
                            self.positions[sym] = None
                            self.risk_states[sym] = None
                            self.breakeven_applied[sym] = False
                            self.save_state()
                    else:
                        if cur_pos is not None and cur_risk is not None:
                            if size < cur_risk.size and not cur_risk.tp1_executed:
                                confirmed, closed_qty, tp1_fill_p = self._is_confirmed_tp1_reduction(sym, cur_risk, size)
                                if confirmed:
                                    cur_risk.tp1_executed = True
                                    cur_risk.tp1_closed_qty = closed_qty
                                    if tp1_fill_p > 0:
                                        cur_risk.tp1_fill_price = tp1_fill_p
                                    cur_risk.size = size
                                    cur_pos["size"] = size
                                    if hasattr(self, "last_exec_prices"):
                                        self.last_exec_prices.pop(sym, None)
                                    if hasattr(self, "last_exec_info"):
                                        self.last_exec_info.pop(sym, None)
                                    trade_log.info(
                                        f"[{sym}] WebSocket: Обнаружено частичное закрытие TP1 на бирже (закрыто: {closed_qty})."
                                    )
                                    tp1_oid_ws = str(getattr(cur_risk, "tp1_order_id", "") or "")
                                    self._db_record_trade(
                                        symbol=sym,
                                        action="ЧАСТИЧНЫЙ TP",
                                        side="Sell" if cur_risk.is_long else "Buy",
                                        strategy=getattr(cur_risk, "strategy_tag", ""),
                                        price=tp1_fill_p or cur_risk.tp1_price,
                                        qty=closed_qty,
                                        order_id=tp1_oid_ws,
                                        exec_type="ExchangeLimitTP1",
                                        pos_ref=getattr(cur_risk, "pos_ref", ""),
                                        is_profit=True,
                                        details=f"WS TP1: закрыто {closed_qty} @ {tp1_fill_p or cur_risk.tp1_price}",
                                    )
                                    self.schedule_closed_pnl_sync(
                                        sym,
                                        order_id=tp1_oid_ws,
                                        margin_used=float(getattr(cur_risk, "margin_used", 0.0) or 0.0),
                                    )
                                    if PARTIAL_TP_MOVE_BE and not cur_risk.breakeven_applied:
                                        be_price = cur_risk.entry_price * (1 + BREAKEVEN_OFFSET if cur_risk.is_long else 1 - BREAKEVEN_OFFSET)
                                        rounded_be = self.round_price(sym, be_price)
                                        pos_idx = self.get_position_idx(sym, cur_risk.side)
                                        try:
                                            r_ts = self._api_call(
                                                self.client.set_trading_stop,
                                                category="linear",
                                                symbol=sym,
                                                stopLoss=self.format_price(sym, rounded_be),
                                                positionIdx=pos_idx
                                            )
                                            if isinstance(r_ts, dict) and r_ts.get("retCode", 0) != 0 and not self._is_stop_not_modified(r_ts):
                                                raise RuntimeError(f"retCode={r_ts.get('retCode')}: {r_ts.get('retMsg')}")
                                            cur_risk.stop_loss = rounded_be
                                            cur_pos["stop_loss"] = rounded_be
                                            cur_risk.breakeven_applied = True
                                            self.breakeven_applied[sym] = True
                                            trade_log.info(f"[{sym}] WebSocket: Стоп перенесен в безубыток: {rounded_be}")
                                        except Exception as e:
                                            if self._is_stop_not_modified(e):
                                                cur_risk.stop_loss = rounded_be
                                                cur_pos["stop_loss"] = rounded_be
                                                cur_risk.breakeven_applied = True
                                                self.breakeven_applied[sym] = True
                                            else:
                                                log.warning(f"[{sym}] WebSocket: Ошибка переноса стопа в безубыток: {e}")
                                    self.save_state()
                                else:
                                    cur_pos["size"] = size
                                    cur_risk.size = size
                                    self.save_state()
                            else:
                                cur_pos["size"] = size
                                cur_risk.size = size
        except Exception as e:
            log.debug(f"Ошибка в WS position handler: {e}")

    def _handle_ws_execution(self, message: dict, ws_gen: Optional[int] = None):
        """Real-time логирование исполнения ордеров по приватному WebSocket (без искажения цены событиями Funding)."""
        if ws_gen is not None and ws_gen != getattr(self, "_private_ws_gen", ws_gen):
            return
        if getattr(config, "SHADOW_MODE", False):
            return
        self.last_ws_private_msg = time.time()
        try:
            data = message.get("data", [])
            items = data if isinstance(data, list) else [data]
            for item in items:
                exec_id = item.get("execId")
                if exec_id:
                    if not hasattr(self, "_seen_exec_ids"):
                        self._seen_exec_ids = deque(maxlen=1000)
                    if exec_id in self._seen_exec_ids:
                        continue
                    self._seen_exec_ids.append(exec_id)

                sym = item.get("symbol")
                exec_p = item.get("execPrice")
                exec_q = item.get("execQty")
                side = item.get("side")
                exec_type = str(item.get("execType", "Trade"))
                if sym and exec_p and exec_type in ("Trade", "BustTrade", "AdlTrade", ""):
                    try:
                        if not hasattr(self, "last_exec_prices"):
                            self.last_exec_prices = {}
                        if not hasattr(self, "last_exec_info"):
                            self.last_exec_info = {}
                        px = float(exec_p)
                        self.last_exec_prices[sym] = px
                        self.last_exec_info[sym] = {
                            "price": px,
                            "side": str(side or ""),
                            "exec_type": exec_type,
                            "order_id": str(item.get("orderId") or ""),
                            "ts": time.time(),
                        }
                    except Exception:
                        pass
                trade_log.info(f"[{sym}] WS Fill ({exec_type}): {side} {exec_q} @ {exec_p}")
                if exec_type == "Funding" and sym:
                    try:
                        f_fee = float(item.get("execFee") or 0.0)
                        if f_fee != 0.0:
                            r_st = getattr(self, "risk_states", {}).get(sym)
                            self._db_record_trade(
                                symbol=sym,
                                action="FUNDING",
                                side=str(side or ""),
                                strategy=getattr(r_st, "strategy_tag", "") if r_st else "",
                                price=float(exec_p or 0.0),
                                qty=float(exec_q or 0.0),
                                funding_usdt=f_fee,
                                exec_type="Funding",
                                pos_ref=getattr(r_st, "pos_ref", "") if r_st else "",
                                details=f"Funding {f_fee:+.4f} USDT",
                            )
                    except Exception:
                        pass
        except Exception as e:
            log.debug(f"Ошибка в WS execution handler: {e}")

    def _handle_ws_order(self, message: dict, ws_gen: Optional[int] = None):
        """Real-time обработка изменений статусов ордеров (PostOnly, Limit TP1 и др.)."""
        if ws_gen is not None and ws_gen != getattr(self, "_private_ws_gen", ws_gen):
            return
        if getattr(config, "SHADOW_MODE", False):
            return
        self.last_ws_private_msg = time.time()
        try:
            data = message.get("data", [])
            items = data if isinstance(data, list) else [data]
            with self.lock:
                if not hasattr(self, "ws_orders"):
                    self.ws_orders = {}
                for item in items:
                    oid = item.get("orderId")
                    sym = item.get("symbol")
                    status = str(item.get("orderStatus", ""))
                    cum_qty = float(item.get("cumExecQty") or 0.0)
                    avg_p = float(item.get("avgPrice") or item.get("price") or 0.0)
                    if oid:
                        self.ws_orders[oid] = {
                            "status": status,
                            "cum_qty": cum_qty,
                            "avg_price": avg_p,
                            "ts": time.time(),
                        }
                        if len(self.ws_orders) > 500:
                            oldest_keys = sorted(
                                self.ws_orders.keys(),
                                key=lambda k: self.ws_orders[k].get("ts", 0.0)
                            )[:len(self.ws_orders) - 400]
                            for k in oldest_keys:
                                self.ws_orders.pop(k, None)

                    if sym and sym in self.risk_states and self.risk_states[sym] is not None:
                        r_st = self.risk_states[sym]
                        if oid and oid == r_st.tp1_order_id:
                            if status == "Filled":
                                r_st.tp1_executed = True
                                if avg_p > 0:
                                    r_st.tp1_fill_price = avg_p
                                if cum_qty > 0:
                                    r_st.tp1_closed_qty = cum_qty
                                self.save_state()
                            elif status in ("Cancelled", "Rejected", "Deactivated"):
                                r_st.tp1_order_id = ""
                                self.save_state()
        except Exception as e:
            log.debug(f"Ошибка в WS order handler: {e}")

    def _check_ws_risk_update(self, symbol: str, price: float):
        """Быстрая проверка трейлинг-стопа и частичного TP по тику из WebSocket с троттлингом."""
        now = time.time()
        if now - self.last_risk_api_call.get(symbol, 0) < 1.5:
            return

        with self.lock:
            pos = self.positions.get(symbol)
            risk = self.risk_states.get(symbol)
            if not pos or not risk:
                return

            risk.update_price_extremes(price, price)

            should_manage = False
            if USE_PARTIAL_TP and not risk.tp1_executed and not risk.tp1_order_id:
                triggered, _, _ = risk.check_partial_tp(price, ratio=PARTIAL_TP_RATIO, offset_pct=BREAKEVEN_OFFSET)
                if triggered:
                    should_manage = True

            if USE_BREAKEVEN and not risk.breakeven_applied and not should_manage:
                be_updated, new_be_sl = risk.check_breakeven(
                    price,
                    trigger_pct=BREAKEVEN_TRIGGER,
                    offset_pct=BREAKEVEN_OFFSET,
                    r_mult=BREAKEVEN_R_MULT
                )
                if be_updated:
                    rounded_be = self.round_price(symbol, new_be_sl)
                    cur_sl = pos.get("stop_loss", 0.0)
                    if (rounded_be > cur_sl) if risk.is_long else (cur_sl == 0.0 or rounded_be < cur_sl):
                        should_manage = True

            if USE_TRAILING_STOP and not risk.native_trailing_active and getattr(risk, "entry_regime", "") != "RANGE" and not should_manage:
                trail_updated, new_trail_sl = risk.check_trailing_stop(
                    current_price=price,
                    high_price=risk.highest_price,
                    low_price=risk.lowest_price,
                    trigger_pct=TRAILING_STOP_TRIGGER,
                    atr_mult=TRAILING_STOP_ATR_MULT,
                    fallback_pct=TRAILING_STOP_FALLBACK,
                    r_mult=TRAILING_STOP_R_MULT
                )
                if trail_updated:
                    rounded_trail = self.round_price(symbol, new_trail_sl)
                    cur_sl = pos.get("stop_loss", 0.0)
                    if (rounded_trail > cur_sl) if risk.is_long else (cur_sl == 0.0 or rounded_trail < cur_sl):
                        should_manage = True

            if should_manage:
                self.last_risk_api_call[symbol] = now
                self.manage_risk_and_stops(symbol, price, risk.highest_price, risk.lowest_price)

    def reload_credentials(self) -> bool:
        """Перезагружает клиент Bybit и WebSocket при переключении режима DEMO/LIVE (без удержания lock на WS exit/connect)."""
        old_pub = None
        old_priv = None
        with self.lock:
            if config.TRADING_MODE == "LIVE" and (
                config.is_placeholder_key(config.API_KEY) or config.is_placeholder_key(config.API_SECRET)
            ):
                log.error("Отклонено переключение в LIVE: заданы плейсхолдерные или пустые LIVE_API_KEY/LIVE_API_SECRET.")
                return False

            log.info(f"Перезагрузка подключения Bybit: Режим {config.TRADING_MODE}")
            sync_exchange_time(config.TESTNET, config.DEMO)
            self.client = HTTP(
                testnet=config.TESTNET,
                api_key=config.API_KEY,
                api_secret=config.API_SECRET,
                demo=config.DEMO,
                recv_window=10000,
                timeout=25,
                max_retries=5,
                retry_delay=2
            )
            if hasattr(self, "_attach_rate_limit_hook"):
                self._attach_rate_limit_hook(self.client)
            old_pub = getattr(self, "ws", None)
            self.ws = None
            self._public_ws_gen = getattr(self, "_public_ws_gen", 0) + 1
            old_priv = getattr(self, "ws_private", None)
            self.ws_private = None
            self._private_ws_gen = getattr(self, "_private_ws_gen", 0) + 1

            # Полная очистка кэшей старого аккаунта
            if hasattr(self, "ws_orders"):
                self.ws_orders.clear()
            if hasattr(self, "last_exec_prices"):
                self.last_exec_prices.clear()
            if hasattr(self, "last_exec_info"):
                self.last_exec_info.clear()
            if hasattr(self, "last_handled_candles"):
                self.last_handled_candles.clear()
            if hasattr(self, "hedge_positions"):
                self.hedge_positions.clear()

            self.positions.clear()
            self.risk_states.clear()
            self.breakeven_applied.clear()

            for sym in getattr(self, "active_symbols", SYMBOLS):
                self.positions[sym] = None
                self.risk_states[sym] = None
                self.breakeven_applied[sym] = False
                self._load_instrument_info(sym)
                self._init_leverage(sym)
                self._detect_position_mode(sym)

            self.sync_positions()

            self.last_known_equity = 0.0
            self.last_equity_ok = False
            self.circuit_breaker_active = False
            new_bal = self.get_wallet_equity()
            self.start_day_balance = new_bal if getattr(self, "last_equity_ok", False) and new_bal > 0 else FALLBACK_USDT
            trade_log.info(f"Режим {config.TRADING_MODE} активирован. Базовый баланс дня: {self.start_day_balance:.2f} USDT")

        self._safe_close_ws(old_pub)
        self._safe_close_ws(old_priv)
        self._init_websocket()
        return True

    @staticmethod
    def _count_decimals(val) -> int:
        try:
            d = Decimal(str(val)).normalize()
            exp = d.as_tuple().exponent
            return max(0, -int(exp)) if isinstance(exp, int) else 0
        except Exception:
            return 3

    def _load_instrument_info(self, symbol: str):
        """Загружает точность цены, шага лота, maxLeverage и минимального нотионала с кэшированием."""
        try:
            resp = self._safe_api_call(self.client.get_instruments_info, category="linear", symbol=symbol)
            info = resp["result"]["list"][0]
            lot_filter = info.get("lotSizeFilter", {})
            price_filter = info.get("priceFilter", {})
            lev_filter = info.get("leverageFilter", {})

            raw_qty_step = lot_filter.get("qtyStep", "0.001")
            raw_tick_size = price_filter.get("tickSize", "0.1")

            qty_step = float(raw_qty_step)
            min_qty = float(lot_filter.get("minOrderQty", 0.001))
            min_notional = float(lot_filter.get("minNotionalValue") or MIN_ORDER_NOTIONAL)
            tick_size = float(raw_tick_size)
            max_lev = float(lev_filter.get("maxLeverage") or LEVERAGE)

            qty_decimals = self._count_decimals(raw_qty_step)
            price_decimals = self._count_decimals(raw_tick_size)

            self.instruments[symbol] = {
                "qty_step": qty_step,
                "min_qty": min_qty,
                "min_notional": min_notional,
                "tick_size": tick_size,
                "qty_decimals": qty_decimals,
                "price_decimals": price_decimals,
                "max_leverage": max_lev,
                "last_loaded_ts": time.time(),
            }
            log.info(
                f"{symbol:<10} | Step: {qty_step} | Min: {min_qty} | MinNotional: {min_notional}$ | Tick: {tick_size} | MaxLev: x{max_lev:g}"
            )
        except Exception as e:
            log.warning(f"Не удалось загрузить параметры для {symbol}: {e}. Используем дефолтные.")
            self.instruments[symbol] = {
                "qty_step": 0.001,
                "min_qty": 0.001,
                "min_notional": MIN_ORDER_NOTIONAL,
                "tick_size": 0.1,
                "qty_decimals": 3,
                "price_decimals": 2,
                "max_leverage": float(LEVERAGE),
                "last_loaded_ts": time.time(),
            }

    def refresh_instruments_if_needed(self, force: bool = False):
        """Суточное обновление кэша метаданных инструментов (tickSize, qtyStep, maxLeverage) без рестарта."""
        now = time.time()
        ttl = float(getattr(config, "INSTRUMENT_CACHE_TTL_SEC", 86400))
        syms = self._get_watch_symbols() if hasattr(self, "_get_watch_symbols") else list(SYMBOLS)
        for sym in syms:
            meta = self.instruments.get(sym)
            last_ts = float(meta.get("last_loaded_ts", 0.0)) if isinstance(meta, dict) else 0.0
            prev_max_lev = float(meta.get("max_leverage", LEVERAGE)) if isinstance(meta, dict) else float(LEVERAGE)
            if force or meta is None or (now - last_ts) >= ttl:
                self._load_instrument_info(sym)
                new_max_lev = float(self.instruments.get(sym, {}).get("max_leverage") or LEVERAGE)
                if new_max_lev < prev_max_lev and new_max_lev < float(LEVERAGE) and not getattr(config, "SHADOW_MODE", False):
                    self._init_leverage(sym)

    def _init_leverage(self, symbol: str):
        """Безопасная установка плеча на бирже с ограничением maxLeverage инструмента."""
        max_lev = float(self.instruments.get(symbol, {}).get("max_leverage") or LEVERAGE)
        eff_lev = max(1, min(int(LEVERAGE), int(max_lev)))
        try:
            self._safe_api_call(
                self.client.set_leverage,
                category="linear",
                symbol=symbol,
                buyLeverage=str(eff_lev),
                sellLeverage=str(eff_lev),
            )
            log.info(f"[{symbol}] Плечо x{eff_lev} успешно установлено")
        except Exception as e:
            if "110043" in str(e) or "Set leverage not modified" in str(e):
                log.info(f"[{symbol}] Плечо уже x{eff_lev}")
            else:
                log.warning(f"[{symbol}] Ошибка установки плеча: {e}")

    def round_qty(self, symbol: str, qty: float) -> float:
        info = self.instruments.get(symbol, {"qty_step": 0.001, "qty_decimals": 3})
        step = info["qty_step"]
        dec = info["qty_decimals"]
        try:
            d_qty = Decimal(str(max(0.0, qty)))
            d_step = Decimal(str(step))
            steps_cnt = (d_qty / d_step).to_integral_value(rounding=ROUND_DOWN)
            return round(float(steps_cnt * d_step), dec)
        except Exception:
            rounded = (qty // step) * step
            return round(rounded, dec)

    def round_price(self, symbol: str, price: float) -> float:
        info = self.instruments.get(symbol, {"tick_size": 0.1, "price_decimals": 2})
        tick = info["tick_size"]
        dec = info["price_decimals"]
        try:
            d_price = Decimal(str(max(0.0, price)))
            d_tick = Decimal(str(tick))
            ticks_cnt = (d_price / d_tick).to_integral_value(rounding=ROUND_HALF_UP)
            return round(float(ticks_cnt * d_tick), dec)
        except Exception:
            rounded = round(round(price / tick) * tick, dec)
            return rounded

    def format_qty(self, symbol: str, qty: float) -> str:
        dec = self.instruments.get(symbol, {}).get("qty_decimals", 3)
        return f"{qty:.{dec}f}"

    def format_price(self, symbol: str, price: float) -> str:
        dec = self.instruments.get(symbol, {}).get("price_decimals", 2)
        return f"{price:.{dec}f}"

    def get_available_balance(self) -> float:
        """
        Получает свободный доступный баланс USDT для новых сделок.
        При исчерпанной марже или ошибке API возвращает 0.0 (блокирует новые входы),
        а не полный walletBalance и не FALLBACK_USDT.
        """
        try:
            resp = self._safe_api_call(self.client.get_wallet_balance, accountType="UNIFIED")
            acc_list = resp["result"]["list"]
            if not acc_list:
                self.last_balance_ok = False
                return 0.0
            acc = acc_list[0]
            tot_avail_raw = acc.get("totalAvailableBalance")
            tot_avail = max(0.0, float(tot_avail_raw)) if tot_avail_raw not in (None, "") else None
            coins = acc.get("coin", [])
            for c in coins:
                if c.get("coin") == "USDT":
                    raw_avail = c.get("availableToWithdraw")
                    avail_val = max(0.0, float(raw_avail)) if raw_avail not in (None, "") else None
                    wb = float(c.get("walletBalance") or 0.0)
                    pos_im = float(c.get("totalPositionIM") or 0.0)
                    ord_im = float(c.get("totalOrderIM") or 0.0)
                    free_wb = max(0.0, wb - pos_im - ord_im)
                    self.last_balance_ok = True
                    if tot_avail is not None:
                        avail_res = min(tot_avail, free_wb) if wb > 0 and (pos_im > 0 or ord_im > 0) else tot_avail
                    elif avail_val is not None:
                        avail_res = min(avail_val, free_wb) if wb > 0 and (pos_im > 0 or ord_im > 0) else avail_val
                    else:
                        avail_res = free_wb
                    self.last_known_balance = avail_res
                    return avail_res
            self.last_balance_ok = True
            avail_res = tot_avail if tot_avail is not None else 0.0
            self.last_known_balance = avail_res
            return avail_res
        except Exception as e:
            log.warning(f"Ошибка получения баланса: {e}. Возвращаем 0.0 USDT (блокировка новых входов).")
            self.last_balance_ok = False
            return 0.0

    def get_wallet_equity(self) -> float:
        """Получает общий капитал (Equity = Balance + Unrealized PnL) с защитой от сетевых сбоев."""
        try:
            resp = self._safe_api_call(self.client.get_wallet_balance, accountType="UNIFIED")
            acc_list = resp["result"]["list"]
            if not acc_list:
                self.last_equity_ok = False
                return self.last_known_equity or FALLBACK_USDT
            coins = acc_list[0]["coin"]
            for c in coins:
                if c["coin"] == "USDT":
                    raw_eq = c.get("equity") or c.get("walletBalance")
                    if raw_eq not in (None, "") and float(raw_eq) > 0:
                        eq = float(raw_eq)
                        self.last_known_equity = eq
                        self.last_equity_ok = True
                        return eq
            self.last_equity_ok = False
            return self.last_known_equity or FALLBACK_USDT
        except Exception as e:
            log.warning(f"Ошибка получения equity: {e}")
            self.last_equity_ok = False
            return self.last_known_equity or FALLBACK_USDT

    def check_daily_circuit_breaker(self):
        """Защита от дневной просадки (Daily Loss Circuit Breaker) + HWM MaxDD-троттлинг."""
        today = datetime.now().date()
        is_shadow_only = bool(getattr(config, "SHADOW_MODE", False)) and getattr(self, "shadow_engine", None) is not None
        pr = self.shadow_engine.portfolio_risk if is_shadow_only else getattr(self, "portfolio_risk", None)

        if is_shadow_only:
            curr_equity = float(self.shadow_engine.equity)
            self.last_known_equity = curr_equity
            self.last_equity_ok = True
            if today != self.start_day:
                self.start_day = today
                self.start_day_balance = curr_equity
                self.circuit_breaker_active = False
                if pr is not None:
                    pr.update_equity(curr_equity, valid=True)
                if hasattr(self, "risk_states"):
                    self.save_state()
                return
            if self.start_day_balance <= 0:
                return
            if pr is not None and curr_equity > 0:
                pr.update_equity(curr_equity, valid=True)
            daily_loss_pct = (self.start_day_balance - curr_equity) / self.start_day_balance
            if daily_loss_pct >= MAX_DAILY_LOSS_PCT and not self.circuit_breaker_active:
                self.circuit_breaker_active = True
                trade_log.critical(
                    f"[SHADOW] Сработал Circuit Breaker! Дневная просадка {daily_loss_pct*100:.2f}% >= {MAX_DAILY_LOSS_PCT*100:.1f}%."
                )
                self.save_state()
                self.panic_close_all()
            return

        if today != self.start_day:
            curr_equity = self.get_wallet_equity()
            if getattr(self, "last_equity_ok", False) and curr_equity > 0:
                self.start_day = today
                self.start_day_balance = curr_equity
                self.circuit_breaker_active = False
                if pr is not None:
                    pr.update_equity(curr_equity, valid=True)
                trade_log.info(f"Новый торговый день. Базовый баланс сброшен на: {self.start_day_balance:.2f} USDT")
                if hasattr(self, "risk_states"):
                    self.save_state()
            elif getattr(self, "last_known_equity", 0.0) > 0:
                self.start_day = today
                self.start_day_balance = self.last_known_equity
                self.circuit_breaker_active = False
                trade_log.info(
                    f"Новый торговый день (кэш equity). Базовый баланс сброшен на: {self.start_day_balance:.2f} USDT"
                )
                if hasattr(self, "risk_states"):
                    self.save_state()
            return

        if self.start_day_balance <= 0:
            return

        curr_equity = self.get_wallet_equity()
        if not getattr(self, "last_equity_ok", False):
            log.warning("Пропуск проверки Circuit Breaker: нет достоверного ответа баланса от API Bybit")
            return

        if pr is not None and curr_equity > 0:
            prev_peak = pr.peak_equity
            prev_pause = pr.drawdown_pause_active
            dd_info = pr.update_equity(curr_equity, valid=True)
            if dd_info["drawdown_pause"] and not prev_pause:
                trade_log.warning(
                    f"Активирован Tier-2 MaxDD Pause: просадка от пика {dd_info['peak_dd_pct']:.2f}% >= {pr.dd_tier2_pct*100:.0f}%. Только закрытие позиций."
                )
            if (pr.peak_equity > prev_peak or dd_info["drawdown_pause"] != prev_pause) and hasattr(self, "risk_states"):
                self.save_state()

        daily_loss_pct = (self.start_day_balance - curr_equity) / self.start_day_balance

        if daily_loss_pct >= MAX_DAILY_LOSS_PCT and not self.circuit_breaker_active:
            self.circuit_breaker_active = True
            trade_log.critical(
                f"Сработал Circuit Breaker! Дневная просадка {daily_loss_pct*100:.2f}% >= "
                f"лимита {MAX_DAILY_LOSS_PCT*100:.1f}%. Торговля остановлена, позиции закрываются!"
            )
            if hasattr(self, "risk_states"):
                self.save_state()
            self.panic_close_all()

    def sync_positions(self):
        """Синхронизирует позиции бота с биржей через REST API (с поддержкой Hedge Mode)."""
        if getattr(config, "SHADOW_MODE", False) and getattr(self, "shadow_engine", None) is not None:
            self._sync_from_shadow_engine()
            return
        with self.lock:
            try:
                resp = self._safe_api_call(self.client.get_positions, category="linear", settleCoin="USDT")
                raw_positions = resp["result"]["list"]

                live_legs = {}  # {(symbol, position_idx): dict}
                live_by_sym = {}
                for p in raw_positions:
                    size = float(p.get("size", 0))
                    if size > 0:
                        sym = p.get("symbol")
                        side = p.get("side")
                        p_idx = int(p.get("positionIdx", 0))
                        leg = {
                            "symbol": sym,
                            "side": side,
                            "position": "long" if side == "Buy" else "short",
                            "size": size,
                            "entry_price": float(p.get("avgPrice") or 0.0),
                            "stop_loss": float(p.get("stopLoss") or 0.0),
                            "take_profit": float(p.get("takeProfit") or 0.0),
                            "unrealised_pnl": float(p.get("unrealisedPnl") or 0.0),
                            "position_idx": p_idx
                        }
                        live_legs[(sym, p_idx)] = leg
                        live_by_sym.setdefault(sym, []).append(leg)

                self.hedge_positions = live_legs
                saved_states = self.load_state()

                live_map = {}
                for sym, legs in live_by_sym.items():
                    if len(legs) == 1:
                        live_map[sym] = legs[0]
                    else:
                        # В Hedge Mode выбираем ногу, соответствующую текущему/сохраненному состоянию
                        cur_p = self.positions.get(sym)
                        sav_s = saved_states.get(sym)
                        target_idx = (
                            int(cur_p.get("position_idx", 0)) if cur_p
                            else (int(sav_s.position_idx) if sav_s else legs[0]["position_idx"])
                        )
                        chosen = next((l for l in legs if l["position_idx"] == target_idx), legs[0])
                        live_map[sym] = chosen
                        log.warning(f"[{sym}] Обнаружены две открытые Hedge-ноги на бирже; отслеживается positionIdx={chosen['position_idx']}.")

                tracked_symbols = list(dict.fromkeys(
                    list(getattr(self, "active_symbols", SYMBOLS))
                    + list(SYMBOLS)
                    + list(self.positions.keys())
                    + list(live_map.keys())
                ))
                for sym in tracked_symbols:
                    old_pos = self.positions.get(sym)
                    new_pos = live_map.get(sym)

                    if old_pos and not new_pos:
                        self._log_exchange_close(sym, old_pos, self.risk_states.get(sym), source="Sync")
                        self.breakeven_applied[sym] = False
                        self.risk_states[sym] = None
                        self.save_state()
                    elif not old_pos and new_pos:
                        trade_log.info(f"[{sym}] Обнаружена открытая позиция {new_pos['position'].upper()} ({new_pos['size']})")
                        if sym in saved_states and saved_states[sym].side.lower() == new_pos["side"].lower():
                            restored = saved_states[sym]
                            if not restored.tp1_executed and new_pos["size"] < restored.size:
                                confirmed, closed_q, tp1_p = self._is_confirmed_tp1_reduction(sym, restored, new_pos["size"])
                                if confirmed:
                                    restored.tp1_executed = True
                                    restored.tp1_closed_qty = closed_q
                                    if tp1_p > 0:
                                        restored.tp1_fill_price = tp1_p
                                    trade_log.info(f"[{sym}] Рестарт: подтверждено срабатывание TP1 за время простоя (закрыто {closed_q}).")
                            restored.size = new_pos["size"]
                            if new_pos["entry_price"] > 0:
                                restored.entry_price = new_pos["entry_price"]
                            if new_pos["stop_loss"] > 0:
                                restored.stop_loss = new_pos["stop_loss"]
                            if new_pos["take_profit"] > 0:
                                restored.take_profit = new_pos["take_profit"]
                            restored.position_idx = new_pos.get("position_idx", 0)
                            if restored.tp1_executed and PARTIAL_TP_MOVE_BE and not restored.breakeven_applied and restored.entry_price > 0:
                                be_price = restored.entry_price * (1 + BREAKEVEN_OFFSET if restored.is_long else 1 - BREAKEVEN_OFFSET)
                                rounded_be = self.round_price(sym, be_price)
                                try:
                                    self._api_call(
                                        self.client.set_trading_stop,
                                        category="linear", symbol=sym,
                                        stopLoss=self.format_price(sym, rounded_be),
                                        positionIdx=restored.position_idx
                                    )
                                    restored.stop_loss = rounded_be
                                    new_pos["stop_loss"] = rounded_be
                                    restored.breakeven_applied = True
                                except Exception as e:
                                    log.warning(f"[{sym}] Рестарт: ошибка переноса стопа в БУ после TP1: {e}")
                            self.risk_states[sym] = restored
                            self.breakeven_applied[sym] = restored.breakeven_applied
                            trade_log.info(f"[{sym}] Восстановлено состояние рисков: TP1={restored.tp1_executed}, SL={restored.stop_loss}")
                        else:
                            self.risk_states[sym] = PositionRiskState(
                                symbol=sym,
                                side=new_pos["side"],
                                entry_price=new_pos["entry_price"],
                                size=new_pos["size"],
                                initial_size=new_pos["size"],
                                stop_loss=new_pos["stop_loss"],
                                take_profit=new_pos["take_profit"],
                                r_mult=PARTIAL_TP_R_MULT,
                                position_idx=new_pos.get("position_idx", 0)
                            )
                            self.breakeven_applied[sym] = False
                        self.save_state()
                    elif old_pos and new_pos:
                        cur_risk = self.risk_states.get(sym)
                        if cur_risk is None or old_pos.get("side") != new_pos.get("side"):
                            self.risk_states[sym] = PositionRiskState(
                                symbol=sym,
                                side=new_pos["side"],
                                entry_price=new_pos["entry_price"],
                                size=new_pos["size"],
                                initial_size=new_pos["size"],
                                stop_loss=new_pos["stop_loss"],
                                take_profit=new_pos["take_profit"],
                                r_mult=PARTIAL_TP_R_MULT,
                                position_idx=new_pos.get("position_idx", 0)
                            )
                            self.breakeven_applied[sym] = False
                            self.save_state()
                        else:
                            if new_pos["size"] < cur_risk.size and not cur_risk.tp1_executed:
                                confirmed, closed_qty, tp1_p = self._is_confirmed_tp1_reduction(sym, cur_risk, new_pos["size"])
                                if confirmed:
                                    cur_risk.tp1_executed = True
                                    cur_risk.tp1_closed_qty = closed_qty
                                    if tp1_p > 0:
                                        cur_risk.tp1_fill_price = tp1_p
                                    cur_risk.size = new_pos["size"]
                                    if hasattr(self, "last_exec_prices"):
                                        self.last_exec_prices.pop(sym, None)
                                    if hasattr(self, "last_exec_info"):
                                        self.last_exec_info.pop(sym, None)
                                    trade_log.info(
                                        f"[{sym}] Sync: Обнаружено частичное закрытие TP1 на бирже (закрыто: {closed_qty})."
                                    )
                                    if PARTIAL_TP_MOVE_BE and not cur_risk.breakeven_applied:
                                        be_price = cur_risk.entry_price * (1 + BREAKEVEN_OFFSET if cur_risk.is_long else 1 - BREAKEVEN_OFFSET)
                                        rounded_be = self.round_price(sym, be_price)
                                        pos_idx = self.get_position_idx(sym, cur_risk.side)
                                        try:
                                            self._api_call(
                                                self.client.set_trading_stop,
                                                category="linear",
                                                symbol=sym,
                                                stopLoss=self.format_price(sym, rounded_be),
                                                positionIdx=pos_idx
                                            )
                                            new_pos["stop_loss"] = rounded_be
                                            cur_risk.stop_loss = rounded_be
                                            cur_risk.breakeven_applied = True
                                            self.breakeven_applied[sym] = True
                                            trade_log.info(f"[{sym}] Sync: Стоп перенесен в безубыток: {rounded_be}")
                                        except Exception as e:
                                            log.warning(f"[{sym}] Sync: Ошибка переноса стопа в безубыток: {e}")
                                    self.save_state()
                                else:
                                    cur_risk.size = new_pos["size"]
                                    self.save_state()
                            else:
                                cur_risk.size = new_pos["size"]
                                if new_pos["size"] > cur_risk.initial_size and not cur_risk.tp1_executed:
                                    cur_risk.initial_size = new_pos["size"]
                                if new_pos["entry_price"] > 0:
                                    cur_risk.entry_price = new_pos["entry_price"]
                                if new_pos["stop_loss"] > 0:
                                    cur_risk.stop_loss = new_pos["stop_loss"]
                                if new_pos["take_profit"] > 0:
                                    cur_risk.take_profit = new_pos["take_profit"]

                    self.positions[sym] = new_pos

            except Exception as e:
                log.error(f"Ошибка синхронизации позиций: {e}")

    def get_open_positions_count(self) -> int:
        return sum(1 for p in self.positions.values() if p is not None)

    def get_klines(self, symbol: str, interval: str = None, limit: int = 150) -> pd.DataFrame:
        tf = self._normalize_kline_interval(interval or getattr(config, "INTERVAL", INTERVAL))[0]
        try:
            resp = self._safe_api_call(
                self.client.get_kline,
                category="linear",
                symbol=symbol,
                interval=tf,
                limit=limit
            )
            data = resp["result"]["list"]
            df = pd.DataFrame(data, columns=[
                "timestamp", "open", "high", "low", "close", "volume", "turnover"
            ])
            df = df.astype({
                "open": float, "high": float, "low": float,
                "close": float, "volume": float
            })
            df["timestamp"] = pd.to_datetime(df["timestamp"].astype("int64"), unit="ms")
            df = df.sort_values("timestamp").reset_index(drop=True)
            if not df.empty:
                with self.lock:
                    if not hasattr(self, "kline_cache"):
                        self.kline_cache = {}
                    if not hasattr(self, "kline_cache_ts"):
                        self.kline_cache_ts = {}
                    self.kline_cache[(symbol.upper(), tf)] = df.tail(700).reset_index(drop=True)
                    self.kline_cache_ts[(symbol.upper(), tf)] = time.time()
            return df
        except Exception as e:
            log.error(f"[{symbol}] Ошибка получения свечей (TF {tf}): {e}")
            return pd.DataFrame()

    def get_klines_fast(self, symbol: str, interval: str = None, limit: int = 150) -> pd.DataFrame:
        """
        Возвращает свечи из WebSocket kline_cache (0 REST-запросов), если кэш прогрет и свеж;
        иначе выполняет REST get_klines и обновляет кэш.
        """
        tf = self._normalize_kline_interval(interval or getattr(config, "INTERVAL", INTERVAL))[0]
        sym = symbol.upper()
        if getattr(config, "USE_WS_KLINE_TRIGGER", True) and getattr(self, "ws", None) is not None:
            ws_alive = not getattr(self.ws, "exited", False)
            if hasattr(self.ws, "is_connected"):
                try:
                    ws_alive = ws_alive and bool(self.ws.is_connected())
                except Exception:
                    ws_alive = False
            if ws_alive:
                tf_sec = self._interval_seconds(tf)
                with self.lock:
                    cached = getattr(self, "kline_cache", {}).get((sym, tf))
                    c_ts = float(getattr(self, "kline_cache_ts", {}).get((sym, tf), 0.0))
                    if (
                        cached is not None
                        and not cached.empty
                        and len(cached) >= limit
                        and (time.time() - c_ts) <= max(180.0, tf_sec * 1.5)
                    ):
                        return cached.tail(limit).reset_index(drop=True).copy()
        return self.get_klines(sym, interval=tf, limit=limit)

    def _get_derivatives_context(self, symbol: str, price_df: Optional[pd.DataFrame] = None) -> tuple[Optional[float], Optional[dict]]:
        """
        Загружает Funding Rate Z-score и Open Interest метрики с Bybit v5 с TTL-кэшем (15 мин)
        для экономии REST-запросов и батареи в Termux.
        """
        sym = symbol.upper()
        now = time.time()
        ttl = float(getattr(config, "DERIVATIVES_CACHE_TTL_SEC", 900))
        if not hasattr(self, "_derivatives_cache"):
            self._derivatives_cache = {}

        fz: Optional[float] = None
        if getattr(config, "USE_FUNDING_ZSCORE_FILTER", True) and hasattr(self.client, "get_funding_rate_history"):
            f_key = ("funding", sym)
            cached_f = self._derivatives_cache.get(f_key)
            if cached_f and (now - cached_f[0]) < ttl:
                fz = cached_f[1]
            else:
                try:
                    win = int(getattr(config, "FUNDING_ZSCORE_WINDOW", 30))
                    resp_f = self._api_call(
                        self.client.get_funding_rate_history,
                        category="linear",
                        symbol=sym,
                        limit=max(40, win + 10),
                    )
                    rows = (resp_f or {}).get("result", {}).get("list", []) if isinstance(resp_f, dict) else []
                    if rows:
                        fdf = pd.DataFrame(rows)
                        fdf["timestamp"] = pd.to_datetime(fdf["fundingRateTimestamp"].astype("int64"), unit="ms")
                        fdf["fundingRate"] = fdf["fundingRate"].astype(float)
                        fdf = fdf.sort_values("timestamp").reset_index(drop=True)
                        fz = MarketRegimeFilter.compute_latest_funding_zscore(fdf, window=win)
                    self._derivatives_cache[f_key] = (now, fz)
                except Exception as e:
                    log.debug(f"[{sym}] Funding Z-score запрос пропущен: {e}")

        oi_metrics: Optional[dict] = None
        if getattr(config, "USE_OI_FILTER", True) and hasattr(self.client, "get_open_interest"):
            o_key = ("oi", sym)
            cached_o = self._derivatives_cache.get(o_key)
            if cached_o and (now - cached_o[0]) < ttl:
                oi_metrics = cached_o[1]
            else:
                try:
                    lb = int(getattr(config, "OI_LOOKBACK_PERIODS", 4))
                    iv = "4h" if str(BENCHMARK_INTERVAL) == "240" and sym == BENCHMARK_SYMBOL else "1h"
                    resp_o = self._api_call(
                        self.client.get_open_interest,
                        category="linear",
                        symbol=sym,
                        intervalTime=iv,
                        limit=max(20, lb + 10),
                    )
                    rows = (resp_o or {}).get("result", {}).get("list", []) if isinstance(resp_o, dict) else []
                    if rows:
                        odf = pd.DataFrame(rows)
                        odf["timestamp"] = pd.to_datetime(odf["timestamp"].astype("int64"), unit="ms")
                        odf["openInterest"] = odf["openInterest"].astype(float)
                        odf = odf.sort_values("timestamp").reset_index(drop=True)
                        oi_metrics = MarketRegimeFilter.compute_oi_metrics(odf, price_df=price_df, lookback=lb)
                    self._derivatives_cache[o_key] = (now, oi_metrics)
                except Exception as e:
                    log.debug(f"[{sym}] Open Interest запрос пропущен: {e}")

        return fz, oi_metrics

    def refresh_universe_if_needed(self, force: bool = False):
        """Периодически пересматривает ликвидную корзину пар, сохраняя открытые позиции."""
        if not getattr(config, "USE_DYNAMIC_UNIVERSE", False):
            return
        now = time.time()
        refresh_sec = float(getattr(config, "UNIVERSE_REFRESH_HOURS", 24.0)) * 3600.0
        if not force and (now - getattr(self, "_last_universe_refresh", 0.0)) < refresh_sec:
            return
        self._last_universe_refresh = now

        selector = getattr(self, "universe_selector", None)
        if selector is None:
            return
        ltf_tf = self._normalize_kline_interval(getattr(config, "INTERVAL", INTERVAL))[0]
        with self.lock:
            pinned = [s for s, p in self.positions.items() if p]
            sh_eng = getattr(self, "shadow_engine", None)
            if sh_eng is not None:
                for sh_s in list(sh_eng.positions.keys()):
                    if sh_s not in pinned:
                        pinned.append(sh_s)
            klines_map = {}
            for sym, df in getattr(self, "kline_cache", {}).items():
                if isinstance(sym, tuple) and sym[1] == ltf_tf:
                    klines_map[sym[0]] = df.copy()
            cur_active = list(getattr(self, "active_symbols", SYMBOLS))

        raw_new_syms = selector.select_symbols(
            self.client,
            fallback_symbols=cur_active,
            klines_map=klines_map,
            pinned_symbols=pinned,
            api_call_fn=self._safe_api_call,
        )
        new_syms = list(dict.fromkeys(str(s).strip().upper() for s in (raw_new_syms or []) if str(s).strip()))
        if not new_syms:
            new_syms = cur_active
        if new_syms != cur_active:
            membership_changed = set(new_syms) != {str(s).strip().upper() for s in cur_active}
            is_shadow_only = bool(getattr(config, "SHADOW_MODE", False))
            if not hasattr(self, "position_modes"):
                self.position_modes = {}
            for s in new_syms:
                if s not in self.instruments:
                    self._load_instrument_info(s)
                    if not is_shadow_only:
                        self._init_leverage(s)
                        self._detect_position_mode(s)
                    else:
                        self.position_modes[s] = "MergedSingle"
            with self.lock:
                self.active_symbols = new_syms
                if not hasattr(self, "last_handled_candles"):
                    self.last_handled_candles = {}
                for s in new_syms:
                    self.positions.setdefault(s, None)
                    self.risk_states.setdefault(s, None)
                    self.last_handled_candles.setdefault(s, None)
                    self.breakeven_applied.setdefault(s, False)
            if membership_changed:
                self._init_public_ws()
            self.save_state()

    def get_higher_trend(self, symbol: str) -> str:
        """Определяет старший тренд (4H EMA) с fail-safe защитой при сбое загрузки."""
        if not USE_MTF_FILTER or not getattr(self.strategy, "use_mtf_filter", True):
            return "neutral"
        try:
            df_htf = self.get_klines_fast(symbol, interval=HIGHER_INTERVAL, limit=max(600, HTF_EMA_PERIOD * 3))
            if (
                df_htf is None
                or df_htf.empty
                or len(df_htf) < HTF_EMA_PERIOD + 5
                or df_htf["close"].iloc[-2:].isna().any()
            ):
                log.warning(f"[{symbol}] Недостаточно данных 4H для MTF-фильтра (fail-safe).")
                return "unavailable"
            htf_idx = self._closed_candle_idx(symbol, HIGHER_INTERVAL, df_htf) if hasattr(self, "_closed_candle_idx") else -2
            try:
                return self.strategy.get_mtf_trend(df_htf, candle_idx=htf_idx)
            except TypeError:
                return self.strategy.get_mtf_trend(df_htf)
        except Exception as e:
            log.warning(f"[{symbol}] Ошибка старшего тренда: {e}")
            return "unavailable"

    def manage_risk_and_stops(self, symbol: str, curr_price: float, high_price: float, low_price: float):
        """
        Управляет рисками активной позиции:
        1. Частичная фиксация прибыли (Partial TP)
        2. Time-Stop при застое позиции (если не дошла до TP1)
        3. Гарантированный перенос стопа в безубыток при достижении BREAKEVEN_TRIGGER
        4. Трейлинг-стоп (Trailing Stop)
        """
        with self.lock:
            pos = self.positions.get(symbol)
            risk = self.risk_states.get(symbol)
            if not pos or not risk:
                return

            if getattr(config, "SHADOW_MODE", False) and getattr(self, "shadow_engine", None) is not None:
                self.shadow_engine.on_tick(symbol, curr_price, high=high_price, low=low_price)
                self._sync_from_shadow_engine(symbol)
                return

            min_notional = float(self.instruments.get(symbol, {}).get("min_notional", MIN_ORDER_NOTIONAL))
            min_qty = float(self.instruments.get(symbol, {}).get("min_qty", 0.001))

            # 0. Time-Stop: проверка застоя позиции до TP1
            if getattr(config, "USE_TIME_STOP", True) and getattr(risk, "opened_at", 0.0) > 0:
                ts_act, ts_sl, ts_reason = risk.check_time_stop(
                    current_price=curr_price,
                    now_ts=time.time(),
                    max_hours=getattr(config, "TIME_STOP_HOURS", 30.0),
                    hard_hours=getattr(config, "TIME_STOP_HARD_HOURS", 36.0),
                    min_r_progress=getattr(config, "TIME_STOP_MIN_R", 0.3),
                    offset_pct=BREAKEVEN_OFFSET,
                )
                if ts_act == "close":
                    trade_log.info(f"[{symbol}] {ts_reason}. Закрываем застойную позицию.")
                    self.close_position(symbol, curr_price, reason=ts_reason)
                    return
                elif ts_act == "tighten_be":
                    rounded_ts_be = self.round_price(symbol, ts_sl)
                    cur_sl = pos.get("stop_loss", 0.0)
                    improved = (rounded_ts_be > cur_sl) if risk.is_long else (cur_sl == 0.0 or rounded_ts_be < cur_sl)
                    if improved:
                        try:
                            pos_idx = self.get_position_idx(symbol, "Buy" if risk.is_long else "Sell")
                            r_ts = self._api_call(
                                self.client.set_trading_stop,
                                category="linear",
                                symbol=symbol,
                                stopLoss=self.format_price(symbol, rounded_ts_be),
                                positionIdx=pos_idx,
                            )
                            if isinstance(r_ts, dict) and r_ts.get("retCode", 0) != 0 and not self._is_stop_not_modified(r_ts):
                                raise RuntimeError(f"retCode={r_ts.get('retCode')}: {r_ts.get('retMsg')}")
                            pos["stop_loss"] = rounded_ts_be
                            risk.stop_loss = rounded_ts_be
                            risk.breakeven_applied = True
                            risk.time_stop_tightened = True
                            self.breakeven_applied[symbol] = True
                            trade_log.info(f"[{symbol}] {ts_reason}: SL={rounded_ts_be}")
                            self._db_record_trade(
                                symbol=symbol,
                                action="БЕЗУБЫТОК",
                                side="Buy" if risk.is_long else "Sell",
                                strategy=getattr(risk, "strategy_tag", ""),
                                price=rounded_ts_be,
                                qty=risk.size,
                                pos_ref=getattr(risk, "pos_ref", ""),
                                is_profit=True,
                                details=f"{ts_reason}: SL={rounded_ts_be}",
                            )
                            self.save_state()
                        except Exception as e:
                            log.warning(f"[{symbol}] Time-Stop ошибка подтяжки БУ: {e}")

            # 1. Частичная фиксация прибыли (Partial TP)
            # Программное закрытие по маркету выполняется только если на бирже НЕ стоит активный лимитный ордер TP1
            if USE_PARTIAL_TP and not risk.tp1_executed:
                triggered, qty_to_close, be_sl = risk.check_partial_tp(
                    curr_price, ratio=PARTIAL_TP_RATIO, offset_pct=BREAKEVEN_OFFSET
                )
                if triggered:
                    can_market_close = True
                    if risk.tp1_order_id:
                        try:
                            ord_chk = self._safe_api_call(
                                self.client.get_open_orders,
                                category="linear", symbol=symbol, orderId=risk.tp1_order_id
                            )
                            open_list = ord_chk.get("result", {}).get("list", [])
                            if open_list:
                                can_market_close = False
                            else:
                                tp1_status = getattr(self, "ws_orders", {}).get(risk.tp1_order_id, {}).get("status", "")
                                if not tp1_status and hasattr(self.client, "get_order_history"):
                                    h_resp = self._safe_api_call(
                                        self.client.get_order_history,
                                        category="linear", symbol=symbol, orderId=risk.tp1_order_id
                                    )
                                    if isinstance(h_resp, dict):
                                        h_list = h_resp.get("result", {}).get("list", [])
                                        if h_list and isinstance(h_list, list):
                                            tp1_status = str(h_list[0].get("orderStatus", ""))

                                if tp1_status in ("Cancelled", "Rejected", "Deactivated"):
                                    log.info(f"[{symbol}] Лимитный TP1 ордер был отменен ({tp1_status}). Закрываем TP1 по рынку.")
                                    risk.tp1_order_id = ""
                                    can_market_close = True
                                else:
                                    risk.tp1_executed = True
                                    can_market_close = False
                                    if PARTIAL_TP_MOVE_BE and not risk.breakeven_applied:
                                        rounded_be = self.round_price(symbol, be_sl)
                                        pos_idx = self.get_position_idx(symbol, "Buy" if risk.is_long else "Sell")
                                        self._safe_api_call(
                                            self.client.set_trading_stop,
                                            category="linear",
                                            symbol=symbol,
                                            stopLoss=self.format_price(symbol, rounded_be),
                                            positionIdx=pos_idx
                                        )
                                        pos["stop_loss"] = rounded_be
                                        risk.stop_loss = rounded_be
                                        risk.breakeven_applied = True
                                        self.breakeven_applied[symbol] = True
                                        trade_log.info(f"[{symbol}] Стоп перенесен в безубыток после биржевого TP1: {rounded_be}")
                                    self.save_state()
                        except Exception as e:
                            log.debug(f"[{symbol}] Проверка биржевого TP1 ордера: {e}")
                            can_market_close = False

                    close_qty = self.round_qty(symbol, qty_to_close)
                    if can_market_close and close_qty >= min_qty and (close_qty * curr_price) >= min_notional:
                        try:
                            side = "Sell" if risk.is_long else "Buy"
                            pos_idx = self.get_position_idx(symbol, "Buy" if risk.is_long else "Sell")
                            tp1_mkt_resp = self._api_call(
                                self.client.place_order,
                                category="linear",
                                symbol=symbol,
                                side=side,
                                orderType="Market",
                                qty=self.format_qty(symbol, close_qty),
                                positionIdx=pos_idx,
                                reduceOnly=True,
                                timeInForce="GoodTillCancel"
                            )
                            if isinstance(tp1_mkt_resp, dict) and tp1_mkt_resp.get("retCode", 0) != 0:
                                raise RuntimeError(f"retCode={tp1_mkt_resp.get('retCode')}: {tp1_mkt_resp.get('retMsg')}")

                            tp1_mkt_oid = str(tp1_mkt_resp.get("result", {}).get("orderId") or "") if isinstance(tp1_mkt_resp, dict) else ""
                            risk.tp1_executed = True
                            risk.tp1_closed_qty = close_qty
                            risk.tp1_fill_price = curr_price
                            risk.size = round(max(0.0, risk.size - close_qty), 8)
                            pos["size"] = risk.size
                            if hasattr(self, "last_exec_prices"):
                                self.last_exec_prices.pop(symbol, None)
                            if hasattr(self, "last_exec_info"):
                                self.last_exec_info.pop(symbol, None)

                            trade_log.info(
                                f"[{symbol}] ЧАСТИЧНЫЙ ТЕЙК-ПРОФИТ (TP1 = {risk.r_mult:g}R): Закрыто {close_qty} @ {curr_price}"
                            )
                            self._db_record_trade(
                                symbol=symbol,
                                action="ЧАСТИЧНЫЙ TP",
                                side=side,
                                strategy=getattr(risk, "strategy_tag", ""),
                                price=curr_price,
                                qty=close_qty,
                                order_id=tp1_mkt_oid,
                                exec_type="MarketTP1",
                                pos_ref=getattr(risk, "pos_ref", ""),
                                is_profit=True,
                                details=f"TP1 ({risk.r_mult:g}R): закрыто {close_qty} @ {curr_price}",
                            )
                            if hasattr(self, "schedule_closed_pnl_sync"):
                                self.schedule_closed_pnl_sync(
                                    symbol,
                                    order_id=tp1_mkt_oid,
                                    margin_used=float(getattr(risk, "margin_used", 0.0) or 0.0),
                                )

                            # Перенос оставшейся части в безубыток
                            if PARTIAL_TP_MOVE_BE:
                                rounded_be = self.round_price(symbol, be_sl)
                                pos_idx = self.get_position_idx(symbol, "Buy" if risk.is_long else "Sell")
                                try:
                                    r_ts = self._api_call(
                                        self.client.set_trading_stop,
                                        category="linear",
                                        symbol=symbol,
                                        stopLoss=self.format_price(symbol, rounded_be),
                                        positionIdx=pos_idx
                                    )
                                    if isinstance(r_ts, dict) and r_ts.get("retCode", 0) != 0 and not self._is_stop_not_modified(r_ts):
                                        raise RuntimeError(f"retCode={r_ts.get('retCode')}: {r_ts.get('retMsg')}")
                                    pos["stop_loss"] = rounded_be
                                    risk.stop_loss = rounded_be
                                    risk.breakeven_applied = True
                                    self.breakeven_applied[symbol] = True
                                    trade_log.info(f"[{symbol}] Стоп перенесен в безубыток после TP1: {rounded_be}")
                                except Exception as be_e:
                                    if self._is_stop_not_modified(be_e):
                                        pos["stop_loss"] = rounded_be
                                        risk.stop_loss = rounded_be
                                        risk.breakeven_applied = True
                                        self.breakeven_applied[symbol] = True
                                    else:
                                        log.warning(f"[{symbol}] Ошибка переноса стопа в БУ после TP1: {be_e}")

                            self.save_state()

                        except Exception as e:
                            log.error(f"[{symbol}] Ошибка исполнения Partial TP: {e}")

            # 2. Безубыток: гарантированный перенос стопа при достижении триггера (>= 1.0R)
            if USE_BREAKEVEN and not risk.breakeven_applied:
                be_updated, new_sl = risk.check_breakeven(
                    current_price=curr_price,
                    trigger_pct=BREAKEVEN_TRIGGER,
                    offset_pct=BREAKEVEN_OFFSET,
                    r_mult=BREAKEVEN_R_MULT
                )
                if be_updated:
                    rounded_sl = self.round_price(symbol, new_sl)
                    current_sl = pos.get("stop_loss", 0.0)
                    should_update = (rounded_sl > current_sl) if risk.is_long else (current_sl == 0.0 or rounded_sl < current_sl)
                    if should_update:
                        try:
                            pos_idx = self.get_position_idx(symbol, "Buy" if risk.is_long else "Sell")
                            r_be = self._safe_api_call(
                                self.client.set_trading_stop,
                                category="linear",
                                symbol=symbol,
                                stopLoss=self.format_price(symbol, rounded_sl),
                                positionIdx=pos_idx
                            )
                            if isinstance(r_be, dict) and r_be.get("retCode", 0) != 0 and not self._is_stop_not_modified(r_be):
                                raise RuntimeError(f"retCode={r_be.get('retCode')}: {r_be.get('retMsg')}")
                            pos["stop_loss"] = rounded_sl
                            risk.stop_loss = rounded_sl
                            risk.breakeven_applied = True
                            self.breakeven_applied[symbol] = True
                            trade_log.info(f"[{symbol}] Безубыток активирован: SL перенесен на {rounded_sl}")
                            self._db_record_trade(
                                symbol=symbol,
                                action="БЕЗУБЫТОК",
                                side="Buy" if risk.is_long else "Sell",
                                strategy=getattr(risk, "strategy_tag", ""),
                                price=rounded_sl,
                                qty=risk.size,
                                pos_ref=getattr(risk, "pos_ref", ""),
                                is_profit=True,
                                details=f"Безубыток: SL перенесен на {rounded_sl}",
                            )
                            self.save_state()
                        except Exception as e:
                            if self._is_stop_not_modified(e):
                                pos["stop_loss"] = rounded_sl
                                risk.stop_loss = rounded_sl
                                risk.breakeven_applied = True
                                self.breakeven_applied[symbol] = True
                                self.save_state()
                            else:
                                log.warning(f"[{symbol}] Ошибка переноса стопа в безубыток: {e}")

            # 3. Трейлинг-стоп (Trailing Stop): программный fallback (если нативный трейлинг на бирже не активен, не в RANGE)
            if USE_TRAILING_STOP and not risk.native_trailing_active and getattr(risk, "entry_regime", "") != "RANGE":
                trail_updated, new_sl = risk.check_trailing_stop(
                    current_price=curr_price,
                    high_price=high_price,
                    low_price=low_price,
                    trigger_pct=TRAILING_STOP_TRIGGER,
                    atr_mult=TRAILING_STOP_ATR_MULT,
                    fallback_pct=TRAILING_STOP_FALLBACK,
                    r_mult=TRAILING_STOP_R_MULT
                )
                if trail_updated:
                    rounded_sl = self.round_price(symbol, new_sl)
                    current_sl = pos.get("stop_loss", 0.0)
                    should_update = (rounded_sl > current_sl) if risk.is_long else (current_sl == 0.0 or rounded_sl < current_sl)

                    if should_update:
                        try:
                            pos_idx = self.get_position_idx(symbol, "Buy" if risk.is_long else "Sell")
                            r_tr = self._safe_api_call(
                                self.client.set_trading_stop,
                                category="linear",
                                symbol=symbol,
                                stopLoss=self.format_price(symbol, rounded_sl),
                                positionIdx=pos_idx
                            )
                            if isinstance(r_tr, dict) and r_tr.get("retCode", 0) != 0 and not self._is_stop_not_modified(r_tr):
                                raise RuntimeError(f"retCode={r_tr.get('retCode')}: {r_tr.get('retMsg')}")
                            pos["stop_loss"] = rounded_sl
                            risk.stop_loss = rounded_sl
                            trade_log.info(
                                f"[{symbol}] Трейлинг-стоп подтянут к {rounded_sl} (High: {risk.highest_price}, Low: {risk.lowest_price})"
                            )
                            self._db_record_trade(
                                symbol=symbol,
                                action="ТРЕЙЛИНГ-СТОП",
                                side="Buy" if risk.is_long else "Sell",
                                strategy=getattr(risk, "strategy_tag", ""),
                                price=rounded_sl,
                                qty=risk.size,
                                pos_ref=getattr(risk, "pos_ref", ""),
                                is_profit=True,
                                details=f"Трейлинг-стоп подтянут к {rounded_sl}",
                            )
                            self.save_state()
                        except Exception as e:
                            if self._is_stop_not_modified(e):
                                pos["stop_loss"] = rounded_sl
                                risk.stop_loss = rounded_sl
                                self.save_state()
                            else:
                                log.warning(f"[{symbol}] Ошибка обновления трейлинг-стопа: {e}")

    def _fetch_order_fill_info(self, symbol: str, order_id: str, use_rest: bool = True) -> tuple[str, float, float]:
        """Получает актуальный статус, cumExecQty и avgPrice ордера из WS-кэша или REST history."""
        status = ""
        cum_qty = 0.0
        avg_price = 0.0

        ws_ord = getattr(self, "ws_orders", {}).get(order_id)
        if isinstance(ws_ord, dict):
            status = str(ws_ord.get("status", ""))
            cum_qty = float(ws_ord.get("cum_qty", 0.0) or 0.0)
            avg_price = float(ws_ord.get("avg_price", 0.0) or 0.0)

        if use_rest and hasattr(self, "client") and hasattr(self.client, "get_order_history"):
            try:
                hist_resp = self._safe_api_call(self.client.get_order_history, category="linear", symbol=symbol, orderId=order_id)
                if isinstance(hist_resp, dict):
                    h_list = hist_resp.get("result", {}).get("list", [])
                    if h_list and isinstance(h_list, list):
                        ord_info = h_list[0]
                        h_status = str(ord_info.get("orderStatus", ""))
                        h_cum = float(ord_info.get("cumExecQty") or 0.0)
                        h_avg = float(ord_info.get("avgPrice") or 0.0)
                        if h_cum >= cum_qty:
                            cum_qty = h_cum
                            if h_avg > 0:
                                avg_price = h_avg
                        if h_status:
                            status = h_status
            except Exception:
                pass

        return status, cum_qty, avg_price

    def open_position(self, symbol: str, side: str, price: float, atr_value: float = 0.0, entry_regime: str = ""):
        with self.lock:
            # 1. Проверка глобальной паузы из дашборда
            if is_bot_paused():
                log.info(f"[{symbol}] Бот на паузе (веб-интерфейс). Вход в позицию отменен.")
                return None

            # 2. Проверка Circuit Breaker
            if self.circuit_breaker_active:
                log.warning(f"[{symbol}] Circuit Breaker активен! Открытие позиций заблокировано.")
                return None

            # 3. Shadow / Paper Mode перехват (без отправки реальных ордеров на биржу)
            if getattr(config, "SHADOW_MODE", False) and getattr(self, "shadow_engine", None) is not None:
                bid_p, ask_p = None, None
                try:
                    t_r = self._safe_api_call(self.client.get_tickers, category="linear", symbol=symbol)
                    if isinstance(t_r, dict):
                        t_l = t_r.get("result", {}).get("list", [])
                        if t_l:
                            bid_p = float(t_l[0].get("bid1Price") or 0.0) or None
                            ask_p = float(t_l[0].get("ask1Price") or 0.0) or None
                except Exception:
                    pass
                ltf_tf = self._normalize_kline_interval(getattr(config, "INTERVAL", INTERVAL))[0]
                k_map = {
                    k[0]: v for k, v in getattr(self, "kline_cache", {}).items()
                    if isinstance(k, tuple) and k[1] == ltf_tf
                }
                sh_pos = self.shadow_engine.open_position(
                    symbol=symbol,
                    side=side,
                    price=price,
                    atr_value=atr_value,
                    bid=bid_p,
                    ask=ask_p,
                    klines_map=k_map,
                    entry_regime=entry_regime,
                    strategy_name=getattr(self.strategy, "name", config.ACTIVE_STRATEGY),
                )
                if sh_pos is not None:
                    self.positions[symbol] = dict(sh_pos)
                    self.risk_states[symbol] = self.shadow_engine.risk_states.get(symbol)
                    self.breakeven_applied[symbol] = False
                return sh_pos

            # 4. Проверка лимита открытых позиций и лимита в одном направлении
            open_count = self.get_open_positions_count()
            if open_count >= MAX_OPEN_POSITIONS:
                log.info(f"[{symbol}] Достигнут лимит позиций ({open_count}/{MAX_OPEN_POSITIONS}). Пропуск.")
                return None

            max_same_dir = getattr(config, "MAX_SAME_DIRECTION_POSITIONS", MAX_OPEN_POSITIONS)
            target_dir = "long" if side == "Buy" else "short"
            same_dir_cnt = sum(1 for p in self.positions.values() if p and p.get("position") == target_dir)
            if same_dir_cnt >= max_same_dir:
                log.info(
                    f"[{symbol}] Достигнут лимит позиций в одном направлении {target_dir.upper()} "
                    f"({same_dir_cnt}/{max_same_dir}). Пропуск."
                )
                return None

            if self.positions.get(symbol):
                return None

            balance = self.get_available_balance()
            if balance < 5.0:
                log.error(f"[{symbol}] Недостаточно средств: {balance:.2f} USDT")
                return None

            # Определяем equity для риск-сайзинга (не вызываем сеть, если тест без client)
            equity = balance
            if getattr(self, "last_equity_ok", False) and float(getattr(self, "last_known_equity", 0.0) or 0.0) > 0:
                equity = float(self.last_known_equity)
            elif hasattr(self, "client"):
                eq = self.get_wallet_equity()
                if getattr(self, "last_equity_ok", False) and eq > 0:
                    equity = eq

            try:
                # Расчет стопов (с поддержкой regime-aware стопов для RegimeSwitcher)
                stop_kw = dict(
                    stop_loss_pct=STOP_LOSS_PCT,
                    take_profit_pct=TAKE_PROFIT_PCT,
                    atr_sl_mult=ATR_SL_MULT,
                    atr_tp_mult=ATR_TP_MULT,
                    use_atr=USE_ATR_STOPS,
                )
                if entry_regime:
                    try:
                        stops = self.strategy.calculate_stops(price, side, atr_value=atr_value, regime=entry_regime, **stop_kw)
                    except TypeError:
                        stops = self.strategy.calculate_stops(price, side, atr_value=atr_value, **stop_kw)
                else:
                    stops = self.strategy.calculate_stops(price, side, atr_value=atr_value, **stop_kw)
                sl = self.round_price(symbol, stops["sl"])
                tp = self.round_price(symbol, stops["tp"])

                # Портфельный риск v3: Heat Cap, Correlation Scaling, MaxDD Throttle, Cooldown
                pr = getattr(self, "portfolio_risk", None)
                base_risk_capital = equity * RISK_PER_TRADE_PCT
                risk_scale_ratio = 1.0
                if pr is not None:
                    ltf_tf = self._normalize_kline_interval(getattr(config, "INTERVAL", INTERVAL))[0]
                    k_map = {
                        k[0]: v for k, v in getattr(self, "kline_cache", {}).items()
                        if isinstance(k, tuple) and k[1] == ltf_tf
                    }
                    budget_ok, adj_risk, budget_note = pr.check_entry_budget(
                        symbol=symbol,
                        is_long=(side == "Buy"),
                        proposed_risk_usdt=base_risk_capital,
                        equity=equity,
                        risk_states=getattr(self, "risk_states", {}),
                        open_positions=self.positions,
                        klines_map=k_map,
                    )
                    if not budget_ok or adj_risk <= 0:
                        log.info(f"[{symbol}] Вход отклонен портфельным риск-менеджером: {budget_note}")
                        return None
                    if base_risk_capital > 0:
                        risk_scale_ratio = min(1.0, adj_risk / base_risk_capital)
                    if budget_note and budget_note != "OK":
                        log.info(f"[{symbol}] Портфельный риск-скейлинг: {budget_note} (scale={risk_scale_ratio:.2f})")
                    base_risk_capital = adj_risk

                # Расчет объема позиции от equity с безопасным потолком <= свободного balance
                max_avail_margin = balance * 0.95
                if USE_RISK_BASED_SIZING and stops.get("sl_dist", 0) > 0:
                    risk_capital = base_risk_capital
                    start_day_bal = float(getattr(self, "start_day_balance", 0.0) or 0.0)
                    last_eq = float(getattr(self, "last_known_equity", 0.0) or equity)
                    if start_day_bal > 0 and getattr(self, "last_equity_ok", False):
                        remaining_daily_loss = max(0.0, (start_day_bal * MAX_DAILY_LOSS_PCT) - max(0.0, start_day_bal - last_eq))
                        if remaining_daily_loss < risk_capital * 0.4:
                            log.warning(
                                f"[{symbol}] Остаток дневного лимита убытка ({remaining_daily_loss:.2f}$) слишком мал "
                                f"для открытия новой сделки (риск {risk_capital:.2f}$). Вход пропущен."
                            )
                            return None
                        risk_capital = min(risk_capital, remaining_daily_loss * 0.8)
                    raw_qty = risk_capital / stops["sl_dist"]
                    target_cap = max(equity * POSITION_SIZE_PCT * risk_scale_ratio, min(max_avail_margin, 10.0))
                    max_margin = min(max_avail_margin, target_cap)
                    max_qty = (max_margin * LEVERAGE) / price
                    raw_qty = min(raw_qty, max_qty)
                    trade_margin = (raw_qty * price) / LEVERAGE
                else:
                    target_cap = max(equity * POSITION_SIZE_PCT * risk_scale_ratio, min(max_avail_margin, 10.0))
                    trade_margin = min(max_avail_margin, target_cap)
                    notional = trade_margin * LEVERAGE
                    raw_qty = notional / price

                qty = self.round_qty(symbol, raw_qty)

                inst_info = self.instruments.get(symbol, {})
                min_qty = float(inst_info.get("min_qty", 0.001))
                if qty < min_qty:
                    log.error(f"[{symbol}] Объем {qty} меньше минимального {min_qty}. Баланс: {balance:.1f} USDT.")
                    return None

                min_notional = float(inst_info.get("min_notional", MIN_ORDER_NOTIONAL))
                order_notional = qty * price
                if order_notional < min_notional:
                    log.warning(
                        f"[{symbol}] Стоимость ордера {order_notional:.2f} USDT меньше минимального "
                        f"нотионала {min_notional:.1f} USDT. Вход отменен."
                    )
                    return None

                if hasattr(self, "last_exec_prices"):
                    self.last_exec_prices.pop(symbol, None)
                if hasattr(self, "last_exec_info"):
                    self.last_exec_info.pop(symbol, None)

                pos_idx = self.get_position_idx(symbol, side)

                # Умное исполнение: Maker Post-Only + Maker Chase (amend_order) + проверка спреда
                executed = False
                needs_set_stop = False
                resp = None
                filled_qty = 0.0
                maker_avg_price = 0.0
                actual_entry_price = price
                entry_order_id = ""
                exec_type_label = "Market"
                max_spread = float(getattr(config, "MAX_ENTRY_SPREAD_PCT", 0.005))

                if USE_SMART_EXECUTION:
                    order_id = None
                    limit_price = price
                    try:
                        tickers_resp = self._safe_api_call(self.client.get_tickers, category="linear", symbol=symbol)
                        t_info = tickers_resp["result"]["list"][0]
                        bid = float(t_info.get("bid1Price") or price)
                        ask = float(t_info.get("ask1Price") or price)
                        if bid > 0 and ask >= bid:
                            init_spread = (ask - bid) / bid
                            if init_spread > max_spread:
                                log.warning(
                                    f"[{symbol}] Аномально широкий спред {init_spread*100:.2f}% > "
                                    f"{max_spread*100:.2f}%. Вход пропущен."
                                )
                                return None

                        limit_price = self.round_price(symbol, bid if side == "Buy" else ask)
                        limit_p_str = self.format_price(symbol, limit_price)

                        maker_resp = self._api_call(
                            self.client.place_order,
                            category="linear",
                            symbol=symbol,
                            side=side,
                            orderType="Limit",
                            price=limit_p_str,
                            qty=self.format_qty(symbol, qty),
                            positionIdx=pos_idx,
                            timeInForce="PostOnly",
                            reduceOnly=False
                        )
                        if isinstance(maker_resp, dict) and maker_resp.get("retCode", 0) != 0:
                            raise RuntimeError(f"PostOnly retCode={maker_resp.get('retCode')}: {maker_resp.get('retMsg')}")
                        order_id = maker_resp.get("result", {}).get("orderId") if isinstance(maker_resp, dict) else None
                        if order_id:
                            entry_order_id = str(order_id)

                        if order_id:
                            trade_log.info(
                                f"[{symbol}] Maker ордер (Limit Post-Only) выставлен: {limit_p_str}. "
                                f"Ожидание исполнения (до {SMART_EXECUTION_TIMEOUT}с)..."
                            )
                            t0 = time.time()
                            chase_count = 0
                            last_chase_ts = t0
                            last_rest_poll_ts = 0.0
                            max_chase_attempts = int(getattr(config, "MAX_MAKER_CHASE_ATTEMPTS", 2))
                            chase_interval = float(getattr(config, "MAKER_CHASE_INTERVAL_SEC", 3.0))
                            max_chase_pct = float(getattr(config, "MAX_MAKER_CHASE_PCT", 0.004))
                            max_chase_atr = float(getattr(config, "MAX_MAKER_CHASE_ATR_MULT", 0.4))
                            max_drift = price * max_chase_pct
                            if atr_value > 0:
                                max_drift = min(max_drift, atr_value * max_chase_atr)

                            while time.time() - t0 < SMART_EXECUTION_TIMEOUT:
                                self.lock.release()
                                try:
                                    time.sleep(0.5)
                                finally:
                                    self.lock.acquire()

                                if is_bot_paused() or self.circuit_breaker_active or self.positions.get(symbol):
                                    log.info(f"[{symbol}] Ожидание Maker прервано (пауза/CB/позиция уже активна).")
                                    break

                                ws_priv = getattr(self, "ws_private", None)
                                ws_priv_alive = ws_priv is not None and not getattr(ws_priv, "exited", False)
                                now_poll = time.time()
                                do_rest_poll = (not ws_priv_alive) or ((now_poll - last_rest_poll_ts) >= 1.5)
                                if do_rest_poll:
                                    last_rest_poll_ts = now_poll

                                ws_st, ws_cum, ws_avg = self._fetch_order_fill_info(symbol, order_id, use_rest=do_rest_poll)
                                if ws_st == "Filled":
                                    executed = True
                                    needs_set_stop = True
                                    filled_qty = ws_cum if ws_cum > 0 else qty
                                    maker_avg_price = ws_avg if ws_avg > 0 else limit_price
                                    actual_entry_price = maker_avg_price
                                    resp = maker_resp
                                    exec_type_label = "Maker"
                                    trade_log.info(f"[{symbol}] Maker ордер исполнен (Filled: {filled_qty})!")
                                    break
                                if ws_cum >= qty * 0.95:
                                    try:
                                        self._api_call(self.client.cancel_order, category="linear", symbol=symbol, orderId=order_id)
                                    except Exception:
                                        pass
                                    _, final_cum, final_avg = self._fetch_order_fill_info(symbol, order_id, use_rest=True)
                                    filled_qty = final_cum if final_cum > 0 else ws_cum
                                    maker_avg_price = final_avg if final_avg > 0 else (ws_avg or limit_price)
                                    qty = self.round_qty(symbol, filled_qty)
                                    actual_entry_price = maker_avg_price
                                    executed = True
                                    needs_set_stop = True
                                    resp = maker_resp
                                    exec_type_label = "Maker"
                                    trade_log.info(f"[{symbol}] Maker >=95% исполнен ({filled_qty}), остаток снят!")
                                    break
                                if ws_st in ("Cancelled", "Rejected", "Deactivated"):
                                    filled_qty = ws_cum
                                    maker_avg_price = ws_avg if ws_avg > 0 else limit_price
                                    log.info(
                                        f"[{symbol}] Maker Post-Only завершен ({ws_st}, "
                                        f"исполнено {filled_qty}/{qty}). Переход на Market."
                                    )
                                    break

                                # Умный Maker Chase через amend_order вслед за bid1/ask1 в рамках max_drift
                                now_c = time.time()
                                if (
                                    getattr(config, "USE_MAKER_CHASE", True)
                                    and SMART_EXECUTION_TIMEOUT >= 2.0
                                    and chase_count < max_chase_attempts
                                    and (now_c - last_chase_ts) >= chase_interval
                                    and hasattr(self.client, "amend_order")
                                ):
                                    last_chase_ts = now_c
                                    try:
                                        t_chk = self._safe_api_call(self.client.get_tickers, category="linear", symbol=symbol)
                                        if isinstance(t_chk, dict):
                                            tl = t_chk.get("result", {}).get("list", [])
                                            if tl:
                                                nbid = float(tl[0].get("bid1Price") or 0.0)
                                                nask = float(tl[0].get("ask1Price") or 0.0)
                                                if nbid > 0 and nask >= nbid and ((nask - nbid) / nbid) <= max_spread:
                                                    target_p = self.round_price(symbol, nbid if side == "Buy" else nask)
                                                    drift_ok = (
                                                        (target_p - price <= max_drift)
                                                        if side == "Buy"
                                                        else (price - target_p <= max_drift)
                                                    )
                                                    improves = (
                                                        (target_p > limit_price)
                                                        if side == "Buy"
                                                        else (target_p < limit_price)
                                                    )
                                                    if target_p > 0 and drift_ok and improves:
                                                        am_resp = self._api_call(
                                                            self.client.amend_order,
                                                            category="linear",
                                                            symbol=symbol,
                                                            orderId=order_id,
                                                            price=self.format_price(symbol, target_p),
                                                        )
                                                        if not isinstance(am_resp, dict) or am_resp.get("retCode", 0) == 0:
                                                            limit_price = target_p
                                                            chase_count += 1
                                                            trade_log.info(
                                                                f"[{symbol}] Maker Chase #{chase_count}: цена переставлена на {self.format_price(symbol, target_p)}"
                                                            )
                                    except Exception as ame:
                                        log.debug(f"[{symbol}] Maker Chase amend_order: {ame}")

                                if do_rest_poll:
                                    try:
                                        ord_chk = self._safe_api_call(self.client.get_open_orders, category="linear", symbol=symbol, orderId=order_id)
                                        open_list = ord_chk.get("result", {}).get("list", []) if isinstance(ord_chk, dict) else []
                                        if not open_list:
                                            hist_status, hist_cum, hist_avg = self._fetch_order_fill_info(symbol, order_id, use_rest=True)
                                            if hist_status == "Filled" or hist_cum >= qty * 0.95:
                                                executed = True
                                                needs_set_stop = True
                                                filled_qty = hist_cum if hist_cum > 0 else qty
                                                qty = self.round_qty(symbol, filled_qty)
                                                maker_avg_price = hist_avg if hist_avg > 0 else limit_price
                                                actual_entry_price = maker_avg_price
                                                resp = maker_resp
                                                exec_type_label = "Maker"
                                                trade_log.info(f"[{symbol}] Maker ордер подтвержден исполненным (Filled: {filled_qty})!")
                                                break
                                            elif hist_status:
                                                filled_qty = hist_cum
                                                maker_avg_price = hist_avg if hist_avg > 0 else limit_price
                                                log.info(
                                                    f"[{symbol}] Maker Post-Only завершен ({hist_status}, "
                                                    f"исполнено {filled_qty}/{qty}). Добираем остаток по Market."
                                                )
                                                break
                                            elif not hasattr(self.client, "get_order_history"):
                                                executed = True
                                                needs_set_stop = True
                                                actual_entry_price = limit_price
                                                resp = maker_resp
                                                exec_type_label = "Maker"
                                                break
                                    except Exception:
                                        pass

                            if not executed:
                                trade_log.info(f"[{symbol}] Maker таймаут/отмена. Отмена остатка и проверка факта исполнения...")
                                try:
                                    self._api_call(self.client.cancel_order, category="linear", symbol=symbol, orderId=order_id)
                                except Exception as ce:
                                    log.debug(f"[{symbol}] cancel_order после таймаута Maker: {ce}")

                                time.sleep(0.15)
                                f_st, f_cum, f_avg = self._fetch_order_fill_info(symbol, order_id, use_rest=True)
                                if f_cum > filled_qty:
                                    filled_qty = f_cum
                                if f_avg > 0:
                                    maker_avg_price = f_avg

                                if hasattr(self.client, "get_positions"):
                                    try:
                                        p_chk = self._safe_api_call(self.client.get_positions, category="linear", symbol=symbol)
                                        if isinstance(p_chk, dict):
                                            for lp in p_chk.get("result", {}).get("list", []):
                                                lsz = float(lp.get("size") or 0.0)
                                                if lsz > 0 and int(lp.get("positionIdx", 0)) == pos_idx and lp.get("side") == side:
                                                    if lsz > filled_qty:
                                                        filled_qty = lsz
                                                    l_avg = float(lp.get("avgPrice") or 0.0)
                                                    if l_avg > 0:
                                                        maker_avg_price = l_avg
                                    except Exception:
                                        pass

                                if f_st == "Filled" or filled_qty >= qty * 0.95:
                                    executed = True
                                    needs_set_stop = True
                                    qty = self.round_qty(symbol, filled_qty if filled_qty > 0 else qty)
                                    actual_entry_price = maker_avg_price if maker_avg_price > 0 else limit_price
                                    resp = maker_resp
                                    exec_type_label = "Maker"
                                    trade_log.info(f"[{symbol}] Maker ордер фактически уже исполнен перед отменой ({qty})! Двойной вход предотвращен.")
                    except Exception as e:
                        log.info(f"[{symbol}] Post-Only отклонен ({e}). Переход на Market ордер.")

                # Если пока ждали Maker, бот встал на паузу/CB — добор по Market НЕ делаем
                if not executed and (is_bot_paused() or self.circuit_breaker_active or self.positions.get(symbol)):
                    if filled_qty >= min_qty and not self.positions.get(symbol):
                        executed = True
                        needs_set_stop = True
                        qty = self.round_qty(symbol, filled_qty)
                        actual_entry_price = maker_avg_price if maker_avg_price > 0 else price
                        exec_type_label = "Maker"
                    else:
                        return None

                # Проверка текущего спреда перед Market-фоллбэком
                if not executed and USE_SMART_EXECUTION and hasattr(self.client, "get_tickers"):
                    try:
                        t_fb = self._safe_api_call(self.client.get_tickers, category="linear", symbol=symbol)
                        if isinstance(t_fb, dict):
                            tl_fb = t_fb.get("result", {}).get("list", [])
                            if isinstance(tl_fb, list) and tl_fb and isinstance(tl_fb[0], dict):
                                fb_bid = float(tl_fb[0].get("bid1Price") or 0.0)
                                fb_ask = float(tl_fb[0].get("ask1Price") or 0.0)
                                if fb_bid > 0 and fb_ask >= fb_bid:
                                    fb_spread = (fb_ask - fb_bid) / fb_bid
                                    if fb_spread > max_spread:
                                        log.warning(
                                            f"[{symbol}] Аномальный спред {fb_spread*100:.2f}% > {max_spread*100:.2f}% "
                                            f"перед Market-фоллбэком."
                                        )
                                        if filled_qty >= min_qty:
                                            executed = True
                                            needs_set_stop = True
                                            qty = self.round_qty(symbol, filled_qty)
                                            actual_entry_price = maker_avg_price if maker_avg_price > 0 else price
                                            exec_type_label = "Maker"
                                        else:
                                            return None
                    except Exception:
                        pass

                # Fallback на Market ордер (на весь объем или неисполненный остаток)
                if not executed:
                    rem_qty = self.round_qty(symbol, max(0.0, qty - filled_qty))
                    if rem_qty >= min_qty and (rem_qty * price) >= min_notional:
                        mkt_ok = False
                        try:
                            resp = self._api_call(
                                self.client.place_order,
                                category="linear",
                                symbol=symbol,
                                side=side,
                                orderType="Market",
                                qty=self.format_qty(symbol, rem_qty),
                                stopLoss=self.format_price(symbol, sl),
                                takeProfit=self.format_price(symbol, tp),
                                positionIdx=pos_idx,
                                timeInForce="GoodTillCancel",
                                reduceOnly=False,
                                closeOnTrigger=False,
                            )
                            if isinstance(resp, dict) and resp.get("retCode", 0) != 0:
                                log.error(f"[{symbol}] Биржа отклонила Market ордер (retCode={resp.get('retCode')}): {resp.get('retMsg')}")
                            else:
                                mkt_ok = True
                        except Exception as me:
                            log.error(f"[{symbol}] Ошибка отправки Market ордера: {me}")

                        if mkt_ok:
                            mkt_oid = resp.get("result", {}).get("orderId", "") if isinstance(resp, dict) else ""
                            if mkt_oid and not entry_order_id:
                                entry_order_id = str(mkt_oid)
                            mkt_avg = 0.0
                            mkt_avg_from_pos = False
                            if mkt_oid:
                                time.sleep(0.15)
                                _, _, mkt_avg = self._fetch_order_fill_info(symbol, mkt_oid, use_rest=True)
                            if mkt_avg <= 0 and hasattr(self.client, "get_positions"):
                                try:
                                    p_chk = self._safe_api_call(self.client.get_positions, category="linear", symbol=symbol)
                                    if isinstance(p_chk, dict):
                                        for lp in p_chk.get("result", {}).get("list", []):
                                            if float(lp.get("size") or 0.0) > 0 and int(lp.get("positionIdx", 0)) == pos_idx:
                                                pos_avg = float(lp.get("avgPrice") or 0.0)
                                                if pos_avg > 0:
                                                    mkt_avg = pos_avg
                                                    mkt_avg_from_pos = True
                                                break
                                except Exception:
                                    pass
                            if mkt_avg <= 0:
                                ws_p = getattr(self, "ws_prices", {}).get(symbol)
                                mkt_avg = float(ws_p) if isinstance(ws_p, (int, float)) and ws_p > 0 else price

                            total_q = self.round_qty(symbol, filled_qty + rem_qty)
                            if mkt_avg_from_pos:
                                actual_entry_price = mkt_avg
                            elif filled_qty > 0 and maker_avg_price > 0 and total_q > 0:
                                actual_entry_price = ((filled_qty * maker_avg_price) + (rem_qty * mkt_avg)) / (filled_qty + rem_qty)
                            else:
                                actual_entry_price = mkt_avg
                            qty = total_q
                            executed = True
                            exec_type_label = "Maker+Market" if filled_qty > 0 else "Market"
                            if filled_qty > 0:
                                needs_set_stop = True
                        else:
                            if filled_qty >= min_qty:
                                qty = self.round_qty(symbol, filled_qty)
                                actual_entry_price = maker_avg_price if maker_avg_price > 0 else price
                                executed = True
                                needs_set_stop = True
                                exec_type_label = "Maker"
                                log.warning(f"[{symbol}] Добор Market не удался, регистрируем частичное Maker-исполнение ({qty}) и ставим SL/TP.")
                            else:
                                return None
                    elif filled_qty >= min_qty:
                        executed = True
                        needs_set_stop = True
                        qty = self.round_qty(symbol, filled_qty)
                        actual_entry_price = maker_avg_price if maker_avg_price > 0 else price
                        exec_type_label = "Maker"
                    else:
                        return None

                if isinstance(resp, dict) and resp.get("retCode", 0) != 0 and not executed:
                    log.error(f"[{symbol}] Биржа отклонила ордер (retCode={resp.get('retCode')}): {resp.get('retMsg')}")
                    return None

                # Пересчет SL/TP от реальной цены входа (avgPrice)
                if actual_entry_price > 0 and price > 0 and abs(actual_entry_price - price) / price > 1e-6:
                    stop_kw2 = dict(
                        stop_loss_pct=STOP_LOSS_PCT,
                        take_profit_pct=TAKE_PROFIT_PCT,
                        atr_sl_mult=ATR_SL_MULT,
                        atr_tp_mult=ATR_TP_MULT,
                        use_atr=USE_ATR_STOPS,
                    )
                    if entry_regime:
                        try:
                            stops = self.strategy.calculate_stops(actual_entry_price, side, atr_value=atr_value, regime=entry_regime, **stop_kw2)
                        except TypeError:
                            stops = self.strategy.calculate_stops(actual_entry_price, side, atr_value=atr_value, **stop_kw2)
                    else:
                        stops = self.strategy.calculate_stops(actual_entry_price, side, atr_value=atr_value, **stop_kw2)
                    new_sl = self.round_price(symbol, stops["sl"])
                    new_tp = self.round_price(symbol, stops["tp"])
                    if new_sl != sl or new_tp != tp:
                        sl, tp = new_sl, new_tp
                        needs_set_stop = True

                trade_margin = (qty * actual_entry_price) / LEVERAGE

                if executed or needs_set_stop:
                    stop_set_ok = False
                    for attempt in range(3):
                        try:
                            ts_resp = self._api_call(
                                self.client.set_trading_stop,
                                category="linear",
                                symbol=symbol,
                                positionIdx=pos_idx,
                                stopLoss=self.format_price(symbol, sl),
                                takeProfit=self.format_price(symbol, tp)
                            )
                            if not isinstance(ts_resp, dict) or ts_resp.get("retCode", 0) == 0 or self._is_stop_not_modified(ts_resp):
                                stop_set_ok = True
                                break
                            log.warning(f"[{symbol}] set_trading_stop retCode={ts_resp.get('retCode')}: {ts_resp.get('retMsg')}")
                        except Exception as e:
                            if self._is_stop_not_modified(e):
                                stop_set_ok = True
                                break
                            log.warning(f"[{symbol}] Ошибка выставления SL/TP (попытка {attempt+1}/3): {e}")
                        time.sleep(0.5)

                    if not stop_set_ok:
                        log.critical(f"[{symbol}] Не удалось выставить SL/TP на бирже! Экстренно закрываем незащищенную позицию.")
                        try:
                            close_side = "Sell" if side == "Buy" else "Buy"
                            self._api_call(
                                self.client.place_order,
                                category="linear", symbol=symbol, side=close_side,
                                orderType="Market", qty=self.format_qty(symbol, qty),
                                positionIdx=pos_idx, reduceOnly=True
                            )
                        except Exception as ce:
                            log.critical(f"[{symbol}] Критическая ошибка экстренного закрытия незащищенной позиции: {ce}")
                        return None

                self.positions[symbol] = {
                    "side": side,
                    "position": "long" if side == "Buy" else "short",
                    "size": qty,
                    "entry_price": actual_entry_price,
                    "stop_loss": sl,
                    "take_profit": tp,
                    "position_idx": pos_idx
                }
                self.breakeven_applied[symbol] = False

                opened_ts = time.time()
                strat_tag = getattr(self.strategy, "name", ACTIVE_STRATEGY) if hasattr(self, "strategy") else ACTIVE_STRATEGY
                pos_ref = f"{symbol}_{int(opened_ts)}"

                risk_state = PositionRiskState(
                    symbol=symbol,
                    side=side,
                    entry_price=actual_entry_price,
                    size=qty,
                    initial_size=qty,
                    stop_loss=sl,
                    take_profit=tp,
                    atr_value=atr_value,
                    r_mult=PARTIAL_TP_R_MULT,
                    position_idx=pos_idx,
                    opened_at=opened_ts,
                    strategy_tag=strat_tag,
                    entry_regime=entry_regime,
                    entry_order_id=entry_order_id,
                    margin_used=trade_margin,
                    pos_ref=pos_ref,
                )
                self.risk_states[symbol] = risk_state

                # Выставление лимитного TP1 (округленного по tickSize!) на стороне биржи (reduceOnly)
                if USE_PARTIAL_TP and USE_EXCHANGE_TP1:
                    tp1_qty = self.round_qty(symbol, qty * PARTIAL_TP_RATIO)
                    tp1_rounded_price = self.round_price(symbol, risk_state.tp1_price)
                    risk_state.tp1_price = tp1_rounded_price
                    tp1_side = "Sell" if side == "Buy" else "Buy"
                    if tp1_qty >= min_qty and tp1_rounded_price > 0 and (tp1_qty * tp1_rounded_price) >= min_notional:
                        try:
                            tp1_resp = self._api_call(
                                self.client.place_order,
                                category="linear",
                                symbol=symbol,
                                side=tp1_side,
                                orderType="Limit",
                                price=self.format_price(symbol, tp1_rounded_price),
                                qty=self.format_qty(symbol, tp1_qty),
                                positionIdx=pos_idx,
                                reduceOnly=True,
                                timeInForce="GTC"
                            )
                            if isinstance(tp1_resp, dict) and tp1_resp.get("retCode", 0) != 0:
                                log.warning(
                                    f"[{symbol}] Биржа отклонила лимитный TP1 (retCode={tp1_resp.get('retCode')}): "
                                    f"{tp1_resp.get('retMsg')}"
                                )
                            else:
                                risk_state.tp1_order_id = tp1_resp.get("result", {}).get("orderId", "") if isinstance(tp1_resp, dict) else ""
                                trade_log.info(
                                    f"[{symbol}] На бирже выставлен лимитный ордер TP1 (reduceOnly): {tp1_qty} @ {self.format_price(symbol, tp1_rounded_price)}"
                                )
                        except Exception as e:
                            log.warning(f"[{symbol}] Не удалось выставить лимитный TP1 на бирже: {e}")

                # Активация нативного трейлинг-стопа Bybit (кроме RANGE Mean-Reversion режима)
                if USE_TRAILING_STOP and USE_NATIVE_TRAILING_STOP and getattr(risk_state, "entry_regime", entry_regime) != "RANGE":
                    trail_dist = self.round_price(
                        symbol,
                        atr_value * TRAILING_STOP_ATR_MULT if atr_value > 0 else (actual_entry_price * TRAILING_STOP_FALLBACK)
                    )
                    eff_trail_trigger = TRAILING_STOP_TRIGGER
                    if risk_state.risk_r > 0 and actual_entry_price > 0 and TRAILING_STOP_R_MULT > 0:
                        eff_trail_trigger = max(
                            TRAILING_STOP_TRIGGER,
                            (risk_state.risk_r * TRAILING_STOP_R_MULT) / actual_entry_price
                        )
                    active_price = self.round_price(
                        symbol,
                        actual_entry_price * (1 + eff_trail_trigger if side == "Buy" else 1 - eff_trail_trigger)
                    )
                    if trail_dist > 0 and active_price > 0:
                        try:
                            nt_resp = self._api_call(
                                self.client.set_trading_stop,
                                category="linear",
                                symbol=symbol,
                                positionIdx=pos_idx,
                                trailingStop=self.format_price(symbol, trail_dist),
                                activePrice=self.format_price(symbol, active_price)
                            )
                            if isinstance(nt_resp, dict) and nt_resp.get("retCode", 0) != 0:
                                log.warning(f"[{symbol}] Биржа отклонила нативный трейлинг (retCode={nt_resp.get('retCode')}): {nt_resp.get('retMsg')}")
                            else:
                                risk_state.native_trailing_active = True
                                trade_log.info(
                                    f"[{symbol}] Нативный трейлинг Bybit активирован: дист={trail_dist}, активация={active_price}"
                                )
                        except Exception as e:
                            log.warning(f"[{symbol}] Биржа отклонила нативный трейлинг ({e}). Работает программный.")

                self.save_state()

                direction_lbl = "LONG" if side == "Buy" else "SHORT"
                trade_log.info(
                    f"[{symbol}] {direction_lbl} открыт | "
                    f"Вход: {actual_entry_price} | Кол-во: {qty} ({trade_margin:.1f}$ маржа) | "
                    f"SL: {sl} | TP: {tp} [{stops['type']}] | x{LEVERAGE}"
                )
                self._db_record_trade(
                    symbol=symbol,
                    action=f"ОТКРЫТИЕ {direction_lbl}",
                    side=side,
                    strategy=strat_tag,
                    price=actual_entry_price,
                    qty=qty,
                    order_id=entry_order_id,
                    exec_type=exec_type_label,
                    pos_ref=pos_ref,
                    is_profit=None,
                    details=f"{direction_lbl} ({exec_type_label}) | SL: {sl} | TP: {tp} | {stops['type']}",
                    ts=opened_ts,
                )
                return resp

            except Exception as e:
                log.error(f"[{symbol}] Ошибка открытия позиции: {e}")
                return None

    def close_position(self, symbol: str, price: float, reason: str = "Сигнал"):
        with self.lock:
            curr_pos = self.positions.get(symbol)
            if not curr_pos:
                return None

            if getattr(config, "SHADOW_MODE", False) and getattr(self, "shadow_engine", None) is not None:
                res = self.shadow_engine.close_position(symbol, price, reason=reason)
                self.positions[symbol] = None
                if hasattr(self, "risk_states"):
                    self.risk_states[symbol] = None
                if hasattr(self, "breakeven_applied"):
                    self.breakeven_applied[symbol] = False
                return res

            cur_risk = getattr(self, "risk_states", {}).get(symbol)
            try:
                tp1_oid = getattr(cur_risk, "tp1_order_id", "") if cur_risk else ""
                if tp1_oid:
                    try:
                        self._api_call(self.client.cancel_order, category="linear", symbol=symbol, orderId=tp1_oid)
                    except Exception as e:
                        log.debug(f"[{symbol}] Снятие лимитного TP1 перед закрытием: {e}")

                side = "Sell" if curr_pos["position"] == "long" else "Buy"
                qty = curr_pos["size"]
                pos_idx = (
                    int(curr_pos["position_idx"])
                    if curr_pos.get("position_idx") is not None
                    else self.get_position_idx(symbol, curr_pos.get("side", "Buy"))
                )

                resp = self._api_call(
                    self.client.place_order,
                    category="linear",
                    symbol=symbol,
                    side=side,
                    orderType="Market",
                    qty=self.format_qty(symbol, qty),
                    positionIdx=pos_idx,
                    reduceOnly=True,
                    timeInForce="GoodTillCancel",
                )
                if isinstance(resp, dict) and resp.get("retCode", 0) != 0:
                    log.error(f"[{symbol}] Ошибка Market-закрытия (retCode={resp.get('retCode')}): {resp.get('retMsg')}")
                    return None

                try:
                    self._api_call(self.client.cancel_all_orders, category="linear", symbol=symbol)
                except Exception as e:
                    log.warning(f"[{symbol}] Ошибка очистки ордеров после закрытия: {e}")

                exit_price = float(price or 0.0)
                close_oid = resp.get("result", {}).get("orderId", "") if isinstance(resp, dict) else ""
                if close_oid:
                    _, _, fill_avg = self._fetch_order_fill_info(symbol, close_oid)
                    if fill_avg > 0:
                        exit_price = fill_avg

                entry = float(curr_pos.get("entry_price") or exit_price)
                is_short = (curr_pos.get("position") == "short")
                dir_sign = -1.0 if is_short else 1.0
                rem_pnl_pct = ((exit_price - entry) / entry) * dir_sign * 100.0 * LEVERAGE if entry > 0 else 0.0
                pnl_pct = rem_pnl_pct

                if cur_risk and cur_risk.tp1_executed and cur_risk.initial_size > 0:
                    init_sz = float(cur_risk.initial_size)
                    tp1_sz = float(cur_risk.tp1_closed_qty or max(0.0, init_sz - float(qty)))
                    tp1_p = float(cur_risk.tp1_fill_price or cur_risk.tp1_price or 0.0)
                    if 0 < tp1_sz < init_sz and float(qty) > 0 and tp1_p > 0 and entry > 0:
                        w1 = tp1_sz / init_sz
                        w2 = float(qty) / init_sz
                        tp1_pnl = ((tp1_p - entry) / entry) * dir_sign * 100.0 * LEVERAGE
                        pnl_pct = (w1 * tp1_pnl) + (w2 * rem_pnl_pct)

                sign = "+" if pnl_pct > 0 else ""
                trade_log.info(f"[{symbol}] Закрыто ({reason}) @ {exit_price} | PnL: {sign}{pnl_pct:.2f}%")

                pr = getattr(self, "portfolio_risk", None)
                if pr is not None:
                    pr.record_closed_trade(symbol, pnl_pct)

                margin_used = float(getattr(cur_risk, "margin_used", 0.0) or (float(qty) * entry / LEVERAGE))
                est_usdt = margin_used * (pnl_pct / 100.0) if margin_used > 0 else None
                action_name = "TIME_STOP" if "Time-Stop" in reason else "ЗАКРЫТИЕ"
                self._db_record_trade(
                    symbol=symbol,
                    action=action_name,
                    side=side,
                    strategy=getattr(cur_risk, "strategy_tag", "") if cur_risk else "",
                    price=exit_price,
                    qty=qty,
                    pnl_pct=pnl_pct,
                    net_pnl_usdt=est_usdt,
                    order_id=str(close_oid or ""),
                    exec_type="Market",
                    pos_ref=getattr(cur_risk, "pos_ref", "") if cur_risk else "",
                    is_profit=(pnl_pct > 0),
                    details=f"{reason} @ {exit_price} | PnL: {sign}{pnl_pct:.2f}%",
                )
                self.schedule_closed_pnl_sync(symbol, order_id=str(close_oid or ""), margin_used=margin_used)

                if hasattr(self, "last_exec_prices"):
                    self.last_exec_prices.pop(symbol, None)
                if hasattr(self, "last_exec_info"):
                    self.last_exec_info.pop(symbol, None)
                self.positions[symbol] = None
                if hasattr(self, "risk_states"):
                    self.risk_states[symbol] = None
                if hasattr(self, "breakeven_applied"):
                    self.breakeven_applied[symbol] = False
                self.save_state()
                return resp

            except Exception as e:
                log.error(f"[{symbol}] Ошибка закрытия: {e}")
                return None

    def panic_close_all(self):
        """Экстренно закрывает ВСЕ позиции (включая символы вне SYMBOLS и обе ноги Hedge Mode) и снимает ордера."""
        with self.lock:
            trade_log.warning("ВЫЗВАНА КОМАНДА PANIC CLOSE! Закрытие всех позиций...")
            is_shadow_only = bool(getattr(config, "SHADOW_MODE", False)) and getattr(self, "shadow_engine", None) is not None
            if is_shadow_only and hasattr(self, "_sync_from_shadow_engine"):
                self._sync_from_shadow_engine()

            sh_keys = list(getattr(self.shadow_engine, "positions", {}).keys()) if getattr(self, "shadow_engine", None) else []
            all_tracked = list(dict.fromkeys(list(SYMBOLS) + list(self.positions.keys()) + sh_keys))
            for sym in all_tracked:
                pos = self.positions.get(sym)
                if pos and pos.get("size", 0) > 0:
                    try:
                        try:
                            price = self.get_price(sym)
                        except Exception:
                            if is_shadow_only:
                                price = float(getattr(self, "ws_prices", {}).get(sym) or pos.get("entry_price") or 0.0)
                                if price <= 0:
                                    raise
                            else:
                                raise
                        self.close_position(sym, price, reason="PANIC BUTTON")
                    except Exception as e:
                        log.error(f"[{sym}] Ошибка Panic Close: {e}")

            if is_shadow_only:
                if hasattr(self, "save_state"):
                    self.save_state()
                return

            # Дозачистка всех оставшихся позиций и ордеров напрямую на бирже
            if hasattr(self, "client") and hasattr(self.client, "get_positions"):
                try:
                    resp = self._safe_api_call(self.client.get_positions, category="linear", settleCoin="USDT")
                    if isinstance(resp, dict):
                        for p in resp.get("result", {}).get("list", []):
                            sz = float(p.get("size") or 0.0)
                            sym = p.get("symbol")
                            if sz > 0 and sym:
                                close_side = "Sell" if p.get("side") == "Buy" else "Buy"
                                p_idx = int(p.get("positionIdx", 0))
                                raw_sz_str = str(p.get("size") or sz)
                                qty_str = self.format_qty(sym, sz) if sym in getattr(self, "instruments", {}) else raw_sz_str
                                try:
                                    r = self._api_call(
                                        self.client.place_order,
                                        category="linear", symbol=sym, side=close_side,
                                        orderType="Market", qty=qty_str,
                                        positionIdx=p_idx, reduceOnly=True
                                    )
                                    if not isinstance(r, dict) or r.get("retCode", 0) == 0:
                                        self._api_call(self.client.cancel_all_orders, category="linear", symbol=sym)
                                        self.positions[sym] = None
                                        if hasattr(self, "risk_states"):
                                            self.risk_states[sym] = None
                                        trade_log.warning(f"[{sym}] PANIC: Закрыта внесписочная/Hedge позиция ({qty_str}).")
                                except Exception as ex:
                                    log.error(f"[{sym}] Ошибка дозачистки Panic Close: {ex}")
                except Exception as e:
                    log.warning(f"Panic Close запрос открытых позиций на бирже: {e}")

            if hasattr(self, "client") and hasattr(self.client, "get_open_orders"):
                try:
                    ord_resp = self._safe_api_call(self.client.get_open_orders, category="linear", settleCoin="USDT")
                    if isinstance(ord_resp, dict):
                        syms_with_orders = {o.get("symbol") for o in ord_resp.get("result", {}).get("list", []) if o.get("symbol")}
                        for sym in syms_with_orders:
                            try:
                                self._api_call(self.client.cancel_all_orders, category="linear", symbol=sym)
                            except Exception:
                                pass
                except Exception:
                    pass
            if hasattr(self, "save_state"):
                self.save_state()

    def _sync_external_control(self):
        """Применяет изменения паузы/режима и shadow-команд из внешнего dashboard.py (через bot_control.json)."""
        ctrl = read_control_state()
        if not ctrl:
            return
        upd_ts = float(ctrl.get("updated_at") or 0.0)
        has_shadow_actions = bool(ctrl.get("pending_shadow_panic", False)) or bool(ctrl.get("pending_shadow_closes"))
        if upd_ts <= getattr(self, "_last_control_ts", 0.0) and not has_shadow_actions:
            return
        if upd_ts > getattr(self, "_last_control_ts", 0.0):
            self._last_control_ts = upd_ts

        if "is_paused" in ctrl and bool(ctrl["is_paused"]) != bool(getattr(self, "is_paused", False)):
            self.is_paused = bool(ctrl["is_paused"])
            set_bot_paused(self.is_paused, persist=False)
            log.info(f"IPC: Состояние паузы обновлено из дашборда -> {'ПАУЗА' if self.is_paused else 'АКТИВЕН'}")

        ext_mode = ctrl.get("trading_mode")
        if ext_mode in ("DEMO", "LIVE") and ext_mode != config.TRADING_MODE:
            prev = (config.TRADING_MODE, config.API_KEY, config.API_SECRET, config.DEMO, config.TESTNET)
            config.TRADING_MODE = ext_mode
            if ext_mode == "DEMO":
                config.API_KEY = config.DEMO_API_KEY
                config.API_SECRET = config.DEMO_API_SECRET
                config.DEMO = True
                config.TESTNET = False
            else:
                config.API_KEY = config.LIVE_API_KEY
                config.API_SECRET = config.LIVE_API_SECRET
                config.DEMO = False
                config.TESTNET = False
            if not self.reload_credentials():
                config.TRADING_MODE, config.API_KEY, config.API_SECRET, config.DEMO, config.TESTNET = prev
                self._last_control_ts = write_control_state(trading_mode=prev[0])

        if getattr(config, "SHADOW_MODE", False) and getattr(self, "shadow_engine", None) is not None and has_shadow_actions:
            pending_panic, pending_closes, consumed_ts = consume_shadow_control_actions()
            self._last_control_ts = max(getattr(self, "_last_control_ts", 0.0), consumed_ts)
            if pending_panic:
                self.panic_close_all()
                self._publish_live_status()
            elif pending_closes:
                for sym in pending_closes:
                    if sym and self.positions.get(sym):
                        try:
                            px = self.get_price(sym)
                        except Exception:
                            px = float(self.positions[sym].get("entry_price") or 0.0)
                        if px > 0:
                            self.close_position(sym, px, reason="Ручное закрытие через UI (IPC)")
                self._publish_live_status()

    def get_price(self, symbol: str, fallback_df=None) -> float:
        """
        Получает актуальную цену символа.
        1. WebSocket (кэш свежее 5 сек) — 0ms, мгновенно из памяти.
        2. REST API Bybit get_tickers.
        3. Fallback цена закрытия последней свечи.
        """
        now = time.time()
        if symbol in self.ws_prices and (now - self.ws_last_update.get(symbol, 0) < 5.0):
            return self.ws_prices[symbol]

        try:
            resp = self._safe_api_call(self.client.get_tickers, category="linear", symbol=symbol)
            p = float(resp["result"]["list"][0]["lastPrice"])
            self.ws_prices[symbol] = p
            self.ws_last_update[symbol] = now
            return p
        except Exception as e:
            if fallback_df is not None and not fallback_df.empty:
                last_p = float(fallback_df.iloc[-1]["close"])
                log.warning(f"[{symbol}] Ошибка цены: {e}. Используем fallback из kline: {last_p}")
                return last_p
            log.error(f"[{symbol}] Ошибка цены: {e}")
            raise

    def _evaluate_market_cycle(self):
        """Выполняет оценку рыночного режима и торговых сигналов по активным парам."""
        bench_df = None
        bench_ok = True
        bench_fz = None
        bench_oim = None
        bench_closed_idx = -2
        ltf_tf = self._normalize_kline_interval(getattr(config, "INTERVAL", INTERVAL))[0]
        htf_tf = self._normalize_kline_interval(getattr(config, "HIGHER_INTERVAL", HIGHER_INTERVAL))[0]
        bm_sym = str(getattr(config, "BENCHMARK_SYMBOL", BENCHMARK_SYMBOL) or BENCHMARK_SYMBOL).strip().upper()
        bm_tf = self._normalize_kline_interval(getattr(config, "BENCHMARK_INTERVAL", BENCHMARK_INTERVAL))[0]

        if USE_MARKET_REGIME_FILTER:
            bench_ok = False
            try:
                bench_df = self.get_klines_fast(
                    bm_sym,
                    interval=bm_tf,
                    limit=max(600, BENCHMARK_EMA_PERIOD * 3),
                )
                if bench_df is not None and not bench_df.empty and len(bench_df) >= max(BENCHMARK_EMA_PERIOD, 30):
                    bench_ok = True
                    bench_closed_idx = self._closed_candle_idx(bm_sym, bm_tf, bench_df) if hasattr(self, "_closed_candle_idx") else -2
                    bench_fz, bench_oim = self._get_derivatives_context(bm_sym, price_df=bench_df)
                    self._last_regime_snapshot = self.regime_filter.get_regime_snapshot(
                        bench_df, funding_zscore=bench_fz, oi_metrics=bench_oim, closed_idx=bench_closed_idx
                    )
                else:
                    log.warning(f"Данные бенчмарка {bm_sym} неполные — входы заблокированы (fail-safe).")
            except Exception as e:
                log.warning(f"Не удалось загрузить бенчмарк {bm_sym} ({e}) — входы заблокированы (fail-safe).")

        min_candles_req = max(SLOW_MA, ST_CONFIRM_EMA, ATR_PERIOD, RSI_PERIOD, 30) + 10
        kline_limit = max(250, min_candles_req + 50)
        watch_list = self._get_watch_symbols()

        for symbol in watch_list:
            try:
                df = self.get_klines_fast(symbol, limit=kline_limit)
                if df.empty or len(df) < min_candles_req:
                    continue

                df_calc = self.strategy.calculate_indicators(df)
                price = self.get_price(symbol, fallback_df=df_calc)

                # Передаем в риск-менеджер реальную текущую цену
                self.manage_risk_and_stops(symbol, price, price, price)

                sig_idx = self._closed_candle_idx(symbol, ltf_tf, df_calc) if hasattr(self, "_closed_candle_idx") else -2
                signal_candle = df_calc.iloc[sig_idx]
                signal = int(signal_candle.get("signal", 0))
                exit_signal = int(signal_candle.get("exit_signal", signal))
                signal_price = float(signal_candle["close"])
                signal_time = signal_candle["timestamp"]
                atr_value = float(signal_candle.get("ATR", 0.0) or 0.0)
                entry_regime = str(signal_candle.get("SIGNAL_REGIME", ""))
                current_pos = self.positions.get(symbol)
                current_risk = getattr(self, "risk_states", {}).get(symbol)

                if signal_time != self.last_handled_candles.get(symbol):
                    self.last_handled_candles[symbol] = signal_time

                    # 1. Приоритетное закрытие встречной позиции по exit_signal / Mean-Reversion BB_middle
                    if exit_signal == 1 and current_pos and current_pos.get("position") == "short":
                        self.close_position(symbol, price, "Смена тренда на LONG")
                        current_pos = None
                    elif exit_signal == -1 and current_pos and current_pos.get("position") == "long":
                        self.close_position(symbol, price, "Смена тренда на SHORT")
                        current_pos = None
                    elif current_pos and current_risk and getattr(current_risk, "entry_regime", "") == "RANGE":
                        if current_risk.is_long and bool(signal_candle.get("mr_exit_long", False)):
                            self.close_position(symbol, price, "Mean-Reversion цель BB_middle")
                            current_pos = None
                        elif (not current_risk.is_long) and bool(signal_candle.get("mr_exit_short", False)):
                            self.close_position(symbol, price, "Mean-Reversion цель BB_middle")
                            current_pos = None

                    # 2. Параллельная оценка Shadow A/B стратегии (если активна отдельная SHADOW_STRATEGY)
                    sh_eng = getattr(self, "shadow_engine", None)
                    if sh_eng is not None and not getattr(config, "SHADOW_MODE", False):
                        htf_df_sh = None
                        htf_idx_sh = -2
                        if getattr(sh_eng.strategy, "use_mtf_filter", False):
                            htf_df_sh = self.get_klines_fast(symbol, interval=htf_tf, limit=max(600, HTF_EMA_PERIOD * 3))
                            if htf_df_sh is not None and not htf_df_sh.empty and hasattr(self, "_closed_candle_idx"):
                                htf_idx_sh = self._closed_candle_idx(symbol, htf_tf, htf_df_sh)
                        def _sh_regime_ok(s_sym, s_sig):
                            if not USE_MARKET_REGIME_FILTER:
                                return True, "OK"
                            if not bench_ok:
                                return False, "no bench"
                            return (
                                self.regime_filter.is_long_allowed(
                                    s_sym, bench_df, funding_zscore=bench_fz, oi_metrics=bench_oim, closed_idx=bench_closed_idx
                                )
                                if s_sig == 1
                                else self.regime_filter.is_short_allowed(
                                    s_sym, bench_df, funding_zscore=bench_fz, oi_metrics=bench_oim, closed_idx=bench_closed_idx
                                )
                            )
                        k_map_sh = {
                            k[0]: v for k, v in getattr(self, "kline_cache", {}).items()
                            if isinstance(k, tuple) and k[1] == ltf_tf
                        }
                        sh_eng.process_candle_signal(
                            symbol=symbol,
                            df=df,
                            price=price,
                            htf_df=htf_df_sh,
                            klines_map=k_map_sh,
                            regime_allowed_fn=_sh_regime_ok,
                            candle_idx=sig_idx,
                            htf_candle_idx=htf_idx_sh,
                        )

                    # Если пара вне текущей активной корзины (осталась только под закрытие), новых входов не открываем
                    if symbol not in getattr(self, "active_symbols", SYMBOLS):
                        continue

                    # 3. Направленная защита от неблагоприятного проскальзывания цены для нового входа
                    if signal == 1 and signal_price > 0:
                        slippage = (price - signal_price) / signal_price
                    elif signal == -1 and signal_price > 0:
                        slippage = (signal_price - price) / signal_price
                    else:
                        slippage = 0.0

                    if signal != 0 and slippage > MAX_SLIPPAGE_PCT:
                        log.warning(
                            f"[{symbol}] Неблагоприятное проскальзывание {slippage*100:.2f}% > "
                            f"{MAX_SLIPPAGE_PCT*100:.1f}%. Вход пропущен."
                        )
                        continue

                    # 4. Проверка фильтра рыночного сентимента + деривативных данных (Funding Z-score + OI)
                    if signal != 0 and USE_MARKET_REGIME_FILTER:
                        if not bench_ok:
                            signal = 0
                        else:
                            sym_fz, sym_oim = self._get_derivatives_context(symbol, price_df=df_calc)
                            if signal == 1:
                                allowed, r_msg = self.regime_filter.is_long_allowed(
                                    symbol, bench_df,
                                    funding_zscore=bench_fz,
                                    oi_metrics=bench_oim,
                                    symbol_funding_zscore=sym_fz,
                                    symbol_oi_metrics=sym_oim,
                                    closed_idx=bench_closed_idx,
                                )
                                if not allowed:
                                    log.info(f"[{symbol}] {r_msg}")
                                    signal = 0
                            elif signal == -1:
                                allowed, r_msg = self.regime_filter.is_short_allowed(
                                    symbol, bench_df,
                                    funding_zscore=bench_fz,
                                    oi_metrics=bench_oim,
                                    symbol_funding_zscore=sym_fz,
                                    symbol_oi_metrics=sym_oim,
                                    closed_idx=bench_closed_idx,
                                )
                                if not allowed:
                                    log.info(f"[{symbol}] {r_msg}")
                                    signal = 0

                    # 5. Проверка фильтра старшего тренда (4H EMA), не применяется в RANGE режиме Switcher
                    use_mtf_here = (
                        USE_MTF_FILTER
                        and getattr(self.strategy, "use_mtf_filter", True)
                        and entry_regime != "RANGE"
                    )
                    htf_trend = self.get_higher_trend(symbol) if (use_mtf_here and signal != 0) else "neutral"

                    if signal == 1 and not self.positions.get(symbol):
                        if use_mtf_here and htf_trend in ("bearish", "unavailable"):
                            log.info(f"[{symbol}] Сигнал LONG отклонен: старший тренд 4H ({htf_trend})")
                        else:
                            self.open_position(symbol, "Buy", price, atr_value=atr_value, entry_regime=entry_regime)

                    elif signal == -1 and not self.positions.get(symbol):
                        if use_mtf_here and htf_trend in ("bullish", "unavailable"):
                            log.info(f"[{symbol}] Сигнал SHORT отклонен: старший тренд 4H ({htf_trend})")
                        else:
                            self.open_position(symbol, "Sell", price, atr_value=atr_value, entry_regime=entry_regime)

                time.sleep(0.1)
            except KeyboardInterrupt:
                raise
            except Exception as sym_err:
                if "10002" in str(sym_err) or "recv_window" in str(sym_err).lower() or "timestamp" in str(sym_err).lower():
                    raise
                log.error(f"[{symbol}] Ошибка обработки пары в цикле: {sym_err}")

    def _publish_live_status(self) -> dict:
        """Публикует актуальный срез состояния (включая shadow balance и shadow_positions) для standalone dashboard.py."""
        is_shadow_only = bool(getattr(config, "SHADOW_MODE", False)) and getattr(self, "shadow_engine", None) is not None
        ltf_tf = self._normalize_kline_interval(getattr(config, "INTERVAL", INTERVAL))[0]

        if is_shadow_only:
            sh_eng = self.shadow_engine
            with self.lock:
                with sh_eng.lock:
                    sh_realized = float(sh_eng.equity or self.start_day_balance or 0.0)
                    used_margin = sum(
                        float(getattr(r, "margin_used", 0.0) or 0.0)
                        for r in getattr(sh_eng, "risk_states", {}).values()
                        if r
                    )
                    shadow_balance = max(0.0, sh_realized - used_margin)
                    shadow_positions = []
                    unrealized_pnl = 0.0
                    lev = int(getattr(sh_eng, "leverage", LEVERAGE) or LEVERAGE)
                    for sym, pos in list(sh_eng.positions.items()):
                        if not pos or float(pos.get("size") or 0.0) <= 0:
                            continue
                        r_st = sh_eng.risk_states.get(sym)
                        entry = float(pos.get("entry_price") or 0.0)
                        size = float(pos.get("size") or 0.0)
                        mark = float(getattr(self, "ws_prices", {}).get(sym) or 0.0)
                        if mark <= 0:
                            k_df = getattr(self, "kline_cache", {}).get((sym, ltf_tf))
                            if k_df is not None and not k_df.empty:
                                mark = float(k_df["close"].iloc[-1])
                        if mark <= 0:
                            mark = entry
                        is_long = pos.get("position") == "long" or str(pos.get("side", "")).lower() == "buy"
                        if entry > 0 and size > 0:
                            diff = (mark - entry) if is_long else (entry - mark)
                            pnl_usd = diff * size
                            pnl_pct = (diff / entry) * 100.0 * lev
                            unrealized_pnl += pnl_usd
                        else:
                            pnl_usd = 0.0
                            pnl_pct = 0.0
                        is_be = bool(
                            getattr(self, "breakeven_applied", {}).get(
                                sym,
                                getattr(r_st, "breakeven_applied", False) if r_st else False,
                            )
                        )
                        shadow_positions.append({
                            "symbol": sym,
                            "side": pos.get("side", "Buy" if is_long else "Sell"),
                            "position": "long" if is_long else "short",
                            "size": size,
                            "entry_price": entry,
                            "mark_price": mark,
                            "pnl": round(pnl_usd, 2),
                            "pnl_pct": round(pnl_pct, 2),
                            "stop_loss": float(pos.get("stop_loss") or 0.0),
                            "take_profit": float(pos.get("take_profit") or 0.0),
                            "breakeven": is_be,
                            "margin": round(float(getattr(r_st, "margin_used", 0.0) or pos.get("margin") or 0.0), 4),
                            "shadow_mode": True,
                        })

                    eq_ref = sh_realized + unrealized_pnl
                    daily_pnl_snap = (
                        (eq_ref - float(self.start_day_balance))
                        if (eq_ref > 0 and self.start_day_balance > 0)
                        else 0.0
                    )
                    pr = sh_eng.portfolio_risk
                    heat_pct = pr.compute_open_heat_pct(sh_eng.risk_states, eq_ref) * 100.0 if pr else 0.0
                    dd_info = (
                        pr.update_equity(eq_ref, valid=False)
                        if pr
                        else {"peak_equity": eq_ref, "peak_dd_pct": 0.0, "risk_scale": 1.0, "drawdown_pause": False}
                    )
                    sh_status = sh_eng.get_status()
                    cooldowns = pr.active_cooldowns() if pr else {}
                    be_map = dict(getattr(self, "breakeven_applied", {}))
                    act_syms = list(getattr(self, "active_symbols", SYMBOLS))
                    reg_snap = dict(getattr(self, "_last_regime_snapshot", {}))
                    cb_active = bool(self.circuit_breaker_active)
                    start_bal = float(self.start_day_balance)

            snap = {
                "shadow_mode": True,
                "circuit_breaker": cb_active,
                "start_day_balance": start_bal,
                "balance": round(shadow_balance, 2),
                "equity": round(eq_ref, 2),
                "unrealized_pnl": round(unrealized_pnl, 2),
                "daily_pnl": round(daily_pnl_snap, 2),
                "breakeven_map": be_map,
                "shadow_positions": shadow_positions,
                "peak_equity": dd_info["peak_equity"],
                "peak_drawdown_pct": dd_info["peak_dd_pct"],
                "risk_scale": dd_info["risk_scale"],
                "drawdown_pause": dd_info["drawdown_pause"],
                "portfolio_heat_pct": round(heat_pct, 2),
                "active_symbols": act_syms,
                "regime_summary": reg_snap,
                "cooldowns": cooldowns,
                "shadow_status": sh_status,
            }
        else:
            with self.lock:
                cached_eq = float(getattr(self, "last_known_equity", 0.0) or 0.0)
                cached_bal = float(getattr(self, "last_known_balance", 0.0) or 0.0)
                daily_pnl_snap = (
                    (cached_eq - float(self.start_day_balance))
                    if (cached_eq > 0 and self.start_day_balance > 0)
                    else 0.0
                )
                pr = getattr(self, "portfolio_risk", None)
                eq_ref = cached_eq if cached_eq > 0 else float(self.start_day_balance or 0.0)
                bal_ref = cached_bal if cached_bal > 0 else eq_ref
                heat_pct = pr.compute_open_heat_pct(getattr(self, "risk_states", {}), eq_ref) * 100.0 if pr else 0.0
                dd_info = (
                    pr.update_equity(eq_ref, valid=False)
                    if pr
                    else {"peak_equity": eq_ref, "peak_dd_pct": 0.0, "risk_scale": 1.0, "drawdown_pause": False}
                )
                sh_eng = getattr(self, "shadow_engine", None)
                snap = {
                    "shadow_mode": False,
                    "circuit_breaker": bool(self.circuit_breaker_active),
                    "start_day_balance": float(self.start_day_balance),
                    "balance": round(bal_ref, 2),
                    "equity": round(eq_ref, 2),
                    "daily_pnl": round(daily_pnl_snap, 2),
                    "breakeven_map": dict(getattr(self, "breakeven_applied", {})),
                    "peak_equity": dd_info["peak_equity"],
                    "peak_drawdown_pct": dd_info["peak_dd_pct"],
                    "risk_scale": dd_info["risk_scale"],
                    "drawdown_pause": dd_info["drawdown_pause"],
                    "portfolio_heat_pct": round(heat_pct, 2),
                    "active_symbols": list(getattr(self, "active_symbols", SYMBOLS)),
                    "regime_summary": dict(getattr(self, "_last_regime_snapshot", {})),
                    "cooldowns": pr.active_cooldowns() if pr else {},
                    "shadow_status": sh_eng.get_status() if sh_eng is not None else None,
                }

        write_live_status(snap)
        self._last_live_status_write = time.time()
        return snap

    def run(self):
        watch_list = self._get_watch_symbols()
        log.info(
            f"=== Запуск цикла мониторинга ({len(watch_list)} пар, Event-Driven WS kline={config.USE_WS_KLINE_TRIGGER}) ===\n"
            f"Лимит позиций: {MAX_OPEN_POSITIONS} | HeatCap: {config.MAX_PORTFOLIO_HEAT_PCT*100:.1f}% | "
            f"MTF: {USE_MTF_FILTER} | ATR: {USE_ATR_STOPS} | Trailing: {USE_TRAILING_STOP} | Partial TP: {USE_PARTIAL_TP}"
        )

        last_log_time = 0.0
        last_kline_eval = 0.0
        last_sync_time = 0.0
        last_cb_check = 0.0
        consecutive_errors = 0
        fallback_poll_sec = float(getattr(config, "KLINE_FALLBACK_POLL_SEC", 180))

        while True:
            try:
                now = time.time()

                # 0. Синхронизация команд от автономного dashboard.py (пауза / смена режима)
                self._sync_external_control()

                # 0.1. Отложенная сверка Net PnL через Bybit get_closed_pnl
                self.process_pending_pnl_sync()

                # 1. Защита от дневной просадки + MaxDD HWM (раз в 30с)
                if now - last_cb_check >= 30.0:
                    self.check_daily_circuit_breaker()
                    last_cb_check = now

                # 1.1. Авто-проверка и переподключение WebSocket после сетевых/DNS сбоев
                self.check_and_reconnect_websockets()

                # 2. Регулярная синхронизация времени (60м), инструментов (24ч) и динамической корзины
                if now - self.last_time_sync > 3600:
                    sync_exchange_time(config.TESTNET, config.DEMO)
                    self.last_time_sync = now
                self.refresh_instruments_if_needed()
                self.refresh_universe_if_needed()

                # 3. Синхронизация открытых позиций (раз в 30с или чаще при отключенном private WS)
                sync_interval = 15.0 if (self.ws_private is None or getattr(self.ws_private, "exited", False)) else 45.0
                if (now - last_sync_time >= sync_interval) and not getattr(config, "SHADOW_MODE", False):
                    self.sync_positions()
                    last_sync_time = now

                # 3.1 Проверка Time-Stop и стопов активных позиций по кэшированным WS ценам
                for sym_open, pos_open in list(self.positions.items()):
                    if pos_open and sym_open in self.ws_prices:
                        wp = self.ws_prices[sym_open]
                        self.manage_risk_and_stops(sym_open, wp, wp, wp)

                # 3.2 Публикуем срез состояния для standalone dashboard.py
                if now - getattr(self, "_last_live_status_write", 0.0) >= 15.0:
                    self._publish_live_status()

                # 4. Периодический отчет о состоянии и снимок equity в SQLite (каждые 10 минут)
                if now - last_log_time > 600:
                    if getattr(config, "SHADOW_MODE", False) and getattr(self, "shadow_engine", None) is not None:
                        sh_eng = self.shadow_engine
                        eq_now = float(sh_eng.equity)
                        used_margin = sum(float(getattr(r, "margin_used", 0.0) or 0.0) for r in sh_eng.risk_states.values() if r)
                        bal = max(0.0, eq_now - used_margin)
                        active = [f"{s}:{p['position']}" for s, p in self.positions.items() if p]
                        active_str = ", ".join(active) if active else "нет"
                        trade_log.info(
                            f"[SHADOW] Баланс: {bal:.2f} USDT | Equity: {eq_now:.2f} | Активные ({len(active)}/{MAX_OPEN_POSITIONS}): [{active_str}]"
                        )
                        if getattr(self, "db", None) is not None and eq_now > 0:
                            self.db.record_equity_snapshot(
                                equity=eq_now, balance=bal, unrealized_pnl=0.0, source="periodic", is_shadow=True,
                            )
                    else:
                        bal = self.get_available_balance()
                        eq_now = self.get_wallet_equity()
                        active = [f"{s}:{p['position']}" for s, p in self.positions.items() if p]
                        active_str = ", ".join(active) if active else "нет"
                        trade_log.info(
                            f"Баланс: {bal:.2f} USDT | Equity: {eq_now:.2f} | Активные ({len(active)}/{MAX_OPEN_POSITIONS}): [{active_str}]"
                        )
                        if getattr(self, "last_equity_ok", False) and eq_now > 0:
                            self._db_record_equity(equity=eq_now, balance=bal, source="periodic")
                    last_log_time = now

                # 5. Event-driven триггер: запуск пересчета сигналов при закрытии свечи по WS (или резервный heartbeat)
                ws_kline_triggered = hasattr(self, "_candle_event") and self._candle_event.is_set()
                heartbeat_due = (now - last_kline_eval) >= fallback_poll_sec
                if ws_kline_triggered or heartbeat_due or last_kline_eval == 0.0:
                    if hasattr(self, "_candle_event"):
                        self._candle_event.clear()
                    self._evaluate_market_cycle()
                    last_kline_eval = time.time()

                consecutive_errors = 0
                if hasattr(self, "_candle_event") and self._candle_event is not None:
                    self._candle_event.wait(timeout=5.0)
                else:
                    time.sleep(5.0)

            except KeyboardInterrupt:
                log.info("Бот остановлен пользователем")
                self._safe_close_ws(getattr(self, "ws", None))
                self._safe_close_ws(getattr(self, "ws_private", None))
                break

            except Exception as e:
                consecutive_errors += 1
                log.error(f"Ошибка в цикле мониторинга ({consecutive_errors}): {e}")
                if "10002" in str(e) or "recv_window" in str(e).lower() or "timestamp" in str(e).lower():
                    log.warning("Обнаружен дрейф времени (код 10002). Ресинхронизация времени с сервером Bybit...")
                    sync_exchange_time(config.TESTNET, config.DEMO)
                    self.last_time_sync = time.time()

                if consecutive_errors >= 10:
                    log.critical("10 ошибок подряд. Пауза 5 минут перед повтором...")
                    time.sleep(300)
                    consecutive_errors = 0
                else:
                    time.sleep(15)


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "backtest":
        days = int(sys.argv[2]) if len(sys.argv) > 2 and sys.argv[2].isdigit() else 90
        target = sys.argv[2] if len(sys.argv) > 2 and not sys.argv[2].isdigit() else "all"
        run_backtest(days=days, target_symbol=target)
    else:
        setup_logging()
        bot = BybitMultiFuturesBot()
        bot.run()
