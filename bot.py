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
from risk_manager import PositionRiskState, MarketRegimeFilter
from web.state import (
    set_bot_instance, is_bot_paused, set_bot_paused,
    write_control_state, read_control_state, write_live_status
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


class BybitMultiFuturesBot:
    def __init__(self):
        setup_logging()
        log.info(f"=== Инициализация Multi-Pair Bot v2.3 | Режим: {config.TRADING_MODE} ===")
        log.info(
            f"Торгуемые пары: {', '.join(SYMBOLS)} | TF: {INTERVAL}m (MTF: {HIGHER_INTERVAL}m) | Плечо: x{LEVERAGE}"
        )
        log.info(
            f"Стратегия: {ACTIVE_STRATEGY} | Трейлинг: {USE_TRAILING_STOP} (Native: {USE_NATIVE_TRAILING_STOP}) | "
            f"Частичный TP: {USE_PARTIAL_TP} (Limit: {USE_EXCHANGE_TP1}) | Smart Exec: {USE_SMART_EXECUTION}"
        )

        # Синхронизация системного времени
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

        # Модульная стратегия и фильтр рыночного сентимента
        strat_use_mtf = USE_MTF_FILTER if ACTIVE_STRATEGY.upper() != "BB_RSI" else False
        self.strategy = get_strategy(
            ACTIVE_STRATEGY,
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
            adx_threshold=ADX_THRESHOLD
        )

        self.regime_filter = MarketRegimeFilter(
            benchmark_symbol=BENCHMARK_SYMBOL,
            ema_period=BENCHMARK_EMA_PERIOD,
            rsi_threshold=BENCHMARK_RSI_THRESHOLD,
            bull_rsi_threshold=BENCHMARK_BULL_RSI_THRESHOLD,
            dump_pct_threshold=BENCHMARK_DUMP_PCT,
            pump_pct_threshold=BENCHMARK_PUMP_PCT,
            dump_lookback_candles=BENCHMARK_DUMP_CANDLES
        )

        self.lock = threading.RLock()
        self.last_time_sync = time.time()
        self.state_file = config.STATE_FILE

        self.instruments = {}
        self.positions = {}
        self.hedge_positions = {}  # {(symbol, positionIdx): dict}
        self.risk_states = {}  # {symbol: PositionRiskState}
        self.last_handled_candles = {}
        self.breakeven_applied = {}
        self.position_modes = {}  # {symbol: "MergedSingle" | "BothSides"}

        self.ws = None           # Public ticker WebSocket
        self.ws_private = None   # Private User Data WebSocket (position, order, execution)
        self._public_ws_gen = 0
        self._private_ws_gen = 0
        self.ws_prices = {}
        self.ws_last_update = {}
        self.ws_orders = {}      # {orderId: {"status": str, "cum_qty": float, "avg_price": float, "ts": float}}
        self.last_exec_prices = {}
        self.last_exec_info = {}
        self._seen_exec_ids = deque(maxlen=1000)
        self.last_risk_api_call = {}
        self.last_ws_public_msg = time.time()
        self.last_ws_private_msg = time.time()
        self.last_ws_reconnect_attempt = time.time()

        # Риск-менеджмент: учет баланса и дневных убытков
        self.last_known_equity = 0.0
        self.last_equity_ok = False
        self.last_balance_ok = False
        self.start_day = datetime.now().date()
        self.start_day_balance = 0.0
        self.circuit_breaker_active = False
        self.is_paused = False
        self._last_control_ts = time.time()
        self._last_live_status_write = 0.0

        # 1. Загрузка параметров инструментов, настройка плеча и режима позиций (Hedge/One-Way)
        for sym in SYMBOLS:
            self._load_instrument_info(sym)
            self._init_leverage(sym)
            self._detect_position_mode(sym)
            self.positions[sym] = None
            self.risk_states[sym] = None
            self.last_handled_candles[sym] = None
            self.breakeven_applied[sym] = False

        # 2. Первичная синхронизация открытых позиций
        self.sync_positions()

        # 3. Фиксация начального баланса
        init_bal = self.get_wallet_equity()
        self.start_day_balance = init_bal if getattr(self, "last_equity_ok", False) and init_bal > 0 else FALLBACK_USDT
        trade_log.info(f"Стартовый баланс: {self.start_day_balance:.2f} USDT")

        # 4. Регистрация глобального экземпляра для веб-дашборда и файла управления
        write_control_state(is_paused=False, trading_mode=config.TRADING_MODE)
        self._last_control_ts = time.time()
        set_bot_instance(self)

        # 5. Запуск веб-дашборда в фоновом демоническом потоке
        self._start_dashboard_thread()

        # 6. Запуск фонового WebSocket клиента (публичный + приватный)
        self._init_websocket()

    def save_state(self):
        """
        Сохраняет активные риск-состояния на диск АТОМАРНО.
        Запись ведется во временный файл .tmp с последующей атомарной заменой (replace),
        что гарантирует целостность файла при внезапных отключениях Termux/сервера.
        """
        try:
            with self.lock:
                data = {}
                for sym, st in self.risk_states.items():
                    if st is not None:
                        data[sym] = st.to_dict()
                tmp_file = self.state_file.with_suffix(".tmp")
                with open(tmp_file, "w", encoding="utf-8") as f:
                    json.dump(data, f, indent=2)
                tmp_file.replace(self.state_file)
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
            return {sym: PositionRiskState.from_dict(d) for sym, d in data.items()}
        except Exception as e:
            log.warning(f"Ошибка загрузки {target_file}: {e}")
            tmp = self.state_file.with_suffix(".tmp")
            if target_file != tmp and tmp.exists():
                try:
                    with open(tmp, "r", encoding="utf-8") as f:
                        data = json.load(f)
                    return {sym: PositionRiskState.from_dict(d) for sym, d in data.items()}
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
        try:
            resp = self.client.get_positions(category="linear", symbol=symbol)
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

    def _api_call(self, fn, *args, max_attempts: int = 3, **kwargs):
        """Обертка для критических REST-вызовов Bybit с backoff при Rate Limit (10006)."""
        last_exc = None
        for attempt in range(max_attempts):
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
            if hasattr(ws_obj, "_handle_incoming_message"):
                ws_obj._handle_incoming_message = lambda *a, **kw: None
            if hasattr(ws_obj, "_on_error"):
                ws_obj._on_error = lambda *a, **kw: None
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

    def _init_public_ws(self):
        """Подключает публичный WebSocket тикеров (рыночные котировки Bybit Demo идут с Mainnet -> demo=False)."""
        with self.lock:
            old_ws = getattr(self, "ws", None)
            self.ws = None
            self._public_ws_gen = getattr(self, "_public_ws_gen", 0) + 1
            gen = self._public_ws_gen

        self._safe_close_ws(old_ws)

        try:
            new_ws = WebSocket(
                testnet=config.TESTNET,
                demo=False,
                channel_type="linear",
                restart_on_error=True
            )
            new_ws.ticker_stream(
                symbol=SYMBOLS,
                callback=lambda msg, g=gen: self._handle_ws_ticker(msg, ws_gen=g)
            )
            with self.lock:
                if gen == getattr(self, "_public_ws_gen", gen):
                    self.ws = new_ws
                    self.last_ws_public_msg = time.time()
                else:
                    self._safe_close_ws(new_ws)
                    return
            log.info(f"Публичный WebSocket подключен и подписан на тикеры {len(SYMBOLS)} пар.")
        except Exception as e:
            log.warning(f"Публичный WebSocket не удалось подключить ({e}). Работаем через REST API fallback.")
            with self.lock:
                if gen == getattr(self, "_public_ws_gen", gen):
                    self.ws = None

    def _init_private_ws(self):
        """Подключает приватный WebSocket User Data Stream (position, execution, order)."""
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
        has_keys = not (config.is_placeholder_key(config.API_KEY) or config.is_placeholder_key(config.API_SECRET))

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
            for item in items:
                sym = item.get("symbol") or topic_sym
                if not sym or (sym not in SYMBOLS and sym not in self.positions):
                    continue
                last_p = item.get("lastPrice")
                if last_p:
                    price = float(last_p)
                    self.ws_prices[sym] = price
                    self.ws_last_update[sym] = now

                    # Real-time реакция на стоп/трейлинг/TP если открыта позиция
                    if self.positions.get(sym) and self.risk_states.get(sym):
                        self._check_ws_risk_update(sym, price)
        except Exception as e:
            log.debug(f"Ошибка в WebSocket ticker handler: {e}")

    def _log_exchange_close(self, sym: str, cur_pos: dict, cur_risk: Optional[PositionRiskState], source: str = "WebSocket"):
        """Формирует и записывает в trade.log строку закрытия позиции на бирже с взвешенным PnL (учитывая TP1)."""
        try:
            entry_price = float(cur_pos.get("entry_price") or (cur_risk.entry_price if cur_risk else 0.0) or 0.0)
            is_short = (
                str(cur_pos.get("position", "")).lower() == "short"
                or str(cur_pos.get("side", "")).lower() == "sell"
            )
            cur_pos_side = str(cur_pos.get("side") or ("Sell" if is_short else "Buy"))

            exec_p = None
            exec_info = getattr(self, "last_exec_info", {}).pop(sym, None)
            if isinstance(exec_info, dict) and exec_info.get("price", 0.0) > 0:
                if not exec_info.get("side") or exec_info["side"].lower() != cur_pos_side.lower():
                    exec_p = float(exec_info["price"])

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
                # Если нативный трейлинг сдвинул стоп за пределы БУ в профит (или БУ не применялся) — это Trailing SL
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

                # Взвешенный PnL с учетом ранее закрытой доли по TP1
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
            else:
                trade_log.info(f"[{sym}] {source}: Позиция закрыта на бирже (size=0).")
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
        self.last_ws_private_msg = time.time()
        try:
            data = message.get("data", [])
            items = data if isinstance(data, list) else [data]
            with self.lock:
                for item in items:
                    sym = item.get("symbol")
                    if not sym or (sym not in SYMBOLS and sym not in self.positions):
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
                            "ts": time.time(),
                        }
                    except Exception:
                        pass
                trade_log.info(f"[{sym}] WS Fill ({exec_type}): {side} {exec_q} @ {exec_p}")
        except Exception as e:
            log.debug(f"Ошибка в WS execution handler: {e}")

    def _handle_ws_order(self, message: dict, ws_gen: Optional[int] = None):
        """Real-time обработка изменений статусов ордеров (PostOnly, Limit TP1 и др.)."""
        if ws_gen is not None and ws_gen != getattr(self, "_private_ws_gen", ws_gen):
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

            if USE_TRAILING_STOP and not risk.native_trailing_active and not should_manage:
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

            for sym in SYMBOLS:
                self.positions[sym] = None
                self.risk_states[sym] = None
                self.breakeven_applied[sym] = False
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
        """Загружает точность цены, шага лота и минимального нотионала для каждого символа."""
        try:
            resp = self.client.get_instruments_info(category="linear", symbol=symbol)
            info = resp["result"]["list"][0]
            lot_filter = info.get("lotSizeFilter", {})
            price_filter = info.get("priceFilter", {})

            raw_qty_step = lot_filter.get("qtyStep", "0.001")
            raw_tick_size = price_filter.get("tickSize", "0.1")

            qty_step = float(raw_qty_step)
            min_qty = float(lot_filter.get("minOrderQty", 0.001))
            min_notional = float(lot_filter.get("minNotionalValue") or MIN_ORDER_NOTIONAL)
            tick_size = float(raw_tick_size)

            qty_decimals = self._count_decimals(raw_qty_step)
            price_decimals = self._count_decimals(raw_tick_size)

            self.instruments[symbol] = {
                "qty_step": qty_step,
                "min_qty": min_qty,
                "min_notional": min_notional,
                "tick_size": tick_size,
                "qty_decimals": qty_decimals,
                "price_decimals": price_decimals
            }
            log.info(
                f"{symbol:<10} | Step: {qty_step} | Min: {min_qty} | MinNotional: {min_notional}$ | Tick: {tick_size}"
            )
        except Exception as e:
            log.warning(f"Не удалось загрузить параметры для {symbol}: {e}. Используем дефолтные.")
            self.instruments[symbol] = {
                "qty_step": 0.001,
                "min_qty": 0.001,
                "min_notional": MIN_ORDER_NOTIONAL,
                "tick_size": 0.1,
                "qty_decimals": 3,
                "price_decimals": 2
            }

    def _init_leverage(self, symbol: str):
        """Безопасная установка плеча на бирже."""
        try:
            self.client.set_leverage(
                category="linear",
                symbol=symbol,
                buyLeverage=str(LEVERAGE),
                sellLeverage=str(LEVERAGE),
            )
            log.info(f"[{symbol}] Плечо x{LEVERAGE} успешно установлено")
        except Exception as e:
            if "110043" in str(e) or "Set leverage not modified" in str(e):
                log.info(f"[{symbol}] Плечо уже x{LEVERAGE}")
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
            resp = self.client.get_wallet_balance(accountType="UNIFIED")
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
                        return min(tot_avail, free_wb) if wb > 0 and (pos_im > 0 or ord_im > 0) else tot_avail
                    if avail_val is not None:
                        return min(avail_val, free_wb) if wb > 0 and (pos_im > 0 or ord_im > 0) else avail_val
                    return free_wb
            self.last_balance_ok = True
            return tot_avail if tot_avail is not None else 0.0
        except Exception as e:
            log.warning(f"Ошибка получения баланса: {e}. Возвращаем 0.0 USDT (блокировка новых входов).")
            self.last_balance_ok = False
            return 0.0

    def get_wallet_equity(self) -> float:
        """Получает общий капитал (Equity = Balance + Unrealized PnL) с защитой от сетевых сбоев."""
        try:
            resp = self.client.get_wallet_balance(accountType="UNIFIED")
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
        """Защита от дневной просадки (Daily Loss Circuit Breaker) с защитой от ложных срабатываний при таймауте API."""
        today = datetime.now().date()
        if today != self.start_day:
            curr_equity = self.get_wallet_equity()
            if getattr(self, "last_equity_ok", False) and curr_equity > 0:
                self.start_day = today
                self.start_day_balance = curr_equity
                self.circuit_breaker_active = False
                trade_log.info(f"Новый торговый день. Базовый баланс сброшен на: {self.start_day_balance:.2f} USDT")
            elif getattr(self, "last_known_equity", 0.0) > 0:
                self.start_day = today
                self.start_day_balance = self.last_known_equity
                self.circuit_breaker_active = False
                trade_log.info(
                    f"Новый торговый день (кэш equity). Базовый баланс сброшен на: {self.start_day_balance:.2f} USDT"
                )
            return

        if self.start_day_balance <= 0:
            return

        curr_equity = self.get_wallet_equity()
        # Если API баланса недоступен (таймаут сети) и вернулся FALLBACK_USDT, НЕ вызываем ложный Circuit Breaker!
        if not getattr(self, "last_equity_ok", False):
            log.warning("Пропуск проверки Circuit Breaker: нет достоверного ответа баланса от API Bybit")
            return

        daily_loss_pct = (self.start_day_balance - curr_equity) / self.start_day_balance

        if daily_loss_pct >= MAX_DAILY_LOSS_PCT and not self.circuit_breaker_active:
            self.circuit_breaker_active = True
            trade_log.critical(
                f"Сработал Circuit Breaker! Дневная просадка {daily_loss_pct*100:.2f}% >= "
                f"лимита {MAX_DAILY_LOSS_PCT*100:.1f}%. Торговля остановлена, позиции закрываются!"
            )
            self.panic_close_all()

    def sync_positions(self):
        """Синхронизирует позиции бота с биржей через REST API (с поддержкой Hedge Mode)."""
        with self.lock:
            try:
                resp = self.client.get_positions(category="linear", settleCoin="USDT")
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

                tracked_symbols = list(dict.fromkeys(list(SYMBOLS) + list(self.positions.keys()) + list(live_map.keys())))
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
        tf = str(interval or INTERVAL)
        try:
            resp = self.client.get_kline(
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
            return df
        except Exception as e:
            log.error(f"[{symbol}] Ошибка получения свечей (TF {tf}): {e}")
            return pd.DataFrame()

    def get_higher_trend(self, symbol: str) -> str:
        """Определяет старший тренд (4H EMA) с fail-safe защитой при сбое загрузки."""
        if not USE_MTF_FILTER or not getattr(self.strategy, "use_mtf_filter", True):
            return "neutral"
        try:
            df_htf = self.get_klines(symbol, interval=HIGHER_INTERVAL, limit=max(600, HTF_EMA_PERIOD * 3))
            if (
                df_htf is None
                or df_htf.empty
                or len(df_htf) < HTF_EMA_PERIOD + 5
                or df_htf["close"].iloc[-2:].isna().any()
            ):
                log.warning(f"[{symbol}] Недостаточно данных 4H для MTF-фильтра (fail-safe).")
                return "unavailable"
            return self.strategy.get_mtf_trend(df_htf)
        except Exception as e:
            log.warning(f"[{symbol}] Ошибка старшего тренда: {e}")
            return "unavailable"

    def manage_risk_and_stops(self, symbol: str, curr_price: float, high_price: float, low_price: float):
        """
        Управляет рисками активной позиции:
        1. Частичная фиксация прибыли (Partial TP на 1.5R)
        2. Гарантированный перенос стопа в безубыток при достижении BREAKEVEN_TRIGGER
        3. Трейлинг-стоп (Trailing Stop)
        """
        with self.lock:
            pos = self.positions.get(symbol)
            risk = self.risk_states.get(symbol)
            if not pos or not risk:
                return

            min_notional = float(self.instruments.get(symbol, {}).get("min_notional", MIN_ORDER_NOTIONAL))
            min_qty = float(self.instruments.get(symbol, {}).get("min_qty", 0.001))

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
                            ord_chk = self.client.get_open_orders(
                                category="linear", symbol=symbol, orderId=risk.tp1_order_id
                            )
                            open_list = ord_chk.get("result", {}).get("list", [])
                            if open_list:
                                can_market_close = False
                            else:
                                tp1_status = getattr(self, "ws_orders", {}).get(risk.tp1_order_id, {}).get("status", "")
                                if not tp1_status and hasattr(self.client, "get_order_history"):
                                    h_resp = self.client.get_order_history(
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
                                        self.client.set_trading_stop(
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
                            r_be = self.client.set_trading_stop(
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

            # 3. Трейлинг-стоп (Trailing Stop): программный fallback (если нативный трейлинг на бирже не активен)
            if USE_TRAILING_STOP and not risk.native_trailing_active:
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
                            r_tr = self.client.set_trading_stop(
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
                            self.save_state()
                        except Exception as e:
                            if self._is_stop_not_modified(e):
                                pos["stop_loss"] = rounded_sl
                                risk.stop_loss = rounded_sl
                                self.save_state()
                            else:
                                log.warning(f"[{symbol}] Ошибка обновления трейлинг-стопа: {e}")

    def _fetch_order_fill_info(self, symbol: str, order_id: str) -> tuple[str, float, float]:
        """Получает актуальный статус, cumExecQty и avgPrice ордера из WS-кэша или REST history."""
        status = ""
        cum_qty = 0.0
        avg_price = 0.0

        ws_ord = getattr(self, "ws_orders", {}).get(order_id)
        if isinstance(ws_ord, dict):
            status = str(ws_ord.get("status", ""))
            cum_qty = float(ws_ord.get("cum_qty", 0.0) or 0.0)
            avg_price = float(ws_ord.get("avg_price", 0.0) or 0.0)

        if hasattr(self, "client") and hasattr(self.client, "get_order_history"):
            try:
                hist_resp = self.client.get_order_history(category="linear", symbol=symbol, orderId=order_id)
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

    def open_position(self, symbol: str, side: str, price: float, atr_value: float = 0.0):
        with self.lock:
            # 1. Проверка глобальной паузы из дашборда
            if is_bot_paused():
                log.info(f"[{symbol}] Бот на паузе (веб-интерфейс). Вход в позицию отменен.")
                return None

            # 2. Проверка Circuit Breaker
            if self.circuit_breaker_active:
                log.warning(f"[{symbol}] Circuit Breaker активен! Открытие позиций заблокировано.")
                return None

            # 3. Проверка лимита открытых позиций и лимита в одном направлении
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
                # Расчет стопов
                stops = self.strategy.calculate_stops(
                    price, side, atr_value=atr_value,
                    stop_loss_pct=STOP_LOSS_PCT,
                    take_profit_pct=TAKE_PROFIT_PCT,
                    atr_sl_mult=ATR_SL_MULT,
                    atr_tp_mult=ATR_TP_MULT,
                    use_atr=USE_ATR_STOPS
                )
                sl = self.round_price(symbol, stops["sl"])
                tp = self.round_price(symbol, stops["tp"])

                # Расчет объема позиции от equity с безопасным потолком <= свободного balance
                max_avail_margin = balance * 0.95
                if USE_RISK_BASED_SIZING and stops.get("sl_dist", 0) > 0:
                    risk_capital = equity * RISK_PER_TRADE_PCT
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
                    target_cap = max(equity * POSITION_SIZE_PCT, min(max_avail_margin, 10.0))
                    max_margin = min(max_avail_margin, target_cap)
                    max_qty = (max_margin * LEVERAGE) / price
                    raw_qty = min(raw_qty, max_qty)
                    trade_margin = (raw_qty * price) / LEVERAGE
                else:
                    target_cap = max(equity * POSITION_SIZE_PCT, min(max_avail_margin, 10.0))
                    trade_margin = min(max_avail_margin, target_cap)
                    notional = trade_margin * LEVERAGE
                    raw_qty = notional / price

                qty = self.round_qty(symbol, raw_qty)

                min_qty = self.instruments[symbol]["min_qty"]
                if qty < min_qty:
                    log.error(f"[{symbol}] Объем {qty} меньше минимального {min_qty}. Баланс: {balance:.1f} USDT.")
                    return None

                min_notional = float(self.instruments[symbol].get("min_notional", MIN_ORDER_NOTIONAL))
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

                # Умное исполнение: попытка Maker (Limit Post-Only) с защитой от гонки отмены и двойного входа
                executed = False
                needs_set_stop = False
                resp = None
                filled_qty = 0.0
                maker_avg_price = 0.0
                actual_entry_price = price

                if USE_SMART_EXECUTION:
                    order_id = None
                    limit_price = price
                    try:
                        tickers_resp = self.client.get_tickers(category="linear", symbol=symbol)
                        t_info = tickers_resp["result"]["list"][0]
                        bid = float(t_info.get("bid1Price") or price)
                        ask = float(t_info.get("ask1Price") or price)
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
                            trade_log.info(
                                f"[{symbol}] Maker ордер (Limit Post-Only) выставлен: {limit_p_str}. "
                                f"Ожидание исполнения (до {SMART_EXECUTION_TIMEOUT}с)..."
                            )
                            t0 = time.time()
                            while time.time() - t0 < SMART_EXECUTION_TIMEOUT:
                                self.lock.release()
                                try:
                                    time.sleep(0.5)
                                finally:
                                    self.lock.acquire()

                                # Если во время ожидания сработала пауза/CB или позиция уже открыта — прерываем ожидание
                                if is_bot_paused() or self.circuit_breaker_active or self.positions.get(symbol):
                                    log.info(f"[{symbol}] Ожидание Maker прервано (пауза/CB/позиция уже активна).")
                                    break

                                ws_st, ws_cum, ws_avg = self._fetch_order_fill_info(symbol, order_id)
                                if ws_st == "Filled":
                                    executed = True
                                    needs_set_stop = True
                                    filled_qty = ws_cum if ws_cum > 0 else qty
                                    maker_avg_price = ws_avg if ws_avg > 0 else limit_price
                                    actual_entry_price = maker_avg_price
                                    resp = maker_resp
                                    trade_log.info(f"[{symbol}] Maker ордер исполнен (Filled: {filled_qty})!")
                                    break
                                if ws_cum >= qty * 0.95:
                                    # Порог >=95% достигнут при PartiallyFilled: обязательно отменяем остаток на бирже!
                                    try:
                                        self._api_call(self.client.cancel_order, category="linear", symbol=symbol, orderId=order_id)
                                    except Exception:
                                        pass
                                    _, final_cum, final_avg = self._fetch_order_fill_info(symbol, order_id)
                                    filled_qty = final_cum if final_cum > 0 else ws_cum
                                    maker_avg_price = final_avg if final_avg > 0 else (ws_avg or limit_price)
                                    qty = self.round_qty(symbol, filled_qty)
                                    actual_entry_price = maker_avg_price
                                    executed = True
                                    needs_set_stop = True
                                    resp = maker_resp
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

                                try:
                                    ord_chk = self.client.get_open_orders(category="linear", symbol=symbol, orderId=order_id)
                                    open_list = ord_chk.get("result", {}).get("list", []) if isinstance(ord_chk, dict) else []
                                    if not open_list:
                                        hist_status, hist_cum, hist_avg = self._fetch_order_fill_info(symbol, order_id)
                                        if hist_status == "Filled" or hist_cum >= qty * 0.95:
                                            executed = True
                                            needs_set_stop = True
                                            filled_qty = hist_cum if hist_cum > 0 else qty
                                            qty = self.round_qty(symbol, filled_qty)
                                            maker_avg_price = hist_avg if hist_avg > 0 else limit_price
                                            actual_entry_price = maker_avg_price
                                            resp = maker_resp
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
                                            break
                                except Exception:
                                    pass

                            if not executed:
                                trade_log.info(f"[{symbol}] Maker таймаут/отмена. Отмена остатка и проверка факта исполнения...")
                                # 1. Отменяем в ОТДЕЛЬНОМ try-блоке (110001 = уже исполнен не должен скрывать историю!)
                                try:
                                    self._api_call(self.client.cancel_order, category="linear", symbol=symbol, orderId=order_id)
                                except Exception as ce:
                                    log.debug(f"[{symbol}] cancel_order после таймаута Maker: {ce}")

                                time.sleep(0.15)
                                # 2. Всегда проверяем фактическое исполнение в истории / WS
                                f_st, f_cum, f_avg = self._fetch_order_fill_info(symbol, order_id)
                                if f_cum > filled_qty:
                                    filled_qty = f_cum
                                if f_avg > 0:
                                    maker_avg_price = f_avg

                                # 3. Дополнительная проверка реальной позиции на бирже (страховка от race condition)
                                if hasattr(self.client, "get_positions"):
                                    try:
                                        p_chk = self.client.get_positions(category="linear", symbol=symbol)
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
                                    trade_log.info(f"[{symbol}] Maker ордер фактически уже исполнен перед отменой ({qty})! Двойной вход предотвращен.")
                    except Exception as e:
                        log.info(f"[{symbol}] Post-Only отклонен ({e}). Переход на Market ордер.")

                # Если пока ждали Maker, бот встал на паузу/CB — добор по Market НЕ делаем (но уже открытую часть защитим)
                if not executed and (is_bot_paused() or self.circuit_breaker_active or self.positions.get(symbol)):
                    if filled_qty >= min_qty and not self.positions.get(symbol):
                        executed = True
                        needs_set_stop = True
                        qty = self.round_qty(symbol, filled_qty)
                        actual_entry_price = maker_avg_price if maker_avg_price > 0 else price
                    else:
                        return None

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
                            mkt_avg = 0.0
                            mkt_avg_from_pos = False
                            if mkt_oid:
                                time.sleep(0.15)
                                _, _, mkt_avg = self._fetch_order_fill_info(symbol, mkt_oid)
                            if mkt_avg <= 0 and hasattr(self.client, "get_positions"):
                                try:
                                    p_chk = self.client.get_positions(category="linear", symbol=symbol)
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
                                # get_positions avgPrice уже отражает средневзвешенную цену всей открытой позиции
                                actual_entry_price = mkt_avg
                            elif filled_qty > 0 and maker_avg_price > 0 and total_q > 0:
                                actual_entry_price = ((filled_qty * maker_avg_price) + (rem_qty * mkt_avg)) / (filled_qty + rem_qty)
                            else:
                                actual_entry_price = mkt_avg
                            qty = total_q
                            executed = True
                            if filled_qty > 0:
                                needs_set_stop = True
                        else:
                            # Market добор упал, но Maker уже частично исполнился -> берем частичный объем под защиту!
                            if filled_qty >= min_qty:
                                qty = self.round_qty(symbol, filled_qty)
                                actual_entry_price = maker_avg_price if maker_avg_price > 0 else price
                                executed = True
                                needs_set_stop = True
                                log.warning(f"[{symbol}] Добор Market не удался, регистрируем частичное Maker-исполнение ({qty}) и ставим SL/TP.")
                            else:
                                return None
                    elif filled_qty >= min_qty:
                        executed = True
                        needs_set_stop = True
                        qty = self.round_qty(symbol, filled_qty)
                        actual_entry_price = maker_avg_price if maker_avg_price > 0 else price
                    else:
                        return None

                if isinstance(resp, dict) and resp.get("retCode", 0) != 0 and not executed:
                    log.error(f"[{symbol}] Биржа отклонила ордер (retCode={resp.get('retCode')}): {resp.get('retMsg')}")
                    return None

                # Пересчет SL/TP от реальной цены входа (avgPrice), если она отличается от предварительной price
                if actual_entry_price > 0 and price > 0 and abs(actual_entry_price - price) / price > 1e-6:
                    stops = self.strategy.calculate_stops(
                        actual_entry_price, side, atr_value=atr_value,
                        stop_loss_pct=STOP_LOSS_PCT,
                        take_profit_pct=TAKE_PROFIT_PCT,
                        atr_sl_mult=ATR_SL_MULT,
                        atr_tp_mult=ATR_TP_MULT,
                        use_atr=USE_ATR_STOPS
                    )
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

                # Регистрация объекта управления рисками
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
                    opened_at=time.time()
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

                # Активация нативного трейлинг-стопа Bybit
                if USE_TRAILING_STOP and USE_NATIVE_TRAILING_STOP:
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

                trade_log.info(
                    f"[{symbol}] {'LONG' if side == 'Buy' else 'SHORT'} открыт | "
                    f"Вход: {actual_entry_price} | Кол-во: {qty} ({trade_margin:.1f}$ маржа) | "
                    f"SL: {sl} | TP: {tp} [{stops['type']}] | x{LEVERAGE}"
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

            cur_risk = getattr(self, "risk_states", {}).get(symbol)
            try:
                # Снимаем только лимитный TP1 перед закрытием, сохраняя защитный SL до подтверждения Market-закрытия
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

                # Отменяем оставшиеся лимитные ордера ТОЛЬКО после подтвержденного закрытия
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
            all_tracked = list(dict.fromkeys(list(SYMBOLS) + list(self.positions.keys())))
            for sym in all_tracked:
                pos = self.positions.get(sym)
                if pos and pos.get("size", 0) > 0:
                    try:
                        price = self.get_price(sym)
                        self.close_position(sym, price, reason="PANIC BUTTON")
                    except Exception as e:
                        log.error(f"[{sym}] Ошибка Panic Close: {e}")

            # Дозачистка всех оставшихся позиций и ордеров напрямую на бирже
            if hasattr(self, "client") and hasattr(self.client, "get_positions"):
                try:
                    resp = self.client.get_positions(category="linear", settleCoin="USDT")
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
                    ord_resp = self.client.get_open_orders(category="linear", settleCoin="USDT")
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
        """Применяет изменения паузы/режима из внешнего dashboard.py (через bot_control.json)."""
        ctrl = read_control_state()
        if not ctrl:
            return
        upd_ts = float(ctrl.get("updated_at") or 0.0)
        if upd_ts <= getattr(self, "_last_control_ts", 0.0):
            return
        self._last_control_ts = upd_ts

        if "is_paused" in ctrl and bool(ctrl["is_paused"]) != bool(self.is_paused):
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
                write_control_state(trading_mode=prev[0])
                self._last_control_ts = time.time()

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
            resp = self.client.get_tickers(category="linear", symbol=symbol)
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

    def run(self):
        log.info(
            f"=== Запуск цикла мониторинга {len(SYMBOLS)} пар ===\n"
            f"Лимит позиций: {MAX_OPEN_POSITIONS} | Маржа: {POSITION_SIZE_PCT*100}% | "
            f"MTF: {USE_MTF_FILTER} | ATR: {USE_ATR_STOPS} | Trailing: {USE_TRAILING_STOP} | Partial TP: {USE_PARTIAL_TP}"
        )

        last_log_time = 0
        consecutive_errors = 0

        while True:
            try:
                # 0. Синхронизация команд от автономного dashboard.py (пауза / смена режима)
                self._sync_external_control()

                # 1. Защита от дневной просадки (Circuit Breaker)
                self.check_daily_circuit_breaker()

                # 1.1. Авто-проверка и переподключение WebSocket после сетевых/DNS сбоев
                self.check_and_reconnect_websockets()

                # 2. Регулярная синхронизация времени с Bybit (каждые 60 минут)
                if time.time() - self.last_time_sync > 3600:
                    sync_exchange_time(config.TESTNET, config.DEMO)
                    self.last_time_sync = time.time()

                # 3. Синхронизируем открытые позиции
                self.sync_positions()

                # 3.1 Публикуем срез состояния для standalone dashboard.py
                if time.time() - getattr(self, "_last_live_status_write", 0.0) >= 15.0:
                    cached_eq = float(getattr(self, "last_known_equity", 0.0) or 0.0)
                    daily_pnl_snap = (cached_eq - float(self.start_day_balance)) if (cached_eq > 0 and self.start_day_balance > 0) else 0.0
                    write_live_status({
                        "circuit_breaker": self.circuit_breaker_active,
                        "start_day_balance": self.start_day_balance,
                        "equity": cached_eq if cached_eq > 0 else self.start_day_balance,
                        "daily_pnl": round(daily_pnl_snap, 2),
                        "breakeven_map": dict(self.breakeven_applied),
                    })
                    self._last_live_status_write = time.time()

                # 4. Периодический отчет о состоянии (каждые 10 минут)
                if time.time() - last_log_time > 600:
                    bal = self.get_available_balance()
                    active = [f"{s}:{p['position']}" for s, p in self.positions.items() if p]
                    active_str = ", ".join(active) if active else "нет"
                    trade_log.info(
                        f"Баланс: {bal:.2f} USDT | Активные ({len(active)}/{MAX_OPEN_POSITIONS}): [{active_str}]"
                    )
                    last_log_time = time.time()

                # 5. Загрузка бенчмарка для фильтра сентимента (BTCUSDT) с достаточным прогревом EMA200 (600 свечей)
                bench_df = None
                bench_ok = True
                if USE_MARKET_REGIME_FILTER:
                    bench_ok = False
                    try:
                        bench_df = self.get_klines(BENCHMARK_SYMBOL, interval=BENCHMARK_INTERVAL, limit=max(600, BENCHMARK_EMA_PERIOD * 3))
                        if bench_df is not None and not bench_df.empty and len(bench_df) >= max(BENCHMARK_EMA_PERIOD, 30):
                            bench_ok = True
                        else:
                            log.warning(f"Данные бенчмарка {BENCHMARK_SYMBOL} неполные — входы заблокированы (fail-safe).")
                    except Exception as e:
                        log.warning(f"Не удалось загрузить бенчмарк {BENCHMARK_SYMBOL} ({e}) — входы заблокированы (fail-safe).")

                # 6. Сканируем каждую пару из корзины
                min_candles_req = max(SLOW_MA, ST_CONFIRM_EMA, ATR_PERIOD, RSI_PERIOD, 30) + 10
                kline_limit = max(250, min_candles_req + 50)
                for symbol in SYMBOLS:
                    try:
                        df = self.get_klines(symbol, limit=kline_limit)
                        if df.empty or len(df) < min_candles_req:
                            continue

                        df = self.strategy.calculate_indicators(df)
                        price = self.get_price(symbol, fallback_df=df)

                        # Передаем в риск-менеджер реальную текущую цену (без экстремумов незакрытой свечи до входа)
                        self.manage_risk_and_stops(symbol, price, price, price)

                        # Сигнал формируется по последней ЗАКРЫТОЙ свече (-2)
                        signal_candle = df.iloc[-2]
                        signal = int(signal_candle.get("signal", 0))
                        exit_signal = int(signal_candle.get("exit_signal", signal))
                        signal_price = signal_candle["close"]
                        signal_time = signal_candle["timestamp"]
                        atr_value = signal_candle.get("ATR", 0.0)
                        current_pos = self.positions.get(symbol)

                        if signal_time != self.last_handled_candles.get(symbol):
                            self.last_handled_candles[symbol] = signal_time

                            # 1. Приоритетное закрытие встречной позиции по exit_signal (не блокируется фильтрами входа!)
                            if exit_signal == 1 and current_pos and current_pos.get("position") == "short":
                                self.close_position(symbol, price, "Смена тренда на LONG")
                                current_pos = None
                            elif exit_signal == -1 and current_pos and current_pos.get("position") == "long":
                                self.close_position(symbol, price, "Смена тренда на SHORT")
                                current_pos = None

                            # 2. Направленная защита от неблагоприятного проскальзывания цены для нового входа
                            if signal == 1:
                                slippage = (price - signal_price) / signal_price
                            elif signal == -1:
                                slippage = (signal_price - price) / signal_price
                            else:
                                slippage = 0.0

                            if signal != 0 and slippage > MAX_SLIPPAGE_PCT:
                                log.warning(
                                    f"[{symbol}] Неблагоприятное проскальзывание {slippage*100:.2f}% > "
                                    f"{MAX_SLIPPAGE_PCT*100:.1f}%. Вход пропущен."
                                )
                                continue

                            # 3. Проверка фильтра рыночного сентимента по BTC (fail-safe при отсутствии bench_df)
                            if signal != 0 and USE_MARKET_REGIME_FILTER:
                                if not bench_ok:
                                    signal = 0
                                elif signal == 1:
                                    allowed, r_msg = self.regime_filter.is_long_allowed(symbol, bench_df)
                                    if not allowed:
                                        log.info(f"[{symbol}] {r_msg}")
                                        signal = 0
                                elif signal == -1:
                                    allowed, r_msg = self.regime_filter.is_short_allowed(symbol, bench_df)
                                    if not allowed:
                                        log.info(f"[{symbol}] {r_msg}")
                                        signal = 0

                            # 4. Проверка фильтра старшего тренда (4H EMA) с fail-safe защитой и открытие позиции
                            use_mtf_here = USE_MTF_FILTER and getattr(self.strategy, "use_mtf_filter", True)
                            htf_trend = self.get_higher_trend(symbol) if (use_mtf_here and signal != 0) else "neutral"

                            if signal == 1:
                                if not self.positions.get(symbol):
                                    if use_mtf_here and htf_trend in ("bearish", "unavailable"):
                                        log.info(f"[{symbol}] Сигнал LONG отклонен: старший тренд 4H ({htf_trend})")
                                    else:
                                        self.open_position(symbol, "Buy", price, atr_value=atr_value)

                                self.last_handled_candles[symbol] = signal_time

                            elif signal == -1:
                                if not self.positions.get(symbol):
                                    if use_mtf_here and htf_trend in ("bullish", "unavailable"):
                                        log.info(f"[{symbol}] Сигнал SHORT отклонен: старший тренд 4H ({htf_trend})")
                                    else:
                                        self.open_position(symbol, "Sell", price, atr_value=atr_value)

                                self.last_handled_candles[symbol] = signal_time

                        time.sleep(0.2)
                    except KeyboardInterrupt:
                        raise
                    except Exception as sym_err:
                        if "10002" in str(sym_err) or "recv_window" in str(sym_err).lower() or "timestamp" in str(sym_err).lower():
                            raise
                        log.error(f"[{symbol}] Ошибка обработки пары в цикле: {sym_err}")

                consecutive_errors = 0
                time.sleep(30)

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
                    time.sleep(30)


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
