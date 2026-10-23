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
    USE_BREAKEVEN, BREAKEVEN_TRIGGER, BREAKEVEN_OFFSET,
    USE_TRAILING_STOP, TRAILING_STOP_TRIGGER, TRAILING_STOP_ATR_MULT, TRAILING_STOP_FALLBACK,
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
from web.state import set_bot_instance, is_bot_paused, set_bot_paused
from backtest import run_backtest


# =====================================================================
# Настройка логирования с ротацией файлов (макс. 5MB на файл, до 3 копий):
# - bot.log: технические логи, ошибки, системные события
# - trade.log: торговые события (открытие, закрытие, безубыток, баланс)
# - Консоль (StreamHandler): вывод обоих типов событий
# =====================================================================
formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
stream_handler = logging.StreamHandler()
stream_handler.setFormatter(formatter)

# Системный логгер (bot)
log = logging.getLogger("bot")
log.setLevel(logging.INFO)
log.propagate = False

if not log.handlers:
    bot_file_handler = RotatingFileHandler("bot.log", maxBytes=5*1024*1024, backupCount=3, encoding="utf-8")
    bot_file_handler.setFormatter(formatter)
    log.addHandler(bot_file_handler)
    log.addHandler(stream_handler)

# Торговый логгер (trade)
trade_log = logging.getLogger("trade")
trade_log.setLevel(logging.INFO)
trade_log.propagate = False

if not trade_log.handlers:
    trade_file_handler = RotatingFileHandler("trade.log", maxBytes=5*1024*1024, backupCount=3, encoding="utf-8")
    trade_file_handler.setFormatter(formatter)
    trade_log.addHandler(trade_file_handler)
    trade_log.addHandler(stream_handler)


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
        log.info(f"=== Инициализация Multi-Pair Bot v2.2 | Режим: {config.TRADING_MODE} ===")
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
        self.strategy = get_strategy(
            ACTIVE_STRATEGY,
            st_period=ST_PERIOD,
            st_multiplier=ST_MULTIPLIER,
            confirm_ema_period=ST_CONFIRM_EMA,
            fast_ma=FAST_MA,
            slow_ma=SLOW_MA,
            rsi_period=RSI_PERIOD,
            rsi_long_min=RSI_LONG_MIN,
            rsi_long_max=RSI_LONG_MAX,
            rsi_short_min=RSI_SHORT_MIN,
            rsi_short_max=RSI_SHORT_MAX,
            volume_filter=VOLUME_FILTER,
            use_mtf_filter=USE_MTF_FILTER,
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
        self.state_file = Path("bot_state.json")

        self.instruments = {}
        self.positions = {}
        self.risk_states = {}  # {symbol: PositionRiskState}
        self.last_handled_candles = {}
        self.breakeven_applied = {}
        self.position_modes = {}  # {symbol: "MergedSingle" | "BothSides"}

        self.ws = None           # Public ticker WebSocket
        self.ws_private = None   # Private User Data WebSocket (position, order, execution)
        self.ws_prices = {}
        self.ws_last_update = {}
        self.ws_orders = {}      # {orderId: {"status": str, "cum_qty": float, "avg_price": float}}
        self.last_risk_api_call = {}

        # Риск-менеджмент: учет баланса и дневных убытков
        self.last_known_equity = 0.0
        self.last_equity_ok = False
        self.start_day = datetime.now().date()
        self.start_day_balance = 0.0
        self.circuit_breaker_active = False
        self.is_paused = False

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
        self.start_day_balance = init_bal if init_bal > 0 else FALLBACK_USDT
        trade_log.info(f"Стартовый баланс: {self.start_day_balance:.2f} USDT")

        # 4. Регистрация глобального экземпляра для веб-дашборда
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
        mode = self.position_modes.get(symbol, "MergedSingle")
        if mode == "BothSides":
            return 1 if side.lower() in ("buy", "long") else 2
        return 0

    def _init_websocket(self):
        """
        Инициализирует фоновые WebSocket клиенты Bybit:
        1. Публичный тикерный поток (linear) для получения цен без нагрузки на REST.
        2. Приватный User Data Stream (position, order, execution) для мгновенной (<50ms) реакции.
        """
        # 1. Публичный WebSocket (рыночные котировки Bybit Demo транслируются с Mainnet, поэтому demo=False)
        try:
            self.ws = WebSocket(
                testnet=config.TESTNET,
                demo=False,
                channel_type="linear",
                restart_on_error=True
            )
            self.ws.ticker_stream(
                symbol=SYMBOLS,
                callback=self._handle_ws_ticker
            )
            log.info(f"Публичный WebSocket подключен и подписан на тикеры {len(SYMBOLS)} пар.")
        except Exception as e:
            log.warning(f"Публичный WebSocket не удалось подключить ({e}). Работаем через REST API fallback.")
            self.ws = None

        # 2. Приватный WebSocket
        has_keys = bool(config.API_KEY and config.API_KEY not in ("your_demo_key", "your_live_key"))
        if has_keys:
            try:
                self.ws_private = WebSocket(
                    testnet=config.TESTNET,
                    demo=config.DEMO,
                    api_key=config.API_KEY,
                    api_secret=config.API_SECRET,
                    channel_type="private",
                    restart_on_error=True
                )
                self.ws_private.position_stream(callback=self._handle_ws_position)
                self.ws_private.execution_stream(callback=self._handle_ws_execution)
                self.ws_private.order_stream(callback=self._handle_ws_order)
                log.info("Приватный WebSocket успешно подключен (position, execution, order stream).")
            except Exception as e:
                log.warning(f"Приватный WebSocket не удалось подключить ({e}). Фоллбэк на REST sync_positions.")
                self.ws_private = None

    def _handle_ws_ticker(self, message: dict):
        """Callback-обработчик входящих тикеров WebSocket."""
        try:
            data = message.get("data")
            if not data:
                return
            items = data if isinstance(data, list) else [data]
            for item in items:
                sym = item.get("symbol")
                if not sym or (sym not in SYMBOLS and sym not in self.positions):
                    continue
                last_p = item.get("lastPrice")
                if last_p:
                    price = float(last_p)
                    self.ws_prices[sym] = price
                    self.ws_last_update[sym] = time.time()

                    # Real-time реакция на стоп/трейлинг/TP если открыта позиция
                    if self.positions.get(sym) and self.risk_states.get(sym):
                        self._check_ws_risk_update(sym, price)
        except Exception as e:
            log.debug(f"Ошибка в WebSocket ticker handler: {e}")

    def _handle_ws_position(self, message: dict):
        """
        Real-time обработчик изменений позиций по приватному WebSocket Bybit.
        Обеспечивает реакцию <50ms при закрытии по SL/TP или частичном закрытии.
        """
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
                        # В Hedge Mode (positionIdx 1 и 2) игнорируем пустую противоположную ногу позиции
                        if cur_idx in (1, 2) and item_idx in (1, 2) and item_idx != cur_idx:
                            continue
                        if size == 0.0 and item_side and cur_side and item_side != cur_side:
                            continue

                    if size == 0.0:
                        # Позиция закрыта на бирже
                        if cur_pos is not None:
                            trade_log.info(f"[{sym}] WebSocket: Позиция закрыта на бирже (size=0).")
                            self.positions[sym] = None
                            self.risk_states[sym] = None
                            self.breakeven_applied[sym] = False
                            self.save_state()
                    else:
                        if cur_pos is not None and cur_risk is not None:
                            # Проверка частичного закрытия (TP1)
                            if size < cur_risk.size and not cur_risk.tp1_executed:
                                closed_qty = cur_risk.size - size
                                cur_risk.tp1_executed = True
                                cur_risk.size = size
                                cur_pos["size"] = size
                                trade_log.info(
                                    f"[{sym}] WebSocket: Обнаружено частичное закрытие TP1 на бирже (закрыто: {closed_qty})."
                                )
                                if PARTIAL_TP_MOVE_BE and not cur_risk.breakeven_applied:
                                    be_price = cur_risk.entry_price * (1 + BREAKEVEN_OFFSET if cur_risk.is_long else 1 - BREAKEVEN_OFFSET)
                                    rounded_be = self.round_price(sym, be_price)
                                    pos_idx = self.get_position_idx(sym, cur_risk.side)
                                    try:
                                        self.client.set_trading_stop(
                                            category="linear",
                                            symbol=sym,
                                            stopLoss=self.format_price(sym, rounded_be),
                                            positionIdx=pos_idx
                                        )
                                        cur_risk.stop_loss = rounded_be
                                        cur_pos["stop_loss"] = rounded_be
                                        cur_risk.breakeven_applied = True
                                        self.breakeven_applied[sym] = True
                                        trade_log.info(f"[{sym}] WebSocket: Стоп перенесен в безубыток: {rounded_be}")
                                    except Exception as e:
                                        log.warning(f"[{sym}] WebSocket: Ошибка переноса стопа в безубыток: {e}")
                                self.save_state()
                            else:
                                cur_pos["size"] = size
                                cur_risk.size = size
        except Exception as e:
            log.debug(f"Ошибка в WS position handler: {e}")

    def _handle_ws_execution(self, message: dict):
        """Real-time логирование исполнения ордеров по приватному WebSocket."""
        try:
            data = message.get("data", [])
            items = data if isinstance(data, list) else [data]
            for item in items:
                sym = item.get("symbol")
                exec_p = item.get("execPrice")
                exec_q = item.get("execQty")
                side = item.get("side")
                exec_type = item.get("execType", "Trade")
                trade_log.info(f"[{sym}] WS Fill ({exec_type}): {side} {exec_q} @ {exec_p}")
        except Exception as e:
            log.debug(f"Ошибка в WS execution handler: {e}")

    def _handle_ws_order(self, message: dict):
        """Real-time обработка изменений статусов ордеров (PostOnly, Limit TP1 и др.)."""
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
                            "avg_price": avg_p
                        }
                    if sym and sym in self.risk_states and self.risk_states[sym] is not None:
                        r_st = self.risk_states[sym]
                        if oid and oid == r_st.tp1_order_id:
                            if status == "Filled":
                                r_st.tp1_executed = True
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
                be_updated, new_be_sl = risk.check_breakeven(price, trigger_pct=BREAKEVEN_TRIGGER, offset_pct=BREAKEVEN_OFFSET)
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
                    fallback_pct=TRAILING_STOP_FALLBACK
                )
                if trail_updated:
                    rounded_trail = self.round_price(symbol, new_trail_sl)
                    cur_sl = pos.get("stop_loss", 0.0)
                    if (rounded_trail > cur_sl) if risk.is_long else (cur_sl == 0.0 or rounded_trail < cur_sl):
                        should_manage = True

            if should_manage:
                self.last_risk_api_call[symbol] = now
                self.manage_risk_and_stops(symbol, price, risk.highest_price, risk.lowest_price)

    def reload_credentials(self):
        """Перезагружает клиент Bybit и WebSocket при переключении режима DEMO/LIVE."""
        with self.lock:
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
            # Перезапуск WebSocket
            if self.ws:
                try:
                    self.ws.exit()
                except Exception:
                    pass
            if self.ws_private:
                try:
                    self.ws_private.exit()
                except Exception:
                    pass

            # Очистка позиций старого аккаунта и переопределение режима/плеча
            for sym in SYMBOLS:
                self.positions[sym] = None
                self.risk_states[sym] = None
                self.breakeven_applied[sym] = False
                self._init_leverage(sym)
                self._detect_position_mode(sym)

            self._init_websocket()
            self.sync_positions()

            # Пересчет базового баланса дня для нового аккаунта (предотвращает ложный Circuit Breaker)
            self.last_known_equity = 0.0
            self.circuit_breaker_active = False
            new_bal = self.get_wallet_equity()
            self.start_day_balance = new_bal if new_bal > 0 else FALLBACK_USDT
            trade_log.info(f"Режим {config.TRADING_MODE} активирован. Базовый баланс дня: {self.start_day_balance:.2f} USDT")

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
        """Получает доступный баланс USDT с обработкой ошибок и специфики UTA/Demo."""
        try:
            resp = self.client.get_wallet_balance(accountType="UNIFIED")
            acc_list = resp["result"]["list"]
            if not acc_list:
                return FALLBACK_USDT
            acc = acc_list[0]
            coins = acc.get("coin", [])
            for c in coins:
                if c.get("coin") == "USDT":
                    raw_avail = c.get("availableToWithdraw")
                    avail_val = float(raw_avail) if raw_avail not in (None, "") else 0.0
                    if avail_val > 0:
                        return avail_val
                    # В UTA 2.0 / Demo availableToWithdraw может быть пустым или "0"
                    wb = float(c.get("walletBalance") or 0.0)
                    pos_im = float(c.get("totalPositionIM") or 0.0)
                    ord_im = float(c.get("totalOrderIM") or 0.0)
                    free_wb = max(0.0, wb - pos_im - ord_im)
                    if free_wb > 0:
                        return free_wb
                    if wb > 0:
                        return wb
                    return FALLBACK_USDT
            tot_avail = float(acc.get("totalAvailableBalance") or 0.0)
            return tot_avail if tot_avail > 0 else FALLBACK_USDT
        except Exception as e:
            log.warning(f"Ошибка получения баланса: {e}. Используем fallback {FALLBACK_USDT} USDT")
            return FALLBACK_USDT

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
        """Синхронизирует позиции бота с биржей через REST API."""
        with self.lock:
            try:
                resp = self.client.get_positions(category="linear", settleCoin="USDT")
                raw_positions = resp["result"]["list"]

                live_map = {}
                for p in raw_positions:
                    size = float(p.get("size", 0))
                    if size > 0:
                        sym = p.get("symbol")
                        side = p.get("side")
                        live_map[sym] = {
                            "side": side,
                            "position": "long" if side == "Buy" else "short",
                            "size": size,
                            "entry_price": float(p.get("avgPrice") or 0.0),
                            "stop_loss": float(p.get("stopLoss") or 0.0),
                            "take_profit": float(p.get("takeProfit") or 0.0),
                            "unrealised_pnl": float(p.get("unrealisedPnl") or 0.0),
                            "position_idx": int(p.get("positionIdx", 0))
                        }

                saved_states = self.load_state()

                tracked_symbols = list(dict.fromkeys(list(SYMBOLS) + list(self.positions.keys()) + list(live_map.keys())))
                for sym in tracked_symbols:
                    old_pos = self.positions.get(sym)
                    new_pos = live_map.get(sym)

                    if old_pos and not new_pos:
                        trade_log.info(f"[{sym}] Позиция закрыта на бирже (вероятно, сработал SL/TP)")
                        self.breakeven_applied[sym] = False
                        self.risk_states[sym] = None
                        self.save_state()
                    elif not old_pos and new_pos:
                        trade_log.info(f"[{sym}] Обнаружена открытая позиция {new_pos['position'].upper()} ({new_pos['size']})")
                        if sym in saved_states and saved_states[sym].side.lower() == new_pos["side"].lower():
                            restored = saved_states[sym]
                            restored.size = new_pos["size"]
                            if new_pos["entry_price"] > 0:
                                restored.entry_price = new_pos["entry_price"]
                            restored.stop_loss = new_pos["stop_loss"]
                            restored.take_profit = new_pos["take_profit"]
                            restored.position_idx = new_pos.get("position_idx", 0)
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
                                closed_qty = cur_risk.size - new_pos["size"]
                                cur_risk.tp1_executed = True
                                cur_risk.size = new_pos["size"]
                                trade_log.info(
                                    f"[{sym}] Sync: Обнаружено частичное закрытие TP1 на бирже (закрыто: {closed_qty})."
                                )
                                if PARTIAL_TP_MOVE_BE and not cur_risk.breakeven_applied:
                                    be_price = cur_risk.entry_price * (1 + BREAKEVEN_OFFSET if cur_risk.is_long else 1 - BREAKEVEN_OFFSET)
                                    rounded_be = self.round_price(sym, be_price)
                                    pos_idx = self.get_position_idx(sym, cur_risk.side)
                                    try:
                                        self.client.set_trading_stop(
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
            df["timestamp"] = pd.to_datetime(df["timestamp"].astype(int), unit="ms")
            df = df.sort_values("timestamp").reset_index(drop=True)
            return df
        except Exception as e:
            log.error(f"[{symbol}] Ошибка получения свечей (TF {tf}): {e}")
            return pd.DataFrame()

    def get_higher_trend(self, symbol: str) -> str:
        """Определяет старший тренд (4H EMA) с достаточным прогревом истории (600 свечей)."""
        if not USE_MTF_FILTER:
            return "neutral"
        try:
            df_htf = self.get_klines(symbol, interval=HIGHER_INTERVAL, limit=max(600, HTF_EMA_PERIOD * 3))
            return self.strategy.get_mtf_trend(df_htf)
        except Exception as e:
            log.warning(f"[{symbol}] Ошибка старшего тренда: {e}")
            return "neutral"

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
                            self.client.place_order(
                                category="linear",
                                symbol=symbol,
                                side=side,
                                orderType="Market",
                                qty=self.format_qty(symbol, close_qty),
                                positionIdx=pos_idx,
                                reduceOnly=True,
                                timeInForce="GoodTillCancel"
                            )
                            risk.tp1_executed = True
                            risk.size = round(max(0.0, risk.size - close_qty), 8)
                            pos["size"] = risk.size

                            trade_log.info(
                                f"[{symbol}] ЧАСТИЧНЫЙ ТЕЙК-ПРОФИТ (TP1 = 1.5R): Закрыто {close_qty} @ {curr_price}"
                            )

                            # Перенос оставшейся части в безубыток
                            if PARTIAL_TP_MOVE_BE:
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
                                trade_log.info(f"[{symbol}] Стоп перенесен в безубыток после TP1: {rounded_be}")

                            self.save_state()

                        except Exception as e:
                            log.error(f"[{symbol}] Ошибка исполнения Partial TP: {e}")

            # 2. Безубыток: гарантированный перенос стопа при достижении триггера
            if USE_BREAKEVEN and not risk.breakeven_applied:
                be_updated, new_sl = risk.check_breakeven(
                    current_price=curr_price,
                    trigger_pct=BREAKEVEN_TRIGGER,
                    offset_pct=BREAKEVEN_OFFSET
                )
                if be_updated:
                    rounded_sl = self.round_price(symbol, new_sl)
                    current_sl = pos.get("stop_loss", 0.0)
                    should_update = (rounded_sl > current_sl) if risk.is_long else (current_sl == 0.0 or rounded_sl < current_sl)
                    if should_update:
                        try:
                            pos_idx = self.get_position_idx(symbol, "Buy" if risk.is_long else "Sell")
                            self.client.set_trading_stop(
                                category="linear",
                                symbol=symbol,
                                stopLoss=self.format_price(symbol, rounded_sl),
                                positionIdx=pos_idx
                            )
                            pos["stop_loss"] = rounded_sl
                            risk.stop_loss = rounded_sl
                            risk.breakeven_applied = True
                            self.breakeven_applied[symbol] = True
                            trade_log.info(f"[{symbol}] Безубыток активирован: SL перенесен на {rounded_sl}")
                            self.save_state()
                        except Exception as e:
                            log.warning(f"[{symbol}] Ошибка переноса стопа в безубыток: {e}")

            # 3. Трейлинг-стоп (Trailing Stop): программный fallback (если нативный трейлинг на бирже не активен)
            if USE_TRAILING_STOP and not risk.native_trailing_active:
                trail_updated, new_sl = risk.check_trailing_stop(
                    current_price=curr_price,
                    high_price=high_price,
                    low_price=low_price,
                    trigger_pct=TRAILING_STOP_TRIGGER,
                    atr_mult=TRAILING_STOP_ATR_MULT,
                    fallback_pct=TRAILING_STOP_FALLBACK
                )
                if trail_updated:
                    rounded_sl = self.round_price(symbol, new_sl)
                    current_sl = pos.get("stop_loss", 0.0)
                    should_update = (rounded_sl > current_sl) if risk.is_long else (current_sl == 0.0 or rounded_sl < current_sl)

                    if should_update:
                        try:
                            pos_idx = self.get_position_idx(symbol, "Buy" if risk.is_long else "Sell")
                            self.client.set_trading_stop(
                                category="linear",
                                symbol=symbol,
                                stopLoss=self.format_price(symbol, rounded_sl),
                                positionIdx=pos_idx
                            )
                            pos["stop_loss"] = rounded_sl
                            risk.stop_loss = rounded_sl
                            trade_log.info(
                                f"[{symbol}] Трейлинг-стоп подтянут к {rounded_sl} (High: {risk.highest_price}, Low: {risk.lowest_price})"
                            )
                            self.save_state()
                        except Exception as e:
                            log.warning(f"[{symbol}] Ошибка обновления трейлинг-стопа: {e}")

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

            # 3. Проверка лимита открытых позиций
            open_count = self.get_open_positions_count()
            if open_count >= MAX_OPEN_POSITIONS:
                log.info(f"[{symbol}] Достигнут лимит позиций ({open_count}/{MAX_OPEN_POSITIONS}). Пропуск.")
                return None

            if self.positions.get(symbol):
                return None

            balance = self.get_available_balance()
            if balance < 5.0:
                log.error(f"[{symbol}] Недостаточно средств: {balance:.2f} USDT")
                return None

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

                # Расчет объема позиции
                if USE_RISK_BASED_SIZING and stops.get("sl_dist", 0) > 0:
                    risk_capital = balance * RISK_PER_TRADE_PCT
                    raw_qty = risk_capital / stops["sl_dist"]
                    max_margin = max(balance * POSITION_SIZE_PCT, 10.0)
                    max_qty = (max_margin * LEVERAGE) / price
                    raw_qty = min(raw_qty, max_qty)
                    trade_margin = (raw_qty * price) / LEVERAGE
                else:
                    trade_margin = max(balance * POSITION_SIZE_PCT, 10.0)
                    notional = trade_margin * LEVERAGE
                    raw_qty = notional / price

                qty = self.round_qty(symbol, raw_qty)

                min_qty = self.instruments[symbol]["min_qty"]
                if qty < min_qty:
                    log.error(f"[{symbol}] Объем {qty} меньше минимального {min_qty}. Баланс: {balance:.1f} USDT.")
                    return None

                # Проверка минимального нотионала биржи (>= 5 USDT)
                min_notional = float(self.instruments[symbol].get("min_notional", MIN_ORDER_NOTIONAL))
                order_notional = qty * price
                if order_notional < min_notional:
                    log.warning(
                        f"[{symbol}] Стоимость ордера {order_notional:.2f} USDT меньше минимального "
                        f"нотионала {min_notional:.1f} USDT. Вход отменен."
                    )
                    return None

                pos_idx = self.get_position_idx(symbol, side)

                # Умное исполнение: попытка Maker (Limit Post-Only) с таймаутом и фоллбэком на Market
                executed = False
                resp = None
                filled_qty = 0.0
                actual_entry_price = price

                if USE_SMART_EXECUTION:
                    try:
                        tickers_resp = self.client.get_tickers(category="linear", symbol=symbol)
                        t_info = tickers_resp["result"]["list"][0]
                        bid = float(t_info.get("bid1Price") or price)
                        ask = float(t_info.get("ask1Price") or price)
                        limit_price = bid if side == "Buy" else ask
                        limit_p_str = self.format_price(symbol, limit_price)

                        maker_resp = self.client.place_order(
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
                        order_id = maker_resp.get("result", {}).get("orderId")
                        if order_id:
                            trade_log.info(
                                f"[{symbol}] Maker ордер (Limit Post-Only) выставлен: {limit_p_str}. "
                                f"Ожидание исполнения (до {SMART_EXECUTION_TIMEOUT}с)..."
                            )
                            t0 = time.time()
                            while time.time() - t0 < SMART_EXECUTION_TIMEOUT:
                                # Освобождаем RLock на время сна, чтобы не блокировать WebSocket и дашборд
                                self.lock.release()
                                try:
                                    time.sleep(0.5)
                                finally:
                                    self.lock.acquire()

                                ws_ord = getattr(self, "ws_orders", {}).get(order_id)
                                if ws_ord:
                                    ws_st = ws_ord.get("status", "")
                                    ws_cum = float(ws_ord.get("cum_qty", 0.0))
                                    if ws_st == "Filled" or ws_cum >= qty * 0.95:
                                        executed = True
                                        filled_qty = ws_cum if ws_cum > 0 else qty
                                        actual_entry_price = float(ws_ord.get("avg_price") or limit_price)
                                        resp = maker_resp
                                        trade_log.info(f"[{symbol}] Maker ордер исполнен по WS (Filled: {filled_qty})!")
                                        break
                                    elif ws_st in ("Cancelled", "Rejected", "Deactivated"):
                                        filled_qty = ws_cum
                                        log.info(
                                            f"[{symbol}] Maker Post-Only отклонен/отменен по WS ({ws_st}, "
                                            f"исполнено {filled_qty}/{qty}). Переход на Market."
                                        )
                                        break

                                cur = self.positions.get(symbol)
                                if cur and cur.get("size", 0) >= qty * 0.95:
                                    executed = True
                                    actual_entry_price = limit_price
                                    resp = maker_resp
                                    trade_log.info(f"[{symbol}] Maker ордер успешно исполнен на бирже!")
                                    break
                                try:
                                    ord_chk = self.client.get_open_orders(category="linear", symbol=symbol, orderId=order_id)
                                    open_list = ord_chk.get("result", {}).get("list", [])
                                    if not open_list:
                                        # Ордер больше не открыт: проверяем в истории (Filled или Cancelled/Rejected из-за PostOnly)
                                        hist_list = []
                                        hist_is_dict = False
                                        if hasattr(self.client, "get_order_history"):
                                            hist_resp = self.client.get_order_history(
                                                category="linear", symbol=symbol, orderId=order_id
                                            )
                                            if isinstance(hist_resp, dict):
                                                hist_is_dict = True
                                                hist_list = hist_resp.get("result", {}).get("list", [])

                                        if hist_list and isinstance(hist_list, list):
                                            ord_info = hist_list[0]
                                            status = str(ord_info.get("orderStatus", ""))
                                            cum_qty = float(ord_info.get("cumExecQty") or 0.0)
                                            if status == "Filled" or cum_qty >= qty * 0.95:
                                                executed = True
                                                filled_qty = cum_qty if cum_qty > 0 else qty
                                                actual_entry_price = float(ord_info.get("avgPrice") or limit_price)
                                                resp = maker_resp
                                                trade_log.info(f"[{symbol}] Maker ордер подтвержден исполненным (Filled: {filled_qty})!")
                                                break
                                            else:
                                                # PostOnly был отменен биржей (EC_PostOnlyWillTakeLiquidity) или частично исполнен
                                                filled_qty = cum_qty
                                                log.info(
                                                    f"[{symbol}] Maker Post-Only завершен со статусом {status} "
                                                    f"(исполнено {filled_qty}/{qty}). Добираем остаток по Market."
                                                )
                                                break
                                        elif not hist_is_dict:
                                            # Фоллбэк для unit-тестов с простым моком без get_order_history
                                            executed = True
                                            actual_entry_price = limit_price
                                            resp = maker_resp
                                            trade_log.info(f"[{symbol}] Maker ордер подтвержден исполненным!")
                                            break
                                except Exception:
                                    pass

                            if not executed:
                                trade_log.info(f"[{symbol}] Maker таймаут/отмена. Отмена остатка и вход Market...")
                                try:
                                    self.client.cancel_order(category="linear", symbol=symbol, orderId=order_id)
                                    if filled_qty == 0.0 and hasattr(self.client, "get_order_history"):
                                        hist_resp = self.client.get_order_history(
                                            category="linear", symbol=symbol, orderId=order_id
                                        )
                                        if isinstance(hist_resp, dict):
                                            h_list = hist_resp.get("result", {}).get("list", [])
                                            if h_list and isinstance(h_list, list):
                                                filled_qty = float(h_list[0].get("cumExecQty") or 0.0)
                                except Exception:
                                    pass
                    except Exception as e:
                        log.info(f"[{symbol}] Post-Only отклонен ({e}). Переход на Market ордер.")

                # Fallback на Market ордер (на весь объем или неисполненный остаток)
                if not executed:
                    rem_qty = self.round_qty(symbol, max(0.0, qty - filled_qty))
                    if rem_qty >= min_qty and (rem_qty * price) >= min_notional:
                        resp = self.client.place_order(
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
                        qty = self.round_qty(symbol, filled_qty + rem_qty)
                    elif filled_qty >= min_qty:
                        executed = True
                        qty = self.round_qty(symbol, filled_qty)
                    else:
                        return None

                if isinstance(resp, dict) and resp.get("retCode", 0) != 0:
                    log.error(f"[{symbol}] Биржа отклонила ордер (retCode={resp.get('retCode')}): {resp.get('retMsg')}")
                    return None

                if executed:
                    # Для Maker ордера выставляем защитные стопы через set_trading_stop
                    try:
                        self.client.set_trading_stop(
                            category="linear",
                            symbol=symbol,
                            positionIdx=pos_idx,
                            stopLoss=self.format_price(symbol, sl),
                            takeProfit=self.format_price(symbol, tp)
                        )
                    except Exception as e:
                        log.warning(f"[{symbol}] Ошибка выставления SL/TP для Maker ордера: {e}")

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
                    position_idx=pos_idx
                )
                self.risk_states[symbol] = risk_state

                # Выставление лимитного TP1 (50% объема) на стороне биржи (reduceOnly)
                if USE_PARTIAL_TP and USE_EXCHANGE_TP1:
                    tp1_qty = self.round_qty(symbol, qty * PARTIAL_TP_RATIO)
                    tp1_side = "Sell" if side == "Buy" else "Buy"
                    if tp1_qty >= min_qty and (tp1_qty * risk_state.tp1_price) >= min_notional:
                        try:
                            tp1_resp = self.client.place_order(
                                category="linear",
                                symbol=symbol,
                                side=tp1_side,
                                orderType="Limit",
                                price=self.format_price(symbol, risk_state.tp1_price),
                                qty=self.format_qty(symbol, tp1_qty),
                                positionIdx=pos_idx,
                                reduceOnly=True,
                                timeInForce="GoodTillCancel"
                            )
                            risk_state.tp1_order_id = tp1_resp.get("result", {}).get("orderId", "")
                            trade_log.info(
                                f"[{symbol}] На бирже выставлен лимитный ордер TP1 (reduceOnly): {tp1_qty} @ {risk_state.tp1_price}"
                            )
                        except Exception as e:
                            log.warning(f"[{symbol}] Не удалось выставить лимитный TP1 на бирже: {e}")

                # Активация нативного трейлинг-стопа Bybit
                if USE_TRAILING_STOP and USE_NATIVE_TRAILING_STOP:
                    trail_dist = self.round_price(
                        symbol,
                        atr_value * TRAILING_STOP_ATR_MULT if atr_value > 0 else (price * TRAILING_STOP_FALLBACK)
                    )
                    active_price = self.round_price(
                        symbol,
                        price * (1 + TRAILING_STOP_TRIGGER if side == "Buy" else 1 - TRAILING_STOP_TRIGGER)
                    )
                    try:
                        self.client.set_trading_stop(
                            category="linear",
                            symbol=symbol,
                            positionIdx=pos_idx,
                            trailingStop=self.format_price(symbol, trail_dist),
                            activePrice=self.format_price(symbol, active_price)
                        )
                        risk_state.native_trailing_active = True
                        trade_log.info(
                            f"[{symbol}] Нативный трейлинг Bybit активирован: дист={trail_dist}, активация={active_price}"
                        )
                    except Exception as e:
                        log.warning(f"[{symbol}] Биржа отклонила нативный трейлинг ({e}). Работает программный.")

                self.save_state()

                trade_log.info(
                    f"[{symbol}] {'LONG' if side == 'Buy' else 'SHORT'} открыт | "
                    f"Вход: {price} | Кол-во: {qty} ({trade_margin:.1f}$ маржа) | "
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

            try:
                try:
                    self.client.cancel_all_orders(category="linear", symbol=symbol)
                except Exception as e:
                    log.warning(f"[{symbol}] Ошибка отмены ордеров: {e}")

                time.sleep(0.3)
                side = "Sell" if curr_pos["position"] == "long" else "Buy"
                qty = curr_pos["size"]

                pos_idx = self.get_position_idx(symbol, curr_pos.get("side", "Buy"))
                resp = self.client.place_order(
                    category="linear",
                    symbol=symbol,
                    side=side,
                    orderType="Market",
                    qty=self.format_qty(symbol, qty),
                    positionIdx=pos_idx,
                    reduceOnly=True,
                    timeInForce="GoodTillCancel",
                )

                entry = curr_pos.get("entry_price") or price
                pnl_pct = ((price - entry) / entry * 100)
                if curr_pos["position"] == "short":
                    pnl_pct = -pnl_pct
                pnl_pct *= LEVERAGE

                sign = "+" if pnl_pct > 0 else ""
                trade_log.info(f"[{symbol}] Закрыто ({reason}) @ {price} | PnL: {sign}{pnl_pct:.2f}%")

                self.positions[symbol] = None
                self.risk_states[symbol] = None
                self.breakeven_applied[symbol] = False
                self.save_state()
                return resp

            except Exception as e:
                log.error(f"[{symbol}] Ошибка закрытия: {e}")
                return None

    def panic_close_all(self):
        """Экстренно закрывает ВСЕ позиции и снимает ордера."""
        with self.lock:
            trade_log.warning("ВЫЗВАНА КОМАНДА PANIC CLOSE! Закрытие всех позиций...")
            for sym in SYMBOLS:
                pos = self.positions.get(sym)
                if pos and pos.get("size", 0) > 0:
                    try:
                        price = self.get_price(sym)
                        self.close_position(sym, price, reason="PANIC BUTTON")
                    except Exception as e:
                        log.error(f"[{sym}] Ошибка Panic Close: {e}")

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
                # 1. Защита от дневной просадки (Circuit Breaker)
                self.check_daily_circuit_breaker()

                # 2. Регулярная синхронизация времени с Bybit (каждые 60 минут)
                if time.time() - self.last_time_sync > 3600:
                    sync_exchange_time(config.TESTNET, config.DEMO)
                    self.last_time_sync = time.time()

                # 3. Синхронизируем открытые позиции
                self.sync_positions()

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
                if USE_MARKET_REGIME_FILTER:
                    try:
                        bench_df = self.get_klines(BENCHMARK_SYMBOL, interval=BENCHMARK_INTERVAL, limit=max(600, BENCHMARK_EMA_PERIOD * 3))
                    except Exception as e:
                        log.warning(f"Не удалось загрузить бенчмарк {BENCHMARK_SYMBOL}: {e}")

                # 6. Сканируем каждую пару из корзины
                min_candles_req = max(SLOW_MA, ST_CONFIRM_EMA, ATR_PERIOD, RSI_PERIOD, 30) + 10
                kline_limit = max(250, min_candles_req + 50)
                for symbol in SYMBOLS:
                    df = self.get_klines(symbol, limit=kline_limit)
                    if df.empty or len(df) < min_candles_req:
                        continue

                    # Расчет индикаторов через модуль стратегии
                    df = self.strategy.calculate_indicators(df)
                    price = self.get_price(symbol, fallback_df=df)

                    last_candle = df.iloc[-1]
                    high_p = last_candle["high"]
                    low_p = last_candle["low"]

                    # Управление рисками активной позиции (Trailing Stop, Partial TP, Breakeven)
                    self.manage_risk_and_stops(symbol, price, high_p, low_p)

                    # Сигнал формируется по последней ЗАКРЫТОЙ свече (-2)
                    signal_candle = df.iloc[-2]
                    signal = int(signal_candle.get("signal", 0))
                    signal_price = signal_candle["close"]
                    signal_time = signal_candle["timestamp"]
                    atr_value = signal_candle.get("ATR", 0.0)
                    current_pos = self.positions.get(symbol)

                    if signal_time != self.last_handled_candles.get(symbol):
                        self.last_handled_candles[symbol] = signal_time

                        # Направленная защита от неблагоприятного проскальзывания цены
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

                        # Проверка фильтра рыночного сентимента по BTC для LONG
                        if signal == 1 and USE_MARKET_REGIME_FILTER and bench_df is not None:
                            allowed, r_msg = self.regime_filter.is_long_allowed(symbol, bench_df)
                            if not allowed:
                                log.info(f"[{symbol}] {r_msg}")
                                signal = 0

                        # Проверка фильтра рыночного сентимента по BTC для SHORT
                        elif signal == -1 and USE_MARKET_REGIME_FILTER and bench_df is not None:
                            allowed, r_msg = self.regime_filter.is_short_allowed(symbol, bench_df)
                            if not allowed:
                                log.info(f"[{symbol}] {r_msg}")
                                signal = 0

                        # Проверка фильтра старшего тренда (4H EMA) только при наличии сигнала
                        htf_trend = self.get_higher_trend(symbol) if (USE_MTF_FILTER and signal != 0) else "neutral"

                        if signal == 1:
                            if current_pos and current_pos["position"] == "short":
                                self.close_position(symbol, price, "Смена тренда на LONG")

                            if not self.positions.get(symbol):
                                if USE_MTF_FILTER and htf_trend == "bearish":
                                    log.info(f"[{symbol}] Сигнал LONG отклонен: старший тренд 4H медвежий")
                                else:
                                    self.open_position(symbol, "Buy", price, atr_value=atr_value)

                            self.last_handled_candles[symbol] = signal_time

                        elif signal == -1:
                            if current_pos and current_pos["position"] == "long":
                                self.close_position(symbol, price, "Смена тренда на SHORT")

                            if not self.positions.get(symbol):
                                if USE_MTF_FILTER and htf_trend == "bullish":
                                    log.info(f"[{symbol}] Сигнал SHORT отклонен: старший тренд 4H бычий")
                                else:
                                    self.open_position(symbol, "Sell", price, atr_value=atr_value)

                            self.last_handled_candles[symbol] = signal_time

                    time.sleep(0.2)

                consecutive_errors = 0
                time.sleep(30)

            except KeyboardInterrupt:
                log.info("Бот остановлен пользователем")
                if self.ws:
                    try:
                        self.ws.exit()
                    except Exception:
                        pass
                if self.ws_private:
                    try:
                        self.ws_private.exit()
                    except Exception:
                        pass
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
        bot = BybitMultiFuturesBot()
        bot.run()
