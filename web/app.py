"""
FastAPI веб-сервер легковесного локального дашборда Bybit Bot (v2.3).
Разработан специально под Termux (Android) с минимальным потреблением CPU и RAM.
Включает:
- Real-time WebSocket соединение (/ws) с неблокирующим executor и TTL-кэшем статуса
- Эндпоинты для интерактивных графиков свечей (/api/candles) и честной кривой капитала (/api/equity_history)
- Полноценный потоковый парсер истории сделок и портфельный бэктестер
"""

from fastapi import FastAPI, Request, Query, Body, Depends, HTTPException, WebSocketException, status, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse, PlainTextResponse
from fastapi.security import HTTPBasic
from fastapi.templating import Jinja2Templates
from starlette.requests import HTTPConnection
from pathlib import Path
from collections import deque
import base64
import logging
import asyncio
import secrets
import time
import threading
from typing import Optional
import pandas as pd

from config import (
    TRADING_MODE, SYMBOLS, ACTIVE_STRATEGY, LEVERAGE, MAX_OPEN_POSITIONS,
    API_KEY, API_SECRET, DEMO, TESTNET, FALLBACK_USDT
)
from web.state import (
    get_bot_instance, is_bot_paused, set_bot_paused, parse_trade_log,
    parse_equity_history, write_control_state, read_control_state, read_live_status
)
import config

log = logging.getLogger("bot")

security = HTTPBasic(auto_error=False)


def verify_dashboard_auth(conn: HTTPConnection):
    """
    Проверка HTTP Basic Auth (поддерживает как обычные HTTP-запросы, так и WebSocket /ws),
    если логин/пароль заданы в .env.
    Если параметры не заданы (пустые), доступ открыт для локального использования.
    """
    user = config.DASHBOARD_USERNAME
    pwd = config.DASHBOARD_PASSWORD
    if not user and not pwd:
        return True

    auth_header = conn.headers.get("Authorization", "")
    req_user, req_pwd = None, None
    if auth_header.lower().startswith("basic "):
        try:
            decoded = base64.b64decode(auth_header[6:].strip()).decode("utf-8")
            if ":" in decoded:
                req_user, req_pwd = decoded.split(":", 1)
        except Exception:
            req_user, req_pwd = None, None

    if req_user is None or req_pwd is None:
        if conn.scope.get("type") == "websocket":
            raise WebSocketException(code=status.WS_1008_POLICY_VIOLATION)
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Необходима авторизация",
            headers={"WWW-Authenticate": "Basic"},
        )

    correct_user = secrets.compare_digest(req_user, user)
    correct_pwd = secrets.compare_digest(req_pwd, pwd)
    if not (correct_user and correct_pwd):
        if conn.scope.get("type") == "websocket":
            raise WebSocketException(code=status.WS_1008_POLICY_VIOLATION)
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Неверный логин или пароль",
            headers={"WWW-Authenticate": "Basic"},
        )
    return True


app = FastAPI(
    title="Bybit Bot v2.3 Dashboard",
    dependencies=[Depends(verify_dashboard_auth)]
)

templates_dir = Path(__file__).resolve().parent / "templates"
templates = Jinja2Templates(directory=str(templates_dir))

# Кэш цен и статуса для предотвращения перегрузки Bybit API
_PRICE_CACHE = {}
_PRICE_CACHE_TTL = 3.0
_STATUS_CACHE = {"key": None, "ts": 0.0, "data": None}
_STATUS_CACHE_TTL = 1.5
_CACHE_LOCK = threading.Lock()


def _invalidate_status_cache():
    with _CACHE_LOCK:
        _STATUS_CACHE["key"] = None
        _STATUS_CACHE["ts"] = 0.0
        _STATUS_CACHE["data"] = None


def get_cached_price(bot_or_client, symbol: str) -> float:
    now = time.time()
    with _CACHE_LOCK:
        cached = _PRICE_CACHE.get(symbol)
        if cached and (now - cached[1] < _PRICE_CACHE_TTL):
            return cached[0]

    if hasattr(bot_or_client, "get_price"):
        p = bot_or_client.get_price(symbol)
    else:
        resp = bot_or_client.get_tickers(category="linear", symbol=symbol)
        p = float(resp["result"]["list"][0]["lastPrice"])

    with _CACHE_LOCK:
        _PRICE_CACHE[symbol] = (p, now)
    return p


_STANDALONE_CLIENT = None
_STANDALONE_LOCK = threading.Lock()
_LAST_STANDALONE_SYNC = 0.0


def get_standalone_client():
    """Создает и кэширует надежный HTTP клиент Bybit для standalone режима со синхронизацией времени раз в час."""
    global _STANDALONE_CLIENT, _LAST_STANDALONE_SYNC
    now = time.time()
    with _STANDALONE_LOCK:
        if _STANDALONE_CLIENT is None or (now - _LAST_STANDALONE_SYNC > 3600):
            from pybit.unified_trading import HTTP
            from pybit import _helpers
            try:
                unauth = HTTP(testnet=config.TESTNET, demo=config.DEMO)
                server_t = int(unauth.get_server_time()["time"])
                local_t = int(time.time() * 1000)
                offset = server_t - local_t
                _helpers.generate_timestamp = lambda: int(time.time() * 1000) + offset
            except Exception as e:
                log.warning(f"Standalone sync exchange time: {e}")

            _STANDALONE_CLIENT = HTTP(
                testnet=config.TESTNET,
                api_key=config.API_KEY,
                api_secret=config.API_SECRET,
                demo=config.DEMO,
                recv_window=10000,
                timeout=25,
                max_retries=5,
                retry_delay=2
            )
            _LAST_STANDALONE_SYNC = now
        return _STANDALONE_CLIENT


@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    """Главная страница дашборда."""
    ctx = {"request": request, "symbols": config.SYMBOLS}
    try:
        return templates.TemplateResponse(request=request, name="index.html", context=ctx)
    except TypeError:
        return templates.TemplateResponse("index.html", ctx)


@app.websocket("/ws")
async def websocket_dashboard(ws: WebSocket):
    """
    WebSocket эндпоинт для real-time пуша статуса бота в дашборд.
    """
    await ws.accept()
    try:
        while True:
            status_data = await get_status()
            await ws.send_json(status_data)
            try:
                msg = await asyncio.wait_for(ws.receive(), timeout=1.5)
                if msg.get("type") == "websocket.disconnect":
                    break
            except asyncio.TimeoutError:
                pass
    except (WebSocketDisconnect, asyncio.CancelledError):
        pass
    except Exception as e:
        log.debug(f"Dashboard WS disconnected: {e}")


def _compute_status_sync() -> dict:
    bot = get_bot_instance()
    mode = config.TRADING_MODE
    paused = is_bot_paused()
    is_mock_bot = bot is not None and "unittest.mock" in type(bot).__module__

    if bot is None:
        ctrl = read_control_state()
        if ctrl.get("trading_mode") in config.VALID_TRADING_MODES:
            mode = ctrl["trading_mode"]

    cache_key = (id(bot) if bot is not None else 0, mode, paused)
    now = time.time()
    if not is_mock_bot:
        with _CACHE_LOCK:
            if (
                _STATUS_CACHE["data"] is not None
                and _STATUS_CACHE["key"] == cache_key
                and (now - _STATUS_CACHE["ts"] < _STATUS_CACHE_TTL)
            ):
                return _STATUS_CACHE["data"]

    cb_active = False
    balance = 0.0
    equity = 0.0
    unrealized_pnl = 0.0
    daily_pnl = 0.0
    positions_data = []

    if bot is not None:
        try:
            balance = float(bot.get_available_balance())
            cb_active = bool(getattr(bot, "circuit_breaker_active", False))

            with getattr(bot, "lock", threading.RLock()):
                positions_items = [(s, dict(p) if p else None) for s, p in bot.positions.items()]

            for sym, pos in positions_items:
                if pos:
                    try:
                        curr_price = get_cached_price(bot, sym)
                    except Exception:
                        curr_price = pos.get("entry_price", 0.0)

                    entry = float(pos.get("entry_price", 0.0) or 0.0)
                    size = float(pos.get("size", 0.0) or 0.0)
                    is_long = pos.get("position") == "long"

                    if entry > 0 and size > 0:
                        diff = (curr_price - entry) if is_long else (entry - curr_price)
                        pnl_usd = diff * size
                        pnl_pct = (diff / entry) * 100 * LEVERAGE
                        unrealized_pnl += pnl_usd
                    else:
                        pnl_usd = 0.0
                        pnl_pct = 0.0

                    positions_data.append({
                        "symbol": sym,
                        "side": pos.get("side", "Buy"),
                        "size": size,
                        "entry_price": entry,
                        "mark_price": curr_price,
                        "pnl": round(pnl_usd, 2),
                        "pnl_pct": round(pnl_pct, 2),
                        "stop_loss": pos.get("stop_loss", 0.0),
                        "take_profit": pos.get("take_profit", 0.0),
                        "breakeven": bot.breakeven_applied.get(sym, False) if hasattr(bot, "breakeven_applied") else False
                    })

            raw_eq = bot.get_wallet_equity() if hasattr(bot, "get_wallet_equity") else None
            if isinstance(raw_eq, (int, float)) and raw_eq > 0:
                equity = float(raw_eq)
            else:
                equity = balance + unrealized_pnl

            start_bal = getattr(bot, "start_day_balance", equity)
            if isinstance(start_bal, (int, float)) and start_bal > 0:
                daily_pnl = equity - float(start_bal)
            else:
                daily_pnl = 0.0

        except Exception as e:
            log.warning(f"Ошибка получения статуса из bot instance: {e}")
    else:
        live_snap = read_live_status()
        cb_active = bool(live_snap.get("circuit_breaker", False))
        be_map = live_snap.get("breakeven_map", {})
        start_bal = float(live_snap.get("start_day_balance") or 0.0)
        try:
            client = get_standalone_client()
            resp = client.get_wallet_balance(accountType="UNIFIED")
            acc_list = resp.get("result", {}).get("list", [])
            if acc_list:
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
                        if tot_avail is not None:
                            balance = min(tot_avail, free_wb) if wb > 0 and (pos_im > 0 or ord_im > 0) else tot_avail
                        elif avail_val is not None:
                            balance = min(avail_val, free_wb) if wb > 0 and (pos_im > 0 or ord_im > 0) else avail_val
                        else:
                            balance = free_wb
                        equity = float(c.get("equity") or wb or balance)
                        break
                else:
                    if tot_avail is not None:
                        balance = tot_avail
                        equity = float(acc.get("totalEquity") or tot_avail)

            pos_resp = client.get_positions(category="linear", settleCoin="USDT")
            raw_pos = pos_resp.get("result", {}).get("list", [])
            for p in raw_pos:
                sz = float(p.get("size", 0))
                if sz > 0:
                    sym = p.get("symbol")
                    side = p.get("side")
                    entry = float(p.get("avgPrice") or 0.0)
                    mark = float(p.get("markPrice") or entry)
                    pnl_usd = float(p.get("unrealisedPnl") or 0.0)
                    unrealized_pnl += pnl_usd

                    is_long = side == "Buy"
                    diff = (mark - entry) if is_long else (entry - mark)
                    pnl_pct = (diff / entry * 100 * LEVERAGE) if entry > 0 else 0.0

                    positions_data.append({
                        "symbol": sym,
                        "side": side,
                        "size": sz,
                        "entry_price": entry,
                        "mark_price": mark,
                        "pnl": round(pnl_usd, 2),
                        "pnl_pct": round(pnl_pct, 2),
                        "stop_loss": float(p.get("stopLoss") or 0.0),
                        "take_profit": float(p.get("takeProfit") or 0.0),
                        "breakeven": bool(be_map.get(sym, False))
                    })

            if start_bal > 0 and equity > 0:
                daily_pnl = equity - start_bal
        except Exception as e:
            log.warning(f"Standalone статус Bybit: {e}")
            if live_snap:
                balance = float(live_snap.get("balance") or FALLBACK_USDT)
                equity = float(live_snap.get("equity") or balance)
                daily_pnl = float(live_snap.get("daily_pnl") or 0.0)

    base_cap = equity if equity > 0 else balance
    unrealized_pct = (unrealized_pnl / base_cap * 100) if base_cap > 0 else 0.0

    result = {
        "mode": mode,
        "is_paused": paused,
        "circuit_breaker": cb_active,
        "balance": round(balance, 2),
        "equity": round(equity, 2),
        "unrealized_pnl": round(unrealized_pnl, 2),
        "unrealized_pnl_pct": round(unrealized_pct, 2),
        "daily_pnl": round(daily_pnl, 2),
        "strategy": config.ACTIVE_STRATEGY,
        "leverage": LEVERAGE,
        "open_positions_count": len(positions_data),
        "max_positions": MAX_OPEN_POSITIONS,
        "symbols": config.SYMBOLS,
        "positions": positions_data
    }

    if not is_mock_bot:
        with _CACHE_LOCK:
            _STATUS_CACHE["key"] = cache_key
            _STATUS_CACHE["ts"] = time.time()
            _STATUS_CACHE["data"] = result
    return result


@app.get("/api/status")
async def get_status():
    """Возвращает актуальный статус бота без блокировки asyncio event loop."""
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(None, _compute_status_sync)


def _get_candles_sync(symbol: str, interval: str, limit: int) -> list[dict]:
    sym = symbol.strip().upper()
    limit = max(1, min(1000, int(limit)))
    bot = get_bot_instance()
    df = pd.DataFrame()
    if bot is not None and hasattr(bot, "get_klines"):
        try:
            df = bot.get_klines(sym, interval=interval, limit=limit)
        except Exception as e:
            log.warning(f"Ошибка получения свечей для графика из bot: {e}")

    if df.empty:
        try:
            client = get_standalone_client()
            resp = client.get_kline(category="linear", symbol=sym, interval=interval, limit=limit)
            raw = resp["result"]["list"]
            df = pd.DataFrame(raw, columns=["timestamp", "open", "high", "low", "close", "volume", "turnover"])
            df = df.astype({"open": float, "high": float, "low": float, "close": float, "volume": float})
            df["timestamp"] = pd.to_datetime(df["timestamp"].astype("int64"), unit="ms")
            df = df.sort_values("timestamp").reset_index(drop=True)
        except Exception as e:
            log.warning(f"Ошибка получения свечей из standalone client: {e}")

    if df.empty:
        return []

    candles = []
    for _, row in df.iterrows():
        t = int(row["timestamp"].timestamp())
        candles.append({
            "time": t,
            "open": float(row["open"]),
            "high": float(row["high"]),
            "low": float(row["low"]),
            "close": float(row["close"]),
            "volume": round(float(row["volume"]), 4)
        })
    return candles


@app.get("/api/candles")
async def get_candles(
    symbol: str = Query("BTCUSDT"),
    interval: str = Query("60"),
    limit: int = Query(120)
):
    """Возвращает свечные данные (OHLCV) для графиков TradingView Lightweight Charts."""
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(None, lambda: _get_candles_sync(symbol, interval, limit))


@app.get("/api/equity_history")
async def get_equity_history():
    """
    Возвращает историческую кривую капитала (Equity Curve)
    на основе реальных снимков капитала и взвешенного PnL из trade.log.
    """
    initial_balance = config.FALLBACK_USDT
    bot = get_bot_instance()
    if bot is not None:
        sb = getattr(bot, "start_day_balance", initial_balance)
        if isinstance(sb, (int, float)) and sb > 0:
            initial_balance = float(sb)

    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(
        None,
        lambda: parse_equity_history(initial_balance=initial_balance, margin_weight=config.POSITION_SIZE_PCT)
    )


@app.get("/api/trades")
async def get_trades():
    """Возвращает историю реальных сделок из trade.log."""
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(None, lambda: parse_trade_log(limit=60))


@app.post("/api/pause")
async def toggle_pause():
    """Переключает состояние паузы (Emergency Pause)."""
    curr = is_bot_paused()
    set_bot_paused(not curr)
    new_state = not curr
    _invalidate_status_cache()
    log.info(f"Дашборд: Бот переведен в состояние {'ПАУЗА' if new_state else 'АКТИВЕН'}")
    return {"status": "ok", "is_paused": new_state}


def _close_exchange_position_safe(client, sym: str, pos_dict: dict):
    """Закрывает позицию по маркету ДО полной отмены защитных стопов."""
    sz = float(pos_dict.get("size", 0))
    if sz <= 0:
        return
    side = "Sell" if pos_dict.get("side") == "Buy" else "Buy"
    pos_idx = int(pos_dict.get("positionIdx", 0))
    resp = client.place_order(
        category="linear",
        symbol=sym,
        side=side,
        orderType="Market",
        qty=str(sz),
        positionIdx=pos_idx,
        reduceOnly=True
    )
    if isinstance(resp, dict) and resp.get("retCode", 0) != 0:
        raise RuntimeError(f"retCode={resp.get('retCode')}: {resp.get('retMsg')}")
    try:
        client.cancel_all_orders(category="linear", symbol=sym)
    except Exception:
        pass


def _panic_close_sync() -> dict:
    bot = get_bot_instance()
    closed = []
    errors = []

    if bot is not None:
        try:
            bot.panic_close_all()
            return {"status": "ok", "message": "Все позиции закрыты, бот переведен на паузу."}
        except Exception as e:
            errors.append(str(e))

    try:
        client = get_standalone_client()
        pos_resp = client.get_positions(category="linear", settleCoin="USDT")
        for p in pos_resp.get("result", {}).get("list", []):
            sz = float(p.get("size", 0))
            sym = p.get("symbol")
            if sz > 0 and sym:
                try:
                    _close_exchange_position_safe(client, sym, p)
                    closed.append(sym)
                except Exception as e:
                    errors.append(f"{sym}: {e}")

        msg = f"Закрыто позиций: {len(closed)}" + (f" | Ошибки: {'; '.join(errors)}" if errors else "")
        return {"status": "ok" if not errors else "error", "message": msg}
    except Exception as e:
        return {"status": "error", "message": str(e)}


@app.post("/api/panic")
async def panic_close():
    """Panic Button: экстренно закрывает все открытые позиции и переводит бота на паузу."""
    set_bot_paused(True)
    _invalidate_status_cache()
    loop = asyncio.get_running_loop()
    res = await loop.run_in_executor(None, _panic_close_sync)
    _invalidate_status_cache()
    return res


def _close_position_sync(symbol: str) -> dict:
    sym = symbol.strip().upper()
    bot = get_bot_instance()

    if bot is not None:
        with getattr(bot, "lock", threading.RLock()):
            tracked_pos = dict(bot.positions.get(sym)) if getattr(bot, "positions", {}).get(sym) else None

        if tracked_pos and float(tracked_pos.get("size") or 0.0) > 0:
            try:
                price = bot.get_price(sym)
                res = bot.close_position(sym, price, reason="Ручное закрытие через UI")
                if res is not None:
                    return {"status": "ok", "message": f"Позиция {sym} закрыта."}
                return {"status": "error", "message": f"Не удалось закрыть позицию {sym} на бирже."}
            except Exception as e:
                return {"status": "error", "message": str(e)}

        # Fallback: проверка внесписочных / Hedge позиций напрямую через клиент бота
        client = getattr(bot, "client", None)
        if client is not None and hasattr(client, "get_positions"):
            try:
                pos_resp = client.get_positions(category="linear", symbol=sym)
                closed_any = False
                if isinstance(pos_resp, dict):
                    for p in pos_resp.get("result", {}).get("list", []):
                        if float(p.get("size") or 0.0) > 0:
                            _close_exchange_position_safe(client, sym, p)
                            closed_any = True
                if closed_any:
                    if hasattr(bot, "sync_positions"):
                        bot.sync_positions()
                    return {"status": "ok", "message": f"Позиция {sym} закрыта."}
            except Exception as e:
                return {"status": "error", "message": str(e)}
        return {"status": "error", "message": f"Открытая позиция по {sym} не найдена."}

    try:
        client = get_standalone_client()
        pos_resp = client.get_positions(category="linear", symbol=sym)
        closed_any = False
        for p in pos_resp.get("result", {}).get("list", []):
            sz = float(p.get("size", 0))
            if sz > 0:
                _close_exchange_position_safe(client, sym, p)
                closed_any = True
        if closed_any:
            return {"status": "ok", "message": f"Позиция {sym} закрыта (standalone)."}
        return {"status": "error", "message": f"Открытая позиция по {sym} не найдена."}
    except Exception as e:
        return {"status": "error", "message": str(e)}


@app.post("/api/close_position")
async def close_position(symbol: str = Query(...)):
    """Закрывает любую открытую позицию по символу."""
    loop = asyncio.get_running_loop()
    res = await loop.run_in_executor(None, lambda: _close_position_sync(symbol))
    _invalidate_status_cache()
    return res


def _set_mode_sync(new_mode: str) -> dict:
    global _STANDALONE_CLIENT, _LAST_STANDALONE_SYNC

    prev_mode = config.TRADING_MODE
    prev_key = config.API_KEY
    prev_sec = config.API_SECRET
    prev_demo = config.DEMO
    prev_test = config.TESTNET

    config.TRADING_MODE = new_mode
    if new_mode == "DEMO":
        config.API_KEY = config.DEMO_API_KEY
        config.API_SECRET = config.DEMO_API_SECRET
        config.DEMO = True
        config.TESTNET = False
    else:
        config.API_KEY = config.LIVE_API_KEY
        config.API_SECRET = config.LIVE_API_SECRET
        config.DEMO = False
        config.TESTNET = False

    with _STANDALONE_LOCK:
        _STANDALONE_CLIENT = None
        _LAST_STANDALONE_SYNC = 0.0

    bot = get_bot_instance()
    if bot is not None and hasattr(bot, "reload_credentials"):
        ok = bot.reload_credentials()
        if ok is False:
            config.TRADING_MODE = prev_mode
            config.API_KEY = prev_key
            config.API_SECRET = prev_sec
            config.DEMO = prev_demo
            config.TESTNET = prev_test
            return {"status": "error", "message": "Не удалось переключить режим: проверьте API ключи в .env"}
    else:
        write_control_state(trading_mode=new_mode)

    log.info(f"Дашборд: Торговый режим изменен на {new_mode}")
    return {"status": "ok", "message": f"Режим успешно изменен на {new_mode}"}


@app.post("/api/set_mode")
async def set_mode(mode: str = Query(...), password: Optional[str] = Query(None)):
    """
    Переключает торговый режим (DEMO или LIVE).
    Переключение в боевой режим LIVE требует ввода пароля DASHBOARD_PASSWORD.
    """
    new_mode = mode.strip().upper()
    if new_mode not in ("DEMO", "LIVE"):
        return {"status": "error", "message": "Допустимые режимы: DEMO или LIVE"}

    if new_mode == "LIVE":
        expected_pwd = config.DASHBOARD_PASSWORD
        if expected_pwd:
            if not password or not secrets.compare_digest(password, expected_pwd):
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail="Неверный пароль для активации боевого режима LIVE!"
                )

    loop = asyncio.get_running_loop()
    res = await loop.run_in_executor(None, lambda: _set_mode_sync(new_mode))
    _invalidate_status_cache()
    return res


@app.post("/api/backtest")
async def api_backtest(payload: dict = Body(...)):
    """Запускает честный бэктест (или портфельную симуляцию для ALL) из UI."""
    from backtest import run_backtest_symbol, run_portfolio_backtest
    symbol = str(payload.get("symbol", "BTCUSDT")).strip()
    try:
        days = max(1, min(365, int(payload.get("days", 90))))
    except (ValueError, TypeError):
        days = 90

    try:
        loop = asyncio.get_running_loop()
        if symbol.lower() == "all":
            port_res = await loop.run_in_executor(
                None, lambda: run_portfolio_backtest(days=days, verbose=False)
            )
            if not port_res or not port_res.get("symbols"):
                return JSONResponse(status_code=400, content={"message": "Нет данных"})

            recent = []
            tdf = port_res.get("tdf")
            if tdf is not None and not tdf.empty:
                for _, t in tdf.tail(20).iterrows():
                    recent.append({
                        "symbol": str(t["symbol"]),
                        "type": str(t["type"]),
                        "entry": float(t["entry"]),
                        "exit": float(t["exit"]),
                        "pnl_pct": float(t["pnl_pct"]),
                        "reason": str(t["reason"]),
                        "exit_time": t["exit_time"].strftime("%Y-%m-%d %H:%M") if hasattr(t["exit_time"], "strftime") else str(t["exit_time"])
                    })

            return {
                "symbol": "ALL",
                "trades": port_res["trades"],
                "win_rate": port_res["win_rate"],
                "pnl_pct": port_res["pnl_pct"],
                "profit_factor": port_res["profit_factor"],
                "max_drawdown": port_res["max_drawdown"],
                "sharpe_ratio": port_res["sharpe_ratio"],
                "recent_trades": recent
            }
        else:
            res = await loop.run_in_executor(None, lambda: run_backtest_symbol(symbol, days=days, verbose=False))
            if not res:
                return JSONResponse(status_code=400, content={"message": f"Нет данных для {symbol}"})

            recent = []
            tdf = res.get("tdf")
            if tdf is not None and not tdf.empty:
                for _, t in tdf.tail(20).iterrows():
                    recent.append({
                        "symbol": str(t["symbol"]),
                        "type": str(t["type"]),
                        "entry": float(t["entry"]),
                        "exit": float(t["exit"]),
                        "pnl_pct": float(t["pnl_pct"]),
                        "reason": str(t["reason"]),
                        "exit_time": t["exit_time"].strftime("%Y-%m-%d %H:%M") if hasattr(t["exit_time"], "strftime") else str(t["exit_time"])
                    })

            return {
                "symbol": res["symbol"],
                "trades": res["trades"],
                "win_rate": res["win_rate"],
                "pnl_pct": res["pnl_pct"],
                "profit_factor": res["profit_factor"],
                "max_drawdown": res["max_drawdown"],
                "sharpe_ratio": res["sharpe_ratio"],
                "recent_trades": recent
            }

    except Exception as e:
        return JSONResponse(status_code=500, content={"message": str(e)})


@app.get("/api/logs/download")
async def download_log(file: str = "bot"):
    """Позволяет скачать bot.log или trade.log."""
    file_path = Path(config.TRADE_LOG_FILE if file == "trade" else config.BOT_LOG_FILE)
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="Файл логов не найден")
    return FileResponse(path=file_path, filename=file_path.name, media_type="text/plain")


@app.get("/api/logs/view", response_class=PlainTextResponse)
async def view_log(file: str = "bot", lines: int = 200):
    """Возвращает последние N строк лога прямо текстом в браузер (1..2000)."""
    lines = max(1, min(2000, int(lines)))
    file_path = Path(config.TRADE_LOG_FILE if file == "trade" else config.BOT_LOG_FILE)
    if not file_path.exists():
        return "Логов пока нет"

    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
        tail = deque(f, maxlen=lines)
        return "".join(tail)
