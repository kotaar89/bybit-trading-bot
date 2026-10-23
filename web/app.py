"""
FastAPI веб-сервер легковесного локального дашборда Bybit Bot (v2.2).
Разработан специально под Termux (Android) с минимальным потреблением CPU и RAM.
Включает:
- Real-time WebSocket соединение (/ws) взамен частых AJAX-запросов
- Защиту переключения в боевой режим LIVE паролем
- Эндпоинты для интерактивных графиков свечей (/api/candles) и кривой капитала (/api/equity_history)
"""

from fastapi import FastAPI, Request, Query, Body, Depends, HTTPException, WebSocketException, status, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse, PlainTextResponse
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from fastapi.templating import Jinja2Templates
from starlette.requests import HTTPConnection
from pathlib import Path
import base64
import logging
import asyncio
import secrets
import time
import threading
from datetime import datetime, timezone
from typing import Optional
import pandas as pd

from config import (
    TRADING_MODE, SYMBOLS, ACTIVE_STRATEGY, LEVERAGE, MAX_OPEN_POSITIONS,
    API_KEY, API_SECRET, DEMO, TESTNET, FALLBACK_USDT
)
from web.state import (
    get_bot_instance, is_bot_paused, set_bot_paused, parse_trade_log
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
    title="Bybit Bot v2.2 Dashboard",
    dependencies=[Depends(verify_dashboard_auth)]
)

templates_dir = Path(__file__).resolve().parent / "templates"
templates = Jinja2Templates(directory=str(templates_dir))

# Кэш цен для предотвращения перегрузки Bybit API запросами из дашборда
_PRICE_CACHE = {}
_PRICE_CACHE_TTL = 3.0  # секунды
_CACHE_LOCK = threading.Lock()


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
    Исключает частый HTTP-поллинг, снижая нагрузку на процессор и экономя батарею (Termux).
    """
    await ws.accept()
    try:
        while True:
            status_data = await get_status()
            await ws.send_json(status_data)
            await asyncio.sleep(1.5)
    except (WebSocketDisconnect, asyncio.CancelledError):
        pass
    except Exception as e:
        log.debug(f"Dashboard WS disconnected: {e}")


@app.get("/api/status")
async def get_status():
    """
    Возвращает актуальный статус бота: баланс, открытые позиции,
    режим торговли, PnL и состояние защитного стопа.
    """
    bot = get_bot_instance()

    mode = config.TRADING_MODE
    paused = is_bot_paused()
    cb_active = False

    balance = FALLBACK_USDT
    equity = FALLBACK_USDT
    unrealized_pnl = 0.0
    daily_pnl = 0.0

    positions_data = []

    if bot is not None:
        # Данные из живого бота в памяти с потокобезопасной блокировкой
        try:
            balance = bot.get_available_balance()
            cb_active = getattr(bot, "circuit_breaker_active", False)

            with getattr(bot, "lock", threading.RLock()):
                positions_items = [(s, dict(p) if p else None) for s, p in bot.positions.items()]

            for sym, pos in positions_items:
                if pos:
                    try:
                        curr_price = get_cached_price(bot, sym)
                    except Exception:
                        curr_price = pos.get("entry_price", 0.0)

                    entry = pos.get("entry_price", 0.0)
                    size = pos.get("size", 0.0)
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
            daily_pnl = (equity if isinstance(raw_eq, (int, float)) and raw_eq > 0 else balance) - start_bal

        except Exception as e:
            log.warning(f"Ошибка получения статуса из bot instance: {e}")

    else:
        # Standalone режим (бот запущен в другом процессе или выключен):
        # Опрашиваем Bybit напрямую через HTTP со синхронизацией
        try:
            client = get_standalone_client()
            resp = client.get_wallet_balance(accountType="UNIFIED")
            acc_list = resp.get("result", {}).get("list", [])
            if acc_list:
                coins = acc_list[0].get("coin", [])
                for c in coins:
                    if c.get("coin") == "USDT":
                        raw_avail = c.get("availableToWithdraw")
                        avail_val = float(raw_avail) if raw_avail not in (None, "") else 0.0
                        wb = float(c.get("walletBalance") or 0.0)
                        pos_im = float(c.get("totalPositionIM") or 0.0)
                        ord_im = float(c.get("totalOrderIM") or 0.0)
                        free_wb = max(0.0, wb - pos_im - ord_im)
                        balance = avail_val if avail_val > 0 else (free_wb if free_wb > 0 else wb)
                        equity = float(c.get("equity") or wb or balance)
                        break

            # Получаем открытые позиции
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
                        "breakeven": False
                    })

        except Exception as e:
            log.warning(f"Standalone статус Bybit: {e}")

    unrealized_pct = (unrealized_pnl / balance * 100) if balance > 0 else 0.0

    return {
        "mode": mode,
        "is_paused": paused,
        "circuit_breaker": cb_active,
        "balance": round(balance, 2),
        "equity": round(equity, 2),
        "unrealized_pnl": round(unrealized_pnl, 2),
        "unrealized_pnl_pct": round(unrealized_pct, 2),
        "daily_pnl": round(daily_pnl, 2),
        "strategy": ACTIVE_STRATEGY,
        "leverage": LEVERAGE,
        "open_positions_count": len(positions_data),
        "max_positions": MAX_OPEN_POSITIONS,
        "symbols": config.SYMBOLS,
        "positions": positions_data
    }


@app.get("/api/candles")
async def get_candles(
    symbol: str = Query("BTCUSDT"),
    interval: str = Query("60"),
    limit: int = Query(120)
):
    """
    Возвращает свечные данные (OHLCV) для графиков TradingView Lightweight Charts.
    """
    sym = symbol.strip().upper()
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
            df["timestamp"] = pd.to_datetime(df["timestamp"].astype(int), unit="ms")
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


@app.get("/api/equity_history")
async def get_equity_history():
    """
    Возвращает историческую кривую капитала (Equity Curve)
    на основе баланса и истории сделок из trade.log.
    """
    trades = parse_trade_log(limit=100)
    initial_balance = config.FALLBACK_USDT
    bot = get_bot_instance()
    if bot is not None:
        initial_balance = getattr(bot, "start_day_balance", initial_balance)

    points = []
    curr_balance = initial_balance

    if not trades:
        now_ts = int(time.time())
        return [{"time": now_ts, "value": round(curr_balance, 2)}]

    sorted_trades = sorted(trades, key=lambda x: x.get("time") or x.get("timestamp") or "")
    for t in sorted_trades:
        pnl = float(t.get("pnl_pct") or 0.0)
        curr_balance += (curr_balance * (pnl / 100.0))
        t_str = t.get("time") or t.get("timestamp") or ""
        try:
            ts = int(datetime.strptime(t_str, "%Y-%m-%d %H:%M:%S").replace(tzinfo=timezone.utc).timestamp())
        except Exception:
            ts = int(time.time())
        if points and ts <= points[-1]["time"]:
            ts = points[-1]["time"] + 1
        points.append({"time": ts, "value": round(curr_balance, 2)})

    return points


@app.get("/api/trades")
async def get_trades():
    """Возвращает историю сделок из trade.log."""
    return parse_trade_log(limit=60)


@app.post("/api/pause")
async def toggle_pause():
    """Переключает состояние паузы (Emergency Pause)."""
    curr = is_bot_paused()
    set_bot_paused(not curr)
    new_state = not curr
    log.info(f"Дашборд: Бот переведен в состояние {'ПАУЗА' if new_state else 'АКТИВЕН'}")
    return {"status": "ok", "is_paused": new_state}


@app.post("/api/panic")
async def panic_close():
    """
    Panic Button: Экстренно закрывает все открытые позиции по маркету
    и отменяет все активные ордера.
    """
    bot = get_bot_instance()
    closed = []
    errors = []

    # При экстренном закрытии обязательно переводим бота на паузу, чтобы не открыть позиции заново
    set_bot_paused(True)

    if bot is not None:
        try:
            bot.panic_close_all()
            return {"status": "ok", "message": "Все позиции закрыты, бот переведен на паузу."}
        except Exception as e:
            errors.append(str(e))

    # Если бот не зарегистрирован или выбросил ошибку, закрываем напрямую через HTTP
    try:
        client = get_standalone_client()
        pos_resp = client.get_positions(category="linear", settleCoin="USDT")
        for p in pos_resp.get("result", {}).get("list", []):
            sz = float(p.get("size", 0))
            sym = p.get("symbol")
            if sz > 0 and sym:
                try:
                    client.cancel_all_orders(category="linear", symbol=sym)
                    side = "Sell" if p.get("side") == "Buy" else "Buy"
                    pos_idx = int(p.get("positionIdx", 0))
                    client.place_order(
                        category="linear",
                        symbol=sym,
                        side=side,
                        orderType="Market",
                        qty=str(sz),
                        positionIdx=pos_idx,
                        reduceOnly=True
                    )
                    closed.append(sym)
                except Exception as e:
                    errors.append(f"{sym}: {e}")

        msg = f"Закрыто позиций: {len(closed)}" + (f" | Ошибки: {'; '.join(errors)}" if errors else "")
        return {"status": "ok", "message": msg}
    except Exception as e:
        return {"status": "error", "message": str(e)}


@app.post("/api/close_position")
async def close_position(symbol: str = Query(...)):
    """Закрывает конкретную позицию по символу."""
    sym = symbol.strip().upper()
    if sym not in config.SYMBOLS and sym != config.BENCHMARK_SYMBOL:
        return {"status": "error", "message": f"Неизвестный символ: {sym}. Разрешены: {', '.join(config.SYMBOLS)}"}

    bot = get_bot_instance()
    if bot is not None:
        try:
            price = bot.get_price(sym)
            bot.close_position(sym, price, reason="Ручное закрытие через UI")
            return {"status": "ok", "message": f"Позиция {sym} закрыта."}
        except Exception as e:
            return {"status": "error", "message": str(e)}

    # Standalone фоллбэк: закрытие позиции напрямую через HTTP клиент Bybit
    try:
        client = get_standalone_client()
        pos_resp = client.get_positions(category="linear", symbol=sym)
        closed_any = False
        for p in pos_resp.get("result", {}).get("list", []):
            sz = float(p.get("size", 0))
            if sz > 0:
                client.cancel_all_orders(category="linear", symbol=sym)
                side = "Sell" if p.get("side") == "Buy" else "Buy"
                pos_idx = int(p.get("positionIdx", 0))
                client.place_order(
                    category="linear",
                    symbol=sym,
                    side=side,
                    orderType="Market",
                    qty=str(sz),
                    positionIdx=pos_idx,
                    reduceOnly=True
                )
                closed_any = True
        if closed_any:
            return {"status": "ok", "message": f"Позиция {sym} закрыта (standalone)."}
        return {"status": "error", "message": f"Открытая позиция по {sym} не найдена."}
    except Exception as e:
        return {"status": "error", "message": str(e)}


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

    global _STANDALONE_CLIENT, _LAST_STANDALONE_SYNC
    with _STANDALONE_LOCK:
        _STANDALONE_CLIENT = None
        _LAST_STANDALONE_SYNC = 0.0

    bot = get_bot_instance()
    if bot is not None and hasattr(bot, "reload_credentials"):
        bot.reload_credentials()

    log.info(f"Дашборд: Торговый режим изменен на {new_mode}")
    return {"status": "ok", "message": f"Режим успешно изменен на {new_mode}"}


@app.post("/api/backtest")
async def api_backtest(payload: dict = Body(...)):
    """Запускает бэктест из UI и возвращает результаты."""
    from backtest import run_backtest_symbol, run_backtest
    symbol = str(payload.get("symbol", "BTCUSDT")).strip()
    try:
        days = max(1, min(365, int(payload.get("days", 90))))
    except (ValueError, TypeError):
        days = 90

    try:
        loop = asyncio.get_running_loop()
        if symbol.lower() == "all":
            results = await loop.run_in_executor(None, lambda: run_backtest(days=days))
            if not results:
                return JSONResponse(status_code=400, content={"message": "Нет данных"})
            tot_trades = sum(r["trades"] for r in results)
            tot_pnl = sum(r["pnl_pct"] for r in results) / len(results)
            avg_wr = sum(r["win_rate"] for r in results) / len(results)
            max_dd = min(r["max_drawdown"] for r in results)
            avg_pf = sum(r["profit_factor"] for r in results) / len(results)
            avg_sh = sum(r["sharpe_ratio"] for r in results) / len(results)

            return {
                "symbol": "ALL",
                "trades": tot_trades,
                "win_rate": avg_wr,
                "pnl_pct": tot_pnl,
                "profit_factor": avg_pf,
                "max_drawdown": max_dd,
                "sharpe_ratio": avg_sh,
                "recent_trades": []
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
                        "symbol": t["symbol"],
                        "type": t["type"],
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
    """Позволяет скачать bot.log или trade.log на компьютер."""
    filename = "trade.log" if file == "trade" else "bot.log"
    file_path = Path(filename)
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="Файл логов не найден")
    return FileResponse(path=file_path, filename=filename, media_type="text/plain")


@app.get("/api/logs/view", response_class=PlainTextResponse)
async def view_log(file: str = "bot", lines: int = 200):
    """Возвращает последние N строк лога прямо текстом в браузер."""
    filename = "trade.log" if file == "trade" else "bot.log"
    file_path = Path(filename)
    if not file_path.exists():
        return "Логов пока нет"

    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
        content = f.readlines()
        return "".join(content[-lines:])
