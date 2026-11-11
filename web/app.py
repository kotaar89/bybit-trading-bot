"""
FastAPI веб-сервер легковесного локального дашборда Bybit Bot (v2.3).
Разработан специально под Termux (Android) с минимальным потреблением CPU и RAM.
Включает:
- Real-time WebSocket соединение (/ws) с неблокирующим executor и TTL-кэшем статуса
- Эндпоинты для интерактивных графиков свечей (/api/candles) и честной кривой капитала (/api/equity_history)
- Полноценный потоковый парсер истории сделок и портфельный бэктестер
"""

from fastapi import FastAPI, Request, Query, Body, Depends, HTTPException, WebSocketException, status, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse, PlainTextResponse, Response
from fastapi.security import HTTPBasic
from fastapi.templating import Jinja2Templates
from starlette.requests import HTTPConnection
from pathlib import Path
from collections import deque
from datetime import datetime
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
from risk_manager import PortfolioRiskManager, PositionRiskState
from storage import BotDatabase
from web.state import (
    get_bot_instance, is_bot_paused, set_bot_paused, parse_trade_log,
    parse_equity_history, write_control_state, read_control_state, read_live_status,
    enqueue_shadow_close,
    get_trade_history, get_equity_curve, get_analytics_summary,
    export_trades_csv, export_trades_json,
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
    title="Bybit Bot v0.3.0 Dashboard",
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
    shadow_mode_cfg = bool(getattr(config, "SHADOW_MODE", False))

    if bot is None:
        ctrl = read_control_state()
        if ctrl.get("trading_mode") in config.VALID_TRADING_MODES:
            mode = ctrl["trading_mode"]

    cache_key = (id(bot) if bot is not None else 0, mode, paused, shadow_mode_cfg)
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
    live_snap = {}
    snap_is_shadow = False
    sh_state = None
    runtime_bundle = {}
    sh_analytics = None

    is_shadow_only = (
        bot is not None
        and not is_mock_bot
        and shadow_mode_cfg
        and getattr(bot, "shadow_engine", None) is not None
        and "unittest.mock" not in type(bot.shadow_engine).__module__
    )

    if bot is not None:
        try:
            if not is_shadow_only:
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

                    pos_entry = {
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
                    }
                    if is_shadow_only:
                        pos_entry["shadow_mode"] = True
                    positions_data.append(pos_entry)

            if is_shadow_only:
                sh_eng = bot.shadow_engine
                sh_realized = float(getattr(sh_eng, "equity", 0.0) or 0.0)
                used_margin = sum(
                    float(getattr(r, "margin_used", 0.0) or 0.0)
                    for r in getattr(sh_eng, "risk_states", {}).values()
                    if r
                )
                balance = max(0.0, sh_realized - used_margin)
                equity = sh_realized + unrealized_pnl
            else:
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
        if shadow_mode_cfg:
            # В SHADOW_MODE никогда не запрашиваем реальный аккаунт Bybit через get_standalone_client()
            snap_is_shadow = (
                isinstance(live_snap, dict)
                and bool(live_snap)
                and (
                    live_snap.get("shadow_mode") is True
                    or "shadow_positions" in live_snap
                    or (
                        live_snap.get("shadow_mode") is not False
                        and isinstance(live_snap.get("shadow_status"), dict)
                    )
                )
            )
            db_file = Path(getattr(config, "DB_FILE", ""))
            if str(db_file) and db_file.exists():
                try:
                    db = BotDatabase(db_file)
                    raw_sh = db.get_runtime_value("shadow_engine_state")
                    if isinstance(raw_sh, dict):
                        sh_state = raw_sh
                    raw_rt = db.get_all_runtime_state()
                    if isinstance(raw_rt, dict):
                        runtime_bundle = raw_rt
                    sh_analytics = db.get_analytics_summary(is_shadow=True)
                except Exception as e:
                    log.debug(f"Standalone shadow DB read error: {e}")

            today_str = str(datetime.now().date())
            if snap_is_shadow and "circuit_breaker" in live_snap:
                cb_active = bool(live_snap.get("circuit_breaker", False))
            elif str(runtime_bundle.get("start_day") or "") == today_str:
                cb_active = bool(runtime_bundle.get("circuit_breaker_active", False))

            be_map = (
                dict(live_snap.get("breakeven_map"))
                if (snap_is_shadow and isinstance(live_snap.get("breakeven_map"), dict))
                else {}
            )
            if isinstance(sh_state, dict) and isinstance(sh_state.get("risk_states"), dict):
                for r_sym, r_dict in sh_state["risk_states"].items():
                    if isinstance(r_dict, dict) and r_sym not in be_map:
                        be_map[r_sym] = bool(r_dict.get("breakeven_applied", False))

            if snap_is_shadow and float(live_snap.get("start_day_balance") or 0.0) > 0:
                start_bal = float(live_snap["start_day_balance"])
            elif str(runtime_bundle.get("start_day") or "") == today_str and float(runtime_bundle.get("start_day_balance") or 0.0) > 0:
                start_bal = float(runtime_bundle["start_day_balance"])
            else:
                start_bal = 0.0

            init_sh_bal = float(getattr(config, "SHADOW_INITIAL_BALANCE", FALLBACK_USDT) or FALLBACK_USDT)
            if snap_is_shadow and isinstance(live_snap.get("shadow_positions"), list):
                used_margin_snap = 0.0
                for p in live_snap["shadow_positions"]:
                    if not isinstance(p, dict):
                        continue
                    sz = float(p.get("size") or 0.0)
                    if sz <= 0:
                        continue
                    sym = str(p.get("symbol") or "").upper()
                    side = str(p.get("side") or "Buy")
                    entry = float(p.get("entry_price") or 0.0)
                    mark = float(p.get("mark_price") or entry)
                    is_long = side.lower() in ("buy", "long") or str(p.get("position", "")).lower() == "long"
                    if p.get("pnl") is not None:
                        pnl_usd = float(p["pnl"])
                    elif entry > 0 and sz > 0:
                        pnl_usd = ((mark - entry) if is_long else (entry - mark)) * sz
                    else:
                        pnl_usd = 0.0
                    if p.get("pnl_pct") is not None:
                        pnl_pct = float(p["pnl_pct"])
                    elif entry > 0:
                        diff = (mark - entry) if is_long else (entry - mark)
                        pnl_pct = (diff / entry) * 100.0 * LEVERAGE
                    else:
                        pnl_pct = 0.0
                    unrealized_pnl += pnl_usd
                    used_margin_snap += float(p.get("margin") or 0.0)
                    positions_data.append({
                        "symbol": sym,
                        "side": side,
                        "size": sz,
                        "entry_price": entry,
                        "mark_price": mark,
                        "pnl": round(pnl_usd, 2),
                        "pnl_pct": round(pnl_pct, 2),
                        "stop_loss": float(p.get("stop_loss") or 0.0),
                        "take_profit": float(p.get("take_profit") or 0.0),
                        "breakeven": bool(p.get("breakeven", be_map.get(sym, False))),
                        "shadow_mode": True,
                    })
                if live_snap.get("unrealized_pnl") is not None:
                    unrealized_pnl = float(live_snap["unrealized_pnl"])
                equity = float(
                    live_snap.get("equity")
                    or (sh_state.get("equity") if isinstance(sh_state, dict) else 0.0)
                    or init_sh_bal
                )
                if live_snap.get("balance") is not None:
                    balance = max(0.0, float(live_snap["balance"]))
                else:
                    balance = max(0.0, equity - unrealized_pnl - used_margin_snap)
            elif isinstance(sh_state, dict):
                sh_realized = float(sh_state.get("equity") or init_sh_bal)
                sh_pos_map = sh_state.get("positions") if isinstance(sh_state.get("positions"), dict) else {}
                sh_risk_map = sh_state.get("risk_states") if isinstance(sh_state.get("risk_states"), dict) else {}
                used_margin = 0.0
                for sym, pos in sh_pos_map.items():
                    if not isinstance(pos, dict):
                        continue
                    sz = float(pos.get("size") or 0.0)
                    if sz <= 0:
                        continue
                    sym_u = str(sym or "").upper()
                    r_dict = sh_risk_map.get(sym) if isinstance(sh_risk_map.get(sym), dict) else {}
                    used_margin += float(r_dict.get("margin_used") or pos.get("margin") or 0.0)
                    entry = float(pos.get("entry_price") or r_dict.get("entry_price") or 0.0)
                    mark = float(pos.get("mark_price") or entry)
                    side = str(pos.get("side") or "Buy")
                    is_long = side.lower() in ("buy", "long") or str(pos.get("position", "")).lower() == "long"
                    diff = (mark - entry) if is_long else (entry - mark)
                    pnl_usd = diff * sz if (entry > 0 and sz > 0) else 0.0
                    pnl_pct = (diff / entry * 100.0 * LEVERAGE) if entry > 0 else 0.0
                    unrealized_pnl += pnl_usd
                    positions_data.append({
                        "symbol": sym_u,
                        "side": side,
                        "size": sz,
                        "entry_price": entry,
                        "mark_price": mark,
                        "pnl": round(pnl_usd, 2),
                        "pnl_pct": round(pnl_pct, 2),
                        "stop_loss": float(pos.get("stop_loss") or 0.0),
                        "take_profit": float(pos.get("take_profit") or 0.0),
                        "breakeven": bool(be_map.get(sym_u, r_dict.get("breakeven_applied", False))),
                        "shadow_mode": True,
                    })
                balance = max(0.0, sh_realized - used_margin)
                equity = sh_realized + unrealized_pnl
                if snap_is_shadow:
                    if float(live_snap.get("equity") or 0.0) > 0:
                        equity = float(live_snap["equity"])
                    if live_snap.get("balance") is not None:
                        balance = max(0.0, float(live_snap["balance"]))
            elif snap_is_shadow:
                sh_st = live_snap.get("shadow_status") if isinstance(live_snap.get("shadow_status"), dict) else {}
                raw_sh_pos = sh_st.get("positions") if isinstance(sh_st.get("positions"), list) else []
                used_margin = 0.0
                for pos in raw_sh_pos:
                    if not isinstance(pos, dict):
                        continue
                    sz = float(pos.get("size") or 0.0)
                    if sz <= 0:
                        continue
                    sym_u = str(pos.get("symbol") or "").upper()
                    entry = float(pos.get("entry_price") or 0.0)
                    mark = float(pos.get("mark_price") or entry)
                    side = str(pos.get("side") or "Buy")
                    used_margin += float(pos.get("margin") or 0.0)
                    positions_data.append({
                        "symbol": sym_u,
                        "side": side,
                        "size": sz,
                        "entry_price": entry,
                        "mark_price": mark,
                        "pnl": 0.0,
                        "pnl_pct": 0.0,
                        "stop_loss": float(pos.get("stop_loss") or 0.0),
                        "take_profit": float(pos.get("take_profit") or 0.0),
                        "breakeven": bool(be_map.get(sym_u, False)),
                        "shadow_mode": True,
                    })
                equity = float(live_snap.get("equity") or sh_st.get("equity") or init_sh_bal)
                balance = (
                    max(0.0, float(live_snap["balance"]))
                    if live_snap.get("balance") is not None
                    else max(0.0, equity - used_margin)
                )
            else:
                balance = init_sh_bal
                equity = init_sh_bal

            if snap_is_shadow and live_snap.get("daily_pnl") is not None:
                daily_pnl = float(live_snap["daily_pnl"])
            elif start_bal > 0 and equity > 0:
                daily_pnl = equity - start_bal
        else:
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

    # v0.3.0 телеметрия портфельного риска, режима рынка, shadow и динамической корзины
    portfolio_heat_pct = 0.0
    max_portfolio_heat_pct = round(float(getattr(config, "MAX_PORTFOLIO_HEAT_PCT", 0.035)) * 100.0, 2)
    peak_equity = round(base_cap, 2) if base_cap > 0 else 0.0
    peak_drawdown_pct = 0.0
    risk_scale = 1.0
    drawdown_pause = False
    active_symbols = list(config.SYMBOLS)
    regime_summary = {}
    cooldowns = {}
    shadow_status = None

    if bot is not None and not is_mock_bot:
        try:
            sh_eng = getattr(bot, "shadow_engine", None)
            pr = (
                getattr(sh_eng, "portfolio_risk", None)
                if is_shadow_only
                else getattr(bot, "portfolio_risk", None)
            )
            r_states = (
                getattr(sh_eng, "risk_states", {})
                if is_shadow_only
                else getattr(bot, "risk_states", {})
            )
            if pr is not None and "unittest.mock" not in type(pr).__module__:
                eq_ref = equity if equity > 0 else balance
                portfolio_heat_pct = round(float(pr.compute_open_heat_pct(r_states, eq_ref)) * 100.0, 2)
                dd_info = pr.update_equity(eq_ref, valid=False)
                peak_equity = round(float(dd_info.get("peak_equity", peak_equity)), 2)
                peak_drawdown_pct = round(float(dd_info.get("peak_dd_pct", 0.0)), 2)
                risk_scale = round(float(dd_info.get("risk_scale", 1.0)), 2)
                drawdown_pause = bool(dd_info.get("drawdown_pause", False))
                cooldowns = pr.active_cooldowns()
            act_syms = getattr(bot, "active_symbols", None)
            if isinstance(act_syms, (list, tuple)) and act_syms:
                active_symbols = list(act_syms)
            reg_snap = getattr(bot, "_last_regime_snapshot", None)
            if isinstance(reg_snap, dict):
                regime_summary = dict(reg_snap)
            if sh_eng is not None and "unittest.mock" not in type(sh_eng).__module__ and hasattr(sh_eng, "get_status"):
                shadow_status = sh_eng.get_status()
        except Exception:
            pass
    elif isinstance(live_snap, dict) and live_snap and (not shadow_mode_cfg or snap_is_shadow):
        try:
            portfolio_heat_pct = round(float(live_snap.get("portfolio_heat_pct") or 0.0), 2)
            peak_equity = round(float(live_snap.get("peak_equity") or peak_equity), 2)
            peak_drawdown_pct = round(float(live_snap.get("peak_drawdown_pct") or 0.0), 2)
            risk_scale = round(float(live_snap.get("risk_scale") if live_snap.get("risk_scale") is not None else 1.0), 2)
            drawdown_pause = bool(live_snap.get("drawdown_pause", False))
            act_syms = live_snap.get("active_symbols")
            if isinstance(act_syms, (list, tuple)) and act_syms:
                active_symbols = list(act_syms)
            if isinstance(live_snap.get("regime_summary"), dict):
                regime_summary = dict(live_snap["regime_summary"])
            if isinstance(live_snap.get("cooldowns"), dict):
                cooldowns = dict(live_snap["cooldowns"])
            if isinstance(live_snap.get("shadow_status"), dict):
                shadow_status = dict(live_snap["shadow_status"])
        except Exception:
            pass
    elif shadow_mode_cfg:
        try:
            if isinstance(sh_state, dict):
                pr_raw = sh_state.get("portfolio_risk")
                r_raw = sh_state.get("risk_states")
                r_states_obj = {}
                if isinstance(r_raw, dict):
                    for r_sym, r_dict in r_raw.items():
                        if isinstance(r_dict, dict):
                            try:
                                r_states_obj[r_sym] = PositionRiskState.from_dict(r_dict)
                            except Exception:
                                pass
                pr_obj = PortfolioRiskManager.from_dict(
                    pr_raw if isinstance(pr_raw, dict) else {},
                    max_heat_pct=float(getattr(config, "MAX_PORTFOLIO_HEAT_PCT", 0.035)),
                    use_correlation_scaling=bool(getattr(config, "USE_CORRELATION_SCALING", True)),
                    corr_lookback_bars=int(getattr(config, "CORR_LOOKBACK_BARS", 336)),
                    corr_threshold=float(getattr(config, "CORR_THRESHOLD", 0.75)),
                    corr_risk_scale=float(getattr(config, "CORR_RISK_SCALE", 0.6)),
                    use_peak_dd_throttle=bool(getattr(config, "USE_PEAK_DD_THROTTLE", True)),
                    dd_tier1_pct=float(getattr(config, "DD_THROTTLE_TIER1_PCT", 0.06)),
                    dd_tier1_mult=float(getattr(config, "DD_THROTTLE_TIER1_MULT", 0.5)),
                    dd_tier2_pct=float(getattr(config, "DD_THROTTLE_TIER2_PCT", 0.12)),
                    dd_recover_pct=float(getattr(config, "DD_THROTTLE_RECOVER_PCT", 0.09)),
                    use_loss_cooldown=bool(getattr(config, "USE_LOSS_STREAK_COOLDOWN", True)),
                    symbol_loss_limit=int(getattr(config, "LOSS_STREAK_SYMBOL_LIMIT", 2)),
                    portfolio_loss_limit=int(getattr(config, "LOSS_STREAK_PORTFOLIO_LIMIT", 3)),
                    cooldown_seconds=float(getattr(config, "LOSS_STREAK_COOLDOWN_BARS", 6)) * 3600.0,
                )
                eq_ref = equity if equity > 0 else balance
                portfolio_heat_pct = round(float(pr_obj.compute_open_heat_pct(r_states_obj, eq_ref)) * 100.0, 2)
                dd_info = pr_obj.update_equity(eq_ref, valid=False)
                peak_equity = round(float(dd_info.get("peak_equity") or peak_equity), 2)
                peak_drawdown_pct = round(float(dd_info.get("peak_dd_pct", 0.0)), 2)
                risk_scale = round(float(dd_info.get("risk_scale", 1.0)), 2)
                drawdown_pause = bool(dd_info.get("drawdown_pause", False))
                cooldowns = pr_obj.active_cooldowns()
            act_syms = runtime_bundle.get("active_symbols")
            if isinstance(act_syms, (list, tuple)) and act_syms:
                active_symbols = list(act_syms)
        except Exception:
            pass

    raw_active = str(getattr(config, "ACTIVE_STRATEGY", "SUPERTREND_PRO")).strip().upper()
    eff_strategy = (
        "REGIME_SWITCHER"
        if getattr(config, "USE_REGIME_SWITCHER", False)
        and raw_active not in ("REGIME_SWITCHER", "ADAPTIVE")
        else raw_active
    )
    if is_shadow_only and getattr(bot.shadow_engine, "strategy_name", ""):
        eff_strategy = str(bot.shadow_engine.strategy_name)
    elif bot is None and shadow_mode_cfg:
        sh_cfg_strat = str(getattr(config, "SHADOW_STRATEGY", "") or "").strip().upper()
        if isinstance(shadow_status, dict) and shadow_status.get("strategy"):
            eff_strategy = str(shadow_status["strategy"]).strip().upper()
        elif sh_cfg_strat:
            eff_strategy = sh_cfg_strat
        if shadow_status is None:
            shadow_status = {
                "strategy": eff_strategy,
                "equity": round(equity, 2),
                "initial_equity": round(float(getattr(config, "SHADOW_INITIAL_BALANCE", FALLBACK_USDT) or FALLBACK_USDT), 2),
                "open_positions_count": len(positions_data),
                "positions": positions_data,
                "peak_equity": peak_equity,
            }
        if isinstance(sh_analytics, dict) and isinstance(shadow_status, dict):
            shadow_status.setdefault("closed_trades", int(sh_analytics.get("total_trades") or 0))
            shadow_status.setdefault("win_rate", float(sh_analytics.get("win_rate") or 0.0))
            shadow_status.setdefault("total_net_pnl_usdt", float(sh_analytics.get("total_net_pnl_usdt") or 0.0))

    result = {
        "mode": mode,
        "is_paused": paused,
        "circuit_breaker": cb_active,
        "balance": round(balance, 2),
        "equity": round(equity, 2),
        "unrealized_pnl": round(unrealized_pnl, 2),
        "unrealized_pnl_pct": round(unrealized_pct, 2),
        "daily_pnl": round(daily_pnl, 2),
        "strategy": eff_strategy,
        "leverage": LEVERAGE,
        "open_positions_count": len(positions_data),
        "max_positions": MAX_OPEN_POSITIONS,
        "symbols": active_symbols,
        "active_symbols": active_symbols,
        "positions": positions_data,
        "portfolio_heat_pct": portfolio_heat_pct,
        "max_portfolio_heat_pct": max_portfolio_heat_pct,
        "peak_equity": peak_equity,
        "peak_drawdown_pct": peak_drawdown_pct,
        "risk_scale": risk_scale,
        "drawdown_pause": drawdown_pause,
        "regime_summary": regime_summary,
        "cooldowns": cooldowns,
        "shadow_mode": shadow_mode_cfg,
        "shadow_strategy": str(getattr(config, "SHADOW_STRATEGY", "") or ""),
        "shadow_status": shadow_status,
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
async def get_equity_history(shadow: bool = Query(False)):
    """
    Возвращает историческую кривую капитала (Equity Curve) из SQLite
    с безопасным fallback на trade.log.
    """
    initial_balance = (
        getattr(config, "SHADOW_INITIAL_BALANCE", config.FALLBACK_USDT)
        if shadow
        else config.FALLBACK_USDT
    )
    bot = get_bot_instance()
    if bot is not None and not shadow:
        sb = getattr(bot, "start_day_balance", initial_balance)
        if isinstance(sb, (int, float)) and sb > 0:
            initial_balance = float(sb)

    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(
        None,
        lambda: get_equity_curve(
            initial_balance=initial_balance,
            margin_weight=config.POSITION_SIZE_PCT,
            is_shadow=shadow,
        ),
    )


@app.get("/api/trades")
async def get_trades(
    limit: int = Query(60, ge=1, le=500),
    include_shadow: bool = Query(False),
    only_shadow: bool = Query(False),
    symbol: Optional[str] = Query(None),
):
    """Возвращает историю сделок из SQLite (или fallback на trade.log)."""
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(
        None,
        lambda: get_trade_history(
            limit=limit,
            include_shadow=include_shadow,
            only_shadow=only_shadow,
            symbol=symbol,
        ),
    )


@app.get("/api/trades/export")
async def export_trades(
    format: str = Query("csv"),
    limit: int = Query(1000, ge=1, le=5000),
    include_shadow: bool = Query(False),
):
    """Экспортирует историю сделок из SQLite в формате CSV или JSON."""
    fmt = format.strip().lower()
    loop = asyncio.get_running_loop()
    if fmt == "json":
        data = await loop.run_in_executor(
            None, lambda: export_trades_json(limit=limit, include_shadow=include_shadow)
        )
        return JSONResponse(
            content=data,
            headers={"Content-Disposition": 'attachment; filename="trades_export.json"'},
        )
    csv_text = await loop.run_in_executor(
        None, lambda: export_trades_csv(limit=limit, include_shadow=include_shadow)
    )
    return Response(
        content=csv_text,
        media_type="text/csv; charset=utf-8",
        headers={"Content-Disposition": 'attachment; filename="trades_export.csv"'},
    )


@app.get("/api/analytics")
async def get_analytics(shadow: bool = Query(False), limit: int = Query(2000, ge=50, le=5000)):
    """Возвращает агрегированную аналитику сделок (WinRate, PF, Net PnL USDT, Fees, Funding, MaxDD) и live-деградацию."""
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(
        None, lambda: get_analytics_summary(is_shadow=shadow, limit=limit)
    )


@app.get("/api/universe")
async def get_universe():
    """Возвращает текущую активную корзину и рейтинг ликвидных инструментов."""
    bot = get_bot_instance()
    is_mock = bot is not None and "unittest.mock" in type(bot).__module__
    enabled = bool(getattr(config, "USE_DYNAMIC_UNIVERSE", False))
    active = list(config.SYMBOLS)
    rankings: list[dict] = []
    last_refresh = 0.0

    if bot is not None and not is_mock:
        act = getattr(bot, "active_symbols", None)
        if isinstance(act, (list, tuple)) and act:
            active = list(act)
        sel = getattr(bot, "universe_selector", None)
        if sel is not None and isinstance(getattr(sel, "last_ranked", None), list):
            rankings = list(sel.last_ranked)
        lr = getattr(bot, "_last_universe_refresh", 0.0) or (getattr(sel, "last_refresh_ts", 0.0) if sel is not None else 0.0)
        if isinstance(lr, (int, float)):
            last_refresh = float(lr)
    else:
        snap = read_live_status()
        if isinstance(snap.get("active_symbols"), list) and snap["active_symbols"]:
            active = list(snap["active_symbols"])

    return {
        "enabled": enabled,
        "active_symbols": active,
        "rankings": rankings,
        "last_refresh": last_refresh,
    }


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
    shadow_mode_cfg = bool(getattr(config, "SHADOW_MODE", False))
    closed = []
    errors = []

    if bot is not None:
        try:
            bot.panic_close_all()
            res = {"status": "ok", "message": "Все позиции закрыты, бот переведен на паузу."}
            if shadow_mode_cfg:
                res["shadow_mode"] = True
            return res
        except Exception as e:
            if shadow_mode_cfg:
                return {"status": "error", "shadow_mode": True, "message": str(e)}
            errors.append(str(e))

    if shadow_mode_cfg:
        live_snap = read_live_status()
        snap_is_shadow = (
            isinstance(live_snap, dict)
            and bool(live_snap)
            and (
                live_snap.get("shadow_mode") is True
                or "shadow_positions" in live_snap
                or (
                    live_snap.get("shadow_mode") is not False
                    and isinstance(live_snap.get("shadow_status"), dict)
                )
            )
        )
        if snap_is_shadow:
            write_control_state(is_paused=True, pending_shadow_panic=True)
            return {
                "status": "ok",
                "queued": True,
                "shadow_mode": True,
                "message": "SHADOW_MODE: бот переведен на паузу, команда закрытия всех теневых позиций отправлена процессу bot.py.",
            }
        write_control_state(is_paused=True)
        return {
            "status": "error",
            "shadow_mode": True,
            "message": "SHADOW_MODE: бот переведен на паузу; теневые позиции управляются запущенным процессом bot.py (реальные биржевые ордера отключены).",
        }

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
    shadow_mode_cfg = bool(getattr(config, "SHADOW_MODE", False))

    if bot is not None:
        with getattr(bot, "lock", threading.RLock()):
            tracked_pos = dict(bot.positions.get(sym)) if getattr(bot, "positions", {}).get(sym) else None

        if tracked_pos and float(tracked_pos.get("size") or 0.0) > 0:
            try:
                price = bot.get_price(sym)
                res = bot.close_position(sym, price, reason="Ручное закрытие через UI")
                if res is not None:
                    out = {"status": "ok", "message": f"Позиция {sym} закрыта."}
                    if shadow_mode_cfg:
                        out["shadow_mode"] = True
                    return out
                if shadow_mode_cfg:
                    return {"status": "error", "shadow_mode": True, "message": f"Не удалось закрыть теневую позицию {sym}."}
                return {"status": "error", "message": f"Не удалось закрыть позицию {sym} на бирже."}
            except Exception as e:
                return {"status": "error", "message": str(e)}

        # Fallback: проверка внесписочных / Hedge позиций напрямую через клиент бота (только НЕ в SHADOW_MODE)
        client = None if shadow_mode_cfg else getattr(bot, "client", None)
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

    if shadow_mode_cfg:
        live_snap = read_live_status()
        snap_is_shadow = (
            isinstance(live_snap, dict)
            and bool(live_snap)
            and (
                live_snap.get("shadow_mode") is True
                or "shadow_positions" in live_snap
                or (
                    live_snap.get("shadow_mode") is not False
                    and isinstance(live_snap.get("shadow_status"), dict)
                )
            )
        )
        if snap_is_shadow:
            sh_pos_list = live_snap.get("shadow_positions")
            if not isinstance(sh_pos_list, list):
                sh_pos_list = (live_snap.get("shadow_status") or {}).get("positions")
            if isinstance(sh_pos_list, list):
                has_open = any(
                    isinstance(p, dict)
                    and str(p.get("symbol") or "").upper() == sym
                    and float(p.get("size") or 0.0) > 0
                    for p in sh_pos_list
                )
                if not has_open:
                    return {
                        "status": "error",
                        "shadow_mode": True,
                        "message": f"Открытая теневая позиция по {sym} не найдена.",
                    }
            enqueue_shadow_close(sym)
            return {
                "status": "ok",
                "queued": True,
                "shadow_mode": True,
                "message": f"SHADOW_MODE: команда закрытия теневой позиции {sym} передана процессу bot.py.",
            }
        return {
            "status": "error",
            "shadow_mode": True,
            "message": f"SHADOW_MODE: теневые позиции (в т.ч. {sym}) управляются запущенным процессом bot.py (реальные биржевые ордера отключены).",
        }

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
