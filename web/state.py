"""
Глобальное состояние и мост между ботом и веб-сервером (v2.3).
Поддерживает:
- Работу как внутри процесса bot.py, так и автономно (dashboard.py) через атомарный файл управления
- Потоковый парсер trade.log с конца файла без загрязнения строками 'Баланс:'
- Построение честной кривой капитала (Equity Curve) по реальным снимкам баланса и взвешенному PnL
"""

from typing import Any, Optional
import csv
import io
import json
import os
import re
import time
import threading
from datetime import datetime
from pathlib import Path

import config
from storage import BotDatabase

_STATE_LOCK = threading.RLock()
_BOT_INSTANCE: Optional[Any] = None
_IS_PAUSED: bool = False
_LAST_CONTROL_WRITE_TS: float = 0.0


def set_bot_instance(bot: Any):
    """Регистрирует активный экземпляр BybitMultiFuturesBot."""
    global _BOT_INSTANCE
    with _STATE_LOCK:
        _BOT_INSTANCE = bot


def get_bot_instance() -> Optional[Any]:
    """Возвращает активный экземпляр бота (если запущен в том же процессе)."""
    with _STATE_LOCK:
        return _BOT_INSTANCE


def _atomic_write_json(path, payload: dict):
    path = Path(path)
    try:
        tmp = path.with_suffix(".tmp")
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        tmp.replace(path)
    except Exception:
        pass


def _safe_read_json(path) -> dict:
    path = Path(path)
    if not path.exists():
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def write_control_state(
    is_paused: Optional[bool] = None,
    trading_mode: Optional[str] = None,
    pending_shadow_panic: Optional[bool] = None,
    pending_shadow_closes: Optional[list[str]] = None,
) -> float:
    """Атомарно сохраняет состояние управления (пауза, режим, shadow-действия) для синхронизации между bot.py и dashboard.py."""
    global _LAST_CONTROL_WRITE_TS
    with _STATE_LOCK:
        cur = _safe_read_json(config.CONTROL_FILE)
        prev_ts = float(cur.get("updated_at") or 0.0)
        now = max(time.time(), prev_ts + 1e-6, _LAST_CONTROL_WRITE_TS + 1e-6)
        if is_paused is not None:
            cur["is_paused"] = bool(is_paused)
        if trading_mode is not None:
            cur["trading_mode"] = str(trading_mode).strip().upper()
        if pending_shadow_panic is not None:
            if pending_shadow_panic:
                cur["pending_shadow_panic"] = True
            else:
                cur.pop("pending_shadow_panic", None)
        if pending_shadow_closes is not None:
            cleaned = [str(s).strip().upper() for s in pending_shadow_closes if str(s).strip()]
            if cleaned:
                cur["pending_shadow_closes"] = list(dict.fromkeys(cleaned))
            else:
                cur.pop("pending_shadow_closes", None)
        cur["updated_at"] = now
        _LAST_CONTROL_WRITE_TS = now
        _atomic_write_json(config.CONTROL_FILE, cur)
        return now


def enqueue_shadow_close(symbol: str) -> float:
    """Атомарно добавляет символ в очередь закрытия shadow-позиций для процесса bot.py."""
    sym = str(symbol or "").strip().upper()
    with _STATE_LOCK:
        cur = _safe_read_json(config.CONTROL_FILE)
        existing = cur.get("pending_shadow_closes")
        lst = list(existing) if isinstance(existing, list) else []
        if sym and sym not in lst:
            lst.append(sym)
        return write_control_state(pending_shadow_closes=lst)


def consume_shadow_control_actions() -> tuple[bool, list[str], float]:
    """Атомарно извлекает и очищает отложенные shadow-команды (panic / close) из файла управления IPC."""
    global _LAST_CONTROL_WRITE_TS
    with _STATE_LOCK:
        cur = _safe_read_json(config.CONTROL_FILE)
        pending_panic = bool(cur.get("pending_shadow_panic", False))
        raw_closes = cur.get("pending_shadow_closes")
        pending_closes = (
            list(dict.fromkeys(str(s).strip().upper() for s in raw_closes if str(s).strip()))
            if isinstance(raw_closes, list)
            else []
        )
        if not pending_panic and not pending_closes:
            return False, [], float(cur.get("updated_at") or 0.0)
        prev_ts = float(cur.get("updated_at") or 0.0)
        now = max(time.time(), prev_ts + 1e-6, _LAST_CONTROL_WRITE_TS + 1e-6)
        cur.pop("pending_shadow_panic", None)
        cur.pop("pending_shadow_closes", None)
        cur["updated_at"] = now
        _LAST_CONTROL_WRITE_TS = now
        _atomic_write_json(config.CONTROL_FILE, cur)
        return pending_panic, pending_closes, now


def read_control_state() -> dict:
    """Читает файл управления для синхронизации standalone dashboard.py и bot.py."""
    return _safe_read_json(config.CONTROL_FILE)


def write_control_action(action: str, symbol: Optional[str] = None) -> dict:
    """Помещает отложенную команду (close_position / panic_close_all) в файл управления IPC."""
    global _LAST_CONTROL_WRITE_TS
    act_name = str(action or "").strip().lower()
    sym_u = str(symbol or "").strip().upper()
    with _STATE_LOCK:
        cur = _safe_read_json(config.CONTROL_FILE)
        now = time.time()
        actions = cur.get("pending_actions")
        if not isinstance(actions, list):
            actions = []
        updated_existing = False
        for item in actions:
            if isinstance(item, dict) and item.get("action") == act_name and str(item.get("symbol") or "").upper() == sym_u:
                item["ts"] = now
                entry = item
                updated_existing = True
                break
        if not updated_existing:
            entry = {
                "id": f"{act_name}:{sym_u}:{int(now * 1000)}",
                "action": act_name,
                "symbol": sym_u,
                "ts": now,
            }
            actions.append(entry)
        cur["pending_actions"] = actions[-50:]
        cur["updated_at"] = now
        _LAST_CONTROL_WRITE_TS = now
        _atomic_write_json(config.CONTROL_FILE, cur)
        return dict(entry)


def consume_control_actions() -> tuple[list[dict], float]:
    """Атомарно извлекает и очищает очередь отложенных команд из файла управления."""
    global _LAST_CONTROL_WRITE_TS
    with _STATE_LOCK:
        cur = _safe_read_json(config.CONTROL_FILE)
        actions = cur.get("pending_actions")
        if not isinstance(actions, list) or not actions:
            return [], float(cur.get("updated_at") or 0.0)
        consumed = [dict(a) for a in actions if isinstance(a, dict)]
        now = time.time()
        cur["pending_actions"] = []
        cur["updated_at"] = now
        _LAST_CONTROL_WRITE_TS = now
        _atomic_write_json(config.CONTROL_FILE, cur)
        return consumed, now


def write_live_status(snapshot: dict):
    """Сохраняет срез состояния живого бота для standalone-дашборда."""
    payload = dict(snapshot)
    payload["updated_at"] = time.time()
    _atomic_write_json(config.LIVE_STATUS_FILE, payload)


def read_live_status(max_age_sec: float = 120.0) -> dict:
    """Читает последний срез живого бота, если он актуален."""
    data = _safe_read_json(config.LIVE_STATUS_FILE)
    if not data:
        return {}
    ts = float(data.get("updated_at") or 0.0)
    if ts > 0 and (time.time() - ts) <= max_age_sec:
        return data
    return {}


def is_bot_live(max_age_sec: float = 120.0) -> bool:
    """True, если процесс bot.py недавно публиковал живой срез состояния."""
    return bool(read_live_status(max_age_sec=max_age_sec))


def is_bot_paused() -> bool:
    global _IS_PAUSED
    with _STATE_LOCK:
        if _BOT_INSTANCE is not None and hasattr(_BOT_INSTANCE, "is_paused"):
            return bool(_BOT_INSTANCE.is_paused)
        ctrl = read_control_state()
        if "is_paused" in ctrl:
            return bool(ctrl["is_paused"])
        return _IS_PAUSED


def set_bot_paused(paused: bool, persist: bool = True):
    global _IS_PAUSED
    with _STATE_LOCK:
        _IS_PAUSED = bool(paused)
        is_mock = _BOT_INSTANCE is not None and "unittest.mock" in type(_BOT_INSTANCE).__module__
        if _BOT_INSTANCE is not None and hasattr(_BOT_INSTANCE, "is_paused"):
            _BOT_INSTANCE.is_paused = bool(paused)
        if persist and not is_mock:
            write_control_state(is_paused=bool(paused))


def _iter_lines_reversed(file_path: Path, chunk_size: int = 8192):
    """Эффективно читает строки текстового файла с конца без загрузки всего файла в RAM."""
    with open(file_path, "rb") as f:
        f.seek(0, os.SEEK_END)
        pos = f.tell()
        remainder = b""
        while pos > 0:
            read_size = min(chunk_size, pos)
            pos -= read_size
            f.seek(pos, os.SEEK_SET)
            chunk = f.read(read_size) + remainder
            lines = chunk.split(b"\n")
            remainder = lines[0]
            for raw_line in reversed(lines[1:]):
                yield raw_line.decode("utf-8", errors="ignore")
        if remainder:
            yield remainder.decode("utf-8", errors="ignore")


def parse_trade_log(limit: int = 50, include_reports: bool = False) -> list[dict]:
    """
    Парсит реальные торговые события из trade.log, читая файл с конца.
    Строки периодических отчетов ('Баланс:', 'Стартовый баланс:', 'WS Fill (Funding)')
    по умолчанию пропускаются, чтобы не вытеснять историю сделок.
    """
    trades: list[dict] = []
    limit = max(1, min(500, int(limit)))
    log_file = Path(config.TRADE_LOG_FILE)
    if not log_file.exists():
        return trades

    try:
        for raw_line in _iter_lines_reversed(log_file):
            line = raw_line.strip()
            if not line:
                continue

            if not include_reports and (
                "Баланс:" in line
                or "Стартовый баланс:" in line
                or "Базовый баланс" in line
                or "WS Fill (Funding)" in line
            ):
                continue

            time_match = re.match(r"^(\d{4}-\d{2}-\d{2}\s\d{2}:\d{2}:\d{2})", line)
            timestamp = time_match.group(1) if time_match else ""

            sym_match = re.search(r"\[([A-Z0-9]+)\]", line)
            symbol = sym_match.group(1) if sym_match else ""
            if symbol in ("INFO", "WARNING", "ERROR", "CRITICAL", "DEBUG"):
                # [INFO] не является парой — ищем следующий [SYMBOL]
                all_syms = re.findall(r"\[([A-Z0-9]+)\]", line)
                non_lvl = [s for s in all_syms if s not in ("INFO", "WARNING", "ERROR", "CRITICAL", "DEBUG")]
                symbol = non_lvl[0] if non_lvl else ""

            details_str = line
            if symbol and f"[{symbol}]" in line:
                details_str = line.split(f"[{symbol}]", 1)[-1].strip()
            elif "] " in line:
                details_str = line.split("] ", 1)[-1].strip()

            price_match = re.search(
                r"(?:@\s*|Вход:\s*|перенесен на\s*|подтянут к\s*|в безубыток(?:\s+после[^:]+)?:\s*)(\d+(?:\.\d+)?)",
                details_str
            )
            price_str = price_match.group(1) if price_match else ""

            def _make_entry(sym_val: str, act_val: str, pnl_s: str = "", pnl_f: float = 0.0, prof_val=None) -> dict:
                return {
                    "timestamp": timestamp,
                    "time": timestamp,
                    "symbol": sym_val,
                    "action": act_val,
                    "event": act_val,
                    "price": price_str,
                    "details": details_str,
                    "raw": details_str,
                    "pnl": pnl_s,
                    "pnl_pct": pnl_f,
                    "is_profit": prof_val
                }

            if "ЧАСТИЧНЫЙ ТЕЙК-ПРОФИТ" in line or "частичное закрытие TP1" in line:
                trades.append(_make_entry(symbol, "ЧАСТИЧНЫЙ TP", "", 0.0, True))
            elif "Трейлинг-стоп подтянут" in line:
                trades.append(_make_entry(symbol, "ТРЕЙЛИНГ-СТОП", "", 0.0, True))
            elif "Стоп перенесен в безубыток" in line or "Безубыток активирован" in line:
                trades.append(_make_entry(symbol, "БЕЗУБЫТОК", "", 0.0, True))
            elif re.search(r"\b(?:LONG|SHORT)\s+открыт\b", line):
                side = "LONG" if "LONG" in line else ("SHORT" if "SHORT" in line else "OPEN")
                trades.append(_make_entry(symbol, f"ОТКРЫТИЕ {side}", "", 0.0, None))
            elif "Закрыто" in line or "Закрыта" in line or "Позиция закрыта на бирже" in line:
                pnl_match = re.search(r"PnL:\s*([+-]?\d+\.?\d*)%", line)
                pnl_val = float(pnl_match.group(1)) if pnl_match else 0.0
                pnl_str = f"{pnl_match.group(1)}%" if pnl_match else ""
                is_profit = (pnl_val > 0) if pnl_match else False
                trades.append(_make_entry(symbol, "ЗАКРЫТИЕ", pnl_str, pnl_val, is_profit))
            elif include_reports and "Баланс:" in line:
                trades.append(_make_entry("SYSTEM", "ОТЧЕТ", "", 0.0, None))

            if len(trades) >= limit:
                break

    except Exception:
        pass

    return trades


def parse_equity_history(
    initial_balance: float = 50.0,
    margin_weight: float = 0.15,
    max_points: int = 200
) -> list[dict]:
    """
    Строит кривую капитала (Equity Curve) из trade.log:
    1. Опирается на реальные снимки капитала ('Стартовый баланс', 'Базовый баланс', 'Баланс ... Активные (0/...)').
    2. Промежуточные закрытия сделок учитывает с весом маржи (margin_weight), а не на 100% депозита.
    3. Корректно переводит локальные метки времени лога в Unix timestamp.
    """
    log_file = Path(config.TRADE_LOG_FILE)
    if not log_file.exists():
        return [{"time": int(time.time()), "value": round(float(initial_balance), 2)}]

    points: list[dict] = []
    curr_equity = float(initial_balance)
    has_real_anchor = False

    try:
        # Ограничиваем хвост до ~4000 последних строк для скорости на Termux
        lines_tail: list[str] = []
        for i, line in enumerate(_iter_lines_reversed(log_file)):
            lines_tail.append(line)
            if i >= 4000:
                break
        lines_tail.reverse()

        for raw_line in lines_tail:
            line = raw_line.strip()
            if not line:
                continue
            tm = re.match(r"^(\d{4}-\d{2}-\d{2}\s\d{2}:\d{2}:\d{2})", line)
            if not tm:
                continue
            try:
                ts = int(datetime.strptime(tm.group(1), "%Y-%m-%d %H:%M:%S").timestamp())
            except Exception:
                continue

            val_updated = False
            start_m = re.search(r"Стартовый баланс:\s*(\d+(?:\.\d+)?)\s*USDT", line)
            reset_m = re.search(r"Базовый баланс сброшен на:\s*(\d+(?:\.\d+)?)\s*USDT", line)
            flat_m = re.search(r"Баланс:\s*(\d+(?:\.\d+)?)\s*USDT\s*\|\s*Активные\s*\(0/", line)

            if start_m:
                curr_equity = float(start_m.group(1))
                has_real_anchor = True
                val_updated = True
            elif reset_m:
                curr_equity = float(reset_m.group(1))
                has_real_anchor = True
                val_updated = True
            elif flat_m:
                curr_equity = float(flat_m.group(1))
                has_real_anchor = True
                val_updated = True
            elif "Закрыто" in line or "Позиция закрыта на бирже" in line:
                pnl_m = re.search(r"PnL:\s*([+-]?\d+\.?\d*)%", line)
                if pnl_m:
                    pnl_pct = float(pnl_m.group(1))
                    if curr_equity > 0:
                        curr_equity += curr_equity * (pnl_pct / 100.0) * margin_weight
                        val_updated = True

            if val_updated:
                if points and ts <= points[-1]["time"]:
                    ts = points[-1]["time"] + 1
                # Не плодим дублирующие плоские точки при неизменном балансе каждые 10 мин
                if points and abs(points[-1]["value"] - round(curr_equity, 2)) < 1e-4 and flat_m and has_real_anchor:
                    continue
                points.append({"time": ts, "value": round(curr_equity, 2)})

    except Exception:
        pass

    if not points:
        return [{"time": int(time.time()), "value": round(float(initial_balance), 2)}]

    if len(points) > max_points:
        points = points[-max_points:]
    return points


# ==============================================================================
# v0.3.0: SQLite Storage + Fallback Helpers + Analytics + CSV/JSON Export
# ==============================================================================

def _db_exists() -> bool:
    try:
        db_file = Path(getattr(config, "DB_FILE", ""))
        return bool(str(db_file)) and db_file.exists()
    except Exception:
        return False


def get_trade_history(
    limit: int = 60,
    include_shadow: bool = False,
    only_shadow: bool = False,
    symbol: Optional[str] = None,
) -> list[dict]:
    """Читает историю сделок из SQLite с безопасным fallback на trade.log."""
    if _db_exists():
        try:
            rows = BotDatabase().get_trades(
                limit=limit,
                include_shadow=include_shadow,
                only_shadow=only_shadow,
                symbol=symbol,
            )
            if rows:
                return rows
        except Exception:
            pass

    if only_shadow:
        return []

    fallback = parse_trade_log(limit=limit)
    if symbol:
        sym_u = symbol.strip().upper()
        fallback = [r for r in fallback if str(r.get("symbol", "")).upper() == sym_u]
    return fallback


def get_equity_curve(
    initial_balance: float = 1000.0,
    max_points: int = 200,
    margin_weight: float = 0.25,
    is_shadow: bool = False,
) -> list[dict]:
    """Возвращает кривую капитала из SQLite с fallback на парсинг trade.log."""
    if _db_exists():
        try:
            pts = BotDatabase().get_equity_history(limit=max_points, is_shadow=is_shadow)
            if pts:
                return pts
        except Exception:
            pass

    if is_shadow:
        return [{"time": int(time.time()), "value": round(float(initial_balance), 2)}]
    return parse_equity_history(
        initial_balance=initial_balance,
        max_points=max_points,
        margin_weight=margin_weight,
    )


def get_analytics_summary(is_shadow: bool = False, limit: int = 2000) -> dict:
    """Агрегированная аналитика по сделкам, монетам, стратегиям и детектору деградации."""
    summary: Optional[dict] = None
    if _db_exists():
        try:
            summary = BotDatabase().get_analytics_summary(is_shadow=is_shadow, limit=limit)
        except Exception:
            summary = None

    if summary is None:
        summary = {
            "is_shadow": bool(is_shadow),
            "total_trades": 0,
            "wins": 0,
            "losses": 0,
            "win_rate": 0.0,
            "profit_factor": 0.0,
            "total_net_pnl_usdt": 0.0,
            "total_fees_usdt": 0.0,
            "total_funding_usdt": 0.0,
            "avg_pnl_pct": 0.0,
            "max_drawdown_pct": 0.0,
            "by_symbol": {},
            "by_strategy": {},
        }
        if not is_shadow:
            legacy = parse_trade_log(limit=min(limit, 500))
            closed = [t for t in legacy if t.get("is_profit") is not None]
            if closed:
                wins = sum(1 for t in closed if t.get("is_profit") is True)
                summary["total_trades"] = len(closed)
                summary["wins"] = wins
                summary["losses"] = len(closed) - wins
                summary["win_rate"] = round(wins / len(closed) * 100.0, 2)

    try:
        from optimizer import load_strategy_profile, detect_live_degradation
        raw_strat = (
            getattr(config, "SHADOW_STRATEGY", None) or getattr(config, "ACTIVE_STRATEGY", "SUPERTREND_PRO")
            if is_shadow
            else getattr(config, "ACTIVE_STRATEGY", "SUPERTREND_PRO")
        )
        use_switcher = getattr(config, "USE_REGIME_SWITCHER", False)
        if (
            (not is_shadow or not getattr(config, "SHADOW_STRATEGY", ""))
            and use_switcher
            and str(raw_strat).strip().upper() not in ("REGIME_SWITCHER", "ADAPTIVE")
        ):
            strat_name = "REGIME_SWITCHER"
        else:
            strat_name = str(raw_strat).strip().upper()
        profile = load_strategy_profile(strat_name)
        summary["degradation"] = detect_live_degradation(summary, profile=profile)
    except Exception:
        summary["degradation"] = {"status": "INSUFFICIENT_DATA", "degraded": False, "reasons": []}

    return summary


def export_trades_csv(limit: int = 1000, include_shadow: bool = False) -> str:
    """Экспортирует историю сделок в CSV из SQLite (или fallback trade.log)."""
    if _db_exists():
        try:
            db = BotDatabase()
            if db.get_trades(limit=1, include_shadow=include_shadow):
                return db.export_trades_csv(limit=limit, include_shadow=include_shadow)
        except Exception:
            pass

    rows = parse_trade_log(limit=limit)
    buf = io.StringIO()
    fields = ["timestamp", "symbol", "action", "price", "pnl", "is_profit", "details"]
    writer = csv.DictWriter(buf, fieldnames=fields, extrasaction="ignore")
    writer.writeheader()
    for r in rows:
        writer.writerow(r)
    return buf.getvalue()


def export_trades_json(limit: int = 1000, include_shadow: bool = False) -> list[dict]:
    """Экспортирует историю сделок в JSON-список из SQLite (или fallback trade.log)."""
    if _db_exists():
        try:
            db = BotDatabase()
            rows = db.export_trades_json(limit=limit, include_shadow=include_shadow)
            if rows:
                return rows
        except Exception:
            pass
    return parse_trade_log(limit=limit)
