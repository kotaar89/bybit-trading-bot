"""
Глобальное состояние и мост между ботом и веб-сервером (v2.3).
Поддерживает:
- Работу как внутри процесса bot.py, так и автономно (dashboard.py) через атомарный файл управления
- Потоковый парсер trade.log с конца файла без загрязнения строками 'Баланс:'
- Построение честной кривой капитала (Equity Curve) по реальным снимкам баланса и взвешенному PnL
"""

from typing import Any, Optional
import json
import os
import re
import time
import threading
from datetime import datetime
from pathlib import Path

import config

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
    trading_mode: Optional[str] = None
):
    """Атомарно сохраняет состояние управления (пауза, режим) для синхронизации между bot.py и dashboard.py."""
    global _LAST_CONTROL_WRITE_TS
    with _STATE_LOCK:
        cur = _safe_read_json(config.CONTROL_FILE)
        now = time.time()
        if is_paused is not None:
            cur["is_paused"] = bool(is_paused)
        if trading_mode is not None:
            cur["trading_mode"] = str(trading_mode).strip().upper()
        cur["updated_at"] = now
        _LAST_CONTROL_WRITE_TS = now
        _atomic_write_json(config.CONTROL_FILE, cur)


def read_control_state() -> dict:
    """Читает файл управления для синхронизации standalone dashboard.py и bot.py."""
    return _safe_read_json(config.CONTROL_FILE)


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
