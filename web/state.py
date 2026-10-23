"""
Глобальное состояние и мост между ботом и веб-сервером.
Позволяет дашборду работать как внутри процесса bot.py, так и автономно.
"""

from typing import Any, Optional
import os
import re
import threading
from collections import deque
from datetime import datetime

_STATE_LOCK = threading.RLock()
_BOT_INSTANCE: Optional[Any] = None
_IS_PAUSED: bool = False


def set_bot_instance(bot: Any):
    """Регистрирует активный экземпляр BybitMultiFuturesBot."""
    global _BOT_INSTANCE
    with _STATE_LOCK:
        _BOT_INSTANCE = bot


def get_bot_instance() -> Optional[Any]:
    """Возвращает активный экземпляр бота (если запущен в том же процессе)."""
    with _STATE_LOCK:
        return _BOT_INSTANCE


def is_bot_paused() -> bool:
    global _IS_PAUSED
    with _STATE_LOCK:
        if _BOT_INSTANCE is not None and hasattr(_BOT_INSTANCE, "is_paused"):
            return _BOT_INSTANCE.is_paused
        return _IS_PAUSED


def set_bot_paused(paused: bool):
    global _IS_PAUSED
    with _STATE_LOCK:
        _IS_PAUSED = paused
        if _BOT_INSTANCE is not None and hasattr(_BOT_INSTANCE, "is_paused"):
            _BOT_INSTANCE.is_paused = paused


def parse_trade_log(limit: int = 50) -> list[dict]:
    """
    Парсит последние строки из trade.log в структурированный список сделок.
    Использует потоковое чтение через deque для экономии памяти на Termux.
    """
    trades = []
    from pathlib import Path
    log_file = Path(__file__).resolve().parent.parent / "trade.log"
    if not log_file.exists():
        return trades

    try:
        with open(log_file, "r", encoding="utf-8", errors="ignore") as f:
            # Читаем только последние строки без загрузки всего файла в память
            recent_lines = deque(f, maxlen=limit * 3)

        # Читаем с конца
        for line in reversed(recent_lines):
            line = line.strip()
            if not line:
                continue

            time_match = re.match(r"^(\d{4}-\d{2}-\d{2}\s\d{2}:\d{2}:\d{2})", line)
            timestamp = time_match.group(1) if time_match else ""

            sym_match = re.search(r"\[([A-Z0-9]+)\]", line)
            symbol = sym_match.group(1) if sym_match else ""
            details_str = line.split("] ", 2)[-1] if "] " in line else line

            price_match = re.search(r"(?:@|Вход:|на|к)\s*(\d+(?:\.\d+)?)", details_str)
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

            # Ищем событие
            if "ЧАСТИЧНЫЙ ТЕЙК-ПРОФИТ" in line:
                trades.append(_make_entry(symbol, "ЧАСТИЧНЫЙ TP (1.5R)", "", 0.0, True))
            elif "Трейлинг-стоп подтянут" in line:
                trades.append(_make_entry(symbol, "ТРЕЙЛИНГ-СТОП", "", 0.0, True))
            elif "Стоп перенесен в безубыток" in line or "Безубыток активирован" in line:
                trades.append(_make_entry(symbol, "БЕЗУБЫТОК", "", 0.0, True))
            elif "открыт" in line:
                side = "LONG" if "LONG" in line else ("SHORT" if "SHORT" in line else "OPEN")
                trades.append(_make_entry(symbol, f"ОТКРЫТИЕ {side}", "", 0.0, None))
            elif "Закрыто" in line:
                pnl_match = re.search(r"PnL:\s*([+-]?\d+\.?\d*)%", line)
                pnl_val = float(pnl_match.group(1)) if pnl_match else 0.0
                pnl_str = f"{pnl_match.group(1)}%" if pnl_match else ""
                is_profit = (pnl_val > 0) if pnl_match else False
                trades.append(_make_entry(symbol, "ЗАКРЫТИЕ", pnl_str, pnl_val, is_profit))
            elif "Баланс:" in line:
                trades.append(_make_entry("SYSTEM", "ОТЧЕТ", "", 0.0, None))

            if len(trades) >= limit:
                break

    except Exception:
        pass

    return trades
