"""
Модуль персистентного хранения данных в SQLite (WAL) для Bybit Bot v0.3.0.
Обеспечивает:
- Хранение истории сделок, закрытых PnL с биржи (get_closed_pnl), снимков equity и runtime-состояния
- Потокобезопасную работу между bot.py и standalone dashboard.py (WAL + busy_timeout)
- Аналитику по монетам и стратегиям (WinRate, Profit Factor, Net PnL USDT, Fees, Funding, MaxDD)
- Экспорт сделок в CSV и JSON без внешних зависимостей (stdlib sqlite3, csv, json)
"""

from __future__ import annotations

import csv
import io
import json
import sqlite3
import threading
import time
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import config

_DB_LOCK = threading.RLock()
_INITIALIZED_PATHS: set[str] = set()

CLOSE_ACTIONS = (
    "ЗАКРЫТИЕ",
    "TIME_STOP",
    "SHADOW_CLOSE",
)


class BotDatabase:
    """Потокобезопасная обертка над SQLite WAL для хранения торговых данных и метрик."""

    def __init__(self, db_path: Optional[Path | str] = None):
        self._explicit_path = Path(db_path) if db_path is not None else None

    @property
    def db_path(self) -> Path:
        return self._explicit_path if self._explicit_path is not None else Path(config.DB_FILE)

    def _ensure_schema(self, conn: sqlite3.Connection):
        path_key = str(self.db_path.resolve())
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute("PRAGMA synchronous=NORMAL;")
        conn.execute("PRAGMA busy_timeout=5000;")

        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS trades (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ts REAL NOT NULL,
                timestamp_str TEXT NOT NULL,
                symbol TEXT NOT NULL,
                action TEXT NOT NULL,
                side TEXT DEFAULT '',
                strategy TEXT DEFAULT '',
                price REAL DEFAULT 0.0,
                qty REAL DEFAULT 0.0,
                pnl_pct REAL DEFAULT 0.0,
                net_pnl_usdt REAL,
                fee_usdt REAL DEFAULT 0.0,
                funding_usdt REAL DEFAULT 0.0,
                order_id TEXT DEFAULT '',
                exec_type TEXT DEFAULT '',
                pos_ref TEXT DEFAULT '',
                is_profit INTEGER,
                is_shadow INTEGER NOT NULL DEFAULT 0,
                details TEXT DEFAULT '',
                closed_pnl_synced INTEGER NOT NULL DEFAULT 0
            );

            CREATE INDEX IF NOT EXISTS idx_trades_shadow_ts ON trades (is_shadow, ts DESC);
            CREATE INDEX IF NOT EXISTS idx_trades_symbol_ts ON trades (symbol, ts DESC);
            CREATE INDEX IF NOT EXISTS idx_trades_order_id ON trades (order_id);

            CREATE TABLE IF NOT EXISTS exchange_closed_pnl (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol TEXT NOT NULL,
                order_id TEXT NOT NULL,
                side TEXT DEFAULT '',
                qty REAL DEFAULT 0.0,
                avg_entry_price REAL DEFAULT 0.0,
                avg_exit_price REAL DEFAULT 0.0,
                closed_pnl REAL DEFAULT 0.0,
                open_fee REAL DEFAULT 0.0,
                close_fee REAL DEFAULT 0.0,
                created_time INTEGER DEFAULT 0,
                updated_time INTEGER DEFAULT 0,
                raw_json TEXT DEFAULT '',
                synced_at REAL NOT NULL,
                UNIQUE(symbol, order_id, updated_time)
            );

            CREATE INDEX IF NOT EXISTS idx_ecp_sym_time ON exchange_closed_pnl (symbol, updated_time DESC);

            CREATE TABLE IF NOT EXISTS equity_snapshots (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ts INTEGER NOT NULL,
                equity REAL NOT NULL,
                balance REAL DEFAULT 0.0,
                unrealized_pnl REAL DEFAULT 0.0,
                source TEXT DEFAULT 'bot',
                is_shadow INTEGER NOT NULL DEFAULT 0
            );

            CREATE INDEX IF NOT EXISTS idx_eq_shadow_ts ON equity_snapshots (is_shadow, ts ASC);

            CREATE TABLE IF NOT EXISTS runtime_state (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL,
                updated_at REAL NOT NULL
            );
            """
        )
        conn.commit()
        _INITIALIZED_PATHS.add(path_key)

    @contextmanager
    def _connect(self):
        target = self.db_path
        target.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(str(target), timeout=10.0, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        try:
            conn.execute("PRAGMA busy_timeout=5000;")
            if str(target.resolve()) not in _INITIALIZED_PATHS:
                self._ensure_schema(conn)
            else:
                # Легкая проверка на случай подмены/удаления файла БД в тестах
                tbl = conn.execute(
                    "SELECT name FROM sqlite_master WHERE type='table' AND name='trades'"
                ).fetchone()
                if tbl is None:
                    self._ensure_schema(conn)
            yield conn
        finally:
            conn.close()

    def init_db(self):
        with _DB_LOCK:
            with self._connect() as conn:
                self._ensure_schema(conn)

    # -----------------------------------------------------------------
    # Торговые события и сверка с get_closed_pnl
    # -----------------------------------------------------------------
    def record_trade_event(
        self,
        symbol: str,
        action: str,
        side: str = "",
        strategy: str = "",
        price: float = 0.0,
        qty: float = 0.0,
        pnl_pct: float = 0.0,
        net_pnl_usdt: Optional[float] = None,
        fee_usdt: float = 0.0,
        funding_usdt: float = 0.0,
        order_id: str = "",
        exec_type: str = "",
        pos_ref: str = "",
        is_profit: Optional[bool] = None,
        is_shadow: bool = False,
        details: str = "",
        ts: Optional[float] = None,
    ) -> int:
        now_ts = float(ts if ts is not None else time.time())
        ts_str = datetime.fromtimestamp(now_ts).strftime("%Y-%m-%d %H:%M:%S")
        profit_int = None if is_profit is None else (1 if is_profit else 0)
        with _DB_LOCK:
            with self._connect() as conn:
                cur = conn.execute(
                    """
                    INSERT INTO trades (
                        ts, timestamp_str, symbol, action, side, strategy,
                        price, qty, pnl_pct, net_pnl_usdt, fee_usdt, funding_usdt,
                        order_id, exec_type, pos_ref, is_profit, is_shadow, details,
                        closed_pnl_synced
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        now_ts,
                        ts_str,
                        str(symbol or "").upper(),
                        str(action or ""),
                        str(side or ""),
                        str(strategy or ""),
                        float(price or 0.0),
                        float(qty or 0.0),
                        float(pnl_pct or 0.0),
                        float(net_pnl_usdt) if net_pnl_usdt is not None else None,
                        float(fee_usdt or 0.0),
                        float(funding_usdt or 0.0),
                        str(order_id or ""),
                        str(exec_type or ""),
                        str(pos_ref or ""),
                        profit_int,
                        1 if is_shadow else 0,
                        str(details or ""),
                        0,
                    ),
                )
                conn.commit()
                return int(cur.lastrowid)

    def upsert_closed_pnl_record(
        self,
        rec: dict,
        margin_used: float = 0.0,
        allow_orphan: bool = True,
        min_orphan_ts: Optional[float] = None,
    ) -> bool:
        """
        Сохраняет запись из Bybit v5 get_closed_pnl и привязывает точный Net PnL (USDT)
        к соответствующей записи закрытия в таблице trades.
        Возвращает True, если новая запись была добавлена или обновлена.
        """
        sym = str(rec.get("symbol") or "").upper()
        oid = str(rec.get("orderId") or "")
        if not sym or not oid:
            return False

        side = str(rec.get("side") or "")
        qty = float(rec.get("closedSize") or rec.get("qty") or 0.0)
        avg_entry = float(rec.get("avgEntryPrice") or 0.0)
        avg_exit = float(rec.get("avgExitPrice") or rec.get("orderPrice") or 0.0)
        closed_pnl = float(rec.get("closedPnl") or 0.0)
        open_fee = float(rec.get("openFee") or 0.0)
        close_fee = float(rec.get("closeFee") or 0.0)
        created_ms = int(rec.get("createdTime") or 0)
        updated_ms = int(rec.get("updatedTime") or created_ms or 0)
        total_fee = abs(open_fee) + abs(close_fee)
        now_ts = time.time()
        rec_sec = (updated_ms / 1000.0) if updated_ms > 0 else now_ts

        with _DB_LOCK:
            with self._connect() as conn:
                existing = conn.execute(
                    "SELECT id FROM exchange_closed_pnl WHERE symbol=? AND order_id=? AND updated_time=?",
                    (sym, oid, updated_ms),
                ).fetchone()
                if existing is not None:
                    return False

                # Ищем соответствующую сделку закрытия в trades (сначала по order_id, затем несверенную по символу и времени)
                target_row = conn.execute(
                    """
                    SELECT id, action, net_pnl_usdt, fee_usdt, pnl_pct, price, qty, pos_ref
                    FROM trades
                    WHERE is_shadow=0 AND symbol=? AND order_id=? AND action IN ('ЗАКРЫТИЕ', 'TIME_STOP', 'ЧАСТИЧНЫЙ TP')
                    ORDER BY id DESC LIMIT 1
                    """,
                    (sym, oid),
                ).fetchone()

                if target_row is None:
                    target_row = conn.execute(
                        """
                        SELECT id, action, net_pnl_usdt, fee_usdt, pnl_pct, price, qty, pos_ref
                        FROM trades
                        WHERE is_shadow=0 AND symbol=? AND closed_pnl_synced=0
                          AND action IN ('ЗАКРЫТИЕ', 'TIME_STOP', 'ЧАСТИЧНЫЙ TP')
                          AND ABS(ts - ?) <= 14400
                        ORDER BY ABS(ts - ?) ASC, id DESC LIMIT 1
                        """,
                        (sym, rec_sec, rec_sec),
                    ).fetchone()

                if target_row is None and (not allow_orphan or (min_orphan_ts is not None and rec_sec < min_orphan_ts)):
                    return False

                conn.execute(
                    """
                    INSERT INTO exchange_closed_pnl (
                        symbol, order_id, side, qty, avg_entry_price, avg_exit_price,
                        closed_pnl, open_fee, close_fee, created_time, updated_time,
                        raw_json, synced_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        sym,
                        oid,
                        side,
                        qty,
                        avg_entry,
                        avg_exit,
                        closed_pnl,
                        open_fee,
                        close_fee,
                        created_ms,
                        updated_ms,
                        json.dumps(rec, ensure_ascii=False),
                        now_ts,
                    ),
                )

                if target_row is not None:
                    new_net = closed_pnl
                    is_prof = 1 if new_net > 0 else 0
                    upd_price = avg_exit if avg_exit > 0 else float(target_row["price"] or 0.0)
                    pnl_pct = float(target_row["pnl_pct"] or 0.0)
                    pos_ref = str(target_row["pos_ref"] or "")
                    has_tp1_sibling = False
                    if pos_ref and str(target_row["action"] or "") in ("ЗАКРЫТИЕ", "TIME_STOP"):
                        sib = conn.execute(
                            "SELECT id, net_pnl_usdt FROM trades WHERE is_shadow=0 AND pos_ref=? AND action='ЧАСТИЧНЫЙ TP' LIMIT 1",
                            (pos_ref,),
                        ).fetchone()
                        has_tp1_sibling = sib is not None
                        if margin_used > 0 and sib is not None and sib["net_pnl_usdt"] is not None:
                            pnl_pct = ((new_net + float(sib["net_pnl_usdt"])) / margin_used) * 100.0
                    if margin_used > 0 and not has_tp1_sibling:
                        pnl_pct = (new_net / margin_used) * 100.0
                    conn.execute(
                        """
                        UPDATE trades
                        SET net_pnl_usdt=?, fee_usdt=?, is_profit=?, price=?, pnl_pct=?,
                            order_id=CASE WHEN order_id='' THEN ? ELSE order_id END,
                            closed_pnl_synced=1
                        WHERE id=?
                        """,
                        (new_net, total_fee, is_prof, upd_price, pnl_pct, oid, target_row["id"]),
                    )
                else:
                    # Если закрытие произошло внешне и еще не залогировано, создаем запись
                    ts_str = datetime.fromtimestamp(rec_sec).strftime("%Y-%m-%d %H:%M:%S")
                    pnl_pct_est = (closed_pnl / margin_used * 100.0) if margin_used > 0 else 0.0
                    conn.execute(
                        """
                        INSERT INTO trades (
                            ts, timestamp_str, symbol, action, side, strategy,
                            price, qty, pnl_pct, net_pnl_usdt, fee_usdt, funding_usdt,
                            order_id, exec_type, pos_ref, is_profit, is_shadow, details,
                            closed_pnl_synced
                        ) VALUES (?, ?, ?, 'ЗАКРЫТИЕ', ?, '', ?, ?, ?, ?, ?, 0.0, ?, 'BybitClosedPnL', '', ?, 0, ?, 1)
                        """,
                        (
                            rec_sec,
                            ts_str,
                            sym,
                            side,
                            avg_exit,
                            qty,
                            pnl_pct_est,
                            closed_pnl,
                            total_fee,
                            oid,
                            1 if closed_pnl > 0 else 0,
                            f"Bybit closedPnl={closed_pnl:+.4f} USDT | Fee={total_fee:.4f} | Entry={avg_entry} -> Exit={avg_exit}",
                        ),
                    )
                conn.commit()
                return True

    def get_trades(
        self,
        limit: int = 60,
        include_shadow: bool = False,
        only_shadow: bool = False,
        symbol: Optional[str] = None,
    ) -> list[dict]:
        limit = max(1, min(1000, int(limit)))
        where = []
        params: list[Any] = []
        if only_shadow:
            where.append("is_shadow = 1")
        elif not include_shadow:
            where.append("is_shadow = 0")
        if symbol:
            where.append("symbol = ?")
            params.append(symbol.strip().upper())

        sql_where = ("WHERE " + " AND ".join(where)) if where else ""
        sql = f"""
            SELECT id, ts, timestamp_str, symbol, action, side, strategy, price, qty,
                   pnl_pct, net_pnl_usdt, fee_usdt, funding_usdt, order_id, exec_type,
                   pos_ref, is_profit, is_shadow, details, closed_pnl_synced
            FROM trades
            {sql_where}
            ORDER BY ts DESC, id DESC
            LIMIT ?
        """
        params.append(limit)

        with _DB_LOCK:
            with self._connect() as conn:
                rows = conn.execute(sql, params).fetchall()

        out: list[dict] = []
        for r in rows:
            pnl_pct = float(r["pnl_pct"] or 0.0)
            net_usdt = r["net_pnl_usdt"]
            if net_usdt is not None:
                pnl_str = f"{float(net_usdt):+.2f}$ ({pnl_pct:+.2f}%)" if pnl_pct != 0 else f"{float(net_usdt):+.2f}$"
            elif pnl_pct != 0:
                pnl_str = f"{pnl_pct:+.2f}%"
            else:
                pnl_str = ""
            px = float(r["price"] or 0.0)
            out.append(
                {
                    "id": r["id"],
                    "ts": float(r["ts"]),
                    "timestamp": r["timestamp_str"],
                    "time": r["timestamp_str"],
                    "symbol": r["symbol"],
                    "action": r["action"],
                    "event": r["action"],
                    "side": r["side"],
                    "strategy": r["strategy"],
                    "price": f"{px:g}" if px > 0 else "",
                    "price_num": px,
                    "qty": float(r["qty"] or 0.0),
                    "pnl": pnl_str,
                    "pnl_pct": round(pnl_pct, 4),
                    "net_pnl_usdt": round(float(net_usdt), 4) if net_usdt is not None else None,
                    "fee_usdt": round(float(r["fee_usdt"] or 0.0), 6),
                    "funding_usdt": round(float(r["funding_usdt"] or 0.0), 6),
                    "order_id": r["order_id"],
                    "exec_type": r["exec_type"],
                    "pos_ref": r["pos_ref"],
                    "is_profit": None if r["is_profit"] is None else bool(r["is_profit"]),
                    "is_shadow": bool(r["is_shadow"]),
                    "details": r["details"],
                    "raw": r["details"],
                    "closed_pnl_synced": bool(r["closed_pnl_synced"]),
                }
            )
        return out

    # -----------------------------------------------------------------
    # Снимки капитала (Equity Curve)
    # -----------------------------------------------------------------
    def record_equity_snapshot(
        self,
        equity: float,
        balance: float = 0.0,
        unrealized_pnl: float = 0.0,
        source: str = "bot",
        is_shadow: bool = False,
        ts: Optional[int] = None,
    ):
        eq_val = float(equity or 0.0)
        if eq_val <= 0:
            return
        now_ts = int(ts if ts is not None else time.time())
        shadow_int = 1 if is_shadow else 0

        with _DB_LOCK:
            with self._connect() as conn:
                last = conn.execute(
                    "SELECT ts, equity FROM equity_snapshots WHERE is_shadow=? ORDER BY ts DESC, id DESC LIMIT 1",
                    (shadow_int,),
                ).fetchone()
                if last is not None:
                    # Не плодим дубликаты при одинаковом equity от периодического тикера
                    if source == "periodic" and abs(float(last["equity"]) - round(eq_val, 4)) < 1e-3 and (now_ts - int(last["ts"])) < 600:
                        return
                    if now_ts <= int(last["ts"]):
                        now_ts = int(last["ts"]) + 1

                conn.execute(
                    """
                    INSERT INTO equity_snapshots (ts, equity, balance, unrealized_pnl, source, is_shadow)
                    VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    (
                        now_ts,
                        round(eq_val, 4),
                        round(float(balance or 0.0), 4),
                        round(float(unrealized_pnl or 0.0), 4),
                        str(source or "bot"),
                        shadow_int,
                    ),
                )
                conn.commit()

    def get_equity_history(self, limit: int = 200, is_shadow: bool = False) -> list[dict]:
        limit = max(1, min(2000, int(limit)))
        shadow_int = 1 if is_shadow else 0
        with _DB_LOCK:
            with self._connect() as conn:
                rows = conn.execute(
                    """
                    SELECT ts, equity FROM (
                        SELECT ts, equity, id FROM equity_snapshots
                        WHERE is_shadow=?
                        ORDER BY ts DESC, id DESC
                        LIMIT ?
                    ) sub
                    ORDER BY ts ASC, id ASC
                    """,
                    (shadow_int, limit),
                ).fetchall()

        points: list[dict] = []
        for r in rows:
            t = int(r["ts"])
            if points and t <= points[-1]["time"]:
                t = points[-1]["time"] + 1
            points.append({"time": t, "value": round(float(r["equity"]), 2)})
        return points

    # -----------------------------------------------------------------
    # Аналитика по монетам и стратегиям
    # -----------------------------------------------------------------
    def get_analytics_summary(self, is_shadow: bool = False, limit: int = 2000) -> dict:
        shadow_int = 1 if is_shadow else 0
        limit_eff = max(1, int(limit))
        with _DB_LOCK:
            with self._connect() as conn:
                trade_rows = conn.execute(
                    """
                    SELECT id, ts, symbol, action, strategy, pnl_pct, net_pnl_usdt,
                           fee_usdt, funding_usdt, pos_ref, is_profit
                    FROM trades
                    WHERE is_shadow=?
                    ORDER BY ts DESC, id DESC
                    LIMIT ?
                    """,
                    (shadow_int, limit_eff),
                ).fetchall()
                trade_rows = list(reversed(trade_rows))
                eq_rows = conn.execute(
                    "SELECT equity FROM equity_snapshots WHERE is_shadow=? ORDER BY ts ASC, id ASC",
                    (shadow_int,),
                ).fetchall()

        # Группировка позиций: по pos_ref если задан, иначе по каждой закрывающей строке
        grouped: dict[str, dict] = {}
        total_fees = 0.0
        total_funding = 0.0
        seq = 0

        for r in trade_rows:
            total_fees += float(r["fee_usdt"] or 0.0)
            total_funding += float(r["funding_usdt"] or 0.0)
            action = str(r["action"] or "")
            is_close_like = action in CLOSE_ACTIONS or action == "ЧАСТИЧНЫЙ TP"
            if not is_close_like:
                continue

            pos_ref = str(r["pos_ref"] or "").strip()
            if not pos_ref:
                if action == "ЧАСТИЧНЫЙ TP":
                    continue
                seq += 1
                pos_ref = f"row_{r['id']}_{seq}"

            entry = grouped.setdefault(
                pos_ref,
                {
                    "symbol": str(r["symbol"] or "UNKNOWN"),
                    "strategy": str(r["strategy"] or "UNKNOWN") or "UNKNOWN",
                    "net_pnl_usdt": 0.0,
                    "has_usdt": False,
                    "pnl_pct": 0.0,
                    "closed": False,
                    "ts": float(r["ts"] or 0.0),
                },
            )
            if r["strategy"]:
                entry["strategy"] = str(r["strategy"])
            if r["net_pnl_usdt"] is not None:
                entry["net_pnl_usdt"] += float(r["net_pnl_usdt"])
                entry["has_usdt"] = True
            if action in CLOSE_ACTIONS:
                # Закрытие хранит уже взвешенный pnl_pct всей позиции или итог
                entry["pnl_pct"] = float(r["pnl_pct"] or 0.0)
                entry["closed"] = True
                entry["ts"] = float(r["ts"] or entry["ts"])

        # Оцениваем только завершенные позиции
        closed_positions = [v for v in grouped.values() if v["closed"]]
        total_trades = len(closed_positions)
        wins = 0
        losses = 0
        gross_profit_usdt = 0.0
        gross_loss_usdt = 0.0
        gross_profit_pct = 0.0
        gross_loss_pct = 0.0
        total_net_pnl_usdt = 0.0
        pnl_pct_sum = 0.0

        by_symbol: dict[str, dict] = {}
        by_strategy: dict[str, dict] = {}

        def _acc(bucket: dict, key: str, is_win: bool, net_u: float, has_u: bool, p_pct: float):
            b = bucket.setdefault(
                key,
                {
                    "trades": 0,
                    "wins": 0,
                    "losses": 0,
                    "win_rate": 0.0,
                    "net_pnl_usdt": 0.0,
                    "pnl_pct_sum": 0.0,
                    "avg_pnl_pct": 0.0,
                    "gross_profit": 0.0,
                    "gross_loss": 0.0,
                    "profit_factor": 0.0,
                },
            )
            b["trades"] += 1
            if is_win:
                b["wins"] += 1
            else:
                b["losses"] += 1
            b["pnl_pct_sum"] += p_pct
            metric_val = net_u if has_u else p_pct
            if has_u:
                b["net_pnl_usdt"] += net_u
            if metric_val > 0:
                b["gross_profit"] += metric_val
            elif metric_val < 0:
                b["gross_loss"] += abs(metric_val)

        for p in closed_positions:
            has_u = p["has_usdt"]
            net_u = p["net_pnl_usdt"]
            p_pct = p["pnl_pct"]
            judge_val = net_u if has_u else p_pct
            is_win = judge_val > 0
            if is_win:
                wins += 1
            else:
                losses += 1
            pnl_pct_sum += p_pct
            if has_u:
                total_net_pnl_usdt += net_u
                if net_u > 0:
                    gross_profit_usdt += net_u
                elif net_u < 0:
                    gross_loss_usdt += abs(net_u)
            if p_pct > 0:
                gross_profit_pct += p_pct
            elif p_pct < 0:
                gross_loss_pct += abs(p_pct)

            _acc(by_symbol, p["symbol"], is_win, net_u, has_u, p_pct)
            _acc(by_strategy, p["strategy"] or "UNKNOWN", is_win, net_u, has_u, p_pct)

        for bucket in (by_symbol, by_strategy):
            for _, b in bucket.items():
                cnt = b["trades"]
                b["win_rate"] = round((b["wins"] / cnt * 100.0) if cnt > 0 else 0.0, 2)
                b["avg_pnl_pct"] = round((b["pnl_pct_sum"] / cnt) if cnt > 0 else 0.0, 2)
                b["net_pnl_usdt"] = round(b["net_pnl_usdt"], 4)
                gl = b["gross_loss"]
                gp = b["gross_profit"]
                b["profit_factor"] = round(min(10.0, gp / gl) if gl > 0 else (10.0 if gp > 0 else 0.0), 2)
                b.pop("gross_profit", None)
                b.pop("gross_loss", None)
                b.pop("pnl_pct_sum", None)

        if gross_loss_usdt > 0 or gross_profit_usdt > 0:
            pf = min(10.0, gross_profit_usdt / gross_loss_usdt) if gross_loss_usdt > 0 else 10.0
        elif gross_loss_pct > 0 or gross_profit_pct > 0:
            pf = min(10.0, gross_profit_pct / gross_loss_pct) if gross_loss_pct > 0 else 10.0
        else:
            pf = 0.0

        # MaxDD по снимкам equity
        max_dd_pct = 0.0
        peak = 0.0
        for row in eq_rows:
            val = float(row["equity"] or 0.0)
            if val > peak:
                peak = val
            if peak > 0:
                dd = ((val - peak) / peak) * 100.0
                if dd < max_dd_pct:
                    max_dd_pct = dd

        return {
            "is_shadow": is_shadow,
            "total_trades": total_trades,
            "wins": wins,
            "losses": losses,
            "win_rate": round((wins / total_trades * 100.0) if total_trades > 0 else 0.0, 2),
            "profit_factor": round(pf, 2),
            "total_net_pnl_usdt": round(total_net_pnl_usdt, 4),
            "total_fees_usdt": round(total_fees, 4),
            "total_funding_usdt": round(total_funding, 4),
            "avg_pnl_pct": round((pnl_pct_sum / total_trades) if total_trades > 0 else 0.0, 2),
            "max_drawdown_pct": round(max_dd_pct, 2),
            "by_symbol": by_symbol,
            "by_strategy": by_strategy,
        }

    # -----------------------------------------------------------------
    # Экспорт сделок (CSV / JSON)
    # -----------------------------------------------------------------
    def export_trades_csv(self, limit: int = 1000, include_shadow: bool = False) -> str:
        rows = self.get_trades(limit=limit, include_shadow=include_shadow)
        buf = io.StringIO()
        fields = [
            "id", "timestamp", "symbol", "action", "side", "strategy",
            "price_num", "qty", "pnl_pct", "net_pnl_usdt", "fee_usdt",
            "funding_usdt", "order_id", "exec_type", "is_shadow", "details",
        ]
        writer = csv.DictWriter(buf, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
        return buf.getvalue()

    def export_trades_json(self, limit: int = 1000, include_shadow: bool = False) -> list[dict]:
        return self.get_trades(limit=limit, include_shadow=include_shadow)

    # -----------------------------------------------------------------
    # Персистирование runtime-состояния (start_day_balance, peak_equity, cooldowns)
    # -----------------------------------------------------------------
    def set_runtime_value(self, key: str, value: Any):
        payload = json.dumps(value, ensure_ascii=False)
        now = time.time()
        with _DB_LOCK:
            with self._connect() as conn:
                conn.execute(
                    """
                    INSERT INTO runtime_state (key, value, updated_at)
                    VALUES (?, ?, ?)
                    ON CONFLICT(key) DO UPDATE SET value=excluded.value, updated_at=excluded.updated_at
                    """,
                    (str(key), payload, now),
                )
                conn.commit()

    def get_runtime_value(self, key: str, default: Any = None) -> Any:
        with _DB_LOCK:
            with self._connect() as conn:
                row = conn.execute(
                    "SELECT value FROM runtime_state WHERE key=?", (str(key),)
                ).fetchone()
        if row is None:
            return default
        try:
            return json.loads(row["value"])
        except Exception:
            return default

    def save_runtime_bundle(self, state_dict: dict):
        now = time.time()
        with _DB_LOCK:
            with self._connect() as conn:
                for k, v in state_dict.items():
                    conn.execute(
                        """
                        INSERT INTO runtime_state (key, value, updated_at)
                        VALUES (?, ?, ?)
                        ON CONFLICT(key) DO UPDATE SET value=excluded.value, updated_at=excluded.updated_at
                        """,
                        (str(k), json.dumps(v, ensure_ascii=False), now),
                    )
                conn.commit()

    def get_all_runtime_state(self) -> dict:
        with _DB_LOCK:
            with self._connect() as conn:
                rows = conn.execute("SELECT key, value FROM runtime_state").fetchall()
        out = {}
        for r in rows:
            try:
                out[r["key"]] = json.loads(r["value"])
            except Exception:
                out[r["key"]] = r["value"]
        return out
