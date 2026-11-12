"""
Комплексный набор unit и интеграционных тестов для Bybit Trading Bot v0.3.0.
Все временные БД, логи и профили изолированы в TemporaryDirectory без мусора в корне проекта.
"""

import csv
import io
import json
import os
import sqlite3
import tempfile
import threading
import time
import unittest
from collections import deque
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd

import config
from storage import BotDatabase
from risk_manager import PositionRiskState, MarketRegimeFilter, PortfolioRiskManager
from strategies import get_strategy
from strategies.regime_switcher import RegimeSwitcherStrategy
from universe import DynamicUniverseSelector
from shadow import ShadowEngine
import bot as bot_mod
from bot import BybitMultiFuturesBot, RateLimitController
import backtest as bt_mod
import optimizer as opt_mod
from web import state as web_state
from web import app as web_app


def _make_ohlcv(n: int = 300, start_price: float = 100.0, trend: float = 0.001, vol: float = 0.01, seed: int = 42, freq: str = "1h") -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    ts = pd.date_range("2025-01-01", periods=n, freq=freq)
    rets = trend + rng.normal(0.0, vol, size=n)
    close = start_price * np.cumprod(1.0 + rets)
    open_p = np.roll(close, 1)
    open_p[0] = start_price
    spread = np.abs(rng.normal(0.003, 0.0015, size=n))
    high = np.maximum(open_p, close) * (1.0 + spread)
    low = np.minimum(open_p, close) * (1.0 - spread)
    volume = rng.uniform(500.0, 2500.0, size=n)
    return pd.DataFrame({
        "timestamp": ts,
        "open": open_p,
        "high": high,
        "low": low,
        "close": close,
        "volume": volume,
        "turnover": (close * volume).astype(str),
    })


def _make_bare_bot(tmpdir: str, db: BotDatabase | None = None) -> BybitMultiFuturesBot:
    b = object.__new__(BybitMultiFuturesBot)
    b.lock = threading.RLock()
    b.state_file = Path(tmpdir) / "bot_state.json"
    b.db = db
    b.positions = {}
    b.risk_states = {}
    b.breakeven_applied = {}
    b.last_handled_candles = {}
    b.position_modes = {}
    b.is_paused = False
    b.shadow_engine = None
    b.active_symbols = ["ETHUSDT", "SOLUSDT"]
    b.circuit_breaker_active = False
    b.start_day = pd.Timestamp("2026-01-10").date()
    b.start_day_balance = 1000.0
    b.last_known_equity = 1000.0
    b.last_equity_ok = True
    b.last_balance_ok = True
    b.last_exec_prices = {}
    b.last_exec_info = {}
    b._seen_exec_ids = deque(maxlen=100)
    b._pending_pnl_sync = []
    b.kline_cache = {}
    b.kline_cache_ts = {}
    b._confirmed_candles = deque(maxlen=50)
    b._candle_event = threading.Event()
    b._public_ws_gen = 1
    b._derivatives_cache = {}
    b._last_regime_snapshot = {}
    b.ws = None
    b.ws_prices = {}
    b.order_ws_cache = {}
    b.rate_limiter = RateLimitController()
    b.portfolio_risk = PortfolioRiskManager(peak_equity=1000.0)
    b.regime_filter = MarketRegimeFilter(ema_period=50, fast_ema_period=20)
    b.strategy = get_strategy("SUPERTREND_PRO", st_period=10, st_multiplier=3.0)
    b.instruments = {
        "ETHUSDT": {
            "qty_step": 0.01, "min_qty": 0.01, "min_notional": 5.0,
            "tick_size": 0.01, "qty_decimals": 2, "price_decimals": 2,
            "max_leverage": 25.0, "last_loaded_ts": time.time(),
        },
        "SOLUSDT": {
            "qty_step": 0.1, "min_qty": 0.1, "min_notional": 5.0,
            "tick_size": 0.01, "qty_decimals": 1, "price_decimals": 2,
            "max_leverage": 20.0, "last_loaded_ts": time.time(),
        },
    }
    b.client = MagicMock()
    return b


class TestV3StorageAndReconciliation(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.db_path = Path(self.tmp.name) / "bot_test.db"
        self.p_db = patch.object(config, "DB_FILE", str(self.db_path))
        self.p_state = patch.object(config, "STATE_FILE", str(Path(self.tmp.name) / "bot_state.json"))
        self.p_ctrl = patch.object(config, "CONTROL_FILE", str(Path(self.tmp.name) / "control.json"))
        self.p_live = patch.object(config, "LIVE_STATUS_FILE", str(Path(self.tmp.name) / "live_status.json"))
        self.p_tlog = patch.object(config, "TRADE_LOG_FILE", str(Path(self.tmp.name) / "trade.log"))
        self.p_blog = patch.object(config, "BOT_LOG_FILE", str(Path(self.tmp.name) / "bot.log"))
        for p in (self.p_db, self.p_state, self.p_ctrl, self.p_live, self.p_tlog, self.p_blog):
            p.start()
            self.addCleanup(p.stop)

    def test_sqlite_wal_closed_pnl_reconciliation_and_exports(self):
        db = BotDatabase(self.db_path)
        now = 1_750_000_000.0
        # 1. Реальная позиция с TP1 + ЗАКРЫТИЕ по pos_ref, сверяемая по order_id и по времени
        db.record_trade_event(
            symbol="ETHUSDT", action="ВХОД", side="Buy", strategy="SUPERTREND_PRO",
            price=3000.0, qty=0.4, order_id="ord-entry-1", pos_ref="pos-eth-1", ts=now - 500,
        )
        conn = sqlite3.connect(str(self.db_path))
        try:
            mode = conn.execute("PRAGMA journal_mode;").fetchone()[0]
        finally:
            conn.close()
        self.assertEqual(str(mode).lower(), "wal")
        db.record_trade_event(
            symbol="ETHUSDT", action="ЧАСТИЧНЫЙ TP", side="Sell", strategy="SUPERTREND_PRO",
            price=3120.0, qty=0.2, pnl_pct=4.0, net_pnl_usdt=20.0, fee_usdt=1.2,
            pos_ref="pos-eth-1", is_profit=True, ts=now - 200,
        )
        db.record_trade_event(
            symbol="ETHUSDT", action="ЗАКРЫТИЕ", side="Sell", strategy="SUPERTREND_PRO",
            price=3150.0, qty=0.2, pnl_pct=5.0, net_pnl_usdt=25.0, fee_usdt=1.3,
            order_id="ord-close-1", pos_ref="pos-eth-1", is_profit=True, ts=now - 50,
        )
        # 2. Убыточная позиция без order_id (сверка по временному окну)
        db.record_trade_event(
            symbol="SOLUSDT", action="ЗАКРЫТИЕ", side="Buy", strategy="REGIME_SWITCHER",
            price=150.0, qty=2.0, pnl_pct=-3.0, net_pnl_usdt=-6.0, fee_usdt=0.5,
            pos_ref="pos-sol-1", is_profit=False, ts=now,
        )
        # 3. Shadow сделка не должна попадать в live-аналитику
        db.record_trade_event(
            symbol="XRPUSDT", action="SHADOW_CLOSE", side="Sell", strategy="BB_RSI",
            price=0.6, qty=100.0, pnl_pct=-99.0, net_pnl_usdt=-500.0,
            pos_ref="sh-xrp-1", is_profit=False, is_shadow=True, ts=now,
        )

        # Сверка get_closed_pnl от Bybit обновляет точные net_pnl_usdt и fee_usdt
        ok1 = db.upsert_closed_pnl_record({
            "symbol": "ETHUSDT", "orderId": "ord-close-1", "side": "Sell",
            "qty": "0.2", "avgEntryPrice": "3000", "avgExitPrice": "3150",
            "closedPnl": "27.4", "openFee": "0.65", "closeFee": "0.65",
            "updatedTime": str(int((now - 50) * 1000)),
        }, margin_used=120.0)
        self.assertTrue(ok1)

        # Сверка ордера без order_id через +-1800s окно
        ok2 = db.upsert_closed_pnl_record({
            "symbol": "SOLUSDT", "orderId": "ord-sol-exch", "side": "Buy",
            "qty": "2.0", "avgEntryPrice": "148", "avgExitPrice": "150",
            "closedPnl": "-5.2", "openFee": "0.2", "closeFee": "0.3",
            "updatedTime": str(int((now + 10) * 1000)),
        }, margin_used=60.0)
        self.assertTrue(ok2)

        # Орфан-закрытие на бирже (если бот не успел записать сам)
        ok3 = db.upsert_closed_pnl_record({
            "symbol": "TONUSDT", "orderId": "ord-ton-orphan", "side": "Sell",
            "qty": "10", "avgEntryPrice": "5.0", "avgExitPrice": "5.5",
            "closedPnl": "4.5", "openFee": "0.1", "closeFee": "0.1",
            "updatedTime": str(int((now + 30) * 1000)),
        }, margin_used=50.0)
        self.assertTrue(ok3)

        # Кривая капитала
        db.record_equity_snapshot(1000.0, balance=1000.0, source="init", ts=int(now - 600))
        db.record_equity_snapshot(1050.0, balance=1050.0, source="trade", ts=int(now - 300))
        db.record_equity_snapshot(987.0, balance=987.0, source="trade", ts=int(now))
        eq_curve = db.get_equity_history(limit=20, is_shadow=False)
        self.assertEqual(len(eq_curve), 3)
        self.assertEqual(eq_curve[-1]["value"], 987.0)

        summary = db.get_analytics_summary(is_shadow=False)
        self.assertEqual(summary["total_trades"], 3)
        self.assertEqual(summary["wins"], 2)
        self.assertEqual(summary["losses"], 1)
        # ETH: 20.0 (TP1) + 27.4 (reconciled close) = 47.4; SOL: -5.2; TON: 4.5 -> 46.7
        self.assertAlmostEqual(summary["total_net_pnl_usdt"], 46.7, places=2)
        self.assertLess(summary["max_drawdown_pct"], 0.0)
        self.assertIn("ETHUSDT", summary["by_symbol"])
        self.assertNotIn("XRPUSDT", summary["by_symbol"])

        # Экспорт CSV/JSON без shadow
        csv_text = db.export_trades_csv(include_shadow=False)
        rows = list(csv.DictReader(io.StringIO(csv_text)))
        self.assertFalse(any(r["symbol"] == "XRPUSDT" for r in rows))
        csv_with_sh = db.export_trades_csv(include_shadow=True)
        rows_sh = list(csv.DictReader(io.StringIO(csv_with_sh)))
        self.assertTrue(any(r["symbol"] == "XRPUSDT" for r in rows_sh))

    def test_bot_reconcile_closed_pnl_and_runtime_persistence_across_restart(self):
        db = BotDatabase(self.db_path)
        b1 = _make_bare_bot(self.tmp.name, db=db)
        b1.start_day_balance = 1234.56
        b1.circuit_breaker_active = True
        b1.portfolio_risk.peak_equity = 1400.0
        b1.portfolio_risk.portfolio_loss_streak = 2
        b1.portfolio_risk.symbol_cooldown_until = {"ETHUSDT": time.time() + 1800}
        b1.risk_states["ETHUSDT"] = PositionRiskState(
            symbol="ETHUSDT", side="Buy", entry_price=3000.0, size=0.1, initial_size=0.1,
            stop_loss=2940.0, take_profit=3150.0, strategy_tag="REGIME_SWITCHER",
        )
        b1.save_state()

        # Эмулируем рестарт нового инстанса бота
        b2 = _make_bare_bot(self.tmp.name, db=db)
        loaded_states = b2.load_state()
        self.assertEqual(list(loaded_states.keys()), ["ETHUSDT"])
        self.assertNotIn("__meta__", loaded_states)
        self.assertEqual(loaded_states["ETHUSDT"].strategy_tag, "REGIME_SWITCHER")

        meta = b2.load_runtime_meta()
        self.assertAlmostEqual(float(meta["start_day_balance"]), 1234.56, places=2)
        self.assertAlmostEqual(float(meta["peak_equity"]), 1400.0, places=2)
        self.assertTrue(bool(meta["circuit_breaker_active"]))
        restored_pr = PortfolioRiskManager.from_dict(meta["portfolio_risk"])
        self.assertEqual(restored_pr.portfolio_loss_streak, 2)
        self.assertIn("ETHUSDT", restored_pr.symbol_cooldown_until)

        # Сверка closed_pnl с Bybit API через bot.reconcile_closed_pnl
        db.record_trade_event(
            symbol="ETHUSDT", action="ЗАКРЫТИЕ", side="Sell",
            price=3100.0, qty=0.1, pnl_pct=10.0, net_pnl_usdt=8.0,
            order_id="close-oid-9", pos_ref="p9", is_profit=True,
        )
        b2.client.get_closed_pnl.return_value = {
            "retCode": 0,
            "result": {
                "list": [{
                    "symbol": "ETHUSDT", "orderId": "close-oid-9", "side": "Sell",
                    "qty": "0.1", "avgEntryPrice": "3000", "avgExitPrice": "3100",
                    "closedPnl": "9.45", "openFee": "0.15", "closeFee": "0.15",
                    "updatedTime": str(int(time.time() * 1000)),
                }]
            },
        }
        synced = b2.reconcile_closed_pnl("ETHUSDT", order_id="close-oid-9", margin_used=60.0)
        self.assertEqual(synced, 1)
        tr = db.get_trades(limit=5)[0]
        self.assertAlmostEqual(tr["net_pnl_usdt"], 9.45, places=2)
        self.assertTrue(tr["closed_pnl_synced"])


class TestV3WebAndAnalytics(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.db_path = Path(self.tmp.name) / "web_test.db"
        self.prof_path = Path(self.tmp.name) / "profiles.json"
        self.tlog_path = Path(self.tmp.name) / "trade.log"
        self.blog_path = Path(self.tmp.name) / "bot.log"
        self.p_db = patch.object(config, "DB_FILE", str(self.db_path))
        self.p_prof = patch.object(config, "PROFILES_FILE", str(self.prof_path))
        self.p_tlog = patch.object(config, "TRADE_LOG_FILE", str(self.tlog_path))
        self.p_blog = patch.object(config, "BOT_LOG_FILE", str(self.blog_path))
        self.p_live = patch.object(config, "LIVE_STATUS_FILE", str(Path(self.tmp.name) / "live_status.json"))
        self.p_ctrl = patch.object(config, "CONTROL_FILE", str(Path(self.tmp.name) / "control.json"))
        for p in (self.p_db, self.p_prof, self.p_tlog, self.p_blog, self.p_live, self.p_ctrl):
            p.start()
            self.addCleanup(p.stop)
        web_state.set_bot_instance(None)
        web_app._invalidate_status_cache()
        self.addCleanup(lambda: web_state.set_bot_instance(None))

    def test_web_sqlite_endpoints_auth_export_and_log_fallback(self):
        import asyncio
        import base64
        from fastapi import HTTPException
        from starlette.requests import HTTPConnection

        self.tlog_path.write_text(
            "2026-01-10 10:00:00 [ETHUSDT] LONG открыт | Вход: 3000.0 | Кол-во: 0.1 (60.0$ маржа) | SL: 2940.0 | TP: 3120.0\n",
            encoding="utf-8",
        )
        self.blog_path.write_text("Bot started OK\n", encoding="utf-8")

        # Пока DB нет — fallback работает из trade.log без создания мусорного DB
        rows_fb = web_state.get_trade_history(limit=10)
        self.assertEqual(len(rows_fb), 1)
        self.assertFalse(self.db_path.exists())

        # Создаем DB с реальными записями
        db = BotDatabase(self.db_path)
        db.record_trade_event(
            symbol="SOLUSDT", action="ЗАКРЫТИЕ", side="Buy", strategy="SUPERTREND_PRO",
            price=180.0, qty=1.0, pnl_pct=6.5, net_pnl_usdt=12.5, fee_usdt=0.4,
            pos_ref="w-sol-1", is_profit=True,
        )
        db.record_equity_snapshot(1012.5, balance=1012.5, source="trade")

        def _make_conn(hdrs: list[tuple[bytes, bytes]]) -> HTTPConnection:
            return HTTPConnection({"type": "http", "method": "GET", "path": "/api/analytics", "headers": hdrs})

        with patch.object(config, "DASHBOARD_USERNAME", "admin"), \
             patch.object(config, "DASHBOARD_PASSWORD", "s3cr3t"):
            # Auth сохранен на уровне FastAPI router и защищает все новые и старые эндпоинты
            app_dep_fns = [getattr(d, "dependency", None) for d in web_app.app.router.dependencies]
            self.assertIn(web_app.verify_dashboard_auth, app_dep_fns)

            with self.assertRaises(HTTPException) as ctx_no:
                web_app.verify_dashboard_auth(_make_conn([]))
            self.assertEqual(ctx_no.exception.status_code, 401)

            bad_tok = base64.b64encode(b"admin:wrong").decode("ascii")
            with self.assertRaises(HTTPException) as ctx_bad:
                web_app.verify_dashboard_auth(_make_conn([(b"authorization", f"Basic {bad_tok}".encode("ascii"))]))
            self.assertEqual(ctx_bad.exception.status_code, 401)

            ok_tok = base64.b64encode(b"admin:s3cr3t").decode("ascii")
            self.assertTrue(web_app.verify_dashboard_auth(_make_conn([(b"authorization", f"Basic {ok_tok}".encode("ascii"))])))

            r_tr = asyncio.run(web_app.get_trades(limit=60, include_shadow=False, only_shadow=False, symbol=None))
            self.assertEqual(r_tr[0]["symbol"], "SOLUSDT")

            r_eq = asyncio.run(web_app.get_equity_history(shadow=False))
            self.assertEqual(r_eq[-1]["value"], 1012.5)

            r_csv = asyncio.run(web_app.export_trades(format="csv", limit=1000, include_shadow=False))
            self.assertIn("text/csv", r_csv.media_type)
            self.assertIn("SOLUSDT", r_csv.body.decode("utf-8"))

            r_json = asyncio.run(web_app.export_trades(format="json", limit=1000, include_shadow=False))
            self.assertEqual(json.loads(r_json.body.decode("utf-8"))[0]["symbol"], "SOLUSDT")

            an_data = asyncio.run(web_app.get_analytics(shadow=False, limit=2000))
            self.assertEqual(an_data["total_trades"], 1)
            self.assertAlmostEqual(an_data["total_net_pnl_usdt"], 12.5, places=2)
            self.assertIn("degradation", an_data)

            # Логи .log по-прежнему доступны
            r_dl = asyncio.run(web_app.download_log(file="trade"))
            self.assertEqual(Path(r_dl.path), self.tlog_path)
            r_vw = asyncio.run(web_app.view_log(file="bot", lines=200))
            self.assertIn("Bot started OK", r_vw)

            # Status с MagicMock не должен падать
            mock_bot = MagicMock()
            mock_bot.get_available_balance.return_value = 500.0
            mock_bot.get_wallet_equity.return_value = 520.0
            mock_bot.positions = {}
            mock_bot.circuit_breaker_active = False
            mock_bot.start_day_balance = 500.0
            web_state.set_bot_instance(mock_bot)
            st_j = asyncio.run(web_app.get_status())
            self.assertIn("portfolio_heat_pct", st_j)
            self.assertIn("peak_drawdown_pct", st_j)

            u_j = asyncio.run(web_app.get_universe())
            self.assertIn("active_symbols", u_j)

        # Проверка шаблона: SRI сохранен и zero innerHTML
        html_text = (Path(__file__).resolve().parent.parent / "web" / "templates" / "index.html").read_text(encoding="utf-8")
        self.assertIn('integrity="sha384-', html_text)
        self.assertNotIn("innerHTML", html_text)


class TestV3DerivativesRegimeAndSwitcher(unittest.TestCase):
    def test_funding_zscore_oi_divergence_and_hybrid_ema_regime(self):
        # Бычий бенчмарк (цена выше EMA и без дампа)
        btc = _make_ohlcv(260, start_price=40000.0, trend=0.002, vol=0.004, seed=7, freq="4h")
        rf = MarketRegimeFilter(
            ema_period=200, fast_ema_period=50, use_hybrid_trend=True,
            funding_z_long_max=2.0, funding_z_short_min=-2.0,
            oi_div_pct=0.03, oi_price_move_pct=0.015,
            extreme_vol_pct=2.50,
        )

        ok, _ = rf.is_long_allowed("ETHUSDT", btc, funding_zscore=0.5, oi_metrics={"oi_change_pct": 0.04, "price_change_pct": 0.02})
        self.assertTrue(ok)

        # Перегретый положительный Funding Z-score >= 2.0 -> блок лонгов
        blk_fz, reason_fz = rf.is_long_allowed("ETHUSDT", btc, funding_zscore=2.4)
        self.assertFalse(blk_fz)
        self.assertIn("Funding Z-score", reason_fz)

        # OI дивергенция: цена растет >= 1.5%, OI падает >= 3% (шорт-сквиз) -> блок лонгов
        blk_oi, reason_oi = rf.is_long_allowed("ETHUSDT", btc, funding_zscore=0.2, oi_metrics={"oi_change_pct": -0.05, "price_change_pct": 0.025})
        self.assertFalse(blk_oi)
        self.assertIn("OI", reason_oi)

        # Медвежий бенчмарк для проверки шортов
        btc_bear = _make_ohlcv(260, start_price=80000.0, trend=-0.002, vol=0.004, seed=8, freq="4h")
        ok_s, _ = rf.is_short_allowed("ETHUSDT", btc_bear, funding_zscore=-0.3)
        self.assertTrue(ok_s)

        blk_sfz, _ = rf.is_short_allowed("ETHUSDT", btc_bear, funding_zscore=-2.3)
        self.assertFalse(blk_sfz)

        # Вынос лонгов: цена падает при падении OI -> блок шортов
        blk_soi, _ = rf.is_short_allowed("ETHUSDT", btc_bear, oi_metrics={"oi_change_pct": -0.06, "price_change_pct": -0.03})
        self.assertFalse(blk_soi)

        # Гибридная быстрая EMA: после затяжного дампа цена пробивает вверх EMA(50) с растущим наклоном, оставаясь < EMA(200)
        n = 240
        closes = np.concatenate([
            np.linspace(60000.0, 58000.0, 120),
            np.linspace(58000.0, 42000.0, 90),
            np.linspace(42000.0, 49500.0, 30),
        ])
        ts = pd.date_range("2025-01-01", periods=n, freq="4h")
        v_df = pd.DataFrame({
            "timestamp": ts,
            "open": closes * 0.998,
            "high": closes * 1.003,
            "low": closes * 0.997,
            "close": closes,
            "volume": np.full(n, 1000.0),
        })
        # Убеждаемся, что на закрытой свече [-2] цена < EMA200, но > EMA50 с восходящим наклоном
        ema200 = v_df["close"].ewm(span=200, adjust=False).mean()
        ema50 = v_df["close"].ewm(span=50, adjust=False).mean()
        self.assertLess(v_df["close"].iloc[-2], ema200.iloc[-2])
        self.assertGreater(v_df["close"].iloc[-2], ema50.iloc[-2])

        ok_recov, _ = rf.is_long_allowed("ETHUSDT", v_df)
        self.assertTrue(ok_recov)

        rf_slow_only = MarketRegimeFilter(ema_period=200, fast_ema_period=50, use_hybrid_trend=False, rsi_threshold=25.0)
        blk_slow, _ = rf_slow_only.is_long_allowed("ETHUSDT", v_df)
        self.assertFalse(blk_slow)

        # Векторный расчёт add_indicators с funding_df и oi_df (без lookahead)
        fdf = pd.DataFrame({
            "timestamp": pd.date_range("2025-01-01", periods=100, freq="8h"),
            "fundingRate": np.concatenate([np.full(90, 0.0001), np.full(10, 0.005)]),
        })
        odf = pd.DataFrame({
            "timestamp": ts,
            "openInterest": np.linspace(1e6, 1.2e6, n),
        })
        ind_btc = MarketRegimeFilter.add_indicators(btc, funding_df=fdf, oi_df=odf)
        self.assertIn("BENCH_FUNDING_Z", ind_btc.columns)
        self.assertIn("BENCH_OI_CHG", ind_btc.columns)
        self.assertIn("BENCH_FAST_EMA", ind_btc.columns)

    def test_regime_switcher_strategy_and_mr_exits(self):
        strat = get_strategy("REGIME_SWITCHER")
        self.assertIsInstance(strat, RegimeSwitcherStrategy)
        strat_alias = get_strategy("ADAPTIVE")
        self.assertIsInstance(strat_alias, RegimeSwitcherStrategy)

        # Четкий направленный тренд -> TREND
        trend_df = _make_ohlcv(220, start_price=100.0, trend=0.006, vol=0.003, seed=11)
        out_t = strat.calculate_indicators(trend_df)
        self.assertIn("REGIME_MODE", out_t.columns)
        self.assertIn("SIGNAL_REGIME", out_t.columns)
        self.assertIn("TREND", set(out_t["REGIME_MODE"].tail(40)))

        # Боковой стационарный флэт (возврат к среднему без тренда) -> RANGE и MR-колонки
        n = 220
        ts = pd.date_range("2025-01-01", periods=n, freq="1h")
        rng_r = np.random.default_rng(123)
        chop = np.empty(n, dtype=float)
        chop[0] = 100.0
        for idx_c in range(1, n):
            chop[idx_c] = 100.0 - 0.65 * (chop[idx_c - 1] - 100.0) + rng_r.normal(0.0, 0.35)
        range_df = pd.DataFrame({
            "timestamp": ts,
            "open": chop,
            "high": chop + 0.3,
            "low": chop - 0.3,
            "close": chop,
            "volume": np.full(n, 1000.0),
        })
        out_r = strat.calculate_indicators(range_df)
        self.assertIn("mr_exit_long", out_r.columns)
        self.assertIn("mr_exit_short", out_r.columns)
        self.assertIn("RANGE", set(out_r["REGIME_MODE"].tail(60)))

        # Стопы разные в RANGE vs TREND
        s_tr = strat.calculate_stops(100.0, "Buy", atr_value=2.0, atr_sl_mult=2.2, atr_tp_mult=4.5, regime="TREND")
        s_mr = strat.calculate_stops(100.0, "Buy", atr_value=2.0, atr_sl_mult=2.2, atr_tp_mult=4.5, regime="RANGE")
        self.assertNotAlmostEqual(s_tr["tp"], s_mr["tp"], places=2)
        self.assertTrue(s_mr["type"].startswith("MR-"))


class TestV3PortfolioRiskAndTimeStop(unittest.TestCase):
    def test_time_stop_stages(self):
        now = 1_750_000_000.0
        st = PositionRiskState(
            symbol="ETHUSDT", side="Buy", entry_price=100.0, size=1.0, initial_size=1.0,
            stop_loss=96.0, take_profit=112.0, atr_value=2.0, opened_at=now - 31 * 3600,
        )
        # При нормальной выдержке до 30ч — none
        act0, _, _ = st.check_time_stop(current_price=100.2, now_ts=now - 5 * 3600, max_hours=30.0, hard_hours=36.0)
        self.assertEqual(act0, "none")

        # После 31ч при малом позитивном прогрессе (0.15R: price=100.6 при R=4 -> 0.15R in (0.05..0.3)) -> tighten_be
        act1, be_sl, _ = st.check_time_stop(current_price=100.6, now_ts=now, max_hours=30.0, hard_hours=36.0, min_r_progress=0.3)
        self.assertEqual(act1, "tighten_be")
        self.assertGreater(be_sl, 100.0)

        # Если цена застряла около нуля/минуса <= 0.05R после max_hours -> close
        act2, _, _ = st.check_time_stop(current_price=99.8, now_ts=now, max_hours=30.0, hard_hours=36.0, min_r_progress=0.3)
        self.assertEqual(act2, "close")

        # Если >= hard_hours и прогресс < min_r_progress -> close
        st.time_stop_tightened = True
        act3, _, _ = st.check_time_stop(current_price=100.8, now_ts=now + 6 * 3600, max_hours=30.0, hard_hours=36.0, min_r_progress=0.3)
        self.assertEqual(act3, "close")

        # После TP1 — не мешаем победителю
        st.tp1_executed = True
        act4, _, _ = st.check_time_stop(current_price=100.1, now_ts=now + 10 * 3600, max_hours=30.0, hard_hours=36.0)
        self.assertEqual(act4, "none")

    def test_portfolio_heat_cap_corr_throttle_and_cooldown(self):
        pr = PortfolioRiskManager(
            max_heat_pct=0.035,
            use_correlation_scaling=True,
            corr_lookback_bars=60,
            corr_threshold=0.75,
            corr_risk_scale=0.6,
            use_peak_dd_throttle=True,
            dd_tier1_pct=0.06,
            dd_tier1_mult=0.5,
            dd_tier2_pct=0.12,
            dd_recover_pct=0.09,
            use_loss_cooldown=True,
            symbol_loss_limit=2,
            portfolio_loss_limit=3,
            cooldown_seconds=3600.0,
            peak_equity=1000.0,
        )

        # 1. Heat Cap: уже открыто риска 28$ (2.8% от 1000$), хотим рискнуть 15$ -> урежется до 7$ (>= 35% от 15$)
        r1 = PositionRiskState(symbol="ETHUSDT", side="Buy", entry_price=100.0, size=14.0, initial_size=14.0, stop_loss=98.0, take_profit=106.0)
        ok_h, adj_h, note_h = pr.check_entry_budget(
            "SOLUSDT", True, proposed_risk_usdt=15.0, equity=1000.0,
            risk_states={"ETHUSDT": r1}, open_positions={"ETHUSDT": {"position": "long"}},
        )
        self.assertTrue(ok_h)
        self.assertAlmostEqual(adj_h, 7.0, places=2)
        self.assertIn("Heat Cap", note_h)

        # Если открытый риск 30$, остаток 5$ < 35% от 15$ (5.25$) -> блокировка (анти-пылевой порог)
        r_dust = PositionRiskState(symbol="ETHUSDT", side="Buy", entry_price=100.0, size=15.0, initial_size=15.0, stop_loss=98.0, take_profit=106.0)
        ok_dust, _, _ = pr.check_entry_budget(
            "SOLUSDT", True, proposed_risk_usdt=15.0, equity=1000.0,
            risk_states={"ETHUSDT": r_dust}, open_positions={"ETHUSDT": {"position": "long"}},
        )
        self.assertFalse(ok_dust)

        # Если открытая позиция уже в безубытке -> ее открытый риск 0$, полный бюджет доступен
        r1_be = PositionRiskState(symbol="ETHUSDT", side="Buy", entry_price=100.0, size=14.0, initial_size=14.0, stop_loss=100.5, take_profit=106.0, breakeven_applied=True)
        self.assertEqual(r1_be.open_risk_usdt(), 0.0)

        # Полное исчерпание Heat Cap -> отклонение входа
        r_full = PositionRiskState(symbol="ETHUSDT", side="Buy", entry_price=100.0, size=18.0, initial_size=18.0, stop_loss=98.0, take_profit=106.0)  # 36$ > 35$
        ok_blk, _, _ = pr.check_entry_budget(
            "SOLUSDT", True, proposed_risk_usdt=10.0, equity=1000.0,
            risk_states={"ETHUSDT": r_full}, open_positions={"ETHUSDT": {"position": "long"}},
        )
        self.assertFalse(ok_blk)

        # 2. Корреляционный скейлинг (идентичные доходности >= 0.75 по модулю в том же направлении)
        df_base = _make_ohlcv(100, start_price=100.0, trend=0.001, vol=0.015, seed=3)
        df_corr = df_base.copy()
        df_corr["close"] = df_corr["close"] * 0.5
        ok_c, adj_c, note_c = pr.check_entry_budget(
            "SOLUSDT", True, proposed_risk_usdt=10.0, equity=1000.0,
            risk_states={}, open_positions={"ETHUSDT": {"position": "long"}},
            klines_map={"ETHUSDT": df_base, "SOLUSDT": df_corr},
        )
        self.assertTrue(ok_c)
        self.assertAlmostEqual(adj_c, 6.0, places=2)
        self.assertIn("Corr", note_c)

        # Противоположное направление (Short против Long) НЕ режется корреляцией
        ok_opp, adj_opp, _ = pr.check_entry_budget(
            "SOLUSDT", False, proposed_risk_usdt=10.0, equity=1000.0,
            risk_states={}, open_positions={"ETHUSDT": {"position": "long"}},
            klines_map={"ETHUSDT": df_base, "SOLUSDT": df_corr},
        )
        self.assertTrue(ok_opp)
        self.assertAlmostEqual(adj_opp, 10.0, places=2)

        # 3. HWM Peak Drawdown Throttle + гистерезис восстановления
        dd1 = pr.update_equity(930.0)  # -7% >= 6%
        self.assertEqual(dd1["risk_scale"], 0.5)
        self.assertFalse(dd1["drawdown_pause"])

        dd2 = pr.update_equity(870.0)  # -13% >= 12%
        self.assertTrue(dd2["drawdown_pause"])
        self.assertEqual(dd2["risk_scale"], 0.0)

        dd_hold = pr.update_equity(900.0)  # -10% > 9% recover threshold -> остается на паузе!
        self.assertTrue(dd_hold["drawdown_pause"])

        dd_rec = pr.update_equity(920.0)  # -8% <= 9% recover -> выходит из паузы (и в tier1 0.5x т.к. 8% >= 6%)
        self.assertFalse(dd_rec["drawdown_pause"])
        self.assertEqual(dd_rec["risk_scale"], 0.5)

        # 4. Loss-Streak Cooldown (2 стопа по паре, 3 по портфелю) + реальная блокировка входа + сброс выигрышем
        t0 = 1_750_000_000.0
        pr.record_closed_trade("ETHUSDT", -10.0, now_ts=t0)
        self.assertNotIn("ETHUSDT", pr.active_cooldowns(now_ts=t0 + 10)["symbol_cooldowns_min"])
        pr.record_closed_trade("ETHUSDT", -10.0, now_ts=t0 + 20)
        self.assertIn("ETHUSDT", pr.active_cooldowns(now_ts=t0 + 30)["symbol_cooldowns_min"])
        ok_cd_sym, _, _ = pr.check_entry_budget("ETHUSDT", True, 10.0, 1000.0, {}, {}, now_ts=t0 + 30)
        self.assertFalse(ok_cd_sym)

        pr.record_closed_trade("SOLUSDT", -10.0, now_ts=t0 + 40)
        self.assertGreater(pr.active_cooldowns(now_ts=t0 + 50)["portfolio_cooldown_min"], 0.0)
        ok_cd_port, _, _ = pr.check_entry_budget("AVAXUSDT", True, 10.0, 1000.0, {}, {}, now_ts=t0 + 50)
        self.assertFalse(ok_cd_port)

        pr.record_closed_trade("SOLUSDT", +25.0, now_ts=t0 + 60)
        self.assertEqual(pr.portfolio_loss_streak, 0)
        self.assertEqual(pr.symbol_loss_streaks.get("SOLUSDT"), 0)


class TestV3ExecutionWsRateLimitAndUniverse(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.p_db = patch.object(config, "DB_FILE", str(Path(self.tmp.name) / "exec_test.db"))
        self.p_prof = patch.object(config, "PROFILES_FILE", str(Path(self.tmp.name) / "profiles.json"))
        self.p_tlog = patch.object(config, "TRADE_LOG_FILE", str(Path(self.tmp.name) / "trade.log"))
        self.p_blog = patch.object(config, "BOT_LOG_FILE", str(Path(self.tmp.name) / "bot.log"))
        self.p_ctrl = patch.object(config, "CONTROL_FILE", str(Path(self.tmp.name) / "control.json"))
        self.p_state = patch.object(config, "STATE_FILE", str(Path(self.tmp.name) / "bot_state.json"))
        self.p_live = patch.object(config, "LIVE_STATUS_FILE", str(Path(self.tmp.name) / "live_status.json"))
        for p in (self.p_db, self.p_prof, self.p_tlog, self.p_blog, self.p_ctrl, self.p_state, self.p_live):
            p.start()
            self.addCleanup(p.stop)
        web_state.set_bot_instance(None)
        web_state.set_bot_paused(False, persist=False)
        self.addCleanup(lambda: (web_state.set_bot_instance(None), web_state.set_bot_paused(False, persist=False)))

    def test_rate_limiter_ws_kline_event_and_zero_rest_cache(self):
        rl = RateLimitController(reserve_threshold=8)
        now_ms = int(time.time() * 1000)
        rl.update_from_headers({
            "X-Bapi-Limit": "120",
            "x-bapi-limit-status": "5",
            "X-Bapi-Limit-Reset-Timestamp": str(now_ms + 300),
        })
        snap = rl.snapshot()
        self.assertEqual(snap["remaining"], 5)
        self.assertEqual(snap["limit"], 120)
        with patch.object(bot_mod.time, "sleep") as m_sleep:
            rl.wait_if_needed()
            self.assertTrue(m_sleep.called)

        b = _make_bare_bot(self.tmp.name)
        b.ws = MagicMock()
        b.ws.exited = False
        b.ws.is_connected.return_value = True

        # Засеваем кэш 200 свечами
        tf = str(config.INTERVAL)
        seed_df = _make_ohlcv(200, start_price=100.0)
        b.kline_cache[("ETHUSDT", tf)] = seed_df.copy()
        b.kline_cache_ts[("ETHUSDT", tf)] = time.time()

        # Устаревшее поколение WS игнорируется
        last_ms = int(seed_df["timestamp"].iloc[-1].timestamp() * 1000) + 3_600_000
        b._handle_ws_kline({
            "topic": f"kline.{tf}.ETHUSDT",
            "data": [{"start": last_ms, "open": "105", "high": "106", "low": "104", "close": "105.5", "volume": "500", "confirm": True}],
        }, ws_gen=999)
        self.assertFalse(b._candle_event.is_set())

        # Незакрытая свеча обновляет кэш, но НЕ будит event-loop
        b._handle_ws_kline({
            "topic": f"kline.{tf}.ETHUSDT",
            "data": [{"start": last_ms, "open": "105", "high": "106", "low": "104", "close": "105.5", "volume": "500", "confirm": False}],
        }, ws_gen=1)
        self.assertFalse(b._candle_event.is_set())
        self.assertEqual(len(b.kline_cache[("ETHUSDT", tf)]), 201)

        # Подтвержденное закрытие свечи будит _candle_event и обслуживается 0 REST-запросами
        b._handle_ws_kline({
            "topic": f"kline.{tf}.ETHUSDT",
            "data": [{"start": last_ms, "open": "105", "high": "107", "low": "104", "close": "106.5", "volume": "900", "confirm": True}],
        }, ws_gen=1)
        self.assertTrue(b._candle_event.is_set())
        res_df = b.get_klines_fast("ETHUSDT", interval=tf, limit=150)
        self.assertEqual(len(res_df), 150)
        self.assertAlmostEqual(float(res_df["close"].iloc[-1]), 106.5, places=2)
        b.client.get_kline.assert_not_called()

    def test_spread_guard_maker_chase_and_daily_instrument_refresh(self):
        b = _make_bare_bot(self.tmp.name)
        b.get_available_balance = MagicMock(return_value=1000.0)
        b.get_wallet_equity = MagicMock(return_value=1000.0)

        # 1. Аномальный начальный спред отклоняет вход без отправки ордера
        b.client.get_tickers.return_value = {
            "retCode": 0,
            "result": {"list": [{"bid1Price": "3000.00", "ask1Price": "3040.00"}]},  # 1.33% > 0.5%
        }
        with patch.object(bot_mod, "USE_SMART_EXECUTION", True), \
             patch.object(config, "MAX_ENTRY_SPREAD_PCT", 0.005):
            with b.lock:
                res_wide = b.open_position("ETHUSDT", "Buy", 3000.0, atr_value=20.0)
        self.assertIsNone(res_wide)
        b.client.place_order.assert_not_called()

        # 2. Maker Chase: переставляет цену через amend_order вслед за bid1 и исполняется как Maker
        b2 = _make_bare_bot(self.tmp.name)
        b2.get_available_balance = MagicMock(return_value=1000.0)
        b2.get_wallet_equity = MagicMock(return_value=1000.0)
        b2.client.get_tickers.side_effect = [
            {"retCode": 0, "result": {"list": [{"bid1Price": "3000.00", "ask1Price": "3001.00"}]}},
            {"retCode": 0, "result": {"list": [{"bid1Price": "3002.00", "ask1Price": "3003.00"}]}},
        ]
        b2.client.place_order.return_value = {"retCode": 0, "result": {"orderId": "mk-1"}}
        b2.client.amend_order.return_value = {"retCode": 0, "result": {"orderId": "mk-1"}}
        b2.client.get_order_history.side_effect = [
            {"retCode": 0, "result": {"list": [{"orderId": "mk-1", "orderStatus": "New", "cumExecQty": "0"}]}},
            {"retCode": 0, "result": {"list": [{"orderId": "mk-1", "orderStatus": "Filled", "cumExecQty": "0.10", "avgPrice": "3002.0"}]}},
        ]
        b2.client.get_open_orders.return_value = {"retCode": 0, "result": {"list": [{"orderId": "mk-1"}]}}

        with patch.object(bot_mod, "USE_SMART_EXECUTION", True), \
             patch.object(bot_mod, "SMART_EXECUTION_TIMEOUT", 4.0), \
             patch.object(config, "USE_MAKER_CHASE", True), \
             patch.object(config, "MAKER_CHASE_INTERVAL_SEC", 0.0), \
             patch.object(config, "MAX_MAKER_CHASE_ATTEMPTS", 2), \
             patch.object(config, "MAX_MAKER_CHASE_PCT", 0.004), \
             patch.object(bot_mod.time, "sleep"):
            with b2.lock:
                ok_mk = b2.open_position("ETHUSDT", "Buy", 3000.0, atr_value=30.0)

        self.assertIsNotNone(ok_mk)
        b2.client.amend_order.assert_called_once()
        self.assertEqual(b2.client.amend_order.call_args.kwargs["price"], "3002.00")

        # 2b. Расширение спреда после таймаута Maker отменяет Market-фоллбэк
        b2b = _make_bare_bot(self.tmp.name)
        b2b.get_available_balance = MagicMock(return_value=1000.0)
        b2b.get_wallet_equity = MagicMock(return_value=1000.0)
        b2b.client.get_tickers.side_effect = [
            {"retCode": 0, "result": {"list": [{"bid1Price": "3000.00", "ask1Price": "3001.00"}]}},  # OK на входе Maker
            {"retCode": 0, "result": {"list": [{"bid1Price": "3000.00", "ask1Price": "3040.00"}]}},  # широкий спред перед Market
        ]
        b2b.client.place_order.return_value = {"retCode": 0, "result": {"orderId": "mk-timeout"}}
        b2b.client.cancel_order.return_value = {"retCode": 0}
        b2b.client.get_open_orders.return_value = {"retCode": 0, "result": {"list": [{"orderId": "mk-timeout"}]}}
        b2b.client.get_order_history.return_value = {"retCode": 0, "result": {"list": [{"orderId": "mk-timeout", "orderStatus": "Cancelled", "cumExecQty": "0"}]}}
        b2b.client.get_positions.return_value = {"retCode": 0, "result": {"list": []}}
        with patch.object(bot_mod, "USE_SMART_EXECUTION", True), \
             patch.object(bot_mod, "SMART_EXECUTION_TIMEOUT", 0.05), \
             patch.object(config, "MAX_ENTRY_SPREAD_PCT", 0.005), \
             patch.object(bot_mod.time, "sleep"):
            with b2b.lock:
                res_fb_blk = b2b.open_position("ETHUSDT", "Buy", 3000.0, atr_value=20.0)
        self.assertIsNone(res_fb_blk)
        mkt_calls = [c for c in b2b.client.place_order.call_args_list if c.kwargs.get("orderType") == "Market"]
        self.assertEqual(len(mkt_calls), 0)

        # 3. Суточное обновление кэша инструментов и ограничение плеча maxLeverage
        b3 = _make_bare_bot(self.tmp.name)
        b3.instruments["ETHUSDT"]["last_loaded_ts"] = time.time() - 90_000.0
        b3.client.get_instruments_info.return_value = {
            "retCode": 0,
            "result": {"list": [{
                "lotSizeFilter": {"qtyStep": "0.001", "minOrderQty": "0.01", "minNotionalValue": "5"},
                "priceFilter": {"tickSize": "0.05"},
                "leverageFilter": {"maxLeverage": "3"},
            }]},
        }
        b3.refresh_instruments_if_needed()
        self.assertEqual(b3.instruments["ETHUSDT"]["tick_size"], 0.05)
        self.assertEqual(b3.instruments["ETHUSDT"]["max_leverage"], 3.0)
        b3._init_leverage("ETHUSDT")
        self.assertEqual(b3.client.set_leverage.call_args.kwargs["buyLeverage"], "3")

    def test_dynamic_universe_selector(self):
        sel = DynamicUniverseSelector(top_n=2, min_turnover_24h=50_000_000, max_spread_pct=0.003, min_vol_pct=0.015, max_vol_pct=0.18, max_abs_funding=0.002)
        tickers = [
            # Отличный кандидат
            {"symbol": "ETHUSDT", "lastPrice": "3000", "bid1Price": "2999.5", "ask1Price": "3000.5", "turnover24h": "900000000", "highPrice24h": "3120", "lowPrice24h": "2970", "fundingRate": "0.0001"},
            # Отличный кандидат #2
            {"symbol": "SOLUSDT", "lastPrice": "150", "bid1Price": "149.95", "ask1Price": "150.05", "turnover24h": "400000000", "highPrice24h": "158", "lowPrice24h": "148", "fundingRate": "0.0002"},
            # Отклоняется: низкий оборот
            {"symbol": "LOWVOLUSDT", "lastPrice": "10", "bid1Price": "9.99", "ask1Price": "10.01", "turnover24h": "5000000", "highPrice24h": "10.5", "lowPrice24h": "9.9", "fundingRate": "0.0001"},
            # Отклоняется: широкий спред
            {"symbol": "WIDESPREADUSDT", "lastPrice": "10", "bid1Price": "9.90", "ask1Price": "10.10", "turnover24h": "200000000", "highPrice24h": "10.5", "lowPrice24h": "9.8", "fundingRate": "0.0001"},
            # Отклоняется: экстремальный фандинг
            {"symbol": "EXTFUNUSDT", "lastPrice": "10", "bid1Price": "9.99", "ask1Price": "10.01", "turnover24h": "200000000", "highPrice24h": "10.5", "lowPrice24h": "9.9", "fundingRate": "0.005"},
        ]
        mock_client = MagicMock()
        mock_client.get_tickers.return_value = {"retCode": 0, "result": {"list": tickers}}
        picked = sel.select_symbols(mock_client, fallback_symbols=["XRPUSDT"], pinned_symbols=["AVAXUSDT"])
        self.assertEqual(picked[:2], ["ETHUSDT", "SOLUSDT"])
        self.assertIn("AVAXUSDT", picked)
        self.assertNotIn("LOWVOLUSDT", picked)
        self.assertNotIn("WIDESPREADUSDT", picked)

        # Ошибка API -> возврат fallback_symbols
        mock_client.get_tickers.side_effect = RuntimeError("timeout")
        fb = sel.select_symbols(mock_client, fallback_symbols=["ETHUSDT", "SOLUSDT"])
        self.assertEqual(fb, ["ETHUSDT", "SOLUSDT"])


class TestV3ShadowPortfolioOptimizerMCAndDSR(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.db_path = Path(self.tmp.name) / "opt_test.db"
        self.prof_path = Path(self.tmp.name) / "strategy_profiles.json"
        self.p_db = patch.object(config, "DB_FILE", str(self.db_path))
        self.p_prof = patch.object(config, "PROFILES_FILE", str(self.prof_path))
        self.p_tlog = patch.object(config, "TRADE_LOG_FILE", str(Path(self.tmp.name) / "trade.log"))
        self.p_blog = patch.object(config, "BOT_LOG_FILE", str(Path(self.tmp.name) / "bot.log"))
        self.p_ctrl = patch.object(config, "CONTROL_FILE", str(Path(self.tmp.name) / "control.json"))
        self.p_live = patch.object(config, "LIVE_STATUS_FILE", str(Path(self.tmp.name) / "live_status.json"))
        for p in (self.p_db, self.p_prof, self.p_tlog, self.p_blog, self.p_ctrl, self.p_live):
            p.start()
            self.addCleanup(p.stop)
        web_state.set_bot_instance(None)
        web_state.set_bot_paused(False, persist=False)
        self.addCleanup(lambda: (web_state.set_bot_instance(None), web_state.set_bot_paused(False, persist=False)))

    def test_shadow_engine_simulation_without_exchange_orders(self):
        db = BotDatabase(self.db_path)
        sh = ShadowEngine(db=db, initial_equity=1000.0, strategy_name="SUPERTREND_PRO", leverage=5)

        # Широкий спред пропускается
        blocked = sh.open_position("ETHUSDT", "Buy", price=3000.0, atr_value=20.0, bid=3000.0, ask=3030.0)
        self.assertIsNone(blocked)

        # Вход без отправки реальных ордеров
        pos = sh.open_position("ETHUSDT", "Buy", price=3000.0, atr_value=20.0, bid=2999.5, ask=3000.5)
        self.assertIsNotNone(pos)
        self.assertEqual(len(sh.positions), 1)
        st = sh.risk_states["ETHUSDT"]
        tp1 = st.tp1_price
        tp2 = st.take_profit

        # Тик до TP1 -> частичный фикс + перенос SL в безубыток
        sh.on_tick("ETHUSDT", price=tp1 + 1.0, high=tp1 + 1.0, low=3000.0)
        self.assertTrue(sh.risk_states["ETHUSDT"].tp1_executed)
        self.assertTrue(sh.risk_states["ETHUSDT"].breakeven_applied)

        # Тик до TP2 -> полное закрытие + запись shadow=True в SQLite
        sh.on_tick("ETHUSDT", price=tp2 + 1.0, high=tp2 + 1.0, low=tp1)
        self.assertEqual(len(sh.positions), 0)
        self.assertGreater(sh.equity, 1000.0)

        sh_trades = db.get_trades(only_shadow=True)
        self.assertGreaterEqual(len(sh_trades), 2)
        self.assertEqual(db.get_trades(include_shadow=False), [])

        # Проверка SHADOW_MODE перехвата на уровне BybitMultiFuturesBot: ордера на биржу НЕ уходят
        b_sh = _make_bare_bot(self.tmp.name, db=db)
        b_sh.shadow_engine = sh
        b_sh.client.get_tickers.return_value = {"retCode": 0, "result": {"list": [{"bid1Price": "149.95", "ask1Price": "150.05"}]}}
        with patch.object(config, "SHADOW_MODE", True):
            with b_sh.lock:
                sh_res = b_sh.open_position("SOLUSDT", "Buy", price=150.0, atr_value=2.0)
        self.assertIsNotNone(sh_res)
        b_sh.client.place_order.assert_not_called()

    def test_monte_carlo_dsr_profiles_and_portfolio_wfa(self):
        # 1. Deflated Sharpe Ratio
        dsr_strong = opt_mod.compute_deflated_sharpe_ratio(observed_sharpe=3.5, n_trials=5, n_obs=180)
        dsr_weak = opt_mod.compute_deflated_sharpe_ratio(observed_sharpe=0.3, n_trials=80, n_obs=60)
        self.assertGreater(dsr_strong["dsr"], 0.90)
        self.assertLess(dsr_weak["dsr"], 0.50)

        # 2. Monte Carlo Bootstrap (500 симуляций)
        tdf_sample = pd.DataFrame({
            "pos_id": [1, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            "usd_delta": [25.0, 35.0, -20.0, 40.0, -15.0, 30.0, 45.0, -25.0, 50.0, 20.0, 35.0],
        })
        mc1 = opt_mod.run_monte_carlo_bootstrap(tdf_sample, n_sims=500, initial_capital=1000.0, seed=42)
        mc2 = opt_mod.run_monte_carlo_bootstrap(tdf_sample, n_sims=500, initial_capital=1000.0, seed=42)
        self.assertEqual(mc1, mc2)
        self.assertLessEqual(mc1["p5_return_pct"], mc1["median_return_pct"])
        self.assertLessEqual(mc1["median_return_pct"], mc1["p95_return_pct"])
        self.assertLessEqual(mc1["worst_5pct_max_dd_pct"], mc1["median_max_dd_pct"])
        self.assertGreater(mc1["prob_profit_pct"], 70.0)

        # 3. Portfolio Rolling WFA на синтетических данных без сети
        df_a = _make_ohlcv(520, start_price=100.0, trend=0.0012, vol=0.012, seed=21)
        df_b = _make_ohlcv(520, start_price=50.0, trend=0.0010, vol=0.014, seed=22)
        base_dfs = {"ETHUSDT": df_a, "SOLUSDT": df_b}
        res_wfa = opt_mod.run_portfolio_rolling_wfa(
            symbols=["ETHUSDT", "SOLUSDT"],
            trials=4,
            n_windows=2,
            strategy_name="SUPERTREND_PRO",
            initial_capital=1000.0,
            mc_sims=200,
            base_dfs_override=base_dfs,
            funding_map_override={"ETHUSDT": None, "SOLUSDT": None},
        )
        self.assertEqual(len(res_wfa["windows"]), 2)
        self.assertIsInstance(res_wfa["best_params"], dict)
        self.assertIn("dsr", res_wfa)

        # 4. Версионирование профиля и детектор live-деградации
        p1 = opt_mod.save_strategy_profile(
            "SUPERTREND_PRO", res_wfa["best_params"],
            oos_metrics={"trades": 25, "win_rate": 56.0, "profit_factor": 1.8, "max_drawdown": -7.5, "sharpe_ratio": 1.9},
            dsr_result=res_wfa["dsr"], mc_result=res_wfa["monte_carlo"], symbols=["ETHUSDT", "SOLUSDT"],
        )
        self.assertEqual(p1["version"], 1)
        p2 = opt_mod.save_strategy_profile(
            "SUPERTREND_PRO", res_wfa["best_params"],
            oos_metrics={"trades": 30, "win_rate": 58.0, "profit_factor": 1.9, "max_drawdown": -7.0, "sharpe_ratio": 2.0},
        )
        self.assertEqual(p2["version"], 2)
        loaded_prof = opt_mod.load_strategy_profile("SUPERTREND_PRO")
        self.assertEqual(loaded_prof["version"], 2)

        deg_few = opt_mod.detect_live_degradation({"total_trades": 5, "win_rate": 20.0, "profit_factor": 0.4}, loaded_prof)
        self.assertEqual(deg_few["status"], "INSUFFICIENT_DATA")

        deg_ok = opt_mod.detect_live_degradation({"total_trades": 20, "win_rate": 55.0, "profit_factor": 1.7, "max_drawdown_pct": -5.0}, loaded_prof)
        self.assertEqual(deg_ok["status"], "OK")
        self.assertFalse(deg_ok["degraded"])

        deg_bad = opt_mod.detect_live_degradation({"total_trades": 20, "win_rate": 32.0, "profit_factor": 0.65, "max_drawdown_pct": -18.0}, loaded_prof)
        self.assertEqual(deg_bad["status"], "DEGRADED")
        self.assertTrue(deg_bad["degraded"])
        self.assertGreaterEqual(len(deg_bad["reasons"]), 2)


class TestV3AuditRegressionFixes(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.db_path = Path(self.tmp.name) / "bot_audit.db"
        self.p_db = patch.object(config, "DB_FILE", str(self.db_path))
        self.p_prof = patch.object(config, "PROFILES_FILE", str(Path(self.tmp.name) / "strategy_profiles.json"))
        self.p_state = patch.object(config, "STATE_FILE", str(Path(self.tmp.name) / "bot_state.json"))
        self.p_ctrl = patch.object(config, "CONTROL_FILE", str(Path(self.tmp.name) / "control.json"))
        self.p_live = patch.object(config, "LIVE_STATUS_FILE", str(Path(self.tmp.name) / "live_status.json"))
        self.p_tlog = patch.object(config, "TRADE_LOG_FILE", str(Path(self.tmp.name) / "trade.log"))
        self.p_blog = patch.object(config, "BOT_LOG_FILE", str(Path(self.tmp.name) / "bot.log"))
        for p in (self.p_db, self.p_prof, self.p_state, self.p_ctrl, self.p_live, self.p_tlog, self.p_blog):
            p.start()
            self.addCleanup(p.stop)

    def test_analytics_recent_2000_window_orphan_filter_and_tp1_sibling(self):
        db = BotDatabase(self.db_path)
        now = 1_760_000_000.0
        # 2000 старых убыточных закрытий + 50 свежих прибыльных (total = 2050)
        bulk_rows = [
            (
                now + i, "2025-10-01 00:00:00", "ETHUSDT", "ЗАКРЫТИЕ", "Sell", "SUPERTREND_PRO",
                3000.0, 0.1, -10.0, -10.0, 0.0, 0.0,
                f"old-{i}", "", f"pos-old-{i}", 0, 0, "", 1,
            )
            for i in range(2000)
        ] + [
            (
                now + 5000 + i, "2025-10-02 00:00:00", "ETHUSDT", "ЗАКРЫТИЕ", "Sell", "SUPERTREND_PRO",
                3100.0, 0.1, 25.0, 25.0, 0.0, 0.0,
                f"new-{i}", "", f"pos-new-{i}", 1, 0, "", 1,
            )
            for i in range(50)
        ]
        with db._connect() as conn:
            conn.executemany(
                """
                INSERT INTO trades (
                    ts, timestamp_str, symbol, action, side, strategy,
                    price, qty, pnl_pct, net_pnl_usdt, fee_usdt, funding_usdt,
                    order_id, exec_type, pos_ref, is_profit, is_shadow, details,
                    closed_pnl_synced
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                bulk_rows,
            )
            conn.commit()

        summary = db.get_analytics_summary(limit=2000)
        self.assertEqual(summary["total_trades"], 2000)
        self.assertEqual(summary["wins"], 50)
        self.assertGreater(summary["win_rate"], 0.0)

        # Проверка защиты от импорта доботовой истории и перезаписи суммарного pnl_pct при TP1 sibling
        res_skip = db.upsert_closed_pnl_record(
            {
                "orderId": "prebot-1",
                "symbol": "SOLUSDT",
                "side": "Buy",
                "avgExitPrice": "150",
                "qty": "1",
                "closedPnl": "5.0",
                "openFee": "0.1",
                "closeFee": "0.1",
                "updatedTime": str(int((now - 1000) * 1000)),
            },
            allow_orphan=False,
            min_orphan_ts=now,
        )
        self.assertFalse(res_skip)

        db.record_trade_event(
            symbol="SOLUSDT", action="ЧАСТИЧНЫЙ TP", side="Sell", strategy="SUPERTREND_PRO",
            price=160.0, qty=1.0, pnl_pct=15.0, order_id="tp1-sib", pos_ref="pos-sib-1",
            is_profit=True, ts=now + 100,
        )
        db.record_trade_event(
            symbol="SOLUSDT", action="ЗАКРЫТИЕ", side="Sell", strategy="SUPERTREND_PRO",
            price=165.0, qty=1.0, pnl_pct=22.0, order_id="close-sib", pos_ref="pos-sib-1",
            is_profit=True, ts=now + 200,
        )
        db.upsert_closed_pnl_record(
            {
                "orderId": "close-sib",
                "symbol": "SOLUSDT",
                "side": "Sell",
                "avgExitPrice": "165",
                "qty": "1.0",
                "closedPnl": "8.0",
                "openFee": "0.2",
                "closeFee": "0.2",
                "updatedTime": str(int((now + 205) * 1000)),
            },
            margin_used=100.0,
        )
        rows_sib = [r for r in db.get_trades(limit=10, symbol="SOLUSDT") if r.get("pos_ref") == "pos-sib-1" and r["action"] == "ЗАКРЫТИЕ"]
        self.assertEqual(len(rows_sib), 1)
        self.assertAlmostEqual(float(rows_sib[0]["pnl_pct"]), 22.0, places=2)
        self.assertAlmostEqual(float(rows_sib[0]["net_pnl_usdt"]), 8.0, places=2)

    def test_wfa_unique_pos_ids_for_monte_carlo_and_switcher_default(self):
        df_a = _make_ohlcv(420, start_price=100.0, trend=0.001, vol=0.01, seed=31)
        df_b = _make_ohlcv(420, start_price=50.0, trend=0.001, vol=0.01, seed=32)
        base_dfs = {"ETHUSDT": df_a, "SOLUSDT": df_b}

        def fake_eval(base_dfs, funding_map, strat_name, params, eval_start_ts=None, end_ts=None, initial_capital=1000.0):
            # Изолированные pos_id 1..4 в каждом окне — без префиксации окон они бы схлопнулись при concat
            tdf = pd.DataFrame({
                "pos_id": [1, 2, 3, 4],
                "usd_delta": [15.0, -5.0, 20.0, 10.0],
                "pnl_pct": [1.5, -0.5, 2.0, 1.0],
            })
            return {
                "trades": 4,
                "win_rate": 75.0,
                "pnl_pct": 4.0,
                "profit_factor": 2.5,
                "max_drawdown": -2.0,
                "sharpe_ratio": 1.8,
                "tdf": tdf,
            }

        with patch.object(opt_mod, "_eval_portfolio_params", side_effect=fake_eval), \
             patch.object(config, "USE_REGIME_SWITCHER", True), \
             patch.object(config, "ACTIVE_STRATEGY", "SUPERTREND_PRO"):
            res = opt_mod.run_portfolio_rolling_wfa(
                symbols=["ETHUSDT", "SOLUSDT"],
                trials=2,
                n_windows=2,
                strategy_name=None,
                mc_sims=50,
                base_dfs_override=base_dfs,
                funding_map_override={"ETHUSDT": None, "SOLUSDT": None},
            )
        self.assertEqual(res["strategy"], "REGIME_SWITCHER")
        self.assertEqual(len(res["windows"]), 2)
        self.assertEqual(res["test_metrics"]["trades"], 8)
        self.assertEqual(res["monte_carlo"]["n_trades"], 8)

    def test_ws_confirmed_candle_uses_idx_minus_one_immediately(self):
        b = _make_bare_bot(self.tmp.name)
        b.last_handled_candles = {}
        b._last_confirmed_kline_ts = {}
        b.ws_last_update = {}
        b.ws = MagicMock()
        b.ws.exited = False
        b.ws.is_connected.return_value = True

        tf = str(config.INTERVAL)
        bm_tf = str(config.BENCHMARK_INTERVAL)
        seed_df = _make_ohlcv(260, start_price=100.0, trend=0.002, vol=0.008, seed=77)
        bench_df = _make_ohlcv(260, start_price=50000.0, trend=0.001, vol=0.005, seed=78, freq="4h")
        b.kline_cache[("ETHUSDT", tf)] = seed_df.copy()
        b.kline_cache_ts[("ETHUSDT", tf)] = time.time()
        b.kline_cache[(config.BENCHMARK_SYMBOL, bm_tf)] = bench_df.copy()
        b.kline_cache_ts[(config.BENCHMARK_SYMBOL, bm_tf)] = time.time()

        fresh_ms = int(seed_df["timestamp"].iloc[-1].timestamp() * 1000) + 3_600_000
        b._handle_ws_kline({
            "topic": f"kline.{tf}.ETHUSDT",
            "data": [{"start": fresh_ms, "open": "120", "high": "122", "low": "119", "close": "121.0", "volume": "1500", "confirm": True}],
        }, ws_gen=1)
        self.assertEqual(b._closed_candle_idx("ETHUSDT", tf, b.kline_cache[("ETHUSDT", tf)]), -1)
        self.assertEqual(b._closed_candle_idx("ETHUSDT", tf, seed_df), -2)

        # Заставляем стратегию выдать сигнал ТОЛЬКО на последней свече (-1), чтобы проверить оценку свежей confirm=True свечи
        calc_stub = b.kline_cache[("ETHUSDT", tf)].copy()
        calc_stub["signal"] = 0
        calc_stub["exit_signal"] = 0
        calc_stub["ATR"] = 1.5
        calc_stub["SIGNAL_REGIME"] = "TREND"
        calc_stub.loc[calc_stub.index[-1], "signal"] = 1

        b.strategy = MagicMock()
        b.strategy.use_mtf_filter = False
        b.strategy.calculate_indicators.return_value = calc_stub
        b.get_price = MagicMock(return_value=121.0)
        b.open_position = MagicMock()

        with patch.object(bot_mod, "USE_MARKET_REGIME_FILTER", False), \
             patch.object(bot_mod, "SYMBOLS", ["ETHUSDT"]):
            b.active_symbols = ["ETHUSDT"]
            b._evaluate_market_cycle()

        b.open_position.assert_called_once()
        self.assertEqual(b.open_position.call_args.args[0], "ETHUSDT")
        self.assertEqual(b.open_position.call_args.args[1], "Buy")

    def test_shadow_panic_close_web_status_and_regime_snapshot_vol(self):
        # 1. MarketRegimeFilter.get_regime_snapshot & symmetric extreme_vol check in is_short_allowed
        rf = MarketRegimeFilter(
            ema_period=50, fast_ema_period=20,
            vol_window=10, extreme_vol_pct=0.05,
        )
        high_vol_df = _make_ohlcv(120, start_price=50000.0, trend=-0.002, vol=0.05, seed=99, freq="4h")
        snap = rf.get_regime_snapshot(high_vol_df, funding_zscore=0.25, oi_metrics={"oi_change_pct": -0.045, "price_change_pct": 0.01}, closed_idx=-1)
        self.assertIsNotNone(snap["btc_close"])
        self.assertIsNotNone(snap["rsi"])
        self.assertIsNotNone(snap["realized_vol"])
        self.assertIn(snap["trend"], ("bullish", "recovery", "bearish"))
        self.assertAlmostEqual(snap["oi_change_pct"], -4.5, places=1)
        self.assertFalse(snap["long_allowed"])
        self.assertFalse(snap["short_allowed"])

        s_ok, s_reason = rf.is_short_allowed("ETHUSDT", high_vol_df, closed_idx=-1)
        self.assertFalse(s_ok)
        self.assertIn("волатильност", s_reason.lower())

        # 2. ShadowEngine рестарт, panic_close_all в SHADOW_MODE и web /api/status
        db = BotDatabase(self.db_path)
        sh1 = ShadowEngine(db=db, initial_equity=1000.0, strategy_name="REGIME_SWITCHER", leverage=5)
        with patch.object(config, "SHADOW_MODE", True), \
             patch.object(config, "USE_RISK_BASED_SIZING", True):
            pos = sh1.open_position("ETHUSDT", "Buy", price=3000.0, atr_value=25.0)
        self.assertIsNotNone(pos)

        # Рестарт ShadowEngine восстанавливает открытую теневую позицию
        sh2 = ShadowEngine(db=db, initial_equity=1000.0, strategy_name="REGIME_SWITCHER", leverage=5)
        self.assertIn("ETHUSDT", sh2.positions)

        b = _make_bare_bot(self.tmp.name, db=db)
        b.shadow_engine = sh2
        b._sync_from_shadow_engine()
        b.get_price = MagicMock(return_value=3030.0)

        with patch.object(config, "SHADOW_MODE", True), \
             patch.object(config, "USE_REGIME_SWITCHER", True), \
             patch.object(config, "ACTIVE_STRATEGY", "SUPERTREND_PRO"):
            web_state.set_bot_instance(b)
            web_app._invalidate_status_cache()
            try:
                status = web_app._compute_status_sync()
                self.assertTrue(status["shadow_mode"])
                self.assertEqual(status["strategy"], "REGIME_SWITCHER")
                self.assertGreater(status["portfolio_heat_pct"], 0.0)
                b.get_available_balance = MagicMock(side_effect=AssertionError("real balance must not be called in SHADOW_MODE"))
                web_app._invalidate_status_cache()
                status2 = web_app._compute_status_sync()
                self.assertEqual(status2["open_positions_count"], 1)

                b.panic_close_all()
            finally:
                web_app._invalidate_status_cache()
                web_state.set_bot_instance(None)

        self.assertEqual(len(sh2.positions), 0)
        b.client.get_positions.assert_not_called()
        b.client.place_order.assert_not_called()
        b.client.cancel_all_orders.assert_not_called()

        # 3. index.html не умножает oi_change_pct на 100 повторно и сохраняет SRI
        html_path = Path(config.BASE_DIR) / "web" / "templates" / "index.html"
        html = html_path.read_text(encoding="utf-8")
        self.assertNotIn("rs.oi_change_pct) * 100", html)
        self.assertIn("integrity=\"sha384-", html)
        self.assertNotIn(".innerHTML", html)

    def test_standalone_dashboard_shadow_mode_no_real_rest_and_db_fallback(self):
        web_state.set_bot_instance(None)
        web_app._invalidate_status_cache()
        self.addCleanup(lambda: (web_state.set_bot_instance(None), web_app._invalidate_status_cache()))

        # 1. Без live_snap и без DB: SHADOW_MODE=True возвращает SHADOW_INITIAL_BALANCE, не создает DB и не вызывает Bybit REST
        with patch.object(config, "SHADOW_MODE", True), \
             patch.object(config, "SHADOW_INITIAL_BALANCE", 1500.0), \
             patch.object(web_app, "get_standalone_client", side_effect=AssertionError("standalone client must not be called in SHADOW_MODE")):
            web_app._invalidate_status_cache()
            st_empty = web_app._compute_status_sync()
            self.assertTrue(st_empty["shadow_mode"])
            self.assertEqual(st_empty["balance"], 1500.0)
            self.assertEqual(st_empty["equity"], 1500.0)
            self.assertEqual(st_empty["open_positions_count"], 0)
            self.assertFalse(self.db_path.exists())

            # Без live_snap: _close_position_sync и _panic_close_sync не трогают реальный клиент
            cl_no_bot = web_app._close_position_sync("ETHUSDT")
            self.assertEqual(cl_no_bot["status"], "error")
            self.assertTrue(cl_no_bot.get("shadow_mode"))
            pn_no_bot = web_app._panic_close_sync()
            self.assertEqual(pn_no_bot["status"], "error")
            self.assertTrue(pn_no_bot.get("shadow_mode"))

        # 2. Создаем ShadowEngine с открытой позицией + пишем срез через bot._publish_live_status()
        db = BotDatabase(self.db_path)
        sh = ShadowEngine(db=db, initial_equity=1000.0, strategy_name="REGIME_SWITCHER", leverage=5)
        with patch.object(config, "SHADOW_MODE", True), \
             patch.object(config, "USE_RISK_BASED_SIZING", True):
            pos = sh.open_position("ETHUSDT", "Buy", price=3000.0, atr_value=25.0)
        self.assertIsNotNone(pos)

        b = _make_bare_bot(self.tmp.name, db=db)
        b.shadow_engine = sh
        b._sync_from_shadow_engine()
        b.ws_prices["ETHUSDT"] = 3060.0  # +2% цена -> +10% ROI (x5)
        b.start_day = pd.Timestamp.now().date()
        b.start_day_balance = 1000.0
        b._last_regime_snapshot = {"trend": "bullish", "btc_close": 95000.0}
        b.save_state()

        with patch.object(config, "SHADOW_MODE", True):
            snap = b._publish_live_status()
        self.assertTrue(snap["shadow_mode"])
        self.assertIn("balance", snap)
        self.assertIn("shadow_positions", snap)
        self.assertEqual(len(snap["shadow_positions"]), 1)
        self.assertLess(snap["balance"], 1000.0)  # часть маржи занята
        self.assertGreater(snap["equity"], 1000.0)  # прибыль по mark price 3060

        # Standalone _compute_status_sync читает live_snap без get_standalone_client()
        web_state.set_bot_instance(None)
        with patch.object(config, "SHADOW_MODE", True), \
             patch.object(web_app, "get_standalone_client", side_effect=AssertionError("standalone client must not be called")):
            web_app._invalidate_status_cache()
            st_live = web_app._compute_status_sync()
            self.assertTrue(st_live["shadow_mode"])
            self.assertEqual(st_live["strategy"], "REGIME_SWITCHER")
            self.assertAlmostEqual(st_live["balance"], snap["balance"], places=2)
            self.assertAlmostEqual(st_live["equity"], snap["equity"], places=2)
            self.assertGreater(st_live["daily_pnl"], 0.0)
            self.assertGreater(st_live["portfolio_heat_pct"], 0.0)
            self.assertEqual(st_live["open_positions_count"], 1)
            self.assertTrue(st_live["positions"][0]["shadow_mode"])
            self.assertEqual(st_live["positions"][0]["mark_price"], 3060.0)
            self.assertEqual(st_live["regime_summary"].get("trend"), "bullish")

            # Standalone IPC close_position и panic при живом live_snap ставят команду в bot_control.json
            cl_miss = web_app._close_position_sync("SOLUSDT")
            self.assertEqual(cl_miss["status"], "error")
            cl_q = web_app._close_position_sync("ETHUSDT")
            self.assertEqual(cl_q["status"], "ok")
            self.assertTrue(cl_q.get("queued"))

            # bot.py потребляет IPC-команду закрытия и закрывает только shadow-позицию
            b.get_price = MagicMock(side_effect=lambda s: {"ETHUSDT": 3060.0, "SOLUSDT": 152.0}.get(s, 100.0))
            b._last_control_ts = 0.0
            b._sync_external_control()
            self.assertEqual(len(sh.positions), 0)
            b.client.place_order.assert_not_called()

            # Еще одна позиция -> проверка IPC panic из standalone dashboard
            with patch.object(config, "USE_RISK_BASED_SIZING", True):
                sh.open_position("SOLUSDT", "Buy", price=150.0, atr_value=2.0)
            b._sync_from_shadow_engine()
            b._publish_live_status()
            pn_q = web_app._panic_close_sync()
            self.assertEqual(pn_q["status"], "ok")
            self.assertTrue(pn_q.get("queued"))
            b._sync_external_control()
            self.assertEqual(len(sh.positions), 0)
            b.client.place_order.assert_not_called()

        # 3. Если live_snap устарел/удален, standalone SHADOW_MODE=True читает shadow_engine_state из SQLite
        with patch.object(config, "SHADOW_MODE", True), \
             patch.object(config, "USE_RISK_BASED_SIZING", True):
            sh.open_position("ETHUSDT", "Buy", price=3000.0, atr_value=25.0)
        b._sync_from_shadow_engine()
        b.save_state()
        live_path = Path(config.LIVE_STATUS_FILE)
        if live_path.exists():
            live_path.unlink()

        with patch.object(config, "SHADOW_MODE", True), \
             patch.object(web_app, "get_standalone_client", side_effect=AssertionError("standalone client must not be called")):
            web_app._invalidate_status_cache()
            st_db = web_app._compute_status_sync()
            self.assertTrue(st_db["shadow_mode"])
            self.assertEqual(st_db["open_positions_count"], 1)
            self.assertEqual(st_db["positions"][0]["symbol"], "ETHUSDT")
            self.assertTrue(st_db["positions"][0]["shadow_mode"])
            self.assertLess(st_db["balance"], 1000.0)
            self.assertGreater(st_db["portfolio_heat_pct"], 0.0)

        # 4. В обычном режиме (SHADOW_MODE=False) standalone по-прежнему обращается к get_standalone_client()
        mock_cli = MagicMock()
        mock_cli.get_wallet_balance.return_value = {
            "result": {"list": [{"totalAvailableBalance": "450.0", "coin": [{"coin": "USDT", "walletBalance": "450.0", "equity": "450.0"}]}]}
        }
        mock_cli.get_positions.return_value = {"result": {"list": []}}
        with patch.object(config, "SHADOW_MODE", False), \
             patch.object(web_app, "get_standalone_client", return_value=mock_cli) as m_cli:
            web_app._invalidate_status_cache()
            st_real = web_app._compute_status_sync()
            self.assertFalse(st_real["shadow_mode"])
            self.assertEqual(st_real["balance"], 450.0)
            m_cli.assert_called_once()

    def test_init_public_ws_kline_stream_intervals_dedup_and_reconnect(self):
        from pybit.unified_trading import WebSocket as RealPybitWS

        created_ws = []

        class FakePybitWS(RealPybitWS):
            def __init__(self, **kwargs):
                self.kwargs = kwargs
                self.standard_private_topics = ["position", "execution", "order", "wallet", "greeks", "spread.order", "spread.execution"]
                self.standard_public_topics = ["system.status"]
                self.callback_directory = {}
                self.subscriptions = {}
                self.ws = MagicMock()
                self.exited = False
                self.handle_error = kwargs.get("restart_on_error", True)
                self.retries = 10
                self.callback = lambda *a, **kw: None
                created_ws.append(self)

            def is_connected(self):
                return not self.exited

            def _validate_public_topic(self):
                return None

            def exit(self):
                self.exited = True

        b = _make_bare_bot(self.tmp.name)
        b.ws_last_update = {}
        b._last_confirmed_kline_ts = {}
        b.last_ws_public_msg = 0.0
        b.last_ws_reconnect_attempt = 0.0
        b.active_symbols = ["ethusdt", "ETHUSDT", "SOLUSDT", "BTCUSDT"]

        # 1. Разные интервалы (60, 240, D) + дубли символов + BTCUSDT уже в списке -> ноль исключений
        with patch.object(bot_mod, "WebSocket", FakePybitWS), \
             patch.object(config, "INTERVAL", "60"), \
             patch.object(config, "HIGHER_INTERVAL", "240"), \
             patch.object(config, "BENCHMARK_INTERVAL", "D"), \
             patch.object(config, "BENCHMARK_SYMBOL", "btcusdt"), \
             patch.object(config, "USE_WS_KLINE_TRIGGER", True):
            b._init_public_ws()

        ws1 = created_ws[-1]
        self.assertIs(b.ws, ws1)
        self.assertIn("kline.60.ETHUSDT", ws1.callback_directory)
        self.assertIn("kline.60.SOLUSDT", ws1.callback_directory)
        self.assertIn("kline.60.BTCUSDT", ws1.callback_directory)
        self.assertIn("kline.240.ETHUSDT", ws1.callback_directory)
        self.assertIn("kline.D.BTCUSDT", ws1.callback_directory)
        self.assertNotIn("kline.D.ETHUSDT", ws1.callback_directory)
        self.assertEqual(len([k for k in ws1.callback_directory if k.startswith("tickers.")]), 3)

        # 2. Совпадающие интервалы с пробелом/разным типом (" 60 ", "60", "60") не вызывают ошибку дубля в _check_callback_directory
        with patch.object(bot_mod, "WebSocket", FakePybitWS), \
             patch.object(config, "INTERVAL", " 60 "), \
             patch.object(config, "HIGHER_INTERVAL", "60"), \
             patch.object(config, "BENCHMARK_INTERVAL", "060"), \
             patch.object(config, "BENCHMARK_SYMBOL", "BTCUSDT"), \
             patch.object(config, "USE_WS_KLINE_TRIGGER", True):
            b._init_public_ws()

        ws2 = created_ws[-1]
        self.assertEqual(len(ws1.callback_directory), 0)
        self.assertEqual(len(ws1.subscriptions), 0)
        self.assertTrue(ws1.exited)
        self.assertIn("kline.60.ETHUSDT", ws2.callback_directory)

        # Старое поколение ws1 не должно срабатывать после переподключения
        seed_df = _make_ohlcv(50, start_price=100.0)
        b.kline_cache[("ETHUSDT", "60")] = seed_df.copy()
        b.kline_cache_ts[("ETHUSDT", "60")] = time.time()
        fresh_ms = int(seed_df["timestamp"].iloc[-1].timestamp() * 1000) + 3_600_000
        b._handle_ws_kline({
            "topic": "kline.60.ETHUSDT",
            "data": [{"start": fresh_ms, "open": "101", "high": "102", "low": "100", "close": "101.5", "volume": "100", "confirm": True}],
        }, ws_gen=1)
        self.assertFalse(b._candle_event.is_set())

        # 3. Динамическое обновление корзины (>10 символов батчинг + переподключение без дублей)
        many_syms = [f"COIN{i}USDT" for i in range(12)]
        b.universe_selector = MagicMock()
        b.universe_selector.select_symbols.return_value = many_syms
        b._last_universe_refresh = 0.0
        b.client.get_instruments_info.return_value = {
            "retCode": 0,
            "result": {"list": [{"lotSizeFilter": {"qtyStep": "0.1", "minOrderQty": "0.1", "minNotionalValue": "5"}, "priceFilter": {"tickSize": "0.01"}, "leverageFilter": {"maxLeverage": "10"}}]},
        }
        with patch.object(bot_mod, "WebSocket", FakePybitWS), \
             patch.object(config, "USE_DYNAMIC_UNIVERSE", True), \
             patch.object(config, "SHADOW_MODE", True), \
             patch.object(config, "INTERVAL", "60"), \
             patch.object(config, "HIGHER_INTERVAL", "240"), \
             patch.object(config, "BENCHMARK_INTERVAL", "240"), \
             patch.object(config, "BENCHMARK_SYMBOL", "BTCUSDT"):
            b.refresh_universe_if_needed(force=True)

        ws3 = created_ws[-1]
        self.assertTrue(ws2.exited)
        self.assertEqual(len([k for k in ws3.callback_directory if k.startswith("tickers.")]), 12)
        self.assertEqual(len([k for k in ws3.callback_directory if k.startswith("kline.60.")]), 13)
        # Проверка батчинга <= 10 аргументов в одном subscribe-фрейме
        for sub_raw in ws3.subscriptions.values():
            sub_args = json.loads(sub_raw)["args"]
            self.assertLessEqual(len(sub_args), 10)

    def test_gitignore_covers_sqlite_and_profiles_and_clean_repo_root(self):
        import subprocess
        root = Path(config.BASE_DIR)
        gi = (root / ".gitignore").read_text(encoding="utf-8")
        for pat in ("*.db", "*.db-wal", "*.db-shm", "bot_data.db*", "strategy_profiles.json", "bot_state.json", "bot_control.json", "bot_live_status.json"):
            self.assertIn(pat, gi)

        # Проверяем, что ни один runtime-артефакт не отслеживается git и корректно игнорируется
        runtime_artifacts = [
            "bot_data.db",
            "bot_data.db-wal",
            "bot_data.db-shm",
            "strategy_profiles.json",
            "bot_state.json",
            "bot_control.json",
            "bot_live_status.json",
        ]
        if (root / ".git").exists():
            tracked = subprocess.run(
                ["git", "ls-files", "--", *runtime_artifacts],
                cwd=str(root),
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(tracked.stdout.strip(), "", f"Runtime artifacts must not be tracked by git: {tracked.stdout}")
            for art in ("bot_data.db", "bot_data.db-wal", "bot_data.db-shm", "strategy_profiles.json"):
                ignored = subprocess.run(
                    ["git", "check-ignore", "-q", art],
                    cwd=str(root),
                    capture_output=True,
                    text=True,
                    check=False,
                )
                self.assertEqual(ignored.returncode, 0, f"{art} must be ignored by .gitignore")

        # Убеждаемся, что пути внутри теста изолированы и запись в БД/профиль не затрагивает корень проекта
        root_db = root / "bot_data.db"
        before_mtime = root_db.stat().st_mtime_ns if root_db.exists() else None
        db = BotDatabase()
        db.set_runtime_value("isolation_check", {"ok": True})
        self.assertEqual(Path(config.DB_FILE).resolve(), self.db_path.resolve())
        if before_mtime is None:
            self.assertFalse(root_db.exists())
        else:
            self.assertEqual(root_db.stat().st_mtime_ns, before_mtime)

    def test_shadow_runtime_meta_ticker_no_double_tick_and_ipc_timestamp_tie(self):
        db = BotDatabase(self.db_path)
        sh = ShadowEngine(db=db, initial_equity=1200.0, strategy_name="SUPERTREND_PRO", leverage=5)
        sh.portfolio_risk.peak_equity = 1350.0
        with patch.object(config, "SHADOW_MODE", True), \
             patch.object(config, "USE_RISK_BASED_SIZING", True):
            pos = sh.open_position("ETHUSDT", "Buy", price=3000.0, atr_value=25.0)
        self.assertIsNotNone(pos)

        b = _make_bare_bot(self.tmp.name, db=db)
        b.ws_last_update = {}
        b.shadow_engine = sh
        b.portfolio_risk = PortfolioRiskManager(peak_equity=0.0)
        b._sync_from_shadow_engine()

        # 1. _build_runtime_meta сохраняет peak_equity именно из shadow_engine.portfolio_risk, а не 0.0
        with patch.object(config, "SHADOW_MODE", True):
            meta = b._build_runtime_meta()
        self.assertAlmostEqual(meta["peak_equity"], 1350.0, places=2)
        self.assertAlmostEqual(meta["portfolio_risk"]["peak_equity"], 1350.0, places=2)

        # 2. _handle_ws_ticker в SHADOW_MODE не дергает _check_ws_risk_update и вызывает on_tick ровно 1 раз
        b._check_ws_risk_update = MagicMock(side_effect=AssertionError("must not run _check_ws_risk_update in SHADOW_MODE"))
        with patch.object(config, "SHADOW_MODE", True), \
             patch.object(sh, "on_tick", wraps=sh.on_tick) as spy_tick:
            b._handle_ws_ticker(
                {"topic": "tickers.ETHUSDT", "data": {"symbol": "ETHUSDT", "lastPrice": "3010.0"}},
                ws_gen=1,
            )
            spy_tick.assert_called_once_with("ETHUSDT", 3010.0)

        # 3. Несовместимый non-shadow live_snap (shadow_mode=False) игнорируется при panic/close в SHADOW_MODE=True
        web_state.write_live_status({"shadow_mode": False, "equity": 500.0, "balance": 500.0})
        web_state.set_bot_instance(None)
        with patch.object(config, "SHADOW_MODE", True), \
             patch.object(web_app, "get_standalone_client", side_effect=AssertionError("standalone client must not be called")):
            pn_stale = web_app._panic_close_sync()
            cl_stale = web_app._close_position_sync("ETHUSDT")
        self.assertEqual(pn_stale["status"], "error")
        self.assertEqual(cl_stale["status"], "error")

        # 4. Monotonic updated_at при совпадении time.time() + защита от clock-tie в _sync_external_control
        with patch.object(web_state.time, "time", return_value=2_000_000.0):
            ts1 = web_state.write_control_state(is_paused=True)
            ts2 = web_state.write_control_state(is_paused=True, pending_shadow_panic=True)
        self.assertGreater(ts2, ts1)

        # Даже если bot._last_control_ts == updated_at, pending_shadow_panic всё равно исполняется
        # И даже если get_price падает, panic_close_all в SHADOW_MODE закрывает по ws_prices/entry_price fallback
        b._last_control_ts = ts2
        b.get_price = MagicMock(side_effect=RuntimeError("ticker timeout"))
        with patch.object(config, "SHADOW_MODE", True):
            b._sync_external_control()
        self.assertEqual(len(sh.positions), 0)
        ctrl_after = web_state.read_control_state()
        self.assertFalse(ctrl_after.get("pending_shadow_panic", False))


if __name__ == "__main__":
    unittest.main()
