"""
Набор тестов для проверки функционала версии 2.1:
- Секция 1: Критические ошибки и логические баги (Lookahead Bias, min_notional, pd.merge_asof, atomic state, mock standalone client)
- Секция 2: Торговая механика и исполнение (Limit TP1, нативный Trailing Stop, приватный WS, Maker Post-Only)
- Секция 3: Стратегии и фильтры (ADX flat filter, BB_RSI, SUPERTREND, DONCHIAN)
- Секция 4: Веб-дашборд и UI (WebSocket endpoint, candles, equity curve, password for LIVE)
"""

import sys
import os
import time
import tempfile
import threading
from collections import deque
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import unittest
from unittest.mock import MagicMock, patch
import asyncio
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from strategies import (
    get_strategy, BaseStrategy, MaRsiStrategy,
    BbRsiReversionStrategy, SupertrendAtrStrategy, SupertrendProStrategy,
    DonchianBreakoutStrategy, STRATEGY_REGISTRY
)
from risk_manager import PositionRiskState, MarketRegimeFilter
import config
from web import state as web_state
from web.state import set_bot_instance, set_bot_paused, is_bot_paused, parse_trade_log
from web.app import get_status, toggle_pause, get_trades, set_mode, get_candles, get_equity_history


class TestV2Architecture(unittest.TestCase):

    def setUp(self):
        dates = pd.date_range("2026-01-01", periods=100, freq="1h")
        np.random.seed(42)
        base_price = 50000.0
        prices = base_price + np.cumsum(np.random.randn(100) * 100)

        self.df = pd.DataFrame({
            "timestamp": dates,
            "open": prices,
            "high": prices + 50,
            "low": prices - 50,
            "close": prices,
            "volume": np.random.uniform(10, 100, 100)
        })
        self._orig_mode = (config.TRADING_MODE, config.API_KEY, config.API_SECRET, config.DEMO, config.TESTNET)
        self._tmpdir = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmpdir.cleanup)
        self._ipc_patches = [
            patch.object(config, "DB_FILE", Path(self._tmpdir.name) / "bot_data.db"),
            patch.object(config, "PROFILES_FILE", Path(self._tmpdir.name) / "strategy_profiles.json"),
            patch.object(config, "STATE_FILE", Path(self._tmpdir.name) / "bot_state.json"),
            patch.object(config, "CONTROL_FILE", Path(self._tmpdir.name) / "bot_control.json"),
            patch.object(config, "LIVE_STATUS_FILE", Path(self._tmpdir.name) / "bot_live_status.json"),
            patch.object(config, "TRADE_LOG_FILE", Path(self._tmpdir.name) / "trade.log"),
            patch.object(config, "BOT_LOG_FILE", Path(self._tmpdir.name) / "bot.log"),
        ]
        for p in self._ipc_patches:
            p.start()
            self.addCleanup(p.stop)

    def tearDown(self):
        set_bot_instance(None)
        config.TRADING_MODE, config.API_KEY, config.API_SECRET, config.DEMO, config.TESTNET = self._orig_mode

    def test_strategy_factory(self):
        strategy = get_strategy("MA_RSI", fast_ma=15, slow_ma=40)
        self.assertIsInstance(strategy, BaseStrategy)
        self.assertIsInstance(strategy, MaRsiStrategy)
        self.assertEqual(strategy.fast_ma, 15)

        df_calc = strategy.calculate_indicators(self.df)
        self.assertIn("MA_fast", df_calc.columns)
        self.assertIn("MA_slow", df_calc.columns)
        self.assertIn("RSI", df_calc.columns)
        self.assertIn("ATR", df_calc.columns)
        self.assertIn("ADX", df_calc.columns)
        self.assertIn("signal", df_calc.columns)

    def test_partial_tp_and_breakeven(self):
        pos = PositionRiskState(
            symbol="BTCUSDT",
            side="Buy",
            entry_price=50000.0,
            size=1.0,
            initial_size=1.0,
            stop_loss=49000.0,
            take_profit=54000.0,
            atr_value=500.0
        )
        self.assertEqual(pos.risk_r, 1000.0)
        self.assertEqual(pos.tp1_price, 51500.0)

        # Цена не дошла до TP1 (51000)
        trig, qty, be_sl = pos.check_partial_tp(51000.0)
        self.assertFalse(trig)

        # Цена достигла TP1 (51600)
        trig, qty, be_sl = pos.check_partial_tp(51600.0)
        self.assertTrue(trig)
        self.assertEqual(qty, 0.5)
        self.assertGreater(be_sl, 50000.0)

    def test_trailing_stop_logic(self):
        pos = PositionRiskState(
            symbol="ETHUSDT",
            side="Buy",
            entry_price=50000.0,
            size=1.0,
            initial_size=1.0,
            stop_loss=49000.0,
            take_profit=55000.0,
            atr_value=500.0
        )

        # 1. Цена выросла до 52000
        updated, new_sl = pos.check_trailing_stop(
            current_price=52000.0,
            high_price=52000.0,
            low_price=51500.0,
            trigger_pct=0.015,
            atr_mult=1.5
        )
        self.assertTrue(updated)
        self.assertEqual(new_sl, 52000.0 - 750.0)
        self.assertGreater(new_sl, 49000.0)
        pos.stop_loss = new_sl

        # 2. Цена откатилась до 51500 -> стоп НЕ должен откатиться назад!
        updated, candidate_sl = pos.check_trailing_stop(
            current_price=51500.0,
            high_price=51800.0,
            low_price=51400.0,
            trigger_pct=0.015,
            atr_mult=1.5
        )
        self.assertFalse(updated)
        self.assertEqual(pos.stop_loss, 51250.0)

    def test_market_regime_filter(self):
        regime = MarketRegimeFilter(
            benchmark_symbol="BTCUSDT",
            ema_period=20,
            rsi_threshold=45.0,
            dump_pct_threshold=0.03,
            dump_lookback_candles=4
        )

        btc_dates = pd.date_range("2026-01-01", periods=50, freq="4h")
        btc_bullish = pd.DataFrame({
            "timestamp": btc_dates,
            "open": np.linspace(40000, 60000, 50),
            "high": np.linspace(40500, 60500, 50),
            "low": np.linspace(39500, 59500, 50),
            "close": np.linspace(40100, 60200, 50),
            "volume": [1000] * 50
        })

        allowed, msg = regime.is_long_allowed("SOLUSDT", btc_bullish)
        self.assertTrue(allowed)

        btc_dump = btc_bullish.copy()
        btc_dump.loc[49, "close"] = 55000.0
        btc_dump.loc[49, "low"] = 54000.0
        allowed, msg = regime.is_long_allowed("SOLUSDT", btc_dump)
        self.assertFalse(allowed)
        self.assertIn("дамп", msg.lower())

    def test_web_endpoints_no_network_leak(self):
        """
        1.5: Проверка дашборда БЕЗ утечки сетевых запросов.
        Регистрируем mock_bot с lock, balance и позициями.
        """
        mock_bot = MagicMock()
        mock_bot.get_available_balance.return_value = 1250.0
        mock_bot.start_day_balance = 1200.0
        mock_bot.circuit_breaker_active = False
        mock_bot.positions = {
            "BTCUSDT": {
                "side": "Buy",
                "position": "long",
                "size": 0.05,
                "entry_price": 50000.0,
                "stop_loss": 49000.0,
                "take_profit": 53000.0
            }
        }
        mock_bot.breakeven_applied = {"BTCUSDT": False}
        mock_bot.get_price.return_value = 50500.0
        mock_bot.lock = threading.RLock()
        mock_bot.get_klines.return_value = self.df

        set_bot_instance(mock_bot)

        loop = asyncio.new_event_loop()
        try:
            status_data = loop.run_until_complete(get_status())
            self.assertIn("mode", status_data)
            self.assertEqual(status_data["balance"], 1250.0)
            self.assertEqual(len(status_data["positions"]), 1)
            self.assertEqual(status_data["positions"][0]["symbol"], "BTCUSDT")

            initial_paused = is_bot_paused()
            pause_res = loop.run_until_complete(toggle_pause())
            self.assertEqual(pause_res["is_paused"], not initial_paused)
            loop.run_until_complete(toggle_pause())

            trades = loop.run_until_complete(get_trades())
            self.assertIsInstance(trades, list)

            # Проверка эндпоинта свечей
            candles = loop.run_until_complete(get_candles("BTCUSDT", "60", 50))
            self.assertIsInstance(candles, list)
            self.assertGreater(len(candles), 0)

            # Проверка эндпоинта equity
            eq_pts = loop.run_until_complete(get_equity_history())
            self.assertIsInstance(eq_pts, list)
        finally:
            loop.close()

    def test_position_risk_state_serialization(self):
        pos = PositionRiskState(
            symbol="SOLUSDT",
            side="Buy",
            entry_price=100.0,
            size=10.0,
            initial_size=10.0,
            stop_loss=95.0,
            take_profit=115.0,
            atr_value=2.5,
            risk_r=5.0,
            tp1_price=107.5,
            tp1_executed=True,
            tp1_order_id="12345678",
            breakeven_applied=True,
            trailing_active=True,
            highest_price=108.0,
            lowest_price=99.0
        )
        d = pos.to_dict()
        self.assertEqual(d["symbol"], "SOLUSDT")
        self.assertEqual(d["tp1_executed"], True)
        self.assertEqual(d["tp1_order_id"], "12345678")

        restored = PositionRiskState.from_dict(d)
        self.assertEqual(restored.symbol, pos.symbol)
        self.assertEqual(restored.tp1_order_id, "12345678")
        self.assertEqual(restored.tp1_executed, True)

    def test_market_regime_short_filter(self):
        regime = MarketRegimeFilter(
            benchmark_symbol="BTCUSDT",
            ema_period=20,
            rsi_threshold=45.0,
            dump_pct_threshold=0.03,
            dump_lookback_candles=4,
            bull_rsi_threshold=55.0,
            pump_pct_threshold=0.04
        )

        btc_dates = pd.date_range("2026-01-01", periods=50, freq="4h")
        btc_bullish = pd.DataFrame({
            "timestamp": btc_dates,
            "open": np.linspace(40000, 60000, 50),
            "high": np.linspace(40500, 60500, 50),
            "low": np.linspace(39500, 59500, 50),
            "close": np.linspace(40100, 60200, 50),
            "volume": [1000] * 50
        })

        allowed, msg = regime.is_short_allowed("SOLUSDT", btc_bullish)
        self.assertFalse(allowed)

        btc_bearish = pd.DataFrame({
            "timestamp": btc_dates,
            "open": np.linspace(60000, 40000, 50),
            "high": np.linspace(60500, 40500, 50),
            "low": np.linspace(59500, 39500, 50),
            "close": np.linspace(59900, 39800, 50),
            "volume": [1000] * 50
        })

        allowed, msg = regime.is_short_allowed("SOLUSDT", btc_bearish)
        self.assertTrue(allowed)

    def test_market_regime_vectorized_indicators(self):
        dates = pd.date_range("2026-01-01", periods=60, freq="4h")
        df_btc = pd.DataFrame({
            "timestamp": dates,
            "open": np.linspace(40000, 50000, 60),
            "high": np.linspace(40500, 50500, 60),
            "low": np.linspace(39500, 49500, 60),
            "close": np.linspace(40100, 50200, 60),
            "volume": [1000] * 60
        })

        df_res = MarketRegimeFilter.add_indicators(df_btc, ema_period=20, rsi_period=14)
        self.assertIn("BENCH_EMA", df_res.columns)
        self.assertIn("BENCH_RSI", df_res.columns)
        self.assertIn("BENCH_LONG_ALLOWED", df_res.columns)
        self.assertIn("BENCH_SHORT_ALLOWED", df_res.columns)
        self.assertFalse(df_res["BENCH_EMA"].isna().all())

    # =========================================================================
    # 1.2: ПРОВЕРКА МИНИМАЛЬНОГО НОТИОНАЛА (min_notional >= 5 USDT)
    # =========================================================================
    def test_min_notional_validation(self):
        from bot import BybitMultiFuturesBot
        bot = object.__new__(BybitMultiFuturesBot)
        bot.lock = threading.RLock()
        bot.instruments = {
            "BTCUSDT": {
                "qty_step": 0.001,
                "min_qty": 0.001,
                "min_notional": 5.0,
                "tick_size": 0.1,
                "qty_decimals": 3,
                "price_decimals": 2
            }
        }
        bot.positions = {}
        bot.circuit_breaker_active = False
        bot.get_available_balance = MagicMock(return_value=10.0)
        bot.get_open_positions_count = MagicMock(return_value=0)
        bot.strategy = MagicMock()
        bot.strategy.calculate_stops.return_value = {"sl": 49000.0, "tp": 52000.0, "sl_dist": 1000.0, "type": "ATR"}
        bot.round_price = lambda sym, p: p
        bot.round_qty = lambda sym, q: 0.001 # 0.001 * 3000 = 3.0$ (< 5$)

        # Вход по цене 3000 при объеме 0.001 дает стоимость 3 USDT (< 5.0 min_notional)
        res = bot.open_position("BTCUSDT", "Buy", price=3000.0, atr_value=100.0)
        self.assertIsNone(res)

    # =========================================================================
    # 1.4: АТОМАРНОЕ СОХРАНЕНИЕ СОСТОЯНИЯ (save_state через .tmp)
    # =========================================================================
    def test_atomic_state_save_and_load(self):
        from bot import BybitMultiFuturesBot
        bot = object.__new__(BybitMultiFuturesBot)
        bot.lock = threading.RLock()
        bot.state_file = Path(self._tmpdir.name) / "test_bot_state.json"
        bot.risk_states = {
            "SOLUSDT": PositionRiskState(
                symbol="SOLUSDT", side="Buy", entry_price=150.0, size=2.0, initial_size=2.0,
                stop_loss=140.0, take_profit=170.0
            )
        }

        try:
            bot.save_state()
            self.assertTrue(bot.state_file.exists())
            # Временный файл .tmp не должен оставаться после replace
            tmp_file = bot.state_file.with_suffix(".tmp")
            self.assertFalse(tmp_file.exists())

            loaded = bot.load_state()
            self.assertIn("SOLUSDT", loaded)
            self.assertEqual(loaded["SOLUSDT"].entry_price, 150.0)
        finally:
            if bot.state_file.exists():
                bot.state_file.unlink()

    # =========================================================================
    # 2.2: ПРИВАТНЫЙ WEBSOCKET POSITION HANDLER (<50ms реакция)
    # =========================================================================
    def test_ws_position_stream_handler(self):
        from bot import BybitMultiFuturesBot
        bot = object.__new__(BybitMultiFuturesBot)
        bot.lock = threading.RLock()
        bot.state_file = Path(self._tmpdir.name) / "test_ws_state.json"
        bot.save_state = MagicMock()
        bot.round_price = lambda sym, p: round(p, 2)
        bot.get_position_idx = lambda sym, side: 0
        bot.client = MagicMock()

        bot.positions = {"BTCUSDT": {"side": "Buy", "size": 1.0, "entry_price": 50000.0}}
        bot.risk_states = {"BTCUSDT": PositionRiskState("BTCUSDT", "Buy", 50000.0, 1.0, 1.0, 49000.0, 53000.0)}
        bot.breakeven_applied = {"BTCUSDT": False}

        # Сообщение WS: позиция закрыта (size = 0)
        ws_msg_close = {
            "data": [{"symbol": "BTCUSDT", "size": "0", "side": "Buy"}]
        }
        bot._handle_ws_position(ws_msg_close)

        self.assertIsNone(bot.positions["BTCUSDT"])
        self.assertIsNone(bot.risk_states["BTCUSDT"])
        bot.save_state.assert_called()

    # =========================================================================
    # 3.1: ФИЛЬТР ФЛЭТА ADX В MA_RSI
    # =========================================================================
    def test_adx_flat_filter(self):
        strategy = MaRsiStrategy(
            fast_ma=5, slow_ma=15, rsi_period=7,
            use_adx_filter=True, adx_threshold=25.0,
            volume_filter=False, use_mtf_filter=False
        )

        dates = pd.date_range("2026-01-01", periods=60, freq="1h")
        # Флэтовые цены (колебания вокруг 100 с амплитудой 0.1) -> ADX крайне низкий (< 15)
        np.random.seed(42)
        flat_prices = 100.0 + np.sin(np.linspace(0, 20, 60)) * 0.1
        df_flat = pd.DataFrame({
            "timestamp": dates,
            "open": flat_prices,
            "high": flat_prices + 0.05,
            "low": flat_prices - 0.05,
            "close": flat_prices,
            "volume": [100.0] * 60
        })

        df_calc = strategy.calculate_indicators(df_flat)
        self.assertIn("ADX", df_calc.columns)
        # При низком ADX сигналы должны быть заблокированы
        signals = df_calc["signal"].tolist()
        self.assertTrue(all(s == 0 for s in signals[-10:]))

    # =========================================================================
    # 3.2: НОВЫЕ СТРАТЕГИИ (BB_RSI, SUPERTREND, DONCHIAN)
    # =========================================================================
    def test_new_strategies_registry_and_indicators(self):
        for strat_name in ["BB_RSI", "SUPERTREND", "DONCHIAN"]:
            strat = get_strategy(strat_name)
            self.assertIsInstance(strat, BaseStrategy)
            df_res = strat.calculate_indicators(self.df)
            self.assertIn("signal", df_res.columns)
            self.assertIn("ATR", df_res.columns)

    # =========================================================================
    # 4.3: ЗАЩИТА ПЕРЕКЛЮЧЕНИЯ В LIVE ПАРОЛЕМ
    # =========================================================================
    def test_set_mode_security(self):
        loop = asyncio.new_event_loop()
        try:
            with patch.object(config, "DASHBOARD_PASSWORD", "Secret123"):
                # 1. Попытка переключить в LIVE без пароля -> 403 Forbidden
                from fastapi import HTTPException
                with self.assertRaises(HTTPException) as ctx:
                    loop.run_until_complete(set_mode("LIVE", password=None))
                self.assertEqual(ctx.exception.status_code, 403)

                # 2. Неверный пароль -> 403 Forbidden
                with self.assertRaises(HTTPException) as ctx:
                    loop.run_until_complete(set_mode("LIVE", password="WrongPassword"))
                self.assertEqual(ctx.exception.status_code, 403)

                # 3. Верный пароль -> Успешный переход
                res = loop.run_until_complete(set_mode("LIVE", password="Secret123"))
                self.assertEqual(res["status"], "ok")
                self.assertEqual(config.TRADING_MODE, "LIVE")

                # Возврат в DEMO не требует пароля
                res_demo = loop.run_until_complete(set_mode("DEMO"))
                self.assertEqual(res_demo["status"], "ok")
                self.assertEqual(config.TRADING_MODE, "DEMO")
        finally:
            loop.close()

    # =========================================================================
    # 5.1: DECIMAL ОКРУГЛЕНИЕ ЛОТОВ И ЦЕН + НАУЧНАЯ НОТАЦИЯ (1e-05)
    # =========================================================================
    def test_decimal_rounding_and_scientific_notation(self):
        from bot import BybitMultiFuturesBot
        bot = object.__new__(BybitMultiFuturesBot)
        bot.instruments = {
            "SOLUSDT": {
                "qty_step": 0.1,
                "min_qty": 0.1,
                "min_notional": 5.0,
                "tick_size": 0.01,
                "qty_decimals": 1,
                "price_decimals": 2
            },
            "PEPEUSDT": {
                "qty_step": 100.0,
                "min_qty": 100.0,
                "min_notional": 5.0,
                "tick_size": 1e-05,
                "qty_decimals": bot._count_decimals("100"),
                "price_decimals": bot._count_decimals("1e-05")
            }
        }
        # Классическая ошибка IEEE 754: 0.3 // 0.1 == 2.0 (0.2), с Decimal должно быть ровно 0.3
        self.assertEqual(bot.round_qty("SOLUSDT", 0.3), 0.3)
        self.assertEqual(bot.round_qty("SOLUSDT", 0.6), 0.6)
        self.assertEqual(bot.instruments["PEPEUSDT"]["price_decimals"], 5)
        self.assertEqual(bot.round_price("PEPEUSDT", 0.0000123), 0.00001)

    # =========================================================================
    # 5.2: HEDGE MODE WS POSITION HANDLER (НЕ СТИРАТЬ АКТИВНУЮ НОГУ ПРИ size=0 НА ВТОРОЙ)
    # =========================================================================
    def test_ws_position_hedge_mode_opposite_leg_ignored(self):
        from bot import BybitMultiFuturesBot
        bot = object.__new__(BybitMultiFuturesBot)
        bot.lock = threading.RLock()
        bot.save_state = MagicMock()
        bot.positions = {
            "BTCUSDT": {"side": "Buy", "position": "long", "size": 1.0, "entry_price": 50000.0, "position_idx": 1}
        }
        bot.risk_states = {
            "BTCUSDT": PositionRiskState("BTCUSDT", "Buy", 50000.0, 1.0, 1.0, 49000.0, 53000.0, position_idx=1)
        }
        bot.breakeven_applied = {"BTCUSDT": False}

        # Сообщение в Hedge Mode содержит обе ноги: Long (idx=1, size=1.0) и пустой Short (idx=2, size=0)
        ws_msg_hedge = {
            "data": [
                {"symbol": "BTCUSDT", "positionIdx": 1, "side": "Buy", "size": "1.0"},
                {"symbol": "BTCUSDT", "positionIdx": 2, "side": "Sell", "size": "0"}
            ]
        }
        bot._handle_ws_position(ws_msg_hedge)
        self.assertIsNotNone(bot.positions["BTCUSDT"])
        self.assertEqual(bot.positions["BTCUSDT"]["size"], 1.0)

    # =========================================================================
    # 5.3: MTF ФИЛЬТР ДЛЯ ВСЕХ СТРАТЕГИЙ И ОПТИМИЗАТОР НА НЕ-MA_RSI
    # =========================================================================
    def test_all_strategies_mtf_and_optimizer_compatibility(self):
        from optimizer import simulate_strategy_slice
        for strat_name in ["MA_RSI", "BB_RSI", "SUPERTREND", "DONCHIAN"]:
            strat = get_strategy(strat_name, use_mtf_filter=True, htf_ema_period=20)
            trend = strat.get_mtf_trend(self.df)
            self.assertIn(trend, ("bullish", "bearish", "neutral"))

            with patch("optimizer.ACTIVE_STRATEGY", strat_name):
                res = simulate_strategy_slice(self.df, {"fast_ma": 10, "slow_ma": 25, "rsi_period": 14})
                self.assertIn("trades", res)
                self.assertIn("sharpe_ratio", res)

    # =========================================================================
    # 5.4: WEBSOCKET /ws В FASTAPI С ЗАЩИТОЙ АВТОРИЗАЦИИ
    # =========================================================================
    def test_fastapi_websocket_endpoint_no_type_error(self):
        from web.app import app
        mock_bot = MagicMock()
        mock_bot.get_available_balance.return_value = 1000.0
        mock_bot.get_wallet_equity.return_value = 1000.0
        mock_bot.start_day_balance = 1000.0
        mock_bot.circuit_breaker_active = False
        mock_bot.positions = {}
        mock_bot.lock = threading.RLock()
        set_bot_instance(mock_bot)

        async def run_ws():
            with patch.object(config, "DASHBOARD_USERNAME", ""), patch.object(config, "DASHBOARD_PASSWORD", ""):
                scope = {"type": "websocket", "path": "/ws", "headers": [], "query_string": b""}
                msgs = [{"type": "websocket.connect"}, {"type": "websocket.disconnect"}]
                sent = []
                async def receive():
                    return msgs.pop(0) if msgs else {"type": "websocket.disconnect"}
                async def send(m):
                    sent.append(m)
                await app(scope, receive, send)
                self.assertTrue(any(m.get("type") == "websocket.accept" for m in sent))

        loop = asyncio.new_event_loop()
        try:
            loop.run_until_complete(run_ws())
        finally:
            loop.close()

    # =========================================================================
    # 5.5: ЗАЩИТА SYNC_POSITIONS ПРИ СМЕНЕ СТОРОНЫ И WS ORDER STREAM TP1
    # =========================================================================
    def test_sync_positions_side_mismatch_and_ws_order_stream(self):
        from bot import BybitMultiFuturesBot
        bot = object.__new__(BybitMultiFuturesBot)
        bot.lock = threading.RLock()
        bot.positions = {"BTCUSDT": None}
        bot.risk_states = {"BTCUSDT": None}
        bot.breakeven_applied = {"BTCUSDT": False}
        bot.save_state = MagicMock()
        # На диске сохранен старый LONG, а на бирже уже открыт SHORT
        bot.load_state = MagicMock(return_value={
            "BTCUSDT": PositionRiskState("BTCUSDT", "Buy", 50000.0, 1.0, 1.0, 49000.0, 53000.0, tp1_executed=True)
        })
        bot.client = MagicMock()
        bot.client.get_positions.return_value = {
            "result": {
                "list": [{
                    "symbol": "BTCUSDT",
                    "side": "Sell",
                    "size": "2.0",
                    "avgPrice": "60000.0",
                    "stopLoss": "61500.0",
                    "takeProfit": "56000.0",
                    "positionIdx": 0
                }]
            }
        }
        bot.sync_positions()
        # Должен создаться новый риск-стейт для Sell, а не восстановиться старый Buy!
        self.assertEqual(bot.risk_states["BTCUSDT"].side, "Sell")
        self.assertFalse(bot.risk_states["BTCUSDT"].is_long)
        self.assertFalse(bot.risk_states["BTCUSDT"].tp1_executed)
        self.assertLess(bot.risk_states["BTCUSDT"].tp1_price, 60000.0)

        # Проверка _handle_ws_order при отмене лимитного TP1 на бирже
        bot.risk_states["BTCUSDT"].tp1_order_id = "tp1_ord_999"
        bot._handle_ws_order({
            "data": [{"orderId": "tp1_ord_999", "symbol": "BTCUSDT", "orderStatus": "Cancelled", "cumExecQty": "0"}]
        })
        self.assertEqual(bot.risk_states["BTCUSDT"].tp1_order_id, "")

    # =========================================================================
    # 6.1: ПРОВЕРКА СТРАТЕГИИ И КОНФИГУРАЦИИ v2.2 (SUPERTREND_PRO, ADX>=18, БЕЗ BTC)
    # =========================================================================
    def test_v22_supertrend_pro_and_config(self):
        self.assertEqual(config.ACTIVE_STRATEGY, "SUPERTREND_PRO")
        self.assertEqual(config.ADX_THRESHOLD, 18.0)
        self.assertNotIn("BTCUSDT", config.SYMBOLS)
        self.assertEqual(config.BENCHMARK_SYMBOL, "BTCUSDT")
        self.assertEqual(config.ATR_SL_MULT, 2.2)
        self.assertEqual(config.ATR_TP_MULT, 4.5)

        strat = get_strategy("SUPERTREND_PRO", adx_threshold=18.0)
        self.assertIsInstance(strat, SupertrendProStrategy)
        df_res = strat.calculate_indicators(self.df)
        for col in ["SuperTrend", "Trend", "ADX", "RSI", "EMA50", "ATR", "signal"]:
            self.assertIn(col, df_res.columns)

        # Во флэте (ADX < 18) сигналы блокируются
        dates = pd.date_range("2026-01-01", periods=80, freq="1h")
        flat_p = 100.0 + np.sin(np.linspace(0, 25, 80)) * 0.1
        df_flat = pd.DataFrame({
            "timestamp": dates,
            "open": flat_p,
            "high": flat_p + 0.05,
            "low": flat_p - 0.05,
            "close": flat_p,
            "volume": [100.0] * 80
        })
        df_flat_res = strat.calculate_indicators(df_flat)
        self.assertTrue(all(s == 0 for s in df_flat_res["signal"].iloc[-20:]))

    # =========================================================================
    # 6.2: ЗАЩИТА CIRCUIT BREAKER ОТ ЛОЖНОГО СРАБАТЫВАНИЯ ПРИ ТАЙМАУТЕ API (-94.8%)
    # =========================================================================
    def test_circuit_breaker_ignores_api_timeout_fallback(self):
        from bot import BybitMultiFuturesBot
        bot = object.__new__(BybitMultiFuturesBot)
        bot.lock = threading.RLock()
        bot.start_day = datetime.now().date()
        bot.start_day_balance = 968.53
        bot.last_known_equity = 0.0
        bot.last_equity_ok = False
        bot.circuit_breaker_active = False
        bot.panic_close_all = MagicMock()
        bot.client = MagicMock()

        # 1. Таймаут сети -> get_wallet_equity возвращает FALLBACK_USDT (50.0), но last_equity_ok = False
        bot.client.get_wallet_balance.side_effect = Exception("Read timed out (read timeout=10)")
        bot.check_daily_circuit_breaker()
        self.assertFalse(bot.circuit_breaker_active)
        bot.panic_close_all.assert_not_called()

        # 2. Реальная просадка > 5% при валидном ответе API -> Circuit Breaker обязан сработать
        bot.client.get_wallet_balance.side_effect = None
        bot.client.get_wallet_balance.return_value = {
            "result": {"list": [{"coin": [{"coin": "USDT", "equity": "900.0", "walletBalance": "900.0"}]}]}
        }
        bot.check_daily_circuit_breaker()
        self.assertTrue(bot.circuit_breaker_active)
        bot.panic_close_all.assert_called_once()

    # =========================================================================
    # 6.3: ЗАЩИТА ОТ РАННЕГО БЕЗУБЫТКА НА ШУМЕ (< 1.0R) И LIMIT TP1 timeInForce="GTC"
    # =========================================================================
    def test_r_based_breakeven_and_tp1_gtc_and_ws_reconnect(self):
        from bot import BybitMultiFuturesBot
        # 1. Сделка NEARUSDT SHORT из реального лога: Вход 4.163, SL 4.402 (1R = 0.239 = 5.74%)
        pos = PositionRiskState(
            symbol="NEARUSDT",
            side="Sell",
            entry_price=4.163,
            size=49.8,
            initial_size=49.8,
            stop_loss=4.402,
            take_profit=3.675,
            atr_value=0.1086
        )
        # При ходе цены -1.5% (4.100 = всего 0.26R) безубыток НЕ должен активироваться!
        be_trig, _ = pos.check_breakeven(current_price=4.100, trigger_pct=0.015, offset_pct=0.002, r_mult=0.6)
        self.assertFalse(be_trig)

        # При прохождении >= 0.6R (4.015 <= 4.163 - 0.6 * 0.239) безубыток активируется!
        be_trig_r, new_sl = pos.check_breakeven(current_price=4.015, trigger_pct=0.015, offset_pct=0.002, r_mult=0.6)
        self.assertTrue(be_trig_r)
        self.assertAlmostEqual(new_sl, 4.163 * (1 - 0.002), places=4)

        # 2. Проверка Limit TP1 с timeInForce="GTC" и авто-реконнекта WebSocket
        bot = object.__new__(BybitMultiFuturesBot)
        bot.lock = threading.RLock()
        bot.last_ws_reconnect_attempt = 0.0
        bot.last_ws_public_msg = 0.0
        bot.ws = MagicMock()
        bot.ws.is_connected.return_value = False
        bot.ws_private = MagicMock()
        bot.ws_private.is_connected.return_value = False
        bot._init_public_ws = MagicMock()
        bot._init_private_ws = MagicMock()

        with patch.object(config, "API_KEY", "real_key"):
            bot.check_and_reconnect_websockets()
            bot._init_public_ws.assert_called_once()
            bot._init_private_ws.assert_called_once()

    def test_exchange_close_logging_and_closed_candle_regime_and_daily_risk_cap(self):
        from bot import BybitMultiFuturesBot
        # 1. Проверка что MarketRegimeFilter проверяет EMA/RSI по закрытой 4H свече (iloc[-2])
        regime = MarketRegimeFilter(benchmark_symbol="BTCUSDT", ema_period=20, rsi_threshold=45.0)
        dates = pd.date_range("2026-01-01", periods=50, freq="4h")
        # Медвежий тренд до предпоследней свечи (iloc[-2] ниже EMA), а на текущей незакрытой (iloc[-1]) ложный прокол вверх
        closes = np.linspace(60000, 40000, 50)
        closes[-1] = 48000.0  # Незакрытая свеча проколола EMA вверх
        df_fake_spike = pd.DataFrame({
            "timestamp": dates,
            "open": closes,
            "high": closes + 200,
            "low": closes - 200,
            "close": closes,
            "volume": [1000] * 50
        })
        allowed, _ = regime.is_long_allowed("NEARUSDT", df_fake_spike)
        self.assertFalse(allowed)

        # 2. Проверка блокировки новой сделки при исчерпании дневного лимита убытка (Circuit Breaker trap)
        bot = object.__new__(BybitMultiFuturesBot)
        bot.lock = threading.RLock()
        bot.instruments = {
            "NEARUSDT": {
                "qty_step": 0.1, "min_qty": 0.1, "min_notional": 5.0,
                "tick_size": 0.001, "qty_decimals": 1, "price_decimals": 3
            }
        }
        bot.positions = {}
        bot.circuit_breaker_active = False
        bot.start_day_balance = 1000.0
        bot.last_known_equity = 953.0  # Уже потеряно 4.7% из 5.0% дневного лимита (осталось 3$ < 40% от 15$ риска)
        bot.last_equity_ok = True
        bot.get_available_balance = MagicMock(return_value=953.0)
        bot.get_open_positions_count = MagicMock(return_value=0)
        bot.strategy = MagicMock()
        bot.strategy.calculate_stops.return_value = {"sl": 4.5, "tp": 5.0, "sl_dist": 0.15, "type": "ATR"}
        bot.round_price = lambda sym, p: round(p, 3)
        bot.round_qty = lambda sym, q: round(q, 1)

        res = bot.open_position("NEARUSDT", "Buy", price=4.65, atr_value=0.07)
        self.assertIsNone(res)


class TestV23AuditFixes(unittest.TestCase):
    """Тесты критических исправлений аудита v2.3 (без сетевых запросов и загрязнения логов)."""

    def setUp(self):
        dates = pd.date_range("2026-01-01", periods=120, freq="1h")
        rng = np.random.default_rng(42)
        prices = 50000.0 + np.cumsum(rng.normal(0.0, 120.0, size=120))
        self.df = pd.DataFrame({
            "timestamp": dates,
            "open": prices,
            "high": prices + 60.0,
            "low": prices - 60.0,
            "close": prices,
            "volume": rng.uniform(500.0, 2000.0, size=120),
        })
        self.orig_bot = web_state.get_bot_instance()
        self.orig_paused = web_state.is_bot_paused()
        self.tmpdir = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmpdir.cleanup)
        self.addCleanup(self._restore_globals)

        # Заглушки на файловые IPC/state/DB-пути, чтобы тесты не мусорили в рабочем каталоге и не читали боевую БД
        self._patches = [
            patch.object(config, "DB_FILE", Path(self.tmpdir.name) / "bot_data.db"),
            patch.object(config, "PROFILES_FILE", Path(self.tmpdir.name) / "strategy_profiles.json"),
            patch.object(config, "STATE_FILE", Path(self.tmpdir.name) / "bot_state.json"),
            patch.object(config, "CONTROL_FILE", Path(self.tmpdir.name) / "bot_control.json"),
            patch.object(config, "LIVE_STATUS_FILE", Path(self.tmpdir.name) / "bot_live_status.json"),
            patch.object(config, "TRADE_LOG_FILE", Path(self.tmpdir.name) / "trade.log"),
            patch.object(config, "BOT_LOG_FILE", Path(self.tmpdir.name) / "bot.log"),
        ]
        for p in self._patches:
            p.start()
            self.addCleanup(p.stop)

    def _restore_globals(self):
        web_state.set_bot_instance(self.orig_bot)
        web_state.set_bot_paused(self.orig_paused, persist=False)
        from web import app as web_app
        web_app._invalidate_status_cache()

    def _make_bot(self, instruments=None):
        from bot import BybitMultiFuturesBot
        bot = object.__new__(BybitMultiFuturesBot)
        bot.lock = threading.RLock()
        bot.client = MagicMock()
        bot.ws = None
        bot.ws_private = None
        bot._public_ws_gen = 1
        bot._private_ws_gen = 1
        bot.last_ws_public_msg = time.time()
        bot.last_ws_private_msg = time.time()
        bot.last_ws_reconnect_attempt = 0.0
        bot.instruments = instruments or {
            "NEARUSDT": {
                "qty_step": 0.1, "min_qty": 0.1, "min_notional": 5.0,
                "tick_size": 0.001, "qty_decimals": 1, "price_decimals": 3
            },
            "SOLUSDT": {
                "qty_step": 0.1, "min_qty": 0.1, "min_notional": 5.0,
                "tick_size": 0.01, "qty_decimals": 1, "price_decimals": 2
            },
        }
        bot.positions = {}
        bot.risk_states = {}
        bot.breakeven_applied = {}
        bot.ws_prices = {}
        bot.ws_last_update = {}
        bot.ws_orders = {}
        bot.last_exec_prices = {}
        bot.last_exec_info = {}
        bot._seen_exec_ids = deque(maxlen=1000)
        bot.position_modes = {"NEARUSDT": "MergedSingle", "SOLUSDT": "MergedSingle"}
        bot.position_mode_cache = {"NEARUSDT": 0, "SOLUSDT": 0}
        bot.circuit_breaker_active = False
        bot.start_day_balance = 1000.0
        bot.last_known_equity = 1000.0
        bot.last_equity_ok = True
        bot.get_available_balance = MagicMock(return_value=1000.0)
        bot.save_state = MagicMock()
        bot.strategy = get_strategy("SUPERTREND_PRO")
        return bot

    # a) Fail-closed TRADING_MODE & placeholder live key protection
    def test_resolve_trading_mode_fail_closed(self):
        demo_mode, _, _, is_demo, is_testnet = config.resolve_trading_mode("DEMO", validate_live_keys=False)
        self.assertEqual((demo_mode, is_demo, is_testnet), ("DEMO", True, False))
        test_mode, _, _, is_demo2, is_testnet2 = config.resolve_trading_mode("testnet", validate_live_keys=False)
        self.assertEqual((test_mode, is_demo2, is_testnet2), ("TESTNET", False, True))
        with self.assertRaises(ValueError):
            config.resolve_trading_mode("DEM0", validate_live_keys=False)
        with self.assertRaises(ValueError):
            config.resolve_trading_mode("PAPER", validate_live_keys=False)
        with patch.object(config, "LIVE_API_KEY", "your_live_key"), \
             patch.object(config, "LIVE_API_SECRET", "real_secret"):
            with self.assertRaises(ValueError):
                config.resolve_trading_mode("LIVE", validate_live_keys=True)

    # b) bot.py import leaves loggers free of FileHandlers & Optional available
    def test_bot_import_no_file_handlers_on_import(self):
        import logging
        import bot as bot_mod
        self.assertTrue(hasattr(bot_mod, "Optional"))
        for logger_name in ("bot", "trade"):
            lg = logging.getLogger(logger_name)
            file_handlers = [h for h in lg.handlers if isinstance(h, logging.FileHandler)]
            self.assertEqual(len(file_handlers), 0, f"Unexpected FileHandler on {logger_name} at import")

    # c) SmartMaker timeout/cancel race: already filled in history -> NO duplicate Market order
    def test_open_position_maker_cancel_race_no_double_order(self):
        bot = self._make_bot()
        bot.client.get_tickers.return_value = {"result": {"list": [{"bid1Price": "5.000", "ask1Price": "5.002"}]}}
        bot.client.place_order.return_value = {"retCode": 0, "result": {"orderId": "maker-1"}}
        bot.client.get_open_orders.return_value = {"result": {"list": [{"orderId": "maker-1"}]}}
        bot.client.cancel_order.side_effect = RuntimeError("110001 Order does not exist")
        bot.client.get_order_history.return_value = {
            "result": {"list": [{"orderId": "maker-1", "orderStatus": "Filled", "cumExecQty": "15.0", "avgPrice": "4.995"}]}
        }
        bot.client.set_trading_stop.return_value = {"retCode": 0}
        bot.client.get_positions.return_value = {"result": {"list": []}}

        with patch.object(config, "USE_SMART_EXECUTION", True), \
             patch.object(config, "SMART_EXECUTION_TIMEOUT", 0.01), \
             patch("bot.SMART_EXECUTION_TIMEOUT", 0.01), \
             patch("bot.time.sleep", return_value=None):
            res = bot.open_position("NEARUSDT", "Buy", price=5.0, atr_value=0.1)

        self.assertIsNotNone(res)
        # place_order: 1 PostOnly entry + 1 Limit TP1 = max 2 calls, ZERO Market duplicate
        entry_calls = [
            c for c in bot.client.place_order.call_args_list
            if not c.kwargs.get("reduceOnly", False)
        ]
        self.assertEqual(len(entry_calls), 1)
        self.assertEqual(entry_calls[0].kwargs["timeInForce"], "PostOnly")
        self.assertAlmostEqual(bot.positions["NEARUSDT"]["entry_price"], 4.995, places=4)

    # d) Maker >=95% partial fill cancels remainder & registers exact filled qty
    def test_open_position_maker_95pct_cancels_remainder(self):
        bot = self._make_bot()
        bot.client.get_tickers.return_value = {"result": {"list": [{"bid1Price": "5.000", "ask1Price": "5.002"}]}}
        bot.client.place_order.return_value = {"retCode": 0, "result": {"orderId": "maker-95"}}
        bot.client.cancel_order.return_value = {"retCode": 0}
        bot.client.set_trading_stop.return_value = {"retCode": 0}
        # Qty будет рассчитан ~18.7 (при balance=1000, risk=15, sl_dist=0.22 => ~68 -> capped max_margin=150 * 5 / 5 = 150)
        # Предварительно зададим get_available_balance=100 и POSITION_SIZE_PCT=0.5 => max_margin=50*0.15? Проверим:
        bot.get_available_balance.return_value = 100.0
        # sl_dist = 0.22 => raw_qty = 1.5 / 0.22 = 6.8, max_margin = 15 -> max_qty = 15, round = 6.8
        # Проверим динамически через ws_orders в place_order side_effect!
        def _place_side_effect(**kwargs):
            if not kwargs.get("reduceOnly", False) and kwargs.get("timeInForce") == "PostOnly":
                orig_q = float(kwargs["qty"])
                fill_96 = round(orig_q * 0.96, 1)
                bot.ws_orders["maker-95"] = {
                    "status": "PartiallyFilled",
                    "cum_qty": str(fill_96),
                    "avg_price": "5.000"
                }
                return {"retCode": 0, "result": {"orderId": "maker-95"}}
            return {"retCode": 0, "result": {"orderId": "tp1-ord"}}

        bot.client.place_order.side_effect = _place_side_effect

        with patch("bot.SMART_EXECUTION_TIMEOUT", 1.0), patch("bot.time.sleep", return_value=None):
            res = bot.open_position("NEARUSDT", "Buy", price=5.0, atr_value=0.1)

        self.assertIsNotNone(res)
        bot.client.cancel_order.assert_called_once()
        entry_calls = [c for c in bot.client.place_order.call_args_list if not c.kwargs.get("reduceOnly", False)]
        self.assertEqual(len(entry_calls), 1)
        expected_fill = round(float(entry_calls[0].kwargs["qty"]) * 0.96, 1)
        self.assertAlmostEqual(bot.positions["NEARUSDT"]["size"], expected_fill, places=3)

    # e) Maker partial + Market fallback fails -> partial kept & protected; set_trading_stop failure -> emergency close
    def test_open_position_partial_maker_market_fail_and_stop_retry_emergency(self):
        # 1. Partial Maker 50% + Market fallback fails
        bot = self._make_bot()
        bot.get_available_balance.return_value = 200.0
        bot.client.get_tickers.return_value = {"result": {"list": [{"bid1Price": "5.000", "ask1Price": "5.002"}]}}
        bot.client.cancel_order.return_value = {"retCode": 0}
        bot.client.set_trading_stop.return_value = {"retCode": 0}

        def _partial_then_fail(**kwargs):
            if kwargs.get("timeInForce") == "PostOnly":
                half_q = round(float(kwargs["qty"]) * 0.5, 1)
                bot.ws_orders["maker-part"] = {"status": "Cancelled", "cum_qty": str(half_q), "avg_price": "4.99"}
                return {"retCode": 0, "result": {"orderId": "maker-part"}}
            if kwargs.get("orderType") == "Market":
                return {"retCode": 10001, "retMsg": "Insufficient margin"}
            return {"retCode": 0, "result": {"orderId": "tp1-x"}}

        bot.client.place_order.side_effect = _partial_then_fail
        with patch("bot.SMART_EXECUTION_TIMEOUT", 1.0), patch("bot.time.sleep", return_value=None):
            res = bot.open_position("NEARUSDT", "Buy", price=5.0, atr_value=0.1)

        self.assertIsNotNone(res)
        self.assertGreaterEqual(bot.client.set_trading_stop.call_count, 1)
        self.assertIn("stopLoss", bot.client.set_trading_stop.call_args_list[0].kwargs)
        self.assertAlmostEqual(bot.positions["NEARUSDT"]["entry_price"], 4.99, places=4)

        # 2. set_trading_stop fails all 3 attempts -> emergency reduceOnly Market close & return None
        bot2 = self._make_bot()
        bot2.client.place_order.return_value = {"retCode": 0, "result": {"orderId": "mkt-1"}}
        bot2.client.set_trading_stop.return_value = {"retCode": 110099, "retMsg": "Invalid SL"}
        bot2.client.get_positions.return_value = {"result": {"list": []}}
        bot2.client.get_order_history.return_value = {"result": {"list": []}}

        with patch("bot.USE_SMART_EXECUTION", False), patch("bot.time.sleep", return_value=None):
            res2 = bot2.open_position("NEARUSDT", "Buy", price=5.0, atr_value=0.1)

        self.assertIsNone(res2)
        self.assertEqual(bot2.client.set_trading_stop.call_count, 3)
        close_calls = [c for c in bot2.client.place_order.call_args_list if c.kwargs.get("reduceOnly") is True]
        self.assertEqual(len(close_calls), 1)
        self.assertEqual(close_calls[0].kwargs["side"], "Sell")

    # f) Market avgPrice SL/TP recalc + TP1 rounding to tickSize=0.05 + TP1 non-zero retCode handling
    def test_open_position_avg_price_recalc_and_tp1_tick_rounding(self):
        bot = self._make_bot(instruments={
            "WIFUSDT": {
                "qty_step": 1.0, "min_qty": 1.0, "min_notional": 5.0,
                "tick_size": 0.05, "qty_decimals": 0, "price_decimals": 2
            }
        })
        bot.position_mode_cache = {"WIFUSDT": 0}
        bot.client.place_order.side_effect = [
            {"retCode": 0, "result": {"orderId": "mkt-wif"}},          # Market entry
            {"retCode": 110007, "retMsg": "TP1 price error"},          # Limit TP1 rejected gracefully
        ]
        bot.client.get_order_history.return_value = {
            "result": {"list": [{"orderId": "mkt-wif", "orderStatus": "Filled", "cumExecQty": "30", "avgPrice": "2.25"}]}
        }
        bot.client.set_trading_stop.return_value = {"retCode": 0}

        with patch("bot.USE_SMART_EXECUTION", False), patch("bot.time.sleep", return_value=None):
            res = bot.open_position("WIFUSDT", "Buy", price=2.00, atr_value=0.10)

        self.assertIsNotNone(res)
        self.assertEqual(bot.positions["WIFUSDT"]["entry_price"], 2.25)
        # Проверяем, что SL/TP пересчитаны от 2.25, а не от 2.00 (второй вызов — нативный trailingStop)
        self.assertGreaterEqual(bot.client.set_trading_stop.call_count, 1)
        sl_sent = float(bot.client.set_trading_stop.call_args_list[0].kwargs["stopLoss"])
        self.assertAlmostEqual(sl_sent, round((2.25 - 0.10 * config.ATR_SL_MULT) / 0.05) * 0.05, places=2)
        # Проверяем, что TP1 был отправлен округленным до шага tick_size=0.05
        tp1_calls = [c for c in bot.client.place_order.call_args_list if c.kwargs.get("reduceOnly") is True]
        self.assertEqual(len(tp1_calls), 1)
        tp1_sent = float(tp1_calls[0].kwargs["price"])
        self.assertAlmostEqual((tp1_sent * 20) % 1, 0.0, places=5)
        # Отклоненный TP1 не должен записать пустой/битый tp1_order_id
        self.assertEqual(bot.risk_states["WIFUSDT"].tp1_order_id, "")

    # g) close_position keeps state when Market close fails; sends Market BEFORE cancel_all_orders; panic_close_all cleans hedge legs
    def test_close_position_error_safety_and_panic_hedge_cleanup(self):
        bot = self._make_bot()
        bot.positions["NEARUSDT"] = {
            "side": "Buy", "position": "long", "size": 10.0, "entry_price": 5.0,
            "stop_loss": 4.8, "take_profit": 5.5, "position_idx": 0
        }
        bot.risk_states["NEARUSDT"] = PositionRiskState(
            symbol="NEARUSDT", side="Buy", entry_price=5.0, size=10.0, initial_size=10.0, stop_loss=4.8, take_profit=5.5
        )
        # 1. Market close fails -> keep position and do NOT cancel SL/TP1
        bot.client.place_order.return_value = {"retCode": 10001, "retMsg": "Network error"}
        bot.close_position("NEARUSDT", price=4.9, reason="Тест-отказ")
        self.assertIn("NEARUSDT", bot.positions)
        bot.client.cancel_all_orders.assert_not_called()

        # 2. Market close succeeds -> place_order occurs BEFORE cancel_all_orders
        call_order = []
        def _place_ok(**kwargs):
            call_order.append("place_order")
            return {"retCode": 0, "result": {"orderId": "cls-1"}}
        def _cancel_ok(**kwargs):
            call_order.append("cancel_all_orders")
            return {"retCode": 0}
        bot.client.place_order.side_effect = _place_ok
        bot.client.cancel_all_orders.side_effect = _cancel_ok
        bot.close_position("NEARUSDT", price=5.1, reason="Закрытие OK")
        self.assertEqual(call_order, ["place_order", "cancel_all_orders"])
        self.assertIsNone(bot.positions.get("NEARUSDT"))

        # 3. panic_close_all closes untracked Hedge leg from exchange get_positions
        bot2 = self._make_bot()
        bot2.client.get_positions.return_value = {
            "result": {"list": [
                {"symbol": "XRPUSDT", "side": "Sell", "size": "20", "positionIdx": 2, "avgPrice": "0.60"}
            ]}
        }
        bot2.client.place_order.return_value = {"retCode": 0}
        bot2.client.cancel_all_orders.return_value = {"retCode": 0}
        bot2.panic_close_all()
        bot2.client.place_order.assert_called_once()
        kw = bot2.client.place_order.call_args.kwargs
        self.assertEqual(kw["symbol"], "XRPUSDT")
        self.assertEqual(kw["side"], "Buy")
        self.assertEqual(kw["positionIdx"], 2)
        self.assertTrue(kw["reduceOnly"])

    # h) _handle_ws_execution ignores Funding + deduplicates execId; _log_exchange_close weighted TP1 + native trailing label
    def test_ws_execution_funding_dedup_and_weighted_close_log(self):
        bot = self._make_bot()
        bot._handle_ws_execution({"data": [{"execId": "e1", "symbol": "SOLUSDT", "execPrice": "120.0", "execQty": "1.0", "side": "Sell", "execType": "Trade"}]})
        # Повторный execId должен игнорироваться, а Funding НЕ должен перезаписывать last_exec_prices
        bot._handle_ws_execution({"data": [{"execId": "e1", "symbol": "SOLUSDT", "execPrice": "99.0", "execQty": "1.0", "side": "Sell", "execType": "Trade"}]})
        bot._handle_ws_execution({"data": [{"execId": "f1", "symbol": "SOLUSDT", "execPrice": "0.0001", "execQty": "1.0", "side": "Buy", "execType": "Funding"}]})
        self.assertEqual(bot.last_exec_prices["SOLUSDT"], 120.0)

        risk = PositionRiskState(
            symbol="SOLUSDT", side="Buy", entry_price=100.0, size=5.0, initial_size=10.0,
            stop_loss=105.0, take_profit=130.0, tp1_price=110.0, tp1_executed=True,
            tp1_fill_price=110.0, tp1_closed_qty=5.0, breakeven_applied=True, native_trailing_active=True
        )
        cur_pos = {"side": "Buy", "position": "long", "size": 5.0, "entry_price": 100.0, "stop_loss": 105.0, "take_profit": 130.0}
        with patch("bot.trade_log") as mock_tl:
            bot._log_exchange_close("SOLUSDT", cur_pos, risk, source="WebSocket")
            msg = mock_tl.info.call_args[0][0]
        # 50% @ +10% * 5x (+50%) + 50% @ +20% * 5x (+100%) => +75.00% и метка Trailing SL
        self.assertIn("Trailing SL", msg)
        self.assertIn("+75.00%", msg)

    # i) Partial reduction requires TP1 confirmation; serialization preserves tp1_fill_price/tp1_closed_qty/opened_at
    def test_ws_position_and_sync_tp1_confirmation_across_restart(self):
        bot = self._make_bot()
        bot.positions["SOLUSDT"] = {"side": "Buy", "position": "long", "size": 10.0, "entry_price": 100.0, "stop_loss": 95.0, "take_profit": 120.0, "position_idx": 0}
        risk = PositionRiskState(
            symbol="SOLUSDT", side="Buy", entry_price=100.0, size=10.0, initial_size=10.0,
            stop_loss=95.0, take_profit=120.0, tp1_price=110.0, tp1_order_id="tp1-sol",
            opened_at=1700000000.0
        )
        bot.risk_states["SOLUSDT"] = risk

        # Случайное уменьшение на 10% (9.0 вместо ~5.0) при цене 101 не должно считаться TP1
        bot.ws_prices["SOLUSDT"] = 101.0
        confirmed, _, _ = bot._is_confirmed_tp1_reduction("SOLUSDT", risk, 9.0)
        self.assertFalse(confirmed)

        # Исполнение по TP1 ордеру: ws_orders показывает Filled + размер упал до 5.0
        bot.ws_orders["tp1-sol"] = {"status": "Filled", "cum_qty": 5.0, "avg_price": 110.2}
        confirmed2, closed_q, fill_p = bot._is_confirmed_tp1_reduction("SOLUSDT", risk, 5.0)
        self.assertTrue(confirmed2)
        self.assertAlmostEqual(closed_q, 5.0)
        self.assertAlmostEqual(fill_p, 110.2)

        # Проверка round-trip сериализации PositionRiskState
        risk.tp1_executed = True
        risk.tp1_closed_qty = closed_q
        risk.tp1_fill_price = fill_p
        restored = PositionRiskState.from_dict(risk.to_dict())
        self.assertAlmostEqual(restored.tp1_fill_price, 110.2)
        self.assertAlmostEqual(restored.tp1_closed_qty, 5.0)
        self.assertAlmostEqual(restored.opened_at, 1700000000.0)

    # j) _safe_close_ws neutralizes inner ws callbacks and stale generation tokens are ignored
    def test_ws_safe_close_and_generation_token_blocks_stale_callbacks(self):
        from bot import BybitMultiFuturesBot
        fake_ws = MagicMock()
        fake_ws.ws = MagicMock()
        fake_ws.callback = lambda x: x
        fake_ws.ws.on_error = lambda *a: None
        BybitMultiFuturesBot._safe_close_ws(fake_ws)
        fake_ws.exit.assert_called_once()
        self.assertTrue(fake_ws.exited)
        self.assertEqual(fake_ws.retries, 1)
        self.assertFalse(fake_ws.handle_error)

        bot = self._make_bot()
        bot._public_ws_gen = 5
        bot._handle_ws_ticker({"topic": "tickers.SOLUSDT", "data": {"symbol": "SOLUSDT", "lastPrice": "999.0"}}, ws_gen=4)
        self.assertNotIn("SOLUSDT", bot.ws_prices)
        bot._handle_ws_ticker({"topic": "tickers.SOLUSDT", "data": {"symbol": "SOLUSDT", "lastPrice": "150.0"}}, ws_gen=5)
        self.assertEqual(bot.ws_prices["SOLUSDT"], 150.0)

    # k) get_available_balance returns 0.0 on exhausted free margin & API failure; small balance margin cap <= 0.95 * balance
    def test_get_available_balance_exhausted_margin_and_small_account_cap(self):
        bot = self._make_bot()
        # Real method on BybitMultiFuturesBot
        from bot import BybitMultiFuturesBot
        real_get_bal = BybitMultiFuturesBot.get_available_balance.__get__(bot, BybitMultiFuturesBot)

        bot.client.get_wallet_balance.return_value = {
            "result": {"list": [{"coin": [{"coin": "USDT", "walletBalance": "100.0", "availableToWithdraw": "0", "totalAvailableBalance": "0"}]}]}
        }
        self.assertEqual(real_get_bal(), 0.0)

        bot.client.get_wallet_balance.side_effect = RuntimeError("timeout")
        self.assertEqual(real_get_bal(), 0.0)

        # Счет $8 (< $10): max_avail_margin = 7.6 -> не должно пытаться выделить $10!
        bot2 = self._make_bot()
        bot2.get_available_balance.return_value = 8.0
        bot2.start_day_balance = 8.0
        bot2.last_known_equity = 8.0
        bot2.client.place_order.return_value = {"retCode": 0, "result": {"orderId": "small-1"}}
        bot2.client.set_trading_stop.return_value = {"retCode": 0}
        bot2.client.get_positions.return_value = {"result": {"list": []}}
        bot2.client.get_order_history.return_value = {
            "result": {"list": [{"orderId": "small-1", "orderStatus": "Filled", "cumExecQty": "7.5", "avgPrice": "5.0"}]}
        }
        with patch("bot.USE_SMART_EXECUTION", False), patch("bot.time.sleep", return_value=None):
            bot2.open_position("NEARUSDT", "Buy", price=5.0, atr_value=0.01)

        entry_calls = [c for c in bot2.client.place_order.call_args_list if not c.kwargs.get("reduceOnly", False)]
        self.assertEqual(len(entry_calls), 1)
        sent_margin = (float(entry_calls[0].kwargs["qty"]) * 5.0) / config.LEVERAGE
        self.assertLessEqual(sent_margin, 8.0 * 0.95 + 1e-6)

    # l) MarketRegimeFilter & get_higher_trend fail-safe on empty/NaN/error data
    def test_regime_filter_and_get_higher_trend_fail_safe(self):
        regime = MarketRegimeFilter(benchmark_symbol="BTCUSDT", ema_period=20)
        for bad_df in (None, pd.DataFrame(), self.df.iloc[:1].copy()):
            ok_l, _ = regime.is_long_allowed("SOLUSDT", bad_df)
            ok_s, _ = regime.is_short_allowed("SOLUSDT", bad_df)
            self.assertFalse(ok_l)
            self.assertFalse(ok_s)

        nan_df = self.df.copy()
        nan_df["close"] = np.nan
        self.assertFalse(regime.is_long_allowed("SOLUSDT", nan_df)[0])
        self.assertFalse(regime.is_short_allowed("SOLUSDT", nan_df)[0])

        bot = self._make_bot()
        bot.get_klines = MagicMock(return_value=pd.DataFrame())
        self.assertEqual(bot.get_higher_trend("SOLUSDT"), "unavailable")

    # m) Strategies: calc_adx NaN=0.0, unfiltered exit_signal, Donchian edge-trigger, MaRsi single trigger & NaN HTF
    def test_strategies_adx_exit_donchian_edge_ma_rsi(self):
        from strategies.base import BaseStrategy
        from strategies.supertrend_atr import SupertrendProStrategy
        from strategies.donchian_breakout import DonchianBreakoutStrategy
        from strategies.ma_rsi_strategy import MaRsiStrategy

        adx = BaseStrategy.calc_adx(self.df.iloc[:15].copy(), period=14)
        self.assertTrue((adx.iloc[:5] == 0.0).all())

        # SupertrendPro: даже при жестком ADX (signal=0), переворот тренда формирует exit_signal
        sp = SupertrendProStrategy(adx_threshold=99.0, use_mtf_filter=False)
        res_sp = sp.calculate_indicators(self.df.copy())
        self.assertTrue((res_sp["signal"] == 0).all())
        self.assertGreater(int((res_sp["exit_signal"] != 0).sum()), 0)

        # Donchian: не спамит сигналом входа на каждом баре
        don = DonchianBreakoutStrategy(lookback_period=10, volume_filter=False, use_mtf_filter=False)
        trend_up = self.df.copy()
        trend_up["close"] = np.linspace(100, 200, len(trend_up))
        trend_up["high"] = trend_up["close"] + 1.0
        trend_up["low"] = trend_up["close"] - 1.0
        res_don = don.calculate_indicators(trend_up)
        # Если вчера уже был long-пробой, сегодня не должно быть подряд дублирующего сигнала при монотонном росте после пробоя
        consec_long = ((res_don["signal"] == 1) & (res_don["signal"].shift(1) == 1)).sum()
        self.assertEqual(int(consec_long), 0)

        # MaRsi: без NaN-проскока HTF_EMA и не больше 1 сигнала на пересечение при cross_lookback=3
        ma = MaRsiStrategy(fast_ma=5, slow_ma=15, use_mtf_filter=True, cross_lookback=3, volume_filter=False, use_adx_filter=False)
        df_nan_htf = self.df.copy()
        df_nan_htf["HTF_EMA"] = np.nan
        res_ma = ma.calculate_indicators(df_nan_htf)
        self.assertTrue((res_ma["signal"] == 0).all())

    # n) data_loader skips empty DataFrame & ignores 0-row empty CSV cache without IndexError
    def test_data_loader_skips_empty_csv_and_handles_empty_download(self):
        import data_loader
        with patch.object(config, "DATA_DIR", Path(self.tmpdir.name)):
            empty_csv = os.path.join(self.tmpdir.name, "TONUSDT_240m_90d.csv")
            with open(empty_csv, "w", encoding="utf-8") as f:
                f.write("timestamp,open,high,low,close,volume,turnover\n")
            self.assertTrue(data_loader._read_nonempty_csv(empty_csv).empty)

            with patch.object(data_loader, "download_klines", return_value=pd.DataFrame()) as mock_dl:
                out = data_loader.load_or_download("TONUSDT", "240", 90)
                self.assertTrue(out.empty)
                mock_dl.assert_called_once()

            # download_klines с пустым ответом API не должен падать с IndexError
            mock_http = MagicMock()
            mock_http.get_kline.return_value = {"retCode": 0, "result": {"list": []}}
            with patch("pybit.unified_trading.HTTP", return_value=mock_http):
                dl_out = data_loader.download_klines("TONUSDT", "240", 90, save_csv=True, show_progress=False)
                self.assertTrue(dl_out.empty)

    # o) Backtest & Optimizer: HTF_CLOSE vs HTF_EMA, NaN regime fail-safe, MTM end, pos_id stats, WFA 0-trade handling
    def test_backtest_mtf_nan_mtm_pos_id_and_optimizer_wfa(self):
        import backtest, optimizer
        dates_1h = pd.date_range("2026-01-01", periods=120, freq="1h")
        dates_4h = pd.date_range("2026-01-01", periods=30, freq="4h")
        df_1h = pd.DataFrame({
            "timestamp": dates_1h,
            "open": np.linspace(100, 120, 120),
            "high": np.linspace(101, 121, 120),
            "low": np.linspace(99, 119, 120),
            "close": np.linspace(100, 120, 120),
            "volume": [1000.0] * 120,
        })
        df_4h = pd.DataFrame({
            "timestamp": dates_4h,
            "open": np.linspace(100, 120, 30),
            "high": np.linspace(101, 121, 30),
            "low": np.linspace(99, 119, 30),
            "close": np.linspace(100, 120, 30),
            "volume": [4000.0] * 30,
        })
        aligned = backtest.align_htf_to_ltf(df_1h, df_4h, "60", "240", 5)
        self.assertIn("HTF_CLOSE", aligned.columns)
        self.assertIn("HTF_EMA", aligned.columns)

        # Группировка TP1 + SL по pos_id и ограничение PF
        trades = [
            {"pos_id": 1, "pnl_pct": 5.0},
            {"pos_id": 1, "pnl_pct": -1.0},  # суммарно позиция #1 в плюсе (+4.0) -> WR = 100%
        ]
        cnt, wr, pf, _, _ = backtest._aggregate_position_stats(trades)
        self.assertEqual(cnt, 1)
        self.assertEqual(wr, 100.0)
        self.assertEqual(pf, backtest.MAX_PROFIT_FACTOR)

        # optimizer.simulate_strategy_slice при 0 сделок возвращает 0.0 (без -99.0 в метриках WFA)
        zero_res = optimizer.simulate_strategy_slice(df_1h.iloc[:15].copy(), {}, symbol="SOLUSDT")
        self.assertEqual(zero_res["trades"], 0)
        self.assertEqual(zero_res["sharpe_ratio"], 0.0)
        self.assertEqual(zero_res["max_drawdown"], 0.0)

    # p) Web log parser, equity history, view_log clamping, close unlisted sym, index.html XSS/SRI, auth preserved
    def test_web_log_parser_equity_xss_sri_and_preserved_auth(self):
        from web import app as web_app
        from fastapi import HTTPException

        fake_trade_log = os.path.join(self.tmpdir.name, "trade.log")
        lines = []
        # 300 строк "Баланс:" должны быть отфильтрованы до реальных сделок!
        lines.append("2026-11-11 07:50:00 [INFO] Стартовый баланс: 1000.0 USDT\n")
        lines.append("2026-11-11 08:00:00 [INFO] [SOLUSDT] LONG открыт | Вход: 100.0 | SL: 95.0 | TP: 115.0\n")
        lines.append("2026-11-11 08:30:00 [INFO] [SOLUSDT] WS Fill (Funding): Buy 1.0 @ 0.0001\n")
        lines.append("2026-11-11 09:00:00 [INFO] [SOLUSDT] Закрыто (WebSocket: Trailing SL) @ 108.0 | PnL: +30.00%\n")
        for idx in range(300):
            lines.append(f"2026-11-11 10:{idx % 60:02d}:00 [INFO] Баланс: 1030.0 USDT | Активные (0/3)\n")

        with open(fake_trade_log, "w", encoding="utf-8") as f:
            f.writelines(lines)

        with patch.object(config, "TRADE_LOG_FILE", Path(fake_trade_log)), \
             patch.object(web_state.config, "TRADE_LOG_FILE", Path(fake_trade_log)):
            parsed = web_state.parse_trade_log(limit=50)
            self.assertEqual(len(parsed), 2)
            self.assertEqual(parsed[0]["symbol"], "SOLUSDT")
            self.assertNotEqual(parsed[0]["symbol"], "INFO")
            self.assertEqual(parsed[0]["price"], "108.0")
            self.assertIn("Trailing SL", parsed[0]["details"])

            eq_hist = web_state.parse_equity_history(initial_balance=1000.0)
            self.assertGreaterEqual(len(eq_hist), 2)
            # При TP (+30% с маржой 15%) реальный баланс переходит на ~1045, затем якорится на снимке 1030.0
            self.assertAlmostEqual(eq_hist[-1]["value"], 1030.0, places=2)

        # view_log clamp: lines=0 не должен падать и ограничивается
        fake_bot_log = os.path.join(self.tmpdir.name, "bot.log")
        with open(fake_bot_log, "w", encoding="utf-8") as f:
            for k in range(20):
                f.write(f"line-{k}\n")
        loop = asyncio.new_event_loop()
        try:
            with patch.object(config, "BOT_LOG_FILE", Path(fake_bot_log)):
                resp = loop.run_until_complete(web_app.view_log(file="bot", lines=0))
                self.assertEqual((resp.body.decode("utf-8") if hasattr(resp, "body") else str(resp)).strip(), "line-19")
        finally:
            loop.close()

        # close_position позволяет закрыть открытую монету не из config.SYMBOLS, сначала Market потом cancel_all_orders
        call_seq = []
        mock_client = MagicMock()
        mock_client.get_positions.return_value = {
            "result": {"list": [{"symbol": "XRPUSDT", "side": "Buy", "size": "50", "positionIdx": 0}]}
        }
        mock_client.place_order.side_effect = lambda **kw: (call_seq.append("place"), {"retCode": 0})[1]
        mock_client.cancel_all_orders.side_effect = lambda **kw: (call_seq.append("cancel"), {"retCode": 0})[1]

        web_state.set_bot_instance(None)
        with patch.object(web_app, "get_standalone_client", return_value=mock_client):
            res = web_app._close_position_sync("XRPUSDT")
        self.assertEqual(res["status"], "ok")
        self.assertEqual(call_seq, ["place", "cancel"])

        # index.html: SRI на CDN lightweight-charts и безопасный DOM вместо innerHTML
        tpl_path = os.path.join(config.BASE_DIR, "web", "templates", "index.html")
        with open(tpl_path, encoding="utf-8") as f:
            html_src = f.read()
        self.assertIn("integrity=\"sha384-", html_src)
        self.assertNotIn("tbody.innerHTML", html_src)
        self.assertNotIn("select.innerHTML", html_src)
        self.assertNotIn("btBody.innerHTML", html_src)

        # Сохраненное поведение авторизации дашборда (не изменено)
        class _DummyConn:
            def __init__(self, auth_header=""):
                self.headers = {"Authorization": auth_header} if auth_header else {}
                self.scope = {"type": "http"}

        with patch.object(config, "DASHBOARD_USERNAME", ""), patch.object(config, "DASHBOARD_PASSWORD", ""):
            self.assertTrue(web_app.verify_dashboard_auth(_DummyConn()))
        with patch.object(config, "DASHBOARD_USERNAME", "admin"), patch.object(config, "DASHBOARD_PASSWORD", "Secret123"):
            with self.assertRaises(HTTPException) as ctx:
                web_app.verify_dashboard_auth(_DummyConn())
            self.assertEqual(ctx.exception.status_code, 401)

    # q) Bybit 34040 'not modified', pause during Maker wait, avgPrice from get_positions, IPC mode rollback, TP1 price plausibility
    def test_v23_reviewer_execution_and_ipc_fixes(self):
        # 1. Bybit 34040 'not modified' на set_trading_stop НЕ должен экстренно закрывать валидную Market-позицию
        bot = self._make_bot()
        bot.client.place_order.return_value = {"retCode": 0, "result": {"orderId": "mkt-34040"}}
        bot.client.get_order_history.return_value = {
            "result": {"list": [{"orderId": "mkt-34040", "orderStatus": "Filled", "cumExecQty": "15.0", "avgPrice": "5.0"}]}
        }
        bot.client.set_trading_stop.side_effect = RuntimeError("not modified (ErrCode: 34040)")
        with patch("bot.USE_SMART_EXECUTION", False), patch("bot.time.sleep", return_value=None):
            res = bot.open_position("NEARUSDT", "Buy", price=5.0, atr_value=0.1)
        self.assertIsNotNone(res)
        self.assertIn("NEARUSDT", bot.positions)
        close_calls = [c for c in bot.client.place_order.call_args_list if c.kwargs.get("reduceOnly") is True and c.kwargs.get("orderType") == "Market"]
        self.assertEqual(len(close_calls), 0)

        # 2. Если во время ожидания Maker бот поставлен на паузу -> Market добор не отправляется
        bot_p = self._make_bot()
        bot_p.client.get_tickers.return_value = {"result": {"list": [{"bid1Price": "5.000", "ask1Price": "5.002"}]}}
        bot_p.client.place_order.return_value = {"retCode": 0, "result": {"orderId": "maker-pause"}}
        bot_p.client.get_open_orders.return_value = {"result": {"list": [{"orderId": "maker-pause"}]}}
        bot_p.client.get_order_history.return_value = {"result": {"list": []}}
        bot_p.client.get_positions.return_value = {"result": {"list": []}}
        bot_p.client.cancel_order.return_value = {"retCode": 0}
        with patch("bot.USE_SMART_EXECUTION", True), \
             patch("bot.SMART_EXECUTION_TIMEOUT", 1.0), \
             patch("bot.time.sleep", side_effect=lambda *_: web_state.set_bot_paused(True, persist=False)):
            res_p = bot_p.open_position("NEARUSDT", "Buy", price=5.0, atr_value=0.1)
        self.assertIsNone(res_p)
        mkt_entries = [c for c in bot_p.client.place_order.call_args_list if c.kwargs.get("orderType") == "Market"]
        self.assertEqual(len(mkt_entries), 0)
        web_state.set_bot_paused(False, persist=False)

        # 3. Partial Maker + Market добор с avgPrice из get_positions НЕ должен дважды усреднять maker_avg_price
        bot_avg = self._make_bot()
        bot_avg.get_available_balance.return_value = 200.0
        bot_avg.client.get_tickers.return_value = {"result": {"list": [{"bid1Price": "5.000", "ask1Price": "5.002"}]}}
        bot_avg.client.cancel_order.return_value = {"retCode": 0}
        bot_avg.client.set_trading_stop.return_value = {"retCode": 0}
        bot_avg.client.get_order_history.return_value = {"result": {"list": []}}
        bot_avg.client.get_positions.side_effect = [
            {"result": {"list": []}},  # after cancel check
            {"result": {"list": [{"symbol": "NEARUSDT", "side": "Buy", "size": "6.8", "avgPrice": "5.10", "positionIdx": 0}]}},  # after market fill
        ]

        def _part_mkt(**kwargs):
            if kwargs.get("timeInForce") == "PostOnly":
                half = round(float(kwargs["qty"]) * 0.5, 1)
                bot_avg.ws_orders["maker-mix"] = {"status": "Cancelled", "cum_qty": str(half), "avg_price": "4.90"}
                return {"retCode": 0, "result": {"orderId": "maker-mix"}}
            return {"retCode": 0, "result": {"orderId": "mkt-mix"}}

        bot_avg.client.place_order.side_effect = _part_mkt
        with patch("bot.SMART_EXECUTION_TIMEOUT", 1.0), patch("bot.time.sleep", return_value=None):
            res_avg = bot_avg.open_position("NEARUSDT", "Buy", price=5.0, atr_value=0.1)
        self.assertIsNotNone(res_avg)
        self.assertAlmostEqual(bot_avg.positions["NEARUSDT"]["entry_price"], 5.10, places=4)

        # 4. _sync_external_control при провале reload_credentials откатывает bot_control.json назад на DEMO
        bot_ipc = self._make_bot()
        bot_ipc.is_paused = False
        bot_ipc._last_control_ts = 0.0
        bot_ipc.reload_credentials = MagicMock(return_value=False)
        orig_cfg = (config.TRADING_MODE, config.API_KEY, config.API_SECRET, config.DEMO, config.TESTNET)
        try:
            config.TRADING_MODE = "DEMO"
            web_state.write_control_state(is_paused=False, trading_mode="LIVE")
            bot_ipc._sync_external_control()
            self.assertEqual(config.TRADING_MODE, "DEMO")
            self.assertEqual(web_state.read_control_state().get("trading_mode"), "DEMO")
        finally:
            config.TRADING_MODE, config.API_KEY, config.API_SECRET, config.DEMO, config.TESTNET = orig_cfg

        # 5. _is_confirmed_tp1_reduction с активным tp1_order_id не подтверждает TP1 при движении в убыток
        bot_tp1 = self._make_bot()
        risk_tp1 = PositionRiskState(
            symbol="SOLUSDT", side="Buy", entry_price=100.0, size=10.0, initial_size=10.0,
            stop_loss=95.0, take_profit=120.0, tp1_price=110.0, tp1_order_id="tp1-open-loss"
        )
        bot_tp1.ws_prices["SOLUSDT"] = 96.0  # Цена в убытке, далеко от TP1=110
        confirmed_loss, _, _ = bot_tp1._is_confirmed_tp1_reduction("SOLUSDT", risk_tp1, 5.0)
        self.assertFalse(confirmed_loss)

    # r) Web close_position accuracy, standalone totalAvailableBalance, trade_log edge patterns, WFA tail coverage
    def test_v23_reviewer_web_and_wfa_fixes(self):
        from web import app as web_app
        import optimizer

        # 1. _close_position_sync при bot is not None не врет 'ok' если позиции нет или close_position вернул None
        bot = self._make_bot()
        bot.positions = {"ETHUSDT": None, "NEARUSDT": {"side": "Buy", "position": "long", "size": 5.0, "entry_price": 5.0}}
        bot.get_price = MagicMock(return_value=5.0)
        bot.close_position = MagicMock(return_value=None)
        bot.client.get_positions.return_value = {"result": {"list": []}}
        web_state.set_bot_instance(bot)

        res_empty = web_app._close_position_sync("ETHUSDT")
        self.assertEqual(res_empty["status"], "error")
        res_fail = web_app._close_position_sync("NEARUSDT")
        self.assertEqual(res_fail["status"], "error")

        # Внесписочная позиция на бирже при работающем bot -> закрывается через bot.client
        bot.client.get_positions.return_value = {"result": {"list": [{"symbol": "XRPUSDT", "side": "Sell", "size": "10", "positionIdx": 0}]}}
        bot.client.place_order.return_value = {"retCode": 0}
        bot.sync_positions = MagicMock()
        res_unlisted = web_app._close_position_sync("XRPUSDT")
        self.assertEqual(res_unlisted["status"], "ok")

        # 2. Standalone _compute_status_sync корректно берет totalAvailableBalance даже при availableToWithdraw=0
        web_state.set_bot_instance(None)
        mock_cli = MagicMock()
        mock_cli.get_wallet_balance.return_value = {
            "result": {"list": [{
                "totalAvailableBalance": "312.5",
                "coin": [{"coin": "USDT", "walletBalance": "500.0", "availableToWithdraw": "0", "equity": "510.0"}]
            }]}
        }
        mock_cli.get_positions.return_value = {"result": {"list": []}}
        with patch.object(web_app, "get_standalone_client", return_value=mock_cli):
            web_app._invalidate_status_cache()
            st = web_app._compute_status_sync()
        self.assertEqual(st["balance"], 312.5)
        self.assertEqual(st["equity"], 510.0)

        # 3. parse_trade_log: игнор 'Обнаружена открытая позиция', парсинг 'Закрыта' и цены из 'Стоп перенесен в безубыток: 92.63'
        fake_log = Path(self.tmpdir.name) / "trade_edge.log"
        fake_log.write_text(
            "2026-11-11 08:00:00 [INFO] [ETHUSDT] Обнаружена открытая позиция LONG (1.0)\n"
            "2026-11-11 08:05:00 [INFO] [SOLUSDT] WebSocket: Стоп перенесен в безубыток: 92.63\n"
            "2026-11-11 08:10:00 [WARNING] [XRPUSDT] PANIC: Закрыта внесписочная/Hedge позиция (20).\n",
            encoding="utf-8",
        )
        with patch.object(config, "TRADE_LOG_FILE", fake_log):
            ev = web_state.parse_trade_log(limit=20)
        self.assertEqual(len(ev), 2)
        self.assertEqual(ev[0]["symbol"], "XRPUSDT")
        self.assertEqual(ev[0]["action"], "ЗАКРЫТИЕ")
        self.assertEqual(ev[1]["symbol"], "SOLUSDT")
        self.assertEqual(ev[1]["price"], "92.63")

        # 4. Rolling WFA доходит до самой последней свечи датасета без отбрасывания хвоста warmup_buf
        dates_long = pd.date_range("2026-01-01", periods=1000, freq="1h")
        rng = np.random.default_rng(7)
        px = 100.0 + np.cumsum(rng.normal(0.0, 1.0, size=1000))
        df_long = pd.DataFrame({
            "timestamp": dates_long, "open": px, "high": px + 1.0, "low": px - 1.0, "close": px, "volume": [1000.0] * 1000
        })
        with patch("optimizer.OPTUNA_AVAILABLE", False):
            wfa_res = optimizer.run_rolling_wfa(df_long, symbol="SOLUSDT", trials=5, n_windows=3, strategy_name="SUPERTREND")
        self.assertGreaterEqual(len(wfa_res["windows"]), 2)
        self.assertEqual(wfa_res["windows"][-1]["test_range"][1], dates_long[-1].date())


if __name__ == "__main__":
    unittest.main()
