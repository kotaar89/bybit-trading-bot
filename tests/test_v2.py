"""
Набор тестов для проверки функционала версии 2.1:
- Секция 1: Критические ошибки и логические баги (Lookahead Bias, min_notional, pd.merge_asof, atomic state, mock standalone client)
- Секция 2: Торговая механика и исполнение (Limit TP1, нативный Trailing Stop, приватный WS, Maker Post-Only)
- Секция 3: Стратегии и фильтры (ADX flat filter, BB_RSI, SUPERTREND, DONCHIAN)
- Секция 4: Веб-дашборд и UI (WebSocket endpoint, candles, equity curve, password for LIVE)
"""

import sys
import os
import threading
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

    def tearDown(self):
        set_bot_instance(None)

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
        bot.state_file = Path("test_bot_state.json")
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
        bot.state_file = Path("test_ws_state.json")
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


if __name__ == "__main__":
    unittest.main()
