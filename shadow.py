"""
Shadow / Paper Mode (v0.3.0):
Симуляция торговли в реальном времени на живом потоке цен и свечей Bybit
без отправки ордеров на биржу. Поддерживает:
- Полный Paper-режим (SHADOW_MODE=true) вместо реальных ордеров
- Параллельный A/B тест альтернативной стратегии (SHADOW_STRATEGY=...)
- Учет комиссий Maker/Taker, спреда, TP1, безубытка, трейлинга, Time-Stop и запись в SQLite
"""

from __future__ import annotations

import logging
import threading
import time
from typing import Optional

import config
from risk_manager import PortfolioRiskManager, PositionRiskState
from strategies import get_strategy

log = logging.getLogger("bot")
trade_log = logging.getLogger("trade")

TAKER_FEE = 0.00055
MAKER_FEE = 0.00020


class ShadowEngine:
    """Движок теневой (Paper) торговли на реальном рыночном потоке Bybit."""

    def __init__(
        self,
        db=None,
        initial_equity: float = 1000.0,
        strategy_name: Optional[str] = None,
        leverage: int = 5,
    ):
        self.lock = threading.RLock()
        self.db = db
        self.initial_equity = float(initial_equity if initial_equity > 0 else config.SHADOW_INITIAL_BALANCE)
        self.equity = self.initial_equity
        self.leverage = int(leverage or config.LEVERAGE)
        self.strategy_name = (strategy_name or config.SHADOW_STRATEGY or config.ACTIVE_STRATEGY).strip().upper()
        self.strategy = self._build_strategy(self.strategy_name)

        self.positions: dict[str, dict] = {}
        self.risk_states: dict[str, PositionRiskState] = {}
        self.last_handled_candles: dict[str, object] = {}
        self.portfolio_risk = PortfolioRiskManager(
            max_heat_pct=config.MAX_PORTFOLIO_HEAT_PCT,
            use_correlation_scaling=config.USE_CORRELATION_SCALING,
            corr_lookback_bars=config.CORR_LOOKBACK_BARS,
            corr_threshold=config.CORR_THRESHOLD,
            corr_risk_scale=config.CORR_RISK_SCALE,
            use_peak_dd_throttle=config.USE_PEAK_DD_THROTTLE,
            dd_tier1_pct=config.DD_THROTTLE_TIER1_PCT,
            dd_tier1_mult=config.DD_THROTTLE_TIER1_MULT,
            dd_tier2_pct=config.DD_THROTTLE_TIER2_PCT,
            dd_recover_pct=config.DD_THROTTLE_RECOVER_PCT,
            use_loss_cooldown=config.USE_LOSS_STREAK_COOLDOWN,
            symbol_loss_limit=config.LOSS_STREAK_SYMBOL_LIMIT,
            portfolio_loss_limit=config.LOSS_STREAK_PORTFOLIO_LIMIT,
            cooldown_seconds=config.LOSS_STREAK_COOLDOWN_BARS * 3600.0,
            peak_equity=self.equity,
        )
        self._load_persisted_state()

    @staticmethod
    def _build_strategy(name: str):
        strat_use_mtf = config.USE_MTF_FILTER if name != "BB_RSI" else False
        return get_strategy(
            name,
            st_period=config.ST_PERIOD,
            st_multiplier=config.ST_MULTIPLIER,
            confirm_ema_period=config.ST_CONFIRM_EMA,
            confirm_lookback=getattr(config, "ST_CONFIRM_LOOKBACK", 1),
            fast_ma=config.FAST_MA,
            slow_ma=config.SLOW_MA,
            rsi_period=config.RSI_PERIOD,
            rsi_long_min=config.RSI_LONG_MIN,
            rsi_long_max=config.RSI_LONG_MAX,
            rsi_short_min=config.RSI_SHORT_MIN,
            rsi_short_max=config.RSI_SHORT_MAX,
            volume_filter=config.VOLUME_FILTER,
            use_mtf_filter=strat_use_mtf,
            htf_ema_period=config.HTF_EMA_PERIOD,
            atr_period=config.ATR_PERIOD,
            cross_lookback=config.SIGNAL_LOOKBACK_CANDLES,
            use_adx_filter=config.USE_ADX_FILTER,
            adx_period=config.ADX_PERIOD,
            adx_threshold=config.ADX_THRESHOLD,
            adx_trend_min=getattr(config, "SWITCHER_ADX_TREND_MIN", 22.0),
            adx_range_max=getattr(config, "SWITCHER_ADX_RANGE_MAX", 18.0),
            er_period=getattr(config, "SWITCHER_ER_PERIOD", 14),
            er_trend_min=getattr(config, "SWITCHER_ER_TREND_MIN", 0.35),
            er_range_max=getattr(config, "SWITCHER_ER_RANGE_MAX", 0.25),
            bb_pctl_window=getattr(config, "SWITCHER_BB_PCTL_WINDOW", 100),
            mr_sl_mult=getattr(config, "SWITCHER_MR_SL_MULT", 1.8),
            mr_tp_mult=getattr(config, "SWITCHER_MR_TP_MULT", 2.5),
        )

    def _save_persisted_state(self):
        if self.db is None:
            return
        try:
            payload = {
                "equity": self.equity,
                "positions": self.positions,
                "risk_states": {s: r.to_dict() for s, r in self.risk_states.items() if r is not None},
                "portfolio_risk": self.portfolio_risk.to_dict(),
            }
            self.db.set_runtime_value("shadow_engine_state", payload)
        except Exception as e:
            log.debug(f"[SHADOW] Ошибка сохранения состояния: {e}")

    def _load_persisted_state(self):
        if self.db is None:
            return
        try:
            data = self.db.get_runtime_value("shadow_engine_state")
            if not isinstance(data, dict):
                return
            if float(data.get("equity") or 0.0) > 0:
                self.equity = float(data["equity"])
            if isinstance(data.get("positions"), dict):
                self.positions = data["positions"]
            if isinstance(data.get("risk_states"), dict):
                self.risk_states = {
                    s: PositionRiskState.from_dict(r)
                    for s, r in data["risk_states"].items()
                    if isinstance(r, dict)
                }
            if isinstance(data.get("portfolio_risk"), dict):
                self.portfolio_risk = PortfolioRiskManager.from_dict(
                    data["portfolio_risk"],
                    max_heat_pct=config.MAX_PORTFOLIO_HEAT_PCT,
                    use_correlation_scaling=config.USE_CORRELATION_SCALING,
                    corr_lookback_bars=config.CORR_LOOKBACK_BARS,
                    corr_threshold=config.CORR_THRESHOLD,
                    corr_risk_scale=config.CORR_RISK_SCALE,
                    use_peak_dd_throttle=config.USE_PEAK_DD_THROTTLE,
                    dd_tier1_pct=config.DD_THROTTLE_TIER1_PCT,
                    dd_tier1_mult=config.DD_THROTTLE_TIER1_MULT,
                    dd_tier2_pct=config.DD_THROTTLE_TIER2_PCT,
                    dd_recover_pct=config.DD_THROTTLE_RECOVER_PCT,
                    use_loss_cooldown=config.USE_LOSS_STREAK_COOLDOWN,
                    symbol_loss_limit=config.LOSS_STREAK_SYMBOL_LIMIT,
                    portfolio_loss_limit=config.LOSS_STREAK_PORTFOLIO_LIMIT,
                    cooldown_seconds=config.LOSS_STREAK_COOLDOWN_BARS * 3600.0,
                )
        except Exception as e:
            log.debug(f"[SHADOW] Ошибка загрузки состояния: {e}")

    def open_position(
        self,
        symbol: str,
        side: str,
        price: float,
        atr_value: float = 0.0,
        bid: Optional[float] = None,
        ask: Optional[float] = None,
        klines_map: Optional[dict] = None,
        entry_regime: str = "",
        strategy_name: Optional[str] = None,
    ) -> Optional[dict]:
        with self.lock:
            sym = symbol.upper()
            if sym in self.positions and self.positions[sym]:
                return None
            if len(self.positions) >= config.MAX_OPEN_POSITIONS:
                return None

            is_long = side.lower() in ("buy", "long")
            target_dir = "long" if is_long else "short"
            same_dir = sum(1 for p in self.positions.values() if p and p.get("position") == target_dir)
            if same_dir >= config.MAX_SAME_DIRECTION_POSITIONS:
                return None

            # Проверка спреда при наличии bid/ask
            if bid and ask and bid > 0 and ask >= bid:
                spread_pct = (ask - bid) / bid
                if spread_pct > config.MAX_ENTRY_SPREAD_PCT:
                    log.info(f"[SHADOW][{sym}] Пропуск входа: широкий спред {spread_pct*100:.2f}%")
                    return None
                fill_price = float(bid if is_long else ask) if config.USE_SMART_EXECUTION else float(ask if is_long else bid)
                fee_rate = MAKER_FEE if config.USE_SMART_EXECUTION else TAKER_FEE
            else:
                fill_price = float(price)
                fee_rate = MAKER_FEE if config.USE_SMART_EXECUTION else TAKER_FEE

            if fill_price <= 0:
                return None

            stops_kwargs = dict(
                entry_price=fill_price,
                side="Buy" if is_long else "Sell",
                atr_value=atr_value,
                stop_loss_pct=config.STOP_LOSS_PCT,
                take_profit_pct=config.TAKE_PROFIT_PCT,
                atr_sl_mult=config.ATR_SL_MULT,
                atr_tp_mult=config.ATR_TP_MULT,
                use_atr=config.USE_ATR_STOPS,
            )
            try:
                stops = self.strategy.calculate_stops(**stops_kwargs, regime=entry_regime) if entry_regime else self.strategy.calculate_stops(**stops_kwargs)
            except TypeError:
                stops = self.strategy.calculate_stops(**stops_kwargs)

            sl_dist = float(stops.get("sl_dist") or (fill_price * config.STOP_LOSS_PCT))
            if sl_dist <= 0:
                return None

            base_risk_usdt = self.equity * config.RISK_PER_TRADE_PCT
            allowed, adj_risk, reason = self.portfolio_risk.check_entry_budget(
                symbol=sym,
                is_long=is_long,
                proposed_risk_usdt=base_risk_usdt,
                equity=self.equity,
                risk_states=self.risk_states,
                open_positions=self.positions,
                klines_map=klines_map,
            )
            if not allowed or adj_risk <= 0:
                log.info(f"[SHADOW][{sym}] Вход отклонен портфельным риском: {reason}")
                return None

            risk_scale_ratio = min(1.0, adj_risk / base_risk_usdt) if base_risk_usdt > 0 else 1.0
            max_margin = self.equity * config.POSITION_SIZE_PCT * risk_scale_ratio
            max_qty = (max_margin * self.leverage) / fill_price
            if getattr(config, "USE_RISK_BASED_SIZING", True):
                raw_qty = adj_risk / sl_dist
                qty = min(raw_qty, max_qty)
            else:
                qty = max_qty
            notional = qty * fill_price
            if notional < config.MIN_ORDER_NOTIONAL:
                return None

            margin_used = notional / self.leverage
            entry_fee = notional * fee_rate
            now_ts = time.time()
            strat_tag = strategy_name or self.strategy_name
            pos_ref = f"SHADOW_{sym}_{int(now_ts)}"

            pos_dict = {
                "symbol": sym,
                "side": "Buy" if is_long else "Sell",
                "position": target_dir,
                "size": qty,
                "entry_price": fill_price,
                "stop_loss": float(stops["sl"]),
                "take_profit": float(stops["tp"]),
                "margin": margin_used,
                "entry_fee": entry_fee,
                "position_idx": 0,
            }
            risk = PositionRiskState(
                symbol=sym,
                side="Buy" if is_long else "Sell",
                entry_price=fill_price,
                size=qty,
                initial_size=qty,
                stop_loss=float(stops["sl"]),
                take_profit=float(stops["tp"]),
                atr_value=atr_value,
                r_mult=config.PARTIAL_TP_R_MULT,
                opened_at=now_ts,
                strategy_tag=strat_tag,
                entry_regime=entry_regime,
                margin_used=margin_used,
                pos_ref=pos_ref,
            )
            self.positions[sym] = pos_dict
            self.risk_states[sym] = risk

            trade_log.info(
                f"[SHADOW][{sym}] {target_dir.upper()} открыт | Вход: {fill_price:.4f} | "
                f"Кол-во: {qty:.4f} ({margin_used:.1f}$ маржа) | SL: {risk.stop_loss:.4f} | TP: {risk.take_profit:.4f}"
            )
            if self.db is not None:
                self.db.record_trade_event(
                    symbol=sym,
                    action="SHADOW_OPEN",
                    side=pos_dict["side"],
                    strategy=strat_tag,
                    price=fill_price,
                    qty=qty,
                    fee_usdt=entry_fee,
                    exec_type="ShadowMaker" if config.USE_SMART_EXECUTION else "ShadowMarket",
                    pos_ref=pos_ref,
                    is_profit=None,
                    is_shadow=True,
                    details=f"[SHADOW] {target_dir.upper()} SL={risk.stop_loss:.4f} TP={risk.take_profit:.4f} ({reason})",
                    ts=now_ts,
                )
            self._save_persisted_state()
            return pos_dict

    def _finalize_close(self, sym: str, exit_price: float, reason: str, fee_rate: float = TAKER_FEE):
        pos = self.positions.get(sym)
        risk = self.risk_states.get(sym)
        if not pos or not risk:
            return None

        entry_p = float(risk.entry_price)
        rem_qty = float(risk.size)
        is_long = risk.is_long
        diff = (exit_price - entry_p) if is_long else (entry_p - exit_price)
        gross_usdt = diff * rem_qty
        exit_fee = abs(exit_price * rem_qty) * fee_rate
        # Добавляем пропорциональную долю входной комиссии
        entry_fee_share = float(pos.get("entry_fee") or 0.0) * (rem_qty / max(1e-12, risk.initial_size))
        net_usdt = gross_usdt - exit_fee - entry_fee_share
        total_leg_fee = exit_fee + entry_fee_share

        self.equity = max(1.0, self.equity + net_usdt)
        margin_base = float(risk.margin_used or pos.get("margin") or 1.0)
        leg_pnl_pct = (net_usdt / margin_base) * 100.0 if margin_base > 0 else 0.0
        # Общий PnL с учетом TP1
        total_pnl_usdt = net_usdt + float(pos.get("tp1_realized_usdt") or 0.0)
        total_pnl_pct = (total_pnl_usdt / margin_base) * 100.0 if margin_base > 0 else leg_pnl_pct

        now_ts = time.time()
        self.portfolio_risk.record_closed_trade(sym, total_pnl_usdt, now_ts=now_ts)
        self.portfolio_risk.update_equity(self.equity, valid=True)

        sign = "+" if total_pnl_usdt > 0 else ""
        trade_log.info(
            f"[SHADOW][{sym}] Закрыто ({reason}) @ {exit_price:.4f} | "
            f"Net PnL: {sign}{total_pnl_usdt:.2f}$ ({sign}{total_pnl_pct:.2f}%)"
        )
        if self.db is not None:
            self.db.record_trade_event(
                symbol=sym,
                action="SHADOW_CLOSE",
                side="Sell" if is_long else "Buy",
                strategy=risk.strategy_tag or self.strategy_name,
                price=exit_price,
                qty=rem_qty,
                pnl_pct=total_pnl_pct,
                net_pnl_usdt=net_usdt,
                fee_usdt=total_leg_fee,
                exec_type="Shadow",
                pos_ref=risk.pos_ref,
                is_profit=(total_pnl_usdt > 0),
                is_shadow=True,
                details=f"[SHADOW] {reason} @ {exit_price:.4f} | Net: {sign}{total_pnl_usdt:.2f}$",
                ts=now_ts,
            )
            self.db.record_equity_snapshot(
                equity=self.equity,
                balance=self.equity,
                unrealized_pnl=0.0,
                source="shadow_close",
                is_shadow=True,
                ts=int(now_ts),
            )

        self.positions.pop(sym, None)
        self.risk_states.pop(sym, None)
        self._save_persisted_state()
        return {"symbol": sym, "exit_price": exit_price, "net_pnl_usdt": total_pnl_usdt, "reason": reason}

    def close_position(self, symbol: str, price: float, reason: str = "Сигнал"):
        with self.lock:
            sym = symbol.upper()
            if sym not in self.positions or not self.positions[sym]:
                return None
            return self._finalize_close(sym, float(price), reason, fee_rate=TAKER_FEE)

    def on_tick(self, symbol: str, price: float, high: Optional[float] = None, low: Optional[float] = None):
        """Проверяет SL, TP1, BE, Trailing, TP2 и Time-Stop на реальном тике без биржевых ордеров."""
        with self.lock:
            sym = symbol.upper()
            pos = self.positions.get(sym)
            risk = self.risk_states.get(sym)
            if not pos or not risk or price <= 0:
                return None

            h = float(high if high is not None else price)
            l = float(low if low is not None else price)
            risk.update_price_extremes(h, l)

            # 1. Stop Loss
            if (risk.is_long and l <= risk.stop_loss) or (not risk.is_long and h >= risk.stop_loss):
                reason = "Trailing SL" if risk.trailing_active else ("Безубыток" if risk.breakeven_applied else "SL")
                return self._finalize_close(sym, risk.stop_loss, reason, fee_rate=TAKER_FEE)

            # 2. Partial TP1 (Maker limit)
            if config.USE_PARTIAL_TP and not risk.tp1_executed:
                trig, close_qty, be_sl = risk.check_partial_tp(
                    h if risk.is_long else l,
                    ratio=config.PARTIAL_TP_RATIO,
                    offset_pct=config.BREAKEVEN_OFFSET,
                )
                if trig and close_qty > 0:
                    tp1_p = float(risk.tp1_price)
                    diff = (tp1_p - risk.entry_price) if risk.is_long else (risk.entry_price - tp1_p)
                    gross = diff * close_qty
                    tp1_fee = (tp1_p * close_qty) * (MAKER_FEE if config.USE_EXCHANGE_TP1 else TAKER_FEE)
                    entry_fee_share = float(pos.get("entry_fee") or 0.0) * (close_qty / max(1e-12, risk.initial_size))
                    net_tp1 = gross - tp1_fee - entry_fee_share
                    self.equity = max(1.0, self.equity + net_tp1)
                    pos["tp1_realized_usdt"] = float(pos.get("tp1_realized_usdt") or 0.0) + net_tp1
                    risk.tp1_executed = True
                    risk.tp1_closed_qty = close_qty
                    risk.tp1_fill_price = tp1_p
                    risk.size = max(0.0, risk.size - close_qty)
                    pos["size"] = risk.size
                    if config.PARTIAL_TP_MOVE_BE:
                        if (risk.is_long and be_sl > risk.stop_loss) or (not risk.is_long and be_sl < risk.stop_loss):
                            risk.stop_loss = be_sl
                            pos["stop_loss"] = be_sl
                        risk.breakeven_applied = True
                    if self.db is not None:
                        self.db.record_trade_event(
                            symbol=sym,
                            action="ЧАСТИЧНЫЙ TP",
                            side="Sell" if risk.is_long else "Buy",
                            strategy=risk.strategy_tag or self.strategy_name,
                            price=tp1_p,
                            qty=close_qty,
                            net_pnl_usdt=net_tp1,
                            fee_usdt=tp1_fee + entry_fee_share,
                            exec_type="ShadowMakerTP1",
                            pos_ref=risk.pos_ref,
                            is_profit=(net_tp1 > 0),
                            is_shadow=True,
                            details=f"[SHADOW] TP1 @ {tp1_p:.4f} | Net: {net_tp1:+.2f}$",
                        )
                    self._save_persisted_state()

            # 3. Take Profit (TP2)
            if (risk.is_long and h >= risk.take_profit) or (not risk.is_long and l <= risk.take_profit):
                reason = "TP2" if risk.tp1_executed else "TP"
                return self._finalize_close(sym, risk.take_profit, reason, fee_rate=MAKER_FEE)

            # 4. Time-Stop
            if config.USE_TIME_STOP:
                ts_act, new_sl, ts_reason = risk.check_time_stop(
                    current_price=price,
                    now_ts=time.time(),
                    max_hours=config.TIME_STOP_HOURS,
                    hard_hours=config.TIME_STOP_HARD_HOURS,
                    min_r_progress=config.TIME_STOP_MIN_R,
                    offset_pct=config.BREAKEVEN_OFFSET,
                )
                if ts_act == "close":
                    return self._finalize_close(sym, price, ts_reason, fee_rate=TAKER_FEE)
                elif ts_act == "tighten_be":
                    risk.stop_loss = new_sl
                    pos["stop_loss"] = new_sl
                    risk.breakeven_applied = True
                    risk.time_stop_tightened = True
                    self._save_persisted_state()

            # 5. Breakeven & Trailing
            if config.USE_BREAKEVEN and not risk.breakeven_applied:
                be_upd, be_sl = risk.check_breakeven(
                    price, config.BREAKEVEN_TRIGGER, config.BREAKEVEN_OFFSET, config.BREAKEVEN_R_MULT
                )
                if be_upd:
                    risk.stop_loss = be_sl
                    pos["stop_loss"] = be_sl
                    risk.breakeven_applied = True
                    self._save_persisted_state()

            if config.USE_TRAILING_STOP and getattr(risk, "entry_regime", "") != "RANGE":
                tr_upd, tr_sl = risk.check_trailing_stop(
                    price, h, l,
                    config.TRAILING_STOP_TRIGGER,
                    config.TRAILING_STOP_ATR_MULT,
                    config.TRAILING_STOP_FALLBACK,
                    config.TRAILING_STOP_R_MULT,
                )
                if tr_upd:
                    risk.stop_loss = tr_sl
                    pos["stop_loss"] = tr_sl
                    self._save_persisted_state()
        return None

    def process_candle_signal(
        self,
        symbol: str,
        df,
        price: float,
        htf_df=None,
        klines_map: Optional[dict] = None,
        regime_allowed_fn=None,
        candle_idx: int = -2,
        htf_candle_idx: int = -2,
    ):
        """Оценивает сигнал параллельной shadow-стратегии на закрытии свечи."""
        if df is None or df.empty or len(df) < 35:
            return
        sym = symbol.upper()
        calc_df = self.strategy.calculate_indicators(df.copy())
        idx = candle_idx if abs(candle_idx) <= len(calc_df) else (-2 if len(calc_df) >= 2 else -1)
        if len(calc_df) < abs(idx):
            return
        sig_candle = calc_df.iloc[idx]
        sig_ts = sig_candle.get("timestamp")
        if sig_ts == self.last_handled_candles.get(sym):
            return
        self.last_handled_candles[sym] = sig_ts

        sig = int(sig_candle.get("signal", 0))
        exit_sig = int(sig_candle.get("exit_signal", sig))
        atr = float(sig_candle.get("ATR", 0.0) or 0.0)
        entry_regime = str(sig_candle.get("SIGNAL_REGIME", ""))

        cur_pos = self.positions.get(sym)
        cur_risk = self.risk_states.get(sym)
        if cur_pos and cur_risk:
            if (exit_sig == 1 and not cur_risk.is_long) or (exit_sig == -1 and cur_risk.is_long):
                self.close_position(sym, price, reason="Смена тренда (Shadow)")
                cur_pos = None
            elif cur_risk.entry_regime == "RANGE":
                if (cur_risk.is_long and bool(sig_candle.get("mr_exit_long", False))) or (
                    not cur_risk.is_long and bool(sig_candle.get("mr_exit_short", False))
                ):
                    self.close_position(sym, price, reason="Mean-Reversion BB_middle (Shadow)")
                    cur_pos = None

        if sig != 0 and not cur_pos:
            if callable(regime_allowed_fn):
                ok, _ = regime_allowed_fn(sym, sig)
                if not ok:
                    return
            if getattr(self.strategy, "use_mtf_filter", False) and htf_df is not None and entry_regime != "RANGE":
                try:
                    htf_trend = self.strategy.get_mtf_trend(htf_df, candle_idx=htf_candle_idx)
                except TypeError:
                    htf_trend = self.strategy.get_mtf_trend(htf_df)
                if (sig == 1 and htf_trend == "bearish") or (sig == -1 and htf_trend == "bullish"):
                    return
            self.open_position(
                symbol=sym,
                side="Buy" if sig == 1 else "Sell",
                price=price,
                atr_value=atr,
                klines_map=klines_map,
                entry_regime=entry_regime,
                strategy_name=self.strategy_name,
            )

    def get_status(self) -> dict:
        with self.lock:
            return {
                "strategy": self.strategy_name,
                "equity": round(self.equity, 2),
                "initial_equity": round(self.initial_equity, 2),
                "open_positions_count": len(self.positions),
                "positions": list(self.positions.values()),
                "peak_equity": round(self.portfolio_risk.peak_equity, 2),
            }
