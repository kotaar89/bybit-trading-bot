"""
Динамический отбор ликвидной корзины торговых пар (Dynamic Universe Selection) v0.3.0.
Ранжирует USDT-перпетуалы Bybit по 24ч обороту, спреду, нормированной волатильности,
экстремальному фандингу и (опционально) трендовому score — снижая Selection Bias.
"""

from __future__ import annotations

import logging
import math
import time
from typing import Optional

import numpy as np
import pandas as pd

log = logging.getLogger("bot")


class DynamicUniverseSelector:
    """Отбирает топ ликвидных и пригодных к торговле USDT-перпетуалов на Bybit."""

    def __init__(
        self,
        top_n: int = 6,
        min_turnover_24h: float = 50_000_000.0,
        max_spread_pct: float = 0.0025,
        min_vol_pct: float = 0.015,
        max_vol_pct: float = 0.18,
        max_abs_funding: float = 0.002,
        blacklist: Optional[list[str]] = None,
    ):
        self.top_n = max(1, int(top_n))
        self.min_turnover_24h = float(min_turnover_24h)
        self.max_spread_pct = float(max_spread_pct)
        self.min_vol_pct = float(min_vol_pct)
        self.max_vol_pct = float(max_vol_pct)
        self.max_abs_funding = float(max_abs_funding)
        self.blacklist = {s.strip().upper() for s in (blacklist or ["BTCUSDT", "USDCUSDT", "FDUSDUSDT", "USDEUSDT"]) if s.strip()}

        self.last_refresh_ts: float = 0.0
        self.last_ranked: list[dict] = []
        self.last_selected: list[str] = []

    @staticmethod
    def _trend_quality_score(df: pd.DataFrame) -> float:
        """Оценка трендовой эффективности (Efficiency Ratio) по последним свечам."""
        if df is None or df.empty or len(df) < 20 or "close" not in df.columns:
            return 0.0
        close = df["close"].astype(float).tail(25)
        net_move = abs(float(close.iloc[-1]) - float(close.iloc[0]))
        total_move = float(close.diff().abs().sum())
        if total_move <= 0 or np.isnan(net_move) or np.isnan(total_move):
            return 0.0
        return float(min(1.0, net_move / total_move))

    def score_tickers(
        self,
        tickers_list: list[dict],
        klines_map: Optional[dict[str, pd.DataFrame]] = None,
    ) -> list[dict]:
        """Оценивает и ранжирует кандидатов из ответа Bybit get_tickers(category='linear')."""
        candidates: list[dict] = []
        for item in tickers_list or []:
            sym = str(item.get("symbol") or "").upper()
            if not sym.endswith("USDT") or sym in self.blacklist:
                continue

            try:
                last_p = float(item.get("lastPrice") or 0.0)
                bid1 = float(item.get("bid1Price") or 0.0)
                ask1 = float(item.get("ask1Price") or 0.0)
                turnover = float(item.get("turnover24h") or 0.0)
                high24 = float(item.get("highPrice24h") or 0.0)
                low24 = float(item.get("lowPrice24h") or 0.0)
                funding = float(item.get("fundingRate") or 0.0)
            except (TypeError, ValueError):
                continue

            if last_p <= 0 or bid1 <= 0 or ask1 <= 0 or ask1 < bid1:
                continue
            if turnover < self.min_turnover_24h:
                continue

            spread_pct = (ask1 - bid1) / bid1
            if spread_pct > self.max_spread_pct:
                continue

            if high24 > 0 and low24 > 0 and high24 >= low24:
                vol_pct = (high24 - low24) / last_p
            else:
                vol_pct = 0.0
            if vol_pct < self.min_vol_pct or vol_pct > self.max_vol_pct:
                continue

            if abs(funding) >= self.max_abs_funding:
                continue

            # Композитный score: ликвидность + умеренная волатильность + узкий спред + качество тренда
            liq_score = math.log1p(turnover / 1_000_000.0)
            spread_penalty = spread_pct * 500.0
            # Пик волатильностного бонуса около 4-8% суточного диапазона
            vol_target_bonus = max(0.0, 1.5 - abs(vol_pct - 0.05) * 15.0)
            trend_bonus = 0.0
            if klines_map and sym in klines_map:
                trend_bonus = self._trend_quality_score(klines_map[sym]) * 2.0

            total_score = liq_score + vol_target_bonus + trend_bonus - spread_penalty
            candidates.append(
                {
                    "symbol": sym,
                    "score": round(total_score, 3),
                    "turnover24h": round(turnover, 0),
                    "spread_pct": round(spread_pct * 100.0, 4),
                    "vol_24h_pct": round(vol_pct * 100.0, 2),
                    "funding_rate": funding,
                    "last_price": last_p,
                }
            )

        candidates.sort(key=lambda x: x["score"], reverse=True)
        return candidates

    def select_symbols(
        self,
        client,
        fallback_symbols: list[str],
        klines_map: Optional[dict[str, pd.DataFrame]] = None,
        pinned_symbols: Optional[list[str]] = None,
        api_call_fn=None,
    ) -> list[str]:
        """
        Загружает тикеры с Bybit и возвращает топ-N символов.
        При сбое или пустой выборке безопасно возвращает fallback_symbols.
        """
        try:
            resp = (
                api_call_fn(client.get_tickers, category="linear")
                if callable(api_call_fn)
                else client.get_tickers(category="linear")
            )
            if not isinstance(resp, dict) or resp.get("retCode", 0) != 0:
                raise RuntimeError(f"retCode={resp.get('retCode') if isinstance(resp, dict) else 'invalid'}")
            raw_list = resp.get("result", {}).get("list", [])
            ranked = self.score_tickers(raw_list, klines_map=klines_map)
            self.last_ranked = ranked
            self.last_refresh_ts = time.time()

            if not ranked:
                log.warning("DynamicUniverse: ни одна пара не прошла фильтры; сохранена текущая корзина.")
                return list(fallback_symbols)

            selected = [c["symbol"] for c in ranked[: self.top_n]]
            # Обязательно сохраняем символы с открытыми позициями (pinned)
            for p_sym in (pinned_symbols or []):
                ps = str(p_sym).upper()
                if ps and ps not in selected:
                    selected.append(ps)

            self.last_selected = list(selected)
            log.info(f"DynamicUniverse обновлена ({len(selected)} пар): {', '.join(selected)}")
            return selected
        except Exception as e:
            log.warning(f"DynamicUniverse ошибка отбора ({e}); сохранена текущая корзина.")
            return list(fallback_symbols)
