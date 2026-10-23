"""
Фабрика и реестр торговых стратегий v2.2.
Позволяет легко выбирать и инстанцировать стратегию по имени из config.py или дашборда.
"""

from typing import Type, Dict
from strategies.base import BaseStrategy
from strategies.ma_rsi_strategy import MaRsiStrategy
from strategies.bb_rsi_reversion import BbRsiReversionStrategy
from strategies.supertrend_atr import SupertrendAtrStrategy, SupertrendProStrategy
from strategies.donchian_breakout import DonchianBreakoutStrategy

STRATEGY_REGISTRY: Dict[str, Type[BaseStrategy]] = {
    "SUPERTREND_PRO": SupertrendProStrategy,
    "MA_RSI": MaRsiStrategy,
    "DEFAULT": SupertrendProStrategy,
    "BB_RSI": BbRsiReversionStrategy,
    "SUPERTREND": SupertrendAtrStrategy,
    "DONCHIAN": DonchianBreakoutStrategy,
}


def register_strategy(name: str, strategy_cls: Type[BaseStrategy]):
    """Регистрирует новую пользовательскую стратегию в реестре."""
    STRATEGY_REGISTRY[name.upper()] = strategy_cls


def get_strategy(name: str = "SUPERTREND_PRO", **kwargs) -> BaseStrategy:
    """
    Фабричный метод создания экземпляра стратегии по ее имени.
    :param name: Название стратегии (например, 'SUPERTREND_PRO', 'MA_RSI', 'BB_RSI', 'SUPERTREND', 'DONCHIAN')
    :param kwargs: Дополнительные параметры для переопределения стандартных
    :return: Экземпляр BaseStrategy
    """
    key = (name or "SUPERTREND_PRO").strip().upper()
    cls = STRATEGY_REGISTRY.get(key)
    if cls is None:
        raise ValueError(
            f"Неизвестная стратегия: '{name}'. "
            f"Доступные стратегии: {list(STRATEGY_REGISTRY.keys())}"
        )
    return cls(**kwargs)


__all__ = [
    "BaseStrategy",
    "MaRsiStrategy",
    "BbRsiReversionStrategy",
    "SupertrendAtrStrategy",
    "SupertrendProStrategy",
    "DonchianBreakoutStrategy",
    "STRATEGY_REGISTRY",
    "register_strategy",
    "get_strategy",
]
