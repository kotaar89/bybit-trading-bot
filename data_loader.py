"""
Модуль загрузки исторических данных с Bybit
Использование:
    python data_loader.py                  # скачать по настройкам из config.py
    python data_loader.py ETHUSDT 60 90    # символ, таймфрейм, дней
"""

import time
import sys
import os
import pandas as pd
from datetime import datetime, timedelta, timezone
from pybit.unified_trading import HTTP
from config import API_KEY, API_SECRET, TESTNET, SYMBOL, INTERVAL
try:
    from config import DEMO
except ImportError:
    DEMO = False


# ─── Маппинг таймфреймов ─────────────────────────────────────
INTERVAL_MINUTES = {
    "1": 1, "3": 3, "5": 5, "15": 15, "30": 30,
    "60": 60, "120": 120, "240": 240, "360": 360,
    "720": 720, "D": 1440, "W": 10080, "M": 43200
}


def download_klines(
    symbol: str = None,
    interval: str = None,
    days: int = 365,
    save_csv: bool = True,
    show_progress: bool = True
) -> pd.DataFrame:
    """
    Скачивает исторические свечи с Bybit.

    Параметры:
        symbol   — торговая пара (например, 'BTCUSDT')
        interval — таймфрейм ('1','5','15','60','240','D' и т.д.)
        days     — сколько дней истории скачать
        save_csv — сохранить ли результат в CSV
        show_progress — показывать прогресс

    Возвращает: pd.DataFrame с колонками:
        timestamp, open, high, low, close, volume, turnover
    """
    symbol   = symbol   or SYMBOL
    interval = interval or INTERVAL

    client = HTTP(
        testnet=TESTNET,
        api_key=API_KEY,
        api_secret=API_SECRET,
        demo=DEMO,
    )

    tf_minutes = INTERVAL_MINUTES.get(str(interval), 15)
    candles_needed = int(days * 1440 / tf_minutes)
    batch_size = 1000

    if show_progress:
        print(f"\nЗагрузка данных с Bybit")
        print(f"   Пара:       {symbol}")
        print(f"   Таймфрейм:  {interval}m")
        print(f"   Период:     {days} дней")
        print(f"   Свечей:     ~{candles_needed}")
        print(f"   Запросов:   ~{(candles_needed // batch_size) + 1}")
        print()

    all_data = []
    end_time = None
    total_fetched = 0
    request_count = 0
    consecutive_errors = 0
    max_retries = 5

    while total_fetched < candles_needed:
        params = dict(
            category="linear",
            symbol=symbol,
            interval=str(interval),
            limit=batch_size
        )
        if end_time:
            params["end"] = end_time

        try:
            resp = client.get_kline(**params)
            if resp.get("retCode", 0) != 0:
                raise RuntimeError(f"Bybit retCode={resp.get('retCode')}: {resp.get('retMsg')}")
            consecutive_errors = 0
        except Exception as e:
            consecutive_errors += 1
            print(f"Ошибка запроса ({consecutive_errors}/{max_retries}): {e}")
            if consecutive_errors >= max_retries:
                print("Превышен лимит попыток загрузки данных с биржи.")
                break
            time.sleep(2)
            continue

        batch = resp.get("result", {}).get("list", [])
        if not batch:
            break

        all_data.extend(batch)
        total_fetched += len(batch)
        request_count += 1

        # Следующий батч — раньше самой старой свечи
        oldest_ts = int(batch[-1][0])
        end_time = oldest_ts - 1

        if show_progress:
            dt = datetime.fromtimestamp(oldest_ts / 1000).strftime("%Y-%m-%d")
            bar_len = min(30, total_fetched * 30 // candles_needed)
            bar = "=" * bar_len + "-" * (30 - bar_len)
            pct = min(100, total_fetched * 100 // candles_needed)
            print(f"\r   [{bar}] {pct}% | {total_fetched} свечей | до {dt}", end="", flush=True)

        # Защита от rate limit
        time.sleep(0.25)

    if show_progress:
        print(f"\r   [{'='*30}] 100% | {total_fetched} свечей загружено      ")
        print()

    if not all_data:
        print("Данные не получены")
        return pd.DataFrame()

    # ─── Формируем DataFrame ─────────────────────────────────
    df = pd.DataFrame(all_data, columns=[
        "timestamp", "open", "high", "low", "close", "volume", "turnover"
    ])

    for col in ["open", "high", "low", "close", "volume", "turnover"]:
        df[col] = df[col].astype(float)

    df["timestamp"] = pd.to_datetime(df["timestamp"].astype(int), unit="ms")
    df = df.drop_duplicates("timestamp").sort_values("timestamp").reset_index(drop=True)

    # Отрезаем ровно нужный период
    cutoff = datetime.now(timezone.utc).replace(tzinfo=None) - timedelta(days=days)
    df = df[df["timestamp"] >= pd.Timestamp(cutoff)].reset_index(drop=True)

    # ─── Сохраняем CSV ───────────────────────────────────────
    if save_csv:
        os.makedirs("data", exist_ok=True)
        filename = f"data/{symbol}_{interval}m_{days}d.csv"
        df.to_csv(filename, index=False)
        if show_progress:
            size_kb = os.path.getsize(filename) / 1024
            print(f"Сохранено: {filename} ({size_kb:.1f} KB)")

    if show_progress:
        print(f"Итого свечей: {len(df)}")
        print(f"   С:  {df['timestamp'].iloc[0]}")
        print(f"   По: {df['timestamp'].iloc[-1]}")
        print()

    return df


def load_or_download(
    symbol: str = None,
    interval: str = None,
    days: int = 365,
    force_download: bool = False
) -> pd.DataFrame:
    """
    Загружает данные из кэша (CSV) если он есть и свежий,
    иначе скачивает с Bybit заново.

    force_download=True — всегда скачивать заново
    """
    symbol   = symbol   or SYMBOL
    interval = interval or INTERVAL

    filename = f"data/{symbol}_{interval}m_{days}d.csv"

    if not force_download and os.path.exists(filename):
        # Проверяем свежесть файла
        mtime = datetime.fromtimestamp(os.path.getmtime(filename))
        age_hours = (datetime.now() - mtime).total_seconds() / 3600
        tf_minutes = INTERVAL_MINUTES.get(str(interval), 15)
        max_age_hours = max(1, tf_minutes / 60 * 2)  # 2 свечи устарело — перекачать

        if age_hours < max_age_hours:
            print(f"Загружаем из кэша: {filename} (обновлен {age_hours:.1f}ч назад)")
            df = pd.read_csv(filename, parse_dates=["timestamp"])
            print(f"Загружено {len(df)} свечей из кэша\n")
            return df
        else:
            print(f"Кэш устарел ({age_hours:.1f}ч), скачиваем заново...\n")

    df = download_klines(symbol, interval, days)
    if df.empty and os.path.exists(filename):
        print(f"Внимание: Не удалось обновить данные с биржи. Используем имеющийся кэш: {filename}")
        return pd.read_csv(filename, parse_dates=["timestamp"])
    return df


# ─── CLI запуск ──────────────────────────────────────────────
if __name__ == "__main__":
    args = sys.argv[1:]

    sym  = args[0] if len(args) > 0 else SYMBOL
    tf   = args[1] if len(args) > 1 else INTERVAL
    days = int(args[2]) if len(args) > 2 else 90

    df = download_klines(sym, tf, days)

    if not df.empty:
        print("\nПревью данных:")
        print(df.tail(5).to_string(index=False))
