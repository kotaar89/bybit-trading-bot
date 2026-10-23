# Bybit Futures Multi-Pair Bot v2.2 (Termux & Server Ready)

Алготрейдинг-бот для линейных фьючерсов **Bybit (USDT Perpetual)** с модульной архитектурой стратегий (`SUPERTREND_PRO`, `MA_RSI`, `BB_RSI`, `DONCHIAN`), адаптивным риск-менеджментом на базе ATR, real-time WebSocket-потоками (`<50ms`), встроенным веб-дашбордом на FastAPI с графиками TradingView и модулем Walk-Forward оптимизации на базе Optuna.

---

## Ключевые возможности v2.2

### 1. Торговые стратегии и фильтрация
* **`SUPERTREND_PRO` (основная стратегия по умолчанию):**
  * Вход по пробою волатильности `SuperTrend (10, 3.0)` с подтверждением объема (`VOL_SMA 20`).
  * **Фильтр силы тренда `ADX(14) >= 18.0`** — отсекает входы в боковике и консолидации.
  * **Подтверждение тренда `EMA(50)` и защита от пиков `RSI(14)`** (`Close >= EMA50` и `RSI <= 72` для Long; `Close <= EMA50` и `RSI >= 28` для Short).
* **Дополнительные встроенные стратегии:** `MA_RSI` (Golden/Death Cross + RSI + ADX с защитой от дублирования сигналов), `BB_RSI` (возврат к среднему по полосам Боллинджера), `SUPERTREND` (классический), `DONCHIAN` (пробой канала Дончиана).
* **Старший тренд MTF (`4H EMA 200`):** Бот загружает 600 свечей старшего таймфрейма (4H) для точной сходимости `EMA(200)` и блокирует сделки против глобального тренда.
* **Фильтр рыночного сентимента по бенчмарку (`Market Regime Filter`):** Анализирует `BTCUSDT` на `4H` (`EMA 200`, `RSI` и резкие дампы/пампы), защищая альткоины от входов против импульса биткоина.

### 2. Продвинутый риск-менеджмент и исполнение
* **Риск-ориентированный расчет позиции:** Автоматический расчет объема от риска на сделку (`RISK_PER_TRADE_PCT = 1.5%` от депозита) с проверкой минимального нотионала Bybit (`>= 5 USDT`) и точным `Decimal`-округлением лотов/цен.
* **Адаптивные стопы по волатильности:** `Stop-Loss = 2.2 * ATR`, `Take-Profit = 4.5 * ATR`.
* **Частичная фиксация прибыли (`TP1 = 1.5R`):** Выставление лимитного `reduceOnly` ордера на бирже на закрытие `50%` позиции при достижении `1.5R` с автоматическим переносом стопа в безубыток (`+0.2%`).
* **Двойной Trailing Stop:** Нативный серверный трейлинг Bybit (`trailingStop` + `activePrice`) + программный трейлинг по `ATR`.
* **Умное исполнение (`Smart Execution`):** Попытка входа лимитным ордером `PostOnly` (Maker) для экономии на комиссиях с автоматическим фоллбэком на `Market` по таймауту.
* **Защитный стоп дня (`Daily Circuit Breaker`):** Остановка торговли и закрытие позиций при достижении дневного лимита просадки (`5%`) с защитой от ложных срабатываний при сетевых таймаутах API.
* **Атомарное сохранение состояния (`bot_state.json`):** Безопасная запись через `.tmp`-файл позволяет боту переживать перезапуски без потери состояния `TP1`, безубытка и трейлинга.

### 3. Веб-дашборд (FastAPI + WebSocket + TradingView Charts)
* Живое обновление метрик по WebSocket (`/ws`) без нагрузки на процессор и батарею в Termux.
* Интерактивные графики свечей (**TradingView Lightweight Charts**) с линиями `Entry`, `SL`, `TP` и кривой капитала (`Equity Curve`).
* Динамический список торгуемых пар из `SYMBOLS`, переключение режимов `DEMO / LIVE` (с защитой паролем), кнопки `Пауза`, закрытия отдельных позиций и экстренного закрытия `PANIC CLOSE`.

---

## Структура проекта

```text
bybit-trading-bot/
├── bot.py                        # Главный торговый движок бота v2.2 (REST + Public/Private WS)
├── config.py                     # Конфигурация и загрузка настроек из .env
├── .env.example                  # Шаблон переменных окружения и ключей API
├── risk_manager.py               # Трейлинг-стоп, Partial TP, Breakeven и Market Regime Filter
├── backtest.py                   # Бэктестер v2.2 (с учетом комиссий, фандинга и проскальзывания)
├── optimizer.py                  # Оптимизатор гиперпараметров (Optuna + Walk-Forward Analysis)
├── data_loader.py                # Загрузчик и кэширование исторических свечей Bybit API v5
├── dashboard.py                  # Автономный запуск веб-дашборда
├── requirements.txt              # Зависимости Python
│
├── strategies/                   # Модульные торговые стратегии
│   ├── __init__.py               # Фабрика и реестр стратегий (get_strategy)
│   ├── base.py                   # Базовый класс BaseStrategy (индикаторы, ATR-стопы, MTF)
│   ├── supertrend_atr.py         # Стратегии SUPERTREND_PRO и SUPERTREND
│   ├── ma_rsi_strategy.py        # Стратегия MA_RSI (Golden/Death Cross + RSI + ADX)
│   ├── bb_rsi_reversion.py       # Стратегия BB_RSI (Bollinger Bands + RSI Mean Reversion)
│   └── donchian_breakout.py      # Стратегия DONCHIAN (пробой канала Дончиана)
│
├── tests/                        # Модульные и интеграционные автотесты
│   └── test_v2.py                # 21 тест архитектуры, рисков, WebSocket, стратегий и UI
│
└── web/                          # Веб-дашборд (FastAPI)
    ├── app.py                    # REST API, WebSocket /ws и Basic Auth
    ├── state.py                  # Потокобезопасное состояние и парсер trade.log
    └── templates/
        └── index.html            # Адаптивный UI дашборда (Tailwind CSS + Lightweight Charts)
```

---

## Быстрый старт

### 1. Установка зависимостей
```bash
pip install -r requirements.txt
```

### 2. Настройка переменных окружения (`.env`)
Скопируйте шаблон `.env.example` в `.env`:
```bash
# Windows (PowerShell / CMD)
copy .env.example .env

# Linux / macOS / Termux
cp .env.example .env
```

Укажите ваши API-ключи Bybit и желаемые настройки в `.env`:
```env
TRADING_MODE=DEMO
DEMO_API_KEY=your_demo_key
DEMO_API_SECRET=your_demo_secret
LIVE_API_KEY=your_live_key
LIVE_API_SECRET=your_live_secret

DASHBOARD_HOST=0.0.0.0
DASHBOARD_PORT=8000
DASHBOARD_USERNAME=admin
DASHBOARD_PASSWORD=change_this_password

ACTIVE_STRATEGY=SUPERTREND_PRO
# Опционально: свой список торгуемых пар (по умолчанию: ETHUSDT,SOLUSDT,AVAXUSDT,NEARUSDT)
# SYMBOLS=ETHUSDT,SOLUSDT,AVAXUSDT,NEARUSDT,INJUSDT,LTCUSDT
```

### 3. Запуск автотестов (рекомендуется перед стартом)
```bash
python -m unittest discover -s tests -v
```

---

## Запуск системы

### Запуск торгового бота со встроенным веб-дашбордом:
```bash
python bot.py
```
> Бот синхронизирует время с сервером Bybit, настроит плечо и режим позиций (`One-Way` / `Hedge Mode`), запустит публичный и приватный WebSocket-потоки и поднимет веб-дашборд на `http://0.0.0.0:8000`.

### Запуск автономного веб-дашборда (отдельно от торгового цикла):
```bash
python dashboard.py
```

---

## Бэктестинг и оптимизация

### Запуск бэктеста (`backtest.py`):
Бэктестер учитывает комиссии тейкера (`0.055%`), динамическое проскальзывание от `ATR`, списания фандинга каждые 8 часов (`00:00, 08:00, 16:00 UTC`), частичный тейк-профит `TP1 (1.5R)`, безубыток и трейлинг-стоп:

```bash
# Сводный бэктест по всей корзине пар из config.py за 90 дней
python backtest.py all 90

# Бэктест по конкретной паре за 90 дней
python backtest.py SOLUSDT 90

# Принудительно перекачать свежие свечи с биржи
python backtest.py ETHUSDT 90 --force
```

### Результаты базовой корзины `v2.2` за 90 дней (`SUPERTREND_PRO`, плечо `x5`):
```text
============================================================================
  СВОДНЫЙ РЕЗУЛЬТАТ ПО КОРЗИНЕ ИЗ 4 ПАР (90 ДНЕЙ) [v2.2]
============================================================================
Пара      Сделок  WinRate   PF      PnL%        MaxDD%    Sharpe  
----------------------------------------------------------------------------
ETHUSDT   14      71.4%     1.39    +12.04%     -15.51%   2.63    
SOLUSDT   14      85.7%     5.41    +76.69%     -17.38%   15.71   
AVAXUSDT  13      69.2%     2.75    +65.91%     -26.57%   7.92    
NEARUSDT  10      70.0%     3.22    +56.60%     -25.53%   8.39    
----------------------------------------------------------------------------
ИТОГО     51      74.5%     --      +52.81% (средний) | +211.24% (суммарный)
============================================================================
```

### Оптимизация гиперпараметров (`optimizer.py`):
```bash
# Оптимизация параметров для SOLUSDT за 90 дней (30 итераций Optuna)
python optimizer.py SOLUSDT --trials 30 --days 90

# Walk-Forward анализ (70% обучение In-Sample / 30% проверка Out-of-Sample)
python optimizer.py SOLUSDT --wfa --split 0.7 --trials 40
```
