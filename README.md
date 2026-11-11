# Bybit Futures Multi-Pair Bot v0.3.0 (Termux & Server Ready)

Алготрейдинг-бот для линейных бессрочных фьючерсов **Bybit (USDT Perpetual)** в `One-Way Mode` с модульной архитектурой стратегий (`SUPERTREND_PRO`, `REGIME_SWITCHER`, `MA_RSI`, `BB_RSI`, `SUPERTREND`, `DONCHIAN`), event-driven обработкой закрытия свечей по WebSocket, локальной базой SQLite (WAL) и сверкой `get_closed_pnl`, деривативным фильтром рыночного режима, портфельным риск-менеджментом, веб-дашбордом на FastAPI и встроенным **Shadow / Paper Mode**.

---

## Ключевые возможности v0.3.0

### 1. Торговые стратегии, переключатель режимов и выбор корзины
* **`SUPERTREND_PRO` (основная стратегия по умолчанию):**
  * Вход по пробою `SuperTrend (10, 3.0)` с подтверждением объёма (`VOL_SMA 20`), фильтром силы тренда `ADX(14) >= 18.0`, подтверждением `EMA(50)` (`ST_CONFIRM_LOOKBACK = 1`) и защитой от перекупленности/перепроданности по `RSI(14)` (`<= 72` для Long / `>= 28` для Short).
* **`REGIME_SWITCHER` (алиас `ADAPTIVE`):**
  * Адаптивный переключатель рыночных режимов **`TREND` ↔ `RANGE`**: комбинирует `SUPERTREND_PRO` в тренде и `BB_RSI` в боковике с гистерезисом по `ADX`, коэффициенту эффективности Кауфмана (`ER`) и процентилю ширины полос Боллинджера, а также mean-reversion выходами (`USE_REGIME_SWITCHER=true` или `ACTIVE_STRATEGY=REGIME_SWITCHER`).
* **Дополнительные встроенные стратегии:** `MA_RSI`, `BB_RSI`, `SUPERTREND`, `DONCHIAN`.
* **Деривативный `Market Regime Filter`:**
  * Комбинирует старший тренд `BTCUSDT 4H EMA(200)` / гибридную `EMA(50)`, `RSI`, защиту от резких импульсов и экстремальной волатильности с **Funding Rate Z-score** и **дивергенцией Open Interest (`OI`)** (`data_loader.py` + `risk_manager.py`).
* **Динамический отбор ликвидной корзины (`DynamicUniverseSelector`):**
  * Опциональный (`USE_DYNAMIC_UNIVERSE=true`) периодический отбор `TOP_N` пар по 24ч обороту, спреду, диапазону волатильности и чёрному списку (`UNIVERSE_BLACKLIST`).

### 2. Портфельный риск-менеджмент, сопровождение и Time-Stop
* **Риск-ориентированный сайзинг и защита от перекоса:** расчёт позиции от риска на сделку (`RISK_PER_TRADE_PCT = 1.5%` от `equity` с ограничением по доступному балансу), `Decimal`-округление лота/цены, проверка нотионала (`>= 5 USDT`) и лимит однонаправленных позиций (`MAX_SAME_DIRECTION_POSITIONS = 2`).
* **Стопы, частичная фиксация и трейлинг:**
  * Начальные уровни: `SL = 2.2 * ATR`, `TP = 4.5 * ATR`.
  * Частичный тейк-профит `TP1 = 1.2R` (лимитный `reduceOnly` на `50%`), перевод стопа в безубыток на `1.1R` (`+0.2%`) и двойной трейлинг (`1.4R`, дистанция `1.5 * ATR` + серверный `trailingStop`).
* **Портфельная защита (`PortfolioRiskManager`):**
  * **Aggregate Portfolio Heat Cap (`MAX_PORTFOLIO_HEAT_PCT = 3.5%`):** ограничение суммарного открытого риска по портфелю.
  * **Корреляционное масштабирование (`USE_CORRELATION_SCALING`):** снижение риска при входе в пары с высокой корреляцией к уже открытым позициям (`CORR_THRESHOLD = 0.75`, множитель `0.6`).
  * **Multi-Tier Peak DD Throttle:** снижение сайзинга при просадке от High-Water Mark (Tier 1 `-6%` → `x0.5`, Tier 2 пауза `-12%`, выход при восстановлении выше `-9%`).
  * **Loss-Streak Cooldown:** временная блокировка новых входов по паре (`2` убытка) или портфелю (`3` убытка) на `LOSS_STREAK_COOLDOWN_BARS = 6` баров.
  * **Дневной `Daily Circuit Breaker`:** остановка торговли и закрытие позиций при дневном убытке `>= 5%` с защитой от таймаутов API.
* **Поэтапный `Time-Stop` (`USE_TIME_STOP`):** сокращение/закрытие застоявшихся позиций через `30ч` / `36ч`, если сделка не достигла `0.3R`.

### 3. Event-driven WebSocket, исполнение, SQLite и лёгкость для Termux
* **WS `kline_stream` триггер (`USE_WS_KLINE_TRIGGER`):** торговый цикл реагирует на закрытие свечи в публичном WebSocket с кэшем закрытых свечей (`get_klines_fast`), резервным REST-поллингом раз в `180с` и суточным кэшированием спецификаций инструментов.
* **Maker Chase (`amend_order`) + фильтр спреда:** отклонение входа при аномальном спреде (`> MAX_ENTRY_SPREAD_PCT = 0.5%`), лимитный вход `PostOnly` с догоном лучшего bid/ask через `amend_order` (до `2` перестановок в пределах `0.4%` / `0.4 * ATR`) и безопасным фоллбэком на `Market`.
* **Адаптивный `RateLimitController`:** контроль заголовков `X-Bapi-Limit-*` и упреждающая пауза перед исчерпанием лимитов.
* **SQLite WAL (`bot_data.db`) + сверка `get_closed_pnl`:**
  * Хранилище сделок, биржевых записей PnL, снимков equity и runtime-состояния (дневной и пиковый балансы, риск-лимиты, `ShadowEngine`) переживает перезапуски без потери метрик.
  * Использует встроенный модуль Python `sqlite3` — устанавливать отдельный сервер БД на **Termux** не требуется, а [`requirements.txt`](requirements.txt) не утяжелён.

### 4. Shadow / Paper Mode и Веб-дашборд (FastAPI)
* **Shadow / Paper Mode (`SHADOW_MODE=true`):** симуляция стратегии (`ACTIVE_STRATEGY` или отдельной `SHADOW_STRATEGY`) на живом потоке котировок и свечей Bybit с учётом комиссий, спреда и портфельного риска — без отправки ордеров и без обращения к приватному кошельку (включая автономный `dashboard.py`).
* **Веб-дашборд:** живые метрики по `/ws`, свечи с уровнями и кривой капитала (TradingView Lightweight Charts), блок аналитики и экспорта сделок (`/api/analytics`, `/api/trades/export` в `CSV`/`JSON`), просмотр ликвидной корзины (`/api/universe`), сохранение выгрузки `.log` (`/api/logs/download`, `/api/logs/view`), кнопки `Пауза`, закрытия позиций, `PANIC CLOSE`, переключение `DEMO / LIVE` и HTTP Basic Auth.

---

## Структура проекта

```text
bybit-trading-bot/
├── bot.py                        # Торговый движок (REST + Public/Private WS, RateLimitController, Maker Chase)
├── config.py                     # Конфигурация из .env и fail-closed валидация режима торговли
├── .env.example                  # Шаблон ключей API и опций v0.3.0
├── .gitignore                    # Исключения для секретов, логов, SQLite (*.db*) и профилей
├── storage.py                    # SQLite (WAL) хранилище сделок, equity, runtime-состояния и экспортов
├── shadow.py                     # Shadow / Paper движок на реальном потоке Bybit
├── universe.py                   # DynamicUniverseSelector — отбор ликвидной корзины пар
├── risk_manager.py               # Trailing/BE/Time-Stop, MarketRegimeFilter (Funding Z/OI) и PortfolioRiskManager
├── backtest.py                   # Бэктестер одиночных пар и портфеля (комиссии, фандинг, проскальзывание, Time-Stop)
├── optimizer.py                  # Оптимизатор Optuna (одиночный и портфельный), Rolling WFA, Monte Carlo и DSR
├── data_loader.py                # Загрузка и кэширование свечей, Funding Rate и Open Interest с Bybit v5
├── dashboard.py                  # Автономный запуск веб-дашборда
├── requirements.txt              # Зависимости Python
│
├── strategies/                   # Модульные торговые стратегии
│   ├── __init__.py               # Реестр стратегий (get_strategy)
│   ├── base.py                   # Базовый класс BaseStrategy (индикаторы, ATR-стопы, MTF)
│   ├── supertrend_atr.py         # SUPERTREND_PRO и SUPERTREND
│   ├── regime_switcher.py        # REGIME_SWITCHER (адаптивное переключение TREND <-> RANGE)
│   ├── ma_rsi_strategy.py        # MA_RSI (Golden/Death Cross + RSI + ADX)
│   ├── bb_rsi_reversion.py       # BB_RSI (Bollinger Bands + RSI Mean Reversion)
│   └── donchian_breakout.py      # DONCHIAN (пробой канала Дончиана)
│
├── tests/                        # Модульные и интеграционные автотесты
│   ├── test_v2.py                # 41 тест базовой архитектуры, WS, исполнения, стратегий и UI
│   └── test_v3.py                # 20 тестов v0.3.0 (SQLite, WS kline, Maker Chase, Portfolio Risk, WFA/MC, Shadow)
│
└── web/                          # Веб-дашборд (FastAPI)
    ├── app.py                    # REST API, WebSocket /ws, экспорт сделок, Shadow-изоляция и Basic Auth
    ├── state.py                  # Потокобезопасный IPC (bot_control.json / bot_live_status.json) и SQLite/лог ридеры
    └── templates/
        └── index.html            # Адаптивный UI дашборда (Lightweight Charts, аналитика, Shadow-индикация)
```

---

## Быстрый старт

### 1. Установка зависимостей
```bash
pip install -r requirements.txt
```
> В Termux база данных SQLite работает через встроенный модуль `sqlite3` стандартной библиотеки Python — устанавливать внешнюю БД не нужно.

### 2. Настройка переменных окружения (`.env`)
Скопируйте шаблон `.env.example` в `.env`:
```bash
# Windows (PowerShell / CMD)
copy .env.example .env

# Linux / macOS / Termux
cp .env.example .env
```

Основные настройки и ключевые опции `v0.3.0` в `.env`:
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
USE_RISK_BASED_SIZING=true
RISK_PER_TRADE_PCT=0.015
MAX_SLIPPAGE_PCT=0.015

# Опциональные параметры v0.3.0 (значения по умолчанию в config.py)
# USE_WS_KLINE_TRIGGER=true
# USE_MAKER_CHASE=true
# MAX_ENTRY_SPREAD_PCT=0.005
# MAX_PORTFOLIO_HEAT_PCT=0.035
# USE_CORRELATION_SCALING=true
# USE_PEAK_DD_THROTTLE=true
# USE_LOSS_STREAK_COOLDOWN=true
# USE_TIME_STOP=true
# USE_REGIME_SWITCHER=false
# USE_DYNAMIC_UNIVERSE=false
# SHADOW_MODE=false
# SHADOW_STRATEGY=
```

### 3. Запуск автотестов (рекомендуется перед стартом)
```bash
python -m unittest discover -s tests -v
```
Актуальный набор включает **61 тест** (`tests/test_v2.py` и `tests/test_v3.py`), покрывающий хранилище SQLite/WAL, сверку PnL, WS `kline_stream`, Maker Chase и спред-фильтр, портфельные риск-лимиты, `REGIME_SWITCHER`, `DynamicUniverseSelector`, оптимизатор, `ShadowEngine` и веб-эндпоинты.

---

## Запуск системы

### Торговый бот со встроенным веб-дашбордом:
```bash
python bot.py
```
> Бот синхронизирует время с Bybit, настроит плечо (`One-Way Mode` по умолчанию), запустит WebSocket-потоки и поднимет веб-дашборд на `http://0.0.0.0:8000` (или в `SHADOW_MODE=true` запустит paper-движок на публичном потоке цен/свечей без реальных ордеров).

### Автономный веб-дашборд (отдельный процесс):
```bash
python dashboard.py
```
> Обменивается состоянием с `bot.py` через `bot_live_status.json`, `bot_control.json` и `bot_data.db`. В `SHADOW_MODE=true` дашборд читает только shadow-метрики и не запрашивает реальный баланс биржи.

---

## Бэктестинг и оптимизация

### Запуск бэктеста (`backtest.py`):
Бэктестер учитывает комиссии Maker/Taker, динамическое проскальзывание, фандинг каждые 8 часов (`00:00, 08:00, 16:00 UTC`), `TP1 (1.2R)`, безубыток (`1.1R`), трейлинг-стоп, `Time-Stop`, деривативный фильтр режима и портфельные лимиты:

```bash
# Сводный и портфельный бэктест по корзине из config.py за 90 дней
python backtest.py all 90

# Бэктест по одной паре за 90 дней
python backtest.py SOLUSDT 90

# Перекачать свежие данные с биржи
python backtest.py ETHUSDT 90 --force
```

Контрольный прогон базовой корзины `SUPERTREND_PRO` (плечо `x5`, 90 дней):
```text
==============================================================================
  СВОДНЫЙ РЕЗУЛЬТАТ ПО КОРЗИНЕ ИЗ 4 ПАР (90 ДНЕЙ) [v2.3]
==============================================================================
Пара      Позиций  PosWR    PF      Счет%      MaxDD%    D.Sharpe 
------------------------------------------------------------------------------
ETHUSDT   10       50.0%    1.20    +0.92%     -5.11%    0.53     
SOLUSDT   9        66.7%    3.50    +8.07%     -2.76%    3.42     
AVAXUSDT  13       53.8%    1.51    +2.16%     -4.91%    0.87     
NEARUSDT  8        62.5%    2.71    +3.34%     -2.70%    1.72     
------------------------------------------------------------------------------
  [ПОРТФЕЛЬНАЯ СИМУЛЯЦИЯ] Капитал: 1000$ -> 1169.97$ (+17.00%)
  Позиций: 36 (откр. на финише: 0) | Pos WR: 61.1% | PF: 2.62 | MaxDD: -8.13% | Daily Sharpe: 3.63
==============================================================================
```

### Оптимизация, Rolling WFA, Monte Carlo и DSR (`optimizer.py`):
```bash
# Одиночная оптимизация + скользящий WFA + Monte Carlo + DSR
python optimizer.py SOLUSDT --trials 30 --days 90 --windows 3 --mc-sims 500

# Портфельная оптимизация сразу по всей корзине с сохранением версионированного профиля
python optimizer.py --portfolio --trials 30 --days 90 --windows 3 --mc-sims 500 --save-profile

# Портфельная оптимизация для конкретной стратегии и списка пар
python optimizer.py --portfolio --strategy REGIME_SWITCHER --symbols ETHUSDT,SOLUSDT,AVAXUSDT --trials 25 --days 120
```
