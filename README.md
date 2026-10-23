# Bybit Futures Multi-Pair Bot v2.0 (Termux & Server Ready)

Алготрейдинг-бот для деривативов Bybit с адаптивным риск-менеджментом, встроенным веб-дашбордом на FastAPI и модулем оптимизации стратегий.



## Структура проекта

```
bybit_bot/
├── bot.py                  # Главный торговый движок бота v2.0
├── config.py               # Загрузка и валидация настроек из .env
├── config.example.py       # Референсный шаблон конфигурации
├── .env.example            # Шаблон переменных окружения и ключей API
├── risk_manager.py         # Модуль трейлинга, частичного TP и режима рынка
├── optimizer.py            # Модуль оптимизации Optuna + Walk-Forward Analysis
├── backtest.py             # Модуль бэктестинга (с фандингом и проскальзыванием)
├── data_loader.py          # Загрузчик исторических свечей с Bybit API v5
├── dashboard.py            # Автономный запуск веб-дашборда
├── requirements.txt        # Список зависимостей Python
│
├── strategies/             # Каталог модульных торговых стратегий
│   ├── __init__.py         # Фабрика и реестр стратегий
│   ├── base.py             # Базовый абстрактный класс BaseStrategy
│   └── ma_rsi_strategy.py  # Стратегия Golden Cross + RSI + Volume + MTF
│
├── tests/                  # Набор тестов (стратегии, риски, WebSocket, Hedge Mode)
│   └── test_v2.py          # Автоматические тесты функционала v2.0
│
└── web/                    # Веб-дашборд (FastAPI)
    ├── app.py              # Эндпоинты REST API и маршрутизация
    ├── state.py            # Общее состояние и парсер trade.log
    └── templates/
        └── index.html      # Адаптивный мобильный интерфейс (Tailwind CSS)
```

---

## Быстрый старт

### 1. Установка зависимостей
```bash
pip install -r requirements.txt
```

### 2. Настройка переменных окружения (`.env`)
Скопируйте файл шаблона `.env.example` в `.env`:
```bash
# Windows
copy .env.example .env

# Linux / Termux
cp .env.example .env
```

Отредактируйте `.env`, указав ваши API-ключи Bybit:
```env
TRADING_MODE=DEMO             # DEMO или LIVE
DEMO_API_KEY=your_demo_key
DEMO_API_SECRET=your_demo_secret
LIVE_API_KEY=your_live_key
LIVE_API_SECRET=your_live_secret

DASHBOARD_HOST=0.0.0.0
DASHBOARD_PORT=8000
```

---

### 3. Проверка тестов (рекомендуется перед запуском):
`ash
python tests/test_v2.py
`

## Запуск системы

### Запуск торгового бота с интегрированным дашбордом:
```bash
python bot.py
```
> Бот инициализирует подключение к Bybit, загрузит параметры контрактов и параллельно в фоновом потоке запустит веб-дашборд по адресу `http://0.0.0.0:8000`.

### Запуск автономного веб-дашборда (без торгового цикла):
```bash
python dashboard.py
```


---

## Бэктестинг и оптимизация

### Запуск бэктеста:
```bash
# Бэктест по одной конкретной паре за 90 дней
python backtest.py BTCUSDT 90

# Сводный бэктест по всей корзине пар
python backtest.py all 90

# Бэктест с принудительной перезагрузкой исторических данных
python backtest.py SOLUSDT 60 --force
```

### Оптимизация параметров (Optuna + WFA):
```bash
# Оптимизация стратегии для BTCUSDT за 90 дней (30 итераций)
python optimizer.py BTCUSDT --trials 30 --days 90

# Оптимизация с валидацией Walk-Forward (70% обучение / 30% тест)
python optimizer.py SOLUSDT --wfa --split 0.7 --trials 40
```

---

