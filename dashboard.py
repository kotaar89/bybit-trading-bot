"""
Автономный запуск веб-дашборда Bybit Bot (v2.0).
Доступен внутри локальной сети Termux: http://192.168.0.100:8000
Использование:
    python dashboard.py
"""

import uvicorn
from config import DASHBOARD_HOST, DASHBOARD_PORT

if __name__ == "__main__":
    print("=" * 60)
    print(f"  Запуск Bybit Bot v2.0 Dashboard")
    print(f"  Хост: {DASHBOARD_HOST} | Порт: {DASHBOARD_PORT}")
    print(f"  Локальный URL: http://{DASHBOARD_HOST}:{DASHBOARD_PORT}")
    print("=" * 60)

    uvicorn.run(
        "web.app:app",
        host=DASHBOARD_HOST,
        port=DASHBOARD_PORT,
        log_level="info"
    )
