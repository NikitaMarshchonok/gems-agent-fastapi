#!/usr/bin/env python3
"""
Запуск FastAPI сервера для Gems Agent
"""
import uvicorn
import os
from pathlib import Path

if __name__ == "__main__":
    # Устанавливаем PYTHONPATH
    project_root = Path(__file__).parent
    os.environ["PYTHONPATH"] = str(project_root)
    
    # Запускаем сервер
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        reload_dirs=[str(project_root / "app")],
        log_level="info"
    )

