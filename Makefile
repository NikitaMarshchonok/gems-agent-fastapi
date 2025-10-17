.PHONY: install run dev test clean

# Установка зависимостей
install:
	pip install -r requirements.txt

# Запуск в режиме разработки
dev:
	python run.py

# Запуск через uvicorn напрямую
run:
	uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

# Запуск в продакшене
prod:
	uvicorn app.main:app --host 0.0.0.0 --port 8000 --workers 4

# Тестирование API
test:
	curl http://localhost:8000/health
	curl http://localhost:8000/templates

# Очистка кэша
clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -name "*.pyc" -delete

# Показать помощь
help:
	@echo "Доступные команды:"
	@echo "  make install  - Установить зависимости"
	@echo "  make dev      - Запустить в режиме разработки"
	@echo "  make run      - Запустить через uvicorn"
	@echo "  make prod     - Запустить в продакшене"
	@echo "  make test     - Тестировать API"
	@echo "  make clean    - Очистить кэш"

