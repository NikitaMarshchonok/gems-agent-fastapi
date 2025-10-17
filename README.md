# 🤖 Gems Agent FastAPI

Аналог Gemini Gems для создания AI агентов с кастомными инструкциями и базой знаний.

## 🚀 Быстрый старт

### 1. Установка зависимостей
```bash
make install
# или
pip install -r requirements.txt
```

### 2. Запуск сервера

#### Через VS Code:
1. Открой проект в VS Code
2. Нажми `F5` или `Ctrl+Shift+P` → "Debug: Start Debugging"
3. Выбери "Run FastAPI Server"

#### Через терминал:
```bash
# Режим разработки
make dev
# или
python run.py

# Прямой запуск
make run
# или
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

### 3. Открыть веб-интерфейс
```
http://localhost:8000/manage
```

## 🎯 Возможности

- ✅ **Создание агентов** с кастомными инструкциями
- ✅ **Загрузка файлов** с drag & drop
- ✅ **База знаний** с RAG поиском
- ✅ **Инструменты** (web_search, calculator, kb_search)
- ✅ **Шаблоны** для быстрого старта
- ✅ **Тестирование** агентов в реальном времени
- ✅ **Редактирование** существующих агентов

## 📁 Структура проекта

```
gems-agent-fastapi/
├── app/
│   ├── main.py          # FastAPI приложение
│   ├── models.py        # Pydantic модели
│   ├── store.py         # Хранение данных
│   ├── llm.py          # LLM интеграция
│   ├── tools.py        # Инструменты агентов
│   └── kb.py           # База знаний
├── data/               # Данные агентов
├── .vscode/            # Конфигурация VS Code
├── .env                # Переменные окружения
├── run.py              # Скрипт запуска
├── Makefile            # Команды для разработки
└── requirements.txt    # Зависимости
```

## 🔧 API эндпоинты

- `GET /health` - Проверка здоровья
- `GET /templates` - Список шаблонов
- `GET /gems` - Список агентов
- `POST /gems` - Создание агента
- `POST /gems/{id}/files` - Загрузка файлов
- `POST /chat` - Чат с агентом
- `GET /manage` - Веб-интерфейс

## ⚙️ Конфигурация

Настройки в файле `.env`:

```env
# LLM Backend
LLM_BACKEND=ollama
EMBED_BACKEND=ollama

# Ollama (по умолчанию)
OLLAMA_BASE_URL=http://127.0.0.1:11434
OLLAMA_MODEL=llama3.1:8b
OLLAMA_EMBED_MODEL=nomic-embed-text

# OpenAI (опционально)
# OPENAI_API_KEY=your_key_here
# OPENAI_MODEL=gpt-4o-mini
```

## 🛠️ Разработка

### Команды Makefile:
```bash
make install  # Установить зависимости
make dev      # Запустить в режиме разработки
make run      # Запустить через uvicorn
make prod     # Запустить в продакшене
make test     # Тестировать API
make clean    # Очистить кэш
```

### VS Code:
- `F5` - Запуск в режиме отладки
- `Ctrl+Shift+P` → "Python: Select Interpreter" - выбор интерпретатора
- `Ctrl+Shift+P` → "Debug: Start Debugging" - запуск отладки

## 📝 Использование

1. **Выбери шаблон** или создай агента с нуля
2. **Напиши инструкции** для агента
3. **Загрузи файлы** для базы знаний
4. **Протестируй агента** в секции тестирования
5. **Используй API** для интеграции

## 🔍 Примеры использования

### Создание агента через API:
```bash
curl -X POST http://localhost:8000/gems \
  -H "Content-Type: application/json" \
  -d '{
    "name": "Travel Assistant",
    "system_prompt": "You are a helpful travel assistant...",
    "tools": ["web_search", "calculator"]
  }'
```

### Чат с агентом:
```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{
    "gem_id": "agent_id",
    "messages": [{"role": "user", "content": "Hello!"}]
  }'
```

## 🎨 Шаблоны агентов

- **Travel Assistant** - Помощник по путешествиям
- **Code Reviewer** - Ревьюер кода
- **Research Assistant** - Исследовательский помощник
- **Customer Support** - Агент поддержки
- **Content Writer** - Копирайтер
- **Data Analyst** - Аналитик данных

## 📄 Лицензия

MIT License