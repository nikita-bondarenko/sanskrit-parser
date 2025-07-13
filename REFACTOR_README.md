# Sanskrit OCR Application - Refactored Architecture

## Overview

Приложение было полностью рефакторировано для улучшения организации кода, поддержки и масштабируемости. Код разделен на логические модули и компоненты.

## Backend Architecture

### Модульная структура

```
backend/
├── main.py              # Основной файл приложения
├── auth.py              # Модуль аутентификации
├── ocr_service.py       # Сервис OCR обработки
├── routes.py            # API маршруты
├── sanskrit_database.py # База данных санскритских текстов
├── text_extractor.py    # Извлечение текста из файлов
└── requirements.txt     # Зависимости
```

### Описание модулей

#### `auth.py`
- Аутентификация администратора
- Управление токенами доступа
- Проверка прав доступа

#### `ocr_service.py`
- Класс `SanskritOCRModel` - нейронная сеть для OCR
- Класс `OCRService` - сервис обработки изображений
- Предобработка изображений
- Постобработка текста
- Конвертация в русские диакритики

#### `routes.py`
- Все API endpoints
- Валидация запросов
- Обработка ошибок
- Логирование метрик

#### `sanskrit_database.py`
- Управление базой данных SQLite
- Поиск и сопоставление текстов
- Статистика базы данных

#### `text_extractor.py`
- Извлечение текста из различных форматов
- Анализ структуры текста
- Определение метаданных

## Frontend Architecture

### Компонентная структура

```
frontend/src/
├── App.tsx                    # Главный компонент
├── types/
│   └── index.ts              # Типы и интерфейсы
├── hooks/
│   └── useAPI.ts             # API хуки
├── utils/
│   └── index.ts              # Утилитарные функции
└── components/
    ├── OCRTab.tsx            # Компонент OCR
    ├── AdminPanel.tsx        # Панель администратора
    ├── DatabaseStats.tsx     # Статистика БД
    └── BookUpload.tsx        # Загрузка книг
```

### Описание компонентов

#### `types/index.ts`
- Все TypeScript интерфейсы
- Типы для API ответов
- Типы состояния приложения

#### `hooks/useAPI.ts`
- Custom hooks для API вызовов
- Обработка HTTP запросов
- Централизованная логика API

#### `utils/index.ts`
- Утилитарные функции
- Валидация файлов
- Форматирование данных
- Работа с буфером обмена

#### `components/OCRTab.tsx`
- Загрузка и обработка изображений
- Отображение результатов OCR
- Информация об источнике

#### `components/AdminPanel.tsx`
- Аутентификация администратора
- Управление сессией
- Статус доступа

#### `components/DatabaseStats.tsx`
- Отображение статистики БД
- Количество текстов, книг, слов

#### `components/BookUpload.tsx`
- Загрузка файлов книг
- Валидация форматов
- Прогресс загрузки

## Key Features

### 🔐 Admin Authentication
- Безопасная аутентификация с паролем "hegopinath"
- JWT-подобные токены с истечением (24 часа)
- Защищенные endpoints для загрузки книг

### 📚 Database Integration
- SQLite база данных для санскритских текстов
- Поиск по фразам и словам
- Fuzzy matching для улучшения точности OCR
- Указание источника найденного текста

### 🔍 Enhanced OCR
- Нейронная сеть для распознавания
- Множественные варианты предобработки
- Интеллектуальная постобработка
- Конвертация IAST и Gaura PT в русские диакритики

### 📖 Multi-format Support
- PDF, DOCX, TXT, DOC, RTF, ODT
- Автоматическое извлечение текста
- Определение структуры (главы, стихи)
- Анализ метаданных

## API Endpoints

### Public Endpoints
- `GET /` - Корневой endpoint
- `GET /health` - Проверка здоровья
- `POST /ocr` - OCR обработка изображений
- `GET /database-stats` - Статистика БД
- `GET /search-books` - Поиск по книгам
- `GET /supported-formats` - Поддерживаемые форматы

### Admin Endpoints
- `POST /admin/login` - Вход администратора
- `POST /upload-book` - Загрузка книг (требует токен)

## Database Schema

### Tables
- `sanskrit_texts` - Основные тексты
- `sanskrit_words` - Индекс слов
- `sanskrit_phrases` - Индекс фраз

### Indexes
- По нормализованному тексту
- По словам и фразам
- По источникам

## Security Features

- Токены с ограниченным временем жизни
- Валидация файлов по типу и размеру
- Защита от XSS и инъекций
- CORS настройки

## Performance Optimizations

- Кэширование статистики БД
- Индексы для быстрого поиска
- Ленивая загрузка компонентов
- Debounce для поиска

## Development

### Backend
```bash
cd backend
pip install -r requirements.txt
python main.py
```

### Frontend
```bash
cd frontend
npm install
npm run dev
```

### Docker
```bash
docker-compose up --build
```

## Future Improvements

1. **Redis для токенов** - Замена in-memory хранения
2. **PostgreSQL** - Более мощная БД для больших объемов
3. **Elasticsearch** - Полнотекстовый поиск
4. **WebSocket** - Реальное время для загрузки
5. **Тесты** - Unit и интеграционные тесты
6. **CI/CD** - Автоматическое развертывание

## Architecture Benefits

### Maintainability
- Четкое разделение ответственности
- Модульная структура
- Типизация TypeScript

### Scalability
- Независимые компоненты
- Легкое добавление новых функций
- Горизонтальное масштабирование

### Testability
- Изолированные модули
- Dependency injection
- Мокирование API вызовов

### Performance
- Ленивая загрузка
- Кэширование
- Оптимизированные запросы

Этот рефакторинг создает основу для дальнейшего развития приложения с улучшенной архитектурой и лучшими практиками разработки. 