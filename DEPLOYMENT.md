# Sanskrit Parser - Deployment Guide

## Развертывание в режиме разработки

### Предварительные требования

1. **Docker и Docker Compose** должны быть установлены
2. **Traefik** должен быть запущен и доступна сеть `traefik_net`
3. **Домен** `sanskrit-parser.bondarenko-nikita.ru` должен быть настроен

### Структура файлов

```
sanskrit-parser/
├── docker-compose.dev.yml      # Development конфигурация
├── docker-compose.prod.yml     # Production конфигурация (для будущего)
├── deploy.sh                   # Скрипт развертывания
├── logs.sh                     # Скрипт просмотра логов
├── stop.sh                     # Скрипт остановки
├── backend/
│   ├── Dockerfile
│   └── main.py                 # API с префиксом /api
├── frontend/
│   ├── Dockerfile
│   ├── Dockerfile.prod         # Production Dockerfile
│   └── nginx.conf              # Nginx конфигурация
```

### Запуск в режиме разработки

#### Обычный запуск:
1. **Запустить приложение:**
   ```bash
   ./deploy.sh
   ```

#### Быстрый запуск (рекомендуется):
1. **Собрать базовый образ (один раз):**
   ```bash
   ./build-base-image.sh
   ```

2. **Быстрый запуск приложения:**
   ```bash
   ./deploy-fast.sh
   ```

#### Управление:
2. **Просмотреть логи:**
   ```bash
   ./logs.sh
   ```

3. **Остановить приложение:**
   ```bash
   ./stop.sh
   ```

### Конфигурация Traefik

В файле `traefik/dynamic_conf.yml` добавлена конфигурация:

```yaml
sanskrit-parser-router:
  rule: Host(`sanskrit-parser.bondarenko-nikita.ru`)
  entryPoints:
    - websecure   
    - web      
  service: sanskrit-parser_svc
  tls:
    certResolver: myresolver

# Service
sanskrit-parser_svc:
  loadBalancer:
    servers:
      - url: "http://sanskrit-parser-frontend:3000"
```

### Особенности Development режима

- **Hot Reload**: Изменения в коде автоматически применяются
- **Volume Mapping**: Код проекта примонтирован в контейнеры
- **Development Server**: Frontend запускается с `pnpm run dev`
- **API Reload**: Backend перезапускается при изменениях с `--reload`

### API Endpoints

Все API endpoints доступны через префикс `/api`:

- `GET /api/health` - Проверка состояния
- `POST /api/ocr` - OCR обработка изображений
- `POST /api/admin/login` - Авторизация админа
- `POST /api/upload-book` - Загрузка книг
- `GET /api/database-stats` - Статистика базы данных

### Доступ к приложению

- **Frontend**: https://sanskrit-parser.bondarenko-nikita.ru
- **API**: https://sanskrit-parser.bondarenko-nikita.ru/api
- **Health Check**: https://sanskrit-parser.bondarenko-nikita.ru/api/health

### Troubleshooting

1. **Проверить сеть Traefik:**
   ```bash
   docker network ls | grep traefik_net
   ```

2. **Проверить статус контейнеров:**
   ```bash
   docker-compose -f docker-compose.dev.yml ps
   ```

3. **Просмотреть логи:**
   ```bash
   docker-compose -f docker-compose.dev.yml logs -f
   ```

4. **Перезапустить приложение:**
   ```bash
   ./stop.sh && ./deploy-fast.sh
   ```

5. **Ошибка "host not allowed":**
   - Убедитесь, что в `frontend/vite.config.ts` добавлен домен в `allowedHosts`
   - После изменения конфигурации перезапустите контейнеры

### Оптимизация сборки

#### Структура Dockerfile:
- **Multi-stage build**: Отдельные этапы для сборки и продакшена
- **Поэтапная установка зависимостей**: Базовые, ML, текстовые
- **Кеширование слоев**: Лучшее использование Docker cache
- **Минимальный финальный образ**: Только runtime зависимости

#### Файлы оптимизации:
- `backend/Dockerfile` - Оптимизированный основной Dockerfile
- `backend/Dockerfile.fast` - Быстрый Dockerfile с предварительно собранным базовым образом
- `backend/.dockerignore` - Исключение ненужных файлов
- `backend/requirements-*.txt` - Разделенные зависимости
- `build-base-image.sh` - Сборка базового образа
- `deploy-fast.sh` - Быстрое развертывание

#### Преимущества:
- **Ускорение сборки в 3-5 раз** при повторных сборках
- **Лучшее кеширование** Docker слоев
- **Меньший размер** финального образа
- **Безопасность**: Не root пользователь

### Переход в Production

Когда разработка будет завершена, можно будет использовать:
- `docker-compose.prod.yml` для production развертывания
- `frontend/Dockerfile.prod` для оптимизированного frontend
- `frontend/nginx.conf` для production nginx конфигурации 