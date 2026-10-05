# detector — CPU

Сервис для поиска выбранных объектов в видео. Загрузите ролик, укажите названия объектов и/или приложите фото; результат — MP4 с рамками и ZIP с детекциями, временными метками и переданными геоданными. В `main` находится CPU-сборка. [GPU-сборка](https://github.com/iljas277/detector/tree/gpu) живёт в отдельной ветке.

## Запуск через Docker

Нужны Docker и Docker Compose:

```bash
git clone https://github.com/iljas277/detector.git
cd detector
docker compose build
docker compose run --rm api python -m video_search.cli weights download
docker compose up -d
```

Откройте <http://127.0.0.1:8000>. Данные и веса хранятся в `./data`; контейнеры API и worker используют один каталог. Поиск только по фото работает и без команды `weights download`. Остановить сервис: `docker compose down`.

## Запуск без Docker

Нужны Python 3.12, `ffmpeg` и `ffprobe`. Установите CPU-версию PyTorch отдельно от остальных зависимостей:

```bash
python3.12 -m venv .venv
.venv/bin/python -m pip install 'torch==2.6.0+cpu' --index-url https://download.pytorch.org/whl/cpu
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -m video_search.cli weights download
```

Запустите в двух терминалах из папки проекта:

```bash
.venv/bin/python -m video_search.cli worker
```

```bash
.venv/bin/uvicorn video_search.api:app --host 127.0.0.1 --port 8000
```

## API

`POST /api/quick/analyze` принимает `multipart/form-data`: обязательный `file` (видео), `objects` (типы объектов через запятую) и/или `reference_image` (фото объекта). Можно передать `start_utc`, `latitude`, `longitude`, `accuracy_m` и `telemetry_json` с координатами источника по времени. Ответ содержит `job_id`; статус — `GET /api/quick/jobs`. Готовые файлы: `GET /api/quick/jobs/{job_id}/video` и `GET /api/quick/jobs/{job_id}/archive` (ZIP с `annotated.mp4` и `detections.json`). Интерактивная схема: <http://127.0.0.1:8000/docs>.

Текстовый поиск выполняет OWLv2. Поиск по фото сопоставляет изображение объекта с кадрами и лучше работает при похожем ракурсе. Результат следует проверять глазами: модель может пропустить объект или поставить ложную рамку. API не содержит авторизации; для доступа извне нужен защищённый шлюз.
