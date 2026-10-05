# detector — GPU

Сервис для поиска выбранных объектов в видео. Загрузите ролик, укажите названия объектов и/или приложите фото; результат — MP4 с рамками и ZIP с детекциями, временными метками и переданными геоданными. Эта ветка использует NVIDIA GPU и CUDA 12.4. [CPU-версия](https://github.com/iljas277/detector/tree/main) находится в `main`.

## Запуск через Docker

Нужны NVIDIA GPU с подходящим драйвером, Docker Compose и [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html):

```bash
git clone https://github.com/iljas277/detector.git
cd detector
git switch gpu
docker compose build
docker compose run --rm api python -m video_search.cli weights download
docker compose up -d
```

Откройте <http://127.0.0.1:8001>. Контейнер worker запрашивает GPU и работает с `VIDEO_SEARCH_DEVICE=cuda`; при недоступной CUDA задача завершится ошибкой. Эта ветка использует образ `detector:gpu` и каталог `./data-gpu`, поэтому они не смешиваются с CPU-версией. Поиск только по фото работает и без команды `weights download`. Остановить сервис: `docker compose down`.

## Запуск без Docker

Нужны Python 3.12, `ffmpeg`, `ffprobe` и рабочая NVIDIA CUDA-среда. Установите CUDA-версию PyTorch отдельно от остальных зависимостей:

```bash
python3.12 -m venv .venv-gpu
.venv-gpu/bin/python -m pip install 'torch==2.6.0+cu124' --index-url https://download.pytorch.org/whl/cu124
.venv-gpu/bin/python -m pip install -r requirements.txt
.venv-gpu/bin/python -m video_search.cli weights download
```

Запустите в двух терминалах из папки проекта:

```bash
.venv-gpu/bin/python -m video_search.cli worker
```

```bash
.venv-gpu/bin/uvicorn video_search.api:app --host 127.0.0.1 --port 8001
```

Локальные данные по умолчанию хранятся в `./data-gpu`.

## API

`POST /api/quick/analyze` принимает `multipart/form-data`: обязательный `file` (видео), `objects` (типы объектов через запятую) и/или `reference_image` (фото объекта). Можно передать `start_utc`, `latitude`, `longitude`, `accuracy_m` и `telemetry_json` с координатами источника по времени. Ответ содержит `job_id`; статус — `GET /api/quick/jobs`. Готовые файлы: `GET /api/quick/jobs/{job_id}/video` и `GET /api/quick/jobs/{job_id}/archive` (ZIP с `annotated.mp4` и `detections.json`). Интерактивная схема: <http://127.0.0.1:8001/docs>.

Текстовый поиск выполняет OWLv2. Поиск по фото сопоставляет изображение объекта с кадрами и лучше работает при похожем ракурсе. Результат следует проверять глазами: модель может пропустить объект или поставить ложную рамку. API не содержит авторизации; для доступа извне нужен защищённый шлюз.
