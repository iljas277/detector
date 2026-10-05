#!/usr/bin/env bash
set -euo pipefail

project_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$project_dir"

for pattern in 'video_search.cli worker' 'uvicorn video_search.api:app'; do
  while read -r pid; do
    [[ -n "$pid" ]] || continue
    process_dir="$(readlink -f "/proc/$pid/cwd" 2>/dev/null || true)"
    if [[ "$process_dir" == "$project_dir" ]]; then
      kill "$pid" 2>/dev/null || true
    fi
  done < <(pgrep -f "$pattern" || true)
done

sleep 1
setsid .venv/bin/python -m video_search.cli worker > data/worker.log 2>&1 < /dev/null &
echo "$!" > data/worker.pid
setsid .venv/bin/uvicorn video_search.api:app --host 127.0.0.1 --port 8765 --no-access-log > data/api.log 2>&1 < /dev/null &
echo "$!" > data/api.pid

for attempt in 1 2 3 4 5 6 7 8 9 10; do
  if curl --silent --fail http://127.0.0.1:8765/api/quick/capabilities > /dev/null \
     && kill -0 "$(cat data/worker.pid)" 2>/dev/null; then
    echo 'Готово: http://127.0.0.1:8765'
    exit 0
  fi
  sleep 1
done

echo 'Сервер не ответил. Проверьте data/api.log и data/worker.log' >&2
exit 1
