const $ = (selector) => document.querySelector(selector);
const esc = (value) => String(value ?? '').replace(/[&<>"']/g, char => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[char]));
let busy = false;
let apiReady = false;
let selectedJobId = null;
let preferredJobId = null;
let manualSelection = false;
const namesRu = {person:'человек',car:'машина',truck:'грузовик',bus:'автобус',motorcycle:'мотоцикл',bicycle:'велосипед',van:'фургон',boat:'лодка',airplane:'самолёт',helicopter:'вертолёт',drone:'дрон',dog:'собака',cat:'кошка',bird:'птица',building:'здание',tree:'дерево',tent:'палатка',awning:'тент',umbrella:'зонт',horse:'лошадь',cow:'корова',backpack:'рюкзак',suitcase:'чемодан'};

async function refresh() {
  try {
    if (!apiReady) {
      const capabilities = await fetch('/api/quick/capabilities');
      if (!capabilities.ok || (await capabilities.json()).version < 7) {
        $('#startButton').disabled = true;
        $('#message').textContent = 'Сервер и обработчик нужно перезапустить для обновлённого поиска по фото.';
        $('#message').className = 'error';
        return;
      }
      apiReady = true;
      $('#startButton').disabled = false;
      $('#message').textContent = '';
    }
    const response = await fetch('/api/quick/jobs');
    if (!response.ok) throw Error('Не удалось получить список видео');
    const jobs = await response.json();
    $('#jobs').innerHTML = jobs.length ? jobs.map(job => {
      const percent = Math.round(100 * job.progress);
      const status = job.status === 'done' ? 'Готово' : job.status === 'running' ? `Обработка: ${percent}%` : job.status === 'queued' ? 'В очереди' : job.status === 'failed' ? 'Ошибка' : 'Отменено';
      const objects = [...job.objects.map(name => namesRu[name] || name), ...(job.has_reference ? ['по фото'] : [])].join(', ') || 'общий обзор';
      return `<div class="item"><img class="thumb" src="${job.preview_url}" alt=""><div><strong>${esc(job.name)}</strong><span class="muted small">Ищем: ${esc(objects)} · ${status} · ${job.duration_s.toFixed(1)} с</span>${job.status === 'running' ? `<div class="bar"><span style="width:${percent}%"></span></div>` : ''}${job.error ? `<p class="error small">${esc(job.error)}</p>` : ''}${job.video_url ? `<div><button type="button" onclick="watch(${job.id}, true)">Смотреть</button> <a class="button" href="${job.video_url}" download="objects-${job.id}.mp4">Скачать MP4</a> ${job.archive_url ? `<a class="button" href="${job.archive_url}" download="objects-${job.id}.zip">Скачать ZIP с данными</a>` : ''}</div>` : ''}</div></div>`;
    }).join('') : '<p class="muted">Пока нет обработанных видео.</p>';
    const ready = preferredJobId == null ? jobs.find(job => job.video_url) : jobs.find(job => job.id === preferredJobId && job.video_url);
    if (!manualSelection && ready && selectedJobId !== ready.id) watch(ready.id, false);
  } catch (error) {
    $('#message').textContent = error.message;
    $('#message').className = 'error';
  }
}

function watch(id, manual = true) {
  manualSelection = manual;
  selectedJobId = id;
  $('#viewerTitle').textContent = `Видео с рамками #${id}`;
  $('#player').src = `/api/quick/jobs/${id}/video`;
  $('#download').href = `/api/quick/jobs/${id}/video`;
  $('#download').download = `objects-${id}.mp4`;
  $('#viewer').classList.add('open');
  if (manual) $('#viewer').scrollIntoView({behavior:'smooth'});
}

$('#file').addEventListener('change', () => {
  const file = $('#file').files[0];
  $('#fileLabel').textContent = file ? file.name : 'Выберите видео или перетащите его сюда';
});

const dropzone = $('#dropzone');
dropzone.addEventListener('dragover', event => {event.preventDefault();dropzone.classList.add('drag')});
dropzone.addEventListener('dragleave', () => dropzone.classList.remove('drag'));
dropzone.addEventListener('drop', event => {
  event.preventDefault();dropzone.classList.remove('drag');
  if (event.dataTransfer.files.length) {
    $('#file').files = event.dataTransfer.files;
    $('#file').dispatchEvent(new Event('change'));
  }
});

$('#uploadForm').addEventListener('submit', async event => {
  event.preventDefault();
  if (!apiReady) return;
  const file = $('#file').files[0];
  const reference = $('#reference').files[0];
  if (!file || busy) return;
  if (!$('#objects').value.trim() && !reference) {
    $('#message').textContent = 'Напишите, что искать, или добавьте фото объекта';
    $('#message').className = 'error';
    return;
  }
  busy = true;
  $('#startButton').disabled = true;
  $('#message').textContent = 'Загружаем видео…';
  $('#message').className = 'muted';
  try {
    const data = new FormData();
    data.set('file', file);
    data.set('objects', $('#objects').value);
    if (reference) data.set('reference_image', reference);
    const response = await fetch('/api/quick/analyze', {method:'POST', body:data});
    const result = await response.json();
    if (!response.ok) throw Error(result.detail || 'Не удалось загрузить видео');
    preferredJobId = result.job_id;
    manualSelection = false;
    $('#message').textContent = result.cached ? 'Готовый результат найден ниже.' : 'Видео добавлено в очередь. Следите за прогрессом ниже.';
    $('#message').className = '';
    $('#file').value = '';
    $('#reference').value = '';
    $('#fileLabel').textContent = 'Выберите видео или перетащите его сюда';
    await refresh();
  } catch (error) {
    $('#message').textContent = error.message;
    $('#message').className = 'error';
  } finally {
    busy = false;
    $('#startButton').disabled = false;
  }
});

refresh();
setInterval(refresh, 3000);
