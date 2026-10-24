const $ = (s, el = document) => el.querySelector(s);
const state = { presets: [], samples: [], preset: null, file: null, sample: null, options: {}, maxSeconds: 8, es: null, timer: null };

async function boot() {
  const r = await fetch('/api/presets');
  const data = await r.json();
  state.presets = data.presets;
  state.samples = data.samples;
  state.maxSeconds = data.max_seconds;
  $('#duration').max = state.maxSeconds;
  renderGrid();
}

function renderGrid() {
  const grid = $('#grid');
  grid.innerHTML = '';
  for (const p of state.presets) {
    const card = document.createElement('button');
    card.className = 'card';
    card.type = 'button';
    card.innerHTML = `
      <video src="/media/${p.preview}" muted loop autoplay playsinline preload="metadata"></video>
      <span class="tag">like Higgsfield ${p.higgsfield}</span>
      <div class="meta"><h3>${p.title}</h3><p>${p.tagline}</p></div>`;
    card.addEventListener('click', () => openSheet(p));
    grid.appendChild(card);
  }
}

function openSheet(p) {
  state.preset = p;
  state.options = {};
  $('#sheet-title').textContent = p.title;
  $('#sheet-tagline').textContent = p.tagline;
  $('#sheet-hf').textContent = `Open-source take on Higgsfield ${p.higgsfield}`;
  $('#best-with').textContent = `Best with: ${p.best_with}`;
  $('#anycam-wrap').hidden = !p.workflow_any_camera;
  renderOptions(p);
  renderSamples();
  resetRun();
  $('#sheet').hidden = false;
}

function closeSheet() {
  $('#sheet').hidden = true;
  if (state.es) state.es.close();
  clearInterval(state.timer);
}

function renderOptions(p) {
  const box = $('#options');
  box.innerHTML = '';
  for (const o of p.options) {
    state.options[o.id] = o.default;
    const wrap = document.createElement('div');
    wrap.className = 'opt';
    if (o.choices) {
      wrap.innerHTML = `<span class="label">${o.label}</span><div class="chips"></div>`;
      const chips = $('.chips', wrap);
      for (const c of o.choices) {
        const b = document.createElement('button');
        b.type = 'button';
        b.className = 'chip' + (c === o.default ? ' sel' : '');
        b.textContent = c;
        b.addEventListener('click', () => {
          state.options[o.id] = c;
          chips.querySelectorAll('.chip').forEach(x => x.classList.toggle('sel', x === b));
        });
        chips.appendChild(b);
      }
    } else {
      wrap.innerHTML = `<label>${o.label} <output>${o.default}</output>
        <input type="range" min="${o.min}" max="${o.max}" step="${o.step}" value="${o.default}"></label>`;
      const input = $('input', wrap), out = $('output', wrap);
      input.addEventListener('input', () => { state.options[o.id] = Number(input.value); out.textContent = input.value; });
    }
    box.appendChild(wrap);
  }
}

function renderSamples() {
  const box = $('#samples');
  box.innerHTML = '';
  for (const s of state.samples) {
    const b = document.createElement('button');
    b.type = 'button';
    b.className = 'sample' + (state.sample === s ? ' sel' : '');
    b.innerHTML = `<img src="/samples/${s.replace(/\.mp4$/, '.jpg')}" alt=""><span>${s.replace(/\.mp4$/, '').replace(/[_-]/g, ' ')}</span>`;
    b.addEventListener('click', () => pickSample(s));
    box.appendChild(b);
  }
}

function pickSample(s) {
  state.sample = s;
  state.file = null;
  $('#file').value = '';
  $('#drop').classList.add('has');
  $('#drop-label').innerHTML = `Sample: <b>${s}</b>`;
  renderSamples();
}

function pickFile(f) {
  if (!f) return;
  state.file = f;
  state.sample = null;
  $('#drop').classList.add('has');
  $('#drop-label').innerHTML = `<b>${f.name}</b> · ${(f.size / 1e6).toFixed(1)} MB`;
  renderSamples();
}

function resetRun() {
  $('#create').hidden = false;
  $('#run').hidden = true;
  $('#result').hidden = true;
  $('#error').hidden = true;
  $('#go').disabled = false;
  $('#fill').style.width = '0%';
}

async function submit(ev) {
  ev.preventDefault();
  if (!state.file && !state.sample) {
    showError('Drop in a clip or pick one of the samples first.');
    return;
  }
  $('#error').hidden = true;
  $('#go').disabled = true;
  const fd = new FormData();
  fd.append('preset', state.preset.id);
  fd.append('options', JSON.stringify(state.options));
  fd.append('start', $('#start').value);
  fd.append('duration', $('#duration').value);
  fd.append('any_camera', $('#anycam').checked ? 'true' : 'false');
  if (state.file) fd.append('file', state.file); else fd.append('sample', state.sample);
  $('#create').hidden = true;
  $('#run').hidden = false;
  $('#stage').textContent = 'Uploading clip…';
  const t0 = Date.now();
  clearInterval(state.timer);
  state.timer = setInterval(() => { $('#elapsed').textContent = `${Math.round((Date.now() - t0) / 1000)}s`; }, 500);
  let r;
  try {
    r = await fetch('/api/jobs', { method: 'POST', body: fd });
  } catch (e) {
    return fail('Could not reach the app server.');
  }
  if (!r.ok) return fail(await r.text());
  const { id } = await r.json();
  follow(id);
}

function follow(id) {
  if (state.es) state.es.close();
  const es = new EventSource(`/api/jobs/${id}/events`);
  state.es = es;
  es.onmessage = (m) => {
    const j = JSON.parse(m.data);
    $('#fill').style.width = `${Math.round(j.progress * 100)}%`;
    $('#stage').textContent = j.stage || j.status;
    if (j.status === 'done') {
      es.close();
      clearInterval(state.timer);
      showResult(id, j);
    } else if (j.status === 'error') {
      es.close();
      fail(j.error || 'The workflow failed.');
    }
  };
  es.onerror = () => { /* EventSource retries on its own */ };
}

function showResult(id, j) {
  $('#run').hidden = true;
  $('#result').hidden = false;
  const v = $('#out');
  v.src = `${j.video}?t=${Date.now()}`;
  v.play().catch(() => {});
  $('#dl').href = j.video;
  $('#wf').href = `/api/jobs/${id}/workflow`;
}

function fail(msg) {
  clearInterval(state.timer);
  $('#run').hidden = true;
  $('#create').hidden = false;
  $('#go').disabled = false;
  showError(msg);
}

function showError(msg) {
  const e = $('#error');
  e.textContent = msg;
  e.hidden = false;
}

document.addEventListener('click', (e) => { if (e.target.matches('[data-close]')) closeSheet(); });
document.addEventListener('keydown', (e) => { if (e.key === 'Escape') closeSheet(); });
$('#create').addEventListener('submit', submit);
$('#again').addEventListener('click', resetRun);
$('#file').addEventListener('change', (e) => pickFile(e.target.files[0]));
const drop = $('#drop');
['dragenter', 'dragover'].forEach(t => drop.addEventListener(t, (e) => { e.preventDefault(); drop.classList.add('over'); }));
['dragleave', 'drop'].forEach(t => drop.addEventListener(t, (e) => { e.preventDefault(); drop.classList.remove('over'); }));
drop.addEventListener('drop', (e) => pickFile(e.dataTransfer.files[0]));
for (const [id, out] of [['start', 'start-out'], ['duration', 'dur-out']]) {
  const el = $('#' + id);
  el.addEventListener('input', () => { $('#' + out).textContent = `${Number(el.value).toFixed(1)}s`; });
}
boot();
