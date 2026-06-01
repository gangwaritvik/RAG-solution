const state = {  
  files: [],  
  queryCount: 0,  
  totalChunks: 0,  
};

function setText(id, val) {  
  const el = document.getElementById(id);  
  if (el) el.textContent = val;  
}

const CHUNK_DESCRIPTIONS = {  
  recursive: 'Splits by character count with smart separator fallback — fast &amp; reliable.',  
  semantic: 'Splits by meaning using embeddings — preserves topic context. Slower, uses embedding API.',  
  sliding: 'Overlapping windows of fixed size — ensures no context is lost at chunk boundaries.',  
  fixed: 'Hard splits at exact character count — simple, predictable, no overlap.',  
};

function updateChunkMode() {  
  const select = document.getElementById('chunkModeSelect');  
  const desc = document.getElementById('chunkModeDesc');  
  const mode = select.value;  
  if (desc) desc.innerHTML = CHUNK_DESCRIPTIONS[mode] || '';  
  select.classList.toggle('semantic-active', mode === 'semantic');  
  toast(mode === 'semantic' ? '🧠 Semantic chunking selected' : '⚡ Recursive chunking selected', 'info');  
}

function getChunkMode() {  
  const select = document.getElementById('chunkModeSelect');  
  return select ? select.value : 'recursive';  
}

const zone = document.getElementById('uploadZone');  
const input = document.getElementById('fileInput');

zone.addEventListener('dragover', function(e) {  
  e.preventDefault();  
  zone.classList.add('drag-over');  
});

zone.addEventListener('dragleave', function() {  
  zone.classList.remove('drag-over');  
});

zone.addEventListener('drop', function(e) {  
  e.preventDefault();  
  zone.classList.remove('drag-over');  
  addFiles(Array.from(e.dataTransfer.files));  
});

input.addEventListener('change', function() {  
  addFiles(Array.from(input.files));  
  input.value = '';  
});

function isSupported(f) {  
  const name = f.name.toLowerCase();  
  return f.type === 'application/pdf'  
    || f.type === 'application/vnd.openxmlformats-officedocument.wordprocessingml.document'  
    || name.endsWith('.pdf')  
    || name.endsWith('.docx')  
    || name.endsWith('.doc');  
}

function addFiles(files) {  
  const allowed = files.filter(f =>  
    f.type === 'application/pdf' ||  
    f.type === 'text/csv' ||  
    f.name.toLowerCase().endsWith('.pdf') ||  
    f.name.toLowerCase().endsWith('.docx') ||  
    f.name.toLowerCase().endsWith('.csv')  
  );

  if (!allowed.length) {  
    toast('Only PDF, DOCX, and CSV files accepted.', 'error');  
    return;  
  }

  allowed.forEach(f => {  
    if (state.files.find(x => x.name === f.name)) return;  
    state.files.push({  
      id: Date.now() + Math.random(),  
      file: f,  
      name: f.name,  
      size: fmtSize(f.size),  
      status: 'pending',  
      chunks: []  
    });  
  });

  renderFiles();

  const btn = document.getElementById('processBtn');  
  if (btn) btn.disabled = !state.files.some(f => f.status === 'pending');

  toast(`${allowed.length} file(s) added`, 'success');  
}  


async function loadExistingDocuments() {  
  try {  
    const res = await fetch('/documents');  
    const data = await res.json();  
    if (data.error || !data.documents || !data.documents.length) return;

    data.documents.forEach(function(doc) {  
      if (state.files.find(function(f) { return f.name === doc.filename; })) return;

      const fileSize = (doc.file_size && doc.file_size > 0)  
        ? fmtSize(doc.file_size)  
        : (doc.chunk_count || 0) + ' chunks';

      state.files.push({  
        id: Date.now() + Math.random(),  
        file: null,  
        name: doc.filename,  
        size: fileSize,  
        status: 'done',  
        chunks: [],  
        chunkMode: doc.chunk_type || 'recursive',  
      });

      state.totalChunks += doc.chunk_count || 0;  
    });

    renderFiles();  
    setText('sChunks', state.totalChunks);  
    setText('sFiles', state.files.length);  
    toast('📂 Restored ' + data.documents.length + ' document(s) from database', 'info');  
  } catch (e) {  
    console.warn('[LOAD EXISTING] Failed:', e.message);  
  }  
}

function renderFiles() {  
  const list = document.getElementById('fileList');  
  const noMsg = document.getElementById('noFiles');  
  if (!list) return;

  if (noMsg) noMsg.style.display = state.files.length ? 'none' : 'block';  
  const existing = list.querySelectorAll('.file-item');  
  existing.forEach(function(el) { el.remove(); });

  state.files.forEach(function(f) {  
    const div = document.createElement('div');  
    div.className = 'file-item';  
    div.id = 'fi-' + f.id;

    const methodBadge = f.chunkMode  
      ? '<span class="chunk-method-badge chunk-badge-' + f.chunkMode + '">' + f.chunkMode.toUpperCase() + '</span>'  
      : '';

    const rechunkBtn = f.status === 'done'  
      ? '<button class="file-rechunk-btn" title="Re-chunk with different method" onclick="rechunkFile(\'' + f.id + '\', \'' + esc(f.name) + '\', event)">🔄</button>'  
      : '';

    const deleteBtn = '<button class="file-delete-btn" title="Delete from vector store" onclick="deleteFile(\'' + f.id + '\', \'' + esc(f.name) + '\', event)">🗑️</button>';

    const progressBar = f.status === 'processing'  
      ? '<div class="progress-wrap"><div class="progress-fill" id="pb-' + f.id + '" style="width:0%"></div></div>'  
      : '';

    let chunkToggle = '';  
    if (f.chunks.length > 0) {  
      const chunksHTML = f.chunks.map(function(c, i) {  
        const pageNum = (c.page != null && c.page !== '?') ? c.page : '?';  
        return '<div class="chunk-card">'  
                    + '<div class="chunk-label">Chunk ' + (i + 1) + ' · Page ' + pageNum + '</div>'  
                    + '<div class="chunk-text">' + esc(c.text) + '</div>'  
                    + '</div>';  
      }).join('');

      chunkToggle = '<div class="chunk-toggle" onclick="toggleChunks(\'' + f.id + '\')">'  
                + '<span id="arr-' + f.id + '">▶</span> Chunks (' + f.chunks.length + ')'  
                + '</div>'  
                + '<div class="chunk-list" id="cl-' + f.id + '">' + chunksHTML + '</div>';  
    }
    div.innerHTML = '<div class="file-head">'  
        + '<span class="file-icon">📄</span>'  
        + '<div class="file-info">'  
        + '<div class="file-name" title="' + esc(f.name) + '">' + esc(f.name) + '</div>'  
        + '<div class="file-meta">' + methodBadge + '</div>'  
        + '<span class="file-size-text">' + f.size + '</span>'  
        + '</div>'  
        + '<span class="file-badge badge-' + f.status + '">' + f.status.toUpperCase() + '</span>'  
        + rechunkBtn  
        + deleteBtn  
        + '</div>'  
        + progressBar  
        + chunkToggle;  

    list.appendChild(div);  
  });

  setText('fileCount', state.files.length);  
  setText('sFiles', state.files.length);  
  setText('sChunks', state.totalChunks);  
  setText('sQueries', state.queryCount);  
}

function toggleChunks(id) {  
  const cl = document.getElementById('cl-' + id);  
  const arr = document.getElementById('arr-' + id);  
  if (!cl) return;  
  const open = cl.classList.toggle('open');  
  if (arr) arr.textContent = open ? '▼' : '▶';  
}

async function ingestFiles() {  
  const pending = state.files.filter(function(f) { return f.status === 'pending' && f.file !== null; });  
  if (!pending.length) return;

  const btn = document.getElementById('processBtn');  
  const chunkMode = getChunkMode();  
  if (btn) btn.disabled = true;

  for (let i = 0; i < pending.length; i++) {  
    const f = pending[i];  
    f.status = 'processing';  
    renderFiles();  
    animateProgress(f.id);

    const form = new FormData();  
    form.append('files', f.file, f.name);  
    form.append('chunk_mode', chunkMode);

    try {  
      const res = await fetch('/ingest', { method: 'POST', body: form });  
      const data = await res.json();

      if (data.error) throw new Error(data.error);

      const doc = data.documents && data.documents[0];  
      if (doc) {  
        f.chunks = doc.chunks || [];  
        f.status = 'done';  
        f.chunkMode = chunkMode;  
        state.totalChunks += f.chunks.length;  
        toast('✅ ' + f.name + ' — ' + f.chunks.length + ' chunks (' + chunkMode + ')', 'success');  
      } else {  
        f.status = 'error';  
        toast('No chunks returned for ' + f.name, 'error');  
      }  
    } catch (e) {  
      console.error('[INGEST ERROR]', e);  
      f.status = 'error';  
      toast('Error: ' + e.message, 'error');  
    }

    renderFiles();  
  }

  if (btn) btn.disabled = !state.files.some(function(f) { return f.status === 'pending'; });  
}

function animateProgress(id) {  
  let p = 0;  
  const iv = setInterval(function() {  
    p += 12 + Math.random() * 15;  
    const bar = document.getElementById('pb-' + id);  
    if (bar) bar.style.width = Math.min(p, 90) + '%';  
    if (p >= 100) clearInterval(iv);  
  }, 250);  
  setTimeout(function() { clearInterval(iv); }, 2500);  
}

async function rechunkFile(id, filename, event) {  
  event.stopPropagation();

  const newMode = getChunkMode();  
  const file = state.files.find(function(f) { return String(f.id) === String(id); });  
  if (!file) { toast('File not found.', 'error'); return; }

  if (file.chunkMode === newMode) {  
    toast('Already chunked with "' + newMode + '" — select a different mode first.', 'info');  
    return;  
  }

  const confirmed = confirm('Re-chunk "' + filename + '" using ' + newMode + ' mode?\n\nThis will delete existing vectors and re-embed.');  
  if (!confirmed) return;

  if (!file.file) {  
    toast('Please select "' + filename + '" from your device to re-chunk.', 'info');  
    const picker = document.createElement('input');  
    picker.type = 'file';  
    picker.accept = '.pdf,.docx,.doc';  
    picker.onchange = async function() {  
      const selected = picker.files[0];  
      if (!selected) { toast('No file selected.', 'error'); return; }  
      if (selected.name !== filename) {  
        toast('Please select the correct file: "' + filename + '"', 'error');  
        return;  
      }  
      file.file = selected;  
      await doRechunk(file, filename, newMode);  
    };  
    picker.click();  
    return;  
  }

  await doRechunk(file, filename, newMode);  
}

async function doRechunk(file, filename, newMode) {  
  try {  
    const delRes = await fetch('/delete', {  
      method: 'POST',  
      headers: { 'Content-Type': 'application/json' },  
      body: JSON.stringify({ filename: filename }),  
    });  
    const delData = await delRes.json();  
    if (delData.error) throw new Error(delData.error);

    state.totalChunks = Math.max(0, state.totalChunks - file.chunks.length);  
    file.chunks = [];  
    file.chunkMode = null;  
    file.status = 'processing';  
    renderFiles();  
    animateProgress(file.id);

    const form = new FormData();  
    form.append('files', file.file, file.name);  
    form.append('chunk_mode', newMode);

    const res = await fetch('/ingest', { method: 'POST', body: form });  
    const data = await res.json();  
    if (data.error) throw new Error(data.error);

    const doc = data.documents && data.documents[0];  
    if (doc) {  
      file.chunks = doc.chunks || [];  
      file.status = 'done';  
      file.chunkMode = newMode;  
      state.totalChunks += file.chunks.length;  
      toast('✅ "' + filename + '" re-chunked — ' + file.chunks.length + ' chunks (' + newMode + ')', 'success');  
    } else {  
      file.status = 'error';  
      toast('Re-chunk failed for ' + filename, 'error');  
    }  
  } catch (e) {  
    console.error('[RECHUNK ERROR]', e);  
    file.status = 'error';  
    toast('Re-chunk failed: ' + e.message, 'error');  
  }

  renderFiles();  
}

function handleKey(e) {  
  if (e.key === 'Enter' && !e.shiftKey) {  
    e.preventDefault();  
    submitQuery();  
  }  
}

function resize(el) {  
  el.style.height = 'auto';  
  el.style.height = Math.min(el.scrollHeight, 150) + 'px';  
}

async function submitQuery() {  
  const queryEl = document.getElementById('queryInput');  
  const query = queryEl ? queryEl.value.trim() : '';  
  if (!query) { toast('Enter a question first.', 'error'); return; }

  const ready = state.files.filter(function(f) { return f.status === 'done'; });  
  if (!ready.length) { toast('Process at least one file first.', 'error'); return; }

  setLoading(true);  
  const t0 = Date.now();

  try {  
    const topKEl = document.getElementById('topK');  
    const tempEl = document.getElementById('temp');

    const res = await fetch('/query', {  
      method: 'POST',  
      headers: { 'Content-Type': 'application/json' },  
      body: JSON.stringify({  
        query: query,  
        top_k: parseInt(topKEl ? topKEl.value : '5') || 5,  
        temperature: parseFloat(tempEl ? tempEl.value : '0.2') || 0.2,  
      }),  
    });

    const data = await res.json();  
    if (data.error) throw new Error(data.error);

    const elapsed = ((Date.now() - t0) / 1000).toFixed(2) + 's';  
    state.queryCount++;  
    setText('sQueries', state.queryCount);

    renderAnswer(query, data.answer, data.sources, elapsed);  
    queryEl.value = '';  
    resize(queryEl);  
  } catch (e) {  
    toast('Error: ' + e.message, 'error');  
  }

  setLoading(false);  
}

function renderAnswer(query, answer, sources, elapsed) {  
  const area = document.getElementById('results');  
  const empty = document.getElementById('emptyState');  
  if (empty) empty.remove();

  const cardId = 'card-' + Date.now();  
  const card = document.createElement('div');  
  card.className = 'answer-card expanded';  
  card.id = cardId;

  let sourcesHTML = '';  
  if (sources && sources.length) {  
    sources.forEach(function(s, i) {  
      const ctype = s.chunk_type || 'recursive';  
      const chunkNum = (s.chunk_index != null ? s.chunk_index : i) + 1;  
      const score = s.score ? s.score.toFixed(3) : '—';  
      sourcesHTML += '<div class="source-card" onclick="this.classList.toggle(\'open\')">'  
                + '<div class="src-top">'  
                + '<span class="src-file">📄 ' + esc(s.filename) + '</span>'  
                + '<span class="src-tag">Chunk #' + chunkNum + '</span>'  
                + '<span class="chunk-type-tag ' + ctype + '">' + ctype + '</span>'  
                + '</div>'  
                + '<div class="src-meta">Page ' + s.page + ' · Similarity: <span class="sim">' + score + '</span></div>'  
                + '<div class="src-preview">' + esc(s.text) + '</div>'  
                + '<div class="src-full">' + esc(s.text) + '</div>'  
                + '</div>';  
    });  
  }

  card.innerHTML = '<div class="card-head" onclick="toggleCard(\'' + cardId + '\')">'  
        + '<div class="card-head-left">'  
        + '<div class="q-tag">Question · GPT-4.1 · ' + elapsed + '</div>'  
        + '<div class="q-text">' + esc(query) + '</div>'  
        + '</div>'  
        + '<div class="card-expand-btn" title="Expand / Collapse">▲</div>'  
        + '</div>'  
        + '<div class="card-body-wrap">'  
        + '<div class="card-body">'  
        + '<div class="a-tag">GPT-4.1 Answer</div>'  
        + '<div class="a-text">' + fmtAnswer(answer) + '</div>'  
        + '</div>'  
        + '<div class="sources-wrap">'  
        + '<div class="sources-label">📌 Retrieved Sources (' + (sources ? sources.length : 0) + ')</div>'  
        + '<div class="sources-grid">' + sourcesHTML + '</div>'  
        + '</div>'  
        + '</div>';

  area.insertBefore(card, area.firstChild);

  if (window.MathJax) {  
    MathJax.typesetPromise([card]).catch(function(err) { console.error('MathJax error:', err); });  
  }

  card.scrollIntoView({ behavior: 'smooth', block: 'start' });  
}

function toggleCard(cardId) {  
  const card = document.getElementById(cardId);  
  if (!card) return;  
  card.classList.toggle('expanded');  
}

function setLoading(on) {  
  const btn = document.getElementById('askBtn');  
  const label = document.getElementById('askLabel');  
  if (btn) btn.disabled = on;  
  if (label) label.innerHTML = on ? '<span class="spin"></span> Thinking…' : '➤ Ask';  
}

async function deleteFile(id, filename, event) {  
  event.stopPropagation();

  const confirmed = confirm('Delete "' + filename + '" from the vector store?');  
  if (!confirmed) return;

  try {  
    const res = await fetch('/delete', {  
      method: 'POST',  
      headers: { 'Content-Type': 'application/json' },  
      body: JSON.stringify({ filename: filename }),  
    });

    const data = await res.json();  
    if (data.error) throw new Error(data.error);

    const file = state.files.find(function(f) { return String(f.id) === String(id); });  
    if (file) {  
      state.totalChunks = Math.max(0, state.totalChunks - file.chunks.length);  
    }  
    state.files = state.files.filter(function(f) { return String(f.id) !== String(id); });

    renderFiles();

    const btn = document.getElementById('processBtn');  
    if (btn) btn.disabled = !state.files.some(function(f) { return f.status === 'pending'; });

    toast('🗑️ "' + filename + '" deleted — ' + data.deleted_vectors + ' vectors removed', 'success');  
  } catch (e) {  
    console.error('[DELETE ERROR]', e);  
    toast('Delete failed: ' + e.message, 'error');  
  }  
}

function esc(str) {  
  return String(str)  
    .replace(/&/g, '&amp;')  
    .replace(/</g, '&lt;')  
    .replace(/>/g, '&gt;')  
    .replace(/"/g, '&quot;');  
}

function fmtAnswer(text) {  
  let out = text;  
  out = out.replace(/^### (.+)$/gm, '<h4 class="ans-h4">$1</h4>');  
  out = out.replace(/^## (.+)$/gm, '<h3 class="ans-h3">$1</h3>');  
  out = out.replace(/^# (.+)$/gm, '<h2 class="ans-h2">$1</h2>');  
  out = out.replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>');  
  out = out.replace(/\*(.+?)\*/g, '<em>$1</em>');  
  out = out.replace(/^---+$/gm, '<hr class="ans-hr"/>');  
  out = out.replace(/^[\-\*] (.+)$/gm, '<li class="ans-li">$1</li>');  
  out = out.replace(/(<li class="ans-li">[\s\S]*?<\/li>)/g, '<ul class="ans-ul">$1</ul>');  
  out = out.replace(/<\/ul>\s*<ul class="ans-ul">/g, '');  
  out = out.replace(/^\d+\. (.+)$/gm, '<li class="ans-li">$1</li>');  
  out = out.replace(/`([^`]+)`/g, '<code class="ans-code">$1</code>');  
  out = out.replace(/\n(?!<(h[2-4]|ul|li|hr))/g, '<br/>');  
  return out;  
}

function fmtSize(b) {  
  if (b < 1024) return b + ' B';  
  if (b < 1048576) return (b / 1024).toFixed(1) + ' KB';  
  return (b / 1048576).toFixed(1) + ' MB';  
}

function toast(msg, type) {  
  type = type || 'info';  
  const wrap = document.getElementById('toasts');  
  if (!wrap) return;  
  const el = document.createElement('div');  
  el.className = 'toast ' + type;  
  el.innerHTML = '<span class="tdot"></span>' + msg;  
  wrap.appendChild(el);  
  setTimeout(function() { el.remove(); }, 3800);  
}

async function clearAndStart() {  
  try {  
    await fetch('/clear', { method: 'POST', headers: { 'Content-Type': 'application/json' } });  
  } catch (e) {  
    console.warn('[CLEAR] Failed:', e.message);  
  }  
}

clearAndStart();  
