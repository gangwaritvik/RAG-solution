var state = {  
    files: [],  
    queryCount: 0,  
    totalChunks: 0  
};

var qsExpanded = true;


function setText(id, val) {  
    var el = document.getElementById(id);  
    if (el) el.textContent = val;  
}

var CHUNK_DESCRIPTIONS = {  
    recursive: 'Splits by character count with smart separator fallback — fast &amp; reliable.',  
    semantic:  'Splits by meaning using embeddings — preserves topic context. Slower, uses embedding API.',  
    sliding:   'Overlapping windows of fixed size — ensures no context is lost at chunk boundaries.',  
    fixed:     'Hard splits at exact character count — simple, predictable, no overlap.'  
};

function updateChunkMode() {  
    var select = document.getElementById('chunkModeSelect');  
    var desc   = document.getElementById('chunkModeDesc');  
    var mode   = select.value;  
    if (desc) desc.innerHTML = CHUNK_DESCRIPTIONS[mode] || '';  
    toast(mode === 'semantic' ? '🧠 Semantic chunking selected' : '⚡ ' + mode + ' chunking selected', 'info');  
}

function getChunkMode() {  
    var select = document.getElementById('chunkModeSelect');  
    return select ? select.value : 'recursive';  
}

/* ══ TOP-K MAX TOGGLE ══ */  
var topKMaxActive = false;

function toggleTopKMax() {  
    topKMaxActive = !topKMaxActive;  
    var btn     = document.getElementById('btnTopKMax');  
    var slider  = document.getElementById('topK');  
    var display = document.getElementById('topKDisplay');

    if (topKMaxActive) {  
        if (btn)     btn.classList.add('active');  
        if (display) display.textContent = 'ALL';  
        if (slider)  slider.disabled = true;  
        toast('Top-K set to MAX — all chunks will be retrieved', 'info');  
    } else {  
        if (btn)     btn.classList.remove('active');  
        if (slider)  slider.disabled = false;  
        var val = slider ? slider.value : '5';  
        if (display) display.textContent = val;  
        toast('Top-K restored to ' + val, 'info');  
    }  
}

function updateTopK(val) {  
    var display = document.getElementById('topKDisplay');  
    if (!topKMaxActive && display) display.textContent = val;  
}

function updateTemp(val) {  
    var display = document.getElementById('tempDisplay');  
    if (display) display.textContent = parseFloat(val).toFixed(1);  
}

function getTopK() {  
    if (topKMaxActive) return state.totalChunks || 9999;  
    var el = document.getElementById('topK');  
    return parseInt(el ? el.value : '5') || 5;  
}

function getTemp() {  
    var el = document.getElementById('temp');  
    return parseFloat(el ? el.value : '0.2') || 0.2;  
}

/* ══ UPLOAD ZONE ══ */  
document.addEventListener('DOMContentLoaded', function() {  
    var zone  = document.getElementById('uploadZone');  
    var input = document.getElementById('fileInput');

    if (!zone || !input) {  
        console.error('[UPLOAD] uploadZone or fileInput not found in DOM');  
        return;  
    }

    zone.addEventListener('dragover', function(e) {  
        e.preventDefault();  
        e.stopPropagation();  
        zone.classList.add('drag-over');  
    });

    zone.addEventListener('dragenter', function(e) {  
        e.preventDefault();  
        e.stopPropagation();  
        zone.classList.add('drag-over');  
    });

    zone.addEventListener('dragleave', function(e) {  
        e.preventDefault();  
        e.stopPropagation();  
        zone.classList.remove('drag-over');  
    });

    zone.addEventListener('drop', function(e) {  
        e.preventDefault();  
        e.stopPropagation();  
        zone.classList.remove('drag-over');  
        var files = e.dataTransfer ? Array.from(e.dataTransfer.files) : [];  
        if (files.length) addFiles(files);  
    });

    zone.addEventListener('click', function() {  
        input.click();  
    });

    input.addEventListener('change', function() {  
        addFiles(Array.from(input.files));  
        input.value = '';  
    });  
});

function isSupported(f) {  
    var name = f.name.toLowerCase();  
    return f.type === 'application/pdf'  
        || f.type === 'application/vnd.openxmlformats-officedocument.wordprocessingml.document'  
        || f.type === 'text/plain'  
        || name.endsWith('.pdf')  
        || name.endsWith('.docx')  
        || name.endsWith('.doc');  
}

function addFiles(files) {  
    var supported = files.filter(isSupported);  
    if (!supported.length) {  
        toast('Only PDF and DOCX files accepted.', 'error');  
        return;  
    }  
    for (var i = 0; i < supported.length; i = i + 1) {  
        var f = supported[i];  
        var exists = false;  
        for (var j = 0; j < state.files.length; j = j + 1) {  
            if (state.files[j].name === f.name) { exists = true; break; }  
        }  
        if (exists) continue;  
        state.files.push({  
            id: Date.now() + Math.random(),  
            file: f,  
            name: f.name,  
            size: fmtSize(f.size),  
            status: 'pending',  
            chunks: [],  
            chunkMode: null  
        });  
    }  
    renderFiles();  
    var btn = document.getElementById('processBtn');  
    var hp  = false;  
    for (var i = 0; i < state.files.length; i = i + 1) {  
        if (state.files[i].status === 'pending') { hp = true; break; }  
    }  
    if (btn) btn.disabled = !hp;  
    toast(supported.length + ' file(s) added', 'success');  
}

/* ══ RENDER FILES ══ */  
function renderFiles() {  
    var list  = document.getElementById('fileList');  
    var noMsg = document.getElementById('noFiles');  
    if (!list) return;

    if (noMsg) noMsg.style.display = state.files.length ? 'none' : 'block';  
    var ex = list.querySelectorAll('.file-item');  
    for (var i = 0; i < ex.length; i = i + 1) { ex[i].remove(); }

    for (var i = 0; i < state.files.length; i = i + 1) {  
        var f   = state.files[i];  
        var div = document.createElement('div');  
        div.className = 'file-item';  
        div.id = 'fi-' + f.id;  
        div.style.animationDelay = (i * 0.05) + 's';

        var methodBadge = f.chunkMode  
            ? '<span class="chunk-method-badge chunk-badge-' + f.chunkMode + '">' + f.chunkMode.toUpperCase() + '</span>'  
            : '';

        var rechunkBtn = f.status === 'done'  
            ? '<button class="file-rechunk-btn" title="Re-chunk" onclick="rechunkFile(\'' + f.id + '\', \'' + esc(f.name) + '\', event)">&#x1F504;</button>'  
            : '';

        var deleteBtn = '<button class="file-delete-btn" title="Delete" onclick="deleteFile(\'' + f.id + '\', \'' + esc(f.name) + '\', event)">&#x1F5D1;</button>';

        var progressBar = f.status === 'processing'  
            ? '<div class="progress-wrap"><div class="progress-fill" id="pb-' + f.id + '" style="width:0%"></div></div>'  
            : '';

        var chunkToggle = '';  
        if (f.chunks.length > 0) {  
            var chunksHTML = '';  
            for (var k = 0; k < f.chunks.length; k = k + 1) {  
                var c       = f.chunks[k];  
                var pageNum = (c.page != null && c.page !== '?') ? c.page : '?';  
                chunksHTML = chunksHTML  
                                        + '<div class="chunk-card" onclick="openChunkModal(\''  
                                        + esc(f.name) + '\', ' + (k + 1) + ', ' + pageNum  
                                        + ', \'' + esc(c.text || '') + '\', \''  
                                        + (f.chunkMode || 'recursive') + '\')">'  
                                        + '<div class="chunk-label">Chunk ' + (k + 1) + ' · Page ' + pageNum + '</div>'  
                                        + '<div class="chunk-text">' + esc(c.text || '') + '</div>'  
                                        + '</div>';  
            }  
            chunkToggle = '<div class="chunk-toggle" onclick="toggleChunks(\'' + f.id + '\')">'  
                                + '<span id="arr-' + f.id + '">&#9658;</span> Chunks (' + f.chunks.length + ')'  
                                + '</div>'  
                                + '<div class="chunk-list" id="cl-' + f.id + '">' + chunksHTML + '</div>';  
        }

        div.innerHTML = '<div class="file-head">'  
                        + '<span class="file-icon">&#128196;</span>'  
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
    }

    setText('fileCount', state.files.length);  
    setText('sFiles',    state.files.length);  
    setText('sChunks',   state.totalChunks);  
    setText('sQueries',  state.queryCount);  
}

function toggleChunks(id) {  
    var cl  = document.getElementById('cl-' + id);  
    var arr = document.getElementById('arr-' + id);  
    if (!cl) return;  
    var open = cl.classList.toggle('open');  
    if (arr) arr.innerHTML = open ? '&#9660;' : '&#9658;';  
}



function toggleQuerySettings(){    
    qsExpanded = !qsExpanded;  
    var body  = document.getElementById('qsBody');  
    var arrow = document.getElementById('qsArrow');  
    if (body)  body.style.display  = qsExpanded ? 'flex' : 'none';    
    if (arrow) arrow.innerHTML = qsExpanded ? '&#x25BE;' : '&#x25B8;';  
}  


/* ══ CHUNK MODAL ══ */  
function openChunkModal(filename, chunkNum, page, text, chunkType) {  
    var modal   = document.getElementById('chunkModal');  
    var titleEl = document.getElementById('modalTitle');  
    var metaEl  = document.getElementById('modalMeta');  
    var bodyEl  = document.getElementById('modalBody');  
    if (!modal) return;  
    titleEl.innerHTML = esc(filename);  
    metaEl.innerHTML  = '<span class="meta-pill">Chunk <strong>' + chunkNum + '</strong></span>'  
                + '<span class="meta-pill">Page <strong>' + page + '</strong></span>'  
                + '<span class="meta-pill">Method <strong>' + chunkType.toUpperCase() + '</strong></span>';  
    bodyEl.textContent = text;  
    modal.classList.add('open');  
    document.body.style.overflow = 'hidden';  
}

function closeChunkModal() {  
    var modal = document.getElementById('chunkModal');  
    if (modal) modal.classList.remove('open');  
    document.body.style.overflow = '';  
}

document.addEventListener('keydown', function(e) {  
    if (e.key === 'Escape') closeChunkModal();  
});

/* ══ INGEST ══ */  
async function ingestFiles() {  
    var pending = [];  
    for (var i = 0; i < state.files.length; i = i + 1) {  
        if (state.files[i].status === 'pending' && state.files[i].file !== null) {  
            pending.push(state.files[i]);  
        }  
    }  
    if (!pending.length) return;

    var btn       = document.getElementById('processBtn');  
    var chunkMode = getChunkMode();  
    if (btn) btn.disabled = true;

    for (var i = 0; i < pending.length; i = i + 1) {  
        var f = pending[i];  
        f.status = 'processing';  
        renderFiles();  
        animateProgress(f.id);

        var form = new FormData();  
        form.append('files',      f.file, f.name);  
        form.append('chunk_mode', chunkMode);

        try {  
            var res  = await fetch('/ingest', { method: 'POST', body: form });  
            var data = await res.json();  
            if (data.error) throw new Error(data.error);

            var doc = data.documents && data.documents[0];  
            if (doc) {  
                f.chunks          = doc.chunks || [];  
                f.status          = 'done';  
                f.chunkMode       = chunkMode;  
                state.totalChunks = state.totalChunks + f.chunks.length;  
                toast('✅ ' + f.name + ' — ' + f.chunks.length + ' chunks', 'success');  
            } else {  
                f.status = 'error';  
                toast('No chunks for ' + f.name, 'error');  
            }  
        } catch (e) {  
            f.status = 'error';  
            toast('Error: ' + e.message, 'error');  
        }

        renderFiles();  
    }

    var hp = false;  
    for (var i = 0; i < state.files.length; i = i + 1) {  
        if (state.files[i].status === 'pending') { hp = true; break; }  
    }  
    if (btn) btn.disabled = !hp;  
}

function animateProgress(id) {  
    var p  = 0;  
    var iv = setInterval(function() {  
        p = p + 12 + Math.random() * 15;  
        var bar = document.getElementById('pb-' + id);  
        if (bar) bar.style.width = Math.min(p, 90) + '%';  
        if (p >= 100) clearInterval(iv);  
    }, 250);  
    setTimeout(function() { clearInterval(iv); }, 2500);  
}

/* ══ RE-CHUNK ══ */  
async function rechunkFile(id, filename, event) {  
    event.stopPropagation();  
    var newMode = getChunkMode();  
    var file    = null;  
    for (var i = 0; i < state.files.length; i = i + 1) {  
        if (String(state.files[i].id) === String(id)) { file = state.files[i]; break; }  
    }  
    if (!file)                      { toast('File not found.', 'error'); return; }  
    if (file.chunkMode === newMode) { toast('Already chunked with "' + newMode + '".', 'info'); return; }

    if (!file.file) {  
        toast('Select "' + filename + '" from your device.', 'info');  
        var picker    = document.createElement('input');  
        picker.type   = 'file';  
        picker.accept = '.pdf,.docx,.doc';  
        picker.onchange = async function() {  
            var s = picker.files[0];  
            if (!s)                  { toast('No file selected.', 'error'); return; }  
            if (s.name !== filename) { toast('Select the correct file.', 'error'); return; }  
            file.file = s;  
            if (!confirm('Re-chunk "' + filename + '" using ' + newMode + '?')) return;  
            await doRechunk(file, filename, newMode);  
        };  
        picker.click();  
        return;  
    }  
    if (!confirm('Re-chunk "' + filename + '" using ' + newMode + '?')) return;  
    await doRechunk(file, filename, newMode);  
}

async function doRechunk(file, filename, newMode) {  
    try {  
        var delRes  = await fetch('/delete', {  
            method:  'POST',  
            headers: { 'Content-Type': 'application/json' },  
            body:    JSON.stringify({ filename: filename })  
        });  
        var delData = await delRes.json();  
        if (delData.error) throw new Error(delData.error);

        state.totalChunks = Math.max(0, state.totalChunks - file.chunks.length);  
        file.chunks    = [];  
        file.chunkMode = null;  
        file.status    = 'processing';  
        renderFiles();  
        animateProgress(file.id);

        var form = new FormData();  
        form.append('files',      file.file, file.name);  
        form.append('chunk_mode', newMode);

        var res  = await fetch('/ingest', { method: 'POST', body: form });  
        var data = await res.json();  
        if (data.error) throw new Error(data.error);

        var doc = data.documents && data.documents[0];  
        if (doc) {  
            file.chunks       = doc.chunks || [];  
            file.status       = 'done';  
            file.chunkMode    = newMode;  
            state.totalChunks = state.totalChunks + file.chunks.length;  
            toast('✅ Re-chunked — ' + file.chunks.length + ' chunks', 'success');  
        } else {  
            file.status = 'error';  
            toast('Re-chunk failed.', 'error');  
        }  
    } catch (e) {  
        file.status = 'error';  
        toast('Re-chunk failed: ' + e.message, 'error');  
    }  
    renderFiles();  
}

/* ══ QUERY — CHAT ══ */  
function handleKey(e) {  
    if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); submitQuery(); }  
}

function resize(el) {  
    el.style.height = 'auto';  
    el.style.height = Math.min(el.scrollHeight, 120) + 'px';  
}

async function submitQuery() {  
    var queryEl = document.getElementById('queryInput');  
    var query   = queryEl ? queryEl.value.trim() : '';  
    if (!query) { toast('Enter a question first.', 'error'); return; }

    var hasReady = false;  
    for (var i = 0; i < state.files.length; i = i + 1) {  
        if (state.files[i].status === 'done') { hasReady = true; break; }  
    }  
    if (!hasReady) { toast('Process at least one file first.', 'error'); return; }

    queryEl.value = '';  
    resize(queryEl);  
    setLoading(true);

    var empty = document.getElementById('chatEmpty');  
    if (empty) empty.remove();

    appendUserMessage(query);  
    var typingId = appendTyping();  
    var t0       = Date.now();

    try {  
        var res  = await fetch('/query', {  
            method:  'POST',  
            headers: { 'Content-Type': 'application/json' },  
            body:    JSON.stringify({  
                query:       query,  
                top_k:       getTopK(),  
                temperature: getTemp()  
            })  
        });  
        var data = await res.json();  
        if (data.error) throw new Error(data.error);

        var elapsed = ((Date.now() - t0) / 1000).toFixed(2) + 's';  
        state.queryCount = state.queryCount + 1;  
        setText('sQueries', state.queryCount);

        removeTyping(typingId);  
        var answerText = (data.answer && typeof data.answer === 'string')  
            ? data.answer  
            : 'No answer returned.';  
        appendAIMessage(answerText, data.sources, elapsed);  
    } catch (e) {  
        removeTyping(typingId);  
        appendAIMessage('Error: ' + e.message, [], '0s');  
        toast('Error: ' + e.message, 'error');  
    }

    setLoading(false);  
}

function appendUserMessage(query) {  
    var area = document.getElementById('chatArea');  
    var row  = document.createElement('div');  
    row.className = 'chat-row';  
    row.innerHTML = '<div class="chat-user"><div class="chat-user-bubble">' + esc(query) + '</div></div>';  
    area.appendChild(row);  
    area.scrollTop = area.scrollHeight;  
}

function appendTyping() {  
    var area = document.getElementById('chatArea');  
    var id   = 'typing-' + Date.now();  
    var row  = document.createElement('div');  
    row.className = 'chat-row';  
    row.id        = id;  
    row.innerHTML = '<div class="chat-ai">'  
                + '<div class="chat-ai-header"><div class="chat-ai-avatar">&#x2B21;</div>GPT-4.1 &middot; thinking&hellip;</div>'  
                + '<div class="chat-typing"><div class="typing-dot"></div><div class="typing-dot"></div><div class="typing-dot"></div></div>'  
                + '</div>';  
    area.appendChild(row);  
    area.scrollTop = area.scrollHeight;  
    return id;  
}

function removeTyping(id) {  
    var el = document.getElementById(id);  
    if (el) el.remove();  
}

function appendAIMessage(answer, sources, elapsed) {  
    var area = document.getElementById('chatArea');  
    var row  = document.createElement('div');  
    row.className = 'chat-row';

    var sourcesHTML = '';  
    if (sources && sources.length) {  
        var chips = '';  
        for (var i = 0; i < sources.length; i = i + 1) {  
            var s        = sources[i];  
            var ctype    = s.chunk_type || 'recursive';  
            var chunkNum = (s.chunk_index != null ? s.chunk_index : i) + 1;  
            var score    = s.score ? s.score.toFixed(2) : '—';  
            chips = chips  
                                + '<div class="source-chip" style="animation-delay:' + (i * 0.05) + 's"'  
                                + ' onclick="openChunkModal(\'' + esc(s.filename) + '\', ' + chunkNum  
                                + ', ' + s.page + ', \'' + esc(s.text || '') + '\', \'' + ctype + '\')">'  
                                + '<span class="chip-file">&#128196; ' + esc(s.filename) + '</span>'  
                                + '<span class="chip-info">pg.' + s.page + '</span>'  
                                + '<span class="chip-sim">' + score + '</span>'  
                                + '</div>';  
        }  
        sourcesHTML = '<div class="chat-sources">'  
                        + '<div class="chat-sources-label">Sources (' + sources.length + ')</div>'  
                        + '<div class="sources-chips">' + chips + '</div>'  
                        + '</div>';  
    }

    row.innerHTML = '<div class="chat-ai">'  
                + '<div class="chat-ai-header">'  
                + '<div class="chat-ai-avatar">&#x2B21;</div>'  
                + 'GPT-4.1 <span style="color:var(--t3);margin-left:6px;">&middot; ' + elapsed + '</span>'  
                + '</div>'  
                + '<div class="chat-ai-bubble">' + fmtAnswer(answer) + '</div>'  
                + sourcesHTML  
                + '</div>';

    area.appendChild(row);  
    area.scrollTop = area.scrollHeight;

    if (window.MathJax) {  
        MathJax.typesetPromise([row]).catch(function(err) { console.error('MathJax:', err); });  
    }  
}

function setLoading(on) {  
    var btn   = document.getElementById('askBtn');  
    var label = document.getElementById('askLabel');  
    if (btn)   btn.disabled   = on;  
    if (label) label.innerHTML = on ? '<span class="spin"></span>' : '&#x25BA;';  
}

/* ══ DELETE ══ */  
async function deleteFile(id, filename, event) {  
    event.stopPropagation();  
    if (!confirm('Delete "' + filename + '" from vector store?')) return;  
    try {  
        var res  = await fetch('/delete', {  
            method:  'POST',  
            headers: { 'Content-Type': 'application/json' },  
            body:    JSON.stringify({ filename: filename })  
        });  
        var data = await res.json();  
        if (data.error) throw new Error(data.error);

        var file = null;  
        for (var i = 0; i < state.files.length; i = i + 1) {  
            if (String(state.files[i].id) === String(id)) { file = state.files[i]; break; }  
        }  
        if (file) state.totalChunks = Math.max(0, state.totalChunks - file.chunks.length);

        var nf = [];  
        for (var i = 0; i < state.files.length; i = i + 1) {  
            if (String(state.files[i].id) !== String(id)) nf.push(state.files[i]);  
        }  
        state.files = nf;  
        renderFiles();

        var btn = document.getElementById('processBtn');  
        var hp  = false;  
        for (var i = 0; i < state.files.length; i = i + 1) {  
            if (state.files[i].status === 'pending') { hp = true; break; }  
        }  
        if (btn) btn.disabled = !hp;  
        toast('🗑️ "' + filename + '" deleted', 'success');  
    } catch (e) {  
        toast('Delete failed: ' + e.message, 'error');  
    }  
}

/* ══ CLEAR ALL ══ */  
async function clearAll() {  
    if (!confirm('Delete ALL documents from the vector store? This cannot be undone.')) return;  
    try {  
        await fetch('/clear', { method: 'POST', headers: { 'Content-Type': 'application/json' } });  
        state.files       = [];  
        state.totalChunks = 0;  
        renderFiles();  
        setText('sChunks', 0);  
        setText('sFiles',  0);  
        toast('🗑️ All documents cleared', 'success');  
    } catch (e) {  
        toast('Clear failed: ' + e.message, 'error');  
    }  
}

/* ══ UTILITIES ══ */  
function esc(str) {  
    return String(str)  
        .replace(/&/g,  '&amp;')  
        .replace(/</g,  '&lt;')  
        .replace(/>/g,  '&gt;')  
        .replace(/"/g,  '&quot;');  
}

function fmtAnswer(text) {  
    var out = (text !== null && text !== undefined) ? String(text) : '';

    out = out.replace(/^### (.+)$/gm, '<h4 class="ans-h4">$1</h4>');  
    out = out.replace(/^## (.+)$/gm,  '<h3 class="ans-h3">$1</h3>');  
    out = out.replace(/^# (.+)$/gm,   '<h2 class="ans-h2">$1</h2>');  
    out = out.replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>');  
    out = out.replace(/\*(.+?)\*/g,     '<em>$1</em>');  
    out = out.replace(/^---+$/gm,       '<hr class="ans-hr"/>');

    // TABLE PARSER — handles tables with or without separator rows  
    var lines  = out.split('\n');  
    var result = [];  
    var i      = 0;

    while (i < lines.length) {  
        var line = lines[i].trim();

        if (line.startsWith('|') && line.endsWith('|')) {  
            // Collect all consecutive table lines  
            var tableLines = [];  
            while (i < lines.length && lines[i].trim().startsWith('|') && lines[i].trim().endsWith('|')) {  
                tableLines.push(lines[i].trim());  
                i = i + 1;  
            }

            // Build HTML table  
            var tableHTML  = '<div class="ans-table-wrap"><table class="ans-table">';  
            var firstRow   = true;

            for (var r = 0; r < tableLines.length; r = r + 1) {  
                var tline = tableLines[r];

                // Skip separator lines like |---|---|  
                if (/^\|[\s\-:|]+\|$/.test(tline)) continue;

                var cells  = tline.split('|');  
                var tag    = firstRow ? 'th' : 'td';  
                var rowHTML = '<tr>';

                for (var c = 0; c < cells.length; c = c + 1) {  
                    var cell = cells[c].trim();  
                    if (cell !== '') {  
                        rowHTML = rowHTML + '<' + tag + '>' + cell + '</' + tag + '>';  
                    }  
                }

                rowHTML   = rowHTML + '</tr>';  
                tableHTML = tableHTML + rowHTML;  
                firstRow  = false;  
            }

            tableHTML = tableHTML + '</table></div>';  
            result.push(tableHTML);

        } else {  
            result.push(lines[i]);  
            i = i + 1;  
        }  
    }

    out = result.join('\n');

    out = out.replace(/^[\-\*] (.+)$/gm, '<li class="ans-li">$1</li>');  
    out = out.replace(/(<li class="ans-li">[\s\S]*?<\/li>)/g, '<ul class="ans-ul">$1</ul>');  
    out = out.replace(/<\/ul>\s*<ul class="ans-ul">/g, '');  
    out = out.replace(/^\d+\. (.+)$/gm,    '<li class="ans-li">$1</li>');  
    out = out.replace(/`([^`]+)`/g,        '<code class="ans-code">$1</code>');  
    out = out.replace(/(\(Page[^)]+\))/g,  '<span class="ans-cite">$1</span>');  
    out = out.replace(/\n(?!<(h[2-4]|ul|li|hr|div|table))/g, '<br/>');

    return out;  
}

function fmtSize(b) {  
    if (b < 1024)    return b + ' B';  
    if (b < 1048576) return (b / 1024).toFixed(1) + ' KB';  
    return (b / 1048576).toFixed(1) + ' MB';  
}

function toast(msg, type) {  
    type = type || 'info';  
    var wrap = document.getElementById('toasts');  
    if (!wrap) return;  
    var el = document.createElement('div');  
    el.className = 'toast ' + type;  
    el.innerHTML = '<span class="tdot"></span>' + msg;  
    wrap.appendChild(el);  
    setTimeout(function() { el.remove(); }, 3800);  
}

