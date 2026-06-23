var state = {
    files: [],
    queryCount: 0,
    totalChunks: 0,
    activeGroupId: null  // Track active conversation group
};

var TOP_K_FALLBACK_MAX = 999;

const API_BASE_URL = 'http://localhost:8000';

var qsExpanded = true;


function setText(id, val) {
    var el = document.getElementById(id);
    if (el) el.textContent = val;
}

var CHUNK_DESCRIPTIONS = {
    recursive: 'Splits by character count with smart separator fallback - fast and reliable.',
    semantic:  'Splits by meaning using embeddings - preserves topic context. Slower, uses embedding API.',
    sliding:   'Overlapping windows of fixed size - ensures no context is lost at chunk boundaries.',
    fixed:     'Hard splits at exact character count - simple, predictable, no overlap.'
};

function updateChunkMode() {
    var select = document.getElementById('chunkModeSelect');
    var desc   = document.getElementById('chunkModeDesc');
    var mode   = select.value;
    if (desc) desc.innerHTML = CHUNK_DESCRIPTIONS[mode] || '';
    toast(mode === 'semantic' ? 'Semantic chunking selected' : mode + ' chunking selected', 'info');
}

function getChunkMode() {
    var select = document.getElementById('chunkModeSelect');
    return select ? select.value : 'recursive';
}

function hasProcessedDocs() {
    for (var i = 0; i < state.files.length; i = i + 1) {
        if (state.files[i].status === 'done') return true;
    }
    return false;
}

function getEffectiveTopKMax() {
    // After ingestion, cap manual Top-K to real available chunks.
    if (state.totalChunks && state.totalChunks > 0) return state.totalChunks;
    return TOP_K_FALLBACK_MAX;
}

/* -- TOP-K MANUAL TOGGLE -- */
var topKEnabled = false;  // when off, backend uses automatic per-intent defaults

function syncTopKControlsState() {
    var canUseTopK = hasProcessedDocs() && state.totalChunks > 0;
    var slider = document.getElementById('topK');
    var box = document.getElementById('qsEnable');
    var item = document.getElementById('topKItem');
    var hint = document.getElementById('topKHint');

    if (!slider) return;

    var effectiveMax = getEffectiveTopKMax();
    slider.max = String(effectiveMax);

    if (item) item.classList.toggle('qs-topk-disabled', !topKEnabled || !canUseTopK);
    if (box) box.disabled = !canUseTopK;

    if (!canUseTopK) {
        topKEnabled = false;
        slider.disabled = true;
        if (box) box.checked = false;
        if (hint) hint.textContent = 'Process documents until chunks are available to enable manual Top-K.';
        return;
    }

    slider.disabled = !topKEnabled;

    var current = parseInt(slider.value || '5', 10) || 5;
    if (current > effectiveMax) {
        slider.value = String(effectiveMax);
        current = effectiveMax;
    }

    var display = document.getElementById('topKDisplay');
    if (display) display.textContent = String(current);

    if (hint) {
        hint.textContent = topKEnabled
            ? 'Manual: ' + current + ' chunks per query (max ' + effectiveMax + ').'
            : 'Auto: using per-intent defaults. Toggle to set manually (max ' + effectiveMax + ').';
    }
}

// Enable/disable MANUAL Top-K. When OFF, the Top-K slider is inert and the query
// sends NO override (each intent uses its own default cap). When ON, the slider
// value is sent as an explicit override that wins over defaults.
function toggleTopKEnabled(enabled, silent) {
    if (!(hasProcessedDocs() && state.totalChunks > 0)) {
        topKEnabled = false;
        syncTopKControlsState();
        if (!silent) toast('Process documents first to enable manual Top-K', 'info');
        return;
    }

    topKEnabled = !!enabled;
    syncTopKControlsState();

    if (silent) return;
    var slider = document.getElementById('topK');
    toast(topKEnabled
        ? 'Manual Top-K enabled: using ' + (slider ? slider.value : '5')
        : 'Manual Top-K off: using automatic per-intent defaults', 'info');
}

function updateTopK(val) {
    var display = document.getElementById('topKDisplay');
    if (display) display.textContent = val;
    var hint = document.getElementById('topKHint');
    if (topKEnabled && hint) hint.textContent = 'Manual: ' + val + ' chunks per query (max ' + getEffectiveTopKMax() + ').';
}

function updateTemp(val) {
    var display = document.getElementById('tempDisplay');
    if (display) display.textContent = parseFloat(val).toFixed(1);
}

// Explicit Top-K override to send to the backend, or null when manual Top-K is
// disabled (so the backend falls back to each intent's default cap).
function getTopKOverride() {
    if (!topKEnabled) return null;
    var el = document.getElementById('topK');
    return parseInt(el ? el.value : '5', 10) || 5;
}

function getTopK() {
    var el = document.getElementById('topK');
    return parseInt(el ? el.value : '5', 10) || 5;
}

function getTemp() {
    var el = document.getElementById('temp');
    return parseFloat(el ? el.value : '0.2') || 0.2;
}

/* -- UPLOAD ZONE -- */
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

    // Start with manual Top-K OFF → automatic per-intent defaults.
    toggleTopKEnabled(false, true);
    syncTopKControlsState();
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
    var ex = list.querySelectorAll('.file-row');  
    for (var i = 0; i < ex.length; i = i + 1) { ex[i].remove(); }

    for (var i = 0; i < state.files.length; i = i + 1) {  
        var f   = state.files[i];  
        var row = document.createElement('div');
        row.className = 'file-row file-row--' + f.status;
        row.id = 'fr-' + f.id;
        row.style.animationDelay = (i * 0.05) + 's';

        var methodBadge = f.chunkMode  
            ? '<span class="chunk-method-badge chunk-badge-' + f.chunkMode + '">' + f.chunkMode.toUpperCase() + '</span>'  
            : '';

        var rechunkBtn = f.status === 'done'  
            ? '<button class="file-rechunk-btn file-action-btn file-action-rechunk" title="Re-chunk with different method" onclick="rechunkFile(\'' + f.id + '\', \'' + esc(f.name) + '\', event)">'
            + '<svg class="icon-btn" viewBox="0 0 24 24" aria-hidden="true" focusable="false">'
            + '<path d="M15.55 5.55L11 1v3.07C7.15 4.56 4.85 7.35 4.85 10.5c0 .5.45.95.95.95s.95-.45.95-.95c0-2.5 1.81-4.63 4.19-5.15V11l4.55-4.45zM19.4 15.95c-.5 0-.95.45-.95.95 0 2.5-1.81 4.63-4.19 5.15V13l-4.55 4.45 4.55 4.45v-3.07c3.85-.86 6.15-3.65 6.15-6.8 0-.5-.45-.95-.95-.95z"></path>'
            + '</svg>'
            + 'Rechunk</button>'  
            : '';

        var viewChunksBtn = f.chunks.length > 0  
            ? '<button class="file-rechunk-btn file-action-btn file-action-view" title="View all chunks" onclick="openChunksWindow(\'' + f.id + '\', \'' + esc(f.name) + '\', event)">'
            + '<svg class="icon-btn" viewBox="0 0 24 24" aria-hidden="true" focusable="false">'
            + '<path d="M3 13h2v-2H3v2zm0 4h2v-2H3v2zm0-8h2V7H3v2zm4 4h14v-2H7v2zm0 4h14v-2H7v2zM7 7v2h14V7H7z"></path>'
            + '</svg>'
            + 'Chunks</button>'  
            : '';

        var deleteBtn = f.status !== 'processing'  
            ? '<button class="file-delete-btn file-action-btn file-action-delete" title="Delete file" aria-label="Delete file" onclick="deleteFile(\'' + f.id + '\', \'' + esc(f.name) + '\', event)">'
            + '<svg class="icon-trash" viewBox="0 0 24 24" aria-hidden="true" focusable="false">'
            + '<path d="M9 3h6l1 2h4v2H4V5h4l1-2zm1 7h2v8h-2v-8zm4 0h2v8h-2v-8zM7 10h2v8H7v-8z"></path>'
            + '</svg>'
            + 'Delete'
            + '</button>'
            : '';

        var progressBar = f.status === 'processing'  
            ? '<div class="progress-wrap"><div class="progress-fill" id="pb-' + f.id + '" style="width:0%"></div></div>'  
            : '';

        row.innerHTML = '<div class="file-item" id="fi-' + f.id + '">'  
            + '<div class="file-head">'  
            + '<div class="file-main">'
            + '<span class="file-icon">&#128196;</span>'  
            + '<div class="file-info">'  
            + '<div class="file-name-row">'
            + '<div class="file-name" title="' + esc(f.name) + '">' + esc(f.name) + '</div>'  
            + '<span class="file-size-inline">(' + f.size + ')</span>'
            + '</div>'
            + '</div>'  
            + '</div>'
            + '<div class="file-bottom-meta">' + methodBadge + '<span class="file-badge badge-' + f.status + '">' + f.status.toUpperCase() + '</span></div>'
            + '</div>'  
            + progressBar
            + '</div>'
            + '<div class="file-actions">' + viewChunksBtn + rechunkBtn + deleteBtn + '</div>';

        list.appendChild(row);
    }

    setText('fileCount', state.files.length);  
    setText('sFiles',    state.files.length);  
    setText('sChunks',   state.totalChunks);  
    setText('sQueries',  state.queryCount);  
    syncTopKControlsState();
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

function openChunksWindow(fileId, filename, event) {
    event.stopPropagation();
    var file = null;
    for (var i = 0; i < state.files.length; i = i + 1) {
        if (String(state.files[i].id) === String(fileId)) { file = state.files[i]; break; }
    }
    if (!file || !file.chunks) { toast('File not found', 'error'); return; }

    var modal = document.getElementById('chunksWindowModal');
    var titleEl = document.getElementById('chunksWindowTitle');
    var bodyEl = document.getElementById('chunksWindowBody');
    var searchEl = document.getElementById('chunksSearchInput');
    
    if (!modal) return;
    
    titleEl.textContent = 'All Chunks · ' + esc(filename);
    searchEl.value = '';
    
    // Store current file for filtering
    window.currentChunksFile = file;
    
    var chunksHTML = '';
    for (var k = 0; k < file.chunks.length; k = k + 1) {
        var c = file.chunks[k];
        var pageNum = (c.page != null && c.page !== '?') ? c.page : '?';
        chunksHTML += '<div class="chunk-window-card" data-chunk-index="' + k + '" data-chunk-text="' + esc(c.text || '').toLowerCase() + '" onclick="toggleChunkExpand(this)" title="Click to expand / collapse">'
            + '<div class="chunk-window-header">'
            + '<div class="chunk-window-label">Chunk ' + (k + 1) + '</div>'
            + '<div class="chunk-window-meta">Page ' + pageNum + ' • ' + (file.chunkMode || 'recursive').toUpperCase() + '</div>'
            + '</div>'
            + '<div class="chunk-window-text">' + esc(c.text || '') + '</div>'
            + '<button class="chunk-window-copy" onclick="copyChunkToClipboard(\'' + esc(c.text || '').replace(/'/g, "\\'") + '\', event)">Copy</button>'
            + '</div>';
    }
    
    bodyEl.innerHTML = chunksHTML;
    modal.classList.add('open');
    document.body.style.overflow = 'hidden';
    searchEl.focus();
}

function closeChunksWindow() {
    var modal = document.getElementById('chunksWindowModal');
    if (modal) modal.classList.remove('open');
    document.body.style.overflow = '';
    window.currentChunksFile = null;
}

function filterChunks() {
    var searchEl = document.getElementById('chunksSearchInput');
    var query = searchEl.value.toLowerCase();
    var cards = document.querySelectorAll('.chunk-window-card');
    
    for (var i = 0; i < cards.length; i = i + 1) {
        var card = cards[i];
        var text = card.getAttribute('data-chunk-text');
        if (text.indexOf(query) >= 0) {
            card.style.display = '';
        } else {
            card.style.display = 'none';
        }
    }
}

function copyChunkToClipboard(text, event) {
    event.stopPropagation();
    navigator.clipboard.writeText(text).then(function() {
        toast('Chunk copied to clipboard', 'success');
    }).catch(function() {
        toast('Failed to copy', 'error');
    });
}

// Toggle a chunk card in the "All Chunks" window between the clamped 3-line
// preview and the full chunk text. The Copy button stops propagation, so it
// never triggers this toggle.
function toggleChunkExpand(card) {
    if (!card) return;
    card.classList.toggle('expanded');
}

document.addEventListener('keydown', function(e) {
    if (e.key === 'Escape') {
        closeChunkModal();
        closeChunksWindow();
    }
});

/* ══ POLLING FOR FILE COMPLETION ══ */
function pollForFileCompletion(fileObj) {
    var pollCount = 0;
    var maxPolls = 120; // 2 minutes max (120 * 1 second)
    
    var pollInterval = setInterval(async function() {
        pollCount = pollCount + 1;
        
        try {
            // Query backend for status - use a simple OPTIONS request or status endpoint
            var statusRes = await fetch(API_BASE_URL + '/status', { method: 'GET' });
            
            if (!statusRes.ok) {
                clearInterval(pollInterval);
                return;
            }
            
            var statusData = await statusRes.json();
            
            // Check ingestion_status first (works even if file has 0 chunks)
            var fileIngestionStatus = statusData.ingestion_status && statusData.ingestion_status[fileObj.name];
            
            if (fileIngestionStatus && fileIngestionStatus.status === 'completed') {
                fileObj.status = 'done';
                
                // Update chunk count from ingestion status
                fileObj.chunks = []; // Reset chunks array
                state.totalChunks = statusData.total_vectors || 0;
                
                // POPULATE CHUNKS FOR THIS FILE FROM STATUS RESPONSE
                if (statusData.files && Array.isArray(statusData.files)) {
                    for (var fi = 0; fi < statusData.files.length; fi = fi + 1) {
                        var fileStatus = statusData.files[fi];
                        // Find matching file in state.files
                        for (var sfi = 0; sfi < state.files.length; sfi = sfi + 1) {
                            if (state.files[sfi].name === fileStatus.filename) {
                                state.files[sfi].chunks = fileStatus.chunks || [];
                                // Preserve the mode recorded at upload time; only fall
                                // back to the polled file's mode if it's missing.
                                if (!state.files[sfi].chunkMode) {
                                    state.files[sfi].chunkMode = fileObj.chunkMode;
                                }
                                break;
                            }
                        }
                    }
                }
                
                // Show appropriate message based on chunk count
                if (fileIngestionStatus.chunks === 0) {
                    toast('⚠️ ' + fileObj.name + ' — completed but no content extracted (file may be image-only, encrypted, or invalid)', 'warning');
                } else {
                    toast('✅ ' + fileObj.name + ' — completed (' + fileIngestionStatus.chunks + ' chunks)', 'success');
                }
                renderFiles();
                clearInterval(pollInterval);
                return;
            }
            
            // Stop after max polls
            if (pollCount >= maxPolls) {
                fileObj.status = 'done'; // Assume done after timeout
                toast('⏱ ' + fileObj.name + ' — processing timeout (marked as done)', 'warning');
                renderFiles();
                clearInterval(pollInterval);
            }
        } catch (e) {
            // Network error - continue polling
            if (pollCount >= maxPolls) {
                clearInterval(pollInterval);
            }
        }
    }, 1000); // Poll every 1 second
}

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
            var res  = await fetch(API_BASE_URL + '/ingest', { method: 'POST', body: form });  
            var data = await res.json();  
            
            // Handle 202 Accepted - processing in background
            if (res.status === 202) {
                f.status = 'processing';
                f.chunkMode = chunkMode;  // remember the mode used so the badge shows after polling
                toast('⏳ ' + f.name + ' — processing in background', 'info');
                renderFiles();
                // Start polling for completion
                pollForFileCompletion(f);
                continue;
            }
            
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
        var delRes  = await fetch(API_BASE_URL + '/delete', {  
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

        var res  = await fetch(API_BASE_URL + '/ingest', { method: 'POST', body: form });  
        var data = await res.json();  

        // Handle 202 Accepted — ingest now processes in the background.
        // The response no longer contains `documents`, so treat 202 as success
        // (file is being re-chunked) and poll /status until it completes.
        if (res.status === 202) {  
            file.status    = 'processing';  
            file.chunkMode = newMode;  
            toast('⏳ Re-chunking "' + filename + '" in background', 'info');  
            renderFiles();  
            animateProgress(file.id);  
            pollForFileCompletion(file);  
            return;  
        }  

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
    var t0 = Date.now();

    try {
        await streamQuery(query, t0);
    } catch (e) {
        appendAIMessage('Error: ' + e.message, [], '0s');
        toast('Error: ' + e.message, 'error');
    }

    setLoading(false);
}

// Stream the answer from POST /query/stream (Server-Sent Events over fetch).
async function streamQuery(query, t0) {
    var res = await fetch(API_BASE_URL + '/query/stream', {
        method:  'POST',
        headers: { 'Content-Type': 'application/json' },
        body:    JSON.stringify({
            query:          query,
            top_k:          getTopK(),
            temperature:    getTemp(),
            top_k_override: getTopKOverride(),
            group_id:       state.activeGroupId
        })
    });

    if (!res.ok || !res.body) {
        // Non-streaming error (e.g. 400 with JSON body)
        var errData = null;
        try { errData = await res.json(); } catch (e) {}
        throw new Error(errData && errData.error ? errData.error : ('HTTP ' + res.status));
    }

    var msg        = createStreamingAIMessage();
    var fullAnswer = '';
    var sources    = null;
    var finalAnswer = null;
    var firstToken = false;

    var reader  = res.body.getReader();
    var decoder = new TextDecoder('utf-8');
    var buffer  = '';

    while (true) {
        var chunk = await reader.read();
        if (chunk.done) break;
        buffer += decoder.decode(chunk.value, { stream: true });

        // SSE frames are separated by a blank line.
        var frames = buffer.split('\n\n');
        buffer = frames.pop();  // keep incomplete trailing frame

        for (var i = 0; i < frames.length; i = i + 1) {
            var line = frames[i].trim();
            if (line.indexOf('data:') !== 0) continue;
            var payload = line.slice(5).trim();
            if (!payload) continue;

            var ev;
            try { ev = JSON.parse(payload); } catch (e) { continue; }

            if (ev.type === 'meta') {
                sources = ev.retrieved_chunks || [];
                if (ev.group_id) state.activeGroupId = ev.group_id;
                // Live progress: tell the user retrieval finished and how many
                // sources are being read, so the wait before the first token
                // (classification + retrieval + map phase) doesn't look frozen.
                if (!firstToken && msg.thinkingEl) {
                    var n = sources ? sources.length : 0;
                    msg.thinkingEl.querySelector('.think-label').textContent =
                        n > 0 ? ('Reading ' + n + ' source' + (n === 1 ? '' : 's') + '\u2026')
                              : 'Generating answer\u2026';
                }
            } else if (ev.type === 'token') {
                // First token: drop the thinking indicator and switch to live text.
                if (!firstToken) {
                    firstToken = true;
                    if (msg.thinkingEl) { msg.thinkingEl.remove(); msg.thinkingEl = null; }
                }
                fullAnswer += ev.text;
                msg.bubble.textContent = fullAnswer;
                scrollChat();
            } else if (ev.type === 'done') {
                finalAnswer = (typeof ev.answer === 'string' && ev.answer) ? ev.answer : fullAnswer;
            } else if (ev.type === 'error') {
                throw new Error(ev.error || 'stream error');
            }
        }
    }

    // Finalize: render full markdown + LaTeX, attach sources, stamp elapsed time.
    var answerText = (finalAnswer != null) ? finalAnswer : (fullAnswer || 'No answer returned.');
    var elapsed = ((Date.now() - t0) / 1000).toFixed(2) + 's';

    msg.bubble.classList.remove('chat-ai-streaming');
    msg.bubble.innerHTML = fmtAnswer(answerText);
    if (msg.sourcesSlot) msg.sourcesSlot.innerHTML = buildSourcesHTML(sources);
    if (msg.elapsedEl) msg.elapsedEl.innerHTML = '&middot; ' + elapsed;

    state.queryCount = state.queryCount + 1;
    setText('sQueries', state.queryCount);

    if (window.MathJax) {
        MathJax.typesetPromise([msg.row]).catch(function(err) { console.error('MathJax:', err); });
    }
    scrollChat();
}

function scrollChat() {
    var area = document.getElementById('chatArea');
    if (area) area.scrollTop = area.scrollHeight;
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

    var sourcesHTML = buildSourcesHTML(sources);

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

// Build the "Sources (N)" chips block from a list of retrieved chunks.
// Rendered as a collapsible dropdown — collapsed by default so a large source
// list (e.g. 94 chunks) doesn't flood the chat. Click the header to expand.
function buildSourcesHTML(sources) {
    if (!sources || !sources.length) return '';
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
    var sid = 'src-' + (sourcesSeq++);
    return '<div class="chat-sources">'
            + '<div class="chat-sources-label" onclick="toggleSources(\'' + sid + '\')">'
            + '<span class="src-arrow" id="arrow-' + sid + '">&#9658;</span> '
            + 'Sources (' + sources.length + ')'
            + '</div>'
            + '<div class="sources-chips" id="' + sid + '">' + chips + '</div>'
            + '</div>';
}

var sourcesSeq = 0;

// Expand/collapse a sources dropdown.
function toggleSources(id) {
    var chipsEl = document.getElementById(id);
    var arrow   = document.getElementById('arrow-' + id);
    if (!chipsEl) return;
    var open = chipsEl.classList.toggle('open');
    if (arrow) arrow.innerHTML = open ? '&#9660;' : '&#9658;';
}

// Create an empty AI message shell that tokens can be streamed into. Returns the
// row plus the live bubble element. The bubble uses pre-wrap so raw streamed text
// reads naturally; it is re-rendered with full markdown/MathJax when streaming ends.
function createStreamingAIMessage() {
    var area = document.getElementById('chatArea');
    var row  = document.createElement('div');
    row.className = 'chat-row';
    row.innerHTML = '<div class="chat-ai">'
                + '<div class="chat-ai-header">'
                + '<div class="chat-ai-avatar">&#x2B21;</div>'
                + 'GPT-4.1 <span class="chat-ai-elapsed" style="color:var(--t3);margin-left:6px;">&middot; streaming&hellip;</span>'
                + '</div>'
                + '<div class="chat-ai-bubble chat-ai-streaming">'
                + '<span class="think"><span class="think-label">Understanding your question&hellip;</span>'
                + '<span class="think-dots"><i></i><i></i><i></i></span></span>'
                + '</div>'
                + '<div class="chat-ai-sources-slot"></div>'
                + '</div>';
    area.appendChild(row);
    area.scrollTop = area.scrollHeight;
    return {
        row: row,
        bubble: row.querySelector('.chat-ai-bubble'),
        thinkingEl: row.querySelector('.think'),
        sourcesSlot: row.querySelector('.chat-ai-sources-slot'),
        elapsedEl: row.querySelector('.chat-ai-elapsed'),
    };
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
        var res  = await fetch(API_BASE_URL + '/delete', {  
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
        await fetch(API_BASE_URL + '/clear', { method: 'POST', headers: { 'Content-Type': 'application/json' } });  
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
// Remove non-printable PDF-extraction artifacts (control chars like \x01, \x02,
// \x04, \x1f and the Unicode replacement char) that render as █ boxes in the UI.
function cleanText(str) {
    return String(str)
        // strip C0 controls except tab(\x09)/newline(\x0A)/carriage-return(\x0D)
        .replace(/[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]/g, '')
        // strip the Unicode replacement character and the box-drawing fallback glyph
        .replace(/[\uFFFD\u25A0\u2588]/g, '');
}

function esc(str) {  
    return cleanText(str)  
        .replace(/&/g,  '&amp;')  
        .replace(/</g,  '&lt;')  
        .replace(/>/g,  '&gt;')  
        .replace(/"/g,  '&quot;');  
}

function fmtAnswer(text) {  
    var out = (text !== null && text !== undefined) ? cleanText(text) : '';

    // Protect LaTeX math spans from the markdown transforms below. We pull them out
    // into placeholders, format everything else, then restore them verbatim so MathJax
    // receives clean delimiters. Without this, the '\n -> <br/>' step shatters multi-line
    // \[ ... \] blocks and the bold/italic regexes eat characters inside \( ... \),
    // making formulas render blank. Display delimiters are extracted before inline ones.
    var mathStore = [];
    function stashMath(m) {
        mathStore.push(m);
        return '@@MATH' + (mathStore.length - 1) + '@@';
    }
    out = out.replace(/\$\$[\s\S]+?\$\$/g, stashMath);   // $$ ... $$  (display)
    out = out.replace(/\\\[[\s\S]+?\\\]/g, stashMath);   // \[ ... \]  (display)
    out = out.replace(/\\\([\s\S]+?\\\)/g, stashMath);   // \( ... \)  (inline)
    out = out.replace(/\$[^$\n]+?\$/g,     stashMath);   // $ ... $    (inline)

    // Some models wrap the whole answer in bare heading markers on their own lines
    // (a lone "##" at the very top and bottom). A heading marker with no text is never
    // meaningful, so drop any line that is ONLY '#' characters, then trim the blank
    // edges. Real headings like "## Heading" have text after the space and are kept.
    out = out.replace(/^[ \t]*#{1,6}[ \t]*$/gm, '');
    out = out.replace(/^\s+|\s+$/g, '');

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

                // split('|') on "| a | b |" yields a leading and trailing '' from the
                // outer pipes — drop ONLY those. Interior cells are kept even when empty,
                // otherwise a blank cell makes every following cell shift left and the
                // columns misalign (e.g. a colour row with no letter).
                if (cells.length && cells[0].trim() === '') cells.shift();
                if (cells.length && cells[cells.length - 1].trim() === '') cells.pop();

                for (var c = 0; c < cells.length; c = c + 1) {  
                    var cell = cells[c].trim();  
                    rowHTML = rowHTML + '<' + tag + '>' + cell + '</' + tag + '>';  
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

    // List markers: a dash/asterisk OR a literal bullet glyph the model used as the
    // marker itself (•, ●, ∙, ·, ‣, ▪, ◦, ○, ⦁, ⁃). Treating a literal bullet as a marker
    // means it becomes a proper <li> (single CSS bullet) instead of rendering as raw text.
    out = out.replace(/^[ \t]*(?:[\-\*]|[•·‣▪◦●∙○⦁⁃])[ \t]+(.+)$/gm, '<li class="ans-li">$1</li>');  
    out = out.replace(/(<li class="ans-li">[\s\S]*?<\/li>)/g, '<ul class="ans-ul">$1</ul>');  
    out = out.replace(/<\/ul>\s*<ul class="ans-ul">/g, '');  
    out = out.replace(/^\d+\. (.+)$/gm,    '<li class="ans-li">$1</li>');  
    // Strip ANY redundant leading list marker(s) the model left INSIDE a list item, so
    // they don't render as a SECOND bullet next to the CSS one (.ans-li::before). Covers:
    //   • a literal bullet glyph (•, ●, ·, ‣, ▪, ◦, ○, ⦁, ∙, ◘, ⁃), possibly repeated, and
    //   • a stray "-" or "*" that is ITSELF followed by whitespace (a leftover dash/star
    //     marker, e.g. from "- - text" or "* - text") — but NOT a leading minus on a value
    //     like "-5 degrees", where the "-" is followed by a digit/letter, not a space.
    // The marker may sit INSIDE the item's bold/italic markers (e.g. "- **• Heading**" ->
    // "<li><strong>• Heading</strong>"), since bold/italic convert before lists — so allow
    // optional <strong>/<em> open tags between the <li> and the marker, and keep them.
    out = out.replace(/(<li class="ans-li">)((?:<(?:strong|em)>)*)(?:[\s\u00a0]*(?:[•·‣▪◦●∙○⦁◘⁃]|[-*](?=[\s\u00a0])))+[\s\u00a0]*/g, '$1$2');  
    out = out.replace(/`([^`]+)`/g,        '<code class="ans-code">$1</code>');  
    out = out.replace(/(\(Page[^)]+\))/g,  '<span class="ans-cite">$1</span>');  

    // LINKS — make any URL clickable.
    // 1) Markdown links [text](url) -> <a>.
    out = out.replace(/\[([^\]]+)\]\(((?:https?:\/\/|www\.)[^\s)]+)\)/g, function (_, txt, url) {
        var href = /^www\./i.test(url) ? 'https://' + url : url;
        return '<a class="ans-link" href="' + href + '" target="_blank" rel="noopener noreferrer">' + txt + '</a>';
    });
    // 2) Bare URLs -> <a>. Stash any HTML already built (anchors from step 1, code spans)
    //    so the autolinker can NEVER reach a URL inside an existing tag/attribute — this
    //    prevents nested <a> from a markdown link's href and linkifying URLs inside <code>.
    var htmlStore = [];
    out = out.replace(/<a\b[\s\S]*?<\/a>|<code\b[\s\S]*?<\/code>/g, function (m) {
        htmlStore.push(m);
        return '@@HTML' + (htmlStore.length - 1) + '@@';
    });
    out = out.replace(/((?:https?:\/\/|www\.)[^\s<>()]+[^\s<>().,;:!?'"])/g, function (url) {
        var href = /^www\./i.test(url) ? 'https://' + url : url;
        return '<a class="ans-link" href="' + href + '" target="_blank" rel="noopener noreferrer">' + url + '</a>';
    });
    out = out.replace(/@@HTML(\d+)@@/g, function (_, n) { return htmlStore[Number(n)]; });

    out = out.replace(/\n(?!<(h[2-4]|ul|li|hr|div|table))/g, '<br/>');

    // Restore protected LaTeX spans verbatim (after <br/> insertion so display-math
    // newlines stay intact for MathJax).
    out = out.replace(/@@MATH(\d+)@@/g, function (_, n) { return mathStore[Number(n)]; });

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

