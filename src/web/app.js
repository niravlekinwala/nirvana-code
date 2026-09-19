// API-key shim: when the server was started with --api-key, every /v1/*
// call carries it; a 401 prompts once and remembers the key in this browser.
(function () {
  const KEY = 'nirvana_api_key';
  const orig = window.fetch.bind(window);
  const stored = () => { try { return localStorage.getItem(KEY); } catch (_) { return null; } };
  const remember = (k) => { try { localStorage.setItem(KEY, k); } catch (_) {} };
  const withAuth = (init, key) => {
    const next = Object.assign({}, init || {});
    const h = new Headers(next.headers || {});
    h.set('Authorization', 'Bearer ' + key);
    next.headers = h;
    return next;
  };
  window.fetch = async function (input, init) {
    const url = typeof input === 'string' ? input : (input && input.url) || '';
    const isApi = url.startsWith('/v1/');
    const key = stored();
    let res = await orig(input, isApi && key ? withAuth(init, key) : init);
    if (res.status === 401 && isApi) {
      const entered = window.prompt('This Nirvana Code server requires an API key.\nIt was printed in the terminal where the server started.');
      if (entered && entered.trim()) {
        remember(entered.trim());
        res = await orig(input, withAuth(init, entered.trim()));
      }
    }
    return res;
  };
})();

// State
let isGenerating = false;
let abortController = null;
const chatHistory = [];

// Context Window State
let currentCtxCapacity = 4096;
let currentCtxUsed = 0;

// Project Workspace State
let allProjectFiles = [];
let currentProjectName = '';
let previewedFile = null;

// DOM Elements - Core Chat
const messagesViewport = document.getElementById('messagesViewport');
const promptInput = document.getElementById('promptInput');
const sendBtn = document.getElementById('sendBtn');
const sendLabel = document.getElementById('sendLabel');
const sidebar = document.getElementById('sidebar');
const sidebarToggle = document.getElementById('sidebarToggle');
const heroSplash = document.getElementById('heroSplash');
const templateSelect = document.getElementById('templateSelect');
const toast = document.getElementById('toast');

// Telemetry Elements
const metricTtft = document.getElementById('metricTtft');
const metricSpeed = document.getElementById('metricSpeed');
const metricTokens = document.getElementById('metricTokens');
const specModel = document.getElementById('specModel');
const modelSelect = document.getElementById('modelSelect');
const modelStatus = document.getElementById('modelStatus');
const checkNgramSpec = document.getElementById('checkNgramSpec');

// Context Window Elements
const contextHud = document.getElementById('contextHud');
const contextNumbers = document.getElementById('contextNumbers');
const contextFill = document.getElementById('contextFill');
const contextSelect = document.getElementById('contextSelect');
const customContextRow = document.getElementById('customContextRow');
const customContextInput = document.getElementById('customContextInput');
const btnApplyCustomContext = document.getElementById('btnApplyCustomContext');
const sidebarContextStats = document.getElementById('sidebarContextStats');
const sidebarContextFill = document.getElementById('sidebarContextFill');

// Project Workspace Elements
const projPathInput = document.getElementById('projPathInput');
const btnScanProject = document.getElementById('btnScanProject');
const projNameText = document.getElementById('projNameText');
const projTypeText = document.getElementById('projTypeText');
const projFilesCountText = document.getElementById('projFilesCountText');
const projFilterInput = document.getElementById('projFilterInput');
const projFileList = document.getElementById('projFileList');
const filePreviewModal = document.getElementById('filePreviewModal');
const modalFileName = document.getElementById('modalFileName');
const modalFileContent = document.getElementById('modalFileContent');
const modalCloseBtn = document.getElementById('modalCloseBtn');
const modalDismissBtn = document.getElementById('modalDismissBtn');
const modalAttachBtn = document.getElementById('modalAttachBtn');

// Attachment State & Elements
let currentAttachment = null;
let attachmentProcessingPromise = null;
const attachmentBar = document.getElementById('attachmentBar');
const attachmentThumb = document.getElementById('attachmentThumb');
const attachmentName = document.getElementById('attachmentName');
const attachmentMeta = document.getElementById('attachmentMeta');
const attachmentRemoveBtn = document.getElementById('attachmentRemoveBtn');
const fileInput = document.getElementById('fileInput');
const attachBtn = document.getElementById('attachBtn');
const dropZone = document.getElementById('dropZone');

// ==========================================
// 1. SIDEBAR TAB NAVIGATION
// ==========================================
function switchSidebarTab(tabName) {
  const tabs = {
    conversations: { btn: document.getElementById('tabConversationsBtn'), panel: document.getElementById('panelConversations') },
    projects: { btn: document.getElementById('tabProjectsBtn'), panel: document.getElementById('panelProjects') },
    engine: { btn: document.getElementById('tabEngineBtn'), panel: document.getElementById('panelEngine') },
  };

  Object.keys(tabs).forEach(name => {
    const t = tabs[name];
    if (t.btn && t.panel) {
      if (name === tabName) {
        t.btn.classList.add('active');
        t.panel.classList.add('active');
      } else {
        t.btn.classList.remove('active');
        t.panel.classList.remove('active');
      }
    }
  });
}

document.getElementById('tabConversationsBtn').addEventListener('click', () => switchSidebarTab('conversations'));
document.getElementById('tabProjectsBtn').addEventListener('click', () => switchSidebarTab('projects'));
document.getElementById('tabEngineBtn').addEventListener('click', () => switchSidebarTab('engine'));

// ==========================================
// 2. CONTEXT WINDOW MANAGEMENT
// ==========================================
function updateContextBar(used, capacity) {
  currentCtxUsed = Math.max(0, used);
  if (capacity && capacity > 0) {
    currentCtxCapacity = capacity;
  }
  const pct = Math.min(100, Math.max(0, (currentCtxUsed / currentCtxCapacity) * 100));
  const pctStr = pct.toFixed(1) + '%';
  const text = `${currentCtxUsed.toLocaleString()} / ${currentCtxCapacity.toLocaleString()} tokens (${pctStr})`;

  if (contextNumbers) contextNumbers.textContent = text;
  if (sidebarContextStats) sidebarContextStats.textContent = text;

  const fills = [contextFill, sidebarContextFill];
  fills.forEach(f => {
    if (!f) return;
    f.style.width = pctStr;
    f.classList.remove('context-fill-ok', 'context-fill-warn', 'context-fill-crit');
    if (pct < 65) {
      f.classList.add('context-fill-ok');
    } else if (pct < 85) {
      f.classList.add('context-fill-warn');
    } else {
      f.classList.add('context-fill-crit');
    }
  });
}

async function fetchContextInfo() {
  try {
    const res = await fetch('/v1/context');
    if (res.ok) {
      const info = await res.json();
      currentCtxCapacity = info.ctx_size || 4096;
      if (info.kv_mode) {
        const badgeKv = document.getElementById('badgeKv');
        const specKv = document.getElementById('specKv');
        if (badgeKv) badgeKv.textContent = `KV: ${info.kv_mode}`;
        if (specKv) specKv.textContent = `${info.kv_mode} (Peak Speed)`;
      }
      if (info.backend) {
        const specChip = document.getElementById('specChip');
        if (specChip) specChip.textContent = `Apple Silicon (${info.backend})`;
      }

      // Match dropdown
      const opts = Array.from(contextSelect.options).map(o => o.value);
      if (opts.includes(String(currentCtxCapacity))) {
        contextSelect.value = String(currentCtxCapacity);
        customContextRow.style.display = 'none';
      } else {
        contextSelect.value = 'custom';
        customContextRow.style.display = 'flex';
        customContextInput.value = currentCtxCapacity;
      }
      updateContextBar(currentCtxUsed, currentCtxCapacity);
    }
  } catch (e) {
    console.warn('Context info fetch error:', e);
  }
}

async function applyContextSize(size) {
  if (!size || isNaN(size) || size < 512 || size > 262144) {
    showToast('⚠️ Context size must be between 512 and 262,144 tokens');
    return;
  }
  showToast(`⏳ Resizing context to ${size.toLocaleString()} tokens...`);
  try {
    const res = await fetch('/v1/context', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ ctx_size: size })
    });
    const data = await res.json();
    if (res.ok) {
      currentCtxCapacity = data.ctx_size || size;
      updateContextBar(currentCtxUsed, currentCtxCapacity);
      showToast(`✔ Context window set to ${currentCtxCapacity.toLocaleString()} tokens`);
    } else {
      showToast(`❌ ${data.error || 'Failed to resize context'}`);
    }
  } catch (e) {
    showToast(`❌ Error: ${e.message}`);
  }
}

contextSelect.addEventListener('change', () => {
  if (contextSelect.value === 'custom') {
    customContextRow.style.display = 'flex';
    customContextInput.focus();
  } else {
    customContextRow.style.display = 'none';
    applyContextSize(parseInt(contextSelect.value, 10));
  }
});

btnApplyCustomContext.addEventListener('click', () => {
  const val = parseInt(customContextInput.value, 10);
  applyContextSize(val);
});

contextHud.addEventListener('click', () => {
  switchSidebarTab('engine');
  sidebar.classList.remove('collapsed');
  contextSelect.scrollIntoView({ behavior: 'smooth' });
});

// ==========================================
// 3. CONVERSATIONS MANAGEMENT (PERSISTENCE)
// ==========================================
const CONV_STORAGE_KEY = 'nirvana_conversations_v1';
let conversations = [];
let currentConvId = null;

function loadConversationsFromStorage() {
  try {
    const raw = localStorage.getItem(CONV_STORAGE_KEY);
    if (raw) {
      conversations = JSON.parse(raw);
    }
  } catch (e) {
    console.error('Failed to parse conversations', e);
  }

  if (!Array.isArray(conversations) || conversations.length === 0) {
    const initId = 'conv_' + Date.now();
    conversations = [{
      id: initId,
      title: 'New Conversation',
      createdAt: Date.now(),
      updatedAt: Date.now(),
      messages: [],
      ctxUsed: 0,
    }];
  }

  currentConvId = conversations[0].id;
  renderConversationList();
  loadActiveConversation();
}

function saveConversationsToStorage() {
  try {
    localStorage.setItem(CONV_STORAGE_KEY, JSON.stringify(conversations));
  } catch (e) {
    console.error('Failed to save conversations', e);
  }
}

function renderConversationList() {
  const listEl = document.getElementById('convList');
  if (!listEl) return;
  listEl.innerHTML = '';

  // Sort by recency
  conversations.sort((a, b) => (b.updatedAt || 0) - (a.updatedAt || 0));

  conversations.forEach(c => {
    const card = document.createElement('div');
    card.className = `conv-card ${c.id === currentConvId ? 'active' : ''}`;
    card.onclick = () => switchConversation(c.id);

    const info = document.createElement('div');
    info.className = 'conv-info';

    const title = document.createElement('div');
    title.className = 'conv-title';
    title.textContent = c.title || 'Conversation';

    const time = document.createElement('div');
    time.className = 'conv-time';
    const msgCount = c.messages ? c.messages.length : 0;
    const timeStr = formatRelativeTime(c.updatedAt || c.createdAt);
    time.textContent = `${msgCount} msgs • ${timeStr}`;

    info.appendChild(title);
    info.appendChild(time);

    const delBtn = document.createElement('button');
    delBtn.className = 'conv-del-btn';
    delBtn.title = 'Delete conversation';
    delBtn.innerHTML = '✕';
    delBtn.onclick = (e) => {
      e.stopPropagation();
      deleteConversation(c.id);
    };

    card.appendChild(info);
    card.appendChild(delBtn);
    listEl.appendChild(card);
  });
}

function formatRelativeTime(ts) {
  if (!ts) return '';
  const diffSec = Math.floor((Date.now() - ts) / 1000);
  if (diffSec < 60) return 'Just now';
  if (diffSec < 3600) return `${Math.floor(diffSec / 60)}m ago`;
  if (diffSec < 86400) return `${Math.floor(diffSec / 3600)}h ago`;
  return new Date(ts).toLocaleDateString();
}

function switchConversation(convId) {
  if (isGenerating) {
    showToast('⚠️ Please wait or stop active generation first');
    return;
  }
  currentConvId = convId;
  renderConversationList();
  loadActiveConversation();
}

function loadActiveConversation() {
  const conv = conversations.find(c => c.id === currentConvId) || conversations[0];
  if (!conv) return;
  currentConvId = conv.id;

  chatHistory.length = 0;
  messagesViewport.innerHTML = '';

  if (!conv.messages || conv.messages.length === 0) {
    messagesViewport.appendChild(heroSplash);
    heroSplash.style.display = 'block';
    currentCtxUsed = 0;
    updateContextBar(0, currentCtxCapacity);
  } else {
    heroSplash.style.display = 'none';
    let estTokens = 0;
    conv.messages.forEach(m => {
      chatHistory.push({ role: m.role, content: m.content, attachmentInfo: m.attachmentInfo });
      appendMessage(m.role, m.content, m.attachmentInfo, false);
      estTokens += Math.max(1, Math.round(m.content.length / 4));
    });
    messagesViewport.scrollTop = messagesViewport.scrollHeight;
    currentCtxUsed = conv.ctxUsed || estTokens;
    updateContextBar(currentCtxUsed, currentCtxCapacity);
  }
}

function createNewConversation() {
  if (isGenerating) {
    showToast('⚠️ Please wait for current generation to finish');
    return;
  }
  const newConv = {
    id: 'conv_' + Date.now(),
    title: 'New Conversation',
    createdAt: Date.now(),
    updatedAt: Date.now(),
    messages: [],
    ctxUsed: 0,
  };
  conversations.unshift(newConv);
  currentConvId = newConv.id;
  saveConversationsToStorage();
  renderConversationList();
  loadActiveConversation();
  showToast('✨ New conversation ready');
}

function deleteConversation(id) {
  if (conversations.length <= 1) {
    conversations[0].messages = [];
    conversations[0].title = 'New Conversation';
    conversations[0].updatedAt = Date.now();
    conversations[0].ctxUsed = 0;
    saveConversationsToStorage();
    renderConversationList();
    loadActiveConversation();
    showToast('Conversation reset');
    return;
  }

  conversations = conversations.filter(c => c.id !== id);
  if (currentConvId === id) {
    currentConvId = conversations[0].id;
  }
  saveConversationsToStorage();
  renderConversationList();
  loadActiveConversation();
  showToast('Conversation deleted');
}

function saveCurrentConversation(newTokens = null) {
  const conv = conversations.find(c => c.id === currentConvId);
  if (!conv) return;

  conv.messages = chatHistory.map(m => ({
    role: m.role,
    content: m.content,
    attachmentInfo: m.attachmentInfo || null,
  }));
  conv.updatedAt = Date.now();

  if (conv.title === 'New Conversation') {
    const firstUser = chatHistory.find(m => m.role === 'user');
    if (firstUser && firstUser.content) {
      const clean = firstUser.content.replace(/\s+/g, ' ').trim();
      conv.title = clean.length > 36 ? clean.slice(0, 36) + '...' : clean;
    }
  }

  if (newTokens !== null) {
    conv.ctxUsed = newTokens;
  }

  saveConversationsToStorage();
  renderConversationList();
}

document.getElementById('btnNewChat').addEventListener('click', createNewConversation);

// ==========================================
// 4. PROJECT WORKSPACE & FILE MANAGER
// ==========================================
async function scanProjectDirectory(customPath) {
  const path = customPath !== undefined ? customPath : projPathInput.value.trim();

  btnScanProject.textContent = '⏳';
  btnScanProject.disabled = true;

  try {
    const res = await fetch('/v1/project/scan', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ path: path || undefined })
    });
    const data = await res.json();
    if (res.ok) {
      projPathInput.value = data.root_path;
      currentProjectName = data.project_name;
      projNameText.textContent = data.project_name;
      projTypeText.textContent = `[${data.project_type}]`;
      projFilesCountText.textContent = `${data.total_files} files`;
      allProjectFiles = data.files || [];
      renderProjectFileList(allProjectFiles);
      showToast(`📁 Indexed ${data.total_files} files in ${data.project_name}`);
    } else {
      showToast(`❌ Scan error: ${data.error || 'Failed to scan'}`);
    }
  } catch (err) {
    showToast(`❌ Network error: ${err.message}`);
  } finally {
    btnScanProject.textContent = 'Scan';
    btnScanProject.disabled = false;
  }
}

function getFileIcon(ext) {
  switch ((ext || '').toLowerCase()) {
    case 'rs': return '🦀';
    case 'py': return '🐍';
    case 'js':
    case 'ts':
    case 'tsx':
    case 'jsx': return '⚡';
    case 'go': return '🐹';
    case 'c':
    case 'cpp':
    case 'h':
    case 'hpp': return '⚙️';
    case 'html':
    case 'css': return '🌐';
    case 'json':
    case 'toml':
    case 'yaml':
    case 'yml': return '📋';
    case 'md': return '📝';
    case 'sh':
    case 'zsh': return '💻';
    case 'pdf': return '📄';
    default: return '📄';
  }
}

function formatBytes(bytes) {
  if (!bytes || bytes === 0) return '0 B';
  if (bytes < 1024) return bytes + ' B';
  if (bytes < 1024 * 1024) return (bytes / 1024).toFixed(1) + ' KB';
  return (bytes / (1024 * 1024)).toFixed(1) + ' MB';
}

function renderProjectFileList(files) {
  const filterVal = (projFilterInput.value || '').toLowerCase().trim();

  const filtered = filterVal
    ? files.filter(f => f.relative_path.toLowerCase().includes(filterVal) || f.file_name.toLowerCase().includes(filterVal))
    : files;

  if (filtered.length === 0) {
    projFileList.innerHTML = '<div style="padding:16px; text-align:center; color:var(--text-dim); font-size:12px;">No matching files found</div>';
    return;
  }

  projFileList.innerHTML = '';
  filtered.slice(0, 150).forEach(f => {
    const row = document.createElement('div');
    row.className = 'proj-file-row';

    const left = document.createElement('div');
    left.className = 'proj-file-left';
    left.title = `${f.relative_path} (${formatBytes(f.size_bytes)})`;

    const icon = document.createElement('span');
    icon.textContent = getFileIcon(f.extension);

    const name = document.createElement('span');
    name.className = 'proj-file-path';
    name.textContent = f.relative_path;

    left.appendChild(icon);
    left.appendChild(name);

    const actions = document.createElement('div');
    actions.className = 'proj-file-actions';

    const viewBtn = document.createElement('button');
    viewBtn.className = 'proj-action-btn';
    viewBtn.title = 'View file code';
    viewBtn.textContent = '👁 View';
    viewBtn.onclick = () => viewProjectFile(f.relative_path);

    const attachBtn = document.createElement('button');
    attachBtn.className = 'proj-action-btn';
    attachBtn.title = 'Attach file to prompt';
    attachBtn.textContent = '📎 Attach';
    attachBtn.onclick = () => attachProjectFile(f.relative_path, f.file_name, f.size_bytes);

    actions.appendChild(viewBtn);
    actions.appendChild(attachBtn);

    row.appendChild(left);
    row.appendChild(actions);
    projFileList.appendChild(row);
  });
}

async function viewProjectFile(relPath) {
  const rootPath = projPathInput.value.trim();
  showToast(`⏳ Loading ${relPath}...`);
  try {
    const res = await fetch('/v1/project/file', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ root_path: rootPath, file_path: relPath })
    });
    const data = await res.json();
    if (res.ok) {
      previewedFile = {
        path: relPath,
        content: data.content,
        size: data.size_bytes,
      };
      modalFileName.textContent = `${relPath} (${formatBytes(data.size_bytes)})`;
      modalFileContent.textContent = data.content;
      filePreviewModal.classList.add('show');
    } else {
      showToast(`❌ Error: ${data.error || 'Failed to read file'}`);
    }
  } catch (err) {
    showToast(`❌ Network error: ${err.message}`);
  }
}

async function attachProjectFile(relPath, fileName, sizeBytes) {
  const rootPath = projPathInput.value.trim();
  showToast(`⏳ Attaching ${fileName}...`);
  try {
    const res = await fetch('/v1/project/file', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ root_path: rootPath, file_path: relPath })
    });
    const data = await res.json();
    if (res.ok) {
      currentAttachment = {
        filename: fileName,
        file_type: 'CODE',
        size_bytes: data.size_bytes,
        metadata_summary: `${relPath} (${formatBytes(data.size_bytes)})`,
        extracted_text: data.content,
        data: null
      };
      attachmentBar.style.display = 'block';
      attachmentThumb.innerHTML = getFileIcon((fileName.split('.').pop() || ''));
      attachmentName.textContent = fileName;
      attachmentMeta.textContent = `✔ Project File: ${currentAttachment.metadata_summary}`;
      promptInput.focus();
      showToast(`📎 Attached ${fileName} to prompt`);
    } else {
      showToast(`❌ Could not attach: ${data.error || 'Error'}`);
    }
  } catch (err) {
    showToast(`❌ Error: ${err.message}`);
  }
}

// Modal Events
modalAttachBtn.addEventListener('click', () => {
  if (previewedFile) {
    const fileName = previewedFile.path.split('/').pop() || 'file';
    currentAttachment = {
      filename: fileName,
      file_type: 'CODE',
      size_bytes: previewedFile.size,
      metadata_summary: `${previewedFile.path} (${formatBytes(previewedFile.size)})`,
      extracted_text: previewedFile.content,
      data: null
    };
    attachmentBar.style.display = 'block';
    attachmentThumb.innerHTML = getFileIcon((fileName.split('.').pop() || ''));
    attachmentName.textContent = fileName;
    attachmentMeta.textContent = `✔ Project File: ${currentAttachment.metadata_summary}`;
    filePreviewModal.classList.remove('show');
    promptInput.focus();
    showToast(`📎 Attached ${fileName} to prompt`);
  }
});

[modalCloseBtn, modalDismissBtn].forEach(b => {
  if (b) b.addEventListener('click', () => filePreviewModal.classList.remove('show'));
});

filePreviewModal.addEventListener('click', (e) => {
  if (e.target === filePreviewModal) {
    filePreviewModal.classList.remove('show');
  }
});

btnScanProject.addEventListener('click', () => scanProjectDirectory());
projFilterInput.addEventListener('input', () => renderProjectFileList(allProjectFiles));

// Workspace Quick Action Prompts
document.getElementById('btnPromptArch').addEventListener('click', () => {
  const name = currentProjectName || 'this workspace';
  const path = projPathInput.value.trim();
  promptInput.value = `Analyze the architecture of project "${name}" (${path}). Explain the overall system design, primary modules, communication pathways, and concurrency model.`;
  promptInput.style.height = 'auto';
  promptInput.style.height = promptInput.scrollHeight + 'px';
  promptInput.focus();
});

document.getElementById('btnPromptAudit').addEventListener('click', () => {
  const name = currentProjectName || 'this workspace';
  promptInput.value = `Perform an in-depth security and bug audit of "${name}". Inspect for edge cases, potential panics, race conditions, memory leaks, or improper error handling.`;
  promptInput.style.height = 'auto';
  promptInput.style.height = promptInput.scrollHeight + 'px';
  promptInput.focus();
});

document.getElementById('btnPromptOptimize').addEventListener('click', () => {
  const name = currentProjectName || 'this workspace';
  promptInput.value = `Review the computation bottlenecks in "${name}". Recommend processor-level optimizations specifically for Apple Silicon (Metal 3 GPU shaders, NEON/AMX vectorization, cache tiling, zero-copy LPDDR5 bandwidth utilization).`;
  promptInput.style.height = 'auto';
  promptInput.style.height = promptInput.scrollHeight + 'px';
  promptInput.focus();
});

// ==========================================
// 5. ATTACHMENT & FILE UPLOADS
// ==========================================
attachBtn.addEventListener('click', () => fileInput.click());
attachmentRemoveBtn.addEventListener('click', clearAttachment);

fileInput.addEventListener('change', (e) => {
  if (e.target.files && e.target.files.length > 0) {
    processUploadedFile(e.target.files[0]);
  }
  fileInput.value = '';
});

function clearAttachment() {
  currentAttachment = null;
  attachmentProcessingPromise = null;
  attachmentBar.style.display = 'none';
  attachmentThumb.innerHTML = '📎';
  attachmentName.textContent = '';
  attachmentMeta.textContent = '';
}

function processUploadedFile(file) {
  if (!file) return;

  currentAttachment = {
    filename: file.name,
    file_type: (file.name.split('.').pop() || 'file').toUpperCase(),
    size_bytes: file.size,
    metadata_summary: 'Extracting on Apple Silicon...',
    extracted_text: null,
    data: null,
  };

  attachmentBar.style.display = 'block';
  attachmentName.textContent = file.name;
  attachmentMeta.textContent = '⏳ Extracting text on Apple Silicon (PDFKit / Vision OCR)...';

  if (file.type && file.type.startsWith('image/')) {
    const url = URL.createObjectURL(file);
    attachmentThumb.innerHTML = `<img src="${url}" alt="thumbnail">`;
  } else {
    const ext = (file.name.split('.').pop() || '').toLowerCase();
    attachmentThumb.innerHTML = `<span style="font-size:16px;">${getFileIcon(ext)}</span>`;
  }

  attachmentProcessingPromise = new Promise((resolve) => {
    const reader = new FileReader();
    reader.onload = async function() {
      const base64Data = reader.result;
      if (currentAttachment) {
        currentAttachment.data = base64Data;
      }

      try {
        const res = await fetch('/v1/attachments/process', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            filename: file.name,
            data: base64Data
          })
        });
        if (res.ok) {
          const result = await res.json();
          if (currentAttachment) {
            currentAttachment.extracted_text = result.extracted_text;
            currentAttachment.metadata_summary = result.metadata_summary;
            currentAttachment.file_type = result.file_type;
          }
          attachmentMeta.textContent = `✔ ${result.file_type}: ${result.metadata_summary}`;
          showToast(`📎 Attached: ${file.name} (${result.metadata_summary})`);
          resolve(result);
        } else {
          let errMsg = `HTTP ${res.status}`;
          try {
            const errJson = await res.json();
            errMsg = errJson.error || errMsg;
          } catch {
            try {
              const errTxt = await res.text();
              if (errTxt) errMsg = errTxt;
            } catch {}
          }
          attachmentMeta.textContent = `⚠️ Extraction fallback: ${errMsg}`;
          showToast(`⚠️ ${errMsg}`);
          resolve(null);
        }
      } catch (err) {
        attachmentMeta.textContent = `Attached (${err.message})`;
        showToast(`📎 File attached`);
        resolve(null);
      }
    };
    reader.onerror = function() {
      showToast(`❌ Failed to read file`);
      resolve(null);
    };
    reader.readAsDataURL(file);
  });
}

// Clipboard screenshot paste
window.addEventListener('paste', (e) => {
  const items = e.clipboardData && e.clipboardData.items;
  if (!items) return;
  for (let i = 0; i < items.length; i++) {
    if (items[i].kind === 'file') {
      const file = items[i].getAsFile();
      if (file) {
        const ext = (file.type && file.type.split('/')[1]) || 'png';
        const name = file.name && file.name !== 'image.png' ? file.name : `screenshot_${Date.now()}.${ext}`;
        const renamed = new File([file], name, { type: file.type || 'image/png' });
        processUploadedFile(renamed);
        showToast('📸 Screenshot pasted from clipboard!');
        break;
      }
    }
  }
});

// Drag and Drop
['dragenter', 'dragover'].forEach(eventName => {
  window.addEventListener(eventName, (e) => {
    e.preventDefault();
    e.stopPropagation();
    dropZone.classList.add('drag-active');
  }, false);
});

['dragleave', 'drop'].forEach(eventName => {
  window.addEventListener(eventName, (e) => {
    e.preventDefault();
    e.stopPropagation();
    dropZone.classList.remove('drag-active');
  }, false);
});

window.addEventListener('drop', (e) => {
  if (e.dataTransfer && e.dataTransfer.files && e.dataTransfer.files.length > 0) {
    processUploadedFile(e.dataTransfer.files[0]);
  }
});

// ==========================================
// 6. SIDEBAR & KEYBOARD CONTROLS
// ==========================================
sidebarToggle.addEventListener('click', () => {
  sidebar.classList.toggle('collapsed');
  const isCollapsed = sidebar.classList.contains('collapsed');
  localStorage.setItem('nirvana_sidebar_collapsed', isCollapsed ? '1' : '0');
});

if (localStorage.getItem('nirvana_sidebar_collapsed') === '1') {
  sidebar.classList.add('collapsed');
}

window.addEventListener('keydown', (e) => {
  if ((e.ctrlKey || e.metaKey) && e.key.toLowerCase() === 'b') {
    e.preventDefault();
    sidebarToggle.click();
  }
  if ((e.ctrlKey || e.metaKey) && e.key.toLowerCase() === 'o') {
    e.preventDefault();
    copyFullConversation();
  }
  if (e.key === 'Escape') {
    if (isGenerating) {
      e.preventDefault();
      forceUnstickAndReset();
    }
  }
});

// Auto-resize textarea
promptInput.addEventListener('input', () => {
  promptInput.style.height = 'auto';
  promptInput.style.height = Math.min(promptInput.scrollHeight, 200) + 'px';
});

promptInput.addEventListener('keydown', (e) => {
  if (e.key === 'Enter' && !e.shiftKey && !e.altKey) {
    e.preventDefault();
    handleSend();
  }
});

sendBtn.addEventListener('click', handleSend);

function showToast(msg) {
  toast.textContent = msg;
  toast.classList.add('show');
  setTimeout(() => toast.classList.remove('show'), 2500);
}

// Code-block copy buttons carry their text in data-copy (entity-escaped by
// escapeHtml, decoded by the browser) so model output never lands in an
// inline event handler.
document.addEventListener('click', (e) => {
  const btn = e.target.closest('button[data-copy]');
  if (btn) copyText(btn.dataset.copy);
});

function copyText(text) {
  navigator.clipboard.writeText(text).then(() => {
    showToast('✔ Copied to clipboard!');
  }).catch(() => {
    showToast('❌ Copy failed');
  });
}

function copyFullConversation() {
  if (chatHistory.length === 0) {
    showToast('No conversation to copy');
    return;
  }
  let full = '';
  chatHistory.forEach(m => {
    full += `### ${m.role.toUpperCase()}\n${m.content}\n\n`;
  });
  copyText(full.trim());
}

document.getElementById('btnCopyAll').addEventListener('click', copyFullConversation);

document.getElementById('btnClearChat').addEventListener('click', () => {
  chatHistory.length = 0;
  messagesViewport.innerHTML = '';
  messagesViewport.appendChild(heroSplash);
  heroSplash.style.display = 'block';
  currentCtxUsed = 0;
  updateContextBar(0, currentCtxCapacity);

  const conv = conversations.find(c => c.id === currentConvId);
  if (conv) {
    conv.messages = [];
    conv.ctxUsed = 0;
    conv.updatedAt = Date.now();
    saveConversationsToStorage();
    renderConversationList();
  }
  showToast('✔ Conversation & KV cache cleared');
});

// ==========================================
// 7. MODEL SELECTION & DISCOVERY
// ==========================================
async function loadModelInfo() {
  try {
    const res = await fetch('/v1/models');
    const data = await res.json();
    if (data.data && data.data.length > 0) {
      modelSelect.innerHTML = '';
      const current = data.current_model || '';
      data.data.forEach(m => {
        const opt = document.createElement('option');
        opt.value = m.id;
        const isMlx = m.owned_by === 'MLX' || (m.display_name && m.display_name.includes('[MLX]'));
        let baseName = (m.display_name || m.id).replace(/\s*\[(MLX|GGUF)\]/gi, '').trim();
        const backendTag = isMlx ? ' [MLX]' : ' [GGUF]';
        const sizeStr = m.size_bytes ? ` (${m.size_bytes >= 1073741824 ? (m.size_bytes / (1024*1024*1024)).toFixed(1) + ' GB' : Math.round(m.size_bytes / (1024 * 1024)) + ' MB'})` : '';
        opt.textContent = `${baseName}${backendTag}${sizeStr}`;
        if (m.active || m.id === current) {
          opt.selected = true;
        }
        modelSelect.appendChild(opt);
      });
      specModel.textContent = current || data.data[0].id;
    }
  } catch (err) {
    specModel.textContent = 'Nirvana Silicon';
  }
}

modelSelect.addEventListener('change', async () => {
  const selected = modelSelect.value;
  if (!selected) return;

  showToast(`⏳ Loading ${selected} onto Apple Silicon Metal GPU...`);
  modelStatus.style.color = 'var(--neon-amber)';
  modelStatus.textContent = '⏳ Loading into Unified LPDDR5 RAM...';
  modelSelect.disabled = true;
  sendBtn.disabled = true;

  try {
    const res = await fetch('/v1/models/load', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ model: selected })
    });
    const result = await res.json();
    if (res.ok) {
      specModel.textContent = result.model || selected;
      modelStatus.style.color = 'var(--neon-green)';
      modelStatus.textContent = '✔ Active on Metal 3 GPU';
      showToast(`✔ Loaded: ${result.model || selected}`);
      await fetchContextInfo();
    } else {
      showToast(`❌ Error: ${result.error || 'Failed to switch model'}`);
      modelStatus.style.color = 'var(--neon-magenta)';
      modelStatus.textContent = '❌ Model load failed';
    }
  } catch (err) {
    showToast(`❌ Network error: ${err.message}`);
    modelStatus.style.color = 'var(--neon-magenta)';
    modelStatus.textContent = '❌ Switch error';
  } finally {
    modelSelect.disabled = false;
    sendBtn.disabled = false;
  }
});

async function showInteractiveModelList() {
  try {
    const res = await fetch('/v1/models');
    const data = await res.json();
    if (!data.data || data.data.length === 0) {
      showToast('No models installed');
      return;
    }

    heroSplash.style.display = 'none';

    const msgDiv = document.createElement('div');
    msgDiv.className = 'message system';
    msgDiv.style.border = '1px solid var(--neon-cyan)';
    msgDiv.style.borderRadius = '8px';
    msgDiv.style.padding = '14px';
    msgDiv.style.marginBottom = '16px';
    msgDiv.style.background = 'rgba(0, 240, 255, 0.04)';

    const headerDiv = document.createElement('div');
    headerDiv.style.fontWeight = '700';
    headerDiv.style.color = 'var(--neon-cyan)';
    headerDiv.style.marginBottom = '10px';
    headerDiv.style.display = 'flex';
    headerDiv.style.alignItems = 'center';
    headerDiv.style.justifyContent = 'space-between';
    headerDiv.innerHTML = `
      <span>⚡ AVAILABLE APPLE SILICON MODELS (${data.data.length} INSTALLED)</span>
      <span style="font-size: 11px; color: var(--text-dim); font-weight: normal;">Type /model &lt;1..${data.data.length}&gt; or click below</span>
    `;

    const bodyDiv = document.createElement('div');
    bodyDiv.style.display = 'flex';
    bodyDiv.style.flexDirection = 'column';
    bodyDiv.style.gap = '8px';

    data.data.forEach((m, idx) => {
      const isCurrent = m.active || m.id === data.current_model;
      const sizeMb = m.size_bytes ? Math.round(m.size_bytes / (1024 * 1024)) : 0;
      const sizeLabel = sizeMb >= 1024 ? (sizeMb / 1024).toFixed(1) + ' GB' : sizeMb + ' MB';
      const isMlx = m.owned_by === 'MLX' || (m.display_name && m.display_name.includes('[MLX]'));
      const backendBadge = isMlx
        ? '<span class="badge" style="background: rgba(255, 100, 200, 0.15); color: #ff66cc; border: 1px solid rgba(255, 100, 200, 0.4); margin-left: 8px; font-size: 10px; padding: 2px 6px;">Apple MLX</span>'
        : '<span class="badge" style="background: rgba(0, 240, 255, 0.15); color: var(--neon-cyan); border: 1px solid rgba(0, 240, 255, 0.4); margin-left: 8px; font-size: 10px; padding: 2px 6px;">Metal 3 GGUF</span>';
      const backendDesc = isMlx ? 'Apple Silicon MLX &bull; GPU Unified Memory' : 'Apple Silicon Metal 3 GPU &bull; Unified Memory';

      const card = document.createElement('div');
      card.style.background = isCurrent ? 'rgba(0, 255, 159, 0.08)' : 'rgba(255, 255, 255, 0.03)';
      card.style.border = `1px solid ${isCurrent ? 'var(--neon-green)' : 'var(--border-dim)'}`;
      card.style.borderRadius = '6px';
      card.style.padding = '10px 14px';
      card.style.display = 'flex';
      card.style.alignItems = 'center';
      card.style.justifyContent = 'space-between';
      card.style.gap = '12px';

      const info = document.createElement('div');
      const cleanName = (m.display_name || m.id).replace(/\s*\[(MLX|GGUF)\]/gi, '').trim();
      info.innerHTML = `
        <div style="display: flex; align-items: center; flex-wrap: wrap; gap: 4px;">
          <span style="color: var(--neon-amber); font-weight: 700; margin-right: 6px;">[${idx + 1}]</span>
          <strong style="color: ${isCurrent ? 'var(--neon-green)' : 'var(--text-bright)'}; font-size: 13px;">${escapeHtml(cleanName)}</strong>
          ${backendBadge}
        </div>
        <div style="font-size: 11px; color: var(--text-dim); margin-top: 3px;">
          ${sizeLabel} &bull; ${backendDesc}
        </div>
      `;

      const actionDiv = document.createElement('div');
      if (isCurrent) {
        actionDiv.innerHTML = '<span class="badge badge-green" style="padding: 4px 10px;">● ACTIVE</span>';
      } else {
        const btn = document.createElement('button');
        btn.className = 'copy-btn';
        btn.style.background = 'var(--neon-cyan)';
        btn.style.color = 'var(--bg-dark)';
        btn.style.fontWeight = '700';
        btn.style.cursor = 'pointer';
        btn.style.padding = '5px 12px';
        btn.style.fontSize = '12px';
        btn.textContent = '⚡ Select Model';
        btn.onclick = () => window.switchModelDirect(m.id);
        actionDiv.appendChild(btn);
      }

      card.appendChild(info);
      card.appendChild(actionDiv);
      bodyDiv.appendChild(card);
    });

    msgDiv.appendChild(headerDiv);
    msgDiv.appendChild(bodyDiv);
    messagesViewport.appendChild(msgDiv);
    messagesViewport.scrollTop = messagesViewport.scrollHeight;

    sidebar.classList.remove('collapsed');
  } catch (err) {
    showToast(`❌ Failed to list models: ${err.message}`);
  }
}

window.switchModelDirect = async function(modelId) {
  showToast(`⏳ Loading ${modelId}...`);
  try {
    const res = await fetch('/v1/models/load', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ model: modelId })
    });
    const result = await res.json();
    if (res.ok) {
      showToast(`✔ Switched to ${result.model || modelId}`);
      await loadModelInfo();
      await fetchContextInfo();
      await showInteractiveModelList();
    } else {
      showToast(`❌ ${result.error || 'Switch failed'}`);
    }
  } catch (e) {
    showToast(`❌ Error: ${e.message}`);
  }
};

// ==========================================
// 8. MARKDOWN & RENDERING
// ==========================================
function renderMarkdown(content) {
  if (!content) return '';

  // 1. Process <think> ... </think> or unclosed in-flight <think> ...
  let processed = content;
  const thinkMatches = [];

  // Replace completed <think> ... </think>
  processed = processed.replace(/<think>([\s\S]*?)<\/think>/gi, (match, thought) => {
    const idx = thinkMatches.length;
    thinkMatches.push({ thought: thought.trim(), active: false });
    return `__THINK_PLACEHOLDER_${idx}__`;
  });

  // Replace unclosed in-flight <think> ...
  processed = processed.replace(/<think>([\s\S]*)$/gi, (match, thought) => {
    const idx = thinkMatches.length;
    thinkMatches.push({ thought: thought.trim(), active: true });
    return `__THINK_PLACEHOLDER_${idx}__`;
  });

  // 2. Process code blocks
  const codeBlockRegex = /```([a-zA-Z0-9_\-\+]*)\n([\s\S]*?)```/g;
  let html = '';
  let lastIndex = 0;
  let match;

  while ((match = codeBlockRegex.exec(processed)) !== null) {
    const textBefore = processed.substring(lastIndex, match.index);
    if (textBefore) {
      html += parseTextFormatting(textBefore);
    }

    const lang = match[1] || 'code';
    const codeContent = match[2];
    const escapedCode = escapeHtml(codeContent);

    html += `
      <div class="code-block-wrapper">
        <div class="code-block-header">
          <span>${lang}</span>
          <button class="copy-btn" data-copy="${escapeHtml(codeContent)}">📋 Copy Code</button>
        </div>
        <pre><code>${escapedCode}</code></pre>
      </div>
    `;
    lastIndex = match.index + match[0].length;
  }

  const remaining = processed.substring(lastIndex);
  if (remaining) {
    html += parseTextFormatting(remaining);
  }

  // 3. Restore thinking blocks with muted styling
  thinkMatches.forEach((tm, idx) => {
    const placeholder = `__THINK_PLACEHOLDER_${idx}__`;
    const escapedThought = escapeHtml(tm.thought);
    const thinkingHtml = `
      <details class="thinking-block ${tm.active ? 'thinking-active' : ''}" open>
        <summary class="thinking-header">
          <span class="thinking-icon ${tm.active ? 'thinking-pulse' : ''}">💭</span>
          <span class="thinking-title">${tm.active ? 'Thinking...' : 'Thought Process'}</span>
          <span class="thinking-badge ${tm.active ? 'thinking-badge-active' : ''}">${tm.active ? 'In Progress' : 'Reasoning'}</span>
        </summary>
        <div class="thinking-content">${escapedThought}</div>
      </details>
    `;
    // Function replacer: a literal replacement string would interpret "$&" / "$'" in the thought
    html = html.replace(placeholder, () => thinkingHtml);
  });

  return html;
}

function parseTextFormatting(text) {
  return text
    .split('\n\n')
    .map(p => {
      let line = escapeHtml(p);
      line = line.replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>');
      line = line.replace(/`([^`]+)`/g, '<code style="background: rgba(255,255,255,0.1); padding: 2px 5px; border-radius: 3px;">$1</code>');
      return `<p>${line.replace(/\n/g, '<br>')}</p>`;
    })
    .join('');
}

function escapeHtml(str) {
  return (str || '')
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;')
    .replace(/'/g, '&#039;');
}

function appendMessage(role, initialContent = '', attachmentInfo = null, autoScroll = true) {
  heroSplash.style.display = 'none';

  const msgDiv = document.createElement('div');
  msgDiv.className = `message ${role}`;

  const headerDiv = document.createElement('div');
  headerDiv.className = 'message-header';
  headerDiv.textContent = role === 'user' ? 'USER >' : '⚡ NIRVANA ASSISTANT';

  const bodyDiv = document.createElement('div');
  bodyDiv.className = 'message-body';

  if (attachmentInfo) {
    const badge = document.createElement('div');
    badge.className = 'msg-attachment-badge';
    const ext = (attachmentInfo.file_type || '').toUpperCase();
    const icon = ext === 'PDF' ? '📄' : (ext === 'IMAGE' ? '🖼️' : (ext === 'CODE' ? '💻' : '📎'));
    badge.innerHTML = `${icon} <strong>${escapeHtml(attachmentInfo.filename || 'attachment')}</strong> <span style="opacity:0.8;">[${escapeHtml(attachmentInfo.metadata_summary || attachmentInfo.file_type || 'File')}]</span>`;
    bodyDiv.appendChild(badge);
  }

  if (initialContent) {
    const textWrapper = document.createElement('div');
    textWrapper.innerHTML = renderMarkdown(initialContent);
    bodyDiv.appendChild(textWrapper);
  }

  msgDiv.appendChild(headerDiv);
  msgDiv.appendChild(bodyDiv);

  if (role === 'assistant') {
    const footerDiv = document.createElement('div');
    footerDiv.className = 'message-footer';
    const copyBtn = document.createElement('button');
    copyBtn.className = 'msg-action-btn';
    copyBtn.innerHTML = '📋 Copy Response';
    copyBtn.addEventListener('click', () => {
      copyText(bodyDiv.innerText);
    });
    footerDiv.appendChild(copyBtn);
    msgDiv.appendChild(footerDiv);
  }

  messagesViewport.appendChild(msgDiv);
  if (autoScroll) {
    messagesViewport.scrollTop = messagesViewport.scrollHeight;
  }

  return bodyDiv;
}

function detectRepetitionLoop(text) {
  if (!text || text.length < 60) return false;
  const tail = text.slice(-240);
  for (let len = 6; len <= 40; len++) {
    if (tail.length >= len * 4) {
      const pat = tail.slice(-len);
      const prev1 = tail.slice(-len * 2, -len);
      const prev2 = tail.slice(-len * 3, -len * 2);
      const prev3 = tail.slice(-len * 4, -len * 3);
      if (pat === prev1 && prev1 === prev2 && prev2 === prev3) {
        return true;
      }
    }
  }
  return false;
}

async function forceUnstickAndReset() {
  showToast('🛑 Halting tasks & unsticking engine...');
  if (abortController) {
    abortController.abort();
  }
  isGenerating = false;
  sendLabel.textContent = 'Send';
  sendBtn.classList.remove('stop-btn');

  try {
    await Promise.all([
      fetch('/v1/chat/stop', { method: 'POST' }),
      fetch('/v1/engine/reset', { method: 'POST' })
    ]);
    showToast('✔ Engine reset & unstick complete');
  } catch (e) {
    showToast(`Reset notice: ${e.message}`);
  }
}

const btnEmergencyStop = document.getElementById('btnEmergencyStop');
if (btnEmergencyStop) btnEmergencyStop.addEventListener('click', forceUnstickAndReset);

const btnForceResetEngine = document.getElementById('btnForceResetEngine');
if (btnForceResetEngine) btnForceResetEngine.addEventListener('click', forceUnstickAndReset);

// ==========================================
// 9. INFERENCE SEND & SSE STREAMING
// ==========================================
async function handleSend() {
  if (isGenerating) {
    if (abortController) {
      abortController.abort();
    }
    fetch('/v1/chat/stop', { method: 'POST' }).catch(() => {});
    isGenerating = false;
    sendLabel.textContent = 'Send';
    sendBtn.classList.remove('stop-btn');
    showToast('⏹ Generation stopped');
    return;
  }

  const input = promptInput.value.trim();
  if (!input && !currentAttachment) return;

  // Handle slash commands
  if (input.startsWith('/')) {
    const parts = input.split(' ');
    const cmd = parts[0].toLowerCase();
    if (cmd === '/model' || cmd === '/models' || cmd === '/switch' || cmd === '/load') {
      promptInput.value = '';
      promptInput.style.height = 'auto';
      if (parts.length > 1) {
        const target = parts.slice(1).join(' ');
        window.switchModelDirect(target);
      } else {
        await showInteractiveModelList();
      }
      return;
    } else if (cmd === '/clear') {
      promptInput.value = '';
      promptInput.style.height = 'auto';
      document.getElementById('btnClearChat').click();
      return;
    } else if (cmd === '/detach') {
      promptInput.value = '';
      promptInput.style.height = 'auto';
      clearAttachment();
      showToast('✔ Attachment removed');
      return;
    } else if (cmd === '/attach' || cmd === '/file') {
      promptInput.value = '';
      promptInput.style.height = 'auto';
      if (parts.length > 1) {
        const path = parts.slice(1).join(' ');
        const fname = path.split('/').pop() || 'attachment';
        showToast(`⏳ Attaching ${fname}...`);
        fetch('/v1/attachments/process', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ filename: fname, data: path })
        }).then(r => r.json()).then(res => {
          if (res.status === 'success') {
            currentAttachment = res;
            attachmentBar.style.display = 'block';
            attachmentName.textContent = res.filename;
            attachmentMeta.textContent = `✔ ${res.file_type}: ${res.metadata_summary}`;
            showToast(`📎 Attached: ${res.filename}`);
          } else {
            showToast(`❌ Failed: ${res.error || 'Could not attach'}`);
          }
        }).catch(e => showToast(`❌ Error: ${e.message}`));
      } else {
        fileInput.click();
      }
      return;
    }
  }

  if (attachmentProcessingPromise) {
    if (attachmentMeta) attachmentMeta.textContent = '⏳ Finalizing extraction before sending...';
    await attachmentProcessingPromise;
    attachmentProcessingPromise = null;
  }

  const attachedToSend = currentAttachment;
  clearAttachment();

  const userText = input || `Please analyze the attached ${attachedToSend ? (attachedToSend.file_type || 'file') : 'file'} (${attachedToSend ? attachedToSend.filename : ''}).`;
  const userMsgObj = { role: 'user', content: userText, attachmentInfo: attachedToSend };
  chatHistory.push(userMsgObj);
  appendMessage('user', userText, attachedToSend);
  saveCurrentConversation();

  promptInput.value = '';
  promptInput.style.height = 'auto';

  // Prepare generation
  isGenerating = true;
  sendLabel.textContent = 'Stop';
  sendBtn.classList.add('stop-btn');

  const assistantBody = appendMessage('assistant', '');
  let fullAssistantText = '';
  const startTime = performance.now();
  let firstTokenTime = null;
  let tokenCount = 0;

  abortController = new AbortController();

  try {
    const systemPrompt = "You are Nirvana Code, an ultra-fast Apple Silicon coding assistant. Provide clean, correct, high-performance code.";
    const messages = [
      { role: 'system', content: systemPrompt },
      ...chatHistory
    ];

    const useNgram = checkNgramSpec ? checkNgramSpec.checked : true;

    const response = await fetch('/v1/chat/completions', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        model: modelSelect.value || undefined,
        messages,
        stream: true,
        temperature: 0.7,
        max_tokens: 2048,
        ngram_speculative: useNgram,
        attachment: attachedToSend ? {
          filename: attachedToSend.filename,
          data: attachedToSend.extracted_text ? null : attachedToSend.data,
          extracted_text: attachedToSend.extracted_text,
          metadata_summary: attachedToSend.metadata_summary,
          file_type: attachedToSend.file_type,
        } : undefined,
      }),
      signal: abortController.signal,
    });

    if (!response.ok) {
      throw new Error(`HTTP error ${response.status}`);
    }

    const reader = response.body.getReader();
    const decoder = new TextDecoder('utf-8');
    let buffer = '';

    while (true) {
      const { done, value } = await reader.read();
      if (done) break;

      buffer += decoder.decode(value, { stream: true });
      const lines = buffer.split('\n');
      buffer = lines.pop();

      for (const line of lines) {
        const trimmed = line.trim();
        if (!trimmed || trimmed.startsWith(':')) continue;
        if (trimmed === 'data: [DONE]') break;

        if (trimmed.startsWith('data: ')) {
          try {
            const parsed = JSON.parse(trimmed.substring(6));

            // Process stats payload if present
            if (parsed.stats) {
              const stats = parsed.stats;
              if (stats.ttft_ms) metricTtft.textContent = `${Math.round(stats.ttft_ms)} ms`;
              if (stats.tokens_per_sec) metricSpeed.textContent = `${stats.tokens_per_sec.toFixed(1)} tok/s`;
              if (stats.total_tokens) metricTokens.textContent = stats.total_tokens;
              if (stats.context_used !== undefined && stats.context_capacity !== undefined) {
                updateContextBar(stats.context_used, stats.context_capacity);
              }
            }

            const piece = parsed.choices?.[0]?.delta?.content;
            if (piece) {
              if (firstTokenTime === null) {
                firstTokenTime = performance.now();
                const ttft = Math.round(firstTokenTime - startTime);
                metricTtft.textContent = `${ttft} ms`;
              }
              tokenCount++;
              fullAssistantText += piece;

              if (detectRepetitionLoop(fullAssistantText)) {
                console.warn('Degeneration repetition loop detected, halting generation');
                if (abortController) abortController.abort();
                fetch('/v1/chat/stop', { method: 'POST' }).catch(() => {});
                isGenerating = false;
                sendLabel.textContent = 'Send';
                sendBtn.classList.remove('stop-btn');
                fullAssistantText += "\n\n⚠️ *[Generation halted: repetition degeneration loop detected]*";
                assistantBody.innerHTML = renderMarkdown(fullAssistantText);
                showToast('⚠️ Repetition loop detected — generation halted');
                break;
              }

              assistantBody.innerHTML = renderMarkdown(fullAssistantText);
              messagesViewport.scrollTop = messagesViewport.scrollHeight;

              const elapsedSec = (performance.now() - firstTokenTime) / 1000;
              if (elapsedSec > 0.1) {
                const tps = (tokenCount / elapsedSec).toFixed(1);
                metricSpeed.textContent = `${tps} tok/s`;
              }
              metricTokens.textContent = tokenCount;
            }
          } catch (e) {}
        }
      }
    }

    // Save assistant response
    chatHistory.push({ role: 'assistant', content: fullAssistantText });
    saveCurrentConversation();

  } catch (err) {
    if (err.name !== 'AbortError') {
      assistantBody.innerHTML += `<p style="color: var(--neon-magenta)">❌ Error: ${escapeHtml(String(err && err.message || err))}</p>`;
    }
  } finally {
    isGenerating = false;
    sendLabel.textContent = 'Send';
    sendBtn.classList.remove('stop-btn');
  }
}

// ==========================================
// 10. INITIALIZATION
// ==========================================
switchSidebarTab('conversations');
loadConversationsFromStorage();
loadModelInfo();
fetchContextInfo();
scanProjectDirectory();
