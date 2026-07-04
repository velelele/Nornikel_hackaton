const messagesEl = document.getElementById("messages");
const chatForm = document.getElementById("chatForm");
const chatInput = document.getElementById("chatInput");
const sendBtn = chatForm?.querySelector('.btn-send');
const uploadBtn = document.getElementById("uploadBtn");
const folderUploadBtn = document.getElementById("folderUploadBtn");
const fileInput = document.getElementById("fileInput");
const folderInput = document.getElementById("folderInput");
const fastStageFolderInput = document.getElementById("fastStageFolderInput");
const uploadProgress = document.getElementById("uploadProgress");
const uploadProgressText = document.getElementById("uploadProgressText");
const statusBadge = document.getElementById("statusBadge");
const entityCount = document.getElementById("entityCount");
const documentCount = document.getElementById("documentCount");
const documentList = document.getElementById("documentList");
const menuCountMobile = document.getElementById("menuCountMobile");
const labelsPreview = document.getElementById("labelsPreview");
const statsToggle = document.getElementById("statsToggle");
const statsPanel = document.getElementById("statsPanel");
const sidebarBackdrop = document.getElementById("sidebarBackdrop");
const chatSessionList = document.getElementById("chatSessionList");
const newChatBtn = document.getElementById("newChatBtn");
const toast = document.getElementById("toast");
const fastIndexTab = document.getElementById("fastIndexTab");
const exactIndexTab = document.getElementById("exactIndexTab");
const fastIndexPanel = document.getElementById("fastIndexPanel");
const exactIndexPanel = document.getElementById("exactIndexPanel");
const fastFolderStage1Btn = document.getElementById("fastFolderStage1Btn");
const exactFilesBtn = document.getElementById("exactFilesBtn");
const exactFolderBtn = document.getElementById("exactFolderBtn");
const stage2StartBtn = document.getElementById("stage2StartBtn");
const stage2ModeSelect = document.getElementById("stage2ModeSelect");
const stage2ChunksInput = document.getElementById("stage2ChunksInput");
const stage2ThemeIdsInput = document.getElementById("stage2ThemeIdsInput");
const themeHints = document.getElementById("themeHints");
const ingestionStatus = document.getElementById("ingestionStatus");
const operationPanel = document.getElementById("operationPanel");
const operationTitle = document.getElementById("operationTitle");
const operationPercent = document.getElementById("operationPercent");
const operationProgressBar = document.getElementById("operationProgressBar");
const operationProgressFill = document.getElementById("operationProgressFill");
const operationProgressText = document.getElementById("operationProgressText");
const operationCurrent = document.getElementById("operationCurrent");

const STORAGE_KEY = "nico-chat-sessions-v2";
let appSymbol = "◈";
let appTitle = "NiCo";
const WELCOME_TEXT =
  "◈NiCo готов к работе: загружайте статьи, отчёты, протоколы и презентации — система поможет находить связи и отвечать на вопросы по корпусу.";

let queryMode = "hybrid";
let sessions = [];
let activeSessionId = null;

const STATS_POLL_MS = 3000;
const TRACK_POLL_MS = 2500;
const TRACK_POLL_TIMEOUT_MS = 25 * 60 * 1000;
const INFLIGHT_DOC_STATUSES = new Set([
  "pending",
  "processing",
  "parsing",
  "analyzing",
  "preprocessed",
]);

let statsPollTimer = null;
let activeTrackPolls = 0;
let lastHealth = null;
let lastStats = null;
let chatReady = false;
let uploadInProgress = false;
let activeIngestionRunId = null;
let ingestionPollTimer = null;

const FILE_UPLOAD_BATCH_SIZE = 3;
const FOLDER_UPLOAD_BATCH_SIZE = 1;
const FAST_FOLDER_UPLOAD_BATCH_SIZE = 3;
const FAST_STAGE_UPLOAD_BATCH_SIZE = 10;

let pendingFileUploadOptions = { source: "files", profile: "balanced", flow: "exact" };
let pendingFolderUploadOptions = { source: "folder", profile: "fast_fill", flow: "fast_stage1" };
let activeIndexMode = "fast";

// Hints are used only for browser folder uploads.
// When a user selects the whole root_dir, Chromium sends paths as
// root_dir/collection/theme/file. In that case the first artificial
// root segment should be stripped before sending names to backend.
const TOPIC_COLLECTION_HINTS = new Set([
  "журналы", "журнал", "journals", "journal", "articles", "papers",
  "конференции", "конференция", "conferences", "conference", "proceedings",
  "патенты", "патент", "patents", "patent",
  "отчеты", "отчёты", "отчет", "отчёт", "reports", "report",
  "нормативы", "стандарты", "гост", "gost", "standards", "standard",
  "презентации", "presentations", "presentation",
  "книги", "books", "book",
]);


function renderBrandText(element, value, { markOnly = false } = {}) {
  if (!element) return;
  const raw = String(value || (markOnly ? "◈" : "NiCo")).trim() || (markOnly ? "◈" : "NiCo");
  const diamondMatch = raw.match(/^([◈◇◆✦✧])\s*(.*)$/u);
  const diamond = diamondMatch ? diamondMatch[1] : "";
  const word = (diamondMatch ? diamondMatch[2] : raw).trim();
  element.innerHTML = "";
  if (diamond || markOnly) {
    const diamondSpan = document.createElement("span");
    diamondSpan.className = "brand-diamond";
    diamondSpan.textContent = diamond || "◈";
    element.appendChild(diamondSpan);
  }
  if (!markOnly && word) {
    const wordSpan = document.createElement("span");
    wordSpan.className = "brand-word";
    wordSpan.textContent = word;
    element.appendChild(wordSpan);
  }
}

function sleep(ms) {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

function hasInflightDocuments(documents = []) {
  return documents.some((doc) => INFLIGHT_DOC_STATUSES.has(doc.status));
}

function startStatsPolling() {
  if (statsPollTimer) return;
  statsPollTimer = setInterval(() => {
    refreshStats().catch(() => {});
  }, STATS_POLL_MS);
}

function stopStatsPolling() {
  if (!statsPollTimer) return;
  clearInterval(statsPollTimer);
  statsPollTimer = null;
}

function updateStatsPolling(stats) {
  if (uploadInProgress || stats?.pipeline_busy || hasInflightDocuments(stats?.documents) || activeTrackPolls > 0) {
    startStatsPolling();
    return;
  }
  stopStatsPolling();
}

function isMobileLayout() {
  return window.matchMedia("(max-width: 768px)").matches;
}

function getKnowledgeReadiness(stats, health) {
  const documents = stats?.documents || [];
  const processedCount = stats?.document_count ?? documents.filter((doc) => doc.status === "processed").length;
  const processingCount = stats?.processing_count ?? 0;
  const busy = Boolean(stats?.pipeline_busy || health?.pipeline_busy || activeTrackPolls > 0 || uploadInProgress);
  const inflight = hasInflightDocuments(documents);
  const ready = Boolean((stats?.knowledge_ready ?? (processedCount > 0 && !processingCount && !busy && !inflight)) && health?.rag_ready !== false);
  return { ready, processedCount, processingCount, busy, inflight };
}

function setChatAvailability(stats = lastStats, health = lastHealth) {
  const state = getKnowledgeReadiness(stats, health);
  chatReady = state.ready;

  if (chatInput) {
    chatInput.disabled = !chatReady;
    chatInput.placeholder = chatReady
      ? "Задайте вопрос по загруженным документам..."
      : !health?.rag_ready
        ? "Сначала настройте LLM/embeddings и дождитесь инициализации базы..."
        : state.processedCount === 0
          ? "Сначала загрузите документы и дождитесь построения графа знаний..."
          : "Граф знаний строится. Чат станет доступен после завершения обработки...";
  }
  if (sendBtn) sendBtn.disabled = !chatReady;

  return state;
}

function setPipelineStatus(stats, health) {
  if (!statusBadge) return;

  const state = setChatAvailability(stats || lastStats, health || lastHealth);

  if (health && !health.rag_ready) {
    statusBadge.textContent = "Требуется настройка LLM";
    statusBadge.className = "badge glass-badge error";
    return;
  }

  if (state.busy || state.processingCount > 0 || state.inflight) {
    const activeRun = (health?.active_ingestion_runs || stats?.active_ingestion_runs || [])[0];
    statusBadge.textContent = activeRun
      ? `${activeRun.stage === "stage1" ? "Stage 1" : "Stage 2"}: ${activeRun.processed || 0}/${activeRun.total || "?"}`
      : state.processingCount > 0
        ? `Строится граф (${state.processingCount})…`
        : "Строится граф знаний…";
    statusBadge.className = "badge glass-badge";
    return;
  }

  if (state.ready) {
    statusBadge.textContent = stats?.stage2_ready ? "Stage 2: KG готов" : "Поиск готов";
    statusBadge.className = "badge glass-badge ok";
    return;
  }

  statusBadge.textContent = state.processedCount > 0 ? "Граф не готов" : "Загрузите документы";
  statusBadge.className = "badge glass-badge";
}

function createId() {
  return crypto.randomUUID?.() || `${Date.now()}-${Math.random().toString(16).slice(2)}`;
}

function createSession(title = "Новый чат") {
  return {
    id: createId(),
    title,
    messages: [],
    updatedAt: new Date().toISOString(),
    titleGenerated: false,
  };
}

function getActiveSession() {
  return sessions.find((session) => session.id === activeSessionId) || null;
}

function saveSessions() {
  localStorage.setItem(
    STORAGE_KEY,
    JSON.stringify({ activeSessionId, sessions: sessions.slice(0, 50) }),
  );
}

function loadSessions() {
  try {
    const raw = localStorage.getItem(STORAGE_KEY);
    if (!raw) return false;
    const data = JSON.parse(raw);
    sessions = Array.isArray(data.sessions) ? data.sessions : [];
    activeSessionId = data.activeSessionId || sessions[0]?.id || null;
    return sessions.length > 0;
  } catch {
    sessions = [];
    activeSessionId = null;
    return false;
  }
}

function setStatsPanelOpen(open) {
  if (!statsPanel || !sidebarBackdrop || !statsToggle) return;
  statsPanel.classList.toggle("open", open);
  sidebarBackdrop.hidden = !open;
  sidebarBackdrop.classList.toggle("visible", open);
  statsToggle.setAttribute("aria-expanded", String(open));
  document.body.style.overflow = open ? "hidden" : "";
}

function showToast(text, isError = false, options = {}) {
  const { persistent = false, success = false } = options;
  toast.textContent = text;
  toast.hidden = false;
  toast.classList.toggle("error", isError);
  toast.classList.toggle("success", success && !isError);
  clearTimeout(showToast.timer);
  if (!persistent) {
    showToast.timer = setTimeout(() => {
      toast.hidden = true;
    }, 4000);
  }
}

function hideToast() {
  clearTimeout(showToast.timer);
  toast.hidden = true;
}

function showConfirm({
  title = "Подтвердите действие",
  message = "",
  hint = "",
  confirmLabel = "Подтвердить",
  cancelLabel = "Отмена",
  danger = false,
} = {}) {
  const modal = document.getElementById("confirmModal");
  const titleEl = document.getElementById("confirmTitle");
  const messageEl = document.getElementById("confirmMessage");
  const hintEl = document.getElementById("confirmHint");
  const cancelBtn = document.getElementById("confirmCancel");
  const okBtn = document.getElementById("confirmOk");

  if (!modal || !titleEl || !messageEl || !hintEl || !cancelBtn || !okBtn) {
    return Promise.resolve(false);
  }

  return new Promise((resolve) => {
    titleEl.textContent = title;
    messageEl.textContent = message;
    hintEl.textContent = hint;
    hintEl.hidden = !hint;
    cancelBtn.textContent = cancelLabel;
    okBtn.textContent = confirmLabel;
    okBtn.classList.toggle("btn-danger", danger);
    okBtn.classList.toggle("btn-primary", !danger);

    const finish = (result) => {
      modal.hidden = true;
      modal.setAttribute("aria-hidden", "true");
      document.body.classList.remove("modal-open");
      document.removeEventListener("keydown", onKeydown);
      resolve(result);
    };

    const onKeydown = (event) => {
      if (event.key === "Escape") finish(false);
    };

    cancelBtn.onclick = () => finish(false);
    okBtn.onclick = () => finish(true);
    modal.querySelectorAll("[data-confirm-dismiss]").forEach((el) => {
      el.onclick = () => finish(false);
    });

    modal.hidden = false;
    modal.setAttribute("aria-hidden", "false");
    document.body.classList.add("modal-open");
    document.addEventListener("keydown", onKeydown);
    cancelBtn.focus();
  });
}

function setButtonLoading(button, active, isPrimary = false) {
  if (!button) return;
  button.disabled = active;
  button.setAttribute("aria-busy", String(active));
  button.classList.toggle("is-loading", active && isPrimary);

  const spinner = button.querySelector(".btn-spinner");
  if (spinner) spinner.hidden = !(active && isPrimary);

  const fullLabel = button.querySelector(".btn-text-full");
  const shortLabel = button.querySelector(".btn-text-short");

  if (active && isPrimary) {
    if (fullLabel) {
      if (!button.dataset.originalFullLabel) {
        button.dataset.originalFullLabel = fullLabel.textContent;
      }
      fullLabel.textContent = "Загрузка…";
    }
    if (shortLabel) {
      if (!button.dataset.originalShortLabel) {
        button.dataset.originalShortLabel = shortLabel.textContent;
      }
      shortLabel.textContent = "…";
    }
  } else if (!active) {
    if (fullLabel && button.dataset.originalFullLabel) {
      fullLabel.textContent = button.dataset.originalFullLabel;
    }
    if (shortLabel && button.dataset.originalShortLabel) {
      shortLabel.textContent = button.dataset.originalShortLabel;
    }
  }
}

function setIndexMode(mode) {
  activeIndexMode = mode === "exact" ? "exact" : "fast";
  const fast = activeIndexMode === "fast";

  fastIndexTab?.classList.toggle("active", fast);
  exactIndexTab?.classList.toggle("active", !fast);
  fastIndexTab?.setAttribute("aria-selected", fast ? "true" : "false");
  exactIndexTab?.setAttribute("aria-selected", fast ? "false" : "true");

  if (fastIndexPanel) {
    fastIndexPanel.hidden = !fast;
    fastIndexPanel.classList.toggle("active", fast);
  }
  if (exactIndexPanel) {
    exactIndexPanel.hidden = fast;
    exactIndexPanel.classList.toggle("active", !fast);
  }

  if (ingestionStatus && !activeIngestionRunId && !uploadInProgress) {
    ingestionStatus.textContent = fast
      ? "Быстрый запуск: выберите папку. После Stage 1 поиск будет доступен сразу; граф можно достроить отдельной кнопкой."
      : "Полная сборка: прежняя логика с немедленным построением LightRAG runtime для небольших наборов документов.";
  }
}


function formatProgressPercent(processed, total) {
  if (!Number.isFinite(total) || total <= 0) return null;
  const value = Math.max(0, Math.min(100, Math.round((processed / total) * 100)));
  return value;
}

function setOperationProgress({
  active = false,
  title = "Операция не запущена",
  processed = 0,
  total = 0,
  text = "Выберите режим индексирования.",
  current = "",
  error = false,
  done = false,
} = {}) {
  if (!operationPanel) return;
  operationPanel.classList.toggle("active", active || done || error);
  operationPanel.classList.toggle("error", Boolean(error));
  operationPanel.classList.toggle("done", Boolean(done) && !error);

  const numericProcessed = Number(processed) || 0;
  const numericTotal = Number(total) || 0;
  const percent = formatProgressPercent(numericProcessed, numericTotal);
  // Use an indeterminate animation only when the backend cannot provide a denominator.
  // Previously any non-empty current message forced indeterminate mode, so Stage 2
  // looked like it had no progress bar even when processed/total were known.
  const showActivity = active && percent === null;
  if (operationTitle) operationTitle.textContent = title;
  if (operationProgressText) operationProgressText.textContent = text;
  if (operationCurrent) operationCurrent.textContent = current || "";

  if (operationProgressBar) {
    operationProgressBar.classList.toggle("indeterminate", showActivity);
    operationProgressBar.setAttribute("aria-valuenow", percent === null ? "0" : String(percent));
  }
  if (operationProgressFill) {
    operationProgressFill.style.width = showActivity ? "38%" : (percent === null ? "35%" : `${percent}%`);
  }
  if (operationPercent) {
    operationPercent.textContent = percent === null ? (active ? "идёт" : "—") : `${percent}%`;
  }
}

function setUploadLoading(active, message = "Загрузка документов…", source = "files") {
  setButtonLoading(uploadBtn, active, source === "files");
  setButtonLoading(folderUploadBtn, active, source === "folder");

  if (active) {
    setOperationProgress({ active: true, title: message.includes("Stage 1") ? "Быстрый запуск" : "Полная сборка", text: message });
  }

  if (uploadProgress && uploadProgressText) {
    // Progress is now rendered inside the Indexing panel; keep the old floating
    // bubble disabled to avoid duplicated controls.
    uploadProgress.hidden = true;
    if (active) uploadProgressText.textContent = message;
  }
}

function renderMessage(role, text, sources = []) {
  const wrapper = document.createElement("div");
  wrapper.className = `message ${role}`;

  const avatar = document.createElement("div");
  avatar.className = "avatar";
  avatar.setAttribute("aria-hidden", "true");
  avatar.textContent = role === "user" ? "Вы" : appSymbol;

  const bubble = document.createElement("div");
  bubble.className = role === "assistant" ? "bubble glass-bubble" : "bubble";

  const textEl = document.createElement("p");
  textEl.className = "bubble-text";
  textEl.textContent = text;
  bubble.appendChild(textEl);

  if (role === "assistant" && sources.length) {
    const sourcesEl = document.createElement("div");
    sourcesEl.className = "sources";

    const label = document.createElement("span");
    label.className = "sources-label";
    label.textContent = "Источники:";
    sourcesEl.appendChild(label);

    sources.forEach((source, index) => {
      if (index > 0) sourcesEl.appendChild(document.createTextNode(", "));
      const item = document.createElement("span");
      item.className = "source-item";
      item.title = source.file_path || source.filename;
      const label = source.reference_id
        ? `[${source.reference_id}] ${source.filename}`
        : source.filename;
      item.textContent = label;
      sourcesEl.appendChild(item);
    });

    bubble.appendChild(sourcesEl);
  }

  wrapper.appendChild(avatar);
  wrapper.appendChild(bubble);
  messagesEl.appendChild(wrapper);
}

function renderWelcome() {
  renderMessage("assistant", WELCOME_TEXT);
}

function renderChatMessages() {
  messagesEl.innerHTML = "";
  const session = getActiveSession();
  if (!session || !session.messages.length) {
    renderWelcome();
    return;
  }
  for (const msg of session.messages) {
    renderMessage(msg.role, msg.content, msg.sources || []);
  }
  scrollMessagesToBottom();
}

function formatSessionMeta(session) {
  const date = formatSessionDate(session.updatedAt);
  const count = session.messages?.length || 0;
  const msgLabel =
    count === 0 ? "пустой" : count === 1 ? "1 сообщ." : `${count} сообщ.`;
  return `${date} · ${msgLabel}`;
}

function renderSessionList() {
  if (!chatSessionList) return;
  chatSessionList.innerHTML = "";

  const sorted = [...sessions].sort(
    (a, b) => new Date(b.updatedAt).getTime() - new Date(a.updatedAt).getTime(),
  );

  if (!sorted.length) {
    const empty = document.createElement("li");
    empty.className = "chat-session-empty";
    empty.innerHTML = "<span>Нет сохранённых чатов</span><span>Нажмите «+ Новый», чтобы начать</span>";
    chatSessionList.appendChild(empty);
    return;
  }

  for (const session of sorted) {
    const item = document.createElement("li");
    item.className = `chat-session-item${session.id === activeSessionId ? " active" : ""}`;

    const openBtn = document.createElement("button");
    openBtn.type = "button";
    openBtn.className = "chat-session-open";
    openBtn.title = session.title;

    const titleEl = document.createElement("span");
    titleEl.className = "chat-session-title";
    titleEl.textContent = session.title;

    const metaEl = document.createElement("span");
    metaEl.className = "chat-session-meta";
    metaEl.textContent = formatSessionMeta(session);

    openBtn.appendChild(titleEl);
    openBtn.appendChild(metaEl);
    openBtn.addEventListener("click", () => switchSession(session.id));

    const deleteBtn = document.createElement("button");
    deleteBtn.type = "button";
    deleteBtn.className = "chat-session-delete";
    deleteBtn.setAttribute("aria-label", "Удалить чат");
    deleteBtn.textContent = "×";
    deleteBtn.addEventListener("click", (event) => {
      event.stopPropagation();
      deleteSession(session.id);
    });

    item.appendChild(openBtn);
    item.appendChild(deleteBtn);
    chatSessionList.appendChild(item);
  }
}

function formatSessionDate(iso) {
  const date = new Date(iso);
  const now = new Date();
  const sameDay =
    date.getDate() === now.getDate() &&
    date.getMonth() === now.getMonth() &&
    date.getFullYear() === now.getFullYear();
  if (sameDay) {
    return date.toLocaleTimeString("ru-RU", { hour: "2-digit", minute: "2-digit" });
  }
  return date.toLocaleDateString("ru-RU", { day: "numeric", month: "short" });
}

function switchSession(sessionId) {
  if (activeSessionId === sessionId) {
    if (isMobileLayout()) setStatsPanelOpen(false);
    return;
  }
  activeSessionId = sessionId;
  saveSessions();
  renderSessionList();
  renderChatMessages();
  if (isMobileLayout()) setStatsPanelOpen(false);
}

function startNewChat() {
  const empty = sessions.find((s) => !s.messages.length);
  if (empty) {
    switchSession(empty.id);
    return;
  }
  const session = createSession();
  sessions.unshift(session);
  activeSessionId = session.id;
  saveSessions();
  renderSessionList();
  renderChatMessages();
  chatInput.focus();
  if (isMobileLayout()) setStatsPanelOpen(false);
}

function deleteSession(sessionId) {
  sessions = sessions.filter((session) => session.id !== sessionId);
  if (!sessions.length) {
    const session = createSession();
    sessions.push(session);
    activeSessionId = session.id;
  } else if (activeSessionId === sessionId) {
    activeSessionId = sessions[0].id;
  }
  saveSessions();
  renderSessionList();
  renderChatMessages();
}

function appendMessage(role, text, sources = []) {
  const session = getActiveSession();
  if (!session) return;

  session.messages.push({
    role,
    content: text,
    sources: sources.length ? sources : undefined,
  });
  session.updatedAt = new Date().toISOString();
  saveSessions();
  renderMessage(role, text, sources);
  scrollMessagesToBottom();
  renderSessionList();
}

async function maybeUpdateSessionTitle() {
  const session = getActiveSession();
  if (!session || session.titleGenerated || session.messages.length < 2) return;

  const userMessages = session.messages.filter((m) => m.role === "user");
  if (!userMessages.length) return;

  try {
    const payload = session.messages.map(({ role, content }) => ({ role, content }));
    const result = await api("/api/chat/title", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ messages: payload }),
    });
    if (result.title) {
      session.title = result.title;
      session.titleGenerated = true;
      session.updatedAt = new Date().toISOString();
      saveSessions();
      renderSessionList();
    }
  } catch {
    const first = userMessages[0].content.trim();
    session.title = first.length > 48 ? `${first.slice(0, 48)}…` : first;
    session.titleGenerated = true;
    saveSessions();
    renderSessionList();
  }
}

function appendTypingIndicator() {
  const wrapper = document.createElement("div");
  wrapper.className = "message assistant typing-indicator";
  wrapper.id = "typingIndicator";
  wrapper.innerHTML = `
    <div class="avatar" aria-hidden="true">${appSymbol}</div>
    <div class="bubble glass-bubble"><p class="bubble-text">${appSymbol} думает…</p></div>
  `;
  messagesEl.appendChild(wrapper);
  scrollMessagesToBottom();
}

function removeTypingIndicator() {
  document.getElementById("typingIndicator")?.remove();
}

function scrollMessagesToBottom() {
  messagesEl.scrollTop = messagesEl.scrollHeight;
}

function autoResizeTextarea() {
  chatInput.style.height = "auto";
  chatInput.style.height = `${Math.min(chatInput.scrollHeight, window.innerHeight * 0.3)}px`;
}

function updateKnowledgeCounts(documents, entities) {
  const docText = String(documents ?? "—");
  const entityText = String(entities ?? "—");
  if (documentCount) documentCount.textContent = docText;
  if (entityCount) entityCount.textContent = entityText;
  if (menuCountMobile) {
    menuCountMobile.textContent =
      documents == null || entities == null ? "—" : `${docText} · ${entityText}`;
  }
}

function canDeleteDocument(doc) {
  if (!doc?.id || doc.deletable === false) return false;
  return !INFLIGHT_DOC_STATUSES.has(doc.status);
}

async function deleteDocument(doc) {
  if (!canDeleteDocument(doc)) return;

  const confirmed = await showConfirm({
    title: "Удалить документ?",
    message: doc.filename,
    hint: "Граф знаний и связанные фрагменты будут удалены без возможности восстановления.",
    confirmLabel: "Удалить",
    cancelLabel: "Отмена",
    danger: true,
  });
  if (!confirmed) return;

  try {
    const result = await api(`/api/knowledge/documents/${encodeURIComponent(doc.id)}`, {
      method: "DELETE",
    });
    showToast(
      result.message || `Документ «${doc.filename}» удалён из базы знаний.`,
      false,
      { success: true },
    );
    await refreshStats();
  } catch (error) {
    showToast(error.message, true);
    setOperationProgress({ error: true, title: "Построение графа знаний / Stage 2", text: error.message });
  }
}

function renderDocumentList(documents = []) {
  if (!documentList) return;
  documentList.innerHTML = "";

  if (!documents.length) {
    const empty = document.createElement("li");
    empty.className = "document-empty";
    empty.textContent = "Пока нет загруженных документов";
    documentList.appendChild(empty);
    return;
  }

  for (const doc of documents) {
    const item = document.createElement("li");
    item.className = `document-item glass-inset status-${doc.status || "unknown"}`;

    const main = document.createElement("div");
    main.className = "document-main";

    const name = document.createElement("span");
    name.className = "document-name";
    name.title = doc.filename;
    name.textContent = doc.filename;

    const meta = document.createElement("span");
    meta.className = "document-meta";
    const statusLabel = formatDocumentStatus(doc.status);
    const chunks = doc.chunks ? `${doc.chunks} фрагм.` : "";
    const chars = doc.chars ? `${doc.chars.toLocaleString("ru-RU")} симв.` : "";
    const theme = doc.theme_id ? `тема: ${doc.theme_id}` : "";
    meta.textContent = [statusLabel, chunks, chars, theme].filter(Boolean).join(" · ");

    main.appendChild(name);
    main.appendChild(meta);

    if (doc.status === "failed" && doc.error) {
      const error = document.createElement("span");
      error.className = "document-error";
      error.title = doc.error;
      error.textContent = doc.error;
      main.appendChild(error);
    }

    item.appendChild(main);

    if (canDeleteDocument(doc)) {
      const deleteBtn = document.createElement("button");
      deleteBtn.type = "button";
      deleteBtn.className = "document-delete";
      deleteBtn.setAttribute("aria-label", `Удалить ${doc.filename}`);
      deleteBtn.title = "Удалить из базы знаний";
      deleteBtn.textContent = "×";
      deleteBtn.addEventListener("click", () => deleteDocument(doc));
      item.appendChild(deleteBtn);
    }

    documentList.appendChild(item);
  }
}

function formatDocumentStatus(status) {
  switch (status) {
    case "processed":
      return "готово";
    case "pending":
      return "в очереди";
    case "processing":
    case "parsing":
    case "analyzing":
    case "preprocessed":
      return "обработка…";
    case "failed":
      return "ошибка";
    default:
      return status || "";
  }
}

async function api(path, options = {}) {
  const response = await fetch(path, options);
  const data = await response.json().catch(() => ({}));
  if (!response.ok) {
    let detail = data.detail || "Ошибка запроса";
    if (response.status === 404 && detail === "Not Found") {
      detail = "Сервис не поддерживает удаление. Перезапустите сервер (python run.py).";
    }
    throw new Error(typeof detail === "string" ? detail : "Ошибка запроса");
  }
  return data;
}

async function loadConfig() {
  const config = await api("/api/config");
  queryMode = config.query_mode || "hybrid";
  appSymbol = config.app_symbol || appSymbol;
  appTitle = config.app_title || appTitle;
  document.title = appTitle;
  const brandTitle = document.querySelector(".brand-title");
  const brandMark = document.querySelector(".brand-mark");
  renderBrandText(brandTitle, appTitle);
  renderBrandText(brandMark, appSymbol, { markOnly: true });
  const profiles = config.ingestion_profiles || {};
  const retrievalProfile = profiles.overnight_retrieval_kg || {};
  if (stage2ChunksInput && retrievalProfile.max_chunks_per_document_for_graph) {
    stage2ChunksInput.value = String(retrievalProfile.max_chunks_per_document_for_graph);
    stage2ChunksInput.placeholder = String(retrievalProfile.max_chunks_per_document_for_graph);
  }
  updateStage2Controls();
}

async function refreshHealth() {
  try {
    lastHealth = await api("/api/health");
    setPipelineStatus(lastStats, lastHealth);
  } catch {
    lastHealth = null;
    statusBadge.textContent = "Сервис недоступен";
    statusBadge.className = "badge glass-badge error";
  }
}


function renderThemeHints(themes = []) {
  if (!themeHints) return;
  themeHints.innerHTML = "";
  const values = [];
  const seen = new Set();
  for (const row of themes || []) {
    for (const value of [row.theme_id, row.collection, row.theme_name]) {
      const text = String(value || "").trim();
      if (!text || seen.has(text)) continue;
      seen.add(text);
      values.push(text);
    }
  }
  for (const value of values.slice(0, 300)) {
    const option = document.createElement("option");
    option.value = value;
    themeHints.appendChild(option);
  }
}

function updateStage2Controls() {
  const selectedMode = stage2ModeSelect?.value || "retrieval_kg";
  const lightragMode = selectedMode === "compressed_lightrag" || selectedMode === "full_kg";
  if (stage2ChunksInput) {
    stage2ChunksInput.disabled = selectedMode === "retrieval_kg";
    stage2ChunksInput.placeholder = selectedMode === "retrieval_kg" ? "не требуется" : (selectedMode === "compressed_lightrag" ? "4" : "6");
  }
  if (stage2ThemeIdsInput) {
    stage2ThemeIdsInput.placeholder = lightragMode
      ? "необязательно: конференции, ALTA или theme_id; пусто = все темы"
      : "необязательно: конференции, ALTA или theme_id";
  }
}

async function refreshStats() {
  try {
    const stats = await api("/api/knowledge/stats");
    lastStats = stats;
    updateKnowledgeCounts(stats.document_count ?? 0, stats.entities);
    renderDocumentList(stats.documents || []);
    renderThemeHints(stats.themes || []);
    const activeRun = (stats.active_ingestion_runs || [])[0];
    labelsPreview.textContent = activeRun
      ? `${activeRun.stage === "stage1" ? "Stage 1" : "Stage 2"}: ${activeRun.processed || 0}/${activeRun.total || "?"}; ok=${activeRun.ok || 0}; failed=${activeRun.failed || 0}; ${activeRun.current || ""}`
      : stats.labels_preview?.length
        ? `Примеры сущностей: ${stats.labels_preview.join(", ")}`
        : stats.processing_count
          ? `Обрабатывается документов: ${stats.processing_count}. Граф обновится по завершении.`
          : stats.stage2_ready
            ? `Stage 2 выполнен: построен граф знаний для тем ${stats.stage2_ready_theme_count || 0}; LightRAG runtime: ${stats.runtime_ready_theme_count || 0}. Поиск использует KG/retrieval-слой поверх knowledge_store.`
            : stats.knowledge_ready
              ? "Поиск доступен. Stage 2 ещё не выполнен для графа знаний."
              : stats.document_count
              ? "Документы есть, но индекс ещё не готов."
              : "Загрузите документы, чтобы построить базу знаний.";
    setPipelineStatus(stats, lastHealth);
    updateStatsPolling(stats);
    return stats;
  } catch {
    lastStats = null;
    updateKnowledgeCounts(null, null);
    renderDocumentList([]);
    labelsPreview.textContent = "";
    setChatAvailability(null, lastHealth);
    return null;
  }
}

function setIngestionButtons(active) {
  if (fastFolderStage1Btn) fastFolderStage1Btn.disabled = active || uploadInProgress;
  if (exactFilesBtn) exactFilesBtn.disabled = active || uploadInProgress;
  if (exactFolderBtn) exactFolderBtn.disabled = active || uploadInProgress;
  if (stage2StartBtn) stage2StartBtn.disabled = active || uploadInProgress;
}

function renderIngestionRun(run) {
  if (!run) return;
  const total = run.total || 0;
  const processed = run.processed || 0;
  const ok = run.ok || 0;
  const failed = run.failed || 0;
  const skipped = run.skipped || 0;
  const phase = run.phase || run.status || "";
  const detail = run.progress_detail || "";
  const stageName = run.stage === "stage1" ? "Быстрый запуск / Stage 1" : "Достройка графа / Stage 2";
  const current = run.current || "";
  const lastError = run.errors?.length ? run.errors[run.errors.length - 1].error : "";
  const lastMessage = run.messages?.length ? run.messages[run.messages.length - 1].message : "";
  const text = `${run.status} · ${phase} · обработано ${processed}/${total || "?"}; ok=${ok}; skipped=${skipped}; failed=${failed}`;
  const currentLine = lastError
    ? `Ошибка: ${lastError}`
    : [current, detail, lastMessage].filter(Boolean).join(" · ");

  if (ingestionStatus) {
    ingestionStatus.textContent = `${stageName}: ${text}${currentLine ? `; ${currentLine}` : ""}`;
  }
  setOperationProgress({
    active: ["queued", "running", "cancelling"].includes(run.status),
    done: run.status === "completed",
    error: !["queued", "running", "cancelling", "completed"].includes(run.status),
    title: stageName,
    processed,
    total,
    text,
    current: currentLine,
  });
}

async function pollIngestionRun(runId) {
  activeIngestionRunId = runId;
  setIngestionButtons(true);
  if (ingestionPollTimer) clearInterval(ingestionPollTimer);

  const tick = async () => {
    try {
      const run = await api(`/api/ingestion/runs/${encodeURIComponent(runId)}`);
      renderIngestionRun(run);
      await refreshHealth();
      await refreshStats();
      if (!["queued", "running", "cancelling"].includes(run.status)) {
        clearInterval(ingestionPollTimer);
        ingestionPollTimer = null;
        activeIngestionRunId = null;
        setIngestionButtons(false);
        showToast(
          run.status === "completed"
            ? `${run.stage === "stage1" ? "Stage 1" : "Stage 2"} завершён`
            : `${run.stage === "stage1" ? "Stage 1" : "Stage 2"}: ${run.status}`,
          run.status !== "completed",
          { success: run.status === "completed" },
        );
      }
    } catch (error) {
      clearInterval(ingestionPollTimer);
      ingestionPollTimer = null;
      activeIngestionRunId = null;
      setIngestionButtons(false);
      showToast(error.message, true);
    }
  };

  await tick();
  ingestionPollTimer = setInterval(tick, 3000);
}


function supportedUploadExtension(name) {
  const suffix = String(name || "").toLowerCase().match(/\.[^.]+$/)?.[0] || "";
  const accept = (fileInput?.accept || folderInput?.accept || "")
    .split(",")
    .map((item) => item.trim().toLowerCase())
    .filter(Boolean);
  return !accept.length || accept.includes(suffix);
}

async function collectDirectoryFiles(directoryHandle, prefix = "") {
  const files = [];
  const uploadNames = new Map();
  const entries = [];

  for await (const [name, handle] of directoryHandle.entries()) {
    entries.push([name, handle]);
  }
  entries.sort((a, b) => a[0].localeCompare(b[0], "ru"));

  for (const [name, handle] of entries) {
    const relative = prefix ? `${prefix}/${name}` : name;
    if (handle.kind === "directory") {
      const nested = await collectDirectoryFiles(handle, relative);
      for (const file of nested.files) files.push(file);
      for (const [file, uploadName] of nested.uploadNames.entries()) {
        uploadNames.set(file, uploadName);
      }
    } else if (handle.kind === "file" && supportedUploadExtension(name)) {
      const file = await handle.getFile();
      uploadNames.set(file, relative);
      files.push(file);
    }
  }
  return { files, uploadNames };
}

function openFastStageFolderInput() {
  pendingFolderUploadOptions = { source: "folder", profile: "fast_fill", flow: "fast_stage1_staged" };
  const input = fastStageFolderInput || folderInput;
  if (!input) {
    showToast("В интерфейсе не найден input для выбора папки. Перезагрузите страницу.", true);
    return;
  }
  if (!supportsFolderPicker()) {
    showToast("Этот браузер не поддерживает загрузку папок. Используйте Chromium/Chrome/Edge или CLI stage1_fast_fill.py.", true);
    return;
  }
  input.value = "";
  input.click();
}

async function startFastStage1FolderUpload() {
  if (uploadInProgress) {
    showToast("Дождитесь завершения текущей загрузки.", true);
    return;
  }

  // Chromium sometimes ignores click() on a reused hidden folder input after
  // previous uploads or after a layout re-render. Use the File System Access API
  // first when it is available, because it is a direct directory picker tied to
  // this button click. Fall back to a dedicated hidden webkitdirectory input.
  if (typeof window.showDirectoryPicker === "function") {
    try {
      setOperationProgress({
        active: true,
        title: "Быстрый запуск",
        text: "Открытие выбора папки…",
      });
      const directoryHandle = await window.showDirectoryPicker({ mode: "read" });
      const { files, uploadNames } = await collectDirectoryFiles(directoryHandle);
      if (!files.length) {
        showToast("В выбранной папке не найдено поддерживаемых документов.", true);
        setOperationProgress({ error: true, title: "Быстрый запуск", text: "Поддерживаемые файлы не найдены." });
        return;
      }
      await stagedStage1FolderUpload(files, uploadNames);
      return;
    } catch (error) {
      if (error?.name === "AbortError") {
        setOperationProgress({ title: "Операция не запущена", text: "Выбор папки отменён." });
        return;
      }
      console.warn("showDirectoryPicker failed, falling back to webkitdirectory input", error);
      // Continue to the webkitdirectory input fallback below.
    }
  }

  openFastStageFolderInput();
}

async function startStage2FromWeb() {
  const rawSelectedMode = stage2ModeSelect?.value || "retrieval_kg";
  const selectedMode = rawSelectedMode === "compressed_kg" ? "retrieval_kg" : rawSelectedMode;
  const graphMode = selectedMode === "compressed_lightrag" ? "compressed_kg" : selectedMode;
  const rawChunks = stage2ChunksInput?.value.trim();
  const maxChunksPerDocument = rawChunks ? Number(rawChunks) : null;
  const themeIds = (stage2ThemeIdsInput?.value || "")
    .split(",")
    .map((item) => item.trim())
    .filter(Boolean);
  const buildRuntimeIndex = selectedMode === "compressed_lightrag" || selectedMode === "full_kg";
  setOperationProgress({
    active: true,
    title: "Построение графа знаний / Stage 2",
    text: buildRuntimeIndex ? "Построение LightRAG-графа с контролем таймаута…" : "Построение поискового графа знаний без записи LightRAG-графа…",
  });
  const profile = selectedMode === "full_kg"
    ? "overnight_full"
    : selectedMode === "compressed_lightrag"
      ? "overnight_compressed_lightrag"
      : "overnight_retrieval_kg";
  const defaultChunks = selectedMode === "compressed_lightrag" ? 4 : 6;
  try {
    const result = await api("/api/ingestion/stage2/start", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        profile,
        graph_mode: graphMode,
        build_runtime_index: buildRuntimeIndex,
        theme_ids: themeIds,
        wait: buildRuntimeIndex,
        timeout_sec: buildRuntimeIndex ? (selectedMode === "full_kg" ? 900 : 600) : 0,
        max_chunks_per_document: selectedMode === "retrieval_kg" ? 0 : (Number.isFinite(maxChunksPerDocument) && maxChunksPerDocument > 0 ? maxChunksPerDocument : defaultChunks),
        rebuild_graph_metrics_after: false,
        rebuild_theme_embeddings_after: false,
        build_global_router_after: true,
        force: true,
      }),
    });
    showToast(result.message || "Stage 2 запущен", false, { success: true });
    await pollIngestionRun(result.run_id);
  } catch (error) {
    showToast(error.message, true);
    setOperationProgress({ error: true, title: "Построение графа знаний / Stage 2", text: error.message });
  }
}

fastIndexTab?.addEventListener("click", () => setIndexMode("fast"));
exactIndexTab?.addEventListener("click", () => setIndexMode("exact"));
fastFolderStage1Btn?.addEventListener("click", startFastStage1FolderUpload);
exactFilesBtn?.addEventListener("click", () => openFilePicker({ source: "files", profile: "balanced", flow: "exact" }));
exactFolderBtn?.addEventListener("click", () => openFolderPicker({ source: "folder", profile: "balanced", flow: "exact" }));
stage2ModeSelect?.addEventListener("change", updateStage2Controls);
stage2StartBtn?.addEventListener("click", startStage2FromWeb);

async function pollTrackStatus(trackId, { silent = false } = {}) {
  if (!trackId) return null;

  activeTrackPolls += 1;
  startStatsPolling();
  setPipelineStatus(lastStats, lastHealth);

  const deadline = Date.now() + TRACK_POLL_TIMEOUT_MS;
  let lastTrack = null;

  try {
    while (Date.now() < deadline) {
      const track = await api(`/api/knowledge/track/${encodeURIComponent(trackId)}`);
      lastTrack = track;
      const total = track.total_count || 0;
      const done = (track.processed_count || 0) + (track.failed_count || 0);
      if (uploadProgressText && uploadInProgress && total) {
        uploadProgressText.textContent = `Строится граф: ${done}/${total} объектов…`;
      }
      if (total) {
        setOperationProgress({
          active: true,
          title: "Полная сборка: построение LightRAG runtime",
          processed: done,
          total,
          text: `Строится граф: ${done}/${total} объектов`,
        });
      }
      await refreshStats();

      if (track.is_complete) {
        if (!silent) {
          if (track.processed_count > 0 && track.failed_count === 0) {
            showToast(`Обработано документов: ${track.processed_count}`);
          } else if (track.processed_count > 0) {
            showToast(
              `Готово ${track.processed_count}, ошибок ${track.failed_count}`,
              track.failed_count > 0,
            );
          } else if (track.failed_count > 0) {
            showToast(`Не удалось обработать документы (${track.failed_count})`, true);
          }
        }
        return track;
      }

      await sleep(TRACK_POLL_MS);
    }

    if (!silent) showToast("Обработка ещё идёт — статус обновится автоматически.");
    return lastTrack;
  } catch (error) {
    if (!silent) showToast(error.message, true);
    throw error;
  } finally {
    activeTrackPolls = Math.max(0, activeTrackPolls - 1);
    await refreshHealth();
    await refreshStats();
  }
}

if (statsToggle) {
  statsToggle.addEventListener("click", () => {
    const isOpen = statsPanel?.classList.contains("open");
    setStatsPanelOpen(!isOpen);
  });
}

if (newChatBtn) {
  newChatBtn.addEventListener("click", startNewChat);
}

if (sidebarBackdrop) {
  sidebarBackdrop.addEventListener("click", () => setStatsPanelOpen(false));
}

window.addEventListener("resize", () => {
  if (!isMobileLayout()) setStatsPanelOpen(false);
  autoResizeTextarea();
});

if (window.visualViewport) {
  const onViewportChange = () => {
    if (!isMobileLayout()) {
      document.documentElement.style.removeProperty("--keyboard-offset");
      return;
    }
    const offset = Math.max(
      0,
      window.innerHeight - window.visualViewport.height - window.visualViewport.offsetTop,
    );
    document.documentElement.style.setProperty("--keyboard-offset", `${offset}px`);
    scrollMessagesToBottom();
  };
  window.visualViewport.addEventListener("resize", onViewportChange);
  window.visualViewport.addEventListener("scroll", onViewportChange);
}

chatInput.addEventListener("input", autoResizeTextarea);

chatInput.addEventListener("keydown", (event) => {
  if (event.key === "Enter" && !event.shiftKey && !isMobileLayout()) {
    event.preventDefault();
    chatForm.requestSubmit();
  }
});

chatForm.addEventListener("submit", async (event) => {
  event.preventDefault();
  if (!chatReady) {
    showToast("Чат станет доступен после Stage 1 или полной сборки.", true);
    setChatAvailability(lastStats, lastHealth);
    return;
  }

  const message = chatInput.value.trim();
  if (!message) return;

  const session = getActiveSession();
  if (!session) return;

  appendMessage("user", message);
  chatInput.value = "";
  autoResizeTextarea();
  chatInput.disabled = true;
  if (sendBtn) sendBtn.disabled = true;
  appendTypingIndicator();

  const historyForRequest = session.messages
    .slice(0, -1)
    .slice(-20)
    .map(({ role, content }) => ({ role, content }));

  try {
    const result = await api("/api/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ message, mode: queryMode, history: historyForRequest }),
    });
    removeTypingIndicator();
    appendMessage("assistant", result.answer, result.sources || []);
    await maybeUpdateSessionTitle();
  } catch (error) {
    removeTypingIndicator();
    appendMessage("assistant", `Ошибка: ${error.message}`);
  } finally {
    setChatAvailability(lastStats, lastHealth);
    if (chatReady) chatInput.focus();
  }
});

function openFilePicker(options = {}) {
  if (uploadInProgress || !fileInput) return;
  pendingFileUploadOptions = {
    source: "files",
    profile: options.profile || "balanced",
    flow: options.flow || "exact",
  };
  fileInput.value = "";
  fileInput.click();
}

uploadBtn?.addEventListener("click", () => {
  if (uploadBtn.disabled) return;
  openFilePicker({ source: "files", profile: "balanced", flow: "exact" });
});

folderUploadBtn?.addEventListener("click", () => {
  if (folderUploadBtn.disabled || uploadInProgress) return;
  openFolderPicker({ source: "folder", profile: "fast_fill", flow: "fast_stage1" });
});

function supportsFolderPicker() {
  const probe = document.createElement("input");
  probe.type = "file";
  return "webkitdirectory" in probe || "directory" in probe;
}

function openFolderPicker(options = {}) {
  pendingFolderUploadOptions = {
    source: options.source || "folder",
    profile: options.profile || "fast_fill",
    flow: options.flow || "fast_stage1",
  };
  if (!supportsFolderPicker()) {
    showToast("Этот браузер не поддерживает загрузку папок. Используйте Chromium/Chrome/Edge или CLI ingest_topics.py.", true);
    return;
  }

  // Prefer the persistent hidden input from index.html. It is more reliable
  // than a short-lived dynamic input in Chromium-based browsers.
  if (folderInput) {
    folderInput.value = "";
    folderInput.click();
    return;
  }

  const input = document.createElement("input");
  input.type = "file";
  input.multiple = true;
  input.accept = fileInput?.accept || "";
  input.style.position = "fixed";
  input.style.left = "-9999px";
  input.setAttribute("webkitdirectory", "");
  input.setAttribute("directory", "");
  input.webkitdirectory = true;
  input.addEventListener("change", async () => {
    try {
      if (pendingFolderUploadOptions?.flow === "fast_stage1_staged") {
        await stagedStage1FolderUpload(input.files);
      } else {
        await uploadSelectedFiles(input.files, pendingFolderUploadOptions);
      }
    } finally {
      input.remove();
    }
  });
  document.body.appendChild(input);
  input.click();
}

function chunkArray(items, size) {
  const chunks = [];
  for (let index = 0; index < items.length; index += size) {
    chunks.push(items.slice(index, index + size));
  }
  return chunks;
}

function splitUploadPath(path) {
  return String(path || "")
    .replaceAll("\\", "/")
    .split("/")
    .map((part) => part.trim())
    .filter((part) => part && part !== "." && part !== "..");
}

function normalizeTopicPart(part) {
  return String(part || "").trim().toLowerCase().replaceAll("ё", "е");
}

function looksLikeCollection(part) {
  return TOPIC_COLLECTION_HINTS.has(normalizeTopicPart(part));
}

function rawUploadName(file) {
  return file.webkitRelativePath || file.name || "без_имени";
}

function normalizeUploadName(file, uploadNames = null) {
  return uploadNames?.get(file) || rawUploadName(file);
}

function buildFolderUploadNames(files) {
  const uploadNames = new Map();
  const fileList = Array.from(files || []);
  const paths = fileList.map((file) => splitUploadPath(rawUploadName(file)));

  const relativePathCount = fileList.filter((file) => Boolean(file.webkitRelativePath)).length;
  if (relativePathCount === 0) {
    showToast("Браузер не передал относительные пути. Документы попадут в тему misc/unclassified.", true);
  }

  const candidatePaths = paths.filter((parts) => parts.length >= 2);
  const firstSegment = candidatePaths[0]?.[0];
  const sameTopFolder = Boolean(firstSegment) && candidatePaths.every((parts) => parts[0] === firstSegment);
  const shouldStripSelectedRoot = Boolean(
    sameTopFolder &&
      candidatePaths.some((parts) => parts.length >= 4) &&
      !looksLikeCollection(firstSegment) &&
      candidatePaths.some((parts) => looksLikeCollection(parts[1])),
  );

  fileList.forEach((file, index) => {
    let parts = paths[index];
    if (shouldStripSelectedRoot && parts.length > 1) {
      parts = parts.slice(1);
    }
    uploadNames.set(file, parts.join("/") || file.name || "без_имени");
  });

  if (shouldStripSelectedRoot) {
    console.info("◈NiCo folder upload: stripped selected root folder", firstSegment);
  }
  return uploadNames;
}

async function uploadStage1StagingBatch(sessionId, batch, uploadNames = null) {
  const formData = new FormData();
  for (const file of batch) {
    formData.append("files", file, normalizeUploadName(file, uploadNames));
  }

  const response = await fetch(`/api/ingestion/stage1/upload-batch?session_id=${encodeURIComponent(sessionId)}`, {
    method: "POST",
    body: formData,
  });
  const data = await response.json().catch(() => ({}));
  if (!response.ok) {
    const detail = Array.isArray(data.detail)
      ? data.detail.map((item) => item.msg || item).join("; ")
      : data.detail;
    throw new Error(detail || "Не удалось сохранить файлы во временную папку Stage 1");
  }
  return data;
}

async function startStagedStage1(sessionId) {
  return api("/api/ingestion/stage1/staged/start", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      session_id: sessionId,
      limit: null,
      profile: "fast_fill",
      skip_existing: true,
      build_runtime_index: false,
      cleanup_after: true,
      rebuild_graph_metrics_after: false,
      rebuild_theme_embeddings_after: false,
      build_global_router_after: true,
    }),
  });
}

async function stagedStage1FolderUpload(files, explicitUploadNames = null) {
  const selected = Array.from(files || []);
  if (!selected.length) return;
  if (uploadInProgress) {
    showToast("Дождитесь завершения текущей загрузки.", true);
    return;
  }

  const uploadNames = explicitUploadNames || buildFolderUploadNames(selected);
  const batches = chunkArray(selected, FAST_STAGE_UPLOAD_BATCH_SIZE);
  const sessionId = crypto.randomUUID?.() || `${Date.now()}-${Math.random().toString(16).slice(2)}`;

  uploadInProgress = true;
  setIngestionButtons(true);
  startStatsPolling();
  if (ingestionStatus) {
    ingestionStatus.textContent = "Быстрый запуск: загрузка файлов во временное хранилище. Парсинг запустится фоном после загрузки.";
  }
  setOperationProgress({
    active: true,
    title: "Быстрый запуск: подготовка Stage 1",
    processed: 0,
    total: selected.length,
    text: "Загрузка выбранной папки на сервер без парсинга",
  });

  let saved = 0;
  let failed = 0;
  try {
    for (let index = 0; index < batches.length; index += 1) {
      const batch = batches[index];
      const batchFrom = index * FAST_STAGE_UPLOAD_BATCH_SIZE + 1;
      const batchTo = Math.min((index + 1) * FAST_STAGE_UPLOAD_BATCH_SIZE, selected.length);
      const current = normalizeUploadName(batch[0], uploadNames);
      setOperationProgress({
        active: true,
        title: "Быстрый запуск: загрузка папки",
        processed: batchFrom - 1,
        total: selected.length,
        text: `Загрузка файлов на сервер: ${batchFrom}-${batchTo}/${selected.length}`,
        current,
      });
      const data = await uploadStage1StagingBatch(sessionId, batch, uploadNames);
      saved += data.saved || 0;
      failed += data.failed || 0;
      setOperationProgress({
        active: true,
        title: "Быстрый запуск: загрузка папки",
        processed: batchTo,
        total: selected.length,
        text: `Файлы загружены: ${batchTo}/${selected.length}; ошибок сохранения ${failed}`,
        current: normalizeUploadName(batch[batch.length - 1], uploadNames),
      });
    }

    setOperationProgress({
      active: true,
      title: "Быстрый запуск / Stage 1",
      processed: selected.length,
      total: selected.length,
      text: "Файлы загружены. Запуск фонового парсинга и заполнения knowledge_store…",
    });

    const result = await startStagedStage1(sessionId);
    showToast(result.message || "Stage 1 запущен", false, { success: true });
    if (ingestionStatus) {
      ingestionStatus.textContent = `Stage 1 запущен: загружено ${saved}, ошибок загрузки ${failed}. Прогресс ниже обновляется с сервера.`;
    }
    uploadInProgress = false;
    setIngestionButtons(true);
    await pollIngestionRun(result.run_id);
  } catch (error) {
    showToast(error.message, true);
    if (ingestionStatus) ingestionStatus.textContent = `Ошибка Stage 1: ${error.message}`;
    setOperationProgress({ error: true, title: "Быстрый запуск / Stage 1", text: error.message });
  } finally {
    uploadInProgress = false;
    setIngestionButtons(false);
    if (folderInput) folderInput.value = "";
    if (fastStageFolderInput) fastStageFolderInput.value = "";
    await refreshHealth();
    await refreshStats();
  }
}

async function uploadBatch(batch, uploadNames = null, profile = "balanced") {
  const formData = new FormData();
  for (const file of batch) {
    formData.append("files", file, normalizeUploadName(file, uploadNames));
  }

  const response = await fetch(`/api/knowledge/upload/batch?profile=${encodeURIComponent(profile)}`, {
    method: "POST",
    body: formData,
  });
  const data = await response.json().catch(() => ({}));
  if (!response.ok) {
    const detail = Array.isArray(data.detail)
      ? data.detail.map((item) => item.msg || item).join("; ")
      : data.detail;
    throw new Error(detail || "Не удалось загрузить документы");
  }
  return data;
}

async function uploadSelectedFiles(files, { source = "files", profile = "balanced", flow = "exact" } = {}) {
  const selected = Array.from(files || []);
  if (!selected.length) return;
  if (uploadInProgress) {
    showToast("Дождитесь завершения текущей загрузки.", true);
    return;
  }

  const isFastStage1 = profile === "fast_fill" || flow === "fast_stage1";
  const isExact = !isFastStage1;
  const batchSize = source === "folder"
    ? (isFastStage1 ? FAST_FOLDER_UPLOAD_BATCH_SIZE : FOLDER_UPLOAD_BATCH_SIZE)
    : FILE_UPLOAD_BATCH_SIZE;
  const uploadNames = source === "folder" ? buildFolderUploadNames(selected) : null;
  const uploadProfile = profile || (isFastStage1 ? "fast_fill" : "balanced");
  const batches = chunkArray(selected, batchSize);
  const label = source === "folder" ? "из папки" : "";
  const modeLabel = isFastStage1 ? "Быстрый запуск / Stage 1" : "Полная сборка";

  uploadInProgress = true;
  setIngestionButtons(true);
  setUploadLoading(true, `${modeLabel}: подготовка ${selected.length} документов ${label}…`.replace("  ", " "), source);
  startStatsPolling();

  let succeeded = 0;
  let failed = 0;
  let firstTrackId = "";

  try {
    for (let index = 0; index < batches.length; index += 1) {
      const batch = batches[index];
      const batchFrom = index * batchSize + 1;
      const batchTo = Math.min((index + 1) * batchSize, selected.length);
      const firstName = normalizeUploadName(batch[0], uploadNames);
      const progressMessage = isFastStage1
        ? `Stage 1: сохранение в knowledge_store ${batchFrom}-${batchTo}/${selected.length}`
        : `Полная сборка: LightRAG runtime ${batchFrom}-${batchTo}/${selected.length}`;
      if (uploadProgressText) {
        uploadProgressText.textContent = `${progressMessage} — ${firstName}`;
      }
      setOperationProgress({
        active: true,
        title: isFastStage1 ? "Быстрый запуск / Stage 1" : "Полная сборка",
        processed: batchFrom - 1,
        total: selected.length,
        text: progressMessage,
        current: firstName,
      });

      const data = await uploadBatch(batch, uploadNames, uploadProfile);
      succeeded += data.succeeded || 0;
      failed += data.failed || 0;
      if (!firstTrackId && data.track_id) firstTrackId = data.track_id;
      setOperationProgress({
        active: true,
        title: isFastStage1 ? "Быстрый запуск / Stage 1" : "Полная сборка",
        processed: batchTo,
        total: selected.length,
        text: `${isFastStage1 ? "Сохранение в knowledge_store" : "Отправка в LightRAG runtime"}: ${batchTo}/${selected.length}`,
        current: normalizeUploadName(batch[batch.length - 1], uploadNames),
      });

      if (failed && !succeeded) {
        showToast(data.message || "Не удалось загрузить документы", true);
      }
      await refreshStats();
    }

    if (isFastStage1) {
      if (uploadProgressText) uploadProgressText.textContent = "Stage 1: обновление маршрутизатора тем…";
      setOperationProgress({
        active: true,
        title: "Быстрый запуск / Stage 1",
        processed: selected.length,
        total: selected.length,
        text: "Обновление маршрутизатора тем",
      });
      try {
        await api("/api/themes/rebuild-router", { method: "POST" });
      } catch (error) {
        console.warn("Router rebuild after Stage 1 upload failed", error);
      }
      await refreshStats();
      showToast(`Stage 1 завершён: сохранено ${succeeded}, ошибок ${failed}. Поиск доступен.`, failed > 0, { success: succeeded > 0 });
      if (ingestionStatus) {
        ingestionStatus.textContent = `Быстрый запуск завершён: сохранено ${succeeded}, ошибок ${failed}. Поиск доступен; граф можно достроить кнопкой «Достроить граф».`;
      }
      setOperationProgress({
        done: failed === 0,
        error: failed > 0,
        title: "Быстрый запуск / Stage 1",
        processed: selected.length,
        total: selected.length,
        text: `Поиск доступен. Сохранено ${succeeded}, ошибок ${failed}.`,
      });
      return;
    }

    const message = `Полная сборка: поставлено ${succeeded}, ошибок ${failed}.`;
    showToast(message, failed > 0, { success: succeeded > 0 });
    if (ingestionStatus) ingestionStatus.textContent = message;
    setOperationProgress({
      done: failed === 0,
      error: failed > 0,
      title: "Полная сборка",
      processed: selected.length,
      total: selected.length,
      text: message,
    });

    if (firstTrackId) {
      await pollTrackStatus(firstTrackId, { silent: true });
    }
  } catch (error) {
    showToast(error.message, true);
    if (ingestionStatus) ingestionStatus.textContent = `Ошибка загрузки: ${error.message}`;
    setOperationProgress({ error: true, title: isFastStage1 ? "Быстрый запуск / Stage 1" : "Полная сборка", text: error.message });
  } finally {
    uploadInProgress = false;
    setUploadLoading(false, "", source);
    setIngestionButtons(false);
    if (fileInput) fileInput.value = "";
    if (folderInput) folderInput.value = "";
    if (fastStageFolderInput) fastStageFolderInput.value = "";
    await refreshHealth();
    await refreshStats();
  }
}

fileInput?.addEventListener("change", async () => {
  await uploadSelectedFiles(fileInput.files, pendingFileUploadOptions);
});

fastStageFolderInput?.addEventListener("change", async () => {
  await stagedStage1FolderUpload(fastStageFolderInput.files);
});

folderInput?.addEventListener("change", async () => {
  if (pendingFolderUploadOptions?.flow === "fast_stage1_staged") {
    await stagedStage1FolderUpload(folderInput.files);
  } else {
    await uploadSelectedFiles(folderInput.files, pendingFolderUploadOptions);
  }
});

(async function init() {
  autoResizeTextarea();
  if (!loadSessions()) {
    const session = createSession();
    sessions.push(session);
    activeSessionId = session.id;
    saveSessions();
  }
  renderSessionList();
  renderChatMessages();
  setIndexMode("fast");
  setChatAvailability(null, null);

  const [configResult, healthResult, statsResult] = await Promise.allSettled([
    loadConfig(),
    refreshHealth(),
    refreshStats(),
  ]);

  if (configResult.status === "rejected") {
    showToast(configResult.reason?.message || "Не удалось загрузить настройки", true);
  }
  if (healthResult.status === "rejected") {
    showToast("Сервис недоступен", true);
  }
  if (statsResult.status === "rejected") {
    showToast("Не удалось загрузить статистику базы знаний", true);
  }
})();
