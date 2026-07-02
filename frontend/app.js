const messagesEl = document.getElementById("messages");
const chatForm = document.getElementById("chatForm");
const chatInput = document.getElementById("chatInput");
const uploadBtn = document.getElementById("uploadBtn");
const fileInput = document.getElementById("fileInput");
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

const STORAGE_KEY = "nico-chat-sessions-v1";
const WELCOME_TEXT =
  "Привет! Я NiCo AI — ваш робот-помощник. Загружайте статьи, отчёты и описания экспериментов — помогу находить связи и отвечать на вопросы по вашим документам.";

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
  if (stats?.pipeline_busy || hasInflightDocuments(stats?.documents) || activeTrackPolls > 0) {
    startStatsPolling();
    return;
  }
  stopStatsPolling();
}

function isMobileLayout() {
  return window.matchMedia("(max-width: 768px)").matches;
}

function setPipelineStatus(stats, health) {
  if (!statusBadge) return;

  if (health && !health.rag_ready) {
    statusBadge.textContent = "Требуется настройка LLM";
    statusBadge.className = "badge glass-badge error";
    return;
  }

  const processingCount = stats?.processing_count ?? 0;
  if (stats?.pipeline_busy || processingCount > 0 || activeTrackPolls > 0) {
    statusBadge.textContent =
      processingCount > 0
        ? `Обработка документов (${processingCount})…`
        : "Обработка документов…";
    statusBadge.className = "badge glass-badge";
    return;
  }

  statusBadge.textContent = health?.rag_ready ? "База знаний готова" : "Подключение…";
  statusBadge.className = `badge glass-badge ${health?.rag_ready ? "ok" : ""}`;
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

function setUploadLoading(active, message = "Загрузка документов…") {
  if (!uploadBtn) return;

  uploadBtn.disabled = active;
  uploadBtn.setAttribute("aria-busy", String(active));
  uploadBtn.classList.toggle("is-loading", active);

  const spinner = uploadBtn.querySelector(".btn-spinner");
  if (spinner) spinner.hidden = !active;

  const fullLabel = uploadBtn.querySelector(".btn-text-full");
  const shortLabel = uploadBtn.querySelector(".btn-text-short");

  if (active) {
    if (fullLabel) {
      if (!uploadBtn.dataset.originalFullLabel) {
        uploadBtn.dataset.originalFullLabel = fullLabel.textContent;
      }
      fullLabel.textContent = "Загрузка…";
    }
    if (shortLabel) {
      if (!uploadBtn.dataset.originalShortLabel) {
        uploadBtn.dataset.originalShortLabel = shortLabel.textContent;
      }
      shortLabel.textContent = "…";
    }
  } else {
    if (fullLabel && uploadBtn.dataset.originalFullLabel) {
      fullLabel.textContent = uploadBtn.dataset.originalFullLabel;
    }
    if (shortLabel && uploadBtn.dataset.originalShortLabel) {
      shortLabel.textContent = uploadBtn.dataset.originalShortLabel;
    }
  }

  if (uploadProgress && uploadProgressText) {
    uploadProgress.hidden = !active;
    if (active) uploadProgressText.textContent = message;
  }
}

function renderMessage(role, text, sources = []) {
  const wrapper = document.createElement("div");
  wrapper.className = `message ${role}`;

  const avatar = document.createElement("div");
  avatar.className = "avatar";
  avatar.setAttribute("aria-hidden", "true");
  avatar.textContent = role === "user" ? "Вы" : "NiCo";

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
    <div class="avatar" aria-hidden="true">NiCo</div>
    <div class="bubble glass-bubble"><p class="bubble-text">Николя думает…</p></div>
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
  if (!doc?.id) return false;
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
    meta.textContent = [statusLabel, chunks, chars].filter(Boolean).join(" · ");

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

let lastHealth = null;

async function loadConfig() {
  const config = await api("/api/config");
  queryMode = config.query_mode || "hybrid";
}

async function refreshHealth() {
  try {
    lastHealth = await api("/api/health");
    setPipelineStatus(null, lastHealth);
  } catch {
    lastHealth = null;
    statusBadge.textContent = "Сервис недоступен";
    statusBadge.className = "badge glass-badge error";
  }
}

async function refreshStats() {
  try {
    const stats = await api("/api/knowledge/stats");
    updateKnowledgeCounts(stats.document_count ?? 0, stats.entities);
    renderDocumentList(stats.documents || []);
    labelsPreview.textContent = stats.labels_preview?.length
      ? `Примеры сущностей: ${stats.labels_preview.join(", ")}`
      : stats.processing_count
        ? `Обрабатывается документов: ${stats.processing_count}. Граф обновится по завершении.`
        : stats.document_count
          ? "Граф знаний строится по загруженным документам."
          : "Загрузите документы, чтобы построить базу.";
    setPipelineStatus(stats, lastHealth);
    updateStatsPolling(stats);
    return stats;
  } catch {
    updateKnowledgeCounts(null, null);
    renderDocumentList([]);
    labelsPreview.textContent = "";
    return null;
  }
}

async function pollTrackStatus(trackId) {
  if (!trackId) return;

  activeTrackPolls += 1;
  startStatsPolling();
  setPipelineStatus(null, lastHealth);

  const deadline = Date.now() + TRACK_POLL_TIMEOUT_MS;

  try {
    while (Date.now() < deadline) {
      const track = await api(`/api/knowledge/track/${encodeURIComponent(trackId)}`);
      await refreshStats();

      if (track.is_complete) {
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
        return;
      }

      await sleep(TRACK_POLL_MS);
    }

    showToast("Обработка ещё идёт — статус обновится автоматически.");
  } catch (error) {
    showToast(error.message, true);
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
  const message = chatInput.value.trim();
  if (!message) return;

  const session = getActiveSession();
  if (!session) return;

  appendMessage("user", message);
  chatInput.value = "";
  autoResizeTextarea();
  chatInput.disabled = true;
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
    chatInput.disabled = false;
    chatInput.focus();
  }
});

uploadBtn.addEventListener("click", () => {
  if (uploadBtn.disabled) return;
  fileInput.click();
});

fileInput.addEventListener("change", async () => {
  const files = Array.from(fileInput.files || []);
  if (!files.length) return;

  const progressMessage =
    files.length === 1
      ? `Загружаю «${files[0].name}»…`
      : `Загружаю ${files.length} документов…`;

  setUploadLoading(true, progressMessage);

  try {
    const formData = new FormData();
    for (const file of files) {
      formData.append("files", file);
    }

    const response = await fetch("/api/knowledge/upload/batch", {
      method: "POST",
      body: formData,
    });
    const data = await response.json();
    if (!response.ok) {
      const detail = Array.isArray(data.detail)
        ? data.detail.map((item) => item.msg || item).join("; ")
        : data.detail;
      throw new Error(detail || "Не удалось загрузить документы");
    }

    for (const item of data.results || []) {
      if (item.status === "failed" || !item.success) {
        showToast(`${item.filename}: ${item.message}`, true);
      }
    }

    if (data.message) {
      showToast(data.message, data.succeeded === 0);
    }

    if (data.track_id) {
      pollTrackStatus(data.track_id);
    } else {
      await refreshStats();
    }
  } catch (error) {
    showToast(error.message, true);
  } finally {
    setUploadLoading(false);
    fileInput.value = "";
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
