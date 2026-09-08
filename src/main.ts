import { api, BackendStatus, RecordingSession, Phrase, ModelDownloadProgress } from "./api.ts";
import { invoke } from "@tauri-apps/api/core";

// Window & Titlebar Controls
const appTitlebar = document.getElementById("app-titlebar") as HTMLElement | null;
const btnWinMinimize = document.getElementById("btn-win-minimize") as HTMLButtonElement | null;
const btnWinMaximize = document.getElementById("btn-win-maximize") as HTMLButtonElement | null;
const btnWinClose = document.getElementById("btn-win-close") as HTMLButtonElement | null;
const winMaximizeIcon = document.getElementById("win-maximize-icon") as HTMLElement | null;
const winRestoreIcon = document.getElementById("win-restore-icon") as HTMLElement | null;

// DOM Elements
const tabRecord = document.getElementById("tab-record") as HTMLButtonElement;
const tabHistory = document.getElementById("tab-history") as HTMLButtonElement;
const viewRecord = document.getElementById("view-record") as HTMLElement;
const viewHistory = document.getElementById("view-history") as HTMLElement;

// Status & Diagnostics (Sidebar)
const btnStatusIndicator = document.getElementById("btn-status-indicator") as HTMLButtonElement;
const statusCircleDot = document.getElementById("status-circle-dot") as HTMLElement;
const diagnosticsPopover = document.getElementById("diagnostics-popover") as HTMLElement;
const btnCloseDiag = document.getElementById("btn-close-diag") as HTMLButtonElement;
const diagBackendVal = document.getElementById("diag-backend-val") as HTMLElement;
const diagModelVal = document.getElementById("diag-model-val") as HTMLElement;
const diagDeviceVal = document.getElementById("diag-device-val") as HTMLElement;
const diagMicVal = document.getElementById("diag-mic-val") as HTMLElement;
const diagSystemVal = document.getElementById("diag-system-val") as HTMLElement;
const diagSessionVal = document.getElementById("diag-session-val") as HTMLElement;
const recordingsCountSpan = document.getElementById("recordings-count") as HTMLElement;
const btnOpenSettings = document.getElementById("btn-open-settings") as HTMLButtonElement;

// Recording Idle View & Model Gate
const idleScreen = document.getElementById("idle-screen") as HTMLElement;
const activeRecordingScreen = document.getElementById("active-recording-screen") as HTMLElement;
const modelGateCard = document.getElementById("model-gate-card") as HTMLElement;
const modelGateTag = document.getElementById("model-gate-tag") as HTMLElement;
const modelGateTitle = document.getElementById("model-gate-title") as HTMLElement;
const modelGateDesc = document.getElementById("model-gate-desc") as HTMLElement;
const btnDownloadModel = document.getElementById("btn-download-model") as HTMLButtonElement;
const modelProgressContainer = document.getElementById("model-progress-container") as HTMLElement;
const modelProgressBar = document.getElementById("model-progress-bar") as HTMLElement;
const modelProgressPct = document.getElementById("model-progress-pct") as HTMLElement;
const modelProgressBytes = document.getElementById("model-progress-bytes") as HTMLElement;
const modelProgressSpeed = document.getElementById("model-progress-speed") as HTMLElement;
const modelProgressEta = document.getElementById("model-progress-eta") as HTMLElement;

const btnStartMic = document.getElementById("btn-start-mic") as HTMLButtonElement;
const btnStartBoth = document.getElementById("btn-start-both") as HTMLButtonElement;
const btnStopRecording = document.getElementById("btn-stop-recording") as HTMLButtonElement;

// Active Recording Elements
const activeTimer = document.getElementById("active-timer") as HTMLElement;
const activeSessionModeBadge = document.getElementById("active-session-mode-badge") as HTMLElement;
const vadBar1 = document.getElementById("vad-bar-1") as HTMLElement;
const vadBar2 = document.getElementById("vad-bar-2") as HTMLElement;
const vadBar3 = document.getElementById("vad-bar-3") as HTMLElement;
const systemVadBar1 = document.getElementById("system-vad-bar-1") as HTMLElement;
const systemVadBar2 = document.getElementById("system-vad-bar-2") as HTMLElement;
const systemVadBar3 = document.getElementById("system-vad-bar-3") as HTMLElement;
const splashScreen = document.getElementById("splash-screen") as HTMLElement | null;
const splashStatusText = document.getElementById("splash-status-text") as HTMLElement | null;
const labelMic = document.getElementById("label-mic") as HTMLElement | null;
const labelSystem = document.getElementById("label-system") as HTMLElement | null;
const meterCardSystem = document.getElementById("meter-card-system") as HTMLElement;
const livePhrasesList = document.getElementById("live-phrases-list") as HTMLElement;
const emptyPhrasesHint = document.getElementById("empty-phrases-hint") as HTMLElement;

// History Elements
const btnRefreshHistory = document.getElementById("btn-refresh-history") as HTMLButtonElement;
const sessionsList = document.getElementById("sessions-list") as HTMLElement;
const emptyDetailHint = document.getElementById("empty-detail-hint") as HTMLElement;
const sessionLoadingState = document.getElementById("session-loading-state") as HTMLElement;
const sessionContentCard = document.getElementById("session-content-card") as HTMLElement;
const sessionTitleInput = document.getElementById("session-title-input") as HTMLInputElement;
const sessionDateDisplay = document.getElementById("session-date-display") as HTMLElement;
const sessionDurationDisplay = document.getElementById("session-duration-display") as HTMLElement;
const sessionModeDisplay = document.getElementById("session-mode-display") as HTMLElement;
const interactivePhrasesContainer = document.getElementById("interactive-phrases-container") as HTMLElement;
const btnRetranscribe = document.getElementById("btn-retranscribe") as HTMLButtonElement;
const finalTranscriptCard = document.getElementById("final-transcript-card") as HTMLElement;
const finalTranscriptText = document.getElementById("final-transcript-text") as HTMLTextAreaElement;
const transcriptHqStatus = document.getElementById("transcript-hq-status") as HTMLElement;
const appStatusMessage = document.getElementById("app-status-message") as HTMLElement;
const btnCopyTranscript = document.getElementById("btn-copy-transcript") as HTMLButtonElement;
const btnDownloadTxt = document.getElementById("btn-download-txt") as HTMLButtonElement;
const btnDownloadSrt = document.getElementById("btn-download-srt") as HTMLButtonElement;
const btnDeleteSession = document.getElementById("btn-delete-session") as HTMLButtonElement;

// HQ Retranscription Modal Elements
const hqProgressModal = document.getElementById("hq-progress-modal") as HTMLElement;
const hqModalProgressBar = document.getElementById("hq-modal-progress-bar") as HTMLElement;
const hqModalPct = document.getElementById("hq-modal-pct") as HTMLElement;
const hqModalChunks = document.getElementById("hq-modal-chunks") as HTMLElement;

// Audio Player Elements
const audioElement = document.getElementById("session-audio-element") as HTMLAudioElement;
const btnPlayerPlay = document.getElementById("btn-player-play") as HTMLButtonElement;
const playIcon = document.getElementById("play-icon") as HTMLElement;
const pauseIcon = document.getElementById("pause-icon") as HTMLElement;
const playerScrubSlider = document.getElementById("player-scrub-slider") as HTMLInputElement;
const playerCurrentTime = document.getElementById("player-current-time") as HTMLElement;
const playerTotalTime = document.getElementById("player-total-time") as HTMLElement;
const speedBtns = document.querySelectorAll(".speed-btn");

// Settings Modal Elements
const settingsModal = document.getElementById("settings-modal") as HTMLElement;
const btnCloseSettings = document.getElementById("btn-close-settings") as HTMLButtonElement;
const btnSaveSettings = document.getElementById("btn-save-settings") as HTMLButtonElement;
const btnRefreshDevices = document.getElementById("btn-refresh-devices") as HTMLButtonElement;
const btnPreloadModel = document.getElementById("btn-preload-model") as HTMLButtonElement;
const selectMicDevice = document.getElementById("select-mic-device") as HTMLSelectElement;
const selectSystemDevice = document.getElementById("select-system-device") as HTMLSelectElement;
const selectLanguage = document.getElementById("select-language") as HTMLSelectElement;
const settingsModelStatusText = document.getElementById("settings-model-status-text") as HTMLElement;

// State Variables
let selectedMicIndex: number = -1; // -1 means Windows Default
let selectedSystemIndex: number = -1; // -1 means Windows Default
let selectedLanguage = "en";
let currentModelStatus: string = "not_downloaded";
let latestStatus: BackendStatus | null = null;
let targetHqSessionId: string | null = null;

type RecordingOperation = "idle" | "starting" | "recording" | "stopping";
let recordingOperation: RecordingOperation = "idle";
let isRecording = false;
let backendOnline = false;
let sessionRequestGeneration = 0;
let historyRequestGeneration = 0;
let statusPollInFlight = false;
let statusMessageTimer: number | null = null;
let recordingTimerInterval: number | null = null;
let recordingStartTime = 0;
let lastRecordingError: string | null = null;
let retranscribeRequestInFlight = false;
let transcriptSavingSessionId: string | null = null;
let deletingSessionId: string | null = null;

let currentSessionId: string | null = null;
let activeViewingSession: RecordingSession | null = null;
let lastEditedTranscriptSource: "final_transcript" | "phrases" | null = null;
const livePhraseCards = new Map<string, HTMLElement>();

let activeSpeakingCardYou: HTMLElement | null = null;
let activeSpeakingCardThem: HTMLElement | null = null;
let speakingCardTimerYou: number | null = null;
let speakingCardTimerThem: number | null = null;
let splashDismissed = false;

function dismissSplashScreen() {
  if (splashDismissed || !splashScreen) return;
  splashDismissed = true;
  if (splashStatusText) splashStatusText.textContent = "Ready!";
  splashScreen.classList.add("fade-out");
  window.setTimeout(() => {
    splashScreen.classList.add("hidden");
  }, 450);
}

// Initialize Application
async function initApp() {
  loadSavedSettings();
  setupTabs();
  setupWindowControls();
  setupDiagnostics();
  setupHqModal();
  setupSettingsModal();
  setupAudioPlayer();
  setupHistoryActions();
  setupModelDownloadAction();

  // Connect WebSocket
  api.connectWebSocket(
    async () => {
      backendOnline = true;
      updateStatusCircleAndDiagnostics();
      await refreshStatus();
      await refreshDevices();
      await refreshHistory();
      dismissSplashScreen();
    },
    () => {
      backendOnline = false;
      updateStatusCircleAndDiagnostics();
      syncRecordControls();
    }
  );

  // Listen to WebSocket messages
  api.onWsMessage(handleWebSocketMessage);

  // Initial load attempt (only if backend is already running)
  await refreshStatus();
  if (backendOnline) {
    await refreshDevices();
    await refreshHistory();
    dismissSplashScreen();
  }

  // Safety fallback: ensure splash screen doesn't block UI indefinitely
  window.setTimeout(() => {
    if (!splashDismissed) {
      dismissSplashScreen();
    }
  }, 10000);

  // Poll as a fallback for missed WebSocket events. Never overlap slow polls.
  window.setInterval(async () => {
    if (statusPollInFlight) return;
    statusPollInFlight = true;
    try {
      const wasOnline = backendOnline;
      await refreshStatus();
      if (backendOnline) {
        if (!splashDismissed) {
          await refreshDevices();
          await refreshHistory();
          dismissSplashScreen();
        } else if (!wasOnline || selectMicDevice.children.length === 0) {
          await refreshDevices();
          await refreshHistory();
        }
      }
    } finally {
      statusPollInFlight = false;
    }
  }, 2000);
}

// WebSocket Message Handler
function handleWebSocketMessage(msg: any) {
  if (msg.type === "vad_meter") {
    handleVadMeter(msg);
  } else if (msg.type === "phrase_pending") {
    handlePhrasePending(msg.phrase);
  } else if (msg.type === "phrase_transcribed") {
    handlePhraseTranscribed(msg.phrase);
  } else if (msg.type === "model_status") {
    updateModelStatus(msg.status, msg.error);
  } else if (msg.type === "model_download_progress") {
    updateDownloadProgress(msg.progress);
  } else if (msg.type === "recording_started") {
    currentSessionId = msg.session?.id || currentSessionId;
    const modeLabel = msg.session?.mode === "mic_only" ? "Mic Only" : "Mic + System";
    if (!isRecording) enterRecording(modeLabel, false);
  } else if (msg.type === "recording_stopped") {
    if (!currentSessionId || msg.session_id === currentSessionId) leaveRecording();
  } else if (msg.type === "recording_error") {
    if (!msg.session_id || msg.session_id === currentSessionId) showRecordingError(msg.error);
  } else if (msg.type === "hq_pass_progress") {
    handleHqPassProgress(msg.progress);
  } else if (msg.type === "hq_pass_completed") {
    handleHqPassCompleted(msg.result);
  } else if (msg.type === "hq_pass_failed") {
    handleHqPassFailed(msg.result);
  }
}

function updateVadBars(bars: HTMLElement[], rms: number, isSpeaking: boolean) {
  if (isSpeaking) {
    // Keep the same three-bar visual treatment for microphone and computer audio.
    const h1 = Math.min(100, Math.max(50, Math.round(55 + rms * 100)));
    const h2 = Math.min(100, Math.max(80, Math.round(80 + rms * 60)));
    const h3 = Math.min(100, Math.max(60, Math.round(65 + rms * 80)));
    [h1, h2, h3].forEach((height, index) => {
      bars[index].style.height = `${height}%`;
      bars[index].classList.add("speaking");
    });
  } else {
    bars.forEach((bar) => {
      bar.style.height = "4px";
      bar.classList.remove("speaking");
    });
  }
}

function handleVadMeter(payload: any) {
  if (!isRecording) return;

  const micRms = Math.min(1, Math.max(0, payload.mic?.rms || 0));
  const systemRms = Math.min(1, Math.max(0, payload.system?.rms || 0));
  const isMicSpeaking = !!payload.mic?.is_speaking;
  const isSystemSpeaking = !!payload.system?.is_speaking;

  updateVadBars([vadBar1, vadBar2, vadBar3], micRms, isMicSpeaking);
  if (labelMic) labelMic.classList.toggle("speaking", isMicSpeaking);

  updateVadBars([systemVadBar1, systemVadBar2, systemVadBar3], systemRms, isSystemSpeaking);
  if (labelSystem) labelSystem.classList.toggle("speaking", isSystemSpeaking);

  manageSpeakingSkeleton("You", isMicSpeaking);
  manageSpeakingSkeleton("Them", isSystemSpeaking);
}

function manageSpeakingSkeleton(speaker: "You" | "Them", isSpeaking: boolean) {
  if (speaker === "You") {
    if (isSpeaking) {
      if (speakingCardTimerYou) {
        window.clearTimeout(speakingCardTimerYou);
        speakingCardTimerYou = null;
      }
      if (!activeSpeakingCardYou) {
        activeSpeakingCardYou = createSpeakingSkeletonCard("You");
        emptyPhrasesHint.classList.add("hidden");
        livePhrasesList.prepend(activeSpeakingCardYou);
        scrollLivePhrasesToTop();
      }
    } else if (activeSpeakingCardYou && !speakingCardTimerYou) {
      speakingCardTimerYou = window.setTimeout(() => {
        if (activeSpeakingCardYou) {
          activeSpeakingCardYou.remove();
          activeSpeakingCardYou = null;
          if (livePhraseCards.size === 0 && !activeSpeakingCardThem) {
            emptyPhrasesHint.classList.remove("hidden");
          }
        }
        speakingCardTimerYou = null;
      }, 2500);
    }
  } else {
    if (isSpeaking) {
      if (speakingCardTimerThem) {
        window.clearTimeout(speakingCardTimerThem);
        speakingCardTimerThem = null;
      }
      if (!activeSpeakingCardThem) {
        activeSpeakingCardThem = createSpeakingSkeletonCard("Them");
        emptyPhrasesHint.classList.add("hidden");
        livePhrasesList.prepend(activeSpeakingCardThem);
        scrollLivePhrasesToTop();
      }
    } else if (activeSpeakingCardThem && !speakingCardTimerThem) {
      speakingCardTimerThem = window.setTimeout(() => {
        if (activeSpeakingCardThem) {
          activeSpeakingCardThem.remove();
          activeSpeakingCardThem = null;
          if (livePhraseCards.size === 0 && !activeSpeakingCardYou) {
            emptyPhrasesHint.classList.remove("hidden");
          }
        }
        speakingCardTimerThem = null;
      }, 2500);
    }
  }
}

function createSpeakingSkeletonCard(speaker: "You" | "Them"): HTMLElement {
  const card = document.createElement("div");
  card.className = "phrase-card phrase-card-skeleton live-speaking-skeleton";
  card.dataset.speaker = speaker;
  card.setAttribute("aria-busy", "true");
  card.setAttribute("aria-label", `${speaker} speech detected`);
  card.innerHTML = `
    <span class="phrase-speaker">${speaker}:</span>
    <span class="phrase-skeleton-text" aria-hidden="true">
      <span class="phrase-skeleton-line" style="width: 78%"></span>
      <span class="phrase-skeleton-line" style="width: 48%"></span>
    </span>
  `;
  return card;
}

function speakerLabelForPhrase(phrase: Phrase): string {
  const speaker = (phrase.speaker || "").trim().toLowerCase();
  return ["system audio", "system / remote", "system", "remote", "them"].includes(speaker) ? "Them" : "You";
}

function scrollLivePhrasesToTop() {
  window.requestAnimationFrame(() => {
    livePhrasesList.scrollTo({ top: 0, behavior: "smooth" });
  });
}

function handlePhrasePending(phrase: Phrase) {
  if (!phrase || !phrase.phrase_id || (phrase.session_id && phrase.session_id !== currentSessionId)) return;
  emptyPhrasesHint.classList.add("hidden");

  // WebSocket reconnects can deliver an event more than once; keep one row per phrase.
  if (livePhraseCards.has(phrase.phrase_id)) return;

  const speaker = speakerLabelForPhrase(phrase);
  let phraseCard: HTMLElement;

  if (speaker === "You" && activeSpeakingCardYou) {
    phraseCard = activeSpeakingCardYou;
    activeSpeakingCardYou = null;
    if (speakingCardTimerYou) {
      window.clearTimeout(speakingCardTimerYou);
      speakingCardTimerYou = null;
    }
    phraseCard.classList.remove("live-speaking-skeleton");
  } else if (speaker === "Them" && activeSpeakingCardThem) {
    phraseCard = activeSpeakingCardThem;
    activeSpeakingCardThem = null;
    if (speakingCardTimerThem) {
      window.clearTimeout(speakingCardTimerThem);
      speakingCardTimerThem = null;
    }
    phraseCard.classList.remove("live-speaking-skeleton");
  } else {
    phraseCard = document.createElement("div");
    phraseCard.className = "phrase-card phrase-card-skeleton";
    livePhrasesList.prepend(phraseCard);
  }

  phraseCard.dataset.phraseId = phrase.phrase_id;
  phraseCard.setAttribute("aria-busy", "true");
  phraseCard.setAttribute("aria-label", `${speaker} phrase is being transcribed`);

  const duration = Math.max(0.5, Number(phrase.duration) || 0.5);
  const lineCount = duration >= 3.5 ? 3 : 2;
  const widths = lineCount === 3 ? ["92%", "76%", "48%"] : ["88%", "58%"];
  const skeletonLines = widths
    .map((width) => `<span class="phrase-skeleton-line" style="width: ${width}"></span>`)
    .join("");
  phraseCard.innerHTML = `
    <span class="phrase-speaker">${escapeHtml(speaker)}:</span>
    <span class="phrase-skeleton-text" aria-hidden="true">${skeletonLines}</span>
  `;

  livePhraseCards.set(phrase.phrase_id, phraseCard);
  scrollLivePhrasesToTop();
}

function handlePhraseTranscribed(phrase: Phrase) {
  if (!phrase || !phrase.phrase_id || (phrase.session_id && phrase.session_id !== currentSessionId)) return;
  emptyPhrasesHint.classList.add("hidden");

  if (phrase.error) showMessage(`A speech segment could not be transcribed: ${phrase.error}`, true);

  const speaker = speakerLabelForPhrase(phrase);
  let phraseCard = livePhraseCards.get(phrase.phrase_id);

  if (!phraseCard) {
    if (speaker === "You" && activeSpeakingCardYou) {
      phraseCard = activeSpeakingCardYou;
      activeSpeakingCardYou = null;
      if (speakingCardTimerYou) {
        window.clearTimeout(speakingCardTimerYou);
        speakingCardTimerYou = null;
      }
      phraseCard.classList.remove("live-speaking-skeleton");
    } else if (speaker === "Them" && activeSpeakingCardThem) {
      phraseCard = activeSpeakingCardThem;
      activeSpeakingCardThem = null;
      if (speakingCardTimerThem) {
        window.clearTimeout(speakingCardTimerThem);
        speakingCardTimerThem = null;
      }
      phraseCard.classList.remove("live-speaking-skeleton");
    } else {
      phraseCard = document.createElement("div");
      livePhrasesList.prepend(phraseCard);
    }
  }

  const wasPending = phraseCard.classList.contains("phrase-card-skeleton");
  phraseCard.className = `phrase-card${phrase.error ? " error" : ""}`;
  phraseCard.dataset.phraseId = phrase.phrase_id;
  phraseCard.removeAttribute("aria-busy");
  phraseCard.setAttribute("aria-label", `${speaker}: ${phrase.text || "This segment could not be transcribed."}`);

  phraseCard.innerHTML = `
    <span class="phrase-speaker">${escapeHtml(speaker)}:</span>
    <span class="phrase-text${wasPending ? " phrase-text-revealed" : ""}">${escapeHtml(phrase.text || "This segment could not be transcribed.")}</span>
  `;

  if (!phraseCard.parentElement) livePhrasesList.prepend(phraseCard);
  livePhraseCards.set(phrase.phrase_id, phraseCard);
  scrollLivePhrasesToTop();
}

function showRecordingError(error: string) {
  if (!error || error === lastRecordingError) return;
  lastRecordingError = error;
  btnStopRecording.textContent = "Stop & save what was captured";
  showMessage(`Audio capture problem: ${error}. Check Settings, then stop this recording.`, true);
}

function handleHqPassProgress(payload: any) {
  if (!payload) return;
  if (targetHqSessionId && payload.session_id && payload.session_id !== targetHqSessionId) return;
  const pct = Math.max(0, Math.min(100, Math.round(payload.percent || 0)));
  hqModalProgressBar.style.width = `${pct}%`;
  hqModalPct.textContent = `${pct}%`;
  hqModalChunks.textContent = `Processing chunk ${payload.processed_chunks || 0} of ${payload.total_chunks || 0}…`;
}

function handleHqPassCompleted(result: any) {
  const finishedSessionId = result?.session_id;
  if (targetHqSessionId && targetHqSessionId === finishedSessionId) {
    hideHqModal();
    targetHqSessionId = null;
    switchTab("history");
    if (finishedSessionId) void selectSession(finishedSessionId);
    showMessage("High-quality transcription complete.");
  } else if (activeViewingSession?.id === finishedSessionId) {
    hideHqModal();
    void selectSession(finishedSessionId);
    showMessage("High-quality transcription complete.");
  }
  void refreshHistory();
}

function handleHqPassFailed(result: any) {
  const failedSessionId = result?.session_id;
  if (targetHqSessionId && targetHqSessionId === failedSessionId) {
    hideHqModal();
    targetHqSessionId = null;
    switchTab("history");
    if (failedSessionId) void selectSession(failedSessionId);
  } else if (activeViewingSession?.id === failedSessionId) {
    hideHqModal();
    void selectSession(failedSessionId);
  }
  showMessage(`High-quality transcription failed: ${result?.error || "Unknown error"}`, true);
  void refreshHistory();
}

function showMessage(message: string, isError = false) {
  if (statusMessageTimer) window.clearTimeout(statusMessageTimer);
  appStatusMessage.textContent = message;
  appStatusMessage.classList.toggle("error", isError);
  appStatusMessage.classList.remove("hidden");
  statusMessageTimer = window.setTimeout(() => appStatusMessage.classList.add("hidden"), isError ? 9000 : 4500);
}

function modelCanRecord(): boolean {
  return (
    currentModelStatus === "ready" ||
    currentModelStatus === "transcribing" ||
    currentModelStatus === "not_loaded" ||
    currentModelStatus === "loading"
  );
}

function syncRecordControls() {
  const canStart = backendOnline && modelCanRecord() && recordingOperation === "idle" && !isRecording;
  btnStartMic.disabled = !canStart;
  btnStartBoth.disabled = !canStart;
  btnStartMic.title = canStart
    ? "Record microphone only"
    : !backendOnline
      ? "Backend is offline"
      : !modelCanRecord()
        ? "Download the transcription model before recording"
        : "A recording action is already in progress";
  btnStartBoth.title = canStart ? "Record microphone and system audio" : btnStartMic.title;
  if (activeViewingSession) renderSessionProcessingState(activeViewingSession);
}

function updateStatusCircleAndDiagnostics() {
  if (!backendOnline) {
    statusCircleDot.className = "status-circle-dot status-offline";
    btnStatusIndicator.title = "Backend: Offline • Click for diagnostics";
    diagBackendVal.textContent = "Offline (Disconnected)";
    diagBackendVal.style.color = "var(--danger)";
    diagModelVal.textContent = "Unknown (Backend Offline)";
    diagModelVal.style.color = "var(--text-muted)";
    diagDeviceVal.textContent = "Unknown";
    diagDeviceVal.style.color = "var(--text-muted)";
    diagMicVal.textContent = "Unavailable";
    diagMicVal.style.color = "var(--text-muted)";
    diagSystemVal.textContent = "Unavailable";
    diagSystemVal.style.color = "var(--text-muted)";
    diagSessionVal.textContent = "Offline";
    diagSessionVal.style.color = "var(--text-muted)";
    return;
  }

  diagBackendVal.textContent = "Online";
  diagBackendVal.style.color = "var(--text-primary)";
  diagModelVal.style.color = "var(--text-primary)";
  diagDeviceVal.style.color = "var(--text-primary)";
  diagMicVal.style.color = "var(--text-primary)";
  diagSystemVal.style.color = "var(--text-primary)";

  // Compute device
  if (latestStatus?.device === "cuda") {
    diagDeviceVal.textContent = `CUDA (${latestStatus.cuda_device_name || "GPU"})`;
  } else if (latestStatus?.device === "cpu") {
    diagDeviceVal.textContent = "CPU";
  } else {
    diagDeviceVal.textContent = latestStatus?.device || "Detecting…";
  }

  // Model status string for diagnostics
  let modelDesc = "Checking…";
  if (currentModelStatus === "ready") {
    modelDesc = "Ready in memory (GPU/CPU)";
  } else if (currentModelStatus === "not_loaded") {
    modelDesc = "Ready on disk (Loads on record)";
  } else if (currentModelStatus === "loading") {
    modelDesc = "Loading weights…";
  } else if (currentModelStatus === "downloading") {
    const pct = latestStatus?.download_progress?.percent;
    modelDesc = pct !== undefined ? `Downloading (${Math.round(pct)}%)` : "Downloading…";
  } else if (currentModelStatus === "transcribing") {
    modelDesc = "Transcribing audio";
  } else if (currentModelStatus === "error") {
    modelDesc = `Error: ${latestStatus?.model_error || "Failed"}`;
  } else if (currentModelStatus === "not_downloaded") {
    modelDesc = "Not downloaded";
  } else {
    modelDesc = currentModelStatus;
  }
  diagModelVal.textContent = modelDesc;

  // Active session
  if (isRecording) {
    diagSessionVal.textContent = `Recording (${currentSessionId || "Active"})`;
    diagSessionVal.style.color = "var(--danger)";
  } else if (activeViewingSession) {
    diagSessionVal.textContent = `Viewing: ${activeViewingSession.title || activeViewingSession.id}`;
    diagSessionVal.style.color = "var(--text-primary)";
  } else {
    diagSessionVal.textContent = "Idle";
    diagSessionVal.style.color = "var(--text-muted)";
  }

  // Selected audio devices
  const currentMicText = selectMicDevice.selectedIndex >= 0 && selectMicDevice.options[selectMicDevice.selectedIndex]
    ? selectMicDevice.options[selectMicDevice.selectedIndex].textContent || "Default"
    : "Default";
  diagMicVal.textContent = currentMicText;

  const currentSysText = selectSystemDevice.selectedIndex >= 0 && selectSystemDevice.options[selectSystemDevice.selectedIndex]
    ? selectSystemDevice.options[selectSystemDevice.selectedIndex].textContent || "Default"
    : "Default";
  diagSystemVal.textContent = currentSysText;

  // Single Status Circle Dot Color:
  // Green: ready or not_loaded (everything fine, ready to record)
  // Yellow: loading, downloading, transcribing (in progress / busy)
  // Red: error or not_downloaded (needs user attention)
  if (currentModelStatus === "ready" || currentModelStatus === "not_loaded") {
    statusCircleDot.className = "status-circle-dot status-good";
    btnStatusIndicator.title = `System: Ready (${currentModelStatus === "ready" ? "In memory" : "On-demand"}) • Click for diagnostics`;
  } else if (
    currentModelStatus === "loading" ||
    currentModelStatus === "downloading" ||
    currentModelStatus === "transcribing"
  ) {
    statusCircleDot.className = "status-circle-dot status-warn";
    btnStatusIndicator.title = `System: Busy (${currentModelStatus}) • Click for diagnostics`;
  } else {
    statusCircleDot.className = "status-circle-dot status-bad";
    btnStatusIndicator.title = `System: Attention needed (${currentModelStatus === "error" ? "Model error" : "Model not downloaded"}) • Click for diagnostics`;
  }
}

function setupDiagnostics() {
  btnStatusIndicator.addEventListener("click", (e) => {
    e.stopPropagation();
    diagnosticsPopover.classList.toggle("hidden");
    updateStatusCircleAndDiagnostics();
  });

  btnCloseDiag.addEventListener("click", (e) => {
    e.stopPropagation();
    diagnosticsPopover.classList.add("hidden");
  });

  document.addEventListener("click", (e) => {
    if (!diagnosticsPopover.classList.contains("hidden")) {
      const target = e.target as HTMLElement;
      if (!diagnosticsPopover.contains(target) && !btnStatusIndicator.contains(target)) {
        diagnosticsPopover.classList.add("hidden");
      }
    }
  });

  document.addEventListener("keydown", (e) => {
    if (e.key === "Escape" && !diagnosticsPopover.classList.contains("hidden")) {
      diagnosticsPopover.classList.add("hidden");
    }
  });
}

function setupHqModal() {
  hqProgressModal.addEventListener("click", (e) => {
    if (e.target === hqProgressModal) {
      hideHqModal();
    }
  });
}

function showHqModal(title = "High-Quality Transcription") {
  const titleEl = document.getElementById("hq-modal-title");
  if (titleEl) titleEl.textContent = title;
  hqModalProgressBar.style.width = "0%";
  hqModalPct.textContent = "0%";
  hqModalChunks.textContent = "Preparing audio segments…";
  hqProgressModal.classList.remove("hidden");
}

function hideHqModal() {
  hqProgressModal.classList.add("hidden");
}

function updateDownloadProgress(progress: ModelDownloadProgress) {
  if (!progress) return;
  currentModelStatus = "downloading";
  const percent = Math.max(0, Math.min(100, Number(progress.percent) || 0));

  modelGateCard.classList.remove("hidden");
  modelGateTag.textContent = "DOWNLOADING";
  modelGateTitle.textContent = "Cohere Transcribe";
  modelGateDesc.textContent = `Downloading ${progress.filename}...`;

  btnDownloadModel.textContent = "Downloading…";
  btnDownloadModel.disabled = true;
  btnDownloadModel.setAttribute("aria-busy", "true");

  modelProgressContainer.classList.remove("hidden");
  modelProgressContainer.setAttribute("aria-valuenow", String(percent));
  modelProgressBar.style.width = `${percent}%`;
  modelProgressPct.textContent = `${percent.toFixed(percent % 1 ? 1 : 0)}%`;
  modelProgressBytes.textContent = progress.total_mb > 0
    ? `${progress.downloaded_mb.toFixed(1)} / ${progress.total_mb.toFixed(1)} MB`
    : `${progress.downloaded_mb.toFixed(1)} MB downloaded`;
  modelProgressSpeed.textContent = progress.speed_mb_s && progress.speed_mb_s > 0 ? `${progress.speed_mb_s.toFixed(1)} MB/s` : "Calculating speed…";
  if (progress.eta_s && progress.eta_s > 0) {
    const mins = Math.floor(progress.eta_s / 60);
    const secs = Math.floor(progress.eta_s % 60);
    modelProgressEta.textContent = `About ${mins ? `${mins}m ` : ""}${secs}s left`;
  } else {
    modelProgressEta.textContent = "";
  }

  settingsModelStatusText.textContent = `Status: Downloading ${progress.filename} (${Math.round(percent)}%)`;
  updateStatusCircleAndDiagnostics();
  syncRecordControls();
}

function updateModelStatus(status: string, error?: string) {
  currentModelStatus = status;
  if (status !== "downloading") btnDownloadModel.removeAttribute("aria-busy");

  if (status === "ready" || status === "transcribing") {
    modelGateCard.classList.add("hidden");
    modelProgressContainer.classList.add("hidden");

    settingsModelStatusText.textContent = status === "transcribing" ? "Status: Transcribing" : "Status: Ready in memory";
    btnPreloadModel.textContent = "Model Loaded";
    btnPreloadModel.disabled = true;
  } else if (status === "not_loaded") {
    modelGateCard.classList.add("hidden");
    modelProgressContainer.classList.add("hidden");

    settingsModelStatusText.textContent = "Status: Downloaded (Loads when recording starts)";
    btnPreloadModel.textContent = "Preload Model";
    btnPreloadModel.disabled = false;
  } else if (status === "loading") {
    modelGateCard.classList.remove("hidden");
    modelGateTag.textContent = "LOADING";
    modelGateTitle.textContent = "Cohere Transcribe";
    modelGateDesc.textContent = "Loading weights into memory (~5 seconds)...";

    btnDownloadModel.textContent = "Loading...";
    btnDownloadModel.disabled = true;
    modelProgressContainer.classList.add("hidden");

    settingsModelStatusText.textContent = "Status: Loading weights into memory...";
    btnPreloadModel.textContent = "Loading...";
    btnPreloadModel.disabled = true;
  } else if (status === "downloading") {
    modelGateCard.classList.remove("hidden");
    modelGateTag.textContent = "DOWNLOADING";
    btnDownloadModel.textContent = "Downloading...";
    btnDownloadModel.disabled = true;
    modelProgressContainer.classList.remove("hidden");

    settingsModelStatusText.textContent = "Status: Downloading...";
    btnPreloadModel.textContent = "Downloading...";
    btnPreloadModel.disabled = true;
  } else if (status === "error") {
    modelGateCard.classList.remove("hidden");
    modelGateTag.textContent = "ERROR";
    modelGateTitle.textContent = "Model Error";
    modelGateDesc.textContent = error || "Download or initialization failed.";

    btnDownloadModel.textContent = "Retry Download";
    btnDownloadModel.disabled = false;
    modelProgressContainer.classList.add("hidden");

    settingsModelStatusText.textContent = `Status: Error (${error || "Failed"})`;
    btnPreloadModel.textContent = "Retry Download";
    btnPreloadModel.disabled = false;
  } else {
    // not_downloaded
    modelGateCard.classList.remove("hidden");
    modelGateTag.textContent = "REQUIRED";
    modelGateTitle.textContent = "Cohere Transcribe";
    modelGateDesc.textContent = "Download model weights (3.8 GB) before starting.";

    btnDownloadModel.textContent = "Download Model";
    btnDownloadModel.disabled = false;
    modelProgressContainer.classList.add("hidden");

    settingsModelStatusText.textContent = "Status: Not Downloaded";
    btnPreloadModel.textContent = "Download Model";
    btnPreloadModel.disabled = false;
  }
  updateStatusCircleAndDiagnostics();
  syncRecordControls();
  if (activeViewingSession) renderSessionProcessingState(activeViewingSession);
}

async function refreshStatus() {
  try {
    const status: BackendStatus = await api.getStatus();
    latestStatus = status;
    backendOnline = true;
    if (status.download_progress) updateDownloadProgress(status.download_progress);
    else updateModelStatus(status.model_status, status.model_error);

    if (status.is_recording && !isRecording) {
      currentSessionId = status.active_session_id || null;
      const modeLabel = status.active_recording_mode === "mic_only" ? "Mic Only" : "Mic + System";
      enterRecording(modeLabel, false, status.recording_duration || 0);
    } else if (status.is_recording && isRecording && recordingOperation !== "stopping" && typeof status.recording_duration === "number") {
      recordingStartTime = Date.now() - status.recording_duration * 1000;
    } else if (!status.is_recording && isRecording && recordingOperation !== "stopping") {
      leaveRecording();
    }
    if (status.recording_error) showRecordingError(status.recording_error);
    updateStatusCircleAndDiagnostics();
    syncRecordControls();
  } catch {
    backendOnline = false;
    updateStatusCircleAndDiagnostics();
    syncRecordControls();
  }
}

async function refreshDevices() {
  try {
    const data = await api.getDevices();

    // 1. Populate Mic Selector
    const currentMic = selectedMicIndex ?? -1;
    selectMicDevice.innerHTML = "";

    const defaultMicOpt = document.createElement("option");
    defaultMicOpt.value = "-1";
    defaultMicOpt.textContent = `Default: ${data.default_mic?.name || "Windows Default"}`;
    if (currentMic === -1) {
      defaultMicOpt.selected = true;
    }
    selectMicDevice.appendChild(defaultMicOpt);

    data.microphones.forEach((m) => {
      const opt = document.createElement("option");
      opt.value = String(m.index);
      const isWinDefault = data.default_mic && m.index === data.default_mic.index;
      opt.textContent = `${m.name} ${isWinDefault ? "(Default Device)" : ""}`;
      if (currentMic === m.index) {
        opt.selected = true;
      }
      selectMicDevice.appendChild(opt);
    });

    // 2. Populate System Audio Selector
    const currentSys = selectedSystemIndex ?? -1;
    selectSystemDevice.innerHTML = "";

    const defaultSysOpt = document.createElement("option");
    defaultSysOpt.value = "-1";
    defaultSysOpt.textContent = `Default: ${data.default_system?.name || "Windows Default Loopback"}`;
    if (currentSys === -1) {
      defaultSysOpt.selected = true;
    }
    selectSystemDevice.appendChild(defaultSysOpt);

    data.system_devices.forEach((s) => {
      const opt = document.createElement("option");
      opt.value = String(s.index);
      const isWinDefault = data.default_system && s.index === data.default_system.index;
      opt.textContent = `${s.name} ${isWinDefault ? "(Default Device)" : ""}`;
      if (currentSys === s.index) {
        opt.selected = true;
      }
      selectSystemDevice.appendChild(opt);
    });
    updateStatusCircleAndDiagnostics();
  } catch (e) {
    console.error("Failed to query audio devices:", e);
  }
}

// Download Trigger Action
function setupModelDownloadAction() {
  const triggerDownload = async () => {
    if (currentModelStatus === "downloading" || currentModelStatus === "loading") return;
    currentModelStatus = "downloading";
    syncRecordControls();
    btnDownloadModel.textContent = "Starting…";
    btnDownloadModel.disabled = true;
    btnDownloadModel.setAttribute("aria-busy", "true");
    modelProgressContainer.classList.remove("hidden");
    modelProgressBar.style.width = "0%";
    modelProgressPct.textContent = "0%";
    modelProgressBytes.textContent = "Connecting to Hugging Face…";
    modelProgressSpeed.textContent = "";
    modelProgressEta.textContent = "";

    try {
      await api.downloadModel();
      await refreshStatus();
    } catch (err: any) {
      updateModelStatus("error", err.message);
      showMessage(`Model download could not start: ${err.message}`, true);
    }
  };

  btnDownloadModel.addEventListener("click", triggerDownload);
}

// Recording Controls
btnStartMic.addEventListener("click", () => startRecording("mic_only"));
btnStartBoth.addEventListener("click", () => startRecording("mic_and_system"));
btnStopRecording.addEventListener("click", stopRecording);

async function startRecording(mode: "mic_only" | "mic_and_system") {
  if (recordingOperation !== "idle") return;
  if (!backendOnline || !modelCanRecord()) {
    showMessage(!backendOnline ? "Backend is offline. Reconnect before recording." : "Download the transcription model before starting a recording.", true);
    return;
  }
  recordingOperation = "starting";
  btnStartMic.setAttribute("aria-busy", "true");
  btnStartBoth.setAttribute("aria-busy", "true");
  syncRecordControls();
  showMessage("Starting audio capture…");
  try {
    const res = await api.startRecording({
      mode,
      language: selectedLanguage,
      mic_device_index: selectedMicIndex >= 0 ? selectedMicIndex : undefined,
      system_device_index: selectedSystemIndex >= 0 ? selectedSystemIndex : undefined,
    });
    currentSessionId = res.session_id;
    enterRecording(mode === "mic_only" ? "Mic Only" : "Mic + System", true);
    showMessage("Recording started.");
  } catch (err: any) {
    recordingOperation = "idle";
    syncRecordControls();
    showMessage(`Could not start recording: ${err.message}`, true);
  } finally {
    btnStartMic.removeAttribute("aria-busy");
    btnStartBoth.removeAttribute("aria-busy");
  }
}

function enterRecording(modeLabel: string, resetPhrases: boolean, elapsedSeconds = 0) {
  isRecording = true;
  recordingOperation = "recording";
  lastRecordingError = null;
  idleScreen.classList.add("hidden");
  activeRecordingScreen.classList.remove("hidden");
  activeSessionModeBadge.textContent = modeLabel;
  meterCardSystem.classList.toggle("hidden", modeLabel === "Mic Only");
  btnStopRecording.disabled = false;
  btnStopRecording.textContent = "Stop Recording";
  if (resetPhrases) {
    livePhraseCards.clear();
    livePhrasesList.innerHTML = "";
    emptyPhrasesHint.classList.remove("hidden");
    livePhrasesList.appendChild(emptyPhrasesHint);
    if (activeSpeakingCardYou) { activeSpeakingCardYou.remove(); activeSpeakingCardYou = null; }
    if (activeSpeakingCardThem) { activeSpeakingCardThem.remove(); activeSpeakingCardThem = null; }
    if (speakingCardTimerYou) { clearTimeout(speakingCardTimerYou); speakingCardTimerYou = null; }
    if (speakingCardTimerThem) { clearTimeout(speakingCardTimerThem); speakingCardTimerThem = null; }
  }
  recordingStartTime = Date.now() - Math.max(0, elapsedSeconds) * 1000;
  updateTimerDisplay();
  if (recordingTimerInterval) clearInterval(recordingTimerInterval);
  recordingTimerInterval = window.setInterval(updateTimerDisplay, 500);
  updateStatusCircleAndDiagnostics();
  syncRecordControls();
}

function leaveRecording() {
  isRecording = false;
  recordingOperation = "idle";
  lastRecordingError = null;
  if (recordingTimerInterval) clearInterval(recordingTimerInterval);
  recordingTimerInterval = null;
  activeRecordingScreen.classList.add("hidden");
  idleScreen.classList.remove("hidden");
  livePhraseCards.clear();
  if (activeSpeakingCardYou) { activeSpeakingCardYou.remove(); activeSpeakingCardYou = null; }
  if (activeSpeakingCardThem) { activeSpeakingCardThem.remove(); activeSpeakingCardThem = null; }
  if (speakingCardTimerYou) { clearTimeout(speakingCardTimerYou); speakingCardTimerYou = null; }
  if (speakingCardTimerThem) { clearTimeout(speakingCardTimerThem); speakingCardTimerThem = null; }
  if (labelMic) labelMic.classList.remove("speaking");
  if (labelSystem) labelSystem.classList.remove("speaking");
  updateVadBars([vadBar1, vadBar2, vadBar3], 0, false);
  updateVadBars([systemVadBar1, systemVadBar2, systemVadBar3], 0, false);
  btnStopRecording.disabled = false;
  btnStopRecording.textContent = "Stop Recording";
  btnStopRecording.removeAttribute("aria-busy");
  updateStatusCircleAndDiagnostics();
  syncRecordControls();
}

function updateTimerDisplay() {
  const elapsed = Math.floor((Date.now() - recordingStartTime) / 1000);
  const hrs = Math.floor(elapsed / 3600);
  const mins = Math.floor((elapsed % 3600) / 60);
  const secs = elapsed % 60;
  activeTimer.textContent = `${pad(hrs)}:${pad(mins)}:${pad(secs)}`;
}

async function stopRecording() {
  if (!isRecording || recordingOperation !== "recording") return;
  recordingOperation = "stopping";
  btnStopRecording.disabled = true;
  btnStopRecording.setAttribute("aria-busy", "true");
  btnStopRecording.textContent = "Finalizing audio…";
  showMessage("Saving audio and starting the high-quality transcript…");
  const stoppedSessionId = currentSessionId;
  targetHqSessionId = stoppedSessionId;

  showHqModal("High-Quality Transcription");

  try {
    await api.stopRecording();
    leaveRecording();
    await refreshHistory();
  } catch (error: any) {
    hideHqModal();
    targetHqSessionId = null;
    try {
      const status = await api.getStatus();
      if (!status.is_recording) {
        leaveRecording();
        await refreshHistory();
        if (stoppedSessionId) {
          switchTab("history");
          await selectSession(stoppedSessionId);
        }
        showMessage("Recording stopped. The confirmation response was interrupted, but the saved session was recovered.");
        return;
      }
    } catch {
      // Preserve the recording UI when the backend cannot confirm either state.
    }
    recordingOperation = "recording";
    btnStopRecording.disabled = false;
    btnStopRecording.removeAttribute("aria-busy");
    btnStopRecording.textContent = "Retry Stop";
    showMessage(`Could not stop recording: ${error.message}. Recording may still be active; retry Stop.`, true);
  }
}

// History & Playback
async function refreshHistory() {
  const requestGeneration = ++historyRequestGeneration;
  sessionsList.setAttribute("aria-busy", "true");
  if (!sessionsList.children.length) sessionsList.innerHTML = `<div class="list-state">Loading recordings…</div>`;
  try {
    const sessions = await api.listRecordings();
    if (requestGeneration !== historyRequestGeneration) return;
    recordingsCountSpan.textContent = String(sessions.length);

    // If there was an active "History unavailable" message, clear it now that recordings loaded
    if (appStatusMessage.textContent?.startsWith("History unavailable")) {
      appStatusMessage.classList.add("hidden");
      if (statusMessageTimer) {
        window.clearTimeout(statusMessageTimer);
        statusMessageTimer = null;
      }
    }

    sessionsList.innerHTML = "";
    if (sessions.length === 0) {
      sessionsList.innerHTML = `<div style="padding: 20px; color: var(--text-muted); text-align: center; font-size: 14px;">No recordings yet.</div>`;
      return;
    }

    sessions.forEach((s) => {
      const item = document.createElement("button");
      item.type = "button";
      item.className = `session-item ${activeViewingSession?.id === s.id ? "active" : ""}`;
      item.dataset.id = s.id;

      const statusLabel = s.status === "processing_hq"
        ? "Transcribing"
        : s.status === "hq_error"
          ? "Needs retry"
          : s.status === "interrupted"
            ? "Interrupted"
            : "";
      const statusClass = s.status === "hq_error" ? "error" : s.status === "interrupted" ? "warn" : "";
      item.innerHTML = `
        <div class="session-item-title">${escapeHtml(s.title || "Untitled Meeting")}</div>
        <div class="session-item-sub">
          <span>${formatDate(s.created_at)}</span>
          <span>${statusLabel ? `<span class="session-item-status ${statusClass}">${statusLabel}</span>` : formatSeconds(s.duration || 0)}</span>
        </div>
      `;

      item.addEventListener("click", () => selectSession(s.id));
      sessionsList.appendChild(item);
    });
  } catch (error: any) {
    if (requestGeneration === historyRequestGeneration) {
      if (backendOnline) {
        sessionsList.innerHTML = `<div class="list-state error">Could not load recordings. Use Refresh to retry.</div>`;
        showMessage(`History unavailable: ${error.message}`, true);
      } else {
        sessionsList.innerHTML = `<div class="list-state">Waiting for backend connection…</div>`;
      }
    }
  } finally {
    if (requestGeneration === historyRequestGeneration) sessionsList.setAttribute("aria-busy", "false");
  }
}

function renderSessionProcessingState(session: RecordingSession) {
  const processing = session.status === "processing_hq";
  const failed = session.status === "hq_error";
  const interrupted = session.status === "interrupted";
  const completed = session.status === "completed";

  finalTranscriptCard.classList.remove("hidden");
  const startingRetranscription = retranscribeRequestInFlight;
  finalTranscriptCard.classList.toggle("processing", processing || startingRetranscription);
  finalTranscriptText.disabled = processing || transcriptSavingSessionId === session.id;

  transcriptHqStatus.classList.remove("error", "warn", "processing");
  if (processing) {
    transcriptHqStatus.textContent = "Processing HQ…";
    transcriptHqStatus.classList.add("processing");
  } else if (failed) {
    transcriptHqStatus.textContent = "HQ failed";
    transcriptHqStatus.classList.add("error");
  } else if (interrupted) {
    transcriptHqStatus.textContent = "Closed abruptly";
    transcriptHqStatus.classList.add("warn");
  } else {
    transcriptHqStatus.textContent = "HQ complete";
  }

  // Only show re-transcribe button when useful: hide if completed normally
  if (completed && !failed && !interrupted) {
    btnRetranscribe.classList.add("hidden");
  } else {
    btnRetranscribe.classList.remove("hidden");
    const canRetranscribe = !processing && !startingRetranscription && !isRecording && recordingOperation === "idle" && backendOnline && modelCanRecord();
    btnRetranscribe.disabled = !canRetranscribe;
    btnRetranscribe.textContent = startingRetranscription
      ? "Starting…"
      : processing
        ? "Transcribing…"
        : failed
          ? "Retry transcription"
          : interrupted
            ? "Transcribe recording"
            : "Re-transcribe";
    btnRetranscribe.title = startingRetranscription
      ? "Starting the high-quality pass"
      : processing
        ? "A high-quality pass is already running"
        : isRecording
          ? "Stop the active recording before re-transcribing"
          : !backendOnline
            ? "Backend is offline"
            : !modelCanRecord()
              ? "Download the model before transcribing"
              : "Run high-quality transcription pass";
  }
  btnDeleteSession.disabled = deletingSessionId !== null || session.status === "recording" || processing;
}

async function selectSession(sessionId: string) {
  const requestGeneration = ++sessionRequestGeneration;
  emptyDetailHint.classList.add("hidden");
  sessionContentCard.classList.add("hidden");
  sessionLoadingState.classList.remove("hidden");
  sessionContentCard.setAttribute("aria-busy", "true");
  try {
    const session = await api.getRecording(sessionId);
    if (requestGeneration !== sessionRequestGeneration) return;
    activeViewingSession = session;
    lastEditedTranscriptSource = null;

    // Highlight in list
    document.querySelectorAll(".session-item").forEach((el) => {
      el.classList.toggle("active", (el as HTMLElement).dataset.id === sessionId);
    });

    sessionLoadingState.classList.add("hidden");
    sessionContentCard.classList.remove("hidden");

    sessionTitleInput.value = session.title || "";
    sessionDateDisplay.textContent = formatDate(session.created_at);
    sessionDurationDisplay.textContent = formatSeconds(session.duration || 0);
    sessionModeDisplay.textContent = session.mode === "mic_only" ? "Mic Only" : "Mic + System";

    // Keep the transcript visible throughout processing and failures.
    finalTranscriptText.value = session.final_transcript || "";
    renderSessionProcessingState(session);
    if (session.status === "hq_error" && session.status_error) {
      showMessage(`Final transcription failed: ${session.status_error}`, true);
    } else if (session.status === "interrupted") {
      showMessage("This recording was closed abruptly. Preserved audio and partial transcript are available.");
    }

    // Set audio player source
    audioElement.src = api.getAudioUrl(session.id);
    audioElement.load();
    resetAudioControls();

    // Render interactive phrases
    renderInteractivePhrases(session);
    updateStatusCircleAndDiagnostics();
  } catch (error: any) {
    if (requestGeneration === sessionRequestGeneration) {
      sessionLoadingState.classList.add("hidden");
      emptyDetailHint.classList.remove("hidden");
      showMessage(`Could not load recording: ${error.message}`, true);
    }
  } finally {
    if (requestGeneration === sessionRequestGeneration) sessionContentCard.setAttribute("aria-busy", "false");
  }
}

function renderInteractivePhrases(session: RecordingSession) {
  interactivePhrasesContainer.innerHTML = "";

  const phrases = session.phrases || [];
  if (phrases.length === 0) {
    interactivePhrasesContainer.innerHTML = `<div style="color: var(--text-muted); font-size: 15px; padding: 16px 0;">No speech segments transcribed.</div>`;
    return;
  }

  phrases.forEach((phrase, idx) => {
    const row = document.createElement("div");
    row.className = "interactive-phrase-row";

    const speakerLabel = speakerLabelForPhrase(phrase);

    row.innerHTML = `
      <button type="button" class="interactive-phrase-time" title="Jump to ${formatSeconds(phrase.start_time)} in audio">[${formatSeconds(phrase.start_time)}]</button>
      <div class="interactive-phrase-body">
        <span class="interactive-phrase-speaker">${escapeHtml(speakerLabel)}:</span>
        <textarea class="phrase-text-input" rows="1">${escapeHtml(phrase.text)}</textarea>
      </div>
    `;

    // Jump audio to start time
    const timeBtn = row.querySelector(".interactive-phrase-time") as HTMLElement;
    timeBtn.addEventListener("click", () => {
      audioElement.currentTime = phrase.start_time;
      if (audioElement.paused) void audioElement.play().catch(() => showMessage("Audio could not be played.", true));
    });

    const textarea = row.querySelector(".phrase-text-input") as HTMLTextAreaElement;

    // Auto-resize textarea so it expands naturally with text content
    const autoResize = () => {
      textarea.style.height = "auto";
      textarea.style.height = `${textarea.scrollHeight}px`;
    };

    textarea.addEventListener("input", autoResize);
    window.requestAnimationFrame(autoResize);

    // In-place phrase edit on input, blur, and change
    let lastSavedText = phrase.text;
    let phraseDebounceTimer: number | null = null;

    const savePhrase = async () => {
      if (!activeViewingSession || activeViewingSession.id !== session.id) return;
      const newText = textarea.value;
      if (newText === lastSavedText) return;
      const previous = session.phrases[idx].text;
      session.phrases[idx].text = newText;
      lastSavedText = newText;
      lastEditedTranscriptSource = "phrases";
      try {
        await api.updateRecording(session.id, { phrases: session.phrases });
        showMessage("Phrase saved.");
      } catch (error: any) {
        session.phrases[idx].text = previous;
        textarea.value = previous;
        lastSavedText = previous;
        autoResize();
        showMessage(`Phrase was not saved: ${error.message}`, true);
      }
    };

    textarea.addEventListener("input", () => {
      autoResize();
      if (!activeViewingSession || activeViewingSession.id !== session.id) return;
      session.phrases[idx].text = textarea.value;
      lastEditedTranscriptSource = "phrases";
      if (phraseDebounceTimer) window.clearTimeout(phraseDebounceTimer);
      phraseDebounceTimer = window.setTimeout(savePhrase, 600);
    });

    textarea.addEventListener("blur", () => {
      if (phraseDebounceTimer) {
        window.clearTimeout(phraseDebounceTimer);
        phraseDebounceTimer = null;
      }
      void savePhrase();
    });

    textarea.addEventListener("change", () => {
      if (phraseDebounceTimer) {
        window.clearTimeout(phraseDebounceTimer);
        phraseDebounceTimer = null;
      }
      void savePhrase();
    });

    interactivePhrasesContainer.appendChild(row);
  });
}

function resetAudioControls() {
  audioElement.pause();
  playIcon.classList.remove("hidden");
  pauseIcon.classList.add("hidden");
  playerScrubSlider.value = "0";
  playerCurrentTime.textContent = "00:00";
  playerTotalTime.textContent = "00:00";
}

// Audio Player Handlers
function setupAudioPlayer() {
  btnPlayerPlay.addEventListener("click", () => {
    if (audioElement.paused) {
      void audioElement.play().catch(() => showMessage("Audio could not be played.", true));
    } else {
      audioElement.pause();
    }
  });

  audioElement.addEventListener("play", () => {
    playIcon.classList.add("hidden");
    pauseIcon.classList.remove("hidden");
  });

  audioElement.addEventListener("pause", () => {
    playIcon.classList.remove("hidden");
    pauseIcon.classList.add("hidden");
  });

  audioElement.addEventListener("timeupdate", () => {
    const cur = audioElement.currentTime;
    const dur = audioElement.duration || 0;
    playerCurrentTime.textContent = formatSeconds(cur);
    if (dur > 0) {
      playerScrubSlider.value = String((cur / dur) * 100);
    }
  });

  audioElement.addEventListener("loadedmetadata", () => {
    playerTotalTime.textContent = formatSeconds(audioElement.duration || 0);
  });

  audioElement.addEventListener("ended", () => {
    playIcon.classList.remove("hidden");
    pauseIcon.classList.add("hidden");
  });

  playerScrubSlider.addEventListener("input", () => {
    const pct = parseFloat(playerScrubSlider.value) / 100;
    if (audioElement.duration) {
      audioElement.currentTime = pct * audioElement.duration;
    }
  });

  speedBtns.forEach((btn) => {
    btn.addEventListener("click", () => {
      speedBtns.forEach((b) => b.classList.remove("active"));
      btn.classList.add("active");
      const speed = parseFloat((btn as HTMLElement).dataset.speed || "1.0");
      audioElement.playbackRate = speed;
    });
  });
}

// History Actions (Auto-save Title, Retranscribe, Copy, Export, Delete)
function setupHistoryActions() {
  btnRefreshHistory.addEventListener("click", async () => {
    btnRefreshHistory.disabled = true;
    btnRefreshHistory.setAttribute("aria-busy", "true");
    await refreshHistory();
    btnRefreshHistory.disabled = false;
    btnRefreshHistory.removeAttribute("aria-busy");
  });

  // Auto-save title directly on typing (debounced) and on blur; no explicit button needed
  let titleDebounceTimer: number | null = null;
  const saveTitleNow = async () => {
    if (!activeViewingSession) return;
    const session = activeViewingSession;
    const newTitle = sessionTitleInput.value.trim() || "Untitled Meeting";
    if (session.title === newTitle) return;
    try {
      await api.updateRecording(session.id, { title: newTitle });
      session.title = newTitle;
      const itemTitleEl = document.querySelector(`.session-item[data-id="${session.id}"] .session-item-title`);
      if (itemTitleEl) itemTitleEl.textContent = newTitle;
    } catch (error: any) {
      showMessage(`Title was not saved: ${error.message}`, true);
    }
  };

  sessionTitleInput.addEventListener("input", () => {
    if (titleDebounceTimer) window.clearTimeout(titleDebounceTimer);
    titleDebounceTimer = window.setTimeout(saveTitleNow, 600);
  });

  sessionTitleInput.addEventListener("blur", () => {
    if (titleDebounceTimer) {
      window.clearTimeout(titleDebounceTimer);
      titleDebounceTimer = null;
    }
    void saveTitleNow();
  });

  btnRetranscribe.addEventListener("click", async () => {
    if (!activeViewingSession || btnRetranscribe.disabled) return;
    const session = activeViewingSession;
    retranscribeRequestInFlight = true;
    btnRetranscribe.setAttribute("aria-busy", "true");
    renderSessionProcessingState(session);
    targetHqSessionId = session.id;
    showHqModal("Re-transcribing Audio…");
    try {
      await api.retranscribeRecording(session.id);
      session.status = "processing_hq";
      session.status_error = null;
      if (activeViewingSession?.id === session.id) renderSessionProcessingState(session);
      void refreshHistory();
    } catch (err: any) {
      hideHqModal();
      targetHqSessionId = null;
      showMessage(`Re-transcription could not start: ${err.message}`, true);
      if (activeViewingSession?.id === session.id) renderSessionProcessingState(session);
    } finally {
      retranscribeRequestInFlight = false;
      btnRetranscribe.removeAttribute("aria-busy");
      if (activeViewingSession) renderSessionProcessingState(activeViewingSession);
    }
  });

  function getCurrentTranscriptToCopy(): string {
    if (!activeViewingSession) return "";

    // 1. Immediately sync whatever is currently in the DOM
    if (finalTranscriptText) {
      activeViewingSession.final_transcript = finalTranscriptText.value;
    }

    const phraseTextareas = interactivePhrasesContainer.querySelectorAll<HTMLTextAreaElement>(".phrase-text-input");
    phraseTextareas.forEach((ta, i) => {
      if (activeViewingSession?.phrases && activeViewingSession.phrases[i]) {
        activeViewingSession.phrases[i].text = ta.value;
      }
    });

    // 2. Determine what to copy based on what the user edited or what is available
    if (lastEditedTranscriptSource === "phrases" && activeViewingSession.phrases && activeViewingSession.phrases.length > 0) {
      return activeViewingSession.phrases.map((p) => `${speakerLabelForPhrase(p)}: ${p.text}`).join("\n");
    }

    if (activeViewingSession.final_transcript && activeViewingSession.final_transcript.trim()) {
      return activeViewingSession.final_transcript.trim();
    }

    if (activeViewingSession.phrases && activeViewingSession.phrases.length > 0) {
      return activeViewingSession.phrases.map((p) => `${speakerLabelForPhrase(p)}: ${p.text}`).join("\n");
    }

    return "";
  }

  let finalTranscriptDebounceTimer: number | null = null;
  const saveFinalTranscriptNow = async () => {
    if (!activeViewingSession) return;
    const session = activeViewingSession;
    const newText = finalTranscriptText.value;
    if (session.final_transcript === newText && !transcriptSavingSessionId) return;
    const previous = session.final_transcript;
    session.final_transcript = newText;
    lastEditedTranscriptSource = "final_transcript";
    transcriptSavingSessionId = session.id;
    try {
      await api.updateRecording(session.id, { final_transcript: newText });
      showMessage("Transcript saved.");
    } catch (error: any) {
      if (activeViewingSession?.id === session.id) {
        session.final_transcript = previous;
        finalTranscriptText.value = previous;
      }
      showMessage(`Transcript was not saved: ${error.message}`, true);
    } finally {
      if (transcriptSavingSessionId === session.id) transcriptSavingSessionId = null;
      if (activeViewingSession?.id === session.id) renderSessionProcessingState(session);
    }
  };

  finalTranscriptText.addEventListener("input", () => {
    if (!activeViewingSession) return;
    activeViewingSession.final_transcript = finalTranscriptText.value;
    lastEditedTranscriptSource = "final_transcript";
    if (finalTranscriptDebounceTimer) window.clearTimeout(finalTranscriptDebounceTimer);
    finalTranscriptDebounceTimer = window.setTimeout(saveFinalTranscriptNow, 600);
  });

  finalTranscriptText.addEventListener("blur", () => {
    if (finalTranscriptDebounceTimer) {
      window.clearTimeout(finalTranscriptDebounceTimer);
      finalTranscriptDebounceTimer = null;
    }
    void saveFinalTranscriptNow();
  });

  finalTranscriptText.addEventListener("change", () => {
    if (finalTranscriptDebounceTimer) {
      window.clearTimeout(finalTranscriptDebounceTimer);
      finalTranscriptDebounceTimer = null;
    }
    void saveFinalTranscriptNow();
  });

  btnCopyTranscript.addEventListener("click", async () => {
    if (!activeViewingSession) return;
    const text = getCurrentTranscriptToCopy();
    if (!text.trim()) {
      showMessage("There is no transcript to copy yet.", true);
      return;
    }
    try {
      await navigator.clipboard.writeText(text);
      btnCopyTranscript.textContent = "Copied!";
      window.setTimeout(() => { btnCopyTranscript.textContent = "Copy"; }, 1500);
    } catch {
      showMessage("Clipboard access was denied. Select and copy the transcript manually.", true);
    }
  });

  btnDownloadTxt.addEventListener("click", () => {
    if (!activeViewingSession) return;
    getCurrentTranscriptToCopy();
    let text = `Title: ${activeViewingSession.title}\nDate: ${activeViewingSession.created_at}\nDuration: ${formatSeconds(activeViewingSession.duration || 0)}\n\n`;
    if (activeViewingSession.final_transcript) {
      text += `--- Full Transcript ---\n${activeViewingSession.final_transcript}\n\n`;
    }
    if (activeViewingSession.phrases && activeViewingSession.phrases.length > 0) {
      text += `--- Phrases ---\n` + activeViewingSession.phrases.map((p) => `[${formatSeconds(p.start_time)}] ${speakerLabelForPhrase(p)}: ${p.text}`).join("\n");
    }
    downloadFile(`${activeViewingSession.id}.txt`, text, "text/plain");
  });

  btnDownloadSrt.addEventListener("click", () => {
    if (!activeViewingSession) return;
    getCurrentTranscriptToCopy();
    const srtContent = (activeViewingSession.phrases || [])
      .map((p, i) => `${i + 1}\n${toSrtTime(p.start_time)} --> ${toSrtTime(p.end_time)}\n${speakerLabelForPhrase(p)}: ${p.text}\n`)
      .join("\n");
    downloadFile(`${activeViewingSession.id}.srt`, srtContent, "text/plain");
  });

  btnDeleteSession.addEventListener("click", async () => {
    if (!activeViewingSession || btnDeleteSession.disabled) return;
    const session = activeViewingSession;
    if (!confirm(`Delete recording "${session.title}"?`)) return;
    deletingSessionId = session.id;
    btnDeleteSession.disabled = true;
    btnDeleteSession.setAttribute("aria-busy", "true");
    btnDeleteSession.textContent = "Deleting…";
    try {
      await api.deleteRecording(session.id);
      if (activeViewingSession?.id === session.id) {
        activeViewingSession = null;
        sessionContentCard.classList.add("hidden");
        emptyDetailHint.classList.remove("hidden");
      }
      showMessage("Recording deleted.");
      void refreshHistory();
    } catch (error: any) {
      showMessage(`Recording was not deleted: ${error.message}`, true);
    } finally {
      if (deletingSessionId === session.id) deletingSessionId = null;
      btnDeleteSession.removeAttribute("aria-busy");
      btnDeleteSession.textContent = "Delete";
      if (activeViewingSession) renderSessionProcessingState(activeViewingSession);
    }
  });
}

// Settings Modal
function loadSavedSettings() {
  try {
    const saved = JSON.parse(localStorage.getItem("sayso.settings") || "{}");
    selectedMicIndex = Number.isInteger(saved.mic) ? saved.mic : -1;
    selectedSystemIndex = Number.isInteger(saved.system) ? saved.system : -1;
    selectedLanguage = typeof saved.language === "string" ? saved.language : "en";
    selectLanguage.value = selectedLanguage;
  } catch {
    localStorage.removeItem("sayso.settings");
  }
}

function setupSettingsModal() {
  let previouslyFocused: HTMLElement | null = null;
  const closeModal = () => {
    settingsModal.classList.add("hidden");
    previouslyFocused?.focus();
  };
  const openModal = () => {
    previouslyFocused = document.activeElement as HTMLElement;
    settingsModal.classList.remove("hidden");
    void refreshDevices();
    window.setTimeout(() => btnCloseSettings.focus(), 0);
  };

  btnOpenSettings.addEventListener("click", openModal);
  btnCloseSettings.addEventListener("click", closeModal);
  settingsModal.addEventListener("click", (event) => {
    if (event.target === settingsModal) closeModal();
  });
  document.addEventListener("keydown", (event) => {
    if (event.key === "Escape" && !settingsModal.classList.contains("hidden")) closeModal();
  });

  btnRefreshDevices.addEventListener("click", async () => {
    btnRefreshDevices.textContent = "Refreshing…";
    btnRefreshDevices.disabled = true;
    btnRefreshDevices.setAttribute("aria-busy", "true");
    await refreshDevices();
    btnRefreshDevices.textContent = "Refresh Devices";
    btnRefreshDevices.disabled = false;
    btnRefreshDevices.removeAttribute("aria-busy");
  });

  btnPreloadModel.addEventListener("click", async () => {
    if (currentModelStatus === "not_downloaded" || currentModelStatus === "error") {
      closeModal();
      btnDownloadModel.click();
    } else {
      btnPreloadModel.textContent = "Loading...";
      btnPreloadModel.disabled = true;
      try {
        await api.preloadModel();
      } catch (err: any) {
        btnPreloadModel.disabled = false;
        btnPreloadModel.textContent = "Retry Load";
        showMessage(`Model could not be loaded: ${err.message}`, true);
      }
    }
  });

  selectMicDevice.addEventListener("change", () => {
    const v = parseInt(selectMicDevice.value, 10);
    selectedMicIndex = isNaN(v) || v < 0 ? -1 : v;
  });

  selectSystemDevice.addEventListener("change", () => {
    const v = parseInt(selectSystemDevice.value, 10);
    selectedSystemIndex = isNaN(v) || v < 0 ? -1 : v;
  });

  btnSaveSettings.addEventListener("click", () => {
    const micVal = parseInt(selectMicDevice.value, 10);
    selectedMicIndex = isNaN(micVal) || micVal < 0 ? -1 : micVal;

    const sysVal = parseInt(selectSystemDevice.value, 10);
    selectedSystemIndex = isNaN(sysVal) || sysVal < 0 ? -1 : sysVal;

    selectedLanguage = selectLanguage.value;
    localStorage.setItem("sayso.settings", JSON.stringify({
      mic: selectedMicIndex,
      system: selectedSystemIndex,
      language: selectedLanguage,
    }));
    showMessage("Settings saved.");
    closeModal();
  });
}

// Navigation Tabs
function setupTabs() {
  tabRecord.addEventListener("click", () => switchTab("record"));
  tabHistory.addEventListener("click", () => switchTab("history"));
}

function switchTab(tab: "record" | "history") {
  if (tab === "record") {
    tabRecord.classList.add("active");
    tabHistory.classList.remove("active");
    tabRecord.setAttribute("aria-selected", "true");
    tabHistory.setAttribute("aria-selected", "false");
    viewRecord.classList.add("active");
    viewHistory.classList.remove("active");
  } else {
    tabRecord.classList.remove("active");
    tabHistory.classList.add("active");
    tabRecord.setAttribute("aria-selected", "false");
    tabHistory.setAttribute("aria-selected", "true");
    viewRecord.classList.remove("active");
    viewHistory.classList.add("active");
    void refreshHistory();
  }
}

// Window Controls (Tauri IPC with Graceful Web Fallback)
function isTauriEnvironment(): boolean {
  return typeof window !== "undefined" && Boolean((window as any).__TAURI_INTERNALS__ || (window as any).__TAURI__);
}

async function updateMaximizeState() {
  if (!isTauriEnvironment()) return;
  try {
    const isMax = await invoke<boolean>("is_window_maximized");
    if (typeof isMax === "boolean" && winMaximizeIcon && winRestoreIcon) {
      winMaximizeIcon.classList.toggle("hidden", isMax);
      winRestoreIcon.classList.toggle("hidden", !isMax);
      btnWinMaximize?.setAttribute("title", isMax ? "Restore" : "Enlarge");
      btnWinMaximize?.setAttribute("aria-label", isMax ? "Restore" : "Enlarge");
    }
  } catch {
    // Ignore if not in Tauri
  }
}

function setupWindowControls() {
  btnWinMinimize?.addEventListener("click", async () => {
    if (isTauriEnvironment()) {
      try {
        await invoke("minimize_window");
      } catch (err) {
        console.warn("[Tauri] minimize error", err);
      }
    }
  });

  btnWinMaximize?.addEventListener("click", async () => {
    if (isTauriEnvironment()) {
      try {
        await invoke("toggle_maximize_window");
        window.setTimeout(() => void updateMaximizeState(), 60);
      } catch (err) {
        console.warn("[Tauri] toggle_maximize error", err);
      }
    } else {
      if (!document.fullscreenElement) {
        document.documentElement.requestFullscreen().catch(() => {});
      } else {
        document.exitFullscreen().catch(() => {});
      }
    }
  });

  btnWinClose?.addEventListener("click", async () => {
    if (isTauriEnvironment()) {
      try {
        await invoke("close_window");
      } catch (err) {
        console.warn("[Tauri] close error", err);
      }
    } else {
      window.close();
    }
  });

  if (appTitlebar) {
    appTitlebar.addEventListener("dblclick", async (e) => {
      if (!(e.target as HTMLElement).closest(".titlebar-btn")) {
        if (isTauriEnvironment()) {
          try {
            await invoke("toggle_maximize_window");
            window.setTimeout(() => void updateMaximizeState(), 60);
          } catch (err) {
            console.warn("[Tauri] toggle_maximize error", err);
          }
        }
      }
    });
  }

  window.addEventListener("resize", () => {
    void updateMaximizeState();
  });

  void updateMaximizeState();
}

// Utility Helpers
function pad(n: number): string {
  return n < 10 ? `0${n}` : String(n);
}

function formatSeconds(secs: number): string {
  const s = Math.max(0, Math.floor(Number.isFinite(secs) ? secs : 0));
  const mins = Math.floor(s / 60);
  const remainderSecs = s % 60;
  return `${pad(mins)}:${pad(remainderSecs)}`;
}

function toSrtTime(secs: number): string {
  secs = Math.max(0, Number.isFinite(secs) ? secs : 0);
  const h = Math.floor(secs / 3600);
  const m = Math.floor((secs % 3600) / 60);
  const s = Math.floor(secs % 60);
  const ms = Math.floor((secs - Math.floor(secs)) * 1000);
  return `${pad(h)}:${pad(m)}:${pad(s)},${ms.toString().padStart(3, "0")}`;
}

function formatDate(isoStr?: string): string {
  if (!isoStr) return "";
  const d = new Date(isoStr);
  if (Number.isNaN(d.getTime())) return "Unknown date";
  return d.toLocaleDateString(undefined, { month: "short", day: "numeric", hour: "2-digit", minute: "2-digit" });
}

function escapeHtml(text: string): string {
  const div = document.createElement("div");
  div.textContent = text;
  return div.innerHTML;
}

function downloadFile(filename: string, content: string, type: string) {
  const blob = new Blob([content], { type });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = filename;
  document.body.appendChild(a);
  a.click();
  a.remove();
  window.setTimeout(() => URL.revokeObjectURL(url), 1000);
}

// Start
initApp();
