export interface AudioDevice {
  index: number;
  name: string;
  channels?: number;
  sample_rate?: number;
}

export interface DevicesResponse {
  default_mic: { index: number; name: string };
  default_system: { index: number; name: string };
  microphones: AudioDevice[];
  system_devices: AudioDevice[];
}

export interface ModelDownloadProgress {
  filename: string;
  percent: number;
  downloaded_mb: number;
  total_mb: number;
  speed_mb_s?: number;
  eta_s?: number;
}

export interface BackendStatus {
  status: string;
  is_recording: boolean;
  active_session_id?: string;
  active_recording_mode?: "mic_only" | "mic_and_system";
  recording_duration?: number;
  recording_error?: string | null;
  model_status: "not_downloaded" | "downloading" | "not_loaded" | "loading" | "ready" | "transcribing" | "error";
  is_model_downloaded?: boolean;
  is_model_loaded?: boolean;
  download_progress?: ModelDownloadProgress;
  model_error?: string;
  default_mic?: { index: number; name: string };
  default_system?: { index: number; name: string };
  device?: string;
  cuda_device_name?: string | null;
}

export interface Phrase {
  session_id?: string;
  phrase_id: string;
  speaker: string;
  start_time: number;
  end_time: number;
  duration: number;
  text: string;
  latency_s?: number;
  error?: string;
  status?: "pending" | "transcribed";
}

export interface RecordingSession {
  id: string;
  title: string;
  created_at: string;
  duration: number;
  mode: "mic_only" | "mic_and_system";
  language: string;
  status: "recording" | "processing_hq" | "completed" | "hq_error" | "interrupted";
  status_error?: string | null;
  phrases: Phrase[];
  final_transcript: string;
  audio_file: string;
}

export interface VADMeterPayload {
  type: "vad_meter";
  mic: { rms: number; prob: number; is_speaking: boolean };
  system: { rms: number; prob: number; is_speaking: boolean };
  duration: number;
}

function getBaseUrls() {
  const location = typeof window !== "undefined" ? window.location : undefined;
  // Vite's development server proxies /api and /ws. Tauri production uses
  // the `tauri.localhost` origin, which would return index.html for `/api/*`
  // and cause JSON parsing errors. Only use relative URLs for known dev ports.
  const isViteDevServer = Boolean(
    location &&
    location.protocol.startsWith("http") &&
    (location.hostname === "localhost" || location.hostname === "127.0.0.1") &&
    location.port === "41765"
  );

  if (isViteDevServer && location) {
    const wsProto = location.protocol === "https:" ? "wss:" : "ws:";
    return {
      primaryApi: "", // Uses Vite proxy in dev mode
      fallbackApi: "http://127.0.0.1:48653",
      primaryWs: `${wsProto}//${location.host}/ws`,
      fallbackWs: "ws://127.0.0.1:48653/ws",
    };
  }
  return {
    primaryApi: "http://127.0.0.1:48653",
    fallbackApi: "http://localhost:48653",
    primaryWs: "ws://127.0.0.1:48653/ws",
    fallbackWs: "ws://localhost:48653/ws",
  };
}

export class ApiClient {
  private ws: WebSocket | null = null;
  private wsListeners: Array<(msg: any) => void> = [];
  public isConnected = false;
  private activeApiBase = "";
  private wsGeneration = 0;
  private reconnectTimer: ReturnType<typeof setTimeout> | null = null;
  private shouldReconnect = false;
  private connectionReported = false;

  constructor() {
    this.activeApiBase = getBaseUrls().primaryApi;
  }

  private async fetchWithTimeout(url: string, options?: RequestInit): Promise<Response> {
    if (options?.signal) return fetch(url, options);
    const controller = new AbortController();
    const timer = setTimeout(() => controller.abort(), 15_000);
    try {
      return await fetch(url, { ...options, signal: controller.signal });
    } catch (error) {
      if (controller.signal.aborted) throw new Error("The backend did not respond within 15 seconds");
      throw error;
    } finally {
      clearTimeout(timer);
    }
  }

  private async request(path: string, options?: RequestInit): Promise<Response> {
    const urls = getBaseUrls();
    const method = (options?.method || "GET").toUpperCase();
    try {
      const response = await this.fetchWithTimeout(`${this.activeApiBase}${path}`, options);
      // Safe reads can probe another base; mutations must never be replayed.
      if (response.status !== 404 || method !== "GET" || this.activeApiBase !== urls.primaryApi) return response;
    } catch (error) {
      if (method !== "GET") throw error;
    }
    const altBase = this.activeApiBase === urls.primaryApi ? urls.fallbackApi : urls.primaryApi;
    const response = await this.fetchWithTimeout(`${altBase}${path}`, options);
    if (response.ok) this.activeApiBase = altBase;
    return response;
  }

  private async json<T>(path: string, options?: RequestInit): Promise<T> {
    const response = await this.request(path, options);
    if (!response.ok) {
      const body = await response.json().catch(() => null);
      throw new Error(body?.detail || `Request failed (HTTP ${response.status})`);
    }
    return response.json();
  }

  async getStatus(): Promise<BackendStatus> {
    return this.json("/api/status");
  }

  async getDevices(): Promise<DevicesResponse> {
    return this.json("/api/devices");
  }

  async startRecording(params: {
    mode: "mic_only" | "mic_and_system";
    language?: string;
    title?: string;
    mic_device_index?: number;
    system_device_index?: number;
  }): Promise<{ session_id: string; status: string; session: RecordingSession }> {
    const res = await this.request("/api/record/start", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(params),
    });
    if (!res.ok) {
      const err = await res.json().catch(() => ({ detail: "Failed to start recording" }));
      throw new Error(err.detail || "Failed to start recording");
    }
    return res.json();
  }

  async stopRecording(): Promise<{ session_id: string; duration: number; status: string }> {
    return this.json("/api/record/stop", { method: "POST" });
  }

  async listRecordings(): Promise<RecordingSession[]> {
    return this.json("/api/recordings");
  }

  async getRecording(sessionId: string): Promise<RecordingSession> {
    return this.json(`/api/recordings/${encodeURIComponent(sessionId)}`);
  }

  getAudioUrl(sessionId: string): string {
    const base = this.activeApiBase || "http://127.0.0.1:48653";
    return `${base}/api/recordings/${encodeURIComponent(sessionId)}/audio`;
  }

  async updateRecording(
    sessionId: string,
    updates: { title?: string; final_transcript?: string; phrases?: Phrase[] }
  ): Promise<RecordingSession> {
    return this.json(`/api/recordings/${encodeURIComponent(sessionId)}`, {
      method: "PATCH",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(updates),
    });
  }

  async deleteRecording(sessionId: string): Promise<boolean> {
    await this.json(`/api/recordings/${encodeURIComponent(sessionId)}`, { method: "DELETE" });
    return true;
  }

  async retranscribeRecording(sessionId: string): Promise<{ session_id: string; status: string }> {
    const res = await this.request(`/api/recordings/${encodeURIComponent(sessionId)}/retranscribe`, {
      method: "POST",
    });
    if (!res.ok) {
      const err = await res.json().catch(() => ({ detail: "Failed to start retranscription" }));
      throw new Error(err.detail || `HTTP error ${res.status}`);
    }
    return res.json();
  }

  async preloadModel(): Promise<any> {
    return this.json("/api/model/preload", { method: "POST" });
  }

  async downloadModel(): Promise<any> {
    return this.json("/api/model/download", { method: "POST" });
  }

  connectWebSocket(onOpen?: () => void, onClose?: () => void) {
    this.disconnect();
    this.shouldReconnect = true;
    const generation = ++this.wsGeneration;
    const urls = getBaseUrls();

    const reportClosed = () => {
      if (!this.connectionReported) return;
      this.connectionReported = false;
      this.isConnected = false;
      onClose?.();
    };
    const schedule = (nextUrl: string) => {
      if (!this.shouldReconnect || generation !== this.wsGeneration || this.reconnectTimer) return;
      this.reconnectTimer = setTimeout(() => {
        this.reconnectTimer = null;
        attemptConnect(nextUrl);
      }, 2000);
    };
    const attemptConnect = (wsUrl: string) => {
      if (!this.shouldReconnect || generation !== this.wsGeneration) return;
      let socket: WebSocket;
      try {
        socket = new WebSocket(wsUrl);
        this.ws = socket;
      } catch {
        reportClosed();
        schedule(wsUrl === urls.primaryWs ? urls.fallbackWs : urls.primaryWs);
        return;
      }
      socket.onopen = () => {
        if (socket !== this.ws || generation !== this.wsGeneration) return;
        this.isConnected = true;
        if (!this.connectionReported) {
          this.connectionReported = true;
          onOpen?.();
        }
      };
      socket.onclose = () => {
        if (socket !== this.ws || generation !== this.wsGeneration) return;
        this.ws = null;
        reportClosed();
        schedule(wsUrl === urls.primaryWs ? urls.fallbackWs : urls.primaryWs);
      };
      socket.onerror = () => {
        if (socket === this.ws) reportClosed();
      };
      socket.onmessage = (event) => {
        if (socket !== this.ws || generation !== this.wsGeneration) return;
        try {
          const data = JSON.parse(event.data);
          this.wsListeners.forEach((listener) => listener(data));
        } catch {
          console.warn("Ignored malformed WebSocket message");
        }
      };
    };
    attemptConnect(urls.primaryWs);
  }

  disconnect() {
    this.shouldReconnect = false;
    this.wsGeneration++;
    if (this.reconnectTimer) clearTimeout(this.reconnectTimer);
    this.reconnectTimer = null;
    const socket = this.ws;
    this.ws = null;
    this.isConnected = false;
    this.connectionReported = false;
    try { socket?.close(); } catch {}
  }

  onWsMessage(listener: (msg: any) => void): () => void {
    this.wsListeners.push(listener);
    return () => { this.wsListeners = this.wsListeners.filter((item) => item !== listener); };
  }
}

export const api = new ApiClient();
