import { afterEach, describe, expect, it, vi } from "vitest";
import { ApiClient } from "./api";

afterEach(() => {
  vi.restoreAllMocks();
  vi.unstubAllGlobals();
});

describe("ApiClient", () => {
  it("starts disconnected", () => {
    const client = new ApiClient();
    expect(client.isConnected).toBe(false);
  });

  it("constructs an encoded audio URL", () => {
    const client = new ApiClient();
    expect(client.getAudioUrl("sess_123")).toBe("http://127.0.0.1:48653/api/recordings/sess_123/audio");
    expect(client.getAudioUrl("sess space")).toContain("sess%20space/audio");
  });

  it("uses the backend directly from the Tauri production origin", () => {
    vi.stubGlobal("window", {
      location: { protocol: "http:", hostname: "tauri.localhost", host: "tauri.localhost", port: "80" },
    });
    const client = new ApiClient();
    expect(client.getAudioUrl("sess_123")).toBe("http://127.0.0.1:48653/api/recordings/sess_123/audio");
  });

  it("encodes session id when opening session folder", async () => {
    const fetchMock = vi.fn().mockResolvedValue(new Response(
      JSON.stringify({ opened: true, path: "D:\\test" }),
      { status: 200, headers: { "Content-Type": "application/json" } },
    ));
    vi.stubGlobal("fetch", fetchMock);

    const client = new ApiClient();
    const res = await client.openSessionFolder("sess 123");
    expect(res.opened).toBe(true);
    expect(String(fetchMock.mock.calls[0][0])).toContain("/api/recordings/sess%20123/open_folder");
  });

  it("surfaces backend detail for a failed mutation", async () => {
    const fetchMock = vi.fn().mockResolvedValue(new Response(
      JSON.stringify({ detail: "Recording is already stopping" }),
      { status: 409, headers: { "Content-Type": "application/json" } },
    ));
    vi.stubGlobal("fetch", fetchMock);

    await expect(new ApiClient().stopRecording()).rejects.toThrow("Recording is already stopping");
    expect(fetchMock).toHaveBeenCalledTimes(1);
  });

  it("never replays a mutation on the fallback backend", async () => {
    const fetchMock = vi.fn().mockRejectedValue(new Error("connection lost"));
    vi.stubGlobal("fetch", fetchMock);

    await expect(new ApiClient().downloadModel()).rejects.toThrow("connection lost");
    expect(fetchMock).toHaveBeenCalledTimes(1);
  });

  it("retries a safe GET against the alternate loopback hostname", async () => {
    const fetchMock = vi.fn()
      .mockRejectedValueOnce(new Error("primary unavailable"))
      .mockResolvedValueOnce(new Response(JSON.stringify({ status: "online" }), {
        status: 200,
        headers: { "Content-Type": "application/json" },
      }));
    vi.stubGlobal("fetch", fetchMock);

    await expect(new ApiClient().getStatus()).resolves.toMatchObject({ status: "online" });
    expect(fetchMock).toHaveBeenCalledTimes(2);
    expect(String(fetchMock.mock.calls[1][0])).toContain("localhost:48653/api/status");
  });

  it("exposes model lifecycle actions", () => {
    const client = new ApiClient();
    expect(typeof client.downloadModel).toBe("function");
    expect(typeof client.preloadModel).toBe("function");
    expect(typeof client.unloadModel).toBe("function");
    expect(typeof client.setRecordingLanguage).toBe("function");
    expect(typeof client.retranscribeRecording).toBe("function");
    expect(typeof client.getFfmpegStatus).toBe("function");
    expect(typeof client.uploadAudio).toBe("function");
  });

  it("calls getFfmpegStatus endpoint", async () => {
    const fetchMock = vi.fn().mockResolvedValue(new Response(
      JSON.stringify({ installed: true, version: "ffmpeg 6.0" }),
      { status: 200, headers: { "Content-Type": "application/json" } },
    ));
    vi.stubGlobal("fetch", fetchMock);

    const client = new ApiClient();
    const res = await client.getFfmpegStatus();
    expect(res.installed).toBe(true);
    expect(res.version).toBe("ffmpeg 6.0");
    expect(String(fetchMock.mock.calls[0][0])).toContain("/api/system/ffmpeg");
  });

  it("sends FormData when calling uploadAudio", async () => {
    const fetchMock = vi.fn().mockResolvedValue(new Response(
      JSON.stringify({ session_id: "sess_upload_1", status: "processing_hq", job_id: "job_1", session: {} }),
      { status: 200, headers: { "Content-Type": "application/json" } },
    ));
    vi.stubGlobal("fetch", fetchMock);

    const client = new ApiClient();
    const file = new File(["fake audio"], "test.mp3", { type: "audio/mp3" });
    const res = await client.uploadAudio(file, "en", "My Audio");
    expect(res.session_id).toBe("sess_upload_1");
    expect(fetchMock.mock.calls[0][1].method).toBe("POST");
    expect(fetchMock.mock.calls[0][1].body).toBeInstanceOf(FormData);
  });

  it("includes device names in startRecording payload", async () => {
    const fetchMock = vi.fn().mockResolvedValue(new Response(
      JSON.stringify({ session_id: "sess_rec_1", status: "recording", session: {} }),
      { status: 200, headers: { "Content-Type": "application/json" } },
    ));
    vi.stubGlobal("fetch", fetchMock);

    const client = new ApiClient();
    await client.startRecording({
      mode: "mic_only",
      mic_device_name: "Headphones Mic",
      system_device_name: "Headphones Loopback",
    });
    const body = JSON.parse(fetchMock.mock.calls[0][1].body);
    expect(body.mic_device_name).toBe("Headphones Mic");
    expect(body.system_device_name).toBe("Headphones Loopback");
  });
});
