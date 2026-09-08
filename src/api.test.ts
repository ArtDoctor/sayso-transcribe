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
    expect(client.getAudioUrl("sess_123")).toBe("http://127.0.0.1:8765/api/recordings/sess_123/audio");
    expect(client.getAudioUrl("sess space")).toContain("sess%20space/audio");
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
    expect(String(fetchMock.mock.calls[1][0])).toContain("localhost:8765/api/status");
  });

  it("exposes model lifecycle actions", () => {
    const client = new ApiClient();
    expect(typeof client.downloadModel).toBe("function");
    expect(typeof client.preloadModel).toBe("function");
    expect(typeof client.retranscribeRecording).toBe("function");
  });
});
