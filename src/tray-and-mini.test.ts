import { describe, expect, it } from "vitest";

declare const require: any;
declare const __dirname: string;
const fs = require("fs");
const path = require("path");

describe("System Tray and Mini Widget Integration", () => {
  const rootDir = path.resolve(__dirname, "..");
  const indexHtml = fs.readFileSync(path.join(rootDir, "index.html"), "utf-8");
  const tauriConf = JSON.parse(fs.readFileSync(path.join(rootDir, "src-tauri", "tauri.conf.json"), "utf-8"));
  const stylesCss = fs.readFileSync(path.join(rootDir, "src", "styles.css"), "utf-8");
  const libRs = fs.readFileSync(path.join(rootDir, "src-tauri", "src", "lib.rs"), "utf-8");
  const mainTs = fs.readFileSync(path.join(rootDir, "src", "main.ts"), "utf-8");

  describe("index.html mini widget structure", () => {
    it("does not have the recording timer in the mini widget", () => {
      // #mini-timer should not exist in the mini floating widget
      expect(indexHtml).not.toMatch(/id=["']mini-timer["']/);
    });

    it("does not have the white circle / rec-dot in the mini widget", () => {
      const widgetSectionMatch = indexHtml.match(/id=["']mini-floating-widget["'][\s\S]*?<\/div>\s*<\/div>\s*<\/div>/);
      expect(widgetSectionMatch).not.toBeNull();
      const widgetHtml = widgetSectionMatch![0];
      expect(widgetHtml).not.toContain("rec-dot");
      expect(widgetHtml).not.toContain("mini-rec-status");
    });

    it("contains the 3-bar vad meter matching the main application", () => {
      expect(indexHtml).toContain('id="mini-widget-body"');
      expect(indexHtml).toContain('id="mini-vad-meter"');
      expect(indexHtml).toContain('id="mini-vad-1"');
      expect(indexHtml).toContain('id="mini-vad-2"');
      expect(indexHtml).toContain('id="mini-vad-3"');
      expect(indexHtml).not.toContain('id="mini-vad-4"');
    });

    it("contains open app button and stop recording button", () => {
      expect(indexHtml).toContain('id="btn-mini-open-app"');
      expect(indexHtml).toContain('id="btn-mini-stop-rec"');
    });

    it("has drag region attributes on the draggable elements", () => {
      expect(indexHtml).toMatch(/id=["']mini-floating-widget["'][^>]*data-tauri-drag-region/);
      expect(indexHtml).toMatch(/id=["']mini-widget-body["'][^>]*data-tauri-drag-region/);
      expect(indexHtml).toMatch(/id=["']mini-vad-meter["'][^>]*data-tauri-drag-region/);
    });
  });

  describe("tauri.conf.json mini window configuration", () => {
    const miniWin = tauriConf.app.windows.find((w: any) => w.label === "mini");

    it("has mini window defined with compact dimensions and transparent settings", () => {
      expect(miniWin).toBeDefined();
      expect(miniWin.transparent).toBe(true);
      expect(miniWin.decorations).toBe(false);
      expect(miniWin.alwaysOnTop).toBe(true);
      expect(miniWin.resizable).toBe(false);
      expect(miniWin.width).toBeLessThanOrEqual(180);
      expect(miniWin.height).toBeLessThanOrEqual(60);
    });
  });

  describe("styles.css draggable and voice activation regions", () => {
    it("configures drag regions for mini widget and no-drag for buttons", () => {
      expect(stylesCss).toContain(".mini-widget-body");
      expect(stylesCss).toContain("-webkit-app-region: drag;");
      expect(stylesCss).toContain("-webkit-app-region: no-drag;");
      expect(stylesCss).toContain("cursor: move;");
    });

    it("configures mini-vad-bar animation matching the main app vad-bar", () => {
      expect(stylesCss).toContain("align-items: flex-end;");
      expect(stylesCss).toContain(".mini-vad-bar.speaking");
      expect(stylesCss).toMatch(/\.mini-vad-bar\s*\{[^}]*transition:[^;]*0\.12s cubic-bezier/);
    });
  });

  describe("Rust lib.rs tray and window visibility coordination", () => {
    it("initializes tray with 'Start recording'", () => {
      expect(libRs).toContain('"Start recording"');
    });

    it("tracks recording state and coordinates mini window visibility when main is closed", () => {
      expect(libRs).toContain("static IS_RECORDING: AtomicBool");
      expect(libRs).toContain("fn update_mini_window_visibility");
      expect(libRs).toContain("recording && !main_visible");
    });

    it("manages TrayState and updates tray recording state dynamically", () => {
      expect(libRs).toContain("struct TrayState");
      expect(libRs).toContain("fn update_tray_recording_state");
      expect(libRs).toContain('"Stop recording"');
      expect(libRs).toContain('"Start recording"');
      expect(libRs).toContain("update_tray_recording_state");
    });

    it("unminimizes and focuses window in show_main_window and show_mini_window", () => {
      expect(libRs).toMatch(/fn show_mini_window[\s\S]*?unminimize\(\)/);
      expect(libRs).toMatch(/fn show_main_window[\s\S]*?unminimize\(\)/);
    });
  });

  describe("main.ts event coordination and transitions", () => {
    it("attaches mouse drag handler to mini widget body", () => {
      expect(mainTs).toContain("getCurrentWindow");
      expect(mainTs).toContain("startDragging");
    });

    it("hides mini window and opens main window when clicking open button on floating widget", () => {
      expect(mainTs).toMatch(/btnMiniOpenApp[\s\S]*?invoke\(["']show_main_window["']\)[\s\S]*?invoke\(["']hide_mini_window["']\)/);
    });

    it("coordinates stop through the mini widget via show_main_window, hide_mini_window, and request_stop_recording", () => {
      expect(mainTs).toContain('emit("request_stop_recording")');
      expect(mainTs).toContain('listen("request_stop_recording"');
    });

    it("transitions directly to the finished recording in history on stop", () => {
      expect(mainTs).toMatch(/switchTab\(["']history["']\);\s*await selectSession\(finalSessionId\)/);
    });

    it("updates tray recording state when entering and leaving recording", () => {
      expect(mainTs).toContain('invoke("update_tray_recording_state", { isRecording: true })');
      expect(mainTs).toContain('invoke("update_tray_recording_state", { isRecording: false })');
    });

    it("uses unified updateVadBars animation combining mic and system audio in mini meter", () => {
      expect(mainTs).toMatch(/function updateMiniVad[\s\S]*?Math\.max\(micRms,\s*sysRms\)[\s\S]*?updateVadBars\(miniVadBars/);
    });

    it("moves user to history page when high-quality pass completes", () => {
      expect(mainTs).toMatch(/handleHqPassCompleted[\s\S]*?switchTab\(["']history["']\)/);
      expect(mainTs).toMatch(/handleHqPassCompleted[\s\S]*?selectSession\(finishedSessionId\)/);
    });
  });
});
