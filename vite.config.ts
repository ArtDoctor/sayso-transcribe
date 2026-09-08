import { defineConfig } from "vite";

// @ts-expect-error process is a nodejs global
const host = process.env.TAURI_DEV_HOST;

export default defineConfig(async () => ({
  clearScreen: false,
  server: {
    port: 41765,
    strictPort: true,
    host: host || false,
    hmr: host
      ? {
          protocol: "ws",
          host,
          port: 41766,
        }
      : undefined,
    watch: {
      ignored: ["**/src-tauri/**", "**/recordings/**", "**/backend/**"],
    },
    proxy: {
      "/api": {
        target: "http://127.0.0.1:48653",
        changeOrigin: true,
      },
      "/ws": {
        target: "ws://127.0.0.1:48653",
        ws: true,
      },
    },
  },
  preview: {
    port: 41767,
    strictPort: true,
    host: "127.0.0.1",
  },
}));
