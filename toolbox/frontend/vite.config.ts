import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

const apiTarget = process.env.VITE_API_PROXY_TARGET ?? "http://127.0.0.1:45700";

export default defineConfig({
  base: "/ui/",
  plugins: [react()],
  server: {
    proxy: {
      "/ui/api": {
        target: apiTarget,
        changeOrigin: true,
      },
      "/health": {
        target: apiTarget,
        changeOrigin: true,
      },
      "/object-search": {
        target: apiTarget,
        changeOrigin: true,
      },
      "^/[^/]+/object-search": {
        target: apiTarget,
        changeOrigin: true,
      },
    },
  },
});
