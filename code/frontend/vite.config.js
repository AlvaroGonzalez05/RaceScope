import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    // Proxy API calls to the FastAPI backend in dev mode.
    // This avoids CORS entirely: the browser sees only :5173.
    proxy: {
      "/api": "http://localhost:8000",
      "/strategy": "http://localhost:8000",  // legacy route
      "/compare":  "http://localhost:8000",  // legacy route
    },
  },
});
