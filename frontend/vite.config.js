import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
// Dev: proxy API calls to FastAPI. Prod: `npm run build` -> dist/, served by FastAPI on the same origin.
const api = "http://127.0.0.1:8000";
export default defineConfig({
  plugins: [react()],
  server: { proxy: { "/predict": api, "/model-info": api, "/health": api } },
});
