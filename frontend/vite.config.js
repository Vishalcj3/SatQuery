import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// base "./" keeps asset paths relative so the build also works when served
// from a subfolder or opened via a simple static server.
export default defineConfig({
  base: "./",
  plugins: [react()],
  server: { port: 5173, host: "127.0.0.1" },
});
