import { defineConfig } from 'vite';
import { svelte } from '@sveltejs/vite-plugin-svelte';

const api = process.env.MPN_API ?? 'http://127.0.0.1:8840';

export default defineConfig({
  base: './',
  plugins: [svelte()],
  build: { outDir: 'dist', emptyOutDir: true, target: 'es2022', sourcemap: false },
  server: {
    host: '127.0.0.1',
    port: Number(process.env.MPN_WEB_PORT ?? 5174),
    strictPort: true,
    // Host and Origin are passed through unchanged: the API refuses writes whose Origin differs from Host.
    proxy: { '/api': { target: api, changeOrigin: false }, '/v1': { target: api, changeOrigin: false } },
  },
});
