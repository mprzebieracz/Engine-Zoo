import { sveltekit } from '@sveltejs/kit/vite';
import { defineConfig } from 'vite';

export default defineConfig({
  plugins: [sveltekit()],
  server: {
    proxy: {
      '/api': {
        target: process.env.ENGINE_ZOO_API ?? 'http://127.0.0.1:8080',
        changeOrigin: true
      }
    }
  }
});
