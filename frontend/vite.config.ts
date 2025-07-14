import { defineConfig } from 'vite'
import preact from '@preact/preset-vite'

export default defineConfig({
  plugins: [preact()],
  server: {
    host: '0.0.0.0',
    port: 3000,
    allowedHosts: [
      'localhost',
      'sanskrit-parser.bondarenko-nikita.ru'
    ],
    proxy: {
      '/api': {
        target: 'http://sanskrit-parser-backend:8000',
        changeOrigin: true,
        secure: false
      }
    }
  }
}) 