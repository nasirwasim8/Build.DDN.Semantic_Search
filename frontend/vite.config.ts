import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    proxy: {
      '/api': {
        target: 'http://localhost:8001',  // Updated to match new backend port
        changeOrigin: true,
      },
      '/health': {
        target: 'http://localhost:8001',  // Updated to match new backend port
        changeOrigin: true,
      },
      '/metrics': {
        target: 'http://localhost:8001',  // Updated to match new backend port
        changeOrigin: true,
      },
    },
  },
})
