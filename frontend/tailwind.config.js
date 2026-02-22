/** @type {import('tailwindcss').Config} */
module.exports = {
  content: [
    './app/**/*.{js,ts,jsx,tsx,mdx}',
    './components/**/*.{js,ts,jsx,tsx,mdx}',
  ],
  theme: {
    extend: {
      colors: {
        dark: {
          900: '#0a0e1a',
          800: '#0f1525',
          700: '#141d35',
          600: '#1a2545',
          500: '#1e2d52',
        },
        brand: {
          blue: '#4a9eff',
          green: '#00d26a',
          red: '#ff4b4b',
          yellow: '#f6ad55',
          purple: '#9f7aea',
        },
      },
      fontFamily: {
        sans: ['Inter', 'system-ui', 'sans-serif'],
        mono: ['JetBrains Mono', 'monospace'],
      },
    },
  },
  plugins: [],
}
