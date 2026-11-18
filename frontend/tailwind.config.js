/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        // The app's palette (src/styles/appTheme.ts), by its custom properties
        ink: 'var(--oz-bg)',
        fg: 'var(--oz-fg)',
        dim: 'var(--oz-fg-dim)',
        faint: 'var(--oz-fg-faint)',
        rule: 'var(--oz-rule)',
        'rule-strong': 'var(--oz-rule-strong)',
        fill: 'var(--oz-fill)',
        'fill-strong': 'var(--oz-fill-strong)',
        volt: 'var(--oz-accent)',
        'on-volt': 'var(--oz-on-accent)',
        mark: 'var(--oz-mark)',
        danger: 'var(--oz-danger)',
        warn: 'var(--oz-warn)',
        ok: 'var(--oz-ok)',
      },
      fontFamily: {
        sans: ['Inter', 'system-ui', 'sans-serif'],
        mono: ['JetBrains Mono', 'Fira Code', 'monospace'],
      },
    },
  },
  plugins: [],
}
