/**
 * Theme hook — reads the background preference from the settings store.
 * Background: dark | light | system (resolved to dark/light)
 */

import { useSettingsStore } from '../stores/settingsStore'

export type EffectiveBg = 'dark' | 'light'

export function useTheme() {
  const settings = useSettingsStore((s) => s.settings)

  const rawBg = settings?.ui?.background ?? 'system'

  const background: EffectiveBg =
    rawBg === 'light' || (rawBg === 'system' && !window.matchMedia('(prefers-color-scheme: dark)').matches)
      ? 'light'
      : 'dark'

  const isLight = background === 'light'

  return {
    background,
    isLight,
  }
}

/**
 * Theme-aware colors for inline style props.
 * Use in components that rely on React inline styles instead of CSS classes.
 */
export function useThemeColors() {
  const { isLight } = useTheme()

  // All readable text tiers share one full-contrast color; hierarchy comes from size/weight.
  const text = isLight ? '#1d1d1f' : '#ffffff'

  return {
    text,
    textStrong: text,
    textMid: text,
    textSub: text,
    textMuted: text,
    textFaint: text,
    textDisabled: isLight ? 'rgba(0,0,0,0.4)' : 'rgba(255,255,255,0.4)',
    border: isLight ? 'rgba(0,0,0,0.1)' : 'rgba(255,255,255,0.1)',
    borderStrong: isLight ? 'rgba(0,0,0,0.15)' : 'rgba(255,255,255,0.15)',
    borderHover: isLight ? 'rgba(0,0,0,0.2)' : 'rgba(255,255,255,0.25)',
    surface: isLight ? 'rgba(0,0,0,0.03)' : 'rgba(255,255,255,0.03)',
    surfaceHover: isLight ? 'rgba(0,0,0,0.05)' : 'rgba(255,255,255,0.05)',
    surfaceActive: isLight ? 'rgba(0,0,0,0.08)' : 'rgba(255,255,255,0.1)',
    inputBg: isLight ? 'rgba(0,0,0,0.03)' : 'rgba(0,0,0,0.4)',
    deepBg: isLight ? 'rgba(0,0,0,0.04)' : 'rgba(0,0,0,0.2)',
    spinnerTrack: isLight ? 'rgba(0,0,0,0.15)' : 'rgba(255,255,255,0.3)',
    spinnerHead: isLight ? '#1d1d1f' : '#fff',
    isLight,
  }
}
