/**
 * Theme hook — reads background and interface preferences from settings store.
 * Background: dark | light | glow | system (resolved to dark/light/glow)
 * Interface: default | glass
 */

import { useSettingsStore } from '../stores/settingsStore'

export type EffectiveBg = 'dark' | 'light' | 'glow'
export type InterfaceStyle = 'default' | 'glass'

export function useTheme() {
  const settings = useSettingsStore((s) => s.settings)

  const rawBg = settings?.ui?.background ?? 'glow'
  const interfaceStyle: InterfaceStyle = settings?.ui?.interface ?? 'glass'

  const background: EffectiveBg =
    rawBg === 'system'
      ? window.matchMedia('(prefers-color-scheme: dark)').matches
        ? 'dark'
        : 'light'
      : rawBg

  const isLight = background === 'light'

  return {
    background,
    interfaceStyle,
    isGlow: background === 'glow',
    isGlass: interfaceStyle === 'glass',
    isDefault: interfaceStyle === 'default',
    isLight,
  }
}

/**
 * Theme-aware colors for inline style props.
 * Use in components that rely on React inline styles instead of CSS classes.
 */
export function useThemeColors() {
  const { isLight } = useTheme()

  return {
    text: isLight ? '#1d1d1f' : '#fff',
    textStrong: isLight ? '#1d1d1f' : 'rgba(255,255,255,0.95)',
    textMid: isLight ? 'rgba(0,0,0,0.6)' : 'rgba(255,255,255,0.7)',
    textSub: isLight ? 'rgba(0,0,0,0.55)' : 'rgba(255,255,255,0.5)',
    textMuted: isLight ? 'rgba(0,0,0,0.5)' : 'rgba(255,255,255,0.4)',
    textFaint: isLight ? 'rgba(0,0,0,0.4)' : 'rgba(255,255,255,0.35)',
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
