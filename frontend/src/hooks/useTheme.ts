/**
 * Theme hook — reads the background preference from the settings store.
 * Background: dark | light | system (resolved to dark/light)
 */

import { useSettingsStore } from '../stores/settingsStore'
import { appPalette, type AppPalette } from '../styles/appTheme'

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

/** The app's colors as literals (styles/appTheme.ts), for canvases and anything else CSS can't reach. */
export function useAppPalette(): AppPalette {
  return appPalette(useTheme().isLight)
}

/**
 * Theme-aware colors for inline style props.
 * Use in components that rely on React inline styles instead of CSS classes.
 */
export function useThemeColors() {
  const p = useAppPalette()

  return {
    text: p.fg,
    textStrong: p.fg,
    textMid: p.fg,
    textSub: p.fg,
    // Supporting text: descriptions, hints, meta
    textMuted: p.fgDim,
    textFaint: p.fgDim,
    textDisabled: p.fgFaint,
    border: p.rule,
    borderStrong: p.ruleStrong,
    borderHover: p.fg,
    surface: 'transparent',
    surfaceHover: p.fill,
    surfaceActive: p.fillStrong,
    inputBg: 'transparent',
    deepBg: p.fill,
    spinnerTrack: p.rule,
    spinnerHead: p.fg,
    accent: p.accent,
    onAccent: p.onAccent,
    mark: p.mark,
    danger: p.danger,
    warn: p.warn,
    ok: p.ok,
    bg: p.bg,
    isLight: p.isLight,
  }
}
