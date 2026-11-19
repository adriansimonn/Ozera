/**
 * The app's pages, in nav order: the NavBar's links and the choices for the default landing page.
 */

import { Cpu, Layers, Microscope, Shapes, Sparkles, Zap, type LucideIcon } from 'lucide-react'
import type { UserSettings } from '../stores/settingsStore'

export interface AppPage {
  path: string
  label: string
  icon: LucideIcon
}

export const APP_PAGES: AppPage[] = [
  { path: '/generate', label: 'Generate', icon: Sparkles },
  { path: '/training', label: 'Custom Models', icon: Cpu },
  { path: '/attribution', label: 'Attribution', icon: Zap },
  { path: '/sae', label: 'SAE Analysis', icon: Layers },
  { path: '/probes', label: 'Probe Lab', icon: Microscope },
  { path: '/geometry', label: 'Geometry Lab', icon: Shapes },
]

const isPage = (path: unknown): path is string => APP_PAGES.some((p) => p.path === path)

// The landing page from this browser's last-seen settings, for when they haven't loaded yet
const CACHE_KEY = 'ozera.landingPage'

function cachedLandingPath(): string | null {
  try {
    const path = localStorage.getItem(CACHE_KEY)
    return isPage(path) ? path : null
  } catch {
    return null
  }
}

/**
 * Where "into the app" leads: the user's default landing page, or Generate when it's unset or
 * not a page (as "/" is, which Generate's path used to be). Before the settings have loaded,
 * the page they named last time in this browser.
 */
export function landingPath(settings: UserSettings | null): string {
  const path = settings ? settings.ui?.default_page : cachedLandingPath()
  return isPage(path) ? path : '/generate'
}

export function cacheLandingPath(path: string | null) {
  try {
    if (path === null) localStorage.removeItem(CACHE_KEY)
    else localStorage.setItem(CACHE_KEY, path)
  } catch {
    // Storage unavailable: the landing page is Generate until the settings have loaded
  }
}
