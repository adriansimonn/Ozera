/**
 * The app's pages, in nav order: the NavBar's links and the choices for the default landing page.
 */

import { Cpu, Layers, Microscope, Search, Shapes, Sparkles, Zap, type LucideIcon } from 'lucide-react'
import type { UserSettings } from '../stores/settingsStore'

export interface AppPage {
  path: string
  label: string
  icon: LucideIcon
}

export const APP_PAGES: AppPage[] = [
  { path: '/', label: 'Generate', icon: Sparkles },
  { path: '/training', label: 'Custom Models', icon: Cpu },
  { path: '/patching', label: 'Activation Patching', icon: Zap },
  { path: '/analysis', label: 'Attention Analysis', icon: Search },
  { path: '/sae', label: 'SAE Analysis', icon: Layers },
  { path: '/probes', label: 'Probe Lab', icon: Microscope },
  { path: '/geometry', label: 'Geometry Lab', icon: Shapes },
]

const isPage = (path: unknown): path is string => APP_PAGES.some((p) => p.path === path)

/** The user's default landing page; Generate when it's unset or not a page. */
export function landingPath(settings: UserSettings | null): string {
  const path = settings?.ui?.default_page
  return isPage(path) ? path : '/'
}

// The landing page from this browser's last-seen settings, so a fresh visit needn't wait for them
const CACHE_KEY = 'ozera.landingPage'

export function cachedLandingPath(): string | null {
  try {
    const path = localStorage.getItem(CACHE_KEY)
    return isPage(path) ? path : null
  } catch {
    return null
  }
}

export function cacheLandingPath(path: string | null) {
  try {
    if (path === null) localStorage.removeItem(CACHE_KEY)
    else localStorage.setItem(CACHE_KEY, path)
  } catch {
    // Storage unavailable: fresh visits wait for the settings instead
  }
}
