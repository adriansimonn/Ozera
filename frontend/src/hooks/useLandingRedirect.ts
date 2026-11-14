/**
 * Sends a fresh visit to "/" (opened from outside the app, with no query) to the user's default
 * landing page, once per app load. Returns true while that page isn't known yet or the redirect
 * is on its way, so "/" holds off rendering Generate.
 */

import { useEffect, useRef, useState } from 'react'
import { useLocation, useNavigate } from 'react-router-dom'
import { cachedLandingPath, landingPath } from '../lib/appPages'
import { useAuthStore } from '../stores/authStore'
import { useSettingsStore } from '../stores/settingsStore'

// How long a fresh visit with no cached landing page waits for the settings before showing Generate
const SETTINGS_WAIT_MS = 3000

export function useLandingRedirect(): boolean {
  const location = useLocation()
  const navigate = useNavigate()
  const authReady = useAuthStore((s) => s.initialized)
  const user = useAuthStore((s) => s.user)
  const settings = useSettingsStore((s) => s.settings)
  const settingsLoading = useSettingsStore((s) => s.isLoading)
  const [gaveUp, setGaveUp] = useState(false)

  // Where the app was opened. A reload of a page reached in the app has a history key, so it stays put.
  const entry = useRef<{ key: string; fresh: boolean; cached: string | null } | null>(null)
  if (entry.current === null) {
    const fresh = location.key === 'default' && location.pathname === '/' && !location.search && !location.hash
    entry.current = { key: location.key, fresh, cached: fresh ? cachedLandingPath() : null }
  }
  const onEntry = entry.current.fresh && location.key === entry.current.key

  // The target, fixed once known, so a later settings refetch can't hide or move the page
  const target = useRef<string | null>(null)
  if (onEntry && target.current === null) {
    if (entry.current.cached) target.current = entry.current.cached
    else if (authReady && (!user || !settingsLoading)) target.current = landingPath(settings)
    else if (gaveUp) target.current = '/'
  }
  const decided = target.current

  useEffect(() => {
    if (!onEntry || decided !== null) return
    const timer = setTimeout(() => setGaveUp(true), SETTINGS_WAIT_MS)
    return () => clearTimeout(timer)
  }, [onEntry, decided])

  useEffect(() => {
    if (onEntry && decided !== null && decided !== '/') navigate(decided, { replace: true })
  }, [onEntry, decided, navigate])

  return onEntry && decided !== '/'
}
