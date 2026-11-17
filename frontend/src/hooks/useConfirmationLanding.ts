/**
 * "/" is the homepage, with one exception: the link in a sign-up confirmation email. Supabase
 * sends that to the site's root with the new session in the address, and such a visit goes on
 * into the app once the session is in. Returns true while that's pending, so "/" holds off
 * rendering the homepage.
 */

import { useEffect, useState } from 'react'
import { useNavigate } from 'react-router-dom'
import { landingPath } from '../lib/appPages'
import { useAuthStore } from '../stores/authStore'
import { useSettingsStore } from '../stores/settingsStore'

// Read as the app loads: Supabase clears the session from the address once it has taken it
const OPENED_BY_CONFIRMATION =
  window.location.pathname === '/' && /[#?&](access_token|code)=/.test(window.location.hash + window.location.search)

export function useConfirmationLanding(): boolean {
  const navigate = useNavigate()
  const authReady = useAuthStore((s) => s.initialized)
  const user = useAuthStore((s) => s.user)
  const [pending, setPending] = useState(OPENED_BY_CONFIRMATION)

  useEffect(() => {
    if (!pending || !authReady) return
    setPending(false)
    // A link that has expired signs nobody in: that visit stays on the homepage
    if (user) navigate(landingPath(useSettingsStore.getState().settings), { replace: true })
  }, [pending, authReady, user, navigate])

  return pending
}
