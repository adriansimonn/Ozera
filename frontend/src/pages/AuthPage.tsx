/**
 * The sign-in page, at "/auth": the form to the left, an animation over the rest, as the
 * homepage's hero is laid out. One page for signing in and for creating an account.
 */

import { useLayoutEffect, useState, type CSSProperties, type FormEvent, type MouseEvent } from 'react'
import { Eye, EyeOff } from 'lucide-react'
import { useLocation, useNavigate } from 'react-router-dom'
import { ReliefMark } from '../components/auth/ReliefMark'
import { THEME, readableOn } from '../components/home/homeTheme'
import { landingPath } from '../lib/appPages'
import { useAuthStore } from '../stores/authStore'
import { useSettingsStore } from '../stores/settingsStore'
import { appPalette } from '../styles/appTheme'
import './AuthPage.css'

type AuthMode = 'login' | 'signup'

// The page is dark, as the homepage's hero is, whatever the app's background is set to
const PALETTE = appPalette(false)

// The themes' colors, as the custom properties AuthPage.css reads
const THEME_VARS = {
  '--ink': THEME.ink,
  '--accent': THEME.accent,
  '--on-accent': readableOn(THEME.accent),
  '--danger': PALETTE.danger,
  '--ok': PALETTE.ok,
} as CSSProperties

const COPY = {
  login: { title: 'Sign in.', submit: 'Sign in', working: 'Signing in', other: 'No account?', switchTo: 'Sign up' },
  signup: { title: 'Create account.', submit: 'Create account', working: 'Creating account', other: 'Have an account?', switchTo: 'Sign in' },
}

const messageOf = (err: unknown) => (err instanceof Error ? err.message : String(err))

interface PasswordFieldProps {
  label: string
  hint?: string
  value: string
  onChange: (value: string) => void
  autoComplete: string
  disabled: boolean
}

/** A password field with a button that shows what's typed. */
function PasswordField({ label, hint, value, onChange, autoComplete, disabled }: PasswordFieldProps) {
  const [shown, setShown] = useState(false)
  return (
    <label className="oza-field">
      <span className="oza-label">
        {label}
        {hint && <small>{hint}</small>}
      </span>
      <span className="oza-control">
        <input
          className="oza-input oza-input--reveal"
          type={shown ? 'text' : 'password'}
          value={value}
          onChange={(e) => onChange(e.target.value)}
          required
          minLength={8}
          autoComplete={autoComplete}
          disabled={disabled}
        />
        <button
          type="button"
          className="oza-reveal"
          aria-label={shown ? `Hide ${label.toLowerCase()}` : `Show ${label.toLowerCase()}`}
          aria-pressed={shown}
          onClick={() => setShown(!shown)}
        >
          {shown ? <Eye strokeWidth={2} /> : <EyeOff strokeWidth={2} />}
        </button>
      </span>
    </label>
  )
}

export default function AuthPage() {
  const navigate = useNavigate()
  const location = useLocation()
  const login = useAuthStore((s) => s.login)
  const signup = useAuthStore((s) => s.signup)
  const [mode, setMode] = useState<AuthMode>('login')

  const [email, setEmail] = useState('')
  const [password, setPassword] = useState('')
  const [fullName, setFullName] = useState('')
  const [confirmPassword, setConfirmPassword] = useState('')
  const [error, setError] = useState<string | null>(null)
  const [checkEmail, setCheckEmail] = useState(false)
  const [busy, setBusy] = useState(false)

  // The app is laid out under a page zoom (index.css); this page is drawn at true size
  useLayoutEffect(() => {
    const root = document.documentElement
    root.classList.add('oz-auth-active')
    return () => root.classList.remove('oz-auth-active')
  }, [])

  // Back to the page that sent the user here; opened directly or from the homepage, on to their
  // landing page
  const leave = () => {
    if (location.key === 'default' || location.state?.fromHome) navigate(landingPath(useSettingsStore.getState().settings), { replace: true })
    else navigate(-1)
  }

  const switchMode = (newMode: AuthMode) => {
    setEmail('')
    setPassword('')
    setConfirmPassword('')
    setFullName('')
    setError(null)
    setCheckEmail(false)
    setMode(newMode)
  }

  const handleLogin = async () => {
    try {
      await login(email, password)
      leave()
    } catch (err) {
      setError(messageOf(err))
    }
  }

  const handleSignup = async () => {
    if (password !== confirmPassword) {
      setError('Passwords do not match')
      return
    }
    if (password.length < 8) {
      setError('Password must be at least 8 characters')
      return
    }
    try {
      await signup(email, password, fullName || undefined)
      leave()
    } catch (err) {
      if (messageOf(err) === 'CHECK_EMAIL') setCheckEmail(true)
      else setError(messageOf(err))
    }
  }

  const submit = async (e: FormEvent) => {
    e.preventDefault()
    if (busy) return
    setError(null)
    setBusy(true)
    try {
      await (mode === 'login' ? handleLogin() : handleSignup())
    } finally {
      setBusy(false)
    }
  }

  const home = (e: MouseEvent) => {
    if (e.metaKey || e.ctrlKey || e.shiftKey || e.button !== 0) return
    e.preventDefault()
    navigate('/')
  }

  const copy = COPY[mode]

  return (
    <div className="oz-auth" style={THEME_VARS}>
      <header className="oza-nav">
        <a className="oza-brand" href="/" onClick={home}>
          <img src="/LogoTransparentWhiteText.png" alt="Ozera" />
        </a>
      </header>

      <main className="oza-panel">
        <div className="oza-body">
          <h1 className="oza-title">
            <span className="oza-line" key={mode}>
              <span>{copy.title}</span>
            </span>
          </h1>

          <div className="oza-fade" key={mode}>
            {error && (
              <p className="oza-note oza-note--error" role="alert">
                {error}
              </p>
            )}
            {checkEmail && (
              <p className="oza-note oza-note--ok" role="status">
                Check your email for a confirmation link to finish signing up.
              </p>
            )}

            <form className="oza-form" onSubmit={submit}>
              {mode === 'signup' && (
                <label className="oza-field">
                  <span className="oza-label">
                    Name
                    <small>Optional</small>
                  </span>
                  <input
                    className="oza-input"
                    type="text"
                    value={fullName}
                    onChange={(e) => setFullName(e.target.value)}
                    autoComplete="name"
                    autoFocus
                    disabled={busy}
                  />
                </label>
              )}
              <label className="oza-field">
                <span className="oza-label">Email</span>
                <input
                  className="oza-input"
                  type="email"
                  value={email}
                  onChange={(e) => setEmail(e.target.value)}
                  required
                  placeholder="you@example.com"
                  autoComplete="email"
                  autoFocus={mode === 'login'}
                  disabled={busy}
                />
              </label>
              <PasswordField
                label="Password"
                hint={mode === 'signup' ? '8 characters or more' : undefined}
                value={password}
                onChange={setPassword}
                autoComplete={mode === 'login' ? 'current-password' : 'new-password'}
                disabled={busy}
              />
              {mode === 'signup' && (
                <PasswordField
                  label="Confirm password"
                  value={confirmPassword}
                  onChange={setConfirmPassword}
                  autoComplete="new-password"
                  disabled={busy}
                />
              )}
              <button type="submit" className="oza-btn" disabled={busy}>
                {busy ? copy.working : copy.submit}
                {busy ? (
                  <span className="oza-cells" aria-hidden="true">
                    <i />
                    <i />
                    <i />
                  </span>
                ) : (
                  <span className="oza-btn__arrow">→</span>
                )}
              </button>
            </form>

            <p className="oza-switch">
              {copy.other}
              <button type="button" className="oza-link" onClick={() => switchMode(mode === 'login' ? 'signup' : 'login')} disabled={busy}>
                {copy.switchTo}
              </button>
            </p>
          </div>
        </div>
      </main>

      <ReliefMark className="oza-canvas" busy={busy} />
    </div>
  )
}
