import { useState, useRef, useEffect, useCallback } from 'react'
import authVideo from '../assets/videos/AuthAnimation.mp4'
import { useNavigate } from 'react-router-dom'
import { useAuthStore } from '../stores/authStore'

type AuthMode = 'login' | 'signup'

const GRID_SPACING = 28
const BASE_DOT_SIZE = 1.2
const MAX_ARROW_LEN = 16
const INFLUENCE_RADIUS = 160

export default function AuthPage() {
  const navigate = useNavigate()
  const { login, signup, isLoading } = useAuthStore()
  const [mode, setMode] = useState<AuthMode>('login')
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const mouseRef = useRef({ x: -1000, y: -1000 })
  const rafRef = useRef<number>(0)
  const loadingRef = useRef(false)
  const timeRef = useRef(0)

  const [email, setEmail] = useState('')
  const [password, setPassword] = useState('')
  const [error, setError] = useState<string | null>(null)
  const [fullName, setFullName] = useState('')
  const [confirmPassword, setConfirmPassword] = useState('')
  const [checkEmail, setCheckEmail] = useState(false)
  const [showPassword, setShowPassword] = useState(false)
  const [showConfirmPassword, setShowConfirmPassword] = useState(false)

  const EyeToggle = ({ visible, onToggle }: { visible: boolean; onToggle: () => void }) => (
    <button
      type="button"
      onClick={onToggle}
      style={{
        position: 'absolute',
        right: '0.75rem',
        top: '50%',
        transform: 'translateY(-50%)',
        background: 'none',
        border: 'none',
        cursor: 'pointer',
        padding: '0.2rem',
        display: 'flex',
        alignItems: 'center',
        color: '#ffffff',
      }}
    >
      {visible ? (
        <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round">
          <path d="M1 12s4-8 11-8 11 8 11 8-4 8-11 8-11-8-11-8z" />
          <circle cx="12" cy="12" r="3" />
        </svg>
      ) : (
        <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round">
          <path d="M17.94 17.94A10.07 10.07 0 0 1 12 20c-7 0-11-8-11-8a18.45 18.45 0 0 1 5.06-5.94" />
          <path d="M9.9 4.24A9.12 9.12 0 0 1 12 4c7 0 11 8 11 8a18.5 18.5 0 0 1-2.16 3.19" />
          <path d="M14.12 14.12a3 3 0 1 1-4.24-4.24" />
          <line x1="1" y1="1" x2="23" y2="23" />
        </svg>
      )}
    </button>
  )

  const resetForm = () => {
    setEmail('')
    setPassword('')
    setConfirmPassword('')
    setFullName('')
    setError(null)
    setCheckEmail(false)
  }

  const switchMode = (newMode: AuthMode) => {
    resetForm()
    setMode(newMode)
  }

  const handleLogin = async (e: React.FormEvent) => {
    e.preventDefault()
    setError(null)
    try {
      await login(email, password)
      navigate(-1)
    } catch (err: any) {
      setError(err.message)
    }
  }

  const handleSignup = async (e: React.FormEvent) => {
    e.preventDefault()
    setError(null)
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
      navigate(-1)
    } catch (err: any) {
      if (err.message === 'CHECK_EMAIL') {
        setCheckEmail(true)
      } else {
        setError(err.message)
      }
    }
  }

  useEffect(() => {
    loadingRef.current = isLoading
  }, [isLoading])

  const drawField = useCallback(() => {
    const canvas = canvasRef.current
    if (!canvas) return
    const ctx = canvas.getContext('2d')
    if (!ctx) return

    const rect = canvas.getBoundingClientRect()
    const w = rect.width
    const h = rect.height
    const dpr = window.devicePixelRatio || 1
    canvas.width = w * dpr
    canvas.height = h * dpr
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0)

    ctx.clearRect(0, 0, w, h)

    timeRef.current += 0.015
    const t = timeRef.current
    const loading = loadingRef.current

    // Build list of attractors: mouse + orbiting ones when loading
    const attractors: { x: number; y: number; strength: number }[] = []

    if (!loading) {
      attractors.push({ x: mouseRef.current.x, y: mouseRef.current.y, strength: 1 })
    } else {
      // Attractors that sweep across the full screen edges and corners
      const attractorPaths = [
        // Top edge, sweeping left to right
        { x: w * (0.5 + 0.55 * Math.cos(t * 0.7)), y: h * 0.05, strength: 0.8 },
        // Bottom edge, sweeping right to left
        { x: w * (0.5 + 0.55 * Math.cos(t * 0.7 + Math.PI)), y: h * 0.95, strength: 0.8 },
        // Left edge, sweeping up and down
        { x: w * 0.02, y: h * (0.5 + 0.45 * Math.sin(t * 0.9)), strength: 0.9 },
        // Right edge, sweeping up and down (opposite phase)
        { x: w * 0.98, y: h * (0.5 + 0.45 * Math.sin(t * 0.9 + Math.PI)), strength: 0.9 },
        // Corner orbits
        { x: w * (0.1 + 0.08 * Math.cos(t * 1.3)), y: h * (0.1 + 0.08 * Math.sin(t * 1.3)), strength: 0.7 },
        { x: w * (0.9 + 0.08 * Math.cos(t * 1.1 + 1)), y: h * (0.9 + 0.08 * Math.sin(t * 1.1 + 1)), strength: 0.7 },
      ]
      for (const a of attractorPaths) {
        attractors.push(a)
      }
    }

    for (let x = GRID_SPACING / 2; x < w; x += GRID_SPACING) {
      for (let y = GRID_SPACING / 2; y < h; y += GRID_SPACING) {
        // Accumulate influence from all attractors
        let totalDx = 0
        let totalDy = 0
        let maxInfluence = 0

        for (const a of attractors) {
          const dx = a.x - x
          const dy = a.y - y
          const dist = Math.sqrt(dx * dx + dy * dy)
          const influence = Math.max(0, 1 - dist / INFLUENCE_RADIUS) * a.strength
          const eased = influence * influence
          if (eased > 0.01) {
            const angle = Math.atan2(dy, dx)
            totalDx += Math.cos(angle) * eased
            totalDy += Math.sin(angle) * eased
            maxInfluence = Math.max(maxInfluence, eased)
          }
        }

        if (maxInfluence > 0.01) {
          const angle = Math.atan2(totalDy, totalDx)
          const magnitude = Math.sqrt(totalDx * totalDx + totalDy * totalDy)
          const len = Math.min(magnitude, 1) * MAX_ARROW_LEN
          const ex = x + Math.cos(angle) * len
          const ey = y + Math.sin(angle) * len

          ctx.beginPath()
          ctx.moveTo(x, y)
          ctx.lineTo(ex, ey)
          ctx.strokeStyle = `rgba(255, 255, 255, ${0.25 + maxInfluence * 0.5})`
          ctx.lineWidth = 1 + maxInfluence * 0.8
          ctx.stroke()

          const headLen = 2.5 + maxInfluence * 2
          ctx.beginPath()
          ctx.moveTo(ex, ey)
          ctx.lineTo(
            ex - Math.cos(angle - 0.45) * headLen,
            ey - Math.sin(angle - 0.45) * headLen,
          )
          ctx.moveTo(ex, ey)
          ctx.lineTo(
            ex - Math.cos(angle + 0.45) * headLen,
            ey - Math.sin(angle + 0.45) * headLen,
          )
          ctx.strokeStyle = `rgba(255, 255, 255, ${0.2 + maxInfluence * 0.45})`
          ctx.lineWidth = 0.8 + maxInfluence * 0.5
          ctx.stroke()
        } else {
          ctx.beginPath()
          ctx.arc(x, y, BASE_DOT_SIZE, 0, Math.PI * 2)
          ctx.fillStyle = 'rgba(255, 255, 255, 0.25)'
          ctx.fill()
        }
      }
    }

    rafRef.current = requestAnimationFrame(drawField)
  }, [])

  useEffect(() => {
    const handleMouseMove = (e: MouseEvent) => {
      mouseRef.current = { x: e.clientX, y: e.clientY }
    }
    window.addEventListener('mousemove', handleMouseMove)
    rafRef.current = requestAnimationFrame(drawField)
    return () => {
      window.removeEventListener('mousemove', handleMouseMove)
      cancelAnimationFrame(rafRef.current)
    }
  }, [drawField])

  return (
    <div style={{
      position: 'fixed',
      inset: 0,
      background: '#1a1a1a',
      display: 'flex',
      alignItems: 'center',
      justifyContent: 'center',
      overflow: 'hidden',
      zIndex: 10,
    }}>
      {/* Vector field canvas */}
      <canvas
        ref={canvasRef}
        style={{
          position: 'absolute',
          inset: 0,
          width: '100%',
          height: '100%',
          pointerEvents: 'none',
        }}
      />

      {/* Main card */}
      <div style={{
        position: 'relative',
        zIndex: 2,
        display: 'flex',
        width: '1056px',
        maxWidth: '92vw',
        minHeight: '624px',
        background: 'rgba(0, 0, 0, 0.7)',
        backdropFilter: 'blur(32px)',
        border: '1px solid rgba(255, 255, 255, 0.08)',
        borderRadius: '0px',
        overflow: 'hidden',
      }}>
        {/* Left half — auth forms */}
        <div style={{
          width: '50%',
          flexShrink: 0,
          padding: '3rem',
          display: 'flex',
          flexDirection: 'column',
        }}>
          {/* Logo */}
          <img
            src="/LogoTransparentWhiteText.png"
            alt="Ozera"
            style={{ height: '38px', width: 'auto', marginBottom: '2rem', alignSelf: 'flex-start' }}
          />

          <h1 style={{
            fontSize: '1.7rem',
            fontWeight: 600,
            color: '#fff',
            marginBottom: '0.4rem',
          }}>
            {mode === 'login' ? 'Log In' : 'Sign Up'}
          </h1>
          <p style={{
            fontSize: '1rem',
            color: '#ffffff',
            marginBottom: '2.1rem',
          }}>
            {mode === 'login' ? 'Sign in to your account' : 'Get started with Ozera'}
          </p>

          {error && (
            <div style={{
              marginBottom: '1.25rem',
              padding: '0.75rem 1rem',
              background: 'rgba(239, 68, 68, 0.1)',
              border: '1px solid rgba(239, 68, 68, 0.25)',
              borderRadius: '0px',
              color: '#f87171',
              fontSize: '1rem',
            }}>
              {error}
            </div>
          )}

          {checkEmail && (
            <div style={{
              marginBottom: '1.25rem',
              padding: '0.75rem 1rem',
              background: 'rgba(16, 185, 129, 0.1)',
              border: '1px solid rgba(16, 185, 129, 0.25)',
              borderRadius: '0px',
              color: '#34d399',
              fontSize: '1rem',
            }}>
              Check your email for a confirmation link to complete signup.
            </div>
          )}

          {/* Login Form */}
          {mode === 'login' && (
            <form onSubmit={handleLogin} style={{ display: 'flex', flexDirection: 'column', gap: '1.3rem', flex: 1 }}>
              <div>
                <label style={labelStyle}>Email</label>
                <input
                  type="email"
                  value={email}
                  onChange={(e) => setEmail(e.target.value)}
                  required
                  style={inputStyle}
                  placeholder="you@example.com"
                  disabled={isLoading}
                />
              </div>
              <div>
                <label style={labelStyle}>Password</label>
                <div style={{ position: 'relative' }}>
                  <input
                    type={showPassword ? 'text' : 'password'}
                    value={password}
                    onChange={(e) => setPassword(e.target.value)}
                    required
                    minLength={8}
                    style={{ ...inputStyle, paddingRight: '2.5rem' }}
                    placeholder="••••••••"
                    disabled={isLoading}
                  />
                  <EyeToggle visible={showPassword} onToggle={() => setShowPassword(!showPassword)} />
                </div>
              </div>
              <div style={{ flex: 1 }} />
              <button type="submit" disabled={isLoading} style={submitStyle}>
                {isLoading ? 'Logging in...' : 'Login'}
              </button>
              <div style={switchTextStyle}>
                Don't have an account?{' '}
                <button type="button" onClick={() => switchMode('signup')} style={switchBtnStyle}>
                  Sign up
                </button>
              </div>
            </form>
          )}

          {/* Signup Form */}
          {mode === 'signup' && (
            <form onSubmit={handleSignup} style={{ display: 'flex', flexDirection: 'column', gap: '1.3rem', flex: 1 }}>
              <div>
                <label style={labelStyle}>
                  Full Name <span style={{ color: '#ffffff' }}>(optional)</span>
                </label>
                <input
                  type="text"
                  value={fullName}
                  onChange={(e) => setFullName(e.target.value)}
                  style={inputStyle}
                  placeholder="John Doe"
                  disabled={isLoading}
                />
              </div>
              <div>
                <label style={labelStyle}>Email</label>
                <input
                  type="email"
                  value={email}
                  onChange={(e) => setEmail(e.target.value)}
                  required
                  style={inputStyle}
                  placeholder="you@example.com"
                  disabled={isLoading}
                />
              </div>
              <div>
                <label style={labelStyle}>Password</label>
                <div style={{ position: 'relative' }}>
                  <input
                    type={showPassword ? 'text' : 'password'}
                    value={password}
                    onChange={(e) => setPassword(e.target.value)}
                    required
                    minLength={8}
                    style={{ ...inputStyle, paddingRight: '2.5rem' }}
                    placeholder="••••••••"
                    disabled={isLoading}
                  />
                  <EyeToggle visible={showPassword} onToggle={() => setShowPassword(!showPassword)} />
                </div>
                <p style={{ marginTop: '0.4rem', fontSize: '0.84rem', color: '#ffffff' }}>
                  Minimum 8 characters
                </p>
              </div>
              <div>
                <label style={labelStyle}>Confirm Password</label>
                <div style={{ position: 'relative' }}>
                  <input
                    type={showConfirmPassword ? 'text' : 'password'}
                    value={confirmPassword}
                    onChange={(e) => setConfirmPassword(e.target.value)}
                    required
                    minLength={8}
                    style={{ ...inputStyle, paddingRight: '2.5rem' }}
                    placeholder="••••••••"
                    disabled={isLoading}
                  />
                  <EyeToggle visible={showConfirmPassword} onToggle={() => setShowConfirmPassword(!showConfirmPassword)} />
                </div>
              </div>
              <div style={{ flex: 1 }} />
              <button type="submit" disabled={isLoading} style={submitStyle}>
                {isLoading ? 'Creating account...' : 'Sign Up'}
              </button>
              <div style={switchTextStyle}>
                Already have an account?{' '}
                <button type="button" onClick={() => switchMode('login')} style={switchBtnStyle}>
                  Login
                </button>
              </div>
            </form>
          )}
        </div>

        {/* Right half — video placeholder */}
        <div style={{
          width: '50%',
          flexShrink: 0,
          background: '#111',
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
          overflow: 'hidden',
        }}>
          <video
            src={authVideo}
            autoPlay
            muted
            loop
            playsInline
            style={{
              width: '100%',
              height: '100%',
              objectFit: 'cover',
              display: 'block',
            }}
          />
        </div>
      </div>
    </div>
  )
}

const labelStyle: React.CSSProperties = {
  display: 'block',
  fontSize: '0.84rem',
  fontWeight: 500,
  color: '#ffffff',
  marginBottom: '0.5rem',
  textTransform: 'uppercase',
  letterSpacing: '0.06em',
}

const inputStyle: React.CSSProperties = {
  width: '100%',
  padding: '0.84rem 1.08rem',
  background: 'rgba(255, 255, 255, 0.05)',
  border: '1px solid rgba(255, 255, 255, 0.12)',
  borderRadius: '0px',
  color: '#ffffff',
  fontSize: '1rem',
  outline: 'none',
  transition: 'border-color 0.2s',
  boxSizing: 'border-box',
}

const submitStyle: React.CSSProperties = {
  width: '100%',
  padding: '0.84rem 1.2rem',
  background: 'rgba(255, 255, 255, 0.1)',
  border: '1px solid rgba(255, 255, 255, 0.12)',
  borderRadius: '0px',
  color: '#ffffff',
  fontSize: '1rem',
  fontWeight: 500,
  cursor: 'pointer',
  transition: 'all 0.2s',
}

const switchTextStyle: React.CSSProperties = {
  textAlign: 'center',
  fontSize: '1rem',
  color: '#ffffff',
}

const switchBtnStyle: React.CSSProperties = {
  background: 'none',
  border: 'none',
  color: '#ffffff',
  fontWeight: 500,
  cursor: 'pointer',
  transition: 'color 0.2s',
  fontSize: '1rem',
}
