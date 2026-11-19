/**
 * The homepage, at "/" for everyone: a signed-in visitor gets their account in the nav.
 * Four sections: the hero, the seven instruments, what the app offers around them, and the models.
 */

import { useEffect, useLayoutEffect, useRef, useState, type CSSProperties, type MouseEvent } from 'react'
import { useNavigate } from 'react-router-dom'
import { AccountMenu } from '../components/home/AccountMenu'
import { CellGlyph } from '../components/home/CellGlyph'
import { FlowMosaic } from '../components/home/FlowMosaic'
import { InstrumentFigure } from '../components/home/InstrumentFigure'
import { TokenField } from '../components/home/TokenField'
import { THEME, readableOn } from '../components/home/homeTheme'
import { landingPath } from '../lib/appPages'
import { useAuthStore } from '../stores/authStore'
import { useSettingsStore } from '../stores/settingsStore'
import './HomePage.css'

const GITHUB_URL = 'https://github.com/adriansimonn/Ozera'

// In the order of InstrumentFigure's figures
const TOOLS = [
  {
    name: 'Generate',
    path: '/generate',
    blurb: 'Stream a completion, then open any layer of it: activations, attention, embeddings, logits.',
  },
  {
    name: 'Activation Patching',
    path: '/patching',
    blurb: 'Patch in activations from another prompt, or ablate them, at any layer, head or position, and compare what the model predicts.',
  },
  {
    name: 'Attention Analysis',
    path: '/analysis',
    blurb: 'Every head classified (induction, copying, positional) and compared across prompts.',
  },
  {
    name: 'SAE Analysis',
    path: '/sae',
    blurb: 'Train sparse autoencoders, browse their features, and read explanations written and scored for you.',
  },
  {
    name: 'Probe Lab',
    path: '/probes',
    blurb: 'Fit a probe at every layer, check it against its controls, then test whether the model uses it.',
  },
  {
    name: 'Geometry Lab',
    path: '/geometry',
    blurb: 'Find the shape a concept takes (a circle, a line, clusters) and steer the model along it.',
  },
  {
    name: 'Custom Models',
    path: '/training',
    blurb: 'Train Ozera’s from-scratch transformer on your own text, with the cost estimated up front.',
  },
]

// What the app offers around its instruments, each under a pictogram (see CellGlyph)
const EXTRAS = [
  {
    name: 'Assistant',
    blurb: 'Ask in plain language. It sets up and runs the experiments for you, and explains what comes back.',
    glyph: ['.#######.', '#.......#', '#.1.2.3.#', '#.......#', '.#######.', '..#......', '.#.......'],
  },
  {
    name: 'Figure export',
    blurb: 'Paper-ready figures with presets for NeurIPS, ICLR and ICML, one at a time or as a batch.',
    glyph: ['....1....', '....2....', '....3....', '..#.4.#..', '...#5#...', '#...6...#', '#########'],
  },
  {
    name: 'GPUs on demand',
    blurb: 'Nothing to install. Runs go to serverless GPUs that start when you do.',
    glyph: ['..1.2..', '.#####.', '8#...#3', '.#.#.#.', '7#...#4', '.#####.', '..6.5..'],
  },
  {
    name: 'Priced up front',
    blurb: 'Pay with credits, only for what you run. Longer jobs show an estimate first.',
    glyph: ['.........', '..######.', '.#......#', '#.1.....#', '.#......#', '..######.', '.........'],
  },
]

// A row of chips each: the open models, then Ozera's own and yours
const MODELS = [
  ['Qwen3', 'Gemma 3', 'SmolLM2', 'SmolLM3'],
  ['Ozera-Nano', 'Ozera-Mini', 'Your own weights'],
]

// The theme's colors, as the custom properties HomePage.css reads
const THEME_VARS = {
  '--ink': THEME.ink,
  '--accent': THEME.accent,
  '--on-accent': readableOn(THEME.accent),
} as CSSProperties

// Seconds each instrument holds the figure while the list plays by itself
const DWELL = 5.2

/** Marks [data-reveal] elements under a root as they scroll into view. */
function useReveal() {
  const rootRef = useRef<HTMLDivElement>(null)
  useEffect(() => {
    const root = rootRef.current
    if (!root) return
    const observer = new IntersectionObserver(
      (entries) => {
        for (const entry of entries) {
          if (!entry.isIntersecting) continue
          entry.target.classList.add('is-in')
          observer.unobserve(entry.target)
        }
      },
      { threshold: 0.2, rootMargin: '0px 0px -6% 0px' },
    )
    root.querySelectorAll('[data-reveal]').forEach((element) => observer.observe(element))
    return () => observer.disconnect()
  }, [])
  return rootRef
}

/** A Supabase session in this browser's storage: the visitor is probably signed in. */
function hasStoredSession(): boolean {
  try {
    for (let i = 0; i < localStorage.length; i++) {
      const key = localStorage.key(i)
      if (key?.startsWith('sb-') && key.endsWith('-auth-token')) return true
    }
  } catch {
    // Storage unavailable: nothing to be signed in with
  }
  return false
}

export default function HomePage() {
  const navigate = useNavigate()
  const user = useAuthStore((s) => s.user)
  const authReady = useAuthStore((s) => s.initialized)
  // Until the session has been checked, a browser holding one isn't offered "Sign in"
  const [storedSession] = useState(hasStoredSession)
  const signedOut = authReady ? !user : !storedSession
  // Where "Open app" leads: the visitor's landing page
  const appPath = useSettingsStore((s) => landingPath(s.settings))
  const rootRef = useReveal()
  const toolsRef = useRef<HTMLElement>(null)

  const [active, setActive] = useState(0)
  // The list plays by itself while it's on screen, until the visitor picks an instrument
  const [toolsInView, setToolsInView] = useState(false)
  const [taken, setTaken] = useState(false)
  const playing = toolsInView && !taken
  // Counts the visitor's picks: picking the instrument that's already open restarts its figure too
  const [picks, setPicks] = useState(0)

  // The app is laid out under a page zoom (index.css); the homepage is drawn at true size
  useLayoutEffect(() => {
    const root = document.documentElement
    root.classList.add('oz-home-active')
    return () => root.classList.remove('oz-home-active')
  }, [])

  useEffect(() => {
    const section = toolsRef.current
    if (!section) return
    const observer = new IntersectionObserver(([entry]) => setToolsInView(entry.isIntersecting), { threshold: 0.3 })
    observer.observe(section)
    return () => observer.disconnect()
  }, [])

  useEffect(() => {
    if (!playing) return
    const timer = setTimeout(() => setActive((i) => (i + 1) % TOOLS.length), DWELL * 1000)
    return () => clearTimeout(timer)
  }, [playing, active])

  /** Leaves for a page of the app, which starts at its top. */
  const go = (path: string, state?: unknown) => {
    navigate(path, { state })
    window.scrollTo(0, 0)
  }
  const openApp = () => go(appPath)
  const link = (path: string, state?: unknown) => (e: MouseEvent) => {
    if (e.metaKey || e.ctrlKey || e.shiftKey || e.button !== 0) return
    e.preventDefault()
    go(path, state)
  }
  // Signing in from here leads on into the app, not back to this page (AuthPage)
  const signIn = link('/auth', { fromHome: true })

  const number = (i: number) => String(i + 1).padStart(2, '0')

  return (
    <div className="oz-home" ref={rootRef} style={THEME_VARS}>
      <header className="ozh-nav">
        <button type="button" className="ozh-brand" data-avoid onClick={() => window.scrollTo({ top: 0, behavior: 'smooth' })}>
          <img src="/LogoTransparentWhiteText.png" alt="Ozera" />
        </button>
        <nav className="ozh-nav__links" aria-label="Site" data-avoid>
          {user ? (
            <AccountMenu onOpen={go} />
          ) : (
            signedOut && (
              <a className="ozh-nav__link" href="/auth" onClick={signIn}>
                Sign in
              </a>
            )
          )}
          <button type="button" className="ozh-btn ozh-btn--small" onClick={openApp}>
            Open app <span className="ozh-btn__arrow">→</span>
          </button>
        </nav>
      </header>

      <section className="ozh-hero">
        <div className="ozh-hero__text">
          <h1 className="ozh-display">
            <span className="ozh-line">
              <span>Inspect</span>
            </span>
            <span className="ozh-line">
              <span>intelligence.</span>
            </span>
          </h1>
          <div className="ozh-fade" style={{ animationDelay: '0.7s' }}>
            <p className="ozh-lede">
              Ozera is a workbench for interpretability research. Run experiments
              and reverse engineer any open model, with no code to write.
            </p>
            <div className="ozh-actions">
              <button type="button" className="ozh-btn" onClick={openApp}>
                Open Ozera <span className="ozh-btn__arrow">→</span>
              </button>
              <a className="ozh-link" href={GITHUB_URL} target="_blank" rel="noreferrer">
                GitHub ↗
              </a>
            </div>
          </div>
        </div>
        <FlowMosaic className="ozh-hero__canvas" avoid="[data-avoid]" />
      </section>

      <section className="ozh-tools" ref={toolsRef}>
        <div className="ozh-head" data-reveal>
          <h2 className="ozh-title">Seven instruments.</h2>
        </div>

        <div className="ozh-tools__body">
          <ol className="ozh-toollist" data-reveal>
            {TOOLS.map((tool, i) => (
              <li key={tool.name} className={`ozh-tool${i === active ? ' is-active' : ''}${playing ? ' is-timed' : ''}`}>
                <button
                  type="button"
                  className="ozh-tool__head"
                  aria-expanded={i === active}
                  onClick={() => {
                    setActive(i)
                    setTaken(true)
                    setPicks((n) => n + 1)
                  }}
                >
                  <span className="ozh-tool__num">{number(i)}</span>
                  <span className="ozh-tool__name">{tool.name}</span>
                </button>
                <div className="ozh-tool__body">
                  <div className="ozh-tool__inner">
                    <p>{tool.blurb}</p>
                    <a className="ozh-link ozh-link--quiet" href={tool.path} onClick={link(tool.path)} tabIndex={i === active ? 0 : -1}>
                      Open ↗
                    </a>
                  </div>
                </div>
                <span className="ozh-tool__bar" key={`${active}-${playing}`} />
              </li>
            ))}
          </ol>

          <div className="ozh-figure" data-reveal>
            <InstrumentFigure className="ozh-figure__canvas" active={active} restart={picks} />
          </div>
        </div>
      </section>

      <section className="ozh-extras">
        <div className="ozh-head" data-reveal>
          <h2 className="ozh-title">Complementary features.</h2>
        </div>
        <ul className="ozh-extras__list">
          {EXTRAS.map((extra, i) => (
            <li key={extra.name} className="ozh-extra">
              <div className="ozh-extra__inner" data-reveal style={{ transitionDelay: `${i * 0.08}s` }}>
                <CellGlyph rows={extra.glyph} />
                <h3>{extra.name}</h3>
                <p>{extra.blurb}</p>
              </div>
            </li>
          ))}
        </ul>
      </section>

      <section className="ozh-models">
        <TokenField className="ozh-models__canvas" avoid="[data-avoid]" />
        <div className="ozh-models__content">
          <h2 className="ozh-display ozh-display--second" data-reveal data-avoid>
            Any model.
            <br />
            Every layer.
          </h2>
          <div className="ozh-chips" data-reveal data-avoid>
            {MODELS.map((row) => (
              <ul key={row[0]}>
                {row.map((model) => (
                  <li key={model}>{model}</li>
                ))}
              </ul>
            ))}
          </div>
        </div>
      </section>

      <footer className="ozh-footer">
        <span className="ozh-footer__brand">
          <img src="/emblem.png" alt="" />
          Ozera © {new Date().getFullYear()}
        </span>
        <nav className="ozh-footer__links" aria-label="Footer">
          <a href={GITHUB_URL} target="_blank" rel="noreferrer">
            GitHub ↗
          </a>
          {signedOut && (
            <a href="/auth" onClick={signIn}>
              Sign in
            </a>
          )}
          <a href={appPath} onClick={link(appPath)}>
            Open app
          </a>
        </nav>
      </footer>
    </div>
  )
}
