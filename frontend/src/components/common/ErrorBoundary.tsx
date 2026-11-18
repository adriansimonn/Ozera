import { Component, type ErrorInfo, type ReactNode } from 'react'
import { THEME } from '../home/homeTheme'

interface Props {
  children: ReactNode
}

interface State {
  hasError: boolean
}

export class ErrorBoundary extends Component<Props, State> {
  constructor(props: Props) {
    super(props)
    this.state = { hasError: false }
  }

  static getDerivedStateFromError(): State {
    return { hasError: true }
  }

  componentDidCatch(error: Error, info: ErrorInfo) {
    console.error('ErrorBoundary caught:', error, info.componentStack)
  }

  render() {
    if (this.state.hasError) {
      // Outside the app's theme (it may be what failed), so the colors are spelled out
      return (
        <div
          style={{
            minHeight: '125vh',
            display: 'flex',
            alignItems: 'center',
            padding: '0 clamp(20px, 3.4vw, 52px)',
            background: THEME.ink,
            color: '#fff',
          }}
        >
          <div style={{ maxWidth: '34rem' }}>
            <h1 style={{ fontSize: '3rem', fontWeight: 500, letterSpacing: '-0.04em', lineHeight: 1, marginBottom: '1.25rem' }}>
              Something went wrong.
            </h1>
            <p style={{ color: 'rgba(255, 255, 255, 0.7)', lineHeight: 1.55, marginBottom: '1.75rem' }}>
              An unexpected error occurred. Please try refreshing the page.
            </p>
            <button
              onClick={() => {
                this.setState({ hasError: false })
                window.location.reload()
              }}
              style={{ padding: '0.85rem 1.2rem', background: '#fff', color: THEME.ink, fontSize: '0.9375rem', fontWeight: 500 }}
            >
              Refresh page →
            </button>
          </div>
        </div>
      )
    }

    return this.props.children
  }
}
