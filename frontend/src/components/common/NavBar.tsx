/**
 * Navigation bar component with glass morphism styling.
 * Supports view mode toggle for pages that use split/single view.
 */

import { Link, useLocation } from 'react-router-dom'
import { Cpu, Square, SplitSquareVertical, Sparkles } from 'lucide-react'

type ViewMode = 'single' | 'split'

interface NavBarProps {
  viewMode?: ViewMode
  onViewModeChange?: (mode: ViewMode) => void
  showViewToggle?: boolean
}

export function NavBar({ viewMode, onViewModeChange, showViewToggle = false }: NavBarProps) {
  const location = useLocation()

  const isActive = (path: string) => location.pathname === path

  return (
    <nav className="navbar">
      <div className="navbar-content">
        <div className="navbar-left">
          <Link to="/" className="navbar-logo">
            <img src="/src/assets/logos/LogoTransparentWhiteText.png" alt="Ozera" />
          </Link>

          <div className="navbar-links">
            <Link
              to="/"
              className={`nav-link ${isActive('/') ? 'active' : ''}`}
            >
              <Sparkles className="nav-icon" />
              Generate
            </Link>
            <Link
              to="/training"
              className={`nav-link ${isActive('/training') ? 'active' : ''}`}
            >
              <Cpu className="nav-icon" />
              Train Model
            </Link>
          </div>
        </div>

        <div className="navbar-right">
          {showViewToggle && onViewModeChange && (
            <div className="view-mode-toggle">
              <button
                onClick={() => onViewModeChange('single')}
                className={`mode-btn ${viewMode === 'single' ? 'active' : ''}`}
                title="Single View Mode"
              >
                <Square className="mode-icon" />
                Single
              </button>
              <button
                onClick={() => onViewModeChange('split')}
                className={`mode-btn ${viewMode === 'split' ? 'active' : ''}`}
                title="Split Screen Mode"
              >
                <SplitSquareVertical className="mode-icon" />
                Split
              </button>
            </div>
          )}
        </div>
      </div>

      <style>{`
        .navbar {
          position: sticky;
          top: 0;
          z-index: 100;
          background: rgba(0, 0, 0, 0.6);
          backdrop-filter: blur(24px) saturate(200%);
          -webkit-backdrop-filter: blur(24px) saturate(200%);
          border-bottom: 1px solid rgba(255, 255, 255, 0.1);
        }

        .navbar-content {
          max-width: 1800px;
          margin: 0 auto;
          padding: 0.75rem 2rem;
          display: flex;
          justify-content: space-between;
          align-items: center;
          gap: 2rem;
        }

        .navbar-left {
          display: flex;
          align-items: center;
          gap: 3rem;
        }

        .navbar-logo {
          display: flex;
          align-items: center;
        }

        .navbar-logo img {
          height: 40px;
          width: auto;
        }

        .navbar-links {
          display: flex;
          align-items: center;
          gap: 0.5rem;
        }

        .nav-link {
          display: flex;
          align-items: center;
          gap: 0.5rem;
          padding: 0.625rem 1rem;
          color: rgba(255, 255, 255, 0.6);
          text-decoration: none;
          font-size: 0.875rem;
          font-weight: 500;
          letter-spacing: 0.025em;
          transition: all 0.2s ease;
          border: 1px solid transparent;
        }

        .nav-link:hover {
          color: rgba(255, 255, 255, 0.9);
          background: rgba(255, 255, 255, 0.05);
        }

        .nav-link.active {
          color: #ffffff;
          background: rgba(255, 255, 255, 0.1);
          border-color: rgba(255, 255, 255, 0.15);
        }

        .nav-icon {
          width: 16px;
          height: 16px;
        }

        .navbar-right {
          display: flex;
          align-items: center;
          gap: 1rem;
        }

        .view-mode-toggle {
          display: flex;
          gap: 0.25rem;
          background: rgba(0, 0, 0, 0.3);
          padding: 0.25rem;
          border: 1px solid rgba(255, 255, 255, 0.12);
        }

        .mode-btn {
          display: flex;
          align-items: center;
          gap: 0.5rem;
          padding: 0.5rem 1rem;
          background: rgba(255, 255, 255, 0.03);
          border: 1px solid rgba(255, 255, 255, 0.1);
          color: rgba(255, 255, 255, 0.6);
          font-size: 0.8rem;
          font-weight: 500;
          cursor: pointer;
          transition: all 0.2s;
          letter-spacing: 0.05em;
          text-transform: uppercase;
        }

        .mode-btn:hover {
          background: rgba(255, 255, 255, 0.08);
          border-color: rgba(255, 255, 255, 0.2);
        }

        .mode-btn.active {
          background: rgba(255, 255, 255, 0.15);
          border-color: rgba(255, 255, 255, 0.3);
          color: #ffffff;
          box-shadow: 0 0 20px rgba(255, 255, 255, 0.1);
        }

        .mode-icon {
          width: 14px;
          height: 14px;
        }

        @media (max-width: 768px) {
          .navbar-content {
            padding: 0.75rem 1rem;
          }

          .navbar-left {
            gap: 1.5rem;
          }

          .navbar-logo img {
            height: 32px;
          }

          .nav-link {
            padding: 0.5rem 0.75rem;
            font-size: 0.8rem;
          }

          .mode-btn {
            padding: 0.4rem 0.75rem;
            font-size: 0.75rem;
          }
        }
      `}</style>
    </nav>
  )
}

export default NavBar
