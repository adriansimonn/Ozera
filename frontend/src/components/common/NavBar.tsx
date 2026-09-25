/**
 * Navigation bar component with glass morphism styling.
 * Supports view mode toggle for pages that use split/single view.
 * Supports custom models page mode toggle (training/upload).
 */

import { Link, useLocation, useNavigate } from 'react-router-dom'
import { Cpu, Square, SplitSquareVertical, Sparkles, GraduationCap, Upload, Zap, Search, Layers } from 'lucide-react'
import { UserMenu } from './UserMenu'
import { useAuthStore } from '../../stores/authStore'

type ViewMode = 'single' | 'split'
export type CustomModelsMode = 'training' | 'upload'

interface NavBarProps {
  viewMode?: ViewMode
  onViewModeChange?: (mode: ViewMode) => void
  showViewToggle?: boolean
  customModelsMode?: CustomModelsMode
  onCustomModelsModeChange?: (mode: CustomModelsMode) => void
  showCustomModelsToggle?: boolean
}

export function NavBar({
  viewMode,
  onViewModeChange,
  showViewToggle = false,
  customModelsMode,
  onCustomModelsModeChange,
  showCustomModelsToggle = false,
}: NavBarProps) {
  const location = useLocation()
  const navigate = useNavigate()
  const { user } = useAuthStore()

  const isActive = (path: string) => location.pathname === path

  return (
    <nav className="navbar">
      <div className="navbar-content">
        <div className="navbar-left">
          <Link to="/" className="navbar-logo">
            <img src="/LogoTransparentWhiteText.png" alt="Ozera" />
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
              Custom Models
            </Link>
            <Link
              to="/patching"
              className={`nav-link ${isActive('/patching') ? 'active' : ''}`}
            >
              <Zap className="nav-icon" />
              Activation Patching
            </Link>
            <Link
              to="/analysis"
              className={`nav-link ${isActive('/analysis') ? 'active' : ''}`}
            >
              <Search className="nav-icon" />
              Attention Analysis
            </Link>
            <Link
              to="/sae"
              className={`nav-link ${isActive('/sae') ? 'active' : ''}`}
            >
              <Layers className="nav-icon" />
              SAE Analysis
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

          {showCustomModelsToggle && onCustomModelsModeChange && (
            <div className="view-mode-toggle">
              <button
                onClick={() => onCustomModelsModeChange('training')}
                className={`mode-btn ${customModelsMode === 'training' ? 'active' : ''}`}
                title="Train a custom model"
              >
                <GraduationCap className="mode-icon" />
                Train
              </button>
              <button
                onClick={() => onCustomModelsModeChange('upload')}
                className={`mode-btn ${customModelsMode === 'upload' ? 'active' : ''}`}
                title="Upload a model file"
              >
                <Upload className="mode-icon" />
                Upload
              </button>
            </div>
          )}

          {/* Auth Section - Show UserMenu if logged in, otherwise show Login/Signup buttons */}
          {user ? (
            <UserMenu />
          ) : (
            <div className="auth-buttons">
              <button onClick={() => navigate('/auth')} className="auth-btn login-btn">
                Login
              </button>
              <button onClick={() => navigate('/auth')} className="auth-btn signup-btn">
                Sign Up
              </button>
            </div>
          )}
        </div>
      </div>

      <style>{`
        .navbar {
          position: fixed;
          top: 0;
          left: 0;
          right: 0;
          z-index: 100;
          background: rgba(0, 0, 0, 0.85);
          border-bottom: 1px solid rgba(255, 255, 255, 0.1);
        }

        [data-bg="light"] .navbar {
          background: rgba(255, 255, 255, 0.85);
          border-bottom: 1px solid rgba(0, 0, 0, 0.1);
        }

        [data-interface="glass"]:not([data-bg="light"]) .navbar {
          background: rgba(0, 0, 0, 0.6);
          backdrop-filter: blur(24px) saturate(200%);
          -webkit-backdrop-filter: blur(24px) saturate(200%);
        }

        [data-bg="light"][data-interface="glass"] .navbar {
          background: rgba(255, 255, 255, 0.6);
          backdrop-filter: blur(24px) saturate(200%);
          -webkit-backdrop-filter: blur(24px) saturate(200%);
        }

        .navbar-content {
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
          color: #ffffff;
          text-decoration: none;
          font-size: 0.875rem;
          font-weight: 500;
          letter-spacing: 0.025em;
          transition: all 0.2s ease;
          border: 1px solid transparent;
        }

        .nav-link:hover {
          color: #ffffff;
          background: rgba(255, 255, 255, 0.05);
        }

        .nav-link.active {
          color: #ffffff;
          background: rgba(255, 255, 255, 0.1);
          border-color: rgba(255, 255, 255, 0.15);
        }

        [data-bg="light"] .nav-link {
          color: #1d1d1f;
        }

        [data-bg="light"] .nav-link:hover {
          color: #1d1d1f;
          background: rgba(0, 0, 0, 0.05);
        }

        [data-bg="light"] .nav-link.active {
          color: #1d1d1f;
          background: rgba(0, 0, 0, 0.08);
          border-color: rgba(0, 0, 0, 0.12);
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

        [data-bg="light"] .view-mode-toggle {
          background: rgba(0, 0, 0, 0.05);
          border-color: rgba(0, 0, 0, 0.1);
        }

        .mode-btn {
          display: flex;
          align-items: center;
          gap: 0.5rem;
          padding: 0.5rem 1rem;
          background: rgba(255, 255, 255, 0.03);
          border: 1px solid rgba(255, 255, 255, 0.1);
          color: #ffffff;
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

        [data-bg="light"] .mode-btn {
          background: rgba(0, 0, 0, 0.03);
          border-color: rgba(0, 0, 0, 0.1);
          color: #1d1d1f;
        }

        [data-bg="light"] .mode-btn:hover {
          background: rgba(0, 0, 0, 0.06);
          border-color: rgba(0, 0, 0, 0.15);
        }

        [data-bg="light"] .mode-btn.active {
          background: rgba(0, 0, 0, 0.1);
          border-color: rgba(0, 0, 0, 0.2);
          color: #1d1d1f;
          box-shadow: 0 0 20px rgba(0, 0, 0, 0.05);
        }

        .mode-icon {
          width: 14px;
          height: 14px;
        }

        .auth-buttons {
          display: flex;
          align-items: center;
          gap: 0.75rem;
        }

        .auth-btn {
          padding: 0.625rem 1.25rem;
          border: 1px solid rgba(255, 255, 255, 0.15);
          font-size: 0.875rem;
          font-weight: 500;
          cursor: pointer;
          transition: all 0.2s ease;
          letter-spacing: 0.025em;
        }

        .login-btn {
          background: transparent;
          color: #ffffff;
        }

        .login-btn:hover {
          background: rgba(255, 255, 255, 0.05);
          color: #ffffff;
          border-color: rgba(255, 255, 255, 0.25);
        }

        .signup-btn {
          background: rgba(255, 255, 255, 0.1);
          color: #ffffff;
          border-color: rgba(255, 255, 255, 0.15);
        }

        .signup-btn:hover {
          background: rgba(255, 255, 255, 0.15);
          color: #ffffff;
          border-color: rgba(255, 255, 255, 0.25);
        }

        [data-bg="light"] .auth-btn {
          border-color: rgba(0, 0, 0, 0.15);
        }

        [data-bg="light"] .login-btn {
          color: #1d1d1f;
        }

        [data-bg="light"] .login-btn:hover {
          background: rgba(0, 0, 0, 0.05);
          color: #1d1d1f;
          border-color: rgba(0, 0, 0, 0.2);
        }

        [data-bg="light"] .signup-btn {
          background: rgba(0, 0, 0, 0.08);
          color: #1d1d1f;
          border-color: rgba(0, 0, 0, 0.15);
        }

        [data-bg="light"] .signup-btn:hover {
          background: rgba(0, 0, 0, 0.12);
          color: #1d1d1f;
          border-color: rgba(0, 0, 0, 0.2);
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

          .auth-btn {
            padding: 0.5rem 1rem;
            font-size: 0.8rem;
          }
        }
      `}</style>
    </nav>
  )
}

export default NavBar
