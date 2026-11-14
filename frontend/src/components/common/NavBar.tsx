/**
 * Navigation bar component.
 */

import { Link, useLocation, useNavigate } from 'react-router-dom'
import { UserMenu } from './UserMenu'
import { AssistantButton } from '../assistant/AssistantPanel'
import { useAuthStore } from '../../stores/authStore'
import { useSettingsStore } from '../../stores/settingsStore'
import { APP_PAGES, landingPath } from '../../lib/appPages'

export function NavBar() {
  const location = useLocation()
  const navigate = useNavigate()
  const { user } = useAuthStore()
  const landing = useSettingsStore((s) => landingPath(s.settings))

  const isActive = (path: string) => location.pathname === path

  return (
    <nav className="navbar">
      <div className="navbar-content">
        <div className="navbar-left">
          <Link to={landing} className="navbar-logo">
            <img src="/LogoTransparentWhiteText.png" alt="Ozera" />
          </Link>

          <div className="navbar-links">
            {APP_PAGES.map(({ path, label, icon: Icon }) => (
              <Link key={path} to={path} className={`nav-link ${isActive(path) ? 'active' : ''}`}>
                <Icon className="nav-icon" />
                {label}
              </Link>
            ))}
          </div>
        </div>

        <div className="navbar-right">
          {/* Auth Section - Show UserMenu if logged in, otherwise show Login/Signup buttons */}
          {user && <AssistantButton />}

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
