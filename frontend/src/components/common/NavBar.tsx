/**
 * The app's nav: the logo, the pages, the assistant and the account. Styled in styles/app.css
 * (.oz-nav), after the homepage's nav.
 */

import { Link, useLocation } from 'react-router-dom'
import { UserMenu } from './UserMenu'
import { AssistantButton } from '../assistant/AssistantPanel'
import { useAuthStore } from '../../stores/authStore'
import { useSettingsStore } from '../../stores/settingsStore'
import { APP_PAGES, landingPath } from '../../lib/appPages'

export function NavBar() {
  const location = useLocation()
  const { user } = useAuthStore()
  const landing = useSettingsStore((s) => landingPath(s.settings))

  return (
    <nav className="oz-nav" aria-label="App">
      <Link to={landing} className="oz-nav__brand">
        <img src="/LogoTransparentWhiteText.png" alt="Ozera" />
      </Link>

      <div className="oz-nav__links">
        {APP_PAGES.map(({ path, label }) => {
          const active = location.pathname === path
          return (
            <Link
              key={path}
              to={path}
              className={`oz-nav__link${active ? ' is-active' : ''}`}
              aria-current={active ? 'page' : undefined}
            >
              {label}
            </Link>
          )
        })}
      </div>

      <div className="oz-nav__end">
        {user ? (
          <>
            <AssistantButton />
            <UserMenu />
          </>
        ) : (
          <>
            <Link to="/auth" className="oz-nav__link">
              Sign in
            </Link>
            <Link to="/auth" className="oz-btn oz-btn--small oz-btn--arrow">
              Sign up
            </Link>
          </>
        )}
      </div>
    </nav>
  )
}

export default NavBar
