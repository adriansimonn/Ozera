/**
 * The signed-in visitor in the homepage's nav: their name, opening a menu with their email,
 * Settings and Sign out.
 */

import { useEffect, useRef, useState } from 'react'
import { useAuthStore } from '../../stores/authStore'

interface Props {
  /** Leaves the homepage for a page of the app */
  onOpen: (path: string) => void
}

export function AccountMenu({ onOpen }: Props) {
  const user = useAuthStore((s) => s.user)
  const logout = useAuthStore((s) => s.logout)
  const [open, setOpen] = useState(false)
  const rootRef = useRef<HTMLDivElement>(null)

  // A click anywhere else, or Escape, closes the menu
  useEffect(() => {
    if (!open) return
    const onPointer = (e: MouseEvent) => {
      if (!rootRef.current?.contains(e.target as Node)) setOpen(false)
    }
    const onKey = (e: KeyboardEvent) => {
      if (e.key === 'Escape') setOpen(false)
    }
    document.addEventListener('mousedown', onPointer)
    document.addEventListener('keydown', onKey)
    return () => {
      document.removeEventListener('mousedown', onPointer)
      document.removeEventListener('keydown', onKey)
    }
  }, [open])

  if (!user) return null

  return (
    <div className="ozh-account" ref={rootRef}>
      <button
        type="button"
        className="ozh-account__button"
        aria-haspopup="menu"
        aria-expanded={open}
        onClick={() => setOpen((o) => !o)}
      >
        <span className="ozh-account__name">{user.full_name || user.email.split('@')[0]}</span>
        <svg className="ozh-account__caret" viewBox="0 0 10 6" aria-hidden="true">
          <path d="M1 1l4 4 4-4" />
        </svg>
      </button>
      {open && (
        <div className="ozh-account__menu" role="menu">
          <p className="ozh-account__email">{user.email}</p>
          <a
            className="ozh-account__item"
            role="menuitem"
            href="/settings"
            onClick={(e) => {
              if (e.metaKey || e.ctrlKey || e.shiftKey || e.button !== 0) return
              e.preventDefault()
              onOpen('/settings')
            }}
          >
            Settings
          </a>
          <button
            type="button"
            className="ozh-account__item"
            role="menuitem"
            onClick={() => {
              setOpen(false)
              logout()
            }}
          >
            Sign out
          </button>
        </div>
      )}
    </div>
  )
}

export default AccountMenu
