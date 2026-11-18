/**
 * The account in the app's nav: the balance and the user's name, opening a menu with their
 * credits, Settings and Sign out. After the homepage's AccountMenu.
 */
import { useState, useRef, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import { useAuthStore } from '../../stores/authStore';
import { useSettingsStore } from '../../stores/settingsStore';
import { usePurchaseCredits } from '../../contexts/PurchaseCreditsContext';

export function UserMenu() {
  const [isOpen, setIsOpen] = useState(false);
  const menuRef = useRef<HTMLDivElement>(null);
  const navigate = useNavigate();
  const { user, logout } = useAuthStore();
  const showBalance = useSettingsStore((s) => s.settings?.credits?.show_balance_in_navbar ?? true);
  const { showPurchaseCredits } = usePurchaseCredits();

  // A click anywhere else, or Escape, closes the menu
  useEffect(() => {
    if (!isOpen) return;
    const handleClickOutside = (event: MouseEvent) => {
      if (menuRef.current && !menuRef.current.contains(event.target as Node)) {
        setIsOpen(false);
      }
    };
    const handleKey = (event: KeyboardEvent) => {
      if (event.key === 'Escape') setIsOpen(false);
    };

    document.addEventListener('mousedown', handleClickOutside);
    document.addEventListener('keydown', handleKey);
    return () => {
      document.removeEventListener('mousedown', handleClickOutside);
      document.removeEventListener('keydown', handleKey);
    };
  }, [isOpen]);

  if (!user) return null;

  const handleLogout = () => {
    logout();
    setIsOpen(false);
  };

  const handleAddCredits = () => {
    setIsOpen(false);
    showPurchaseCredits();
  };

  return (
    <div className="oz-account" ref={menuRef}>
      <button
        type="button"
        className="oz-account__button"
        aria-haspopup="menu"
        aria-expanded={isOpen}
        onClick={() => setIsOpen(!isOpen)}
      >
        {showBalance && (
          <span className="oz-account__balance" title="Available credits">
            ${user.available_balance.toFixed(2)}
          </span>
        )}
        <span className="oz-account__name">{user.full_name || user.email.split('@')[0]}</span>
        <svg className="oz-caret" viewBox="0 0 10 6" aria-hidden="true">
          <path d="M1 1l4 4 4-4" />
        </svg>
      </button>

      {isOpen && (
        <div className="oz-account__menu oz-menu" role="menu">
          <p className="oz-account__email">{user.email}</p>

          <div className="oz-account__credits">
            <span className="oz-account__credits-label">Credits</span>
            <span className="oz-account__credits-value">${user.available_balance.toFixed(2)}</span>
            <button type="button" className="oz-btn oz-btn--small" onClick={handleAddCredits}>
              Add credits
            </button>
            {user.reserved_usd > 0 && (
              <span className="oz-account__credits-note">
                ${user.reserved_usd.toFixed(2)} reserved of ${user.balance_usd.toFixed(2)}
              </span>
            )}
          </div>

          <button
            type="button"
            className="oz-menu__item"
            role="menuitem"
            onClick={() => { navigate('/settings'); setIsOpen(false); }}
          >
            Settings
          </button>
          <button type="button" className="oz-menu__item" role="menuitem" onClick={handleLogout}>
            Sign out
          </button>
        </div>
      )}
    </div>
  );
}
