/**
 * User menu dropdown component.
 */
import { useState, useRef, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import { User, LogOut, Wallet, Plus, Settings } from 'lucide-react';
import { useAuthStore } from '../../stores/authStore';

export function UserMenu() {
  const [isOpen, setIsOpen] = useState(false);
  const menuRef = useRef<HTMLDivElement>(null);
  const navigate = useNavigate();
  const { user, logout } = useAuthStore();

  // Close menu when clicking outside
  useEffect(() => {
    const handleClickOutside = (event: MouseEvent) => {
      if (menuRef.current && !menuRef.current.contains(event.target as Node)) {
        setIsOpen(false);
      }
    };

    document.addEventListener('mousedown', handleClickOutside);
    return () => document.removeEventListener('mousedown', handleClickOutside);
  }, []);

  if (!user) return null;

  const handleLogout = () => {
    logout();
    setIsOpen(false);
  };

  const handleAddCredits = () => {
    setIsOpen(false);
    // Use globally available modal trigger
    if ((window as any).showPurchaseCreditsModal) {
      (window as any).showPurchaseCreditsModal();
    }
  };

  return (
    <div className="relative" ref={menuRef}>
      {/* User button */}
      <button
        onClick={() => setIsOpen(!isOpen)}
        className="flex items-center gap-2 px-3 py-2 bg-transparent border border-transparent hover:bg-white/5 hover:border-white/15 transition-colors"
      >
        <div className="w-8 h-8 rounded-full bg-blue-600 flex items-center justify-center">
          <User size={16} className="text-white" />
        </div>
        <div className="hidden md:block text-left">
          <div className="text-sm font-medium text-white">
            {user.full_name || user.email.split('@')[0]}
          </div>
          <div className="text-xs text-gray-400">
            ${user.available_balance.toFixed(2)} available
          </div>
        </div>
      </button>

      {/* Dropdown menu */}
      {isOpen && (
        <div className="absolute right-0 mt-2 w-64 bg-black border border-gray-700 shadow-lg overflow-hidden z-50">
          {/* User info */}
          <div className="px-4 py-3 border-b border-gray-700">
            <div className="text-sm font-medium text-white truncate">{user.email}</div>
            {user.full_name && (
              <div className="text-xs text-gray-400 truncate">{user.full_name}</div>
            )}
          </div>

          {/* Credit balance */}
          <div className="px-4 py-3 border-b border-gray-700">
            <div className="flex items-center justify-between mb-2">
              <span className="text-sm text-gray-300 flex items-center gap-2">
                <Wallet size={14} />
                Credit Balance
              </span>
              <button
                onClick={handleAddCredits}
                className="flex items-center gap-1 px-2 py-1 text-xs font-medium text-blue-400 hover:text-blue-300 bg-blue-500/10 hover:bg-blue-500/20 transition-colors"
              >
                <Plus size={12} />
                Add
              </button>
            </div>
            <div className="text-lg font-bold text-white">
              ${user.balance_usd.toFixed(2)}
            </div>
            {user.reserved_usd > 0 && (
              <div className="text-xs text-gray-400">
                ${user.reserved_usd.toFixed(2)} reserved
              </div>
            )}
            <div className="text-xs text-green-400 font-medium mt-1">
              ${user.available_balance.toFixed(2)} available
            </div>
          </div>

          {/* Menu items */}
          <div className="py-1">
            <button
              onClick={() => { navigate('/settings'); setIsOpen(false); }}
              className="w-full px-4 py-2 text-left text-sm text-gray-300 hover:bg-[#2a2a2a] flex items-center gap-2 transition-colors"
            >
              <Settings size={14} />
              Settings
            </button>
            <button
              onClick={handleLogout}
              className="w-full px-4 py-2 text-left text-sm text-gray-300 hover:bg-[#2a2a2a] flex items-center gap-2 transition-colors"
            >
              <LogOut size={14} />
              Logout
            </button>
          </div>
        </div>
      )}
    </div>
  );
}
