/**
 * Credit balance display component.
 */
import { Wallet, Plus } from 'lucide-react';
import { useCreditBalance } from '../../hooks/useCredits';
import { useAuthStore } from '../../stores/authStore';

interface CreditBalanceProps {
  onAddCredits?: () => void;
  showAddButton?: boolean;
  compact?: boolean;
}

export function CreditBalance({ onAddCredits, showAddButton = true, compact = false }: CreditBalanceProps) {
  const { user } = useAuthStore();
  const { balance, loading } = useCreditBalance();

  if (!user) return null;

  // Use user data from auth store (updated on login/refresh)
  // or balance from dedicated hook
  const displayBalance = balance || {
    balance_usd: user.balance_usd,
    reserved_usd: user.reserved_usd,
    available_balance: user.available_balance,
  };

  if (compact) {
    return (
      <div className="flex items-center gap-2">
        <Wallet size={16} className="text-gray-400" />
        <span className="text-sm font-medium text-white">
          ${displayBalance.available_balance.toFixed(2)}
        </span>
        {showAddButton && onAddCredits && (
          <button
            onClick={onAddCredits}
            className="p-1 rounded hover:bg-[#2a2a2a] transition-colors"
            title="Add Credits"
          >
            <Plus size={14} className="text-blue-400" />
          </button>
        )}
      </div>
    );
  }

  return (
    <div className="bg-[#1a1a1a] border border-gray-700 rounded-lg p-4">
      <div className="flex items-center justify-between mb-3">
        <div className="flex items-center gap-2">
          <Wallet size={18} className="text-gray-400" />
          <span className="text-sm font-medium text-gray-300">Credit Balance</span>
        </div>
        {showAddButton && onAddCredits && (
          <button
            onClick={onAddCredits}
            className="flex items-center gap-1 px-3 py-1.5 bg-blue-600 hover:bg-blue-700 text-white text-sm font-medium rounded-lg transition-colors"
          >
            <Plus size={14} />
            Add Credits
          </button>
        )}
      </div>

      {loading ? (
        <div className="animate-pulse">
          <div className="h-8 bg-gray-700 rounded w-24 mb-2" />
          <div className="h-4 bg-gray-700 rounded w-32" />
        </div>
      ) : (
        <>
          <div className="text-2xl font-bold text-white mb-1">
            ${displayBalance.balance_usd.toFixed(2)}
          </div>

          {displayBalance.reserved_usd > 0 && (
            <div className="text-xs text-gray-400 mb-1">
              ${displayBalance.reserved_usd.toFixed(2)} reserved for running jobs
            </div>
          )}

          <div className="text-sm text-green-400 font-medium">
            ${displayBalance.available_balance.toFixed(2)} available
          </div>
        </>
      )}
    </div>
  );
}
