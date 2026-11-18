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
        <Wallet size={16} className="text-fg" />
        <span className="text-sm font-medium text-fg">
          ${displayBalance.available_balance.toFixed(2)}
        </span>
        {showAddButton && onAddCredits && (
          <button
            onClick={onAddCredits}
            className="p-1 hover:bg-fill-strong transition-colors"
            title="Add Credits"
          >
            <Plus size={14} className="text-fg" />
          </button>
        )}
      </div>
    );
  }

  return (
    <div className="bg-transparent border border-rule p-4">
      <div className="flex items-center justify-between mb-3">
        <div className="flex items-center gap-2">
          <Wallet size={18} className="text-fg" />
          <span className="text-sm font-medium text-fg">Credit Balance</span>
        </div>
        {showAddButton && onAddCredits && (
          <button
            onClick={onAddCredits}
            className="flex items-center gap-1 px-3 py-1.5 bg-fg hover:bg-volt hover:text-on-volt text-fg text-sm font-medium transition-colors"
          >
            <Plus size={14} />
            Add Credits
          </button>
        )}
      </div>

      {loading ? (
        <div className="animate-pulse">
          <div className="h-8 bg-fill-strong w-24 mb-2" />
          <div className="h-4 bg-fill-strong w-32" />
        </div>
      ) : (
        <>
          <div className="text-2xl font-medium text-fg mb-1">
            ${displayBalance.balance_usd.toFixed(2)}
          </div>

          {displayBalance.reserved_usd > 0 && (
            <div className="text-[0.82rem] text-fg mb-1">
              ${displayBalance.reserved_usd.toFixed(2)} reserved for running jobs
            </div>
          )}

          <div className="text-sm text-ok font-medium">
            ${displayBalance.available_balance.toFixed(2)} available
          </div>
        </>
      )}
    </div>
  );
}
