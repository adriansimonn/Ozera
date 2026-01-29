/**
 * Transaction history display component.
 */
import { useState } from 'react';
import {
  ArrowUpRight,
  CreditCard,
  Cpu,
  RefreshCcw,
  ChevronLeft,
  ChevronRight,
  Loader2,
  Zap,
  Search,
  MessageSquare,
} from 'lucide-react';
import { useTransactions, Transaction } from '../../hooks/useCredits';

const ITEMS_PER_PAGE = 10;

function getTransactionIcon(type: Transaction['transaction_type']) {
  switch (type) {
    case 'credit_purchase':
      return <CreditCard size={16} className="text-green-400" />;
    case 'training_charge':
      return <Cpu size={16} className="text-orange-400" />;
    case 'training_refund':
      return <RefreshCcw size={16} className="text-blue-400" />;
    case 'admin_adjustment':
      return <ArrowUpRight size={16} className="text-purple-400" />;
    case 'inference_charge':
      return <MessageSquare size={16} className="text-yellow-400" />;
    case 'patching_charge':
      return <Zap size={16} className="text-pink-400" />;
    case 'analysis_charge':
      return <Search size={16} className="text-cyan-400" />;
    default:
      return <ArrowUpRight size={16} className="text-gray-400" />;
  }
}

function getTransactionLabel(type: Transaction['transaction_type']) {
  switch (type) {
    case 'credit_purchase':
      return 'Credit Purchase';
    case 'training_charge':
      return 'Training Charge';
    case 'training_refund':
      return 'Training Refund';
    case 'admin_adjustment':
      return 'Adjustment';
    case 'inference_charge':
      return 'Inference';
    case 'patching_charge':
      return 'Patching Experiment';
    case 'analysis_charge':
      return 'Pattern Analysis';
    default:
      return 'Transaction';
  }
}

function formatDate(dateString: string) {
  const date = new Date(dateString);
  return date.toLocaleDateString('en-US', {
    month: 'short',
    day: 'numeric',
    year: 'numeric',
    hour: '2-digit',
    minute: '2-digit',
  });
}

interface TransactionRowProps {
  transaction: Transaction;
}

function TransactionRow({ transaction }: TransactionRowProps) {
  const isPositive = transaction.amount_usd > 0;

  return (
    <div className="flex items-center justify-between py-3 border-b border-gray-800 last:border-0">
      <div className="flex items-center gap-3">
        <div className="w-8 h-8 rounded-full bg-[#2a2a2a] flex items-center justify-center">
          {getTransactionIcon(transaction.transaction_type)}
        </div>
        <div>
          <div className="text-sm font-medium text-white">
            {getTransactionLabel(transaction.transaction_type)}
          </div>
          <div className="text-xs text-gray-400">
            {transaction.description || formatDate(transaction.created_at)}
          </div>
        </div>
      </div>
      <div className={`text-sm font-medium ${isPositive ? 'text-green-400' : 'text-red-400'}`}>
        {isPositive ? '+' : ''}${Math.abs(transaction.amount_usd).toFixed(2)}
      </div>
    </div>
  );
}

interface TransactionHistoryProps {
  compact?: boolean;
  maxItems?: number;
}

export function TransactionHistory({ compact = false, maxItems }: TransactionHistoryProps) {
  const { transactions, total, loading, error, fetchTransactions } = useTransactions();
  const [page, setPage] = useState(0);

  const displayTransactions = maxItems ? transactions.slice(0, maxItems) : transactions;
  const totalPages = Math.ceil(total / ITEMS_PER_PAGE);

  const handlePrevPage = () => {
    if (page > 0) {
      const newPage = page - 1;
      setPage(newPage);
      fetchTransactions(ITEMS_PER_PAGE, newPage * ITEMS_PER_PAGE);
    }
  };

  const handleNextPage = () => {
    if (page < totalPages - 1) {
      const newPage = page + 1;
      setPage(newPage);
      fetchTransactions(ITEMS_PER_PAGE, newPage * ITEMS_PER_PAGE);
    }
  };

  if (loading && transactions.length === 0) {
    return (
      <div className="flex items-center justify-center py-8">
        <Loader2 size={24} className="animate-spin text-gray-400" />
      </div>
    );
  }

  if (error) {
    return (
      <div className="text-center py-8 text-gray-400">
        <p>Failed to load transactions</p>
        <button
          onClick={() => fetchTransactions()}
          className="mt-2 text-sm text-blue-400 hover:text-blue-300"
        >
          Try again
        </button>
      </div>
    );
  }

  if (transactions.length === 0) {
    return (
      <div className="text-center py-8 text-gray-400">
        <p>No transactions yet</p>
        <p className="text-sm mt-1">Purchase credits to get started!</p>
      </div>
    );
  }

  if (compact) {
    return (
      <div className="space-y-0">
        {displayTransactions.map((transaction) => (
          <TransactionRow key={transaction.id} transaction={transaction} />
        ))}
      </div>
    );
  }

  return (
    <div className="bg-[#1a1a1a] border border-gray-700 rounded-lg">
      <div className="p-4 border-b border-gray-700">
        <h3 className="text-sm font-medium text-gray-300">Transaction History</h3>
      </div>

      <div className="p-4">
        {displayTransactions.map((transaction) => (
          <TransactionRow key={transaction.id} transaction={transaction} />
        ))}
      </div>

      {/* Pagination */}
      {!maxItems && totalPages > 1 && (
        <div className="flex items-center justify-between px-4 py-3 border-t border-gray-700">
          <span className="text-sm text-gray-400">
            Page {page + 1} of {totalPages}
          </span>
          <div className="flex items-center gap-2">
            <button
              onClick={handlePrevPage}
              disabled={page === 0 || loading}
              className="p-1 rounded hover:bg-[#2a2a2a] disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
            >
              <ChevronLeft size={18} className="text-gray-400" />
            </button>
            <button
              onClick={handleNextPage}
              disabled={page >= totalPages - 1 || loading}
              className="p-1 rounded hover:bg-[#2a2a2a] disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
            >
              <ChevronRight size={18} className="text-gray-400" />
            </button>
          </div>
        </div>
      )}
    </div>
  );
}
