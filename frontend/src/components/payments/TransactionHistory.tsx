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
  
  Zap,
  Search,
  MessageSquare,
  Layers,
  Microscope,
  Shapes,
  Sparkles,
  Bot } from 'lucide-react';
import { useTransactions, Transaction } from '../../hooks/useCredits';
import { Spinner } from '../common/Spinner'

const ITEMS_PER_PAGE = 10;

function getTransactionIcon(type: Transaction['transaction_type']) {
  switch (type) {
    case 'credit_purchase':
      return <CreditCard size={16} className="text-ok" />;
    case 'training_charge':
      return <Cpu size={16} className="text-warn" />;
    case 'training_refund':
      return <RefreshCcw size={16} className="text-fg" />;
    case 'admin_adjustment':
      return <ArrowUpRight size={16} className="text-fg" />;
    case 'inference_charge':
      return <MessageSquare size={16} className="text-warn" />;
    case 'inference_refund':
      return <RefreshCcw size={16} className="text-fg" />;
    case 'patching_charge':
      return <Zap size={16} className="text-fg" />;
    case 'analysis_charge':
      return <Search size={16} className="text-fg" />;
    case 'sae_charge':
      return <Layers size={16} className="text-fg" />;
    case 'probe_charge':
      return <Microscope size={16} className="text-fg" />;
    case 'geometry_charge':
      return <Shapes size={16} className="text-fg" />;
    case 'geometry_refund':
      return <RefreshCcw size={16} className="text-fg" />;
    case 'autointerp_charge':
      return <Sparkles size={16} className="text-warn" />;
    case 'autointerp_refund':
      return <RefreshCcw size={16} className="text-fg" />;
    case 'circuit_charge':
      return <Zap size={16} className="text-fg" />;
    case 'circuit_refund':
      return <RefreshCcw size={16} className="text-fg" />;
    case 'assistant':
      return <Bot size={16} className="text-sky-400" />;
    default:
      return <ArrowUpRight size={16} className="text-fg" />;
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
    case 'inference_refund':
      return 'Inference Refund';
    case 'patching_charge':
      return 'Patching Experiment';
    case 'analysis_charge':
      return 'Pattern Analysis';
    case 'sae_charge':
      return 'SAE Analysis';
    case 'probe_charge':
      return 'Probe Lab';
    case 'geometry_charge':
      return 'Geometry Lab';
    case 'geometry_refund':
      return 'Geometry Lab Refund';
    case 'autointerp_charge':
      return 'Autointerp';
    case 'autointerp_refund':
      return 'Autointerp Refund';
    case 'circuit_charge':
      return 'Circuit Tracing';
    case 'circuit_refund':
      return 'Circuit Tracing Refund';
    case 'assistant':
      return 'Assistant';
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
    <div className="flex items-center justify-between py-3 border-b border-rule last:border-0">
      <div className="flex items-center gap-3">
        <div className="w-8 h-8 bg-transparent flex items-center justify-center">
          {getTransactionIcon(transaction.transaction_type)}
        </div>
        <div>
          <div className="text-sm font-medium text-fg">
            {getTransactionLabel(transaction.transaction_type)}
          </div>
          <div className="text-[0.82rem] text-fg">
            {transaction.description || formatDate(transaction.created_at)}
          </div>
        </div>
      </div>
      <div className={`text-sm font-medium ${isPositive ? 'text-ok' : 'text-danger'}`}>
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
        <Spinner large />
      </div>
    );
  }

  if (error) {
    return (
      <div className="text-center py-8 text-fg">
        <p>Failed to load transactions</p>
        <button
          onClick={() => fetchTransactions()}
          className="mt-2 text-sm text-fg hover:text-mark"
        >
          Try again
        </button>
      </div>
    );
  }

  if (transactions.length === 0) {
    return (
      <div className="text-center py-8 text-fg">
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
    <div className="bg-transparent border border-rule">
      <div className="p-4 border-b border-rule">
        <h3 className="text-sm font-medium text-fg">Transaction History</h3>
      </div>

      <div className="p-4">
        {displayTransactions.map((transaction) => (
          <TransactionRow key={transaction.id} transaction={transaction} />
        ))}
      </div>

      {/* Pagination */}
      {!maxItems && totalPages > 1 && (
        <div className="flex items-center justify-between px-4 py-3 border-t border-rule">
          <span className="text-sm text-fg">
            Page {page + 1} of {totalPages}
          </span>
          <div className="flex items-center gap-2">
            <button
              onClick={handlePrevPage}
              disabled={page === 0 || loading}
              className="p-1 hover:bg-fill-strong disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
            >
              <ChevronLeft size={18} className="text-fg" />
            </button>
            <button
              onClick={handleNextPage}
              disabled={page >= totalPages - 1 || loading}
              className="p-1 hover:bg-fill-strong disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
            >
              <ChevronRight size={18} className="text-fg" />
            </button>
          </div>
        </div>
      )}
    </div>
  );
}
