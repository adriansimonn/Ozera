/**
 * Hooks for credit and payment functionality.
 */

import { useState, useCallback, useEffect } from 'react';
import { apiClient as axiosClient } from '../api/axios';
import { useAuthStore } from '../stores/authStore';

// Types for credits API
export interface CreditBalance {
  balance_usd: number;
  reserved_usd: number;
  available_balance: number;
}

export interface Transaction {
  id: number;
  amount_usd: number;
  transaction_type: 'credit_purchase' | 'training_charge' | 'training_refund' | 'admin_adjustment' | 'inference_charge' | 'inference_refund' | 'patching_charge' | 'analysis_charge';
  description: string | null;
  stripe_payment_intent_id: string | null;
  training_job_id: string | null;
  created_at: string;
}

export interface TransactionListResponse {
  transactions: Transaction[];
  total: number;
}

export interface GPUPricing {
  gpu_type: string;
  display_name: string;
  rate_per_hour: number;
  description: string;
}

export interface PricingResponse {
  gpu_pricing: GPUPricing[];
  min_purchase: number;
  max_purchase: number;
}

export interface PaymentIntentResponse {
  client_secret: string;
  payment_intent_id: string;
  amount_usd: number;
  credits_usd: number;
}

/**
 * Hook for managing credit balance.
 */
export function useCreditBalance() {
  const [balance, setBalance] = useState<CreditBalance | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const { isAuthenticated, refreshUser } = useAuthStore();

  const fetchBalance = useCallback(async () => {
    if (!isAuthenticated) {
      setBalance(null);
      return;
    }

    setLoading(true);
    setError(null);
    try {
      const response = await axiosClient.get<CreditBalance>('/credits/balance');
      setBalance(response.data);
    } catch (err: any) {
      setError(err.response?.data?.detail || 'Failed to fetch balance');
    } finally {
      setLoading(false);
    }
  }, [isAuthenticated]);

  useEffect(() => {
    if (isAuthenticated) {
      fetchBalance();
    }
  }, [isAuthenticated, fetchBalance]);

  return {
    balance,
    loading,
    error,
    fetchBalance,
    refreshBalance: async () => {
      await fetchBalance();
      await refreshUser(); // Also refresh user data in auth store
    },
  };
}

/**
 * Hook for transaction history.
 */
export function useTransactions() {
  const [transactions, setTransactions] = useState<Transaction[]>([]);
  const [total, setTotal] = useState(0);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const { isAuthenticated } = useAuthStore();

  const fetchTransactions = useCallback(async (limit = 50, offset = 0) => {
    if (!isAuthenticated) {
      setTransactions([]);
      setTotal(0);
      return;
    }

    setLoading(true);
    setError(null);
    try {
      const response = await axiosClient.get<TransactionListResponse>('/credits/transactions', {
        params: { limit, offset },
      });
      setTransactions(response.data.transactions);
      setTotal(response.data.total);
    } catch (err: any) {
      setError(err.response?.data?.detail || 'Failed to fetch transactions');
    } finally {
      setLoading(false);
    }
  }, [isAuthenticated]);

  useEffect(() => {
    if (isAuthenticated) {
      fetchTransactions();
    }
  }, [isAuthenticated, fetchTransactions]);

  return {
    transactions,
    total,
    loading,
    error,
    fetchTransactions,
  };
}

/**
 * Hook for pricing information.
 */
export function usePricing() {
  const [pricing, setPricing] = useState<PricingResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const fetchPricing = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const response = await axiosClient.get<PricingResponse>('/credits/pricing');
      setPricing(response.data);
    } catch (err: any) {
      setError(err.response?.data?.detail || 'Failed to fetch pricing');
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    fetchPricing();
  }, [fetchPricing]);

  return {
    pricing,
    gpuPricing: pricing?.gpu_pricing || [],
    minPurchase: pricing?.min_purchase || 5,
    maxPurchase: pricing?.max_purchase || 500,
    loading,
    error,
    fetchPricing,
  };
}

/**
 * Hook for payment operations.
 */
export function usePayment() {
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const { refreshUser } = useAuthStore();

  const createPaymentIntent = useCallback(async (amountUsd: number): Promise<PaymentIntentResponse> => {
    setLoading(true);
    setError(null);
    try {
      const response = await axiosClient.post<PaymentIntentResponse>('/payments/create-intent', {
        amount_usd: amountUsd,
      });
      return response.data;
    } catch (err: any) {
      const message = err.response?.data?.detail || 'Failed to create payment';
      setError(message);
      throw new Error(message);
    } finally {
      setLoading(false);
    }
  }, []);

  const confirmPayment = useCallback(async (paymentIntentId: string) => {
    // Call backend to confirm payment and add credits, so the balance updates without waiting
    // for the Stripe webhook (and works without webhooks locally). The backend credits each
    // payment once, whichever of this call and the webhook arrives first.
    try {
      const response = await axiosClient.post(`/payments/confirm/${paymentIntentId}`);
      return response.data;
    } catch (err: any) {
      console.error('Failed to confirm payment:', err);
      // Don't throw - the payment succeeded, credits will be added via webhook in production
    }
  }, []);

  const onPaymentSuccess = useCallback(async (paymentIntentId?: string) => {
    // Confirm payment via backend (adds credits without webhook)
    if (paymentIntentId) {
      await confirmPayment(paymentIntentId);
    }
    // Refresh user data to update balance
    await refreshUser();
  }, [refreshUser, confirmPayment]);

  return {
    createPaymentIntent,
    confirmPayment,
    onPaymentSuccess,
    loading,
    error,
    clearError: () => setError(null),
  };
}

/**
 * Hook for Stripe configuration.
 */
export function useStripeConfig() {
  const [publishableKey, setPublishableKey] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    const fetchConfig = async () => {
      try {
        const response = await axiosClient.get<{ publishable_key: string }>('/payments/config');
        setPublishableKey(response.data.publishable_key);
      } catch (err: any) {
        setError(err.response?.data?.detail || 'Failed to fetch Stripe config');
      } finally {
        setLoading(false);
      }
    };

    fetchConfig();
  }, []);

  return {
    publishableKey,
    loading,
    error,
    isConfigured: !!publishableKey,
  };
}
