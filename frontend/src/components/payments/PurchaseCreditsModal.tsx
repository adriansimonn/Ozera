/**
 * Modal for purchasing credits via Stripe.
 */
import { useState, useEffect } from 'react';
import { X, CreditCard, Check, Loader2, AlertCircle, DollarSign } from 'lucide-react';
import { loadStripe } from '@stripe/stripe-js';
import {
  Elements,
  PaymentElement,
  useStripe,
  useElements,
} from '@stripe/react-stripe-js';
import { usePricing, usePayment, useStripeConfig } from '../../hooks/useCredits';
import { useAuthStore } from '../../stores/authStore';

interface PurchaseCreditsModalProps {
  isOpen: boolean;
  onClose: () => void;
}

// Amount input component
function AmountInput({
  amount,
  onChange,
  minAmount,
  maxAmount,
}: {
  amount: string;
  onChange: (value: string) => void;
  minAmount: number;
  maxAmount: number;
}) {
  const numericAmount = parseFloat(amount) || 0;
  const isValid = numericAmount >= minAmount && numericAmount <= maxAmount;

  return (
    <div className="space-y-3">
      <div className="relative">
        <div className="absolute inset-y-0 left-0 pl-4 flex items-center pointer-events-none">
          <DollarSign size={20} className="text-white" />
        </div>
        <input
          type="number"
          value={amount}
          onChange={(e) => onChange(e.target.value)}
          placeholder="Enter amount"
          min={minAmount}
          max={maxAmount}
          step="0.01"
          className="w-full pl-10 pr-4 py-4 bg-black border border-white/10 text-white text-xl font-medium focus:outline-none focus:border-white/40 transition-colors [appearance:textfield] [&::-webkit-outer-spin-button]:appearance-none [&::-webkit-inner-spin-button]:appearance-none"
        />
      </div>

      <div className="flex justify-between text-sm text-white">
        <span>Min: ${minAmount}</span>
        <span>Max: ${maxAmount}</span>
      </div>

      {amount && !isValid && (
        <p className="text-sm text-red-400">
          Amount must be between ${minAmount} and ${maxAmount}
        </p>
      )}

      {amount && isValid && (
        <div className="p-3 bg-green-500/10 border border-green-500/30">
          <p className="text-green-400 text-sm">
            You'll receive <span className="font-bold">${numericAmount.toFixed(2)}</span> in credits
          </p>
        </div>
      )}

      {/* Quick amount buttons */}
      <div className="flex gap-2 flex-wrap">
        {[10, 25, 50, 100].map((quickAmount) => (
          <button
            key={quickAmount}
            onClick={() => onChange(quickAmount.toString())}
            className={`px-4 py-2 border transition-all ${
              parseFloat(amount) === quickAmount
                ? 'border-blue-500 bg-blue-500/10 text-blue-400'
                : 'border-white/10 hover:border-white/20 bg-black text-white'
            }`}
          >
            ${quickAmount}
          </button>
        ))}
      </div>
    </div>
  );
}

// Checkout form component (inside Stripe Elements)
function CheckoutForm({
  clientSecret,
  paymentIntentId,
  amount,
  onSuccess,
  onCancel,
}: {
  clientSecret: string;
  paymentIntentId: string;
  amount: number;
  onSuccess: (paymentIntentId: string) => void;
  onCancel: () => void;
}) {
  const stripe = useStripe();
  const elements = useElements();
  const [processing, setProcessing] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [succeeded, setSucceeded] = useState(false);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();

    if (!stripe || !elements) {
      return;
    }

    setProcessing(true);
    setError(null);

    const { error: submitError } = await elements.submit();
    if (submitError) {
      setError(submitError.message || 'Payment failed');
      setProcessing(false);
      return;
    }

    const { error: confirmError } = await stripe.confirmPayment({
      elements,
      clientSecret,
      confirmParams: {
        return_url: window.location.href,
      },
      redirect: 'if_required',
    });

    if (confirmError) {
      setError(confirmError.message || 'Payment failed');
      setProcessing(false);
    } else {
      setSucceeded(true);
      setProcessing(false);
      // Call onSuccess with payment intent ID to confirm via backend
      setTimeout(() => {
        onSuccess(paymentIntentId);
      }, 1000);
    }
  };

  if (succeeded) {
    return (
      <div className="text-center py-8">
        <div className="w-16 h-16 mx-auto mb-4 rounded-full bg-green-500/20 flex items-center justify-center">
          <Check size={32} className="text-green-400" />
        </div>
        <h3 className="text-xl font-bold text-white mb-2">Payment Successful!</h3>
        <p className="text-white">
          ${amount.toFixed(2)} credits have been added to your account.
        </p>
      </div>
    );
  }

  return (
    <form onSubmit={handleSubmit} className="space-y-4">
      <div className="bg-black border border-white/10 p-4 mb-4">
        <div className="flex justify-between items-center">
          <span className="text-white">Payment Amount</span>
          <span className="text-xl font-bold text-white">${amount.toFixed(2)}</span>
        </div>
        <div className="flex justify-between items-center mt-1">
          <span className="text-white text-sm">Credits Received</span>
          <span className="text-green-400 font-medium">${amount.toFixed(2)}</span>
        </div>
      </div>

      <PaymentElement
        options={{
          layout: 'tabs',
        }}
      />

      {error && (
        <div className="flex items-center gap-2 p-3 bg-red-500/10 border border-red-500/30 text-red-400 text-sm">
          <AlertCircle size={16} />
          {error}
        </div>
      )}

      <div className="flex gap-3">
        <button
          type="button"
          onClick={onCancel}
          disabled={processing}
          className="flex-1 px-4 py-3 bg-white/5 hover:bg-white/10 border border-white/10 hover:border-white/20 text-white transition-colors disabled:opacity-50"
        >
          Cancel
        </button>
        <button
          type="submit"
          disabled={!stripe || processing}
          className="flex-1 px-4 py-3 bg-blue-600 hover:bg-blue-700 text-white font-medium transition-colors disabled:opacity-50 flex items-center justify-center gap-2"
        >
          {processing ? (
            <>
              <Loader2 size={18} className="animate-spin" />
              Processing...
            </>
          ) : (
            <>
              <CreditCard size={18} />
              Pay ${amount.toFixed(2)}
            </>
          )}
        </button>
      </div>
    </form>
  );
}

export function PurchaseCreditsModal({ isOpen, onClose }: PurchaseCreditsModalProps) {
  const { isAuthenticated } = useAuthStore();
  const { minPurchase, maxPurchase, loading: pricingLoading } = usePricing();
  const { createPaymentIntent, onPaymentSuccess, loading: paymentLoading, error: paymentError } = usePayment();
  const { publishableKey, isConfigured } = useStripeConfig();

  const [amount, setAmount] = useState<string>('');
  const [clientSecret, setClientSecret] = useState<string | null>(null);
  const [paymentIntentId, setPaymentIntentId] = useState<string | null>(null);
  const [stripePromise, setStripePromise] = useState<ReturnType<typeof loadStripe> | null>(null);
  const [step, setStep] = useState<'select' | 'checkout'>('select');

  const numericAmount = parseFloat(amount) || 0;
  const isValidAmount = numericAmount >= minPurchase && numericAmount <= maxPurchase;

  // Initialize Stripe
  useEffect(() => {
    if (publishableKey) {
      setStripePromise(loadStripe(publishableKey));
    }
  }, [publishableKey]);

  // Reset state when modal opens/closes
  useEffect(() => {
    if (!isOpen) {
      setAmount('');
      setClientSecret(null);
      setPaymentIntentId(null);
      setStep('select');
    }
  }, [isOpen]);

  const handleProceedToCheckout = async () => {
    if (!isValidAmount) return;

    try {
      const result = await createPaymentIntent(numericAmount);
      setClientSecret(result.client_secret);
      setPaymentIntentId(result.payment_intent_id);
      setStep('checkout');
    } catch (err) {
      // Error is handled by the hook
    }
  };

  const handlePaymentSuccess = async (intentId: string) => {
    await onPaymentSuccess(intentId);
    onClose();
  };

  const handleBack = () => {
    setStep('select');
    setClientSecret(null);
    setPaymentIntentId(null);
  };

  if (!isOpen) return null;

  if (!isAuthenticated) {
    return (
      <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/70 backdrop-blur-sm">
        <div className="bg-black border border-white/10 p-6 w-full max-w-md">
          <p className="text-center text-white">Please log in to purchase credits.</p>
          <button
            onClick={onClose}
            className="w-full mt-4 px-4 py-2 bg-white/5 hover:bg-white/10 border border-white/10 text-white transition-colors"
          >
            Close
          </button>
        </div>
      </div>
    );
  }

  if (!isConfigured) {
    return (
      <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/70 backdrop-blur-sm">
        <div className="bg-black border border-white/10 p-6 w-full max-w-md">
          <div className="flex items-center gap-2 text-yellow-400 mb-4">
            <AlertCircle size={20} />
            <span className="font-medium">Payment Not Available</span>
          </div>
          <p className="text-white text-sm">
            Payment processing is not configured. Please contact support.
          </p>
          <button
            onClick={onClose}
            className="w-full mt-4 px-4 py-2 bg-white/5 hover:bg-white/10 border border-white/10 text-white transition-colors"
          >
            Close
          </button>
        </div>
      </div>
    );
  }

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/70 backdrop-blur-sm">
      <div className="bg-black border border-white/10 w-full max-w-md max-h-[90vh] overflow-y-auto">
        {/* Header */}
        <div className="flex items-center justify-between p-4 border-b border-white/10">
          <h2 className="text-lg font-semibold text-white">
            {step === 'select' ? 'Add Credits' : 'Checkout'}
          </h2>
          <button
            onClick={onClose}
            className="p-1 hover:bg-white/5 transition-colors"
          >
            <X size={20} className="text-white" />
          </button>
        </div>

        {/* Content */}
        <div className="p-4">
          {step === 'select' ? (
            <>
              {pricingLoading ? (
                <div className="flex items-center justify-center py-8">
                  <Loader2 size={24} className="animate-spin text-white" />
                </div>
              ) : (
                <>
                  <p className="text-sm text-white mb-4">
                    Enter the amount you'd like to add. You'll receive exactly what you pay.
                  </p>

                  <AmountInput
                    amount={amount}
                    onChange={setAmount}
                    minAmount={minPurchase}
                    maxAmount={maxPurchase}
                  />

                  {paymentError && (
                    <div className="mt-4 flex items-center gap-2 p-3 bg-red-500/10 border border-red-500/30 text-red-400 text-sm">
                      <AlertCircle size={16} />
                      {paymentError}
                    </div>
                  )}

                  <button
                    onClick={handleProceedToCheckout}
                    disabled={!isValidAmount || paymentLoading}
                    className="w-full mt-4 px-4 py-3 bg-blue-600 hover:bg-blue-700 disabled:bg-gray-700 disabled:cursor-not-allowed text-white font-medium transition-colors flex items-center justify-center gap-2"
                  >
                    {paymentLoading ? (
                      <>
                        <Loader2 size={18} className="animate-spin" />
                        Loading...
                      </>
                    ) : (
                      <>
                        Continue to Payment
                        {isValidAmount && (
                          <span className="text-blue-200">
                            (${numericAmount.toFixed(2)})
                          </span>
                        )}
                      </>
                    )}
                  </button>
                </>
              )}
            </>
          ) : (
            <>
              {clientSecret && stripePromise && paymentIntentId && (
                <Elements
                  stripe={stripePromise}
                  options={{
                    clientSecret,
                    appearance: {
                      theme: 'night',
                      variables: {
                        colorPrimary: '#3b82f6',
                        colorBackground: '#000000',
                        colorText: '#ffffff',
                        colorDanger: '#ef4444',
                        fontFamily: 'system-ui, sans-serif',
                        borderRadius: '0px',
                      },
                    },
                  }}
                >
                  <CheckoutForm
                    clientSecret={clientSecret}
                    paymentIntentId={paymentIntentId}
                    amount={numericAmount}
                    onSuccess={handlePaymentSuccess}
                    onCancel={handleBack}
                  />
                </Elements>
              )}
            </>
          )}
        </div>
      </div>
    </div>
  );
}
