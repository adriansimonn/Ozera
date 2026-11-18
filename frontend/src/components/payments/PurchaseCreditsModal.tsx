/**
 * Modal for purchasing credits via Stripe.
 */
import { useState, useEffect } from 'react';
import { X, Check, AlertCircle } from 'lucide-react';
import { loadStripe } from '@stripe/stripe-js';
import {
  Elements,
  PaymentElement,
  useStripe,
  useElements,
} from '@stripe/react-stripe-js';
import { usePricing, usePayment, useStripeConfig } from '../../hooks/useCredits';
import { useAuthStore } from '../../stores/authStore';
import { useAppPalette } from '../../hooks/useTheme';
import { Spinner } from '../common/Spinner'

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
    <div className="pc-amount">
      <div className="pc-amount-field">
        <span aria-hidden="true">$</span>
        <input
          type="number"
          value={amount}
          onChange={(e) => onChange(e.target.value)}
          placeholder="0.00"
          min={minAmount}
          max={maxAmount}
          step="0.01"
          aria-label="Amount in dollars"
        />
      </div>

      {amount && !isValid ? (
        <p className="pc-error">
          Amount must be between ${minAmount} and ${maxAmount}
        </p>
      ) : (
        <p className="oz-hint">
          ${minAmount} to ${maxAmount}. You receive exactly what you pay.
        </p>
      )}

      {/* Quick amount buttons */}
      <div className="oz-chips">
        {[10, 25, 50, 100].map((quickAmount) => (
          <button
            key={quickAmount}
            type="button"
            onClick={() => onChange(quickAmount.toString())}
            className="oz-chip"
            aria-pressed={parseFloat(amount) === quickAmount}
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
      <div className="oz-note oz-note--ok">
        <Check aria-hidden="true" />
        <span>Payment successful. ${amount.toFixed(2)} in credits have been added to your account.</span>
      </div>
    );
  }

  return (
    <form onSubmit={handleSubmit} className="pc-form">
      <div className="pc-summary">
        <span className="oz-hint">Payment amount</span>
        <span className="pc-summary-value">${amount.toFixed(2)}</span>
      </div>

      <PaymentElement
        options={{
          layout: 'tabs',
        }}
      />

      {error && (
        <div className="oz-note oz-note--error" role="alert">
          <AlertCircle aria-hidden="true" />
          <span>{error}</span>
        </div>
      )}

      <div className="oz-modal__actions">
        <button type="button" onClick={onCancel} disabled={processing} className="oz-btn oz-btn--ghost">
          Back
        </button>
        <button type="submit" disabled={!stripe || processing} className={`oz-btn${processing ? '' : ' oz-btn--arrow'}`}>
          {processing ? (
            <>
              <Spinner />
              Processing...
            </>
          ) : (
            `Pay $${amount.toFixed(2)}`
          )}
        </button>
      </div>
    </form>
  );
}

export function PurchaseCreditsModal({ isOpen, onClose }: PurchaseCreditsModalProps) {
  const { isAuthenticated } = useAuthStore();
  const palette = useAppPalette();
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
      <div className="oz-overlay" onClick={onClose}>
        <div className="oz-modal" role="dialog" aria-modal="true" onClick={(e) => e.stopPropagation()}>
          <h2 className="oz-modal__title">Add credits</h2>
          <p className="oz-modal__text">Please log in to purchase credits.</p>
          <div className="oz-modal__actions">
            <button onClick={onClose} className="oz-btn oz-btn--ghost">
              Close
            </button>
          </div>
        </div>
      </div>
    );
  }

  if (!isConfigured) {
    return (
      <div className="oz-overlay" onClick={onClose}>
        <div className="oz-modal" role="dialog" aria-modal="true" onClick={(e) => e.stopPropagation()}>
          <h2 className="oz-modal__title">Payment not available</h2>
          <p className="oz-modal__text">Payment processing is not configured. Please contact support.</p>
          <div className="oz-modal__actions">
            <button onClick={onClose} className="oz-btn oz-btn--ghost">
              Close
            </button>
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className="oz-overlay">
      <div className="oz-modal purchase-credits" role="dialog" aria-modal="true">
        <button onClick={onClose} className="oz-icon-btn oz-modal__close" aria-label="Close">
          <X />
        </button>
        <h2 className="oz-modal__title">{step === 'select' ? 'Add credits' : 'Checkout'}</h2>

        {step === 'select' ? (
          <>
            {pricingLoading ? (
              <div className="oz-empty">
                <Spinner large />
              </div>
            ) : (
              <>
                <AmountInput
                  amount={amount}
                  onChange={setAmount}
                  minAmount={minPurchase}
                  maxAmount={maxPurchase}
                />

                {paymentError && (
                  <div className="oz-note oz-note--error pc-note" role="alert">
                    <AlertCircle aria-hidden="true" />
                    <span>{paymentError}</span>
                  </div>
                )}

                <button
                  onClick={handleProceedToCheckout}
                  disabled={!isValidAmount || paymentLoading}
                  className={`oz-btn oz-btn--block pc-continue${paymentLoading ? '' : ' oz-btn--arrow'}`}
                >
                  {paymentLoading ? (
                    <>
                      <Spinner />
                      Loading...
                    </>
                  ) : (
                    <>Continue to Payment{isValidAmount && ` ($${numericAmount.toFixed(2)})`}</>
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
                  // Stripe's own fields, in the app's colors
                  appearance: {
                    theme: palette.isLight ? 'stripe' : 'night',
                    variables: {
                      colorPrimary: palette.fg,
                      colorBackground: palette.bg,
                      colorText: palette.fg,
                      colorDanger: palette.danger,
                      fontFamily: 'Inter, system-ui, sans-serif',
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

        <style>{`
          .purchase-credits .pc-amount { display: flex; flex-direction: column; gap: 0.9rem; }
          /* The amount, in the homepage's display type */
          .purchase-credits .pc-amount-field {
            display: flex;
            align-items: baseline;
            gap: 0.35rem;
            padding-bottom: 0.5rem;
            border-bottom: 1px solid var(--oz-rule-strong);
            font-size: 3.5rem;
            font-weight: 500;
            letter-spacing: -0.04em;
            line-height: 1;
            transition: border-color 0.2s ease;
          }
          .purchase-credits .pc-amount-field:focus-within { border-color: var(--oz-fg); }
          .purchase-credits .pc-amount-field span { color: var(--oz-fg-faint); }
          .purchase-credits .pc-amount-field input {
            flex: 1;
            min-width: 0;
            background: transparent;
            border: none;
            outline: none;
            color: var(--oz-fg);
            font: inherit;
            letter-spacing: inherit;
            font-variant-numeric: tabular-nums;
          }
          .purchase-credits .pc-amount-field input::placeholder { color: var(--oz-fg-faint); }
          .purchase-credits .pc-error { color: var(--oz-danger); font-size: 0.875rem; }
          .purchase-credits .pc-note { margin-top: 1rem; }
          .purchase-credits .pc-continue { margin-top: 1.5rem; }
          .purchase-credits .pc-form { display: flex; flex-direction: column; gap: 1.25rem; }
          .purchase-credits .pc-summary { display: flex; align-items: baseline; justify-content: space-between; padding-bottom: 0.9rem; border-bottom: 1px solid var(--oz-rule); }
          .purchase-credits .pc-summary-value { font-size: 2rem; font-weight: 500; letter-spacing: -0.035em; line-height: 1; font-variant-numeric: tabular-nums; }
        `}</style>
      </div>
    </div>
  );
}
