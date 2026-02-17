/**
 * Signup modal component.
 */
import { useState } from 'react';
import { X } from 'lucide-react';
import { useAuthStore } from '../../stores/authStore';

interface SignupModalProps {
  isOpen: boolean;
  onClose: () => void;
  onSwitchToLogin: () => void;
}

export function SignupModal({ isOpen, onClose, onSwitchToLogin }: SignupModalProps) {
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [confirmPassword, setConfirmPassword] = useState('');
  const [fullName, setFullName] = useState('');
  const [error, setError] = useState<string | null>(null);
  const [checkEmail, setCheckEmail] = useState(false);
  const { signup, isLoading } = useAuthStore();

  if (!isOpen) return null;

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setError(null);

    // Validate passwords match
    if (password !== confirmPassword) {
      setError('Passwords do not match');
      return;
    }

    // Validate password length
    if (password.length < 8) {
      setError('Password must be at least 8 characters');
      return;
    }

    try {
      await signup(email, password, fullName || undefined);
      onClose();
      // Reset form
      setEmail('');
      setPassword('');
      setConfirmPassword('');
      setFullName('');
    } catch (err: any) {
      if (err.message === 'CHECK_EMAIL') {
        setCheckEmail(true);
      } else {
        setError(err.message);
      }
    }
  };

  const handleClose = () => {
    setEmail('');
    setPassword('');
    setConfirmPassword('');
    setFullName('');
    setError(null);
    setCheckEmail(false);
    onClose();
  };

  return (
    <div className="fixed inset-0 bg-black/70 backdrop-blur-sm flex items-center justify-center z-50 p-4">
      <div className="bg-black/60 backdrop-filter backdrop-blur-xl border border-gray-800 max-w-md w-full p-8 relative">
        {/* Close button */}
        <button
          onClick={handleClose}
          className="absolute top-6 right-6 text-gray-500 hover:text-white transition-colors"
          aria-label="Close"
        >
          <X size={20} />
        </button>

        {/* Header */}
        <h2 className="text-2xl font-bold text-white mb-8 tracking-tight">Create Account</h2>

        {/* Error message */}
        {error && (
          <div className="mb-6 p-4 bg-red-500/10 border border-red-500/30 text-red-400 text-sm">
            {error}
          </div>
        )}

        {/* Email confirmation message */}
        {checkEmail && (
          <div className="mb-6 p-4 bg-emerald-500/10 border border-emerald-500/30 text-emerald-400 text-sm">
            Check your email for a confirmation link to complete signup.
          </div>
        )}

        {/* Form */}
        <form onSubmit={handleSubmit} className="space-y-5">
          <div>
            <label htmlFor="fullName" className="block text-sm font-medium text-gray-400 mb-2 uppercase tracking-wide text-xs">
              Full Name <span className="text-gray-600">(optional)</span>
            </label>
            <input
              type="text"
              id="fullName"
              value={fullName}
              onChange={(e) => setFullName(e.target.value)}
              className="w-full px-4 py-3 bg-black/40 border border-gray-700 text-white placeholder-gray-600 focus:outline-none focus:border-gray-500 transition-colors"
              placeholder="John Doe"
              disabled={isLoading}
            />
          </div>

          <div>
            <label htmlFor="signup-email" className="block text-sm font-medium text-gray-400 mb-2 uppercase tracking-wide text-xs">
              Email
            </label>
            <input
              type="email"
              id="signup-email"
              value={email}
              onChange={(e) => setEmail(e.target.value)}
              required
              className="w-full px-4 py-3 bg-black/40 border border-gray-700 text-white placeholder-gray-600 focus:outline-none focus:border-gray-500 transition-colors"
              placeholder="you@example.com"
              disabled={isLoading}
            />
          </div>

          <div>
            <label htmlFor="signup-password" className="block text-sm font-medium text-gray-400 mb-2 uppercase tracking-wide text-xs">
              Password
            </label>
            <input
              type="password"
              id="signup-password"
              value={password}
              onChange={(e) => setPassword(e.target.value)}
              required
              minLength={8}
              className="w-full px-4 py-3 bg-black/40 border border-gray-700 text-white placeholder-gray-600 focus:outline-none focus:border-gray-500 transition-colors"
              placeholder="••••••••"
              disabled={isLoading}
            />
            <p className="mt-2 text-xs text-gray-600">Minimum 8 characters</p>
          </div>

          <div>
            <label htmlFor="confirm-password" className="block text-sm font-medium text-gray-400 mb-2 uppercase tracking-wide text-xs">
              Confirm Password
            </label>
            <input
              type="password"
              id="confirm-password"
              value={confirmPassword}
              onChange={(e) => setConfirmPassword(e.target.value)}
              required
              minLength={8}
              className="w-full px-4 py-3 bg-black/40 border border-gray-700 text-white placeholder-gray-600 focus:outline-none focus:border-gray-500 transition-colors"
              placeholder="••••••••"
              disabled={isLoading}
            />
          </div>

          <button
            type="submit"
            disabled={isLoading}
            className="w-full py-3 px-4 bg-white/10 hover:bg-white/15 disabled:bg-gray-800 disabled:cursor-not-allowed text-white font-medium border border-gray-700 hover:border-gray-600 transition-all mt-6"
          >
            {isLoading ? 'Creating account...' : 'Sign Up'}
          </button>
        </form>

        {/* Switch to login */}
        <div className="mt-6 text-center text-sm text-gray-500">
          Already have an account?{' '}
          <button
            onClick={onSwitchToLogin}
            className="text-gray-300 hover:text-white font-medium transition-colors"
          >
            Login
          </button>
        </div>
      </div>
    </div>
  );
}
