/**
 * Login modal component.
 */
import { useState } from 'react';
import { X } from 'lucide-react';
import { useAuthStore } from '../../stores/authStore';

interface LoginModalProps {
  isOpen: boolean;
  onClose: () => void;
  onSwitchToSignup: () => void;
}

export function LoginModal({ isOpen, onClose, onSwitchToSignup }: LoginModalProps) {
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [error, setError] = useState<string | null>(null);
  const { login, isLoading } = useAuthStore();

  if (!isOpen) return null;

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setError(null);

    try {
      await login(email, password);
      onClose();
      // Reset form
      setEmail('');
      setPassword('');
    } catch (err: any) {
      setError(err.message);
    }
  };

  const handleClose = () => {
    setEmail('');
    setPassword('');
    setError(null);
    onClose();
  };

  return (
    <div className="fixed inset-0 bg-black/70 backdrop-blur-sm flex items-center justify-center z-50 p-4">
      <div className="bg-black/60 backdrop-filter backdrop-blur-xl border border-gray-800 max-w-md w-full p-8 relative">
        {/* Close button */}
        <button
          onClick={handleClose}
          className="absolute top-6 right-6 text-white hover:text-white transition-colors"
          aria-label="Close"
        >
          <X size={20} />
        </button>

        {/* Header */}
        <h2 className="text-2xl font-bold text-white mb-8 tracking-tight">Login to Ozera</h2>

        {/* Error message */}
        {error && (
          <div className="mb-6 p-4 bg-red-500/10 border border-red-500/30 text-red-400 text-sm">
            {error}
          </div>
        )}

        {/* Form */}
        <form onSubmit={handleSubmit} className="space-y-5">
          <div>
            <label htmlFor="email" className="block text-sm font-medium text-white mb-2 uppercase tracking-wide text-xs">
              Email
            </label>
            <input
              type="email"
              id="email"
              value={email}
              onChange={(e) => setEmail(e.target.value)}
              required
              className="w-full px-4 py-3 bg-black/40 border border-gray-700 text-white placeholder-gray-500 focus:outline-none focus:border-gray-500 transition-colors"
              placeholder="you@example.com"
              disabled={isLoading}
            />
          </div>

          <div>
            <label htmlFor="password" className="block text-sm font-medium text-white mb-2 uppercase tracking-wide text-xs">
              Password
            </label>
            <input
              type="password"
              id="password"
              value={password}
              onChange={(e) => setPassword(e.target.value)}
              required
              minLength={8}
              className="w-full px-4 py-3 bg-black/40 border border-gray-700 text-white placeholder-gray-500 focus:outline-none focus:border-gray-500 transition-colors"
              placeholder="••••••••"
              disabled={isLoading}
            />
          </div>

          <button
            type="submit"
            disabled={isLoading}
            className="w-full py-3 px-4 bg-white/10 hover:bg-white/15 disabled:bg-gray-800 disabled:cursor-not-allowed text-white font-medium border border-gray-700 hover:border-gray-600 transition-all mt-6"
          >
            {isLoading ? 'Logging in...' : 'Login'}
          </button>
        </form>

        {/* Switch to signup */}
        <div className="mt-6 text-center text-sm text-white">
          Don't have an account?{' '}
          <button
            onClick={onSwitchToSignup}
            className="text-white hover:text-white font-medium transition-colors"
          >
            Sign up
          </button>
        </div>
      </div>
    </div>
  );
}
