/**
 * Protected route component that requires authentication.
 */
import { ReactNode } from 'react';
import { Navigate } from 'react-router-dom';
import { useAuthStore } from '../../stores/authStore';

interface ProtectedRouteProps {
  children: ReactNode;
  redirectTo?: string;
}

export function ProtectedRoute({ children, redirectTo = '/' }: ProtectedRouteProps) {
  const { isAuthenticated } = useAuthStore();

  if (!isAuthenticated) {
    // Redirect to home page (or specified redirect)
    // User can login from there
    return <Navigate to={redirectTo} replace />;
  }

  return <>{children}</>;
}
