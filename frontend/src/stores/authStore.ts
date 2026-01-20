/**
 * Authentication store using Zustand.
 * Manages user authentication state, tokens, and credit balance.
 */
import { create } from 'zustand';
import { persist } from 'zustand/middleware';
import { apiClient } from '../api/axios';

export interface User {
  id: number;
  email: string;
  full_name: string | null;
  created_at: string;
  is_active: boolean;
  is_verified: boolean;
  balance_usd: number;
  reserved_usd: number;
  available_balance: number;
}

interface AuthState {
  // State
  user: User | null;
  token: string | null;
  isAuthenticated: boolean;
  isLoading: boolean;

  // Actions
  login: (email: string, password: string) => Promise<void>;
  signup: (email: string, password: string, fullName?: string) => Promise<void>;
  logout: () => void;
  refreshUser: () => Promise<void>;
  setToken: (token: string) => void;
  clearAuth: () => void;
}

export const useAuthStore = create<AuthState>()(
  persist(
    (set, get) => ({
      // Initial state
      user: null,
      token: null,
      isAuthenticated: false,
      isLoading: false,

      // Login action
      login: async (email: string, password: string) => {
        set({ isLoading: true });
        try {
          const response = await apiClient.post('/auth/login', {
            email,
            password,
          });

          const { access_token } = response.data;
          set({ token: access_token });

          // Fetch user info
          const userResponse = await apiClient.get('/auth/me', {
            headers: {
              Authorization: `Bearer ${access_token}`,
            },
          });

          set({
            user: userResponse.data,
            isAuthenticated: true,
            isLoading: false,
          });
        } catch (error: any) {
          set({ isLoading: false });
          throw new Error(
            error.response?.data?.detail || 'Login failed. Please check your credentials.'
          );
        }
      },

      // Signup action
      signup: async (email: string, password: string, fullName?: string) => {
        set({ isLoading: true });
        try {
          const response = await apiClient.post('/auth/signup', {
            email,
            password,
            full_name: fullName,
          });

          // Auto-login after signup
          const user = response.data;

          // Get token by logging in
          const loginResponse = await apiClient.post('/auth/login', {
            email,
            password,
          });

          const { access_token } = loginResponse.data;

          set({
            user,
            token: access_token,
            isAuthenticated: true,
            isLoading: false,
          });
        } catch (error: any) {
          set({ isLoading: false });
          throw new Error(
            error.response?.data?.detail || 'Signup failed. Please try again.'
          );
        }
      },

      // Logout action
      logout: () => {
        set({
          user: null,
          token: null,
          isAuthenticated: false,
        });
      },

      // Refresh user info (e.g., after credit purchase)
      refreshUser: async () => {
        const { token } = get();
        if (!token) return;

        try {
          const response = await apiClient.get('/auth/me', {
            headers: {
              Authorization: `Bearer ${token}`,
            },
          });

          set({ user: response.data });
        } catch (error) {
          console.error('Failed to refresh user info:', error);
          // If refresh fails due to invalid token, logout
          if ((error as any).response?.status === 401) {
            get().logout();
          }
        }
      },

      // Set token (useful for initial hydration)
      setToken: (token: string) => {
        set({ token });
      },

      // Clear auth state
      clearAuth: () => {
        set({
          user: null,
          token: null,
          isAuthenticated: false,
        });
      },
    }),
    {
      name: 'auth-storage', // LocalStorage key
      partialize: (state) => ({
        // Only persist token, will fetch user on app load
        token: state.token,
      }),
    }
  )
);
