/**
 * Authentication store using Zustand.
 * Manages user state via Supabase Auth — token persistence is handled
 * by Supabase's own localStorage keys, not by this store.
 */
import { create } from 'zustand';
import { supabase } from '../lib/supabase';
import { apiClient } from '../api/axios';
import { useSettingsStore } from './settingsStore';

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
  isAuthenticated: boolean;
  isLoading: boolean;

  // Actions
  initialize: () => Promise<void>;
  login: (email: string, password: string) => Promise<void>;
  signup: (email: string, password: string, fullName?: string) => Promise<void>;
  logout: () => Promise<void>;
  refreshUser: () => Promise<void>;
}

export const useAuthStore = create<AuthState>()((set, get) => ({
  // Initial state
  user: null,
  isAuthenticated: false,
  isLoading: false,

  // Called once on app mount — hydrates user from existing Supabase session
  initialize: async () => {
    set({ isLoading: true });
    try {
      const { data: { session } } = await supabase.auth.getSession();
      if (session) {
        const response = await apiClient.get('/auth/me');
        set({ user: response.data, isAuthenticated: true });
        useSettingsStore.getState().fetchSettings();
      }
    } catch (error) {
      console.error('Failed to initialize auth:', error);
    } finally {
      set({ isLoading: false });
    }

    // Keep store in sync when Supabase session changes (token refresh, sign-out in another tab, etc.)
    supabase.auth.onAuthStateChange(async (event, session) => {
      if (event === 'SIGNED_OUT' || !session) {
        set({ user: null, isAuthenticated: false });
        return;
      }

      if (event === 'SIGNED_IN' || event === 'TOKEN_REFRESHED') {
        try {
          const response = await apiClient.get('/auth/me');
          set({ user: response.data, isAuthenticated: true });
          useSettingsStore.getState().fetchSettings();
        } catch {
          set({ user: null, isAuthenticated: false });
        }
      }
    });
  },

  login: async (email: string, password: string) => {
    set({ isLoading: true });
    try {
      const { error } = await supabase.auth.signInWithPassword({ email, password });
      if (error) throw new Error(error.message);

      // Fetch local user record from our backend
      const response = await apiClient.get('/auth/me');
      set({ user: response.data, isAuthenticated: true, isLoading: false });
      useSettingsStore.getState().fetchSettings();
    } catch (error: any) {
      set({ isLoading: false });
      throw new Error(error.message || 'Login failed. Please check your credentials.');
    }
  },

  signup: async (email: string, password: string, fullName?: string) => {
    set({ isLoading: true });
    try {
      const { data, error } = await supabase.auth.signUp({
        email,
        password,
        options: {
          data: { full_name: fullName },
          emailRedirectTo: window.location.origin,
        },
      });
      if (error) throw new Error(error.message);

      // If Supabase requires email confirmation, session will be null
      if (!data.session) {
        set({ isLoading: false });
        throw new Error('CHECK_EMAIL');
      }

      // Session exists — fetch local user record
      const response = await apiClient.get('/auth/me');
      set({ user: response.data, isAuthenticated: true, isLoading: false });
    } catch (error: any) {
      set({ isLoading: false });
      throw error;
    }
  },

  logout: async () => {
    await supabase.auth.signOut();
    set({ user: null, isAuthenticated: false });
    useSettingsStore.getState().clear();
  },

  refreshUser: async () => {
    const { data: { session } } = await supabase.auth.getSession();
    if (!session) return;

    try {
      const response = await apiClient.get('/auth/me');
      set({ user: response.data, isAuthenticated: true });
    } catch (error) {
      console.error('Failed to refresh user info:', error);
      if ((error as any).response?.status === 401) {
        await get().logout();
      }
    }
  },
}));
