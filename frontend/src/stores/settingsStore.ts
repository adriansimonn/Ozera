/**
 * Settings store using Zustand.
 * Manages user preferences synced with the backend.
 */
import { create } from 'zustand';
import { apiClient } from '../api/axios';

export interface UserSettings {
  credits: {
    low_balance_alert: number;
    show_balance_in_navbar: boolean;
  };
  ui: {
    theme: 'dark' | 'light' | 'liquid_glass' | 'system';
    default_view_mode: 'single' | 'split';
    default_page: string;
  };
}

interface SettingsState {
  settings: UserSettings | null;
  isLoading: boolean;

  fetchSettings: () => Promise<void>;
  updateSettings: (updates: Record<string, any>) => Promise<void>;
  resetSettings: () => Promise<void>;
  clear: () => void;
}

export const useSettingsStore = create<SettingsState>()((set) => ({
  settings: null,
  isLoading: false,

  fetchSettings: async () => {
    set({ isLoading: true });
    try {
      const response = await apiClient.get('/settings');
      set({ settings: response.data });
    } catch (error) {
      console.error('Failed to fetch settings:', error);
    } finally {
      set({ isLoading: false });
    }
  },

  updateSettings: async (updates: Record<string, any>) => {
    // Optimistic update: apply changes to local state immediately
    const previous = useSettingsStore.getState().settings;
    if (previous) {
      const optimistic = { ...previous };
      for (const [section, values] of Object.entries(updates)) {
        if (section in optimistic && typeof values === 'object' && values !== null) {
          (optimistic as any)[section] = { ...(optimistic as any)[section], ...values };
        }
      }
      set({ settings: optimistic });
    }

    try {
      const response = await apiClient.patch('/settings', { settings: updates });
      set({ settings: response.data });
    } catch (error) {
      // Revert to previous state on failure
      if (previous) set({ settings: previous });
      console.error('Failed to update settings:', error);
      throw error;
    }
  },

  resetSettings: async () => {
    try {
      const response = await apiClient.post('/settings/reset');
      set({ settings: response.data });
    } catch (error) {
      console.error('Failed to reset settings:', error);
      throw error;
    }
  },

  clear: () => {
    set({ settings: null });
  },
}));
