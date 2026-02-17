/**
 * Axios instance with Supabase auth interceptor.
 */
import axios from 'axios';
import { getSupabaseToken, supabase } from '../lib/supabase';

const API_BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000';

export const apiClient = axios.create({
  baseURL: API_BASE_URL,
  headers: {
    'Content-Type': 'application/json',
  },
});

// Request interceptor — attach Supabase access token
apiClient.interceptors.request.use(
  (config) => {
    const token = getSupabaseToken();
    if (token) {
      config.headers.Authorization = `Bearer ${token}`;
    }
    return config;
  },
  (error) => Promise.reject(error),
);

// Response interceptor — sign out on 401
apiClient.interceptors.response.use(
  (response) => response,
  (error) => {
    if (error.response?.status === 401) {
      supabase.auth.signOut();
    }
    return Promise.reject(error);
  },
);

export default apiClient;
