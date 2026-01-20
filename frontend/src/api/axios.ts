/**
 * Axios instance with authentication interceptor.
 */
import axios from 'axios';

const API_BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000';

export const apiClient = axios.create({
  baseURL: API_BASE_URL,
  headers: {
    'Content-Type': 'application/json',
  },
});

// Token getter function - will be set by authStore
let getAuthToken: (() => string | null) | null = null;

export const setAuthTokenGetter = (getter: () => string | null) => {
  getAuthToken = getter;
};

// Request interceptor to add JWT token
apiClient.interceptors.request.use(
  (config) => {
    // Try to get token from the getter function (in-memory from Zustand)
    let token: string | null = null;

    if (getAuthToken) {
      token = getAuthToken();
    }

    // Fallback to localStorage if getter not available
    if (!token) {
      const authStorage = localStorage.getItem('auth-storage');
      if (authStorage) {
        try {
          const { state } = JSON.parse(authStorage);
          token = state?.token;
        } catch (error) {
          console.error('Failed to parse auth storage:', error);
        }
      }
    }

    if (token) {
      config.headers.Authorization = `Bearer ${token}`;
    }

    return config;
  },
  (error) => {
    return Promise.reject(error);
  }
);

// Response interceptor to handle 401 errors (unauthorized)
apiClient.interceptors.response.use(
  (response) => response,
  (error) => {
    if (error.response?.status === 401) {
      // Token expired or invalid - clear auth
      localStorage.removeItem('auth-storage');

      // Optionally redirect to home or show login modal
      // window.location.href = '/';
    }

    return Promise.reject(error);
  }
);

export default apiClient;
