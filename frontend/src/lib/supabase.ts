import { createClient } from "@supabase/supabase-js";

const supabaseUrl = import.meta.env.VITE_SUPABASE_URL as string;
const supabaseAnonKey = import.meta.env.VITE_SUPABASE_ANON_KEY as string;

if (!supabaseUrl || !supabaseAnonKey) {
  throw new Error("Missing VITE_SUPABASE_URL or VITE_SUPABASE_ANON_KEY env vars");
}

export const supabase = createClient(supabaseUrl, supabaseAnonKey);

// Cached access token — kept in sync via onAuthStateChange listener.
// This allows getSupabaseToken() to stay synchronous so we don't need
// to make getAuthHeaders() async and touch ~25 call sites in client.ts.
let currentAccessToken: string | null = null;

supabase.auth.onAuthStateChange((_event, session) => {
  currentAccessToken = session?.access_token ?? null;
});

// Hydrate on module load so the token is available immediately if
// Supabase already has a persisted session in localStorage.
supabase.auth.getSession().then(({ data: { session } }) => {
  currentAccessToken = session?.access_token ?? null;
});

export function getSupabaseToken(): string | null {
  return currentAccessToken;
}
