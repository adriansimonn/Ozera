import { useEffect, useState, useCallback } from 'react'
import { BrowserRouter as Router, Routes, Route, useLocation } from 'react-router-dom'
import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import UnifiedPage from './pages/UnifiedPage'
import TrainingPage from './pages/TrainingPage'
import PatchingPlayground from './pages/PatchingPlayground'
import AnalysisPage from './pages/AnalysisPage'
import SAEPage from './pages/SAEPage'
import ProbesPage from './pages/ProbesPage'
import GeometryPage from './pages/GeometryPage'
import SettingsPage from './pages/SettingsPage'
import AuthPage from './pages/AuthPage'
import { ErrorBoundary } from './components/common/ErrorBoundary'
import { PurchaseCreditsModal } from './components/payments/PurchaseCreditsModal'
import { PurchaseCreditsProvider } from './contexts/PurchaseCreditsContext'
import { AssistantHost } from './components/assistant/AssistantPanel'
import { useAuthStore } from './stores/authStore'
import { useTheme } from './hooks/useTheme'

const queryClient = new QueryClient({
  defaultOptions: {
    queries: {
      refetchOnWindowFocus: false,
      retry: 1,
    },
  },
})

function AppShell() {
  const [showPurchaseCredits, setShowPurchaseCredits] = useState(false)
  const { initialize } = useAuthStore()
  const { background } = useTheme()
  const location = useLocation()
  const isAuthPage = location.pathname === '/auth'

  const handleShowPurchaseCredits = useCallback(() => setShowPurchaseCredits(true), [])

  // Hydrate auth state from Supabase session on mount
  useEffect(() => {
    initialize()
  }, [initialize])

  return (
    <PurchaseCreditsProvider onShow={handleShowPurchaseCredits}>
    <div className="min-h-screen bg-black" data-bg={isAuthPage ? undefined : background}>
      <main className="content-container">
            <Routes>
              <Route path="/" element={<UnifiedPage onShowPurchaseCredits={() => setShowPurchaseCredits(true)} />} />
              <Route path="/training" element={<TrainingPage />} />
              <Route path="/patching" element={<PatchingPlayground onShowPurchaseCredits={() => setShowPurchaseCredits(true)} />} />
              <Route path="/analysis" element={<AnalysisPage onShowPurchaseCredits={() => setShowPurchaseCredits(true)} />} />
              <Route path="/sae" element={<SAEPage onShowPurchaseCredits={() => setShowPurchaseCredits(true)} />} />
              <Route path="/probes" element={<ProbesPage onShowPurchaseCredits={() => setShowPurchaseCredits(true)} />} />
              <Route path="/geometry" element={<GeometryPage onShowPurchaseCredits={() => setShowPurchaseCredits(true)} />} />
              <Route path="/settings" element={<SettingsPage />} />
              <Route path="/auth" element={<AuthPage />} />
            </Routes>
          </main>

          {/* The assistant: outside the routes, so it stays open across pages */}
          {!isAuthPage && <AssistantHost />}

          {/* Purchase Credits Modal */}
          <PurchaseCreditsModal
            isOpen={showPurchaseCredits}
            onClose={() => setShowPurchaseCredits(false)}
          />
        </div>
    </PurchaseCreditsProvider>
  )
}

function App() {
  return (
    <ErrorBoundary>
      <QueryClientProvider client={queryClient}>
        <Router>
          <AppShell />
        </Router>
      </QueryClientProvider>
    </ErrorBoundary>
  )
}

export default App
