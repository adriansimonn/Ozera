import { useEffect, useState } from 'react'
import { BrowserRouter as Router, Routes, Route, useLocation } from 'react-router-dom'
import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import UnifiedPage from './pages/UnifiedPage'
import TrainingPage from './pages/TrainingPage'
import PatchingPlayground from './pages/PatchingPlayground'
import AnalysisPage from './pages/AnalysisPage'
import SAEPage from './pages/SAEPage'
import DefaultUnifiedPage from './pages/default/UnifiedPage'
import DefaultTrainingPage from './pages/default/TrainingPage'
import DefaultPatchingPlayground from './pages/default/PatchingPlayground'
import DefaultAnalysisPage from './pages/default/AnalysisPage'
import DefaultSAEPage from './pages/default/SAEPage'
import SettingsPage from './pages/SettingsPage'
import AuthPage from './pages/AuthPage'
import { AnimatedBackground } from './components/common/AnimatedBackground'
import { PurchaseCreditsModal } from './components/payments/PurchaseCreditsModal'
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
  const { background, interfaceStyle, isGlow, isDefault } = useTheme()
  const location = useLocation()
  const isAuthPage = location.pathname === '/auth'

  // Hydrate auth state from Supabase session on mount
  useEffect(() => {
    initialize()
  }, [initialize])

  // Make purchase credits modal available globally via window
  useEffect(() => {
    ;(window as any).showPurchaseCreditsModal = () => setShowPurchaseCredits(true)
  }, [])

  return (
    <div className="min-h-screen bg-black" data-bg={isAuthPage ? undefined : background} data-interface={isAuthPage ? undefined : interfaceStyle}>
      {isGlow && !isAuthPage && <AnimatedBackground />}
      <main className="content-container">
            <Routes>
              <Route
                path="/"
                element={
                  isDefault ? (
                    <DefaultUnifiedPage
                      onShowPurchaseCredits={() => setShowPurchaseCredits(true)}
                    />
                  ) : (
                    <UnifiedPage
                      onShowPurchaseCredits={() => setShowPurchaseCredits(true)}
                    />
                  )
                }
              />
              <Route
                path="/training"
                element={isDefault ? <DefaultTrainingPage /> : <TrainingPage />}
              />
              <Route
                path="/patching"
                element={
                  isDefault ? (
                    <DefaultPatchingPlayground
                      onShowPurchaseCredits={() => setShowPurchaseCredits(true)}
                    />
                  ) : (
                    <PatchingPlayground
                      onShowPurchaseCredits={() => setShowPurchaseCredits(true)}
                    />
                  )
                }
              />
              <Route
                path="/analysis"
                element={
                  isDefault ? (
                    <DefaultAnalysisPage
                      onShowPurchaseCredits={() => setShowPurchaseCredits(true)}
                    />
                  ) : (
                    <AnalysisPage
                      onShowPurchaseCredits={() => setShowPurchaseCredits(true)}
                    />
                  )
                }
              />
              <Route
                path="/sae"
                element={
                  isDefault ? (
                    <DefaultSAEPage
                      onShowPurchaseCredits={() => setShowPurchaseCredits(true)}
                    />
                  ) : (
                    <SAEPage
                      onShowPurchaseCredits={() => setShowPurchaseCredits(true)}
                    />
                  )
                }
              />
              <Route
                path="/settings"
                element={<SettingsPage />}
              />
              <Route
                path="/auth"
                element={<AuthPage />}
              />
            </Routes>
          </main>

          {/* Purchase Credits Modal */}
          <PurchaseCreditsModal
            isOpen={showPurchaseCredits}
            onClose={() => setShowPurchaseCredits(false)}
          />
        </div>
  )
}

function App() {
  return (
    <QueryClientProvider client={queryClient}>
      <Router>
        <AppShell />
      </Router>
    </QueryClientProvider>
  )
}

export default App
