import { useEffect, useState } from 'react'
import { BrowserRouter as Router, Routes, Route } from 'react-router-dom'
import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import UnifiedPage from './pages/UnifiedPage'
import TrainingPage from './pages/TrainingPage'
import PatchingPlayground from './pages/PatchingPlayground'
import AnalysisPage from './pages/AnalysisPage'
import SAEPage from './pages/SAEPage'
import SettingsPage from './pages/SettingsPage'
import AuthPage from './pages/AuthPage'
import { AnimatedBackground } from './components/common/AnimatedBackground'
import { PurchaseCreditsModal } from './components/payments/PurchaseCreditsModal'
import { useAuthStore } from './stores/authStore'

const queryClient = new QueryClient({
  defaultOptions: {
    queries: {
      refetchOnWindowFocus: false,
      retry: 1,
    },
  },
})

function App() {
  const [showPurchaseCredits, setShowPurchaseCredits] = useState(false)
  const { initialize } = useAuthStore()

  // Hydrate auth state from Supabase session on mount
  useEffect(() => {
    initialize()
  }, [initialize])

  // Make purchase credits modal available globally via window
  useEffect(() => {
    ;(window as any).showPurchaseCreditsModal = () => setShowPurchaseCredits(true)
  }, [])

  return (
    <QueryClientProvider client={queryClient}>
      <Router>
        <div className="min-h-screen bg-black">
          <AnimatedBackground />
          <main className="content-container">
            <Routes>
              <Route
                path="/"
                element={
                  <UnifiedPage
                    onShowPurchaseCredits={() => setShowPurchaseCredits(true)}
                  />
                }
              />
              <Route
                path="/training"
                element={<TrainingPage />}
              />
              <Route
                path="/patching"
                element={
                  <PatchingPlayground
                    onShowPurchaseCredits={() => setShowPurchaseCredits(true)}
                  />
                }
              />
              <Route
                path="/analysis"
                element={
                  <AnalysisPage
                    onShowPurchaseCredits={() => setShowPurchaseCredits(true)}
                  />
                }
              />
              <Route
                path="/sae"
                element={
                  <SAEPage
                    onShowPurchaseCredits={() => setShowPurchaseCredits(true)}
                  />
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
      </Router>
    </QueryClientProvider>
  )
}

export default App
