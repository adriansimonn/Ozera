import { useEffect, useState } from 'react'
import { BrowserRouter as Router, Routes, Route } from 'react-router-dom'
import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import UnifiedPage from './pages/UnifiedPage'
import TrainingPage from './pages/TrainingPage'
import { AnimatedBackground } from './components/common/AnimatedBackground'
import { LoginModal } from './components/auth/LoginModal'
import { SignupModal } from './components/auth/SignupModal'
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
  const [showLogin, setShowLogin] = useState(false)
  const [showSignup, setShowSignup] = useState(false)
  const [showPurchaseCredits, setShowPurchaseCredits] = useState(false)
  const { token, refreshUser } = useAuthStore()

  // Refresh user data on mount if token exists
  useEffect(() => {
    if (token) {
      refreshUser()
    }
  }, [token, refreshUser])

  // Make modals available globally via window
  useEffect(() => {
    ;(window as any).showLoginModal = () => setShowLogin(true)
    ;(window as any).showSignupModal = () => setShowSignup(true)
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
                    onShowLogin={() => setShowLogin(true)}
                    onShowSignup={() => setShowSignup(true)}
                  />
                }
              />
              <Route
                path="/training"
                element={
                  <TrainingPage
                    onShowLogin={() => setShowLogin(true)}
                    onShowSignup={() => setShowSignup(true)}
                  />
                }
              />
            </Routes>
          </main>

          {/* Auth Modals */}
          <LoginModal
            isOpen={showLogin}
            onClose={() => setShowLogin(false)}
            onSwitchToSignup={() => {
              setShowLogin(false)
              setShowSignup(true)
            }}
          />
          <SignupModal
            isOpen={showSignup}
            onClose={() => setShowSignup(false)}
            onSwitchToLogin={() => {
              setShowSignup(false)
              setShowLogin(true)
            }}
          />

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
