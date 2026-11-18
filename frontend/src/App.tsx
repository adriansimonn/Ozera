import { useEffect, useLayoutEffect, useState, useCallback } from 'react'
import { BrowserRouter as Router, Routes, Route, Navigate, useLocation } from 'react-router-dom'
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
import HomePage from './pages/HomePage'
import { ErrorBoundary } from './components/common/ErrorBoundary'
import { PurchaseCreditsModal } from './components/payments/PurchaseCreditsModal'
import { PurchaseCreditsProvider } from './contexts/PurchaseCreditsContext'
import { AssistantHost } from './components/assistant/AssistantPanel'
import { useAuthStore } from './stores/authStore'
import { useTheme } from './hooks/useTheme'
import { appPalette, paletteVars } from './styles/appTheme'
import { useConfirmationLanding } from './hooks/useConfirmationLanding'

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
  // "/" is the homepage, signed in or not; the app's pages have paths of their own
  const isHomePage = location.pathname === '/'
  const confirming = useConfirmationLanding()

  const handleShowPurchaseCredits = useCallback(() => setShowPurchaseCredits(true), [])

  // The app's pages take their colors from custom properties on <html> (styles/app.css); the
  // homepage and the sign-in page keep their own
  const isAppPage = !isAuthPage && !isHomePage
  useLayoutEffect(() => {
    if (!isAppPage) return
    const root = document.documentElement
    const vars = paletteVars(appPalette(background === 'light'))
    for (const [name, value] of Object.entries(vars)) root.style.setProperty(name, value)
    root.classList.add('oz-app-active')
    root.dataset.ozBg = background
    return () => {
      for (const name of Object.keys(vars)) root.style.removeProperty(name)
      root.classList.remove('oz-app-active')
      delete root.dataset.ozBg
    }
  }, [isAppPage, background])

  // Hydrate auth state from Supabase session on mount
  useEffect(() => {
    initialize()
  }, [initialize])

  return (
    <PurchaseCreditsProvider onShow={handleShowPurchaseCredits}>
    <div className={isAppPage ? 'oz-app' : 'min-h-screen bg-black'} data-bg={isAppPage ? background : undefined}>
      <main className="content-container">
            <Routes>
              <Route path="/" element={confirming ? null : <HomePage />} />
              <Route path="/generate" element={<UnifiedPage onShowPurchaseCredits={() => setShowPurchaseCredits(true)} />} />
              <Route path="/training" element={<TrainingPage />} />
              <Route path="/patching" element={<PatchingPlayground onShowPurchaseCredits={() => setShowPurchaseCredits(true)} />} />
              <Route path="/analysis" element={<AnalysisPage onShowPurchaseCredits={() => setShowPurchaseCredits(true)} />} />
              <Route path="/sae" element={<SAEPage onShowPurchaseCredits={() => setShowPurchaseCredits(true)} />} />
              <Route path="/probes" element={<ProbesPage onShowPurchaseCredits={() => setShowPurchaseCredits(true)} />} />
              <Route path="/geometry" element={<GeometryPage onShowPurchaseCredits={() => setShowPurchaseCredits(true)} />} />
              <Route path="/settings" element={<SettingsPage />} />
              <Route path="/auth" element={<AuthPage />} />
              <Route path="*" element={<Navigate to="/" replace />} />
            </Routes>
          </main>

          {/* The assistant: outside the routes, so it stays open across pages */}
          {isAppPage && <AssistantHost />}

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
