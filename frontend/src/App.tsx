import { BrowserRouter as Router, Routes, Route } from 'react-router-dom'
import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import GeneratePage from './pages/GeneratePage'
import { VisualizationPage } from './pages/VisualizationPage'

// Pages (to be created)
// import Dashboard from '@pages/dashboard/Dashboard'
// import Playground from '@pages/playground/Playground'
// import Training from '@pages/training/Training'
// import Analysis from '@pages/analysis/Analysis'

const queryClient = new QueryClient({
  defaultOptions: {
    queries: {
      refetchOnWindowFocus: false,
      retry: 1,
    },
  },
})

function App() {
  return (
    <QueryClientProvider client={queryClient}>
      <Router>
        <div className="min-h-screen bg-black">
          {/* Navigation/Layout will go here */}
          <main>
            <Routes>
              <Route path="/" element={<GeneratePage />} />
              <Route path="/visualize" element={<VisualizationPage />} />
              {/* <Route path="/playground" element={<Playground />} /> */}
              {/* <Route path="/training" element={<Training />} /> */}
              {/* <Route path="/analysis" element={<Analysis />} /> */}
            </Routes>
          </main>
        </div>
      </Router>
    </QueryClientProvider>
  )
}

export default App
