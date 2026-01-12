import { BrowserRouter as Router, Routes, Route } from 'react-router-dom'
import { QueryClient, QueryClientProvider } from '@tanstack/react-query'

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
        <div className="min-h-screen bg-gray-50 dark:bg-gray-900">
          {/* Navigation/Layout will go here */}
          <main>
            <Routes>
              <Route path="/" element={
                <div className="flex items-center justify-center h-screen">
                  <div className="text-center">
                    <h1 className="text-4xl font-bold text-gray-900 dark:text-white mb-4">
                      Ozera
                    </h1>
                    <p className="text-lg text-gray-600 dark:text-gray-300">
                      Mechanistic Interpretability Platform
                    </p>
                    <p className="text-sm text-gray-500 dark:text-gray-400 mt-2">
                      Project structure initialized. Ready for development.
                    </p>
                  </div>
                </div>
              } />
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
