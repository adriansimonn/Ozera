/**
 * Text generation page combining model selector and text generator.
 */

import React, { useState } from 'react'
import TextGenerator from '../components/model/TextGenerator'
import ModelSelector from '../components/model/ModelSelector'

export const GeneratePage: React.FC = () => {
  const [selectedModel, setSelectedModel] = useState<'nano' | 'mini'>('nano')

  return (
    <div className="generate-page">
      <div className="page-header">
        <h1>Ozera Text Generation</h1>
        <p className="subtitle">Generate text using Ozera language models</p>
      </div>

      <div className="page-content">
        <div className="sidebar">
          <ModelSelector
            selectedModel={selectedModel}
            onModelSelect={(model) => setSelectedModel(model as 'nano' | 'mini')}
          />
        </div>

        <div className="main-content">
          <TextGenerator
            key={selectedModel}
            defaultModel={selectedModel}
          />
        </div>
      </div>

      <style>{`
        .generate-page {
          min-height: 100vh;
          background: #f5f5f5;
          padding: 2rem;
        }

        .page-header {
          max-width: 1200px;
          margin: 0 auto 2rem;
          text-align: center;
        }

        .page-header h1 {
          margin: 0 0 0.5rem 0;
          font-size: 2.5rem;
          color: #333;
        }

        .subtitle {
          margin: 0;
          color: #666;
          font-size: 1.1rem;
        }

        .page-content {
          max-width: 1200px;
          margin: 0 auto;
          display: grid;
          grid-template-columns: 300px 1fr;
          gap: 2rem;
        }

        @media (max-width: 900px) {
          .page-content {
            grid-template-columns: 1fr;
          }
        }

        .sidebar {
          position: sticky;
          top: 2rem;
          height: fit-content;
        }

        .main-content {
          background: white;
          border-radius: 8px;
          box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
        }
      `}</style>
    </div>
  )
}

export default GeneratePage
