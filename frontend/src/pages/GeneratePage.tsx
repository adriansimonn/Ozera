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
        <h1>Ozera</h1>
        <p className="subtitle">Text Generation</p>
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
          padding: 3rem 2rem;
        }

        .page-header {
          max-width: 1400px;
          margin: 0 auto 3rem;
          text-align: center;
        }

        .page-header h1 {
          margin: 0 0 0.5rem 0;
          font-size: 3.5rem;
          font-weight: 700;
          color: #ffffff;
          letter-spacing: -0.03em;
        }

        .subtitle {
          margin: 0;
          color: rgba(255, 255, 255, 0.5);
          font-size: 1rem;
          font-weight: 400;
          letter-spacing: 0.1em;
          text-transform: uppercase;
        }

        .page-content {
          max-width: 1400px;
          margin: 0 auto;
          display: grid;
          grid-template-columns: 320px 1fr;
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
          background: rgba(0, 0, 0, 0.5);
          backdrop-filter: blur(20px) saturate(180%);
          -webkit-backdrop-filter: blur(20px) saturate(180%);
          border: 1px solid rgba(255, 255, 255, 0.15);
          box-shadow:
            0 8px 32px rgba(0, 0, 0, 0.3),
            inset 0 1px 0 rgba(255, 255, 255, 0.1),
            0 0 0 1px rgba(255, 255, 255, 0.05);
        }
      `}</style>
    </div>
  )
}

export default GeneratePage
