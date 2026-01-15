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
          background: linear-gradient(135deg, #0a0a0a 0%, #1a1a1a 100%);
          padding: 2rem;
        }

        .page-header {
          max-width: 1200px;
          margin: 0 auto 2rem;
          text-align: center;
        }

        .page-header h1 {
          margin: 0 0 0.5rem 0;
          font-size: 3rem;
          font-weight: 700;
          background: linear-gradient(135deg, #00f5ff 0%, #0088ff 100%);
          -webkit-background-clip: text;
          -webkit-text-fill-color: transparent;
          background-clip: text;
          text-shadow: 0 0 40px rgba(0, 245, 255, 0.3);
        }

        .subtitle {
          margin: 0;
          color: #888;
          font-size: 1.1rem;
          font-weight: 300;
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
          background: rgba(20, 20, 20, 0.6);
          backdrop-filter: blur(20px);
          border: 1px solid rgba(255, 255, 255, 0.1);
          border-radius: 16px;
          box-shadow: 0 8px 32px rgba(0, 0, 0, 0.4);
        }
      `}</style>
    </div>
  )
}

export default GeneratePage
