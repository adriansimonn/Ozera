/**
 * SAE (Sparse Autoencoder) Page - Main hub for SAE feature exploration and analysis.
 * Provides tabs for browsing features, analyzing activations, and viewing quality metrics.
 */

import { useState, useCallback } from 'react'
import { NavBar } from '../components/common/NavBar'
import {
  FeatureActivationDisplay,
  FeatureBrowser,
  FeatureTopTokens,
  SparsityDashboard,
  FeatureComparison,
} from '../components/sae'
import {
  Layers,
  Search,
  BarChart3,
  GitCompare,
  Info,
} from 'lucide-react'
import type {
  SequenceFeatureActivations,
  FeatureStats,
  FeatureInterpretation,
  FeatureCatalog,
  SAEQualityMetrics,
  SAETrainingProgress,
  SAEComparisonMetrics,
} from '../types/model'

interface SAEPageProps {
  onShowLogin: () => void
  onShowSignup: () => void
  onShowPurchaseCredits?: () => void
}

type SAEMode = 'browse' | 'analyze' | 'dashboard' | 'compare'

// Mock data for demonstration - replace with API calls when backend is ready
const mockFeatures: FeatureStats[] = Array.from({ length: 100 }, (_, i) => ({
  feature_idx: i,
  activation_frequency: Math.random() * 0.5,
  mean_activation: Math.random() * 2,
  max_activation: Math.random() * 5 + 2,
  polysemanticity: Math.random(),
  suggested_label: [
    'punctuation',
    'named entities',
    'numbers',
    'prepositions',
    'adjectives',
    'verbs',
    'technical terms',
    'common words',
  ][Math.floor(Math.random() * 8)],
  confidence: Math.random() * 0.5 + 0.5,
  unique_tokens: Math.floor(Math.random() * 50) + 5,
}))

const mockCatalog: FeatureCatalog = {
  sae_id: 'mock-sae-1',
  num_features: 100,
  features: mockFeatures,
  dead_features: [3, 17, 42, 67, 89],
  polysemantic_features: [5, 12, 28, 45, 78],
  monosemantic_features: [1, 8, 15, 33, 56, 71, 92],
}

const mockActivations: SequenceFeatureActivations = {
  tokens: ['The', ' quick', ' brown', ' fox', ' jumps', ' over', ' the', ' lazy', ' dog'],
  token_ids: [464, 2159, 6282, 6524, 13407, 731, 278, 17366, 3748],
  per_token_activations: Array.from({ length: 9 }, (_, pos) => ({
    position: pos,
    token: ['The', ' quick', ' brown', ' fox', ' jumps', ' over', ' the', ' lazy', ' dog'][pos],
    token_id: [464, 2159, 6282, 6524, 13407, 731, 278, 17366, 3748][pos],
    top_features: Array.from({ length: 10 }, (_, i) => ({
      feature_idx: Math.floor(Math.random() * 100),
      activation_value: Math.random() * 5,
      rank: i + 1,
      percentile: 1 - (i + 1) / 100,
    })),
    total_active_features: Math.floor(Math.random() * 30) + 10,
    l0_sparsity: Math.floor(Math.random() * 30) + 10,
    l1_norm: Math.random() * 20,
  })),
  feature_activation_matrix: Array.from({ length: 9 }, () =>
    Array.from({ length: 50 }, () => Math.random() * (Math.random() > 0.7 ? 5 : 0))
  ),
  active_features_per_position: Array.from({ length: 9 }, () => Math.floor(Math.random() * 30) + 10),
  most_active_features: [12, 45, 78, 23, 56, 89, 34, 67, 90, 11],
}

const mockInterpretation: FeatureInterpretation = {
  feature_idx: 12,
  top_activating_tokens: [
    { token: 'quick', token_id: 2159, activation_value: 4.5, position: 1, context: ['The', 'quick', 'brown'], prompt: 'The quick brown fox' },
    { token: 'fast', token_id: 3178, activation_value: 4.2, position: 2, context: ['very', 'fast', 'car'], prompt: 'A very fast car' },
    { token: 'rapid', token_id: 5432, activation_value: 3.9, position: 0, context: ['rapid', 'growth', 'in'], prompt: 'Rapid growth in sales' },
  ],
  token_frequency_distribution: {
    'quick': 0.25,
    'fast': 0.20,
    'rapid': 0.15,
    'swift': 0.12,
    'speedy': 0.08,
    'hasty': 0.05,
    'prompt': 0.05,
    'brisk': 0.04,
    'agile': 0.03,
    'nimble': 0.03,
  },
  suggested_label: 'speed/quickness adjectives',
  confidence: 0.87,
  activation_statistics: {
    total_activations: 1523,
    mean_activation: 2.34,
    max_activation: 4.89,
    unique_tokens: 42,
    polysemanticity: 0.23,
  },
}

const mockQualityMetrics: SAEQualityMetrics = {
  sparsity: {
    avg_l0: 23.4,
    l0_std: 8.2,
    sparsity_fraction: 0.023,
    avg_l1: 15.67,
    max_activation: 12.34,
  },
  feature_health: {
    num_features: 1024,
    dead_features: 42,
    dead_feature_fraction: 0.041,
    low_frequency_features: 156,
    high_frequency_features: 3,
    feature_frequency_distribution: Array.from({ length: 1024 }, () => Math.random() * 0.5),
    feature_magnitude_distribution: Array.from({ length: 1024 }, () => Math.random() * 5),
  },
  reconstruction: {
    mse: 0.0023,
    rmse: 0.048,
    normalized_mse: 0.015,
    explained_variance: 0.94,
    cosine_similarity: 0.987,
    relative_reconstruction_error: 0.032,
  },
}

const mockTrainingProgress: SAETrainingProgress[] = Array.from({ length: 50 }, (_, i) => ({
  step: i * 100,
  total_steps: 5000,
  loss: 0.5 * Math.exp(-i / 20) + 0.05,
  reconstruction_loss: 0.4 * Math.exp(-i / 20) + 0.03,
  sparsity_loss: 0.1 * Math.exp(-i / 25) + 0.02,
  avg_l0: 50 - i * 0.5,
  dead_features: Math.floor(30 + i * 0.3),
}))

const mockSaeA = {
  id: 'sae-1',
  name: 'Ozera-Mini L4',
  num_features: 1024,
  layer: 4,
  model: 'Ozera-Mini',
}

const mockSaeB = {
  id: 'sae-2',
  name: 'Ozera-Mini L6',
  num_features: 1024,
  layer: 6,
  model: 'Ozera-Mini',
}

const mockComparisonMetrics: SAEComparisonMetrics = {
  overall_similarity: 0.67,
  matched_features: 456,
  unmatched_a: 284,
  unmatched_b: 284,
  top_matches: Array.from({ length: 20 }, (_, i) => ({
    feature_a: i * 5,
    feature_b: i * 5 + Math.floor(Math.random() * 10),
    similarity: 0.95 - i * 0.02,
    shared_tokens: ['the', 'a', 'an', 'of', 'in'].slice(0, Math.floor(Math.random() * 5) + 1),
    label_a: `Feature ${i * 5} pattern`,
    label_b: `Feature ${i * 5 + Math.floor(Math.random() * 10)} pattern`,
  })),
  divergent_features_a: [100, 200, 300, 400, 500],
  divergent_features_b: [150, 250, 350, 450, 550],
}

export default function SAEPage({
  onShowLogin,
  onShowSignup,
}: SAEPageProps) {
  const [mode, setMode] = useState<SAEMode>('browse')
  const [selectedFeature, setSelectedFeature] = useState<number | null>(null)
  const [selectedFeatures, setSelectedFeatures] = useState<number[]>([])

  const handleFeatureSelect = useCallback((featureIdx: number) => {
    setSelectedFeature(featureIdx)
    setSelectedFeatures(prev =>
      prev.includes(featureIdx)
        ? prev.filter(f => f !== featureIdx)
        : [...prev, featureIdx].slice(-5) // Keep max 5 selected
    )
  }, [])

  const handleComparisonFeatureSelect = useCallback((_saeId: string, featureIdx: number) => {
    setSelectedFeature(featureIdx)
  }, [])

  return (
    <div className="sae-page">
      <NavBar onShowLogin={onShowLogin} onShowSignup={onShowSignup} />

      <div className="sae-content">
        {/* Header */}
        <div className="sae-header">
          <h1>Sparse Autoencoder Analysis</h1>
          <p className="header-description">
            Explore learned features, analyze activations, and evaluate SAE quality
          </p>
        </div>

        {/* Mode Selection */}
        <div className="mode-tabs">
          {[
            { id: 'browse', label: 'Browse Features', icon: Layers },
            { id: 'analyze', label: 'Analyze Activations', icon: Search },
            { id: 'dashboard', label: 'Quality Dashboard', icon: BarChart3 },
            { id: 'compare', label: 'Compare SAEs', icon: GitCompare },
          ].map(({ id, label, icon: Icon }) => (
            <button
              key={id}
              onClick={() => setMode(id as SAEMode)}
              className={`mode-tab ${mode === id ? 'active' : ''}`}
            >
              <Icon size={16} />
              {label}
            </button>
          ))}
        </div>

        {/* Info Banner */}
        <div className="info-banner">
          <Info className="info-icon" />
          <span>
            This page shows mock data for demonstration. Connect to the SAE API endpoints to load real SAE data.
          </span>
        </div>

        {/* Content */}
        <div className="sae-main">
          {/* Browse Features Mode */}
          {mode === 'browse' && (
            <div className="browse-layout">
              <div className="browser-panel">
                <FeatureBrowser
                  catalog={mockCatalog}
                  onFeatureSelect={handleFeatureSelect}
                  selectedFeature={selectedFeature ?? undefined}
                />
              </div>
              <div className="detail-panel">
                {selectedFeature !== null ? (
                  <FeatureTopTokens
                    interpretation={{
                      ...mockInterpretation,
                      feature_idx: selectedFeature,
                    }}
                    maxExamples={15}
                  />
                ) : (
                  <div className="empty-detail">
                    <Layers className="empty-icon" />
                    <h3>Select a Feature</h3>
                    <p>Click on a feature in the browser to see its details</p>
                  </div>
                )}
              </div>
            </div>
          )}

          {/* Analyze Activations Mode */}
          {mode === 'analyze' && (
            <div className="analyze-layout">
              <div className="activation-display-panel">
                <FeatureActivationDisplay
                  activations={mockActivations}
                  selectedFeatures={selectedFeatures}
                  onFeatureSelect={handleFeatureSelect}
                  showTopK={10}
                />
              </div>
              <div className="feature-detail-panel">
                {selectedFeature !== null ? (
                  <FeatureTopTokens
                    interpretation={{
                      ...mockInterpretation,
                      feature_idx: selectedFeature,
                    }}
                    maxExamples={10}
                  />
                ) : (
                  <div className="empty-detail">
                    <Search className="empty-icon" />
                    <h3>Select a Feature</h3>
                    <p>Click on a cell in the activation heatmap to see feature details</p>
                  </div>
                )}
              </div>
            </div>
          )}

          {/* Quality Dashboard Mode */}
          {mode === 'dashboard' && (
            <div className="dashboard-layout">
              <SparsityDashboard
                metrics={mockQualityMetrics}
                trainingProgress={mockTrainingProgress}
              />
            </div>
          )}

          {/* Compare SAEs Mode */}
          {mode === 'compare' && (
            <div className="compare-layout">
              <FeatureComparison
                saeA={mockSaeA}
                saeB={mockSaeB}
                comparison={mockComparisonMetrics}
                onFeatureSelect={handleComparisonFeatureSelect}
              />
              {selectedFeature !== null && (
                <div className="comparison-detail">
                  <FeatureTopTokens
                    interpretation={{
                      ...mockInterpretation,
                      feature_idx: selectedFeature,
                    }}
                    maxExamples={8}
                  />
                </div>
              )}
            </div>
          )}
        </div>
      </div>

      <style>{`
        .sae-page {
          min-height: 100vh;
          padding-top: 70px;
        }

        .sae-content {
          max-width: 1600px;
          margin: 0 auto;
          padding: 2rem;
        }

        .sae-header {
          margin-bottom: 1.5rem;
        }

        .sae-header h1 {
          margin: 0;
          font-size: 1.75rem;
          font-weight: 600;
          color: rgba(255, 255, 255, 0.95);
          letter-spacing: -0.02em;
        }

        .header-description {
          margin: 0.5rem 0 0 0;
          font-size: 0.9rem;
          color: rgba(255, 255, 255, 0.5);
        }

        .mode-tabs {
          display: flex;
          gap: 0.5rem;
          margin-bottom: 1rem;
        }

        .mode-tab {
          display: flex;
          align-items: center;
          gap: 0.5rem;
          padding: 0.625rem 1rem;
          background: rgba(0, 0, 0, 0.3);
          border: 1px solid rgba(255, 255, 255, 0.1);
          color: rgba(255, 255, 255, 0.5);
          font-size: 0.875rem;
          cursor: pointer;
          transition: all 0.2s;
        }

        .mode-tab:hover {
          background: rgba(255, 255, 255, 0.05);
          border-color: rgba(255, 255, 255, 0.2);
          color: rgba(255, 255, 255, 0.7);
        }

        .mode-tab.active {
          background: rgba(168, 85, 247, 0.15);
          border-color: rgba(168, 85, 247, 0.5);
          color: rgba(255, 255, 255, 0.95);
        }

        .info-banner {
          display: flex;
          align-items: center;
          gap: 0.75rem;
          padding: 0.875rem 1rem;
          background: rgba(59, 130, 246, 0.1);
          border: 1px solid rgba(59, 130, 246, 0.25);
          color: rgba(59, 130, 246, 0.9);
          margin-bottom: 1.5rem;
          font-size: 0.85rem;
        }

        .info-icon {
          width: 18px;
          height: 18px;
          flex-shrink: 0;
        }

        .sae-main {
          min-height: 500px;
        }

        /* Browse Layout */
        .browse-layout {
          display: grid;
          grid-template-columns: 1fr 400px;
          gap: 1.5rem;
          align-items: start;
        }

        .browser-panel {
          min-height: 600px;
        }

        .detail-panel {
          position: sticky;
          top: 90px;
        }

        /* Analyze Layout */
        .analyze-layout {
          display: grid;
          grid-template-columns: 1fr 400px;
          gap: 1.5rem;
          align-items: start;
        }

        .activation-display-panel {
          min-height: 500px;
        }

        .feature-detail-panel {
          position: sticky;
          top: 90px;
        }

        /* Dashboard Layout */
        .dashboard-layout {
          display: flex;
          flex-direction: column;
          gap: 1.5rem;
        }

        /* Compare Layout */
        .compare-layout {
          display: flex;
          flex-direction: column;
          gap: 1.5rem;
        }

        .comparison-detail {
          max-width: 600px;
        }

        /* Empty State */
        .empty-detail {
          display: flex;
          flex-direction: column;
          align-items: center;
          justify-content: center;
          padding: 3rem 2rem;
          background: rgba(0, 0, 0, 0.2);
          border: 1px dashed rgba(255, 255, 255, 0.1);
          text-align: center;
          min-height: 300px;
        }

        .empty-icon {
          width: 48px;
          height: 48px;
          color: rgba(255, 255, 255, 0.2);
          margin-bottom: 1rem;
        }

        .empty-detail h3 {
          margin: 0 0 0.5rem 0;
          font-size: 1rem;
          font-weight: 500;
          color: rgba(255, 255, 255, 0.5);
        }

        .empty-detail p {
          margin: 0;
          font-size: 0.85rem;
          color: rgba(255, 255, 255, 0.35);
        }

        @media (max-width: 1200px) {
          .browse-layout,
          .analyze-layout {
            grid-template-columns: 1fr;
          }

          .detail-panel,
          .feature-detail-panel {
            position: static;
          }
        }

        @media (max-width: 768px) {
          .sae-content {
            padding: 1rem;
          }

          .mode-tabs {
            flex-wrap: wrap;
          }

          .mode-tab {
            flex: 1;
            min-width: 140px;
            justify-content: center;
          }
        }
      `}</style>
    </div>
  )
}
