/**
 * Sparsity dashboard for SAE training diagnostics.
 * Shows reconstruction quality, sparsity metrics, and feature health.
 */

import { useEffect, useRef } from 'react'
import * as d3 from 'd3'
import { Activity, TrendingDown, AlertTriangle, CheckCircle, BarChart3 } from 'lucide-react'

interface SparsityMetrics {
  avg_l0: number
  l0_std: number
  sparsity_fraction: number
  avg_l1: number
  max_activation: number
}

interface FeatureHealthMetrics {
  num_features: number
  dead_features: number
  dead_feature_fraction: number
  low_frequency_features: number
  high_frequency_features: number
  feature_frequency_distribution: number[]
  feature_magnitude_distribution: number[]
}

interface ReconstructionMetrics {
  mse: number
  rmse: number
  normalized_mse: number
  explained_variance: number
  cosine_similarity: number
  relative_reconstruction_error: number
}

interface SAEQualityMetrics {
  sparsity: SparsityMetrics
  feature_health: FeatureHealthMetrics
  reconstruction: ReconstructionMetrics
}

interface TrainingProgress {
  step: number
  total_steps: number
  loss: number
  reconstruction_loss: number
  sparsity_loss: number
  avg_l0: number
  dead_features: number
}

interface SparsityDashboardProps {
  metrics: SAEQualityMetrics
  trainingProgress?: TrainingProgress[]
  className?: string
}

export function SparsityDashboard({
  metrics,
  trainingProgress = [],
  className = '',
}: SparsityDashboardProps) {
  const frequencyChartRef = useRef<SVGSVGElement>(null)
  const lossChartRef = useRef<SVGSVGElement>(null)

  // Render feature frequency distribution
  useEffect(() => {
    if (!frequencyChartRef.current) return

    d3.select(frequencyChartRef.current).selectAll('*').remove()

    const svg = d3.select(frequencyChartRef.current)
    const width = 400
    const height = 200
    const margin = { top: 30, right: 20, bottom: 40, left: 50 }
    const innerWidth = width - margin.left - margin.right
    const innerHeight = height - margin.top - margin.bottom

    const frequencies = metrics.feature_health.feature_frequency_distribution
    const numBins = 50
    const binSize = 1 / numBins

    // Create histogram
    const bins = Array(numBins).fill(0)
    frequencies.forEach(f => {
      const binIdx = Math.min(Math.floor(f / binSize), numBins - 1)
      bins[binIdx]++
    })

    const xScale = d3.scaleLinear()
      .domain([0, 1])
      .range([0, innerWidth])

    const yScale = d3.scaleLinear()
      .domain([0, d3.max(bins) || 1])
      .range([innerHeight, 0])

    const g = svg.append('g')
      .attr('transform', `translate(${margin.left},${margin.top})`)

    // Title
    svg.append('text')
      .attr('x', width / 2)
      .attr('y', 15)
      .attr('text-anchor', 'middle')
      .attr('fill', '#e5e7eb')
      .attr('font-size', '12px')
      .text('Feature Activation Frequency Distribution')

    // Bars
    g.selectAll('rect')
      .data(bins)
      .enter()
      .append('rect')
      .attr('x', (_, i) => xScale(i * binSize))
      .attr('y', d => yScale(d))
      .attr('width', Math.max(1, innerWidth / numBins - 1))
      .attr('height', d => innerHeight - yScale(d))
      .attr('fill', '#a855f7')
      .attr('opacity', 0.7)

    // Axes
    g.append('g')
      .attr('transform', `translate(0,${innerHeight})`)
      .call(d3.axisBottom(xScale).ticks(5).tickFormat(d => `${(d as number * 100).toFixed(0)}%`))
      .selectAll('text')
      .attr('fill', '#9ca3af')
      .attr('font-size', '10px')

    g.append('g')
      .call(d3.axisLeft(yScale).ticks(5))
      .selectAll('text')
      .attr('fill', '#9ca3af')
      .attr('font-size', '10px')

    // Axis labels
    svg.append('text')
      .attr('x', width / 2)
      .attr('y', height - 5)
      .attr('text-anchor', 'middle')
      .attr('fill', '#9ca3af')
      .attr('font-size', '10px')
      .text('Activation Frequency')

    svg.append('text')
      .attr('transform', 'rotate(-90)')
      .attr('x', -height / 2)
      .attr('y', 15)
      .attr('text-anchor', 'middle')
      .attr('fill', '#9ca3af')
      .attr('font-size', '10px')
      .text('Feature Count')

  }, [metrics.feature_health.feature_frequency_distribution])

  // Render training loss chart
  useEffect(() => {
    if (!lossChartRef.current || trainingProgress.length === 0) return

    d3.select(lossChartRef.current).selectAll('*').remove()

    const svg = d3.select(lossChartRef.current)
    const width = 400
    const height = 200
    const margin = { top: 30, right: 20, bottom: 40, left: 50 }
    const innerWidth = width - margin.left - margin.right
    const innerHeight = height - margin.top - margin.bottom

    const xScale = d3.scaleLinear()
      .domain([0, d3.max(trainingProgress, d => d.step) || 1])
      .range([0, innerWidth])

    const yScale = d3.scaleLinear()
      .domain([0, d3.max(trainingProgress, d => d.loss) || 1])
      .range([innerHeight, 0])

    const g = svg.append('g')
      .attr('transform', `translate(${margin.left},${margin.top})`)

    // Title
    svg.append('text')
      .attr('x', width / 2)
      .attr('y', 15)
      .attr('text-anchor', 'middle')
      .attr('fill', '#e5e7eb')
      .attr('font-size', '12px')
      .text('Training Loss Over Time')

    // Total loss line
    const lossLine = d3.line<TrainingProgress>()
      .x(d => xScale(d.step))
      .y(d => yScale(d.loss))

    g.append('path')
      .datum(trainingProgress)
      .attr('fill', 'none')
      .attr('stroke', '#a855f7')
      .attr('stroke-width', 2)
      .attr('d', lossLine)

    // Reconstruction loss line
    const reconLine = d3.line<TrainingProgress>()
      .x(d => xScale(d.step))
      .y(d => yScale(d.reconstruction_loss))

    g.append('path')
      .datum(trainingProgress)
      .attr('fill', 'none')
      .attr('stroke', '#22d3ee')
      .attr('stroke-width', 1.5)
      .attr('stroke-dasharray', '4,4')
      .attr('d', reconLine)

    // Axes
    g.append('g')
      .attr('transform', `translate(0,${innerHeight})`)
      .call(d3.axisBottom(xScale).ticks(5))
      .selectAll('text')
      .attr('fill', '#9ca3af')
      .attr('font-size', '10px')

    g.append('g')
      .call(d3.axisLeft(yScale).ticks(5))
      .selectAll('text')
      .attr('fill', '#9ca3af')
      .attr('font-size', '10px')

    // Legend
    const legend = svg.append('g')
      .attr('transform', `translate(${width - 120}, 25)`)

    legend.append('line')
      .attr('x1', 0).attr('x2', 20)
      .attr('y1', 0).attr('y2', 0)
      .attr('stroke', '#a855f7').attr('stroke-width', 2)

    legend.append('text')
      .attr('x', 25).attr('y', 4)
      .attr('fill', '#9ca3af').attr('font-size', '10px')
      .text('Total Loss')

    legend.append('line')
      .attr('x1', 0).attr('x2', 20)
      .attr('y1', 15).attr('y2', 15)
      .attr('stroke', '#22d3ee').attr('stroke-width', 1.5)
      .attr('stroke-dasharray', '4,4')

    legend.append('text')
      .attr('x', 25).attr('y', 19)
      .attr('fill', '#9ca3af').attr('font-size', '10px')
      .text('Recon Loss')

  }, [trainingProgress])

  const { sparsity, feature_health, reconstruction } = metrics

  // Quality indicators
  const getQualityStatus = () => {
    const issues = []
    if (reconstruction.explained_variance < 0.9) issues.push('Low reconstruction quality')
    if (feature_health.dead_feature_fraction > 0.1) issues.push('Many dead features')
    if (sparsity.avg_l0 > 100) issues.push('Low sparsity')
    if (sparsity.avg_l0 < 5) issues.push('Excessive sparsity')
    return issues
  }

  const qualityIssues = getQualityStatus()
  const isHealthy = qualityIssues.length === 0

  return (
    <div className={`bg-black/40 border border-gray-800 ${className}`}>
      {/* Header */}
      <div className="p-4 border-b border-gray-800">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-2">
            <BarChart3 className="w-5 h-5 text-purple-400" />
            <h3 className="text-lg font-semibold text-gray-200 tracking-tight">
              SAE Quality Metrics
            </h3>
          </div>
          {isHealthy ? (
            <div className="flex items-center gap-1 text-green-400 text-sm">
              <CheckCircle className="w-4 h-4" />
              Healthy
            </div>
          ) : (
            <div className="flex items-center gap-1 text-yellow-400 text-sm">
              <AlertTriangle className="w-4 h-4" />
              {qualityIssues.length} issue{qualityIssues.length > 1 ? 's' : ''}
            </div>
          )}
        </div>
      </div>

      {/* Quality issues */}
      {qualityIssues.length > 0 && (
        <div className="px-4 py-3 bg-yellow-500/10 border-b border-yellow-500/20">
          <div className="text-xs text-yellow-400 uppercase tracking-wide mb-2">Potential Issues</div>
          <ul className="text-sm text-yellow-300 space-y-1">
            {qualityIssues.map((issue, idx) => (
              <li key={idx} className="flex items-center gap-2">
                <AlertTriangle className="w-3 h-3" />
                {issue}
              </li>
            ))}
          </ul>
        </div>
      )}

      {/* Main metrics */}
      <div className="grid grid-cols-3 gap-px bg-gray-800">
        {/* Sparsity section */}
        <div className="bg-black/40 p-4">
          <div className="flex items-center gap-2 mb-3">
            <Activity className="w-4 h-4 text-purple-400" />
            <span className="text-xs text-gray-500 uppercase tracking-wide">Sparsity</span>
          </div>
          <div className="space-y-3">
            <MetricRow label="Avg L0" value={sparsity.avg_l0.toFixed(1)} sublabel="features/input" />
            <MetricRow label="L0 Std" value={sparsity.l0_std.toFixed(1)} />
            <MetricRow label="Sparsity" value={`${(sparsity.sparsity_fraction * 100).toFixed(2)}%`} />
            <MetricRow label="Avg L1" value={sparsity.avg_l1.toFixed(2)} />
            <MetricRow label="Max Act" value={sparsity.max_activation.toFixed(3)} />
          </div>
        </div>

        {/* Reconstruction section */}
        <div className="bg-black/40 p-4">
          <div className="flex items-center gap-2 mb-3">
            <TrendingDown className="w-4 h-4 text-cyan-400" />
            <span className="text-xs text-gray-500 uppercase tracking-wide">Reconstruction</span>
          </div>
          <div className="space-y-3">
            <MetricRow label="MSE" value={reconstruction.mse.toExponential(2)} />
            <MetricRow label="RMSE" value={reconstruction.rmse.toFixed(4)} />
            <MetricRow
              label="Explained Var"
              value={`${(reconstruction.explained_variance * 100).toFixed(1)}%`}
              highlight={reconstruction.explained_variance > 0.9}
            />
            <MetricRow label="Cosine Sim" value={reconstruction.cosine_similarity.toFixed(4)} />
            <MetricRow label="Rel Error" value={reconstruction.relative_reconstruction_error.toFixed(4)} />
          </div>
        </div>

        {/* Feature health section */}
        <div className="bg-black/40 p-4">
          <div className="flex items-center gap-2 mb-3">
            <AlertTriangle className="w-4 h-4 text-yellow-400" />
            <span className="text-xs text-gray-500 uppercase tracking-wide">Feature Health</span>
          </div>
          <div className="space-y-3">
            <MetricRow label="Total" value={feature_health.num_features.toLocaleString()} />
            <MetricRow
              label="Dead"
              value={feature_health.dead_features.toLocaleString()}
              sublabel={`${(feature_health.dead_feature_fraction * 100).toFixed(1)}%`}
              warning={feature_health.dead_feature_fraction > 0.1}
            />
            <MetricRow label="Low Freq" value={feature_health.low_frequency_features.toLocaleString()} />
            <MetricRow
              label="High Freq"
              value={feature_health.high_frequency_features.toLocaleString()}
              warning={feature_health.high_frequency_features > 10}
            />
          </div>
        </div>
      </div>

      {/* Charts */}
      <div className="grid grid-cols-2 gap-px bg-gray-800 border-t border-gray-800">
        <div className="bg-black/40 p-4">
          <svg ref={frequencyChartRef} width="100%" height={200} />
        </div>
        <div className="bg-black/40 p-4">
          {trainingProgress.length > 0 ? (
            <svg ref={lossChartRef} width="100%" height={200} />
          ) : (
            <div className="h-[200px] flex items-center justify-center text-gray-600 text-sm">
              No training history available
            </div>
          )}
        </div>
      </div>
    </div>
  )
}

interface MetricRowProps {
  label: string
  value: string
  sublabel?: string
  highlight?: boolean
  warning?: boolean
}

function MetricRow({ label, value, sublabel, highlight, warning }: MetricRowProps) {
  return (
    <div className="flex items-center justify-between">
      <span className="text-xs text-gray-500">{label}</span>
      <div className="text-right">
        <span className={`font-mono text-sm ${
          warning ? 'text-yellow-400' :
          highlight ? 'text-green-400' :
          'text-white'
        }`}>
          {value}
        </span>
        {sublabel && (
          <span className="text-xs text-gray-600 ml-1">{sublabel}</span>
        )}
      </div>
    </div>
  )
}
