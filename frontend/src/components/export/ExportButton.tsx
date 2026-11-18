/**
 * Reusable export button component for visualizations.
 */

import { useState } from 'react'
import { Download } from 'lucide-react'
import { ExportModal } from './ExportModal'
import type { ExportFormat } from '../../types/export'
import { Spinner } from '../common/Spinner'

export type ExportType = 'attention-heatmap' | 'activation-histogram' | 'multi-head-heatmap'

interface ExportButtonProps {
  exportType: ExportType
  activationId: string
  layer?: number
  head?: number
  tokenLabels?: 'text' | 'number'
  className?: string
  variant?: 'icon' | 'button'
  disabled?: boolean
}

export function ExportButton({
  exportType,
  activationId,
  layer,
  head,
  tokenLabels,
  className = '',
  variant = 'icon',
  disabled = false,
}: ExportButtonProps) {
  const [isModalOpen, setIsModalOpen] = useState(false)

  const handleClick = (e: React.MouseEvent) => {
    e.stopPropagation()
    setIsModalOpen(true)
  }

  if (variant === 'icon') {
    return (
      <>
        <button
          onClick={handleClick}
          disabled={disabled}
          className={`p-1.5 text-fg hover:text-mark hover:bg-fill-strong
 transition-colors disabled:opacity-50 disabled:cursor-not-allowed ${className}`}
          title="Export figure"
        >
          <Download className="w-4 h-4" />
        </button>

        <ExportModal
          isOpen={isModalOpen}
          onClose={() => setIsModalOpen(false)}
          exportType={exportType}
          activationId={activationId}
          layer={layer}
          head={head}
          initialTokenLabels={tokenLabels}
        />
      </>
    )
  }

  return (
    <>
      <button
        onClick={handleClick}
        disabled={disabled}
        className={`oz-btn oz-btn--ghost oz-btn--small ${className}`}
      >
        Export
      </button>

      <ExportModal
        isOpen={isModalOpen}
        onClose={() => setIsModalOpen(false)}
        exportType={exportType}
        activationId={activationId}
        layer={layer}
        head={head}
      />
    </>
  )
}

interface QuickExportButtonProps {
  onClick: () => void
  isExporting: boolean
  format: ExportFormat
  className?: string
}

export function QuickExportButton({
  onClick,
  isExporting,
  format,
  className = '',
}: QuickExportButtonProps) {
  return (
    <button
      onClick={onClick}
      disabled={isExporting}
      className={`oz-btn oz-btn--ghost oz-btn--small ${className}`}
    >
      {isExporting && <Spinner />}
      {format.toUpperCase()}
    </button>
  )
}
