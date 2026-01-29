/**
 * Reusable export button component for visualizations.
 */

import { useState } from 'react'
import { Download, Loader2 } from 'lucide-react'
import { ExportModal } from './ExportModal'
import type { ExportFormat } from '../../types/export'

export type ExportType = 'attention-heatmap' | 'activation-histogram' | 'multi-head-heatmap'

interface ExportButtonProps {
  exportType: ExportType
  activationId: string
  layer?: number
  head?: number
  className?: string
  variant?: 'icon' | 'button'
  disabled?: boolean
}

export function ExportButton({
  exportType,
  activationId,
  layer,
  head,
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
          className={`p-1.5 text-gray-400 hover:text-white hover:bg-white/10
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
        />
      </>
    )
  }

  return (
    <>
      <button
        onClick={handleClick}
        disabled={disabled}
        className={`flex items-center gap-2 px-3 py-1.5 text-sm font-medium
          bg-white/5 text-gray-300 hover:bg-white/10 hover:text-white
          border border-gray-700 hover:border-gray-600 transition-all
          disabled:opacity-50 disabled:cursor-not-allowed ${className}`}
      >
        <Download className="w-4 h-4" />
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
      className={`flex items-center gap-1.5 px-2 py-1 text-xs font-medium
        bg-white/5 text-gray-400 hover:text-white hover:bg-white/10
        border border-gray-700 hover:border-gray-600 transition-all
        disabled:opacity-50 disabled:cursor-not-allowed ${className}`}
    >
      {isExporting ? (
        <Loader2 className="w-3 h-3 animate-spin" />
      ) : (
        <Download className="w-3 h-3" />
      )}
      {format.toUpperCase()}
    </button>
  )
}
