import React from 'react'
import { X } from 'lucide-react'

interface ConfirmModalProps {
  isOpen: boolean
  title: string
  message: string
  confirmLabel?: string
  cancelLabel?: string
  onConfirm: () => void
  onCancel: () => void
  isLoading?: boolean
}

export const ConfirmModal: React.FC<ConfirmModalProps> = ({
  isOpen,
  title,
  message,
  confirmLabel = 'Delete',
  cancelLabel = 'Cancel',
  onConfirm,
  onCancel,
  isLoading = false,
}) => {
  if (!isOpen) return null

  return (
    <div
      className="fixed inset-0 bg-black/70 backdrop-blur-sm flex items-center justify-center z-50 p-4"
      onClick={onCancel}
    >
      <div
        className="bg-black/60 backdrop-filter backdrop-blur-xl border border-gray-800 max-w-sm w-full p-6 relative"
        onClick={(e) => e.stopPropagation()}
      >
        <button
          onClick={onCancel}
          className="absolute top-4 right-4 text-gray-500 hover:text-white transition-colors"
          aria-label="Close"
        >
          <X size={18} />
        </button>

        <h2 className="text-lg font-semibold text-white mb-3 tracking-tight">
          {title}
        </h2>

        <p className="text-sm text-gray-400 mb-6 leading-relaxed">
          {message}
        </p>

        <div className="flex gap-3 justify-end">
          <button
            onClick={onCancel}
            disabled={isLoading}
            className="px-4 py-2 bg-white/5 hover:bg-white/10 disabled:bg-gray-800 disabled:cursor-not-allowed text-white text-sm font-medium border border-gray-700 hover:border-gray-600 transition-all"
          >
            {cancelLabel}
          </button>
          <button
            onClick={onConfirm}
            disabled={isLoading}
            className="px-4 py-2 bg-red-500/20 hover:bg-red-500/30 disabled:bg-red-500/10 disabled:cursor-not-allowed text-red-400 text-sm font-medium border border-red-500/30 hover:border-red-500/50 transition-all"
          >
            {isLoading ? 'Deleting...' : confirmLabel}
          </button>
        </div>
      </div>
    </div>
  )
}

export default ConfirmModal
