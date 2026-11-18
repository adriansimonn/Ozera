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
    <div className="oz-overlay" onClick={onCancel}>
      <div className="oz-modal" role="dialog" aria-modal="true" onClick={(e) => e.stopPropagation()}>
        <button onClick={onCancel} className="oz-icon-btn oz-modal__close" aria-label="Close">
          <X />
        </button>

        <h2 className="oz-modal__title">{title}</h2>
        <p className="oz-modal__text">{message}</p>

        <div className="oz-modal__actions">
          <button onClick={onCancel} disabled={isLoading} className="oz-btn oz-btn--ghost">
            {cancelLabel}
          </button>
          <button onClick={onConfirm} disabled={isLoading} className="oz-btn oz-btn--danger">
            {isLoading ? 'Deleting...' : confirmLabel}
          </button>
        </div>
      </div>
    </div>
  )
}

export default ConfirmModal
