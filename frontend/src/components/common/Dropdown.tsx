import { useState, useRef, useEffect, useCallback } from 'react'
import { ChevronDown } from 'lucide-react'
import { useTheme } from '../../hooks/useTheme'

export interface DropdownOption {
  value: string
  label: string
  disabled?: boolean
}

export interface DropdownGroup {
  label: string
  options: DropdownOption[]
}

interface DropdownProps {
  value: string
  onChange: (value: string) => void
  options?: DropdownOption[]
  groups?: DropdownGroup[]
  disabled?: boolean
  id?: string
  className?: string
  style?: React.CSSProperties
  placeholder?: string
}

export function Dropdown({
  value,
  onChange,
  options,
  groups,
  disabled,
  id,
  className = '',
  style,
  placeholder,
}: DropdownProps) {
  const [open, setOpen] = useState(false)
  const ref = useRef<HTMLDivElement>(null)
  const menuRef = useRef<HTMLDivElement>(null)
  const { isLight } = useTheme()

  const allOptions: DropdownOption[] = options
    ? options
    : groups
      ? groups.flatMap((g) => g.options)
      : []

  const selectedLabel =
    allOptions.find((o) => o.value === value)?.label ?? placeholder ?? ''

  const close = useCallback(() => setOpen(false), [])

  useEffect(() => {
    if (!open) return
    const onClickOutside = (e: MouseEvent) => {
      if (ref.current && !ref.current.contains(e.target as Node)) close()
    }
    const onEsc = (e: KeyboardEvent) => {
      if (e.key === 'Escape') close()
    }
    document.addEventListener('mousedown', onClickOutside)
    document.addEventListener('keydown', onEsc)
    return () => {
      document.removeEventListener('mousedown', onClickOutside)
      document.removeEventListener('keydown', onEsc)
    }
  }, [open, close])

  // Reposition menu if it overflows viewport bottom
  useEffect(() => {
    if (!open || !menuRef.current || !ref.current) return
    const menu = menuRef.current
    const rect = menu.getBoundingClientRect()
    const viewportH = window.innerHeight
    if (rect.bottom > viewportH - 8) {
      // Open upward
      const triggerRect = ref.current.getBoundingClientRect()
      menu.style.bottom = `${triggerRect.height}px`
      menu.style.top = 'auto'
    } else {
      menu.style.top = '100%'
      menu.style.bottom = 'auto'
    }
  }, [open])

  const handleSelect = (val: string) => {
    onChange(val)
    close()
  }

  const themeClass = isLight ? 'ozera-dropdown--light' : 'ozera-dropdown--dark'

  return (
    <div
      ref={ref}
      id={id}
      className={`ozera-dropdown ${themeClass} ${disabled ? 'ozera-dropdown--disabled' : ''} ${className}`}
      style={style}
    >
      <button
        type="button"
        className="ozera-dropdown__trigger"
        onClick={() => !disabled && setOpen(!open)}
        disabled={disabled}
      >
        <span className="ozera-dropdown__label">{selectedLabel}</span>
        <ChevronDown
          size={14}
          className={`ozera-dropdown__chevron ${open ? 'ozera-dropdown__chevron--open' : ''}`}
        />
      </button>

      {open && (
        <div ref={menuRef} className="ozera-dropdown__menu">
          {options &&
            options.map((opt) => (
              <button
                key={opt.value}
                type="button"
                className={`ozera-dropdown__item ${opt.value === value ? 'ozera-dropdown__item--active' : ''} ${opt.disabled ? 'ozera-dropdown__item--disabled' : ''}`}
                onClick={() => !opt.disabled && handleSelect(opt.value)}
                disabled={opt.disabled}
              >
                {opt.label}
              </button>
            ))}
          {groups &&
            groups.map((group) => (
              <div key={group.label} className="ozera-dropdown__group">
                <div className="ozera-dropdown__group-label">{group.label}</div>
                {group.options.map((opt) => (
                  <button
                    key={opt.value}
                    type="button"
                    className={`ozera-dropdown__item ${opt.value === value ? 'ozera-dropdown__item--active' : ''} ${opt.disabled ? 'ozera-dropdown__item--disabled' : ''}`}
                    onClick={() => !opt.disabled && handleSelect(opt.value)}
                    disabled={opt.disabled}
                  >
                    {opt.label}
                  </button>
                ))}
              </div>
            ))}
        </div>
      )}
    </div>
  )
}
