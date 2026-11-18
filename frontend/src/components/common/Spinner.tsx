/**
 * Something is working: three square cells lit in turn (styles/app.css), in the text's color and
 * scaled to its size.
 */

interface Props {
  large?: boolean
  className?: string
  /** What's working, for screen readers; without it the spinner is decoration beside a label */
  label?: string
}

export function Spinner({ large, className, label }: Props) {
  return (
    <span
      className={`oz-spinner${large ? ' oz-spinner--large' : ''}${className ? ` ${className}` : ''}`}
      role={label ? 'status' : undefined}
      aria-label={label}
      aria-hidden={label ? undefined : true}
    >
      <i />
      <i />
      <i />
    </span>
  )
}

export default Spinner
