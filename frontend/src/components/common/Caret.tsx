/**
 * The small caret beside something that opens (styles/app.css: .oz-caret). It points down, and
 * turns over when its parent has aria-expanded="true".
 */
export function Caret() {
  return (
    <svg className="oz-caret" viewBox="0 0 10 6" aria-hidden="true">
      <path d="M1 1l4 4 4-4" />
    </svg>
  )
}
