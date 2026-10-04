import { useThemeColors } from '../../hooks/useTheme'

/** A note saying which feature a link asked for, until the text is analyzed. */
export function SaeDeepLinkNote({ feature }: { feature: number }) {
  const tc = useThemeColors()
  return (
    <div
      role="status"
      style={{
        margin: '0 0 0.75rem',
        padding: '0.55rem 0.75rem',
        border: `1px solid ${tc.border}`,
        background: tc.surface,
        color: tc.text,
        fontSize: '0.8rem',
        lineHeight: 1.45,
      }}
    >
      From the Probe Lab: feature <strong>#{feature}</strong>. Press Analyze to see where it fires in this text; its details
      open next to the activations.
    </div>
  )
}
