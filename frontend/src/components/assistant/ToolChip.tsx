/**
 * What the assistant read, did or ran, as a chip under its reply (e.g. "Feature 1234 · Gemma
 * Scope 270M L9"). A run shows what its service charged; while its result is still in memory,
 * clicking it opens the result on its page again.
 */

import { AlertCircle, Check, Eye, Hourglass, Loader2, MousePointerClick, Play, SkipForward } from 'lucide-react'
import { useAssistantStore } from '../../stores/assistantStore'
import type { ToolChipInfo } from '../../types/assistant'

function money(value: number): string {
  return value < 0.01 ? `$${value.toFixed(4)}` : `$${value.toFixed(3)}`
}

export function ToolChip({ tool }: { tool: ToolChipInfo }) {
  const result = useAssistantStore((s) => s.results[tool.id])
  const running = tool.status === 'running' || tool.status === 'pricing'
  const failed = tool.error || tool.status === 'error'
  const Icon = running
    ? Loader2
    : failed
      ? AlertCircle
      : tool.status === 'approval'
        ? Hourglass
        : tool.status === 'skipped'
          ? SkipForward
          : tool.kind === 'ui'
            ? MousePointerClick
            : tool.kind === 'read'
              ? Eye
              : tool.kind === 'run'
                ? Play
                : Check
  const label = (
    <>
      <Icon className={running ? 'ast-spin' : undefined} aria-hidden="true" />
      <span>{tool.chip || tool.name}</span>
      {tool.kind === 'run' && tool.charged != null && tool.status === 'done' && <span className="ast-tool-cost">{money(tool.charged)}</span>}
      {tool.status === 'skipped' && <span className="ast-tool-cost">skipped</span>}
    </>
  )
  const classes = `ast-tool${failed ? ' ast-tool--error' : ''}${tool.status === 'skipped' ? ' ast-tool--skipped' : ''}${tool.kind === 'run' ? ' ast-tool--run' : ''}`
  if (result?.page && !failed) {
    return (
      <button type="button" className={`${classes} ast-tool--open`} title="Open this result on its page" onClick={() => useAssistantStore.getState().applyResult(result)}>
        {label}
      </button>
    )
  }
  return (
    <span className={classes} title={failed ? `${tool.name} failed` : tool.status === 'approval' ? 'Waiting for Run or Skip' : tool.name}>
      {label}
    </span>
  )
}
