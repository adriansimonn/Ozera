/**
 * Registers a page (or a part of one) with the assistant while it's mounted: what's on screen
 * (snapshot, sent with every message), how the assistant's set_inputs fills it, and how a run's
 * result opens in it (load). The bridge can change every render; the registration reads the
 * latest one.
 *
 * Builders for each page are in components/assistant/pages/, shared by a page's variants.
 */

import { useEffect, useRef } from 'react'
import { useAssistantStore, type PageBridge, type Snapshot } from '../stores/assistantStore'
import type { FieldValue } from '../types/assistant'

export interface BridgeParts {
  snapshot: () => Snapshot
  apply: (fields: Record<string, FieldValue>) => void
  /** A run's full result (by its tool's name) to open on the page */
  load?: (tool: string, data: unknown) => void
}

export function useAssistantPage(page: string, bridge: BridgeParts) {
  const latest = useRef(bridge)
  latest.current = bridge
  useEffect(() => {
    const registered: PageBridge = {
      page,
      snapshot: () => latest.current.snapshot(),
      apply: (fields) => latest.current.apply(fields),
      load: (tool, data) => latest.current.load?.(tool, data),
    }
    return useAssistantStore.getState().register(registered)
  }, [page])
}
