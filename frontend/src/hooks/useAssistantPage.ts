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
  /** The controls this part fills (all of the page's when unset); it's sent every field */
  controls?: readonly string[]
  /** The tools whose results this part opens (all when unset); it's sent every result */
  tools?: readonly string[]
  snapshot: () => Snapshot
  /** Return false to leave its fields for later (they wait for a part that takes them) */
  apply: (fields: Record<string, FieldValue>) => boolean | void
  /** A run's full result (by its tool's name) to open on the page; false leaves it for later */
  load?: (tool: string, data: unknown) => boolean | void
}

export function useAssistantPage(page: string, bridge: BridgeParts) {
  const latest = useRef(bridge)
  latest.current = bridge
  useEffect(() => {
    const registered: PageBridge = {
      page,
      // What a part takes is fixed when it registers
      controls: latest.current.controls,
      tools: latest.current.tools,
      snapshot: () => latest.current.snapshot(),
      apply: (fields) => latest.current.apply(fields),
      load: (tool, data) => latest.current.load?.(tool, data),
    }
    return useAssistantStore.getState().register(registered)
  }, [page])
}
