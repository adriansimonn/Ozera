/**
 * The assistant panel's state (zustand): open or not and its width, the chats, the turn being
 * streamed, and the bridge to the page on screen.
 *
 * Pages register a bridge while mounted (hooks/useAssistantPage): a snapshot of what's on screen
 * (sent with every message), an apply function for set_inputs, and a load function that opens a
 * run's result. A page can register in parts (the Generate page's generator and its
 * visualizations); their snapshots merge. Actions and results for a page that isn't mounted yet
 * wait here until it registers. The latest runs' results are kept (in memory only) so their
 * chips can open them again.
 */

import { create } from 'zustand'
import type { QueryClient } from '@tanstack/react-query'
import { assistantClient } from '../api/client'
import { forgetLabels } from '../hooks/useFeatureLabels'
import { PAGE_ROUTES, parseSaeKey } from '../components/assistant/pages/common'
import type {
  AssistantMode,
  AssistantModelId,
  AssistantModels,
  AssistantStreamHandlers,
  ChatDetail,
  ChatInfo,
  FieldValue,
  RunResult,
  ToolChipInfo,
  TurnStatus,
  TurnSummary,
  UiAction,
} from '../types/assistant'

export type Snapshot = Record<string, unknown>

export interface PageBridge {
  page: string
  snapshot: () => Snapshot
  apply: (fields: Record<string, FieldValue>) => void
  load: (tool: string, data: unknown) => void
}

/** A streamed turn's part: text keeps the API call it came in (a declined call's text is dropped). */
export type DraftPart = { type: 'text'; text: string; call: number } | { type: 'tool'; id: string }

/** The turn being streamed: its text and tool calls in order, its tools, and how it's going. */
export interface Draft {
  text: string
  parts: DraftPart[]
  /** The API call streaming */
  call: number
  tools: ToolChipInfo[]
  status: 'waiting' | 'streaming' | 'busy' | 'stopping' | 'running'
  declined: string | null | undefined
  /** It goes on with the last turn (after Run or Skip) instead of starting one */
  resumed: boolean
}

// Run results kept for reopening from their chips
const RESULTS_KEPT = 6

const WIDTH_KEY = 'ozera-assistant-width'
const OPEN_KEY = 'ozera-assistant-open'
export const MIN_WIDTH = 320
export const MAX_WIDTH = 760
const DEFAULT_WIDTH = 420

function readNumber(key: string, fallback: number): number {
  try {
    const value = Number(localStorage.getItem(key))
    return Number.isFinite(value) && value > 0 ? value : fallback
  } catch {
    return fallback
  }
}

function write(key: string, value: string) {
  try {
    localStorage.setItem(key, value)
  } catch {
    // Storage may be unavailable (private windows); the setting just isn't kept
  }
}

/** Where the app is (path and query), and the page's bridges merged into one snapshot. */
function currentSnapshot(bridges: PageBridge[]): Snapshot {
  const route = window.location.pathname + window.location.search
  const merged: Snapshot = {}
  for (const bridge of bridges) {
    try {
      Object.assign(merged, bridge.snapshot())
    } catch (err) {
      console.error('Assistant snapshot failed:', err)
    }
  }
  return { route, ...merged }
}

/** A turn's text: each API call's text as its own paragraph(s). */
function draftText(parts: DraftPart[]): string {
  return parts
    .map((p) => (p.type === 'text' ? p.text : ''))
    .filter((t) => t.trim())
    .join('\n\n')
}

/** Scroll to a control and make it pulse. */
function pulse(page: string, control: string): boolean {
  const element = document.querySelector(`[data-assist="${page}.${control}"]`)
  if (!(element instanceof HTMLElement)) return false
  element.scrollIntoView({ behavior: 'smooth', block: 'center' })
  element.classList.remove('ast-pulse')
  // Restart the animation
  void element.offsetWidth
  element.classList.add('ast-pulse')
  window.setTimeout(() => element.classList.remove('ast-pulse'), 2600)
  return true
}

interface AssistantState {
  open: boolean
  width: number
  models: AssistantModels | null
  chats: ChatInfo[]
  chat: ChatDetail | null
  draft: Draft | null
  /** The message being sent (shown until its turn is stored) */
  sending: string | null
  error: string | null
  bridges: PageBridge[]
  pending: UiAction[]
  /** Run results waiting for their page to mount */
  queued: RunResult[]
  /** The latest runs' results, by tool call ID */
  results: Record<string, RunResult>
  /** Set when Allow found a price had gone up (the card shows the new one) or the runs over the budget */
  notice: string | null
  /** The mode and budget the next new chat starts with */
  newMode: AssistantMode
  newBudget: number | null
  navigator: ((route: string) => void) | null
  queryClient: QueryClient | null
  abort: AbortController | null

  setOpen: (open: boolean) => void
  toggle: () => void
  setWidth: (width: number) => void
  setNavigator: (navigate: ((route: string) => void) | null) => void
  setQueryClient: (client: QueryClient | null) => void
  setError: (error: string | null) => void
  load: () => Promise<void>
  selectChat: (id: number | null) => Promise<void>
  newChat: () => void
  rename: (id: number, title: string) => Promise<void>
  remove: (id: number) => Promise<void>
  setModel: (model: AssistantModelId) => Promise<void>
  setMode: (mode: AssistantMode) => Promise<void>
  setBudget: (budget: number | null) => Promise<void>
  send: (text: string) => Promise<void>
  approve: (decisions: Record<string, 'run' | 'skip'>) => Promise<void>
  stop: () => Promise<void>
  register: (bridge: PageBridge) => () => void
  applyAction: (action: UiAction) => void
  applyResult: (result: RunResult) => void
}

export const useAssistantStore = create<AssistantState>()((set, get) => ({
  open: (() => {
    try {
      return localStorage.getItem(OPEN_KEY) === '1'
    } catch {
      return false
    }
  })(),
  width: Math.min(MAX_WIDTH, Math.max(MIN_WIDTH, readNumber(WIDTH_KEY, DEFAULT_WIDTH))),
  models: null,
  chats: [],
  chat: null,
  draft: null,
  sending: null,
  error: null,
  bridges: [],
  pending: [],
  queued: [],
  results: {},
  notice: null,
  newMode: 'manual',
  newBudget: null,
  navigator: null,
  queryClient: null,
  abort: null,

  setOpen: (open) => {
    write(OPEN_KEY, open ? '1' : '0')
    set({ open })
    if (open && !get().models) get().load()
  },
  toggle: () => get().setOpen(!get().open),
  setWidth: (width) => {
    const clamped = Math.round(Math.min(MAX_WIDTH, Math.max(MIN_WIDTH, width)))
    write(WIDTH_KEY, String(clamped))
    set({ width: clamped })
  },
  setNavigator: (navigator) => set({ navigator }),
  setQueryClient: (queryClient) => set({ queryClient }),
  setError: (error) => set({ error }),

  load: async () => {
    try {
      const [models, chats] = await Promise.all([assistantClient.models(), assistantClient.chats()])
      set({ models, chats, error: null })
      const { chat } = get()
      if (!chat && chats.length > 0) await get().selectChat(chats[0].id)
    } catch (err) {
      set({ error: err instanceof Error ? err.message : 'Failed to load the assistant' })
    }
  },

  selectChat: async (id) => {
    if (get().draft) return
    if (id == null) {
      set({ chat: null })
      return
    }
    try {
      set({ chat: await assistantClient.chat(id), error: null })
    } catch (err) {
      set({ error: err instanceof Error ? err.message : 'Failed to load the chat' })
    }
  },

  newChat: () => {
    if (get().draft) return
    // A new chat starts in manual mode without a budget
    set({ chat: null, error: null, newMode: 'manual', newBudget: null })
  },

  rename: async (id, title) => {
    const updated = await assistantClient.updateChat(id, { title })
    set((s) => ({
      chats: s.chats.map((c) => (c.id === id ? updated : c)),
      chat: s.chat?.id === id ? { ...s.chat, ...updated } : s.chat,
    }))
  },

  remove: async (id) => {
    await assistantClient.deleteChat(id)
    set((s) => ({ chats: s.chats.filter((c) => c.id !== id), chat: s.chat?.id === id ? null : s.chat }))
  },

  setModel: async (model) => {
    const { chat } = get()
    if (!chat) {
      // The next new chat starts on it
      set((s) => ({ models: s.models ? { ...s.models, default: model } : s.models }))
      return
    }
    try {
      const updated = await assistantClient.updateChat(chat.id, { model })
      set((s) => ({ chat: s.chat ? { ...s.chat, ...updated } : s.chat, chats: s.chats.map((c) => (c.id === updated.id ? updated : c)) }))
    } catch (err) {
      set({ error: err instanceof Error ? err.message : 'Failed to switch models' })
    }
  },

  setMode: async (mode) => {
    const { chat } = get()
    if (!chat) {
      set({ newMode: mode })
      return
    }
    // Shown at once; it applies from the chat's next tool calls on
    set((s) => ({ chat: s.chat ? { ...s.chat, mode } : s.chat }))
    try {
      const updated = await assistantClient.updateChat(chat.id, { mode })
      set((s) => ({ chat: s.chat ? { ...s.chat, ...updated } : s.chat, chats: s.chats.map((c) => (c.id === updated.id ? updated : c)) }))
    } catch (err) {
      set((s) => ({ chat: s.chat ? { ...s.chat, mode: chat.mode } : s.chat, error: err instanceof Error ? err.message : 'Failed to switch modes' }))
    }
  },

  setBudget: async (budget) => {
    const { chat } = get()
    if (!chat) {
      set({ newBudget: budget })
      return
    }
    try {
      const updated = await assistantClient.updateChat(chat.id, { budget })
      set((s) => ({
        chat: s.chat ? { ...s.chat, ...updated } : s.chat,
        chats: s.chats.map((c) => (c.id === updated.id ? updated : c)),
        error: s.error === 'BUDGET_REACHED' ? null : s.error,
      }))
    } catch (err) {
      set({ error: err instanceof Error ? err.message : 'Failed to set the budget' })
    }
  },

  send: async (text) => {
    const message = text.trim()
    if (!message || get().draft) return
    set({ error: null, notice: null, sending: message, draft: newDraft(false) })
    let chatId = get().chat?.id
    try {
      if (chatId == null) {
        const created = await assistantClient.createChat(get().models?.default, get().newMode, get().newBudget)
        chatId = created.id
        set((s) => ({ chat: { ...created, turns: [] }, chats: [created, ...s.chats] }))
      }
    } catch (err) {
      set({ draft: null, sending: null, error: err instanceof Error ? err.message : 'Failed to start a chat' })
      return
    }
    const id = chatId
    await streamTurn(id, (handlers, signal) => assistantClient.send(id, message, currentSnapshot(get().bridges), handlers, signal))
  },

  approve: async (decisions) => {
    const { chat, draft } = get()
    if (!chat || draft || !chat.pending?.length) return
    set({ error: null, notice: null, draft: newDraft(true) })
    await streamTurn(chat.id, (handlers, signal) => assistantClient.approve(chat.id, decisions, handlers, signal))
  },

  stop: async () => {
    const { chat, draft } = get()
    if (!chat || !draft) return
    set({ draft: { ...draft, status: 'stopping' } })
    try {
      await assistantClient.stop(chat.id)
    } catch (err) {
      set({ error: err instanceof Error ? err.message : 'Failed to stop' })
    }
  },

  register: (bridge) => {
    set((s) => ({ bridges: [...s.bridges, bridge] }))
    // Actions and results waiting for this page, applied once it has mounted
    window.setTimeout(() => {
      const waiting = get().pending.filter((a) => a.page === bridge.page)
      const results = get().queued.filter((r) => r.page === bridge.page)
      if (!waiting.length && !results.length) return
      set((s) => ({ pending: s.pending.filter((a) => a.page !== bridge.page), queued: s.queued.filter((r) => r.page !== bridge.page) }))
      waiting.forEach((a) => get().applyAction(a))
      results.forEach((r) => deliver(get().bridges.filter((b) => b.page === r.page), r))
    }, 0)
    return () => set((s) => ({ bridges: s.bridges.filter((b) => b !== bridge) }))
  },

  applyAction: (action) => {
    if (action.type === 'navigate') {
      get().navigator?.(action.route)
      return
    }
    const bridges = get().bridges.filter((b) => b.page === action.page)
    if (!bridges.length) {
      set((s) => ({ pending: [...s.pending, action] }))
      return
    }
    if (action.type === 'set_inputs') {
      bridges.forEach((b) => {
        try {
          b.apply(action.fields)
        } catch (err) {
          console.error('Assistant set_inputs failed:', err)
        }
      })
    } else if (action.type === 'highlight') {
      // After any inputs set in the same reply have rendered
      window.setTimeout(() => pulse(action.page, action.control), 150)
    }
  },

  applyResult: (result) => {
    if (result.labels) {
      // A feature was explained or tested: its chips and the pages' labels read it again
      const sae = parseSaeKey(result.labels)
      const client = get().queryClient
      if (sae && client) forgetLabels(client, sae)
    }
    if (!result.page) return
    // Pages register while mounted: a page that isn't gets opened, and takes the result once it registers
    const bridges = get().bridges.filter((b) => b.page === result.page)
    if (result.route) get().navigator?.(result.route)
    else if (!bridges.length) get().navigator?.(PAGE_ROUTES[result.page]?.route ?? '/')
    if (bridges.length) deliver(bridges, result)
    else set((s) => ({ queued: [...s.queued.filter((r) => r.id !== result.id), result] }))
  },
}))

function deliver(bridges: PageBridge[], result: RunResult) {
  bridges.forEach((b) => {
    try {
      b.load(result.tool, result.data)
    } catch (err) {
      console.error('Assistant result failed to open:', err)
    }
  })
}

function newDraft(resumed: boolean): Draft {
  return { text: '', parts: [], call: -1, tools: [], status: resumed ? 'running' : 'waiting', declined: undefined, resumed }
}

/**
 * Stream a turn (a message, or the rest of a paused turn after Run or Skip) into the draft,
 * then replace the draft with the stored chat.
 */
async function streamTurn(chatId: number, start: (handlers: AssistantStreamHandlers, signal: AbortSignal) => Promise<void>) {
  const { getState: get, setState: set } = useAssistantStore
  const abort = new AbortController()
  set({ abort })
  const update = (change: (d: Draft) => Partial<Draft>) =>
    set((s) => (s.draft ? { draft: { ...s.draft, ...change(s.draft) } } : {}))
  let summary: Partial<TurnSummary> | null = null
  try {
    await start({
      onCall: (index) =>
        update((d) => ({ call: index, status: d.status === 'stopping' ? 'stopping' : 'streaming' })),
      onText: (chunk) =>
        update((d) => {
          const parts = [...d.parts]
          const last = parts[parts.length - 1]
          if (last?.type === 'text' && last.call === d.call) parts[parts.length - 1] = { ...last, text: last.text + chunk }
          else parts.push({ type: 'text', text: chunk, call: d.call })
          return { parts, text: draftText(parts), status: d.status === 'stopping' ? 'stopping' : 'streaming' }
        }),
      onTool: (tool) =>
        update((d) => {
          const known = d.tools.some((t) => t.id === tool.id)
          return {
            tools: known ? d.tools.map((t) => (t.id === tool.id ? { ...t, ...tool } : t)) : [...d.tools, tool],
            parts: known ? d.parts : [...d.parts, { type: 'tool', id: tool.id }],
            status: tool.kind === 'run' && tool.status === 'running' ? 'running' : d.status,
          }
        }),
      onUi: (action) => get().applyAction(action),
      onResult: (result) => {
        set((s) => {
          const kept = Object.entries({ ...s.results, [result.id]: result }).slice(-RESULTS_KEPT)
          return { results: Object.fromEntries(kept) }
        })
        get().applyResult(result)
      },
      onStatus: (status) => update((d) => ({ status: status === 'busy' ? 'busy' : d.status })),
      onRefusal: (category) =>
        update((d) => {
          // A declined call's text isn't kept
          const parts = d.parts.filter((p) => p.type !== 'text' || p.call !== d.call)
          return { parts, text: draftText(parts), declined: category }
        }),
      onDone: (s) => {
        summary = s
      },
      onError: (s) => {
        summary = { ...s, status: 'error' as TurnStatus }
        if (s.code === 'INSUFFICIENT_CREDITS') set({ error: 'INSUFFICIENT_CREDITS' })
      },
    }, abort.signal)
  } catch (err) {
    if (err instanceof Error && err.message === 'PRICE_CHANGED') {
      set({ notice: 'A price went up since the card was shown. Check it and run again.' })
    } else if (err instanceof Error && err.message === 'OVER_BUDGET') {
      set({ notice: 'Over this chat’s budget. Raise it in settings, or leave runs out.' })
    } else if (!(err instanceof DOMException && err.name === 'AbortError')) {
      set({ error: err instanceof Error ? err.message : 'The assistant failed' })
    }
  }
  // The stored turn replaces the draft
  try {
    const [chat, chats, models] = await Promise.all([
      assistantClient.chat(chatId),
      assistantClient.chats(),
      assistantClient.models(),
    ])
    set({ chat, chats, models })
  } catch {
    // Keep what's shown
  }
  const s = summary as Partial<TurnSummary> | null
  if (s?.status === 'error' && s.message && get().error == null) set({ error: s.message })
  set({ draft: null, sending: null, abort: null })
}
