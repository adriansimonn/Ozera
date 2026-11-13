/** The assistant (backend/api/assistant.py). */

export type AssistantModelId = 'claude-sonnet-5-5' | 'claude-opus-5-5' | 'claude-haiku-5-5'

export interface AssistantModelInfo {
  id: AssistantModelId
  name: string
  free: boolean
  /** About what a message costs (an estimate, USD) */
  per_message: number
  /** About what a chat's first call holds */
  first_hold: number
}

export interface AssistantModels {
  available: boolean
  default: AssistantModelId
  free_model: AssistantModelId
  free_per_day: number
  free_left: number
  models: AssistantModelInfo[]
}

/** A run waiting for the user's approval (services/assistant_runs.py's quote). */
export interface PendingRun {
  id: string
  name: string
  chip: string
  title: string
  /** [label, value] pairs: what runs */
  details: [string, string][]
  /** The expected charge (USD) */
  estimate: number
  /** The most it holds or needs the balance to cover, when that's more than the estimate */
  hold: number | null
  note: string
}

export interface ChatInfo {
  id: number
  title: string
  model: AssistantModelId
  status: 'idle' | 'running' | 'awaiting_approval'
  tokens: number
  cost: number
  created_at: string
  updated_at: string
  /** Runs waiting for Run or Skip */
  pending: PendingRun[] | null
  /** What runs may cost in all without asking (USD; 0: every run asks) */
  allowance: number
  /** What runs that started without asking have taken from it */
  allowance_spent: number
}

/** A run's quote, as its approval card shows it */
export type RunQuote = Omit<PendingRun, 'id' | 'name' | 'chip'>

export interface ToolChipInfo {
  id: string
  name: string
  kind: 'read' | 'ui' | 'run' | null
  chip: string
  error?: boolean
  status?: 'running' | 'pricing' | 'approval' | 'done' | 'error' | 'skipped'
  /** A run: what its service charged */
  charged?: number
  /** A run the chat's allowance started without asking: its quote */
  auto?: RunQuote
}

export type TurnStatus =
  | 'running' | 'done' | 'max_tokens' | 'refused' | 'stopped' | 'limit' | 'full' | 'error' | 'approval' | 'awaiting_approval' | 'skipped'

export interface ChatTurn {
  turn: number
  text: string
  page: string | null
  reply: string
  tools: ToolChipInfo[]
  status: TurnStatus
  detail: { message?: string; category?: string | null; code?: string }
  cost: number
  free: boolean
  created_at: string
}

export interface ChatDetail extends ChatInfo {
  turns: ChatTurn[]
}

/** A value set_inputs sets (patches come as a list of objects) */
export type FieldValue = string | number | boolean | Record<string, unknown>[]

/** An action a ui tool sends to the page (see core/assistant/pages.py) */
export type UiAction =
  | { type: 'navigate'; page: string; route: string }
  | { type: 'set_inputs'; page: string; fields: Record<string, FieldValue> }
  | { type: 'highlight'; page: string; control: string }

/**
 * A run's full result, for the page it opens in (services/assistant_runs.py): page is null for
 * a run with nothing to open (its feature labels are refreshed instead, `labels` the SAE key).
 */
export interface RunResult {
  id: string
  tool: string
  page: string | null
  route?: string | null
  labels?: string | null
  data: unknown
}

export interface TurnSummary {
  status: TurnStatus
  message?: string
  category?: string | null
  code?: string
  cost: number
  free: boolean
  chat_cost: number | null
  tokens: number | null
  free_left: number
  /** status approval: the balance, for the card */
  balance?: number
}

export interface AssistantStreamHandlers {
  onStart?: (info: { turn: number; free: boolean; model: string; resumed?: boolean }) => void
  onCall?: (index: number) => void
  onText?: (text: string) => void
  onTool?: (tool: ToolChipInfo) => void
  onUi?: (action: UiAction) => void
  onResult?: (result: RunResult) => void
  onApproval?: (runs: PendingRun[]) => void
  onStatus?: (status: string, seconds?: number) => void
  onRefusal?: (category: string | null) => void
  onDone?: (summary: TurnSummary) => void
  onError?: (summary: Partial<TurnSummary> & { message: string }) => void
}
