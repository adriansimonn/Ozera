/**
 * The assistant's panel, docked on the right of every page (ASSISTANT_PLAN.md). It opens from
 * the NavBar or ⌘K / Ctrl+K and stays open across pages (AssistantHost is mounted outside the
 * routes). Docked, the page gets right padding; below DOCK_MIN_VIEWPORT it overlays the page.
 *
 * Chats are private and stored on the backend; replies stream in, and ui actions the assistant
 * sends (open a page, fill inputs, point at a control) go to the page through the store. A reply's
 * tool calls show as rows between its text, in the order it made them. In manual mode a reply's
 * tool calls wait under it on an approval card; in auto mode they run at once (a run's row expands
 * to its quote and shows its charge). Run results open on their pages. The composer's bar holds
 * the chat's settings (its hard budget) and model, which open upward, and the mode toggle beside
 * Send. A chat exports as Markdown.
 */

import { useCallback, useEffect, useLayoutEffect, useMemo, useRef, useState } from 'react'
import { useLocation, useNavigate } from 'react-router-dom'
import { ArrowUp, Bot, Check, ChevronDown, ChevronUp, Download, Hand, Pencil, Plus, Settings2, Square, Trash2, X, Zap } from 'lucide-react'
import { useAuthStore } from '../../stores/authStore'
import { MAX_WIDTH, MIN_WIDTH, useAssistantStore, type DraftPart } from '../../stores/assistantStore'
import { usePurchaseCredits } from '../../contexts/PurchaseCreditsContext'
import { useQueryClient } from '@tanstack/react-query'
import type { ChatTurn, PendingRun, ToolChipInfo, TurnPart } from '../../types/assistant'
import { ConfirmModal } from '../common/ConfirmModal'
import { ApprovalCard } from './ApprovalCard'
import { AssistantMessage } from './AssistantMessage'
import { ToolStep } from './ToolStep'
import { AssistantStyles } from './AssistantStyles'
import { downloadChat } from './exportChat'

// Below this window width (in screen pixels) the panel overlays the page instead of docking
const DOCK_MIN_VIEWPORT = 1100

const SUGGESTIONS: Record<string, string[]> = {
  sae: ['What does the selected feature fire on?', 'Which SAEs here have explained features?', 'Find features about negation'],
  probes: ['Which layer probes best, against the control task?', 'What is selectivity?', 'List my saved probes'],
  geometry: ['Does this concept form a circle at this layer?', 'What does the shuffled baseline mean?', 'Set up months on Gemma 3 270M'],
  generate: ['Explain the top predictions in this generation', 'What does temperature change?', 'Set up a prompt on SmolLM2 135M'],
  patching: ['What does this result show?', 'Zero-ablate each layer’s MLP on this prompt', 'Patching or mean ablation?'],
  analysis: ['Find induction heads in Ozera nano', 'What does a previous-token head do?', 'Explain these head types'],
  other: ['What can I do on this page?', 'What does an atlas cost?', 'Which models can I use?'],
}

const PAGE_OF_ROUTE: Record<string, string> = {
  '/': 'generate', '/sae': 'sae', '/probes': 'probes', '/geometry': 'geometry', '/patching': 'patching', '/analysis': 'analysis',
}

const STATUS_TEXT: Partial<Record<ChatTurn['status'], string>> = {
  stopped: 'Stopped.',
  max_tokens: 'Cut off at the reply limit.',
  limit: 'Reached this message’s limit of tool calls.',
  full: 'This chat is full. Start a new one.',
  skipped: 'Skipped.',
  budget: 'Reached this chat’s budget.',
}

function money(value: number): string {
  if (value === 0) return '$0'
  return value < 0.01 ? `$${value.toFixed(4)}` : `$${value.toFixed(3)}`
}

function tokens(value: number): string {
  return value >= 1000 ? `${(value / 1000).toFixed(1)}k` : String(value)
}

function dollars(value: number): string {
  return `$${value.toFixed(2)}`
}

// Budget presets (USD for the whole chat, replies and runs) and the most a budget can be
const BUDGETS = [1, 5, 20]
const MAX_BUDGET = 1000

type Item = { type: 'text'; text: string; key: string } | { type: 'tools'; tools: ToolChipInfo[]; key: string }

/**
 * A reply as it reads: its text, and each run of consecutive tool calls as one group, in order.
 * A tool no part names (or every tool, from a backend without parts) goes last.
 */
function sequence(parts: (TurnPart | DraftPart)[], tools: ToolChipInfo[]): Item[] {
  const byId = new Map(tools.map((t) => [t.id, t]))
  const seen = new Set<string>()
  const items: Item[] = []
  const addTool = (tool: ToolChipInfo) => {
    if (seen.has(tool.id)) return
    seen.add(tool.id)
    const last = items[items.length - 1]
    if (last?.type === 'tools') last.tools.push(tool)
    else items.push({ type: 'tools', tools: [tool], key: `tools-${tool.id}` })
  }
  for (const part of parts) {
    if (part.type === 'tool') {
      const tool = byId.get(part.id)
      if (tool) addTool(tool)
    } else if (part.text.trim()) {
      const last = items[items.length - 1]
      if (last?.type === 'text') last.text += `\n\n${part.text}`
      else items.push({ type: 'text', text: part.text, key: `text-${items.length}` })
    }
  }
  tools.forEach(addTool)
  return items
}

function ReplyView({ parts, tools }: { parts: (TurnPart | DraftPart)[]; tools: ToolChipInfo[] }) {
  const items = useMemo(() => sequence(parts, tools), [parts, tools])
  return (
    <>
      {items.map((item) =>
        item.type === 'text' ? (
          <AssistantMessage key={item.key} text={item.text} />
        ) : (
          <div key={item.key} className="ast-steps">
            {item.tools.map((t) => (
              <ToolStep key={t.id} tool={t} />
            ))}
          </div>
        ),
      )}
    </>
  )
}

/** A menu that opens up from the composer, and closes on a click outside it or Escape. */
function useMenu() {
  const [open, setOpen] = useState(false)
  const ref = useRef<HTMLDivElement>(null)
  useEffect(() => {
    if (!open) return
    const away = (e: MouseEvent) => {
      if (ref.current && !ref.current.contains(e.target as Node)) setOpen(false)
    }
    const escape = (e: KeyboardEvent) => {
      if (e.key === 'Escape') setOpen(false)
    }
    document.addEventListener('mousedown', away)
    document.addEventListener('keydown', escape)
    return () => {
      document.removeEventListener('mousedown', away)
      document.removeEventListener('keydown', escape)
    }
  }, [open])
  return { open, setOpen, ref }
}

function shortName(name: string): string {
  return name.replace('Claude ', '')
}

/** The chat's model (switching waits for the reply). */
function ModelControl({ disabled }: { disabled: boolean }) {
  const models = useAssistantStore((s) => s.models)
  const chat = useAssistantStore((s) => s.chat)
  const setModel = useAssistantStore((s) => s.setModel)
  const { open, setOpen, ref } = useMenu()
  const current = models?.models.find((m) => m.id === (chat?.model ?? models?.default))

  useEffect(() => {
    if (disabled) setOpen(false)
  }, [disabled, setOpen])

  if (!models || !current) return null
  return (
    <div className="ast-bar-control" ref={ref}>
      <button
        type="button"
        className="ast-bar-button"
        onClick={() => setOpen((o) => !o)}
        disabled={disabled}
        aria-haspopup="menu"
        aria-expanded={open}
        title="Model"
      >
        <span>{shortName(current.name)}</span>
        <ChevronUp aria-hidden="true" />
      </button>
      {open && (
        <div className="ast-menu ast-menu--up" role="menu">
          <div className="ast-menu-head">Model</div>
          {models.models.map((m) => (
            <button
              key={m.id}
              type="button"
              role="menuitemradio"
              aria-checked={m.id === current.id}
              className="ast-menu-item"
              onClick={() => {
                if (m.id !== current.id) setModel(m.id)
                setOpen(false)
              }}
            >
              {m.id === current.id ? <Check aria-hidden="true" /> : <span className="ast-menu-check" />}
              <span className="ast-menu-label">{shortName(m.name)}</span>
              <span className="ast-menu-price">{m.free && models.free_left > 0 ? 'free' : `~${money(m.per_message)}/msg`}</span>
            </button>
          ))}
        </div>
      )}
    </div>
  )
}

type Menu = ReturnType<typeof useMenu>

/** The chat's settings: its hard budget (replies and runs stop once the chat has cost that much). */
function SettingsControl({ menu }: { menu: Menu }) {
  const chat = useAssistantStore((s) => s.chat)
  const newBudget = useAssistantStore((s) => s.newBudget)
  const setBudget = useAssistantStore((s) => s.setBudget)
  const { open, setOpen, ref } = menu
  const budget = chat ? chat.budget : newBudget
  const spent = chat?.spent ?? 0
  const [custom, setCustom] = useState('')

  useEffect(() => {
    if (open) setCustom(budget != null && !BUDGETS.includes(budget) ? String(budget) : '')
  }, [open, budget])

  const choose = (value: number | null) => {
    if (value !== budget) setBudget(value)
    setOpen(false)
  }
  const typed = Number(custom)
  const valid = custom.trim() !== '' && Number.isFinite(typed) && typed >= 0.01 && typed <= MAX_BUDGET
  const options = [...new Set([...BUDGETS, ...(budget != null ? [budget] : [])])].sort((a, b) => a - b)
  return (
    <div className="ast-bar-control" ref={ref}>
      <button
        type="button"
        className="ast-bar-icon"
        onClick={() => setOpen((o) => !o)}
        aria-haspopup="dialog"
        aria-expanded={open}
        aria-label="Chat settings"
        title="Chat settings"
      >
        <Settings2 aria-hidden="true" />
      </button>
      {open && (
        <div className="ast-menu ast-menu--up ast-settings" role="dialog" aria-label="Chat settings">
          <div className="ast-menu-head" title="Everything this chat costs, replies and runs. It stops as soon as it reaches this.">
            Budget
          </div>
          <div role="radiogroup" aria-label="Budget">
            <button type="button" role="radio" aria-checked={budget == null} className="ast-menu-item" onClick={() => choose(null)}>
              {budget == null ? <Check aria-hidden="true" /> : <span className="ast-menu-check" />}
              No limit
            </button>
            {options.map((value) => (
              <button key={value} type="button" role="radio" aria-checked={value === budget} className="ast-menu-item" onClick={() => choose(value)}>
                {value === budget ? <Check aria-hidden="true" /> : <span className="ast-menu-check" />}
                {dollars(value)}
              </button>
            ))}
          </div>
          <form
            className="ast-budget"
            onSubmit={(e) => {
              e.preventDefault()
              if (valid) choose(Math.round(typed * 100) / 100)
            }}
          >
            <span className="ast-budget-input">
              <span aria-hidden="true">$</span>
              <input
                type="number"
                inputMode="decimal"
                min={0.01}
                max={MAX_BUDGET}
                step={0.01}
                value={custom}
                placeholder="Custom"
                aria-label="Custom budget in dollars"
                onChange={(e) => setCustom(e.target.value)}
              />
            </span>
            <button type="submit" className="ast-budget-set" disabled={!valid}>
              Set
            </button>
          </form>
          {chat && (
            <div className={`ast-menu-foot${budget != null && spent >= budget ? ' ast-warn' : ''}`}>
              {budget != null ? `${money(spent)} of ${dollars(budget)} used` : `${money(spent)} used`}
            </div>
          )}
        </div>
      )}
    </div>
  )
}

/** Manual (every tool call waits for approval) or auto (tool calls run without asking); a click switches. */
function ModeControl() {
  const chat = useAssistantStore((s) => s.chat)
  const newMode = useAssistantStore((s) => s.newMode)
  const setMode = useAssistantStore((s) => s.setMode)
  const auto = (chat ? chat.mode : newMode) === 'auto'
  return (
    <button
      type="button"
      className="ast-bar-button ast-mode"
      onClick={() => setMode(auto ? 'manual' : 'auto')}
      aria-pressed={auto}
      title={auto ? 'Auto: tool calls run without asking. Click for manual.' : 'Manual: each tool call waits for your approval. Click for auto.'}
    >
      {auto ? <Zap aria-hidden="true" /> : <Hand aria-hidden="true" />}
      <span>{auto ? 'Auto' : 'Manual'}</span>
    </button>
  )
}

function TurnView({
  turn, last, onContinue, onBudget, busy, approval,
}: {
  turn: ChatTurn
  last: boolean
  onContinue: () => void
  onBudget: () => void
  busy: boolean
  /** The approval card, under the turn whose runs wait */
  approval: React.ReactNode
}) {
  const status =
    turn.status === 'refused'
      ? `Claude declined${turn.detail.category ? ` (${turn.detail.category})` : ''}.`
      : turn.status === 'error'
        ? turn.detail.message || 'The assistant failed.'
        : STATUS_TEXT[turn.status]
  return (
    <div className="ast-turn">
      <div className="ast-user">{turn.text}</div>
      <ReplyView parts={turn.parts ?? (turn.reply ? [{ type: 'text', text: turn.reply }] : [])} tools={turn.tools} />
      {approval}
      <div className="ast-meta">
        {status && <span className={turn.status === 'error' || turn.status === 'refused' ? 'ast-warn' : undefined}>{status}</span>}
        {last && turn.status === 'max_tokens' && (
          <button type="button" className="ast-link" onClick={onContinue} disabled={busy}>
            Continue
          </button>
        )}
        {last && turn.status === 'budget' && (
          <button type="button" className="ast-link" onClick={onBudget}>
            Raise budget
          </button>
        )}
        <span className="ast-cost">{turn.free ? 'free' : money(turn.cost)}</span>
      </div>
    </div>
  )
}

function ChatsMenu({ onClose }: { onClose: () => void }) {
  const { chats, chat, selectChat, newChat, rename, remove } = useAssistantStore()
  const [editing, setEditing] = useState<number | null>(null)
  const [title, setTitle] = useState('')
  const [deleting, setDeleting] = useState<number | null>(null)
  const ref = useRef<HTMLDivElement>(null)

  useEffect(() => {
    const away = (e: MouseEvent) => {
      if (ref.current && !ref.current.contains(e.target as Node) && deleting == null) onClose()
    }
    document.addEventListener('mousedown', away)
    return () => document.removeEventListener('mousedown', away)
  }, [onClose, deleting])

  const save = async (id: number) => {
    if (title.trim()) await rename(id, title.trim()).catch(() => {})
    setEditing(null)
  }

  return (
    <div className="ast-menu" ref={ref} role="menu">
      <button type="button" className="ast-menu-item" onClick={() => { newChat(); onClose() }}>
        <Plus aria-hidden="true" /> New chat
      </button>
      {chats.map((c) => (
        <div key={c.id} className={`ast-menu-row${chat?.id === c.id ? ' active' : ''}`}>
          {editing === c.id ? (
            <>
              <input
                className="ast-rename"
                value={title}
                autoFocus
                maxLength={200}
                onChange={(e) => setTitle(e.target.value)}
                onKeyDown={(e) => {
                  if (e.key === 'Enter') save(c.id)
                  if (e.key === 'Escape') setEditing(null)
                }}
              />
              <button type="button" className="ast-icon" aria-label="Save name" onClick={() => save(c.id)}>
                <Check aria-hidden="true" />
              </button>
            </>
          ) : (
            <>
              <button type="button" className="ast-menu-item ast-menu-title" onClick={() => { selectChat(c.id); onClose() }} title={c.title}>
                <span>{c.title}</span>
                <span className="ast-cost">{money(c.spent ?? c.cost)}</span>
              </button>
              <button type="button" className="ast-icon" aria-label="Rename chat" onClick={() => { setEditing(c.id); setTitle(c.title) }}>
                <Pencil aria-hidden="true" />
              </button>
              <button type="button" className="ast-icon" aria-label="Delete chat" onClick={() => setDeleting(c.id)}>
                <Trash2 aria-hidden="true" />
              </button>
            </>
          )}
        </div>
      ))}
      <ConfirmModal
        isOpen={deleting != null}
        title="Delete chat"
        message="Its messages are deleted for good."
        onConfirm={async () => {
          if (deleting != null) await remove(deleting).catch(() => {})
          setDeleting(null)
        }}
        onCancel={() => setDeleting(null)}
      />
    </div>
  )
}

export function AssistantPanel({ docked }: { docked: boolean }) {
  const { models, chat, draft, sending, error, notice, width, newBudget, setOpen, setWidth, send, approve, stop, setError, load } = useAssistantStore()
  const { showPurchaseCredits } = usePurchaseCredits()
  const location = useLocation()
  const [text, setText] = useState('')
  const [menu, setMenu] = useState(false)
  const [exporting, setExporting] = useState(false)
  const panel = useRef<HTMLDivElement>(null)
  const scroller = useRef<HTMLDivElement>(null)
  const input = useRef<HTMLTextAreaElement>(null)
  const pinned = useRef(true)
  const settings = useMenu()

  useEffect(() => {
    if (!models) load()
    input.current?.focus()
  }, [models, load])

  const current = models?.models.find((m) => m.id === (chat?.model ?? models?.default))
  const freeLeft = models?.free_left ?? 0
  const page = PAGE_OF_ROUTE[location.pathname] ?? 'other'
  const turns = chat?.turns ?? []
  const busy = draft != null

  // Keep the newest text in view unless the user scrolled up to read
  useLayoutEffect(() => {
    const el = scroller.current
    if (el && pinned.current) el.scrollTop = el.scrollHeight
  }, [turns.length, draft?.text, draft?.tools.length, sending])

  const submit = useCallback(
    (message: string) => {
      if (!message.trim() || busy) return
      pinned.current = true
      setText('')
      send(message)
    },
    [busy, send],
  )

  // Resizing from the left edge. The page's CSS zoom scales pointer coordinates but not widths,
  // so the pointer's offset is rescaled by layout width over rendered width (as pointerIn does).
  const onResize = (e: React.PointerEvent) => {
    const el = panel.current
    if (!el) return
    e.preventDefault()
    const target = e.currentTarget as HTMLElement
    target.setPointerCapture(e.pointerId)
    const move = (ev: PointerEvent) => {
      const box = el.getBoundingClientRect()
      const scale = box.width > 0 ? el.offsetWidth / box.width : 1
      setWidth((box.right - ev.clientX) * scale)
    }
    const up = () => {
      target.removeEventListener('pointermove', move)
      target.removeEventListener('pointerup', up)
    }
    target.addEventListener('pointermove', move)
    target.addEventListener('pointerup', up)
  }

  // The composer grows with its lines up to its CSS max-height, then scrolls
  useLayoutEffect(() => {
    const el = input.current
    if (!el) return
    el.style.height = 'auto'
    el.style.height = `${el.scrollHeight}px`
  }, [text, width])

  const lastTurn = turns[turns.length - 1]
  const priceLabel = current
    ? `${shortName(current.name)} · ${current.free && freeLeft > 0 ? `free (${freeLeft} left today)` : `~${money(current.per_message)}/msg`}`
    : ''
  const spend = chat
    ? `chat ${money(chat.spent ?? chat.cost)}${chat.budget != null ? ` of ${dollars(chat.budget)}` : ''}, ${tokens(chat.tokens)} tokens`
    : newBudget != null
      ? `budget ${dollars(newBudget)}`
      : ''
  const notes = [
    lastTurn && !busy && lastTurn.status === 'running' ? 'reply saving…' : '',
    !busy && chat?.status === 'awaiting_approval' ? 'a new message skips the waiting calls' : '',
  ].filter(Boolean)

  const exportChat = async () => {
    if (!chat) return
    setExporting(true)
    try {
      await downloadChat(chat, models)
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Export failed')
    } finally {
      setExporting(false)
    }
  }

  return (
    <aside
      ref={panel}
      className={`ast-panel${docked ? ' ast-panel--docked' : ' ast-panel--overlay'}`}
      style={{ width }}
      aria-label="Assistant"
    >
      <div className="ast-resize" onPointerDown={onResize} role="separator" aria-orientation="vertical" aria-valuemin={MIN_WIDTH} aria-valuemax={MAX_WIDTH} aria-valuenow={width} />

      <header className="ast-header">
        <div className="ast-header-row">
          <button type="button" className="ast-chat-title" onClick={() => setMenu((m) => !m)} disabled={busy} title="Chats">
            <span>{chat?.title ?? 'New chat'}</span>
            <ChevronDown aria-hidden="true" />
          </button>
          <button type="button" className="ast-icon" aria-label="New chat" title="New chat" disabled={busy} onClick={() => useAssistantStore.getState().newChat()}>
            <Plus aria-hidden="true" />
          </button>
          <button
            type="button"
            className="ast-icon"
            aria-label="Export as Markdown"
            title="Export as Markdown"
            disabled={busy || exporting || !chat || turns.length === 0}
            onClick={exportChat}
          >
            <Download aria-hidden="true" />
          </button>
          <button type="button" className="ast-icon" aria-label="Close the assistant" title="Close (⌘K)" onClick={() => setOpen(false)}>
            <X aria-hidden="true" />
          </button>
          {menu && <ChatsMenu onClose={() => setMenu(false)} />}
        </div>
      </header>

      <div
        className="ast-body"
        ref={scroller}
        onScroll={(e) => {
          const el = e.currentTarget
          pinned.current = el.scrollHeight - el.scrollTop - el.clientHeight < 40
        }}
      >
        {models && !models.available && <p className="ast-warn ast-note">The assistant isn’t configured on this server.</p>}
        {turns.length === 0 && !sending && (
          <div className="ast-empty">
            <Bot aria-hidden="true" />
            {SUGGESTIONS[page].map((s) => (
              <button key={s} type="button" className="ast-suggestion" onClick={() => submit(s)}>
                {s}
              </button>
            ))}
          </div>
        )}
        {turns.map((turn, i) => {
          const last = i === turns.length - 1
          const pending: PendingRun[] | null = last && !busy && chat?.status === 'awaiting_approval' && chat.pending?.length ? chat.pending : null
          return (
            <TurnView
              key={turn.turn}
              turn={turn}
              last={last && !busy}
              busy={busy}
              onContinue={() => submit('Continue.')}
              onBudget={() => settings.setOpen(true)}
              approval={pending && <ApprovalCard runs={pending} busy={busy} notice={notice} onDecide={approve} onAddCredits={showPurchaseCredits} />}
            />
          )
        })}
        {draft && (sending || draft.resumed) && (
          <div className="ast-turn">
            {sending && <div className="ast-user">{sending}</div>}
            <ReplyView parts={draft.parts} tools={draft.tools} />
            <div className="ast-meta">
              <span className="ast-working">
                {draft.status === 'stopping'
                  ? 'Stopping…'
                  : draft.status === 'busy'
                    ? 'Claude is busy; retrying…'
                    : draft.status === 'running'
                      ? 'Running…'
                      : draft.declined !== undefined
                        ? 'Claude declined.'
                        : draft.text
                          ? ''
                          : 'Thinking…'}
              </span>
            </div>
          </div>
        )}
      </div>

      <footer className="ast-footer">
        {error && (
          <div className="ast-error" role="alert">
            {error === 'BUDGET_REACHED' ? (
              <>
                <span>This chat has used its budget.</span>
                <button type="button" className="ast-link" onClick={() => settings.setOpen(true)}>
                  Raise budget
                </button>
              </>
            ) : error === 'INSUFFICIENT_CREDITS' ? (
              <>
                <span>Your balance doesn’t cover this model{freeLeft > 0 ? `; Haiku 5.5 has ${freeLeft} free messages today` : ''}.</span>
                <button type="button" className="ast-link" onClick={showPurchaseCredits}>
                  Add credits
                </button>
              </>
            ) : (
              <span>{error}</span>
            )}
            <button type="button" className="ast-icon" aria-label="Dismiss" onClick={() => setError(null)}>
              <X aria-hidden="true" />
            </button>
          </div>
        )}
        <div
          className="ast-composer"
          onMouseDown={(e) => {
            // A click on the box's empty space focuses the text, not the bar
            if (e.target === e.currentTarget || (e.target as HTMLElement).classList.contains('ast-composer-bar')) {
              e.preventDefault()
              input.current?.focus()
            }
          }}
        >
          <textarea
            ref={input}
            value={text}
            rows={1}
            maxLength={8000}
            placeholder="Ask about Ozera, your data, or this page"
            onChange={(e) => setText(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === 'Enter' && !e.shiftKey && !e.nativeEvent.isComposing) {
                e.preventDefault()
                submit(text)
              }
            }}
          />
          <div className="ast-composer-bar">
            <SettingsControl menu={settings} />
            <ModelControl disabled={busy} />
            <div className="ast-bar-end">
              <ModeControl />
              {busy ? (
                <button type="button" className="ast-send" aria-label="Stop" title="Stop" onClick={stop} disabled={draft?.status === 'stopping'}>
                  <Square aria-hidden="true" />
                </button>
              ) : (
                <button type="button" className="ast-send" aria-label="Send" title="Send (Enter)" onClick={() => submit(text)} disabled={!text.trim()}>
                  <ArrowUp aria-hidden="true" />
                </button>
              )}
            </div>
          </div>
        </div>
        {(priceLabel || spend) && (
          <div className="ast-footnote">
            <span className="ast-footnote-model" title={priceLabel}>
              {priceLabel}
            </span>
            {spend && (
              <span className="ast-footnote-chat" title="What this chat has cost (replies and runs), its budget, and the size of its context">
                {spend}
              </span>
            )}
          </div>
        )}
        {notes.length > 0 && <div className="ast-footnote-note">{notes.join(' · ')}</div>}
      </footer>
    </aside>
  )
}

/** Mounts the panel (outside the routes), and handles ⌘K / Ctrl+K, page navigation and docking. */
export function AssistantHost() {
  const { user } = useAuthStore()
  const open = useAssistantStore((s) => s.open)
  const width = useAssistantStore((s) => s.width)
  const navigate = useNavigate()
  const queryClient = useQueryClient()
  const [wide, setWide] = useState(() => window.innerWidth >= DOCK_MIN_VIEWPORT)

  useEffect(() => {
    useAssistantStore.getState().setNavigator((route) => navigate(route))
    return () => useAssistantStore.getState().setNavigator(null)
  }, [navigate])

  // Runs that explain or test a feature make its labels stale
  useEffect(() => {
    useAssistantStore.getState().setQueryClient(queryClient)
    return () => useAssistantStore.getState().setQueryClient(null)
  }, [queryClient])

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if ((e.metaKey || e.ctrlKey) && !e.altKey && !e.shiftKey && e.key.toLowerCase() === 'k' && useAuthStore.getState().user) {
        e.preventDefault()
        useAssistantStore.getState().toggle()
      }
    }
    const onResize = () => setWide(window.innerWidth >= DOCK_MIN_VIEWPORT)
    window.addEventListener('keydown', onKey)
    window.addEventListener('resize', onResize)
    return () => {
      window.removeEventListener('keydown', onKey)
      window.removeEventListener('resize', onResize)
    }
  }, [])

  const shown = open && !!user
  const docked = shown && wide
  useEffect(() => {
    const body = document.body
    if (docked) {
      body.dataset.assistant = 'docked'
      body.style.setProperty('--assist-width', `${width}px`)
    } else {
      delete body.dataset.assistant
      body.style.removeProperty('--assist-width')
    }
    return () => {
      delete body.dataset.assistant
      body.style.removeProperty('--assist-width')
    }
  }, [docked, width])

  return (
    <>
      <AssistantStyles />
      {shown && <AssistantPanel docked={docked} />}
    </>
  )
}

/** The NavBar's button that opens and closes the panel. */
export function AssistantButton() {
  const open = useAssistantStore((s) => s.open)
  const toggle = useAssistantStore((s) => s.toggle)
  const mac = typeof navigator !== 'undefined' && /Mac/.test(navigator.platform)
  return (
    <button
      type="button"
      className={`ast-nav-button${open ? ' active' : ''}`}
      onClick={toggle}
      title={`Assistant (${mac ? '⌘' : 'Ctrl+'}K)`}
      aria-pressed={open}
    >
      <Bot aria-hidden="true" />
      <span>Assistant</span>
    </button>
  )
}
