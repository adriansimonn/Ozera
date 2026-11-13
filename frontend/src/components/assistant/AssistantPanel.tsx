/**
 * The assistant's panel, docked on the right of every page (ASSISTANT_PLAN.md). It opens from
 * the NavBar or ⌘K / Ctrl+K and stays open across pages (AssistantHost is mounted outside the
 * routes). Docked, the page gets right padding; below DOCK_MIN_VIEWPORT it overlays the page.
 *
 * Chats are private and stored on the backend; replies stream in, and ui actions the assistant
 * sends (open a page, fill inputs, point at a control) go to the page through the store. Runs a
 * reply asks for wait under it on an approval card; their results open on their pages. With an
 * allowance set for the chat, runs that fit in it start at once and show as cards with their
 * charges. A chat exports as Markdown.
 */

import { useCallback, useEffect, useLayoutEffect, useMemo, useRef, useState } from 'react'
import { useLocation, useNavigate } from 'react-router-dom'
import { ArrowUp, Bot, Check, ChevronDown, Download, Pencil, Plus, Square, Trash2, X } from 'lucide-react'
import { useAuthStore } from '../../stores/authStore'
import { MAX_WIDTH, MIN_WIDTH, useAssistantStore } from '../../stores/assistantStore'
import { usePurchaseCredits } from '../../contexts/PurchaseCreditsContext'
import { useQueryClient } from '@tanstack/react-query'
import type { AssistantModelId, ChatTurn, PendingRun, ToolChipInfo } from '../../types/assistant'
import { ConfirmModal } from '../common/ConfirmModal'
import { Dropdown } from '../common/Dropdown'
import { ApprovalCard, AutoRunCard } from './ApprovalCard'
import { AssistantMessage } from './AssistantMessage'
import { ToolChip } from './ToolChip'
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
  skipped: 'Runs skipped.',
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

// The allowance choices (USD in all, per chat; 0 asks before every run)
const ALLOWANCES = [0, 0.25, 1, 5, 20]

/** A turn's tools: chips for what it read and did, cards for the runs its allowance started. */
function ToolsView({ tools }: { tools: ToolChipInfo[] }) {
  const chips = tools.filter((t) => !t.auto)
  const auto = tools.filter((t) => t.auto)
  return (
    <>
      {chips.length > 0 && (
        <div className="ast-tools">
          {chips.map((t) => (
            <ToolChip key={t.id} tool={t} />
          ))}
        </div>
      )}
      {auto.map((t) => (
        <AutoRunCard key={t.id} tool={t} />
      ))}
    </>
  )
}

/** The chat's allowance: what runs may cost in all without asking. */
function AllowanceControl() {
  const chat = useAssistantStore((s) => s.chat)
  const newAllowance = useAssistantStore((s) => s.newAllowance)
  const setAllowance = useAssistantStore((s) => s.setAllowance)
  const [open, setOpen] = useState(false)
  const ref = useRef<HTMLDivElement>(null)
  const allowance = chat ? chat.allowance ?? 0 : newAllowance
  const spent = chat?.allowance_spent ?? 0
  const left = Math.max(0, allowance - spent)

  useEffect(() => {
    if (!open) return
    const away = (e: MouseEvent) => {
      if (ref.current && !ref.current.contains(e.target as Node)) setOpen(false)
    }
    document.addEventListener('mousedown', away)
    return () => document.removeEventListener('mousedown', away)
  }, [open])

  const options = [...new Set([...ALLOWANCES, allowance])].sort((a, b) => a - b)
  return (
    <div className="ast-allowance" ref={ref}>
      <button
        type="button"
        className={`ast-allowance-button${allowance > 0 ? ' on' : ''}`}
        onClick={() => setOpen((o) => !o)}
        aria-haspopup="menu"
        aria-expanded={open}
        title="Runs up to this much in all start without asking"
      >
        {allowance > 0 ? `${dollars(left)} left` : 'Ask'}
      </button>
      {open && (
        <div className="ast-menu ast-allowance-menu" role="menu">
          <div className="ast-menu-head">Run without asking</div>
          {options.map((value) => (
            <button
              key={value}
              type="button"
              role="menuitemradio"
              aria-checked={value === allowance}
              className="ast-menu-item"
              onClick={() => {
                setAllowance(value)
                setOpen(false)
              }}
            >
              {value === allowance ? <Check aria-hidden="true" /> : <span className="ast-menu-check" />}
              {value === 0 ? 'Off' : `Up to ${dollars(value)}`}
            </button>
          ))}
          {spent > 0 && <div className="ast-menu-foot">{money(spent)} used in this chat</div>}
        </div>
      )}
    </div>
  )
}

function TurnView({
  turn, last, onContinue, busy, approval,
}: {
  turn: ChatTurn
  last: boolean
  onContinue: () => void
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
      {turn.reply && <AssistantMessage text={turn.reply} />}
      <ToolsView tools={turn.tools} />
      {approval}
      <div className="ast-meta">
        {status && <span className={turn.status === 'error' || turn.status === 'refused' ? 'ast-warn' : undefined}>{status}</span>}
        {last && turn.status === 'max_tokens' && (
          <button type="button" className="ast-link" onClick={onContinue} disabled={busy}>
            Continue
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
                <span className="ast-cost">{money(c.cost)}</span>
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
  const { models, chat, draft, sending, error, notice, width, setOpen, setWidth, setModel, send, approve, stop, setError, load } = useAssistantStore()
  const { showPurchaseCredits } = usePurchaseCredits()
  const location = useLocation()
  const [text, setText] = useState('')
  const [menu, setMenu] = useState(false)
  const [exporting, setExporting] = useState(false)
  const panel = useRef<HTMLDivElement>(null)
  const scroller = useRef<HTMLDivElement>(null)
  const input = useRef<HTMLTextAreaElement>(null)
  const pinned = useRef(true)

  useEffect(() => {
    if (!models) load()
    input.current?.focus()
  }, [models, load])

  const modelId: AssistantModelId | undefined = chat?.model ?? models?.default
  const current = models?.models.find((m) => m.id === modelId)
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

  const modelOptions = useMemo(
    () => (models?.models ?? []).map((m) => ({ value: m.id, label: m.name.replace('Claude ', '') })),
    [models],
  )

  // The composer grows with its lines up to its CSS max-height, then scrolls
  useLayoutEffect(() => {
    const el = input.current
    if (!el) return
    el.style.height = 'auto'
    el.style.height = `${el.scrollHeight}px`
  }, [text, width])

  const lastTurn = turns[turns.length - 1]
  const priceLabel = current
    ? `${current.name.replace('Claude ', '')} · ${current.free && freeLeft > 0 ? `free (${freeLeft} left today)` : `~${money(current.per_message)}/msg`}`
    : ''
  const footnote = [
    chat ? `chat ${money(chat.cost)}, ${tokens(chat.tokens)} tokens` : '',
    lastTurn && !busy && lastTurn.status === 'running' ? 'reply saving…' : '',
    !busy && chat?.status === 'awaiting_approval' ? 'a new message skips the waiting runs' : '',
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
        <div className="ast-header-row ast-model">
          {modelId && (
            <Dropdown value={modelId} onChange={(v) => setModel(v as AssistantModelId)} options={modelOptions} disabled={busy} />
          )}
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
              approval={pending && <ApprovalCard runs={pending} busy={busy} notice={notice} onDecide={approve} onAddCredits={showPurchaseCredits} />}
            />
          )
        })}
        {draft && (sending || draft.resumed) && (
          <div className="ast-turn">
            {sending && <div className="ast-user">{sending}</div>}
            {draft.text && <AssistantMessage text={draft.text} />}
            <ToolsView tools={draft.tools} />
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
            {error === 'INSUFFICIENT_CREDITS' ? (
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
            <AllowanceControl />
            <span className="ast-composer-model" title={priceLabel}>
              {priceLabel}
            </span>
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
        {footnote.length > 0 && (
          <div className="ast-footnote" title="What this chat has cost, and the size of its context">
            {footnote.join(' · ')}
          </div>
        )}
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
