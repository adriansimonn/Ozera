/**
 * A chat as Markdown (ASSISTANT_PLAN.md, Phase 3): each message and reply with its numbers, what
 * the assistant read, showed and ran (runs with their details and charges), and the ozera: links
 * as links into the app. A feature link also carries the feature's stored explanation and score,
 * as its chip shows it, so a claim about it can be checked against it.
 */

import { featureLabels } from '../../hooks/useFeatureLabels'
import type { FeatureLabel } from '../../types/autointerp'
import type { AssistantModels, ChatDetail, ChatTurn, ToolChipInfo } from '../../types/assistant'
import { saeBrowserUrl } from '../lab/saeLinks'
import { PAGE_ROUTES, parseSaeKey } from './pages/common'

const FEATURE_LINK = /\[([^\]]*)\]\((ozera:feature\?[^)\s]+)\)/g
const PAGE_LINK = /\[([^\]]*)\]\(ozera:page\/([a-z_]+)\)/g

const STATUS: Partial<Record<ChatTurn['status'], string>> = {
  stopped: 'Stopped.',
  max_tokens: 'Cut off at the reply limit.',
  limit: 'Reached the limit of tool calls.',
  full: 'The chat was full.',
  skipped: 'Skipped.',
  budget: 'Reached the chat’s budget.',
  refused: 'Claude declined.',
  awaiting_approval: 'Waiting for approval.',
}

interface FeatureRef {
  sae: string
  feature: number
  model?: string
}

function parseFeature(href: string): FeatureRef | null {
  const params = new URLSearchParams(href.slice('ozera:feature?'.length))
  const sae = params.get('sae') ?? ''
  const feature = Number(params.get('n'))
  if (!parseSaeKey(sae) || !Number.isInteger(feature) || feature < 0) return null
  return { sae, feature, model: params.get('model') || undefined }
}

function scopeOf(f: FeatureRef): string {
  return `${f.sae}|${f.model ?? ''}`
}

function money(value: number): string {
  return value < 0.01 ? `$${value.toFixed(4)}` : `$${value.toFixed(3)}`
}

/** Text that sits inside Markdown without starting a link, emphasis or code. */
function plain(text: string): string {
  return text.replace(/\s+/g, ' ').replace(/([\\`*_[\]<>|])/g, '\\$1').trim()
}

function describe(label: FeatureLabel): string {
  const score = label.score != null ? `score ${label.score.toFixed(2)}` : label.status === 'unscored' ? 'unscored' : null
  return [`“${plain(label.explanation)}”`, score, label.token_feature ? 'token feature' : null].filter(Boolean).join(', ')
}

/** Each linked feature's stored label, by "sae|model" then feature (features without one are left out). */
async function labelsOf(texts: string[]): Promise<Map<string, Map<number, string>>> {
  const wanted = new Map<string, { ref: FeatureRef; features: Set<number> }>()
  for (const text of texts) {
    for (const match of text.matchAll(FEATURE_LINK)) {
      const f = parseFeature(match[2])
      if (!f) continue
      const entry = wanted.get(scopeOf(f)) ?? { ref: f, features: new Set<number>() }
      entry.features.add(f.feature)
      wanted.set(scopeOf(f), entry)
    }
  }
  const out = new Map<string, Map<number, string>>()
  await Promise.all(
    [...wanted.entries()].map(async ([scope, { ref, features }]) => {
      try {
        const labels = await featureLabels(parseSaeKey(ref.sae)!, [...features], ref.model)
        out.set(scope, new Map([...labels.entries()].map(([feature, label]) => [feature, describe(label)])))
      } catch {
        // Exported without labels
      }
    }),
  )
  return out
}

function withLinks(text: string, origin: string, labels: Map<string, Map<number, string>>): string {
  return text
    .replace(FEATURE_LINK, (_, label: string, href: string) => {
      const f = parseFeature(href)
      if (!f) return label
      const stored = labels.get(scopeOf(f))?.get(f.feature)
      return `[${label || `#${f.feature}`}](${origin}${saeBrowserUrl(parseSaeKey(f.sae)!, f.feature)})${stored ? ` (${stored})` : ''}`
    })
    .replace(PAGE_LINK, (_, label: string, id: string) => {
      const page = PAGE_ROUTES[id]
      return page ? `[${label || page.title}](${origin}${page.route})` : label
    })
}

function quoted(text: string): string {
  return text
    .split('\n')
    .map((line) => `> ${line}`)
    .join('\n')
}

function runLine(tool: ToolChipInfo): string {
  const state =
    tool.status === 'skipped' ? 'skipped' : tool.status === 'approval' ? 'waiting for approval' : tool.error || tool.status === 'error' ? 'failed' : null
  const charged = tool.charged != null && !state ? money(tool.charged) : null
  const verb = state ? 'Run' : 'Ran'
  if (tool.auto) {
    const details = tool.auto.details.map(([label, value]) => `${label}: ${plain(value)}`).join('; ')
    return `- ${verb} ${plain(tool.auto.title)} (${details}), in auto mode${charged ? `: ${charged}` : ''}${state ? `: ${state}` : ''}`
  }
  return `- ${verb} ${plain(tool.chip || tool.name)}${charged ? `: ${charged}` : ''}${state ? `: ${state}` : ''}`
}

function toolLines(tools: ToolChipInfo[]): string[] {
  const reads = tools.filter((t) => t.kind === 'read').map((t) => plain(t.chip || t.name))
  const shown = tools.filter((t) => t.kind === 'ui').map((t) => plain(t.chip || t.name))
  const lines = []
  if (reads.length) lines.push(`- Read: ${reads.join('; ')}`)
  if (shown.length) lines.push(`- Showed: ${shown.join('; ')}`)
  return [...lines, ...tools.filter((t) => t.kind === 'run').map(runLine)]
}

/** The chat as Markdown, with links into the app at origin. */
export async function chatMarkdown(chat: ChatDetail, models: AssistantModels | null, origin: string): Promise<string> {
  const labels = await labelsOf(chat.turns.map((t) => t.reply))
  const model = models?.models.find((m) => m.id === chat.model)?.name ?? chat.model
  const runs = chat.turns.flatMap((t) => t.tools).reduce((sum, t) => sum + (t.kind === 'run' && t.charged ? t.charged : 0), 0)
  const out = [
    `# ${plain(chat.title)}`,
    '',
    `Ozera assistant · ${model} · exported ${new Date().toISOString().slice(0, 10)} · assistant ${money(chat.cost)}${runs ? ` · runs ${money(runs)}` : ''}`,
  ]
  for (const turn of chat.turns) {
    const page = turn.page ? PAGE_ROUTES[turn.page]?.title : null
    out.push('', '---', '', `**You**${page ? ` · ${page}` : ''}`, '', quoted(turn.text), '')
    if (turn.reply) out.push(withLinks(turn.reply, origin, labels), '')
    const lines = toolLines(turn.tools)
    if (lines.length) out.push(...lines, '')
    const status = turn.status === 'error' ? turn.detail.message || 'The assistant failed.' : STATUS[turn.status]
    out.push(`*${[status, turn.free ? 'free' : money(turn.cost)].filter(Boolean).join(' · ')}*`)
  }
  return out.join('\n') + '\n'
}

/** Download the chat as a .md file. */
export async function downloadChat(chat: ChatDetail, models: AssistantModels | null): Promise<void> {
  const markdown = await chatMarkdown(chat, models, window.location.origin)
  const url = URL.createObjectURL(new Blob([markdown], { type: 'text/markdown;charset=utf-8' }))
  const link = document.createElement('a')
  link.href = url
  link.download = `${chat.title.replace(/[^\w\- ]+/g, '').trim().replace(/\s+/g, '-').slice(0, 60) || 'ozera-chat'}.md`
  document.body.appendChild(link)
  link.click()
  link.remove()
  window.setTimeout(() => URL.revokeObjectURL(url), 1000)
}
