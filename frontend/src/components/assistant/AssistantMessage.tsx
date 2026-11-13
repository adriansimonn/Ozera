/**
 * An assistant reply: Markdown (no raw HTML), with ozera: links as chips. A feature link
 * (ozera:feature?sae=<key>&n=<feature>&model=<model>) shows as a FeatureLink with the feature's
 * stored explanation, so a wrong claim about it is visible next to it; a page link
 * (ozera:page/<id>) opens that page in the app.
 */

import { memo } from 'react'
import ReactMarkdown, { defaultUrlTransform, type Components } from 'react-markdown'
import remarkGfm from 'remark-gfm'
import { useNavigate } from 'react-router-dom'
import { FeatureLink } from '../lab/FeatureLink'
import { PAGE_ROUTES, parseSaeKey } from './pages/common'

/** ozera: links pass; everything else gets react-markdown's own URL check. */
function urlTransform(url: string): string {
  return url.startsWith('ozera:') ? url : defaultUrlTransform(url)
}

function FeatureChip({ href, fallback }: { href: string; fallback: React.ReactNode }) {
  const params = new URLSearchParams(href.slice('ozera:feature?'.length))
  const sae = parseSaeKey(params.get('sae') ?? '')
  const feature = Number(params.get('n'))
  if (!sae || !Number.isInteger(feature) || feature < 0) return <>{fallback}</>
  return (
    <span className="ast-feature">
      <FeatureLink sae={sae} feature={feature} model={params.get('model') ?? undefined} />
    </span>
  )
}

function PageChip({ id, children }: { id: string; children: React.ReactNode }) {
  const navigate = useNavigate()
  const page = PAGE_ROUTES[id]
  if (!page) return <>{children}</>
  return (
    <button type="button" className="ast-page-chip" onClick={() => navigate(page.route)} title={`Open ${page.title}`}>
      {children || page.title}
    </button>
  )
}

const components: Components = {
  a: ({ href, children }) => {
    if (href?.startsWith('ozera:feature?')) return <FeatureChip href={href} fallback={children} />
    if (href?.startsWith('ozera:page/')) return <PageChip id={href.slice('ozera:page/'.length)}>{children}</PageChip>
    if (!href) return <>{children}</>
    return (
      <a href={href} target="_blank" rel="noopener noreferrer">
        {children}
      </a>
    )
  },
}

export const AssistantMessage = memo(function AssistantMessage({ text }: { text: string }) {
  return (
    <div className="ast-markdown">
      <ReactMarkdown remarkPlugins={[remarkGfm]} urlTransform={urlTransform} components={components} skipHtml>
        {text}
      </ReactMarkdown>
    </div>
  )
})
