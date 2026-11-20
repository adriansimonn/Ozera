/**
 * The Graph view: a traced prompt's attribution graph, how much of it is features rather than
 * error, the node that's selected (what its decoder promotes, where it fires, its strongest
 * inputs and outputs), and the feature nodes ranked by influence.
 */

import type { CircuitLab } from '../../hooks/useCircuitLab'
import type { CircuitGraph, GraphNode } from '../../types/circuits'
import { Spinner } from '../common/Spinner'
import { FigureExport } from '../attribution/FigureExport'
import { Findings, Placeholder, Segmented, StatTiles, type Finding } from '../lab/chartKit'
import { useLabColors } from '../lab/labTheme'
import { CircuitGraphChart } from './CircuitGraphChart'
import {
  SHOWN_OPTIONS,
  errorShare,
  logits,
  money,
  nodeEdges,
  nodeName,
  nodeTitle,
  percent,
  showToken,
  shownNodes,
  signedLogits,
  signedPercent,
  type NodeEdge,
} from './circuitUtils'

const LISTED_EDGES = 6
const LISTED_FEATURES = 40

function findings(graph: CircuitGraph): Finding[] {
  const out: Finding[] = []
  const errors = errorShare(graph)
  if (errors > 0.5) {
    out.push({ level: 'warning', text: `Error nodes carry ${percent(errors)} of the target’s attribution: the features explain less than half of it.` })
  } else {
    out.push({ level: 'good', text: `${percent(graph.scores.replacement)} of the target’s attribution flows from the tokens through features; error nodes carry ${percent(errors)}.` })
  }
  if (graph.target.rank > 0) {
    out.push({ level: 'info', text: `${showToken(graph.target.token)} is the model’s #${graph.target.rank + 1} prediction (${percent(graph.target.prob)}); its top is ${showToken(graph.predictions[0]?.token)}.` })
  }
  if (graph.counts.traced_features < graph.counts.active_features) {
    out.push({ level: 'info', text: `${graph.counts.traced_features.toLocaleString()} of ${graph.counts.active_features.toLocaleString()} active features were traced, the most influential first.` })
  }
  out.push({ level: 'info', text: 'Edges are direct effects with attention frozen. Nothing here is tested until you intervene.' })
  return out
}

function EdgeList({ title, edges, lab, hint }: { title: string; edges: NodeEdge[]; lab: CircuitLab; hint: string }) {
  if (edges.length === 0) return null
  return (
    <div className="ct-edges">
      <div className="pl-control-label" title={hint}>
        {title}
      </div>
      <ul>
        {edges.slice(0, LISTED_EDGES).map((edge) => (
          <li key={edge.node.id}>
            <button type="button" className="pl-link" onClick={() => lab.setSelected(edge.node.id)}>
              {nodeName(edge.node)}
            </button>
            <span title={`Direct effect ${signedLogits(edge.weight)}, in the units of the node it enters`}>{signedPercent(edge.weight < 0 ? -edge.share : edge.share)}</span>
          </li>
        ))}
      </ul>
    </div>
  )
}

/** The prompt's tokens, shaded by a feature's activation on each. */
function Fires({ node, graph }: { node: GraphNode; graph: CircuitGraph }) {
  const colors = useLabColors()
  const fires = node.fires ?? []
  const peak = Math.max(1e-9, ...fires)
  return (
    <div className="ct-fires">
      {graph.tokens.map((token, position) => {
        const value = fires[position] ?? 0
        const strength = value > 0 ? 0.12 + 0.6 * (value / peak) : 0
        return (
          <span
            key={position}
            className={`ct-token${position === node.position ? ' is-here' : ''}`}
            style={strength > 0 ? { backgroundColor: `color-mix(in srgb, ${colors.classes[1]} ${Math.round(strength * 100)}%, transparent)` } : undefined}
            title={`${showToken(token)}: ${value.toFixed(2)}`}
          >
            {showToken(token)}
          </span>
        )
      })}
    </div>
  )
}

function NodeDetail({ lab, graph }: { lab: CircuitLab; graph: CircuitGraph }) {
  const index = lab.selected ? graph.nodes.findIndex((n) => n.id === lab.selected) : -1
  if (index < 0) {
    return <div className="pl-empty">Select a node</div>
  }
  const node = graph.nodes[index]
  const { inputs, outputs } = nodeEdges(graph, index)
  const pinned = lab.pinned.includes(node.id)
  return (
    <div className="ct-detail">
      <div className="ct-detail__head">
        <strong>{nodeTitle(node, graph)}</strong>
        {node.kind === 'feature' && (
          <button type="button" className="oz-chip" aria-pressed={pinned} onClick={() => lab.togglePin(node.id)} title="Pinned nodes are the ones an intervention changes">
            {pinned ? 'Pinned' : 'Pin'}
          </button>
        )}
      </div>
      <div className="ct-detail__stats">
        {node.kind === 'logit' ? (
          <>
            <span>
              <b>{percent(node.prob ?? 0)}</b> probability
            </span>
            <span>
              <b>{logits(graph.target.logit)}</b> logits above the mean
            </span>
          </>
        ) : (
          <span title="The share of the target’s attribution that flows through this node">
            <b>{percent(node.influence)}</b> of the target’s attribution
          </span>
        )}
        {(node.kind === 'feature' || node.kind === 'error') && (
          <span title="Its influence with the signs along its paths kept: positive pushes the target up, negative holds it down">
            <b>{signedPercent(node.support)}</b> {node.support >= 0 ? 'supports' : 'opposes'}
          </span>
        )}
        {node.kind === 'feature' && (
          <span>
            <b>{(node.activation ?? 0).toFixed(2)}</b> activation
          </span>
        )}
      </div>
      {node.kind === 'error' && <div className="pl-hint">What this layer’s transcoder didn’t reconstruct here.</div>}
      {node.kind === 'feature' && (
        <>
          <div>
            <div className="pl-control-label" title="The tokens its decoder writes toward, read through the unembedding">
              Promotes
            </div>
            <div className="ct-chips">
              {(node.promotes ?? []).map((token, i) => (
                <span key={i} className="ct-token">
                  {showToken(token)}
                </span>
              ))}
            </div>
          </div>
          <div>
            <div className="pl-control-label" title="The tokens its decoder writes away from">
              Suppresses
            </div>
            <div className="ct-chips">
              {(node.suppresses ?? []).map((token, i) => (
                <span key={i} className="ct-token ct-token--dim">
                  {showToken(token)}
                </span>
              ))}
            </div>
          </div>
          <div>
            <div className="pl-control-label" title="Its activation at every position of the prompt; this node’s position is outlined">
              Fires on
            </div>
            <Fires node={node} graph={graph} />
          </div>
        </>
      )}
      <div className="ct-detail__edges">
        <EdgeList title="Inputs" edges={inputs} lab={lab} hint="The edges into this node, each as the share of the target’s attribution that flows through it (negative: it lowers this node)" />
        <EdgeList title="Outputs" edges={outputs} lab={lab} hint="The edges out of this node, each as the share of the target’s attribution that flows through it (negative: it lowers the node it enters)" />
      </div>
    </div>
  )
}

function FeatureTable({ lab, graph }: { lab: CircuitLab; graph: CircuitGraph }) {
  const features = lab.features.slice(0, LISTED_FEATURES)
  return (
    <div className="ct-table-scroll">
      <table className="pl-table ct-table">
        <thead>
          <tr>
            <th>Feature</th>
            <th>At</th>
            <th>Promotes</th>
            <th>Activation</th>
            <th title="The share of the target’s attribution that flows through the node">Influence</th>
            <th title="Influence with signs kept: positive pushes the target up, negative holds it down">Support</th>
            <th>Pin</th>
          </tr>
        </thead>
        <tbody>
          {features.map((node) => (
            <tr key={node.id} className={node.id === lab.selected ? 'is-active' : undefined} onClick={() => lab.setSelected(node.id)}>
              <td>{nodeName(node)}</td>
              <td className="ct-mono">{showToken(graph.tokens[node.position])}</td>
              <td className="ct-mono ct-promotes">{(node.promotes ?? []).slice(0, 3).map(showToken).join(' ')}</td>
              <td>{(node.activation ?? 0).toFixed(2)}</td>
              <td>{percent(node.influence)}</td>
              <td>{signedPercent(node.support)}</td>
              <td>
                <input
                  type="checkbox"
                  aria-label={`Pin ${nodeName(node)}`}
                  checked={lab.pinned.includes(node.id)}
                  onClick={(e) => e.stopPropagation()}
                  onChange={() => lab.togglePin(node.id)}
                />
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  )
}

export function GraphView({ lab }: { lab: CircuitLab }) {
  const graph = lab.graph
  if (!graph) {
    return (
      <section className="pl-panel">
        {lab.building || lab.loadingGraph ? (
          <div className="oz-empty">
            <Spinner large label="Building the graph" />
            <p>{lab.loadingGraph ? 'Loading the graph' : lab.building?.progress?.stage ?? 'Starting a GPU'}</p>
          </div>
        ) : (
          <Placeholder glyph="circuits">Trace a prompt to see how the model gets to its next word.</Placeholder>
        )}
      </section>
    )
  }

  const shown = shownNodes(graph, lab.shown, [...lab.pinned, ...(lab.selected ? [lab.selected] : [])], lab.showErrors)
  const job = lab.jobs.find((j) => j.job_id === graph.job_id)
  const inputs = graph.scores.target_inputs.absolute
  const inner = graph.nodes.filter((n) => n.kind === 'feature' || n.kind === 'error').length

  return (
    <>
      <section className="pl-panel">
        <div className="pl-run-meta">
          <strong>{graph.model}</strong> · {graph.transcoders.name} · {graph.tokens.length} tokens{graph.chat_template ? ' (chat template)' : ''}
          {job ? ` · ${money(job.cost)}` : ''}
        </div>
        <StatTiles
          tiles={[
            { label: 'Target', value: showToken(graph.target.token), hint: `${percent(graph.target.prob)} · ${logits(graph.target.logit)} logits`, title: 'The token traced, with its probability and its logit above the mean logit' },
            { label: 'Replacement', value: percent(graph.scores.replacement), hint: 'through features', title: 'Of the attribution that reaches the target from tokens and error nodes, the share that starts at tokens' },
            { label: 'Completeness', value: percent(graph.scores.completeness), hint: 'inputs not from error', title: 'Over all nodes, weighted by influence, the share of a node’s inputs that doesn’t come from error nodes' },
            { label: 'Target’s inputs', value: percent(inputs.feature), hint: `features · ${percent(inputs.error)} error`, title: 'The target logit’s direct inputs by kind, as shares of their absolute sum' },
            { label: 'Nodes', value: String(inner), hint: `of ${graph.counts.traced_features.toLocaleString()} traced`, title: 'Feature and error nodes kept after pruning' },
            { label: 'Edges', value: graph.edges.source.length.toLocaleString(), hint: `${graph.seconds.toFixed(1)}s`, title: 'Edges kept after pruning, and how long tracing took' },
          ]}
        />
        <Findings items={findings(graph)} />
      </section>

      <section className="pl-panel">
        <div className="pl-card-title">
          <h3>Graph</h3>
          <div className="ct-graph-controls">
            <div>
              <Segmented
                ariaLabel="Nodes shown"
                value={String(lab.shown)}
                onChange={(value) => lab.setShown(Number(value))}
                options={SHOWN_OPTIONS.map((count) => ({ value: String(count), label: count === 0 ? 'All' : `Top ${count}`, title: count === 0 ? 'Every node kept after pruning' : `The ${count} most influential nodes` }))}
              />
            </div>
            <button type="button" className="oz-chip" aria-pressed={lab.showErrors} onClick={() => lab.setShowErrors(!lab.showErrors)} title="Show the error nodes">
              Errors
            </button>
            <FigureExport
              filename={`circuit-${graph.model}`}
              figure={() => {
                const indices = [...shown]
                const order = new Map(indices.map((index, i) => [index, i]))
                const { source, target, weight, share } = graph.edges
                const edges: number[][] = []
                for (let i = 0; i < source.length && edges.length < 400; i++) {
                  const from = order.get(source[i])
                  const to = order.get(target[i])
                  if (from != null && to != null) edges.push([from, to, weight[i], share[i]])
                }
                return {
                  kind: 'graph',
                  title: `${graph.prompt} → ${showToken(graph.target.token)} (${graph.model})`,
                  graph: {
                    tokens: graph.tokens.map(showToken),
                    num_layers: graph.num_layers,
                    target: showToken(graph.target.token),
                    nodes: indices.map((index) => {
                      const node = graph.nodes[index]
                      return { kind: node.kind, layer: node.layer ?? null, position: node.position, influence: node.influence, support: node.support, pinned: lab.pinned.includes(node.id) }
                    }),
                    edges,
                  },
                }
              }}
            />
          </div>
        </div>
        <CircuitGraphChart graph={graph} shown={shown} selected={lab.selected} pinned={lab.pinned} onSelect={lab.setSelected} />
      </section>

      <div className="pl-results-grid">
        <section className="pl-panel" data-assist="circuits.node">
          <div className="pl-card-title">
            <h3>Node</h3>
          </div>
          <NodeDetail lab={lab} graph={graph} />
        </section>
        <section className="pl-panel">
          <div className="pl-card-title">
            <h3>Top features</h3>
            <span className="pl-note">{lab.pinned.length > 0 ? `${lab.pinned.length} pinned` : ''}</span>
          </div>
          <FeatureTable lab={lab} graph={graph} />
        </section>
      </div>
    </>
  )
}
