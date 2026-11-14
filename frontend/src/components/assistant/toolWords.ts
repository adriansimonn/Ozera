/** A tool call in plain words, for its row in the chat (ToolStep). */

import type { ToolChipInfo } from '../../types/assistant'

/**
 * What each tool does, as [to do, doing, done], and the start of its chip (backend/core/assistant/
 * tools.py and the run and tool services) that the words already say: what's left of the chip
 * follows them. A chip that doesn't start that way follows them whole.
 */
const PHRASES: Record<string, { says: [string, string, string]; prefix: RegExp }> = {
  estimate: { says: ['Estimate the price of', 'Estimating the price of', 'Estimated the price of'], prefix: /^Estimate · / },
  get_activation: { says: ['Read the stored generation', 'Reading the stored generation', 'Read the stored generation'], prefix: /^Stored generation$/ },
  get_atlas: { says: ['Read the atlas of', 'Reading the atlas of', 'Read the atlas of'], prefix: /^Atlas · / },
  get_balance: { says: ['Check your balance', 'Checking your balance', 'Checked your balance'], prefix: /^Balance$/ },
  get_community: { says: ['Read community', 'Reading community', 'Read community'], prefix: /^Community / },
  get_feature: { says: ['Read feature', 'Reading feature', 'Read feature'], prefix: /^Feature / },
  get_probe: { says: ['Read probe', 'Reading probe', 'Read probe'], prefix: /^Probe (· )?/ },
  list_jobs: { says: ['Look up your jobs', 'Looking up your jobs', 'Looked up your jobs'], prefix: /^Jobs$/ },
  list_models: { says: ['Look up the models', 'Looking up the models', 'Looked up the models'], prefix: /^Models$/ },
  list_probes: { says: ['Look up your saved probes', 'Looking up your saved probes', 'Looked up your saved probes'], prefix: /^Saved probes$/ },
  list_saes: { says: ['Look up the SAEs for', 'Looking up the SAEs for', 'Looked up the SAEs for'], prefix: /^SAEs · / },
  search_features: { says: ['Search features for', 'Searching features for', 'Searched features for'], prefix: /^Search / },
  highlight: { says: ['Point to', 'Pointing to', 'Pointed to'], prefix: /^Pointed at / },
  navigate: { says: ['Open', 'Opening', 'Opened'], prefix: /^Opened / },
  set_inputs: { says: ['Fill in', 'Filling in', 'Filled in'], prefix: /^Filled / },
  analyze_sae: { says: ['Analyze the text with', 'Analyzing the text with', 'Analyzed the text with'], prefix: /^Analy[sz]ed? · / },
  build_atlas: { says: ['Start an atlas build for', 'Starting an atlas build for', 'Started an atlas build for'], prefix: /^Atlas job( started)? · / },
  classify_heads: { says: ['Classify the attention heads of', 'Classifying the attention heads of', 'Classified the attention heads of'], prefix: /^(Classify heads|Heads classified) · / },
  compare_probe_sae: { says: ['Compare the probe with', 'Comparing the probe with', 'Compared the probe with'], prefix: /^Probe (\S+ )?vs / },
  explain_feature: { says: ['Explain feature', 'Explaining feature', 'Explained feature'], prefix: /^Explain(ed)? / },
  generate: { says: ['Generate text with', 'Generating text with', 'Generated text with'], prefix: /^Generated? · / },
  generation_test: { says: ['Run a generation test on feature', 'Running a generation test on feature', 'Ran a generation test on feature'], prefix: /^Generation test (· )?/ },
  run_geometry: { says: ['Test the geometry of', 'Testing the geometry of', 'Tested the geometry of'], prefix: /^Geometry · / },
  run_patching: { says: ['Run activation patching on', 'Running activation patching on', 'Ran activation patching on'], prefix: /^Patch(ing|ed) · / },
  start_autointerp_job: { says: ['Start an autointerp job for', 'Starting an autointerp job for', 'Started an autointerp job for'], prefix: /^(Autointerp job|Job started) · / },
  test_hypothesis: { says: ['Test a hypothesis on feature', 'Testing a hypothesis on feature', 'Tested a hypothesis on feature'], prefix: /^Hypothesis (scored · )?/ },
  train_probe: { says: ['Train probes on', 'Training probes on', 'Trained probes on'], prefix: /^Probes( trained)? · / },
  save_probe: { says: ['Save probe', 'Saving probe', 'Saved probe'], prefix: /^Saved? probe · / },
  score_probe: { says: ['Score text with', 'Scoring text with', 'Scored text with'], prefix: /^Scored? · / },
  steer_probe: { says: ['Steer with', 'Steering with', 'Steered with'], prefix: /^Steer(ed)? · / },
  ablate_probe: { says: ['Ablate the direction of', 'Ablating the direction of', 'Ablated the direction of'], prefix: /^Ablat(e|ed) · / },
  generalize_probes: { says: ['Test probe generalization on', 'Testing probe generalization on', 'Tested probe generalization on'], prefix: /^Generalization · / },
  compare_geometry: { says: ['Compare geometry runs', 'Comparing geometry runs', 'Compared geometry runs'], prefix: /^Geometry · / },
  steer_geometry: { says: ['Steer the concept from', 'Steering the concept from', 'Steered the concept from'], prefix: /^Steer(ed)? · / },
  geometry_sae: { says: ['Read the concept through', 'Reading the concept through', 'Read the concept through'], prefix: /^Geometry SAE · / },
  compare_attention: { says: ['Compare attention on', 'Comparing attention on', 'Compared attention on'], prefix: /^Compared? attention · / },
  mine_patterns: { says: ['Mine attention patterns on', 'Mining attention patterns on', 'Mined attention patterns on'], prefix: /^Mined? patterns · / },
}

// What an estimate was for (its chip names the action)
const ACTIONS: Record<string, string> = {
  generate: 'a generation', analyze_sae: 'an SAE analysis', explain_feature: 'a feature explanation',
  test_hypothesis: 'a hypothesis test', generation_test: 'a generation test', start_autointerp_job: 'an autointerp job',
  build_atlas: 'an atlas', train_probe: 'probe training', run_geometry: 'a geometry run', run_patching: 'a patching run',
  classify_heads: 'head classification', compare_probe_sae: 'a probe and SAE comparison', save_probe: 'saving a probe',
  score_probe: 'probe scoring', steer_probe: 'probe steering', ablate_probe: 'a probe ablation',
  generalize_probes: 'a generalization matrix', steer_geometry: 'concept steering', geometry_sae: 'a concept SAE read',
  compare_attention: 'an attention comparison', mine_patterns: 'pattern mining',
}

/** A tool call in words, as [what it does, what on]: "to do" until it runs, then "doing", then "done". */
export function describeTool(tool: ToolChipInfo): [string, string] {
  const phrase = PHRASES[tool.name]
  const chip = (tool.chip || '').trim()
  if (!phrase) {
    const words = tool.name.replace(/_/g, ' ')
    return [words.charAt(0).toUpperCase() + words.slice(1), chip]
  }
  const form = tool.status === 'running' ? 1 : tool.status === 'done' || tool.status === undefined ? 2 : 0
  let detail = phrase.prefix.test(chip) ? chip.replace(phrase.prefix, '').trim() : chip
  if (tool.name === 'estimate') detail = ACTIONS[detail.replace(/ /g, '_')] ?? detail
  // A phrase that ends in a joining word drops it when there's nothing to join
  const says = detail ? phrase.says[form] : phrase.says[form].replace(/ (of|for|with|on|to|in|through|from)$/, '')
  return [says, detail]
}
