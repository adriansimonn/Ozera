"""
The assistant's tools: one fixed list, sorted by name, the same on every page and for every
user (it renders at the start of every prompt, so a stable list keeps the shared cache).

Three kinds:
- read: answered on the backend at once, as the calling user (services/assistant_tools.py).
  Costs nothing beyond tokens.
- ui: checked against the page's controls (core.assistant.pages), then applied by the browser.
  A ui tool only changes what's on screen; it never runs anything charged.
- run: charged by its page's own service, and only after the user approves the run and its
  price (services/assistant_runs.py prices and runs them). The loop stops at a reply with run
  calls and the panel shows an approval card.

Every schema has all parameters required and no unions: a parameter that doesn't apply takes ""
or 0, as its description says. The loop checks every call's input against its schema
(check_input) and answers a bad one with what's wrong, so Claude can call again. Only the tools
with the most structured inputs are strict, because the API compiles all strict schemas of a
request into one grammar and refuses one that's too large ("The compiled grammar is too large"),
well before its documented limits of 20 strict tools, 24 optional and 16 union-typed parameters.
Measured 2026-11-13 on Haiku, Sonnet and Opus 5.5: all 20 tools strict failed, the 12 run tools
plus navigate were the most that compiled, and these five plus six more compiled. After changing
a strict tool, check with scripts/assistant_pilot.py --check-tools.
"""

from dataclasses import dataclass, field
from typing import Callable, Optional

from .pages import INTERVENTIONS, PAGE_IDS, PAGES, PATCH_TYPES

READ = "read"
UI = "ui"
RUN = "run"

# A guard well under what compiled (see above); the grammar's size, not the count, is the limit
MAX_STRICT_TOOLS = 8

ESTIMATE_ACTIONS = (
    "analyze_sae",
    "autointerp_job",
    "build_atlas",
    "classify_heads",
    "compare_probe_sae",
    "explain_feature",
    "generate",
    "generation_test",
    "run_geometry",
    "run_patching",
    "test_hypothesis",
    "train_probe",
)


@dataclass(frozen=True)
class Tool:
    name: str
    kind: str
    description: str
    properties: dict = field(default_factory=dict)
    # A short label for the tool chip under the reply, from the call's arguments (executors may
    # give a better one, e.g. with an SAE's name)
    chip: Callable[[dict], str] = lambda args: ""
    # Whether the API enforces the schema (see the module docstring); check_input checks every tool
    strict: bool = False

    @property
    def input_schema(self) -> dict:
        return {
            "type": "object",
            "properties": self.properties,
            "required": list(self.properties),
            "additionalProperties": False,
        }

    def definition(self) -> dict:
        out = {"name": self.name, "description": self.description, "input_schema": self.input_schema}
        if self.strict:
            out["strict"] = True
        return out


def _string(description: str) -> dict:
    return {"type": "string", "description": description}


def _integer(description: str) -> dict:
    return {"type": "integer", "description": description}


def _number(description: str) -> dict:
    return {"type": "number", "description": description}


def _boolean(description: str) -> dict:
    return {"type": "boolean", "description": description}


def _enum(values, description: str = "") -> dict:
    out: dict = {"type": "string", "enum": list(values)}
    if description:
        out["description"] = description
    return out


_TYPES = {"string": str, "integer": int, "number": (int, float), "boolean": bool, "array": list, "object": dict}
_NAMES = {"string": "text", "integer": "a whole number", "number": "a number", "boolean": "true or false", "array": "a list", "object": "an object"}


def _problem(schema: dict, value, path: str) -> Optional[str]:
    kind = schema.get("type")
    where = path or "the input"
    if kind in _TYPES:
        # JSON's true and false aren't numbers (Python's bool is an int)
        if not isinstance(value, _TYPES[kind]) or (isinstance(value, bool) and kind in ("integer", "number")):
            return f"{where} must be {_NAMES[kind]}"
    if "enum" in schema and value not in schema["enum"]:
        return f"{where} must be one of: {', '.join(map(str, schema['enum']))}"
    if kind == "object":
        properties = schema.get("properties", {})
        missing = [k for k in schema.get("required", []) if k not in value]
        if missing:
            return f"{where} is missing {', '.join(missing)}"
        unknown = [k for k in value if k not in properties]
        if unknown and schema.get("additionalProperties") is False:
            return f"{where} has unknown {', '.join(unknown)}"
        for key, sub in properties.items():
            if key in value:
                problem = _problem(sub, value[key], f"{path}.{key}" if path else key)
                if problem:
                    return problem
    if kind == "array" and "items" in schema:
        for i, item in enumerate(value):
            problem = _problem(schema["items"], item, f"{path}[{i}]")
            if problem:
                return problem
    return None


def check_input(spec: "Tool", args) -> Optional[str]:
    """What's wrong with a call's input against the tool's schema (the first problem), or None."""
    return _problem(spec.input_schema, args, "")


MODEL = _string("A model ID, as list_models gives it (e.g. nano, gemma-3-270m)")
SAE = _string("An SAE key, as list_saes gives it (ozera:<nano|mini>:<layer>, or external:<sae id>)")
FEATURE = _integer("The feature's index")
BRIDGED = tuple(p for p in PAGE_IDS if PAGES[p].bridged)

GEOMETRY_READS = ("value_last", "value_mean", "template_last")
ATLAS_TOKENS = (1_000_000, 2_000_000, 5_000_000)
RUN_NOTE = " Charged: the user sees its price and approves it before it runs."


def _sae_chip(args: dict) -> str:
    return args.get("sae", "")


TOOLS: tuple[Tool, ...] = tuple(sorted((
    Tool(
        "estimate", READ,
        "The price of an action, from the same estimate its page shows (nothing runs). Arguments that "
        "don't apply to the action take \"\" or 0. generate: model, text (prompt), count (max tokens). "
        "analyze_sae: sae, text. explain_feature, test_hypothesis: model, sae. generation_test: model, sae. "
        "autointerp_job: model, sae, count (features, most frequent first). build_atlas: model, sae, count "
        "(corpus tokens). train_probe: model, item (built-in dataset ID). run_geometry: model, item "
        "(built-in concept ID). run_patching: model, text (the longer prompt), count (max tokens). "
        "classify_heads: model, text (the prompt). compare_probe_sae: sae, count (the saved probe's ID). "
        "A run tool's approval card shows the same price, so "
        "there's no need to estimate before calling one.",
        {
            "action": {"type": "string", "enum": list(ESTIMATE_ACTIONS)},
            "model": _string("A model ID, or \"\""),
            "sae": _string("An SAE key, or \"\""),
            "text": _string("The text or prompt, or \"\""),
            "count": _integer("Tokens or features, or 0"),
            "item": _string("A built-in dataset or concept ID, or \"\""),
        },
        lambda a: f"Estimate · {a.get('action', '')}",
    ),
    Tool(
        "get_activation", READ,
        "A stored generation of the user's (the Generate page's activation ID): its model, prompt, tokens, "
        "and the model's top predictions at each position.",
        {"activation_id": _string("The activation ID")},
        lambda a: "Stored generation",
    ),
    Tool(
        "get_atlas", READ,
        "An SAE's atlas on a model (Geometry Lab): what it read, the SAE's fit (L0, unexplained variance), "
        "how many features are alive and explained, and its best-ranked communities with their names, sizes, "
        "top tokens and geometry scores against random groups.",
        {"model": MODEL, "sae": SAE},
        lambda a: f"Atlas · {_sae_chip(a)}",
    ),
    Tool(
        "get_balance", READ,
        "The user's available credit balance, and the free assistant messages left today.",
        {},
        lambda a: "Balance",
    ),
    Tool(
        "get_community", READ,
        "One atlas community: its features with their explanations and statistics, the tokens it fires on, "
        "a few contexts, and its geometry against random groups.",
        {"model": MODEL, "sae": SAE, "community": _integer("The community's ID")},
        lambda a: f"Community {a.get('community', '')} · {_sae_chip(a)}",
    ),
    Tool(
        "get_feature", READ,
        "One SAE feature: its newest explanation and earlier versions, with scores (± and their percentile "
        "among random-direction controls, and the token baseline), its top firing tokens, the tokens its "
        "decoder row promotes and suppresses, how often and how strongly it fires, short contexts, and the "
        "user's own hypotheses and generation tests of it.",
        {"model": MODEL, "sae": SAE, "feature": _integer("The feature's index")},
        lambda a: f"Feature {a.get('feature', '')} · {_sae_chip(a)}",
    ),
    Tool(
        "get_probe", READ,
        "One of the user's saved probes: what it was trained on, its layer, pooling and method, and its "
        "metrics with their baselines (control task, embeddings, majority class).",
        {"probe_id": _integer("The saved probe's ID")},
        lambda a: f"Probe {a.get('probe_id', '')}",
    ),
    Tool(
        "highlight", UI,
        "Scroll to a control on a page and make it pulse, to show the user where it is (e.g. the button "
        "that runs something). The page must be open or opened by navigate first.",
        {
            "page": {"type": "string", "enum": list(BRIDGED)},
            "control": _string("A control ID of that page (see the guide)"),
        },
        lambda a: f"Pointed at {a.get('control', '')}",
    ),
    Tool(
        "list_jobs", READ,
        "The user's recent background jobs: custom model training, atlases and autointerp, with their "
        "status and progress.",
        {},
        lambda a: "Jobs",
    ),
    Tool(
        "list_models", READ,
        "Every model the user can read: Ozera's nano and mini, open-source models and the user's custom "
        "models, with their layers, width, parameters and GPU tier.",
        {},
        lambda a: "Models",
    ),
    Tool(
        "list_probes", READ,
        "The user's saved probes, newest first, with their test AUROC and its control-task baseline.",
        {},
        lambda a: "Saved probes",
    ),
    Tool(
        "list_saes", READ,
        "SAEs that read a model's residual stream (Ozera's for nano and mini, and those the user loaded), "
        "each with its key, layer, width, and its atlas if any: density (L0 as a share of width), whether "
        "its features can be explained, and how many are.",
        {"model": MODEL},
        lambda a: f"SAEs · {a.get('model', '')}",
    ),
    Tool(
        "navigate", UI,
        "Open one of Ozera's pages in the user's browser (nothing runs). To fill a page's inputs, call "
        "set_inputs after it.",
        {"page": {"type": "string", "enum": list(PAGE_IDS)}},
        lambda a: f"Opened {PAGES[a['page']].title if a.get('page') in PAGES else a.get('page', '')}",
    ),
    Tool(
        "search_features", READ,
        "Search an SAE's feature explanations (and its atlas's community names) for words, on its atlas "
        "on a model. Returns matching features with their explanations and scores.",
        {"model": MODEL, "sae": SAE, "query": _string("Words to search for")},
        lambda a: f"Search “{a.get('query', '')}” · {_sae_chip(a)}",
    ),
    Tool(
        "set_inputs", UI,
        "Fill controls on a page (the guide lists each page's controls), without running anything: the "
        "user presses the page's button to run it. Values are text: numbers as digits, toggles as true or "
        "false, SAEs as their keys. A page that isn't open is filled once it opens.",
        {
            "page": {"type": "string", "enum": list(BRIDGED)},
            "fields": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {"control": _string("A control ID"), "value": _string("Its value")},
                    "required": ["control", "value"],
                    "additionalProperties": False,
                },
            },
        },
        lambda a: f"Filled {', '.join(f.get('control', '') for f in a.get('fields', [])[:4])}",
        strict=True,
    ),
    # Run tools (charged; each stops the turn for the user's approval)
    Tool(
        "analyze_sae", RUN,
        "Run a text through an SAE, as the SAE page's Analyze button does: the features active on each "
        "token, with the SAE's L0 and width. The full result opens in the SAE page's Analyze view." + RUN_NOTE,
        {"sae": SAE, "text": _string("The text to analyze")},
        lambda a: f"Analyze · {_sae_chip(a)}",
    ),
    Tool(
        "build_atlas", RUN,
        "Build an SAE's atlas on a model (the Geometry Lab's atlas view): a background GPU job reads "
        "OpenWebText through the SAE for feature statistics, contexts and communities, and replaces any "
        "atlas of it on that corpus. list_jobs follows it." + RUN_NOTE,
        {"model": MODEL, "sae": SAE, "tokens": _integer("Corpus tokens to read: 1000000, 2000000 or 5000000")},
        lambda a: f"Atlas job · {_sae_chip(a)}",
    ),
    Tool(
        "classify_heads", RUN,
        "Attention Analysis: capture a prompt's activations on a model, then classify every attention "
        "head (induction, previous-token, positional, copying), as the Attention Analysis page does. "
        "activation_id reuses one of the page's captures instead (give \"\" to capture the prompt; the "
        "page state lists them). The full result opens on that page." + RUN_NOTE,
        {"model": MODEL, "prompt": _string("The prompt to capture, or \"\" with an activation_id"),
         "activation_id": _string("A capture's ID from the Attention Analysis page, or \"\"")},
        lambda a: f"Classify heads · {a.get('model', '')}",
    ),
    Tool(
        "compare_probe_sae", RUN,
        "Compare one of the user's saved probes with an SAE at the probe's layer, as the Probe Lab's SAE "
        "view does, on the probe's own built-in dataset and split: the features whose decoder directions "
        "are nearest the probe's (against random and shuffled-label directions), the features its score "
        "reads on this data, and sparse probes on the top k features next to a dense probe." + RUN_NOTE,
        {"probe_id": _integer("The saved probe's ID (list_probes gives them)"), "sae": SAE},
        lambda a: f"Probe {a.get('probe_id', '')} vs {_sae_chip(a)}",
    ),
    Tool(
        "explain_feature", RUN,
        "Explain one feature with autointerp, as the SAE page's Explain button does: Claude writes an "
        "explanation from its atlas contexts, and a scorer tests it on held-out contexts against the token "
        "baseline. A feature already explained with the default models comes back as stored, free. The "
        "explanation is shared with everyone who has the SAE." + RUN_NOTE,
        {"model": MODEL, "sae": SAE, "feature": FEATURE},
        lambda a: f"Explain {a.get('feature', '')} · {_sae_chip(a)}",
    ),
    Tool(
        "generate", RUN,
        "Generate text from a prompt with activations captured, as the Generate page's visualize mode "
        "does. It opens on the Generate page; get_activation then reads each position's top predictions." + RUN_NOTE,
        {
            "model": MODEL,
            "prompt": _string("The prompt"),
            "max_tokens": _integer("Tokens to generate, 1 to 300"),
            "temperature": _number("0 to 2 (0 is greedy)"),
            "top_k": _integer("1 to 100, or 0 for the page's default (40)"),
            "until_eos": _boolean("Stop at the model's end-of-sequence token, with max_tokens as the cap"),
        },
        lambda a: f"Generate · {a.get('model', '')}",
        strict=True,
    ),
    Tool(
        "generation_test", RUN,
        "Test whether an explanation is specific enough to write text: Claude writes 10 texts it says fire "
        "the feature and 5 near-misses, and the model reads them. hypothesis_id 0 tests the feature's newest "
        "explanation; otherwise one of the user's scored hypotheses (get_feature gives their IDs). A test "
        "already stored comes back free." + RUN_NOTE,
        {"model": MODEL, "sae": SAE, "feature": FEATURE,
         "hypothesis_id": _integer("A scored hypothesis's ID, or 0 for the newest explanation")},
        lambda a: f"Generation test {a.get('feature', '')} · {_sae_chip(a)}",
    ),
    Tool(
        "run_geometry", RUN,
        "Run the Geometry Lab on a built-in concept: class means at every layer and the shape tests "
        "against shuffled baselines. The full result opens in the Geometry Lab." + RUN_NOTE,
        {
            "model": MODEL,
            "concept": _string("A built-in concept ID (see the guide)"),
            "read": _enum(GEOMETRY_READS, "Where each example is read"),
            "chat_template": _boolean("Read examples in an instruct model's chat template (instruct models only)"),
            "seed": _integer("The run's seed (0 by default)"),
        },
        lambda a: f"Geometry · {a.get('concept', '')} · {a.get('model', '')}",
        strict=True,
    ),
    Tool(
        "run_patching", RUN,
        "Activation Patching: generate from the target prompt with and without the patches, and compare "
        "the outputs. A patch intervention copies the source prompt's activations at that layer and patch "
        "point; the ablations (zero_ablate, mean_ablate, noise_ablate) need no source prompt (give \"\"). "
        "Empty positions or heads mean all; blend_factor 1 replaces fully. The full result opens on the "
        "Activation Patching page." + RUN_NOTE,
        {
            "model": MODEL,
            "source_prompt": _string("The prompt activations are copied from, or \"\" for ablations only"),
            "target_prompt": _string("The prompt generation runs on"),
            "patches": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "layer": _integer("Layer index (0-based)"),
                        "patch_type": _enum(PATCH_TYPES),
                        "intervention_type": _enum(INTERVENTIONS),
                        "positions": {"type": "array", "items": {"type": "integer"}, "description": "Token positions, or [] for all"},
                        "heads": {"type": "array", "items": {"type": "integer"}, "description": "Attention heads (attention patches), or [] for all"},
                        "blend_factor": _number("0 to 1"),
                    },
                    "required": ["layer", "patch_type", "intervention_type", "positions", "heads", "blend_factor"],
                    "additionalProperties": False,
                },
            },
            "max_tokens": _integer("Tokens to generate, 1 to 100"),
            "temperature": _number("0 to 2 (0 for a deterministic comparison)"),
        },
        lambda a: f"Patching · {a.get('model', '')}",
        strict=True,
    ),
    Tool(
        "start_autointerp_job", RUN,
        "Explain and score an atlas's most frequent features in the background (the SAE page's Autointerp "
        "view), through the Batch API, with its random-direction controls; features already explained are "
        "skipped. It holds its upper bound and is charged its usage when it ends. list_jobs follows it." + RUN_NOTE,
        {"model": MODEL, "sae": SAE, "features": _integer("How many of the most frequent features")},
        lambda a: f"Autointerp job · {_sae_chip(a)}",
    ),
    Tool(
        "test_hypothesis", RUN,
        "Score a hypothesis about a feature the way explanations are scored (detection and fuzzing on the "
        "held-out set, next to the token baseline). It's kept private to the user; the same text scored "
        "again comes back free. Write it as a short explanation of what the feature fires on." + RUN_NOTE,
        {"model": MODEL, "sae": SAE, "feature": FEATURE, "hypothesis": _string("The hypothesis, under 200 characters")},
        lambda a: f"Hypothesis {a.get('feature', '')} · {_sae_chip(a)}",
    ),
    Tool(
        "train_probe", RUN,
        "Train the Probe Lab's layer sweep on a built-in dataset: probes at every layer (logistic regression "
        "and difference in means, three poolings) with their control-task and embedding baselines. The full "
        "result opens in the Probe Lab." + RUN_NOTE,
        {
            "model": MODEL,
            "dataset": _string("A built-in dataset ID (see the guide)"),
            "chat_template": _boolean("Read texts in an instruct model's chat template (instruct models only)"),
            "test_fraction": _number("The held-out share: 0.1, 0.2, 0.3 or 0.4"),
            "seed": _integer("The split's seed (0 by default)"),
        },
        lambda a: f"Probes · {a.get('dataset', '')} · {a.get('model', '')}",
        strict=True,
    ),
), key=lambda t: t.name))

BY_NAME = {t.name: t for t in TOOLS}

RUN_TOOLS = tuple(t.name for t in TOOLS if t.kind == RUN)

if sum(t.strict for t in TOOLS) > MAX_STRICT_TOOLS:
    raise RuntimeError(f"{sum(t.strict for t in TOOLS)} strict tools; a request may have {MAX_STRICT_TOOLS}")


def definitions() -> list[dict]:
    """The tools as the Messages API takes them, in their fixed order."""
    return [t.definition() for t in TOOLS]


def tool(name: str) -> Optional[Tool]:
    return BY_NAME.get(name)
