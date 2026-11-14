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
plus navigate were the most that compiled, and five (these three, run_geometry and train_probe)
plus six more compiled. run_geometry and train_probe stopped being strict when they took custom
concepts and rows. After changing a strict tool, check with scripts/assistant_pilot.py --check-tools.
"""

from dataclasses import dataclass, field
from typing import Callable, Optional

from .pages import INTERVENTIONS, PAGE_IDS, PAGES, PATCH_TYPES

READ = "read"
UI = "ui"
RUN = "run"

# A guard well under what compiled (see above); the grammar's size, not the count, is the limit
MAX_STRICT_TOOLS = 8

# Runs estimate prices: every run tool (a test checks it's RUN_TOOLS)
ESTIMATE_ACTIONS = (
    "ablate_probe",
    "analyze_sae",
    "build_atlas",
    "classify_heads",
    "compare_attention",
    "compare_probe_sae",
    "explain_feature",
    "generalize_probes",
    "generate",
    "generation_test",
    "geometry_sae",
    "mine_patterns",
    "run_geometry",
    "run_patching",
    "save_probe",
    "score_probe",
    "start_autointerp_job",
    "steer_geometry",
    "steer_probe",
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
PROBE_ID = _integer("A saved probe's ID (list_probes gives them; save_probe saves one from a sweep)")
BRIDGED = tuple(p for p in PAGE_IDS if PAGES[p].bridged)

GEOMETRY_READS = ("value_last", "value_mean", "template_last")
STRUCTURES = ("cyclic", "ordinal", "categorical")
ATLAS_TOKENS = (1_000_000, 2_000_000, 5_000_000)
# core.geometry.atlas.CORPORA (a test checks they match)
ATLAS_CORPORA = ("openwebtext", "wikitext", "tinystories")
POOLINGS = ("mean", "last", "max")
METHODS = ("logreg", "diff_means")
RUN_NOTE = " Charged: the user sees its price and approves it before it runs."


def _strings(description: str) -> dict:
    return {"type": "array", "items": {"type": "string"}, "description": description}


# A labeled dataset of the user's own, given inline (the Probe Lab's upload)
ROWS = {
    "type": "array",
    "description": "Your own labeled examples instead of a built-in dataset ([] for a built-in one): at least 10 of each "
                   "label, at most 2000, each text under 2000 characters",
    "items": {
        "type": "object",
        "properties": {"text": _string("The example"), "label": _integer("0 or 1 (label_names[0] or [1])")},
        "required": ["text", "label"],
        "additionalProperties": False,
    },
}
LABEL_NAMES = _strings("With rows: the two labels' names, label 0's first (e.g. [\"false\", \"true\"]); [] otherwise")
DATASET_NAME = _string("With rows: the dataset's name; \"\" otherwise")

# A concept of the user's own (the Geometry Lab's custom concept)
CUSTOM_CONCEPT = {
    "type": "object",
    "description": "With concept \"\": a concept of your own. Otherwise give name \"\" and empty lists",
    "properties": {
        "name": _string("Its name"),
        "structure": _enum(STRUCTURES, "cyclic (wraps around, e.g. seasons), ordinal (ordered, e.g. sizes) or categorical (no order)"),
        "values": _strings("5 to 100 values, in order for cyclic and ordinal concepts; one line each"),
        "groups": _strings("Each value's group, for categorical concepts the shape tests compare groups of (e.g. each animal's "
                           "class); [] for none"),
        "templates": _strings("4 to 30 sentences, each with {x} once where a value goes; values × templates at most 2000"),
        "eval_templates": _strings("Up to 4 prompts for steering, each with {x} once and ending where the model would say "
                                   "the next value (e.g. \"After {x} comes\"); [] for none"),
    },
    "required": ["name", "structure", "values", "groups", "templates", "eval_templates"],
    "additionalProperties": False,
}


def _sae_chip(args: dict) -> str:
    return args.get("sae", "")


def _concept_chip(args: dict) -> str:
    custom = args.get("custom") if isinstance(args.get("custom"), dict) else {}
    return args.get("concept") or custom.get("name") or "custom concept"


TOOLS: tuple[Tool, ...] = tuple(sorted((
    Tool(
        "estimate", READ,
        "The price of a run, from the same quote its approval card shows (nothing runs): action is the run "
        "tool's name, input its input as a JSON object in text, exactly as you'd call it. Use it to price a "
        "run the user will start from the page; there's no need to estimate before calling a run tool.",
        {
            "action": {"type": "string", "enum": list(ESTIMATE_ACTIONS)},
            "input": _string("The run tool's input, as JSON (e.g. {\"model\": \"nano\", \"prompt\": \"Hi\", ...})"),
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
        "compare_geometry", READ,
        "Compare two Geometry Lab runs of the same values (free, computed from the kept runs): run_id's class "
        "means at one layer aligned onto other_run_id's at every layer (another layer, model, or the "
        "base/instruct sibling; the same run_id compares its layers), as aligned R² and linear CKA next to "
        "shuffled value pairings. Run IDs come from run_geometry's results and the Geometry Lab's page state. "
        "Opens in the Geometry Lab's compare view.",
        {
            "run_id": _string("The run whose layer is compared"),
            "layer": _integer("Its residual stream position (0 the embeddings, i layer i-1's output)"),
            "other_run_id": _string("The run compared at every layer (the same run_id for its own layers)"),
            "dims": _integer("Top components compared: 2, 3, 4, 6 or 8, or 0 for all"),
        },
        lambda a: f"Geometry · {a.get('run_id', '')} vs {a.get('other_run_id', '')}",
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
        "false, SAEs as their keys, lists one item per line, rows as a JSON array. A page that isn't open is "
        "filled once it opens; a view's controls fill when it shows.",
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
        "ablate_probe", RUN,
        "Probe Lab ablate view: project a saved probe's direction out of the residual stream everywhere (and "
        "control directions removing as much), then measure how far next-token predictions move (KL) and "
        "test AUROC at every layer of probes on the clean and ablated model, refit or not. On the probe's "
        "own built-in dataset and split unless dataset or rows are given." + RUN_NOTE,
        {
            "probe_id": PROBE_ID,
            "dataset": _string("A built-in dataset ID, or \"\" for the probe's own (or with rows)"),
            "rows": ROWS, "label_names": LABEL_NAMES, "dataset_name": DATASET_NAME,
            "test_fraction": _number("The held-out share (0.1 to 0.5), or 0 for the probe's own"),
            "seed": _integer("The split's seed, or -1 for the probe's own"),
        },
        lambda a: f"Ablate · probe {a.get('probe_id', '')}",
    ),
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
        "a corpus through the SAE for feature statistics, contexts and communities, and replaces any "
        "atlas of it on that corpus. list_jobs follows it." + RUN_NOTE,
        {"model": MODEL, "sae": SAE,
         "corpus": _enum(ATLAS_CORPORA, "openwebtext (the default, web text), wikitext (Wikipedia) or tinystories (simple stories)"),
         "tokens": _integer("Corpus tokens to read: 1000000, 2000000 or 5000000")},
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
        "compare_attention", RUN,
        "Attention Analysis compare mode: how two prompts' attention patterns differ on one model, layer by "
        "layer and head by head. Each side captures its prompt, or reuses a capture by activation ID (give "
        "\"\" for the other). The full result opens on that page." + RUN_NOTE,
        {"model": MODEL,
         "prompt_1": _string("The first prompt, or \"\" with activation_id_1"),
         "activation_id_1": _string("A capture's ID, or \"\""),
         "prompt_2": _string("The second prompt, or \"\" with activation_id_2"),
         "activation_id_2": _string("A capture's ID, or \"\"")},
        lambda a: f"Compare attention · {a.get('model', '')}",
    ),
    Tool(
        "compare_probe_sae", RUN,
        "Compare one of the user's saved probes with an SAE at the probe's layer, as the Probe Lab's SAE "
        "view does, on the probe's own built-in dataset and split: the features whose decoder directions "
        "are nearest the probe's (against random and shuffled-label directions), the features its score "
        "reads on this data, and sparse probes on the top k features next to a dense probe. Opens in the "
        "Probe Lab's SAE view." + RUN_NOTE,
        {"probe_id": PROBE_ID, "sae": SAE},
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
        "generalize_probes", RUN,
        "Probe Lab generalize view: train probes (both methods, one pooling, every layer) on each of 2 to 6 "
        "datasets and test them on every other's held-out split, optionally also on the model's base/instruct "
        "sibling, for a train × test AUROC matrix. Opens in the Probe Lab's generalize view." + RUN_NOTE,
        {
            "model": MODEL,
            "datasets": {
                "type": "array",
                "description": "2 to 6 datasets: built-in ones by ID (part main, or ood for its OOD set), or your own rows",
                "items": {
                    "type": "object",
                    "properties": {
                        "dataset": _string("A built-in dataset ID, or \"\" with rows"),
                        "part": _enum(("main", "ood"), "A built-in dataset's main rows or its OOD set"),
                        "rows": ROWS, "label_names": LABEL_NAMES, "name": DATASET_NAME,
                    },
                    "required": ["dataset", "part", "rows", "label_names", "name"],
                    "additionalProperties": False,
                },
            },
            "transfer": _boolean("Also read every dataset with the model's base/instruct sibling and test across the two"),
            "pooling": _enum(POOLINGS, "How a text's tokens are pooled"),
            "chat_template": _boolean("Read texts in an instruct model's chat template"),
            "test_fraction": _number("The held-out share, 0.1 to 0.5 (0.3 by default)"),
            "seed": _integer("The split's seed (0 by default)"),
        },
        lambda a: f"Generalization · {a.get('model', '')}",
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
        "geometry_sae", RUN,
        "Geometry Lab SAE view: read a run's concept through an SAE (at the layer it reads): every feature's "
        "tuning curve over the values, how few features write the class means' shape (capture, shatter, "
        "dilute or missed, next to random feature orders), and the SAE's L0 and unexplained variance on these "
        "activations. Opens in the Geometry Lab's SAE view." + RUN_NOTE,
        {"run_id": _string("A Geometry Lab run's ID (run_geometry's result or the page state)"), "sae": SAE},
        lambda a: f"Geometry SAE · {_sae_chip(a)}",
    ),
    Tool(
        "mine_patterns", RUN,
        "Attention Analysis mine mode: head importance, cross-layer patterns and circuit candidates over a "
        "prompt's attention, captured (or a capture reused by activation ID). The full result opens on that "
        "page." + RUN_NOTE,
        {"model": MODEL, "prompt": _string("The prompt to capture, or \"\" with an activation_id"),
         "activation_id": _string("A capture's ID, or \"\"")},
        lambda a: f"Mine patterns · {a.get('model', '')}",
    ),
    Tool(
        "run_geometry", RUN,
        "Run the Geometry Lab on a concept, built-in or your own: class means at every layer and the shape "
        "tests against shuffled baselines. Its result has a run_id for compare_geometry, steer_geometry and "
        "geometry_sae. The full result opens in the Geometry Lab." + RUN_NOTE,
        {
            "model": MODEL,
            "concept": _string("A built-in concept ID (see the guide), or \"\" with custom"),
            "custom": CUSTOM_CONCEPT,
            "read": _enum(GEOMETRY_READS, "Where each example is read"),
            "chat_template": _boolean("Read examples in an instruct model's chat template (instruct models only)"),
            "seed": _integer("The run's seed (0 by default)"),
        },
        lambda a: f"Geometry · {_concept_chip(a)} · {a.get('model', '')}",
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
        "save_probe", RUN,
        "Save one probe of a Probe Lab sweep to the user's saved probes (free), so score_probe, steer_probe, "
        "ablate_probe, compare_probe_sae and the Generate page can use it. run_id is the sweep's (train_probe's "
        "result, or the Probe Lab's page state); sweeps are kept on the server for a few hours." + RUN_NOTE,
        {
            "run_id": _string("The sweep's run_id"),
            "layer": _integer("The sweep position, as the sweep numbers it (1 to its layers; 0, the embeddings, can't be saved)"),
            "pooling": _enum(POOLINGS),
            "method": _enum(METHODS),
            "name": _string("Its name, or \"\" for one from its dataset and layer"),
        },
        lambda a: f"Save probe · position {a.get('layer', '')}",
    ),
    Tool(
        "score_probe", RUN,
        "Probe Lab score view: a saved probe's score at every token of up to 8 texts, and its pooled score of "
        "each (above 0 reads as label 1). Charged as one GPU request. Opens in the Probe Lab's score view." + RUN_NOTE,
        {"probe_id": PROBE_ID, "texts": _strings("1 to 8 texts, each under 2000 characters")},
        lambda a: f"Score · probe {a.get('probe_id', '')}",
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
        "steer_geometry", RUN,
        "Geometry Lab steer view: walk an eval prompt from a source value to a target value at one layer of a "
        "run's concept, along the class means' curve (manifold) and in a straight line (linear), reading the "
        "next-token probabilities of the values at each step. Not for categorical concepts. Opens in the "
        "Geometry Lab's steer view." + RUN_NOTE,
        {
            "run_id": _string("A Geometry Lab run's ID (run_geometry's result or the page state)"),
            "layer": _integer("The residual stream position steered (0 the embeddings, i layer i-1's output)"),
            "dims": _integer("The class means' top components the curve lives in (1 to 16, under the number of values; 3 is usual)"),
            "source": _string("The value the walk starts at (filled into the eval prompt)"),
            "target": _string("The value it ends at"),
            "steps": _integer("Steps along the way, 4 to 32 (16 by default)"),
            "eval_template": _string("A prompt with {x} once, ending where the model would say a value, or \"\" for the concept's first"),
        },
        lambda a: f"Steer · {a.get('source', '')} → {a.get('target', '')}",
    ),
    Tool(
        "steer_probe", RUN,
        "Probe Lab steer view: generate from a prompt with α × a saved probe's direction added after its layer "
        "(α in class gaps; negative steers toward label 0), next to the unsteered baseline and optionally the "
        "direction ablated; each text comes back with the probe's score and its perplexity under the "
        "unsteered model. Opens in the Probe Lab's steer view." + RUN_NOTE,
        {
            "probe_id": PROBE_ID,
            "prompt": _string("The prompt to continue"),
            "alphas": {"type": "array", "items": {"type": "number"}, "description": "1 to 6 nonzero strengths, at most ±64 (e.g. [-4, -2, 2, 4])"},
            "ablate": _boolean("Also generate with the direction ablated everywhere"),
            "generated_only": _boolean("Steer only the generated tokens, not the prompt's"),
            "max_tokens": _integer("Tokens per generation, 1 to 200 (40 by default)"),
            "temperature": _number("0 to 2 (0 is greedy; otherwise every generation samples from the same seed)"),
        },
        lambda a: f"Steer · probe {a.get('probe_id', '')}",
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
        "Train the Probe Lab's layer sweep on a built-in dataset or your own rows: probes at every layer "
        "(logistic regression and difference in means, three poolings) with their control-task and embedding "
        "baselines. Its result has a run_id for save_probe. The full result opens in the Probe Lab." + RUN_NOTE,
        {
            "model": MODEL,
            "dataset": _string("A built-in dataset ID (see the guide), or \"\" with rows"),
            "rows": ROWS, "label_names": LABEL_NAMES, "dataset_name": DATASET_NAME,
            "chat_template": _boolean("Read texts in an instruct model's chat template (instruct models only)"),
            "test_fraction": _number("The held-out share: 0.1, 0.2, 0.3 or 0.4"),
            "seed": _integer("The split's seed (0 by default)"),
        },
        lambda a: f"Probes · {a.get('dataset') or a.get('dataset_name') or 'your rows'} · {a.get('model', '')}",
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
