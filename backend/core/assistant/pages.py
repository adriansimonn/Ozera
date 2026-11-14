"""
Ozera's pages as the assistant knows them: each page's route, and the controls its UI tools may
fill (set_inputs) or point at (highlight). The frontend registers the same control IDs for each
page (frontend/src/components/assistant/pages/); tests/test_assistant_guide.py checks both sides
and the guide agree.

A ui tool never runs anything: setting a control only changes what's on screen, and actions
(buttons) can only be highlighted, for the user to press.
"""

import json
import re
from dataclasses import dataclass
from typing import Optional

from core.geometry.concepts import builtin_concepts
from core.probes.datasets import builtin_datasets


# Activation Patching's patch points and interventions (api/schemas/patching.py's PatchSpec)
PATCH_TYPES = ("attention", "mlp", "residual", "attn_output", "ff_output", "post_attn", "post_ff")
INTERVENTIONS = ("patch", "zero_ablate", "mean_ablate", "noise_ablate")
# Patches a page or a run may set at most
MAX_PATCHES = 32


@dataclass(frozen=True)
class Control:
    id: str
    label: str
    # text, number, select, toggle, sae (an SAE key, see sae_key), patches (a JSON array of
    # patches, see check_patches), rows (a JSON array of labeled examples, see check_rows),
    # labels (two names, comma-separated), or action (highlight only)
    kind: str
    note: str = ""
    options: tuple[str, ...] = ()
    minimum: Optional[float] = None
    maximum: Optional[float] = None
    integer: bool = False


@dataclass(frozen=True)
class Page:
    id: str
    route: str
    title: str
    # Whether the page reports what's on screen and accepts set_inputs and highlight
    bridged: bool
    controls: tuple[Control, ...] = ()

    def control(self, control_id: str) -> Optional[Control]:
        return next((c for c in self.controls if c.id == control_id), None)


def _dataset_ids() -> tuple[str, ...]:
    return tuple(sorted(builtin_datasets()))


def _concept_ids() -> tuple[str, ...]:
    concepts = builtin_concepts()
    ids = concepts.keys() if isinstance(concepts, dict) else (c["id"] for c in concepts)
    return tuple(sorted(ids))


PAGES: dict[str, Page] = {
    "generate": Page("generate", "/", "Generate", True, (
        Control("prompt", "Prompt", "text"),
        Control("model", "Model", "text", "a model ID from list_models"),
        Control("max_tokens", "Tokens", "number", minimum=1, maximum=300, integer=True),
        Control("until_eos", "Generate until EOS", "toggle"),
        Control("temperature", "Temperature", "number", minimum=0, maximum=2),
        Control("top_k", "Top-k", "number", minimum=1, maximum=100, integer=True),
        Control("mode", "Generation mode", "select", "visualize captures activations; tokens is text only", ("visualize", "tokens")),
        Control("visualization", "Visualization", "select", options=("network", "attention", "activations", "journey", "flow")),
        Control("activation", "Stored generation", "text", "an activation ID; opens that generation's visualizations (free)"),
        Control("generate_button", "Generate", "action", "reads Generate + Visualize in visualize mode"),
    )),
    "training": Page("training", "/training", "Custom Models", False),
    "patching": Page("patching", "/patching", "Activation Patching", True, (
        Control("model", "Model", "text", "a model ID from list_models"),
        Control("source_prompt", "Source prompt", "text", "activations are copied from it (patch interventions)"),
        Control("target_prompt", "Target prompt", "text", "generation runs on it"),
        Control("patches", "Patches", "patches", "a JSON array of {layer, patch_type, intervention_type, positions, heads, blend_factor}; replaces the page's patches"),
        Control("max_tokens", "Max tokens", "number", minimum=1, maximum=100, integer=True),
        Control("temperature", "Temperature", "number", minimum=0, maximum=2),
        Control("run_button", "Run Experiment", "action"),
    )),
    "analysis": Page("analysis", "/analysis", "Attention Analysis", True, (
        Control("mode", "Mode", "select", options=("classify", "compare", "mine")),
        Control("model", "Model", "text", "a model ID from list_models"),
        Control("prompt", "Capture prompt", "text"),
        Control("activation", "Captured activations", "text", "a capture's ID from the page state; selects it"),
        Control("capture_button", "Capture", "action"),
        Control("run_button", "Run analysis", "action", "reads Classify Heads, Compare Patterns or Mine Patterns, by mode"),
    )),
    "sae": Page("sae", "/sae", "SAE Analysis", True, (
        Control("mode", "View", "select", options=("analyze", "browse", "dashboard", "compare", "autointerp", "load")),
        Control("sae", "SAE", "sae", "an SAE key from list_saes"),
        Control("text", "Input text", "text"),
        Control("feature", "Feature", "number", "selected in Browse, or once the text is analyzed", minimum=0, integer=True),
        Control("analyze_button", "Analyze", "action"),
    )),
    "probes": Page("probes", "/probes", "Probe Lab", True, (
        Control("view", "View", "select", options=("sweep", "score", "generalize", "steer", "ablate", "sae")),
        Control("model", "Model", "text", "a model ID from list_models"),
        Control("dataset", "Dataset", "select", "a built-in dataset", _dataset_ids()),
        Control("rows", "Your rows", "rows", "your own labeled examples instead of a built-in dataset (at least 10 per label)"),
        Control("label_names", "Label names", "labels", "with rows: label 0's name, then label 1's"),
        Control("dataset_name", "Dataset name", "text", "with rows"),
        Control("chat_template", "Chat template", "toggle", "instruct models only"),
        Control("test_fraction", "Test split", "select", options=("0.1", "0.2", "0.3", "0.4")),
        Control("seed", "Seed", "number", minimum=0, integer=True),
        Control("pooling", "Pooling", "select", options=("mean", "last", "max")),
        Control("method", "Method", "select", "logistic regression or difference in means", ("logreg", "diff_means")),
        Control("layer", "Layer", "number", "the residual stream position shown (0 is the embeddings)", minimum=0, integer=True),
        Control("train_button", "Train probes", "action"),
        Control("probe", "Probe", "text", "the score, steer, ablate and sae views' probe: sweep (the sweep's selection) or a saved probe's ID"),
        Control("direction", "Direction", "select", "a sweep probe's direction in the steer, ablate and sae views", ("diff_means", "logreg")),
        Control("score_text", "Text to score", "text", "score view"),
        Control("score_button", "Score text", "action"),
        Control("steer_prompt", "Steering prompt", "text", "steer view"),
        Control("steer_alphas", "Strengths", "text", "steer view: comma-separated, e.g. -4, -2, 2, 4"),
        Control("steer_max_tokens", "Max tokens", "number", "steer view", minimum=1, maximum=200, integer=True),
        Control("steer_temperature", "Temperature", "number", "steer view", minimum=0, maximum=2),
        Control("steer_generated_only", "Steer generated tokens only", "toggle", "steer view"),
        Control("steer_ablate", "Also ablate", "toggle", "steer view"),
        Control("steer_button", "Steer", "action"),
        Control("ablate_button", "Run ablation", "action"),
        Control("sae", "SAE", "sae", "sae view: an SAE at the probe's layer"),
        Control("sae_button", "Compare with SAE", "action"),
        Control("gen_datasets", "Datasets", "text", "generalize view: 2 to 6 built-in dataset IDs, comma-separated (id:ood for an OOD set)"),
        Control("gen_transfer", "Base/instruct transfer", "toggle", "generalize view"),
        Control("gen_pooling", "Pooling", "select", "generalize view", ("mean", "last", "max")),
        Control("gen_test_fraction", "Test split", "select", "generalize view", ("0.2", "0.3", "0.4", "0.5")),
        Control("gen_chat_template", "Chat template", "toggle", "generalize view"),
        Control("gen_button", "Run matrix", "action"),
    )),
    "geometry": Page("geometry", "/geometry", "Geometry Lab", True, (
        Control("view", "View", "select", options=("shape", "layers", "similarity", "compare", "steer", "sae", "atlas")),
        Control("model", "Model", "text", "a model ID from list_models"),
        Control("run", "Runs", "text", "a kept run's run_id from the page state"),
        Control("concept", "Concept", "select", "a built-in concept", _concept_ids()),
        Control("concept_name", "Concept name", "text", "your own concept (setting any of these switches to it)"),
        Control("structure", "Structure", "select", "your own concept", ("cyclic", "ordinal", "categorical")),
        Control("values", "Values", "text", "your own concept: one per line, in order (value | group for grouped categorical ones)"),
        Control("templates", "Templates", "text", "your own concept: one per line, each with {x} once"),
        Control("eval_templates", "Eval prompts", "text", "your own concept, for steering: one per line, each with {x}"),
        Control("read", "Read at", "select", "the value's last token, the mean over its tokens, or the template's last token", ("value_last", "value_mean", "template_last")),
        Control("chat_template", "Chat template", "toggle", "instruct models only"),
        Control("seed", "Seed", "number", minimum=0, integer=True),
        Control("layer", "Layer", "number", "the residual stream position shown", minimum=0, integer=True),
        Control("run_button", "Run", "action"),
        Control("compare_with", "Compare with", "text", "compare view: another kept run's run_id (or the same one for its own layers)"),
        Control("compare_dims", "Dims", "select", "compare view", ("2", "3", "4", "6", "8", "all")),
        Control("steer_source", "From", "text", "steer view: a value of the run's concept"),
        Control("steer_target", "To", "text", "steer view: a value of the run's concept"),
        Control("steer_steps", "Steps", "number", "steer view", minimum=4, maximum=32, integer=True),
        Control("steer_dims", "Dims", "number", "steer view", minimum=1, maximum=16, integer=True),
        Control("steer_template", "Eval prompt", "text", "steer view: with {x} once"),
        Control("steer_button", "Steer", "action"),
        Control("sae", "SAE", "sae", "sae view"),
        Control("sae_button", "Read with SAE", "action"),
        Control("atlas_sae", "SAE", "sae", "atlas view"),
        Control("atlas_corpus", "Corpus", "select", "atlas view", ("openwebtext", "wikitext", "tinystories")),
        Control("atlas_tokens", "Tokens", "select", "atlas view", ("1000000", "2000000", "5000000")),
        Control("atlas_button", "Compute atlas", "action", "reads Recompute once the atlas exists"),
    )),
    "settings": Page("settings", "/settings", "Settings", False),
}

PAGE_IDS = tuple(PAGES)

# Built-in SAE keys: "ozera:<nano|mini>:<layer>" (Ozera's residual stream SAEs) and
# "external:<sae_id>" (one the user loaded on the SAE page)
_SAE_KEY = re.compile(r"^(?:ozera:(nano|mini):(\d{1,3})|external:([A-Za-z0-9][A-Za-z0-9._-]*))$")


def sae_ref(key: str) -> Optional[dict]:
    """An SAE key as an SAE reference ({"kind": "ozera", ...} or {"kind": "external", ...}), or None."""
    match = _SAE_KEY.match(key.strip())
    if match is None or ".." in key:
        return None
    if match.group(1):
        return {"kind": "ozera", "model": match.group(1), "layer": int(match.group(2))}
    return {"kind": "external", "sae_id": match.group(3)}


def sae_key(ref: dict) -> str:
    """An SAE reference's key (the frontend's refKey)."""
    if ref.get("kind") == "ozera":
        return f"ozera:{ref['model']}:{ref['layer']}"
    return f"external:{ref['sae_id']}"


def page_for_route(route: str) -> Optional[Page]:
    path = (route or "").split("?")[0].rstrip("/") or "/"
    return next((p for p in PAGES.values() if p.route == path), None)


# Control IDs (page.control) in the assistant's text, with any code or bold marks around them:
# the user sees each as its label instead. The prompt asks for labels; this catches the rest.
_LABELS = {f"{p.id}.{c.id}": c.label for p in PAGES.values() for c in p.controls}
_CONTROL_ID = re.compile(
    r"(?:\*\*)?`?(?<![\w/:.-])("
    + "|".join(re.escape(k) for k in sorted(_LABELS, key=len, reverse=True))
    + r")(?![\w-])`?(?:\*\*)?"
)
# The end of streamed text that may be the start of a control ID
_HELD = re.compile(r"[\w.`*]+\Z")


def control_names(text: str) -> str:
    """Text with the control IDs in it written as their labels (e.g. **Run**)."""
    return _CONTROL_ID.sub(lambda m: f"**{_LABELS[m.group(1)]}**", text) if text else text


class ControlNames:
    """control_names for streamed text: holds back a fragment's end until the next shows what it is."""

    def __init__(self):
        self.held = ""

    def feed(self, text: str) -> str:
        text = self.held + text
        tail = _HELD.search(text)
        self.held = tail.group(0) if tail else ""
        return control_names(text[: len(text) - len(self.held)])

    def flush(self) -> str:
        text, self.held = self.held, ""
        return control_names(text)


def check_patches(value) -> tuple[Optional[list], Optional[str]]:
    """Patches (a list, or JSON text of one) checked as PatchSpec checks them: (patches, None), or (None, why not)."""
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except ValueError:
            return None, "patches is a JSON array of patches"
    if not isinstance(value, list) or not value:
        return None, "patches is a non-empty array of patches"
    if len(value) > MAX_PATCHES:
        return None, f"At most {MAX_PATCHES} patches"
    patches = []
    for i, p in enumerate(value):
        if not isinstance(p, dict):
            return None, f"Patch {i + 1} isn't an object"
        layer = p.get("layer")
        if not isinstance(layer, int) or isinstance(layer, bool) or layer < 0:
            return None, f"Patch {i + 1}'s layer is a whole number from 0"
        if p.get("patch_type") not in PATCH_TYPES:
            return None, f"Patch {i + 1}'s patch_type is one of: {', '.join(PATCH_TYPES)}"
        intervention = p.get("intervention_type", "patch")
        if intervention not in INTERVENTIONS:
            return None, f"Patch {i + 1}'s intervention_type is one of: {', '.join(INTERVENTIONS)}"
        lists = {}
        for key in ("positions", "heads"):
            items = p.get(key)
            if items in (None, []):
                lists[key] = None
            elif isinstance(items, list) and all(isinstance(x, int) and not isinstance(x, bool) and x >= 0 for x in items):
                lists[key] = sorted(set(items))
            else:
                return None, f"Patch {i + 1}'s {key} is a list of whole numbers from 0 ([] for all)"
        blend = p.get("blend_factor", 1.0)
        if isinstance(blend, bool) or not isinstance(blend, (int, float)) or not 0 <= blend <= 1:
            return None, f"Patch {i + 1}'s blend_factor is between 0 and 1"
        patches.append({
            "layer": layer, "patch_type": p["patch_type"], "intervention_type": intervention,
            "positions": lists["positions"], "heads": lists["heads"], "neurons": None, "blend_factor": float(blend),
        })
    return patches, None


# A labeled dataset's limits (core.probes.budget)
MIN_ROWS_PER_LABEL = 10
MAX_ROWS = 2000
MAX_ROW_CHARS = 2000


def check_rows(value) -> tuple[Optional[list], Optional[str]]:
    """Labeled examples (a list, or JSON text of one) as {text, label} dicts: (rows, None), or (None, why not)."""
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except ValueError:
            return None, "rows is a JSON array of {text, label}"
    if not isinstance(value, list) or not value:
        return None, "rows is a non-empty array of {text, label}"
    if len(value) > MAX_ROWS:
        return None, f"At most {MAX_ROWS} rows"
    rows = []
    for i, r in enumerate(value):
        if not isinstance(r, dict) or not isinstance(r.get("text"), str) or not r["text"].strip():
            return None, f"Row {i + 1} needs a text"
        if len(r["text"]) > MAX_ROW_CHARS:
            return None, f"Row {i + 1} is over {MAX_ROW_CHARS} characters"
        if r.get("label") not in (0, 1) or isinstance(r.get("label"), bool):
            return None, f"Row {i + 1}'s label is 0 or 1"
        rows.append({"text": r["text"], "label": r["label"]})
    counts = [sum(1 for r in rows if r["label"] == label) for label in (0, 1)]
    if min(counts) < MIN_ROWS_PER_LABEL:
        return None, f"Each label needs at least {MIN_ROWS_PER_LABEL} rows (these have {counts[0]} and {counts[1]})"
    return rows, None


def check_labels(value) -> tuple[Optional[list], Optional[str]]:
    """Two label names (a list, or text with a comma between them): (names, None), or (None, why not)."""
    names = value if isinstance(value, list) else str(value).split(",")
    names = [str(n).strip() for n in names]
    if len(names) != 2 or not all(names) or any(len(n) > 60 for n in names):
        return None, "Give two label names, label 0's first"
    return names, None


def check_value(control: Control, value: str) -> tuple[Optional[object], Optional[str]]:
    """A value for a control, parsed and checked: (value, None), or (None, why it's refused)."""
    text = value.strip() if control.kind != "text" else value
    if control.kind == "action":
        return None, f"{control.id} is a button: highlight it for the user to press"
    if control.kind == "text":
        return value, None
    if control.kind == "sae":
        return (text, None) if sae_ref(text) else (None, f"{control.id} takes an SAE key such as ozera:nano:3 (see list_saes)")
    if control.kind == "patches":
        return check_patches(text)
    if control.kind == "rows":
        return check_rows(text)
    if control.kind == "labels":
        return check_labels(text)
    if control.kind == "toggle":
        lowered = text.lower()
        if lowered in ("true", "on", "yes", "1"):
            return True, None
        if lowered in ("false", "off", "no", "0"):
            return False, None
        return None, f"{control.id} is on or off: true or false"
    if control.kind == "select":
        match = next((o for o in control.options if o.lower() == text.lower()), None)
        return (match, None) if match is not None else (None, f"{control.id} is one of: {', '.join(control.options)}")
    if control.kind == "number":
        try:
            number = float(text)
        except ValueError:
            return None, f"{control.id} takes a number"
        if control.integer:
            if number != int(number):
                return None, f"{control.id} takes a whole number"
            number = int(number)
        if control.minimum is not None and number < control.minimum or control.maximum is not None and number > control.maximum:
            low = "" if control.minimum is None else f"{control.minimum:g}"
            high = "" if control.maximum is None else f"{control.maximum:g}"
            return None, f"{control.id} is between {low or '-∞'} and {high or '∞'}"
        return number, None
    return None, f"{control.id} can't be set"
