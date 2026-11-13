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
    # patches, see check_patches), or action (highlight only)
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
        Control("generate_button", "Generate", "action"),
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
        Control("run_button", "Run analysis", "action"),
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
        Control("chat_template", "Chat template", "toggle", "instruct models only"),
        Control("test_fraction", "Test split", "select", options=("0.1", "0.2", "0.3", "0.4")),
        Control("seed", "Seed", "number", minimum=0, integer=True),
        Control("pooling", "Pooling", "select", options=("mean", "last", "max")),
        Control("method", "Method", "select", "logistic regression or difference in means", ("logreg", "diff_means")),
        Control("layer", "Layer", "number", "the residual stream position shown (0 is the embeddings)", minimum=0, integer=True),
        Control("train_button", "Train probes", "action"),
    )),
    "geometry": Page("geometry", "/geometry", "Geometry Lab", True, (
        Control("view", "View", "select", options=("shape", "layers", "similarity", "compare", "steer", "sae", "atlas")),
        Control("model", "Model", "text", "a model ID from list_models"),
        Control("concept", "Concept", "select", "a built-in concept", _concept_ids()),
        Control("read", "Read at", "select", "the value's last token, the mean over its tokens, or the template's last token", ("value_last", "value_mean", "template_last")),
        Control("chat_template", "Chat template", "toggle", "instruct models only"),
        Control("seed", "Seed", "number", minimum=0, integer=True),
        Control("layer", "Layer", "number", "the residual stream position shown", minimum=0, integer=True),
        Control("run_button", "Run", "action"),
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
