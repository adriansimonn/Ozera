"""
The assistant's guide and tools name only what exists (ASSISTANT_PLAN.md): every page route,
page control and tool the system prompt mentions, every control the backend accepts is one the
frontend registers and can highlight, the tool schemas stay within strict tool use's limits,
and every run tool can be priced and run. Run from backend/: python -m pytest tests/test_assistant_guide.py
"""

import os
import re
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.assistant.pages import PAGES, check_patches, check_value, sae_key, sae_ref  # noqa: E402
from core.assistant.prompt import ROLE, render_guide, system_prompt  # noqa: E402
from core.assistant.tools import BY_NAME, ESTIMATE_ACTIONS, MAX_STRICT_TOOLS, RUN_TOOLS, TOOLS, definitions  # noqa: E402

FRONTEND = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "frontend", "src")
BRIDGES = os.path.join(FRONTEND, "components", "assistant", "pages")


def test_guide_routes_are_pages():
    routes = {p.route for p in PAGES.values()}
    named = set(re.findall(r"`(/[a-z]*)`", render_guide()))
    assert named, "the guide names page routes"
    assert named <= routes, f"unknown routes: {named - routes}"
    assert {p.route for p in PAGES.values() if p.id != "settings"} <= named | {"/settings"}


def test_guide_controls_exist():
    text = system_prompt()
    named = set(re.findall(r"`([a-z]+)\.([a-z_]+)`", text))
    assert named
    for page_id, control_id in named:
        assert page_id in PAGES, f"unknown page {page_id}"
        assert PAGES[page_id].control(control_id) is not None, f"{page_id} has no control {control_id}"


def test_guide_tools_exist():
    text = system_prompt()
    called = set(re.findall(r"`([a-z_]+)\(\)`", text))
    assert called, "the guide names tools as `name()`"
    assert called <= set(BY_NAME), f"unknown tools: {called - set(BY_NAME)}"
    # Tool-like names anywhere in the role and guide
    mentioned = set(re.findall(r"\b((?:get|list|search|set)_[a-z_]+)\b", text)) | set(re.findall(r"\b(navigate|highlight|estimate)\b", ROLE))
    assert mentioned <= set(BY_NAME), f"unknown tools: {mentioned - set(BY_NAME)}"
    # The rules on runs name every run tool
    runs = ROLE[ROLE.index("# Runs"):]
    assert all(re.search(rf"\b{name}\b", runs) for name in RUN_TOOLS), "the Runs section lists every run tool"


def test_system_prompt_is_static():
    assert system_prompt() == ROLE + "\n\n" + render_guide()
    assert "{{" not in system_prompt()
    assert not re.search(r"\b20\d\d-\d\d-\d\d\b", system_prompt()), "no dates in the system prompt (it would change the shared cache)"


def _walk(schema: dict, path: str):
    """Every object in a schema must be strict: closed, with every property required, and no unions."""
    assert "anyOf" not in schema and not isinstance(schema.get("type"), list), f"{path} uses a union type"
    if schema.get("type") == "object":
        assert schema.get("additionalProperties") is False, f"{path} isn't closed"
        assert sorted(schema.get("required", [])) == sorted(schema.get("properties", {})), f"{path} has optional properties"
        for name, sub in schema.get("properties", {}).items():
            _walk(sub, f"{path}.{name}")
    if schema.get("type") == "array":
        _walk(schema["items"], f"{path}[]")
    for key in ("minimum", "maximum", "minLength", "maxLength"):
        assert key not in schema, f"{path} uses {key}, which strict schemas don't support"


def test_tool_schemas_are_strict():
    tools = definitions()
    assert sum(1 for t in tools if t.get("strict")) <= MAX_STRICT_TOOLS
    assert [t["name"] for t in tools] == sorted(t["name"] for t in tools), "tools keep a fixed, sorted order"
    for t in tools:
        # Strict or not, every schema is written the strict way (closed, everything required)
        _walk(t["input_schema"], t["name"])
    for spec in TOOLS:
        assert spec.kind in {"read", "ui", "run"}
        if spec.kind != "read" or spec.properties and len(spec.properties) > 1:
            assert spec.strict, f"{spec.name} takes inputs the API should check: keep it strict"
    assert len(definitions()) == len(BY_NAME)


def test_run_tools_can_be_priced_and_run():
    from services.assistant_runs import RUNS
    from services.assistant_tools import ESTIMATES, EXECUTORS

    assert set(RUN_TOOLS) == set(RUNS), "every run tool has a quote and an executor, and nothing else does"
    assert not set(RUN_TOOLS) & set(EXECUTORS), "run tools only run through assistant_runs, after approval"
    assert {t.name for t in TOOLS if t.kind != "run"} == set(EXECUTORS)
    assert set(ESTIMATE_ACTIONS) == set(ESTIMATES)


def test_patches_are_checked():
    patches, problem = check_patches('[{"layer": 2, "patch_type": "residual", "intervention_type": "patch", "positions": [3, 1], "heads": [], "blend_factor": 1}]')
    assert problem is None and patches[0]["positions"] == [1, 3] and patches[0]["heads"] is None
    for bad in ("[]", "{}", "x", '[{"layer": -1, "patch_type": "residual"}]', '[{"layer": 0, "patch_type": "nope"}]',
                '[{"layer": 0, "patch_type": "mlp", "blend_factor": 2}]'):
        assert check_patches(bad)[1], bad


def test_sae_keys_round_trip():
    for key in ("ozera:nano:3", "ozera:mini:0", "external:google--gemma-scope-2-270m-pt__resid_post-layer_9"):
        assert sae_key(sae_ref(key)) == key
    for bad in ("ozera:big:1", "external:../x", "nano:3", ""):
        assert sae_ref(bad) is None


def test_values_are_checked():
    generate = PAGES["generate"]
    assert check_value(generate.control("max_tokens"), "300") == (300, None)
    assert check_value(generate.control("max_tokens"), "301")[1]
    assert check_value(generate.control("until_eos"), "true") == (True, None)
    assert check_value(generate.control("mode"), "Visualize") == ("visualize", None)
    assert check_value(generate.control("generate_button"), "x")[1]


def _frontend_files() -> list[str]:
    out = []
    for root, _, files in os.walk(FRONTEND):
        out += [os.path.join(root, f) for f in files if f.endswith((".ts", ".tsx"))]
    return out


@pytest.mark.skipif(not os.path.isdir(BRIDGES), reason="needs the frontend next to the backend")
def test_frontend_registers_the_same_controls():
    for page in PAGES.values():
        if not page.bridged:
            continue
        with open(os.path.join(BRIDGES, f"{page.id}.ts")) as f:
            source = f.read()
        declared = set(re.findall(r"id: '([a-z_]+)'", source))
        expected = {c.id for c in page.controls}
        assert declared == expected, f"{page.id}: frontend {sorted(declared)} vs backend {sorted(expected)}"


@pytest.mark.skipif(not os.path.isdir(BRIDGES), reason="needs the frontend next to the backend")
def test_every_control_can_be_highlighted():
    sources = ""
    for path in _frontend_files():
        with open(path) as f:
            sources += f.read()
    missing = [
        f"{p.id}.{c.id}" for p in PAGES.values() if p.bridged for c in p.controls if f'"{p.id}.{c.id}"' not in sources
    ]
    assert not missing, f"controls without a data-assist target: {missing}"
