"""
The assistant's run tools (core.assistant.tools lists them; ASSISTANT_PLAN.md, Phase 2). Each
has a quote, its price for the approval card (computed as its page prices it), and an
executor, which runs once the user has approved it, as the user, through its page's own route
function: it's checked, charged and stored exactly as when the page runs it. What the run's
own service charged is recorded (credit_service.recording_charges) and reported with it.

A run returns a compact summary for Claude (its numbers with their baselines, and a `link`
for every feature) and its full result for the page ({"page", "data"}, and a "route" to open
when there's one), which the browser opens there. A run never spends the assistant's own
credits: its service charges it, free messages or not.
"""

import asyncio
import logging
from typing import Awaitable, Callable, Optional

from fastapi import HTTPException
from sqlalchemy.orm import Session

from core.assistant.loop import RunQuote, ToolOutcome
from core.assistant.pages import PAGES, check_patches
from core.assistant.tools import ATLAS_TOKENS, GEOMETRY_READS
from db import SessionLocal
from models.database import FeatureHypothesis, User, UserExternalSAE
from services.assistant_tools import ToolContext, _atlas, _explanation_brief, _parse_sae, _round, _short, feature_link

logger = logging.getLogger(__name__)

# Limits on what a run may be given (the pages' own limits)
MAX_TEXT_CHARS = 20_000
MAX_GENERATE_TOKENS = 300
MAX_PATCHING_TOKENS = 100
TEST_FRACTIONS = (0.1, 0.2, 0.3, 0.4)
# Characters of a text the approval card and a summary show
CARD_CHARS = 240
SUMMARY_TEXT_CHARS = 1500
# Items a summary lists at most
STRONGEST_FEATURES = 15
HEADS_PER_TYPE = 5


def _bad(detail: str):
    raise HTTPException(status_code=400, detail=detail)


def _cut(text: Optional[str], chars: int = CARD_CHARS) -> str:
    text = " ".join((text or "").split())
    return text if len(text) <= chars else text[: chars - 1] + "…"


def _money(value: float) -> str:
    return f"${value:.2f}" if value >= 0.1 else f"${value:.3f}"


def _text(args: dict, key: str, what: str, required: bool = True, limit: int = MAX_TEXT_CHARS) -> str:
    value = args.get(key)
    value = value if isinstance(value, str) else ""
    if required and not value.strip():
        _bad(f"Give the {what}")
    if len(value) > limit:
        _bad(f"The {what} is over {limit} characters")
    return value


def _int(args: dict, key: str, low: int, high: Optional[int], what: str) -> int:
    value = args.get(key)
    if isinstance(value, bool) or not isinstance(value, (int, float)) or value != int(value):
        _bad(f"{what} is a whole number")
    value = int(value)
    if value < low or (high is not None and value > high):
        _bad(f"{what} is between {low} and {high}" if high is not None else f"{what} is at least {low}")
    return value


def _float(args: dict, key: str, low: float, high: float, what: str) -> float:
    value = args.get(key)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        _bad(f"{what} is a number")
    if not low <= value <= high:
        _bad(f"{what} is between {low:g} and {high:g}")
    return float(value)


def _model_name(model_id: str) -> str:
    from core.open_source import OPEN_SOURCE_MODELS

    if model_id in OPEN_SOURCE_MODELS:
        return OPEN_SOURCE_MODELS[model_id].display_name
    return f"Ozera {model_id.title()}" if model_id in ("nano", "mini") else model_id


def _check_model(db: Session, user: User, model_id: str) -> str:
    from services.custom_models import resolve_model

    model_id = (model_id or "").strip()
    if not model_id:
        _bad("Give a model (list_models gives the IDs)")
    resolve_model(db, user.id, model_id)
    return model_id


async def _specs(db: Session, user: User, model_id: str):
    from api.probes import _model_specs

    return await _model_specs(db, user.id, model_id)


def _links_model(model_id: Optional[str]) -> str:
    return model_id or ""


# Generate

def _generate_request(db: Session, user: User, args: dict):
    from api.main import GenerateRequest

    model = _check_model(db, user, args.get("model", ""))
    prompt = _text(args, "prompt", "prompt")
    top_k = int(args.get("top_k") or 0) or 40
    if not 1 <= top_k <= 100:
        _bad("top_k is between 1 and 100 (0 for 40)")
    return GenerateRequest(
        prompt=prompt, model=model,
        max_tokens=_int(args, "max_tokens", 1, MAX_GENERATE_TOKENS, "max_tokens"),
        temperature=_float(args, "temperature", 0, 2, "temperature"),
        top_k=top_k, stop_at_eos=bool(args.get("until_eos")),
    )


async def quote_generate(db: Session, user: User, ctx: ToolContext, args: dict) -> RunQuote:
    from api.main import _check_visualizable, generation_estimate

    body = _generate_request(db, user, args)
    await _check_visualizable(db, user.id, body)
    estimate = generation_estimate(body)
    tokens = f"until end of sequence, at most {body.max_tokens}" if body.stop_at_eos else f"{body.max_tokens}"
    return RunQuote(
        title="Generate with activations",
        details=[["Model", _model_name(body.model)], ["Prompt", _cut(body.prompt)], ["Tokens", tokens],
                 ["Sampling", f"temperature {body.temperature:g}, top-k {body.top_k}"]],
        estimate=estimate, note="At most this; charged on the tokens it generates",
    )


async def run_generate(db: Session, user: User, ctx: ToolContext, args: dict) -> ToolOutcome:
    from api.main import generate_with_activations_for

    body = _generate_request(db, user, args)
    result = await generate_with_activations_for(db, user, body)
    activation_id = result.get("activation_id")
    out = {
        "activation_id": activation_id, "model": body.model, "prompt_tokens": result.get("prompt_tokens"),
        "generated_tokens": result.get("generated_tokens"), "finish_reason": result.get("finish_reason"),
        "text": _cut(result.get("text"), SUMMARY_TEXT_CHARS),
        "note": "Opened on the Generate page; get_activation reads each position's top predictions",
    }
    return ToolOutcome(out, chip=f"Generated · {_model_name(body.model)}", result={
        "page": "generate",
        "route": f"{PAGES['generate'].route}?id={activation_id}" if activation_id else None,
        "data": {"activation_id": activation_id, "prompt": body.prompt, "model": body.model},
    })


# SAE analysis

def _sae_target(db: Session, user: User, args: dict) -> tuple[str, dict, str, Optional[str]]:
    """(the SAE key, its reference, its name, the model ID its features link with) for a run, checked like the SAE page."""
    from core.open_source.registry import model_for_hf_id
    from services.model_specs import BASE_MODEL_CONFIGS

    key = (args.get("sae") or "").strip()
    ref, _ = _parse_sae(key)
    if ref["kind"] == "ozera":
        layers = BASE_MODEL_CONFIGS[ref["model"]].num_layers
        if ref["layer"] >= layers:
            _bad(f"Ozera {ref['model']} has {layers} layers")
        return key, ref, f"Ozera {ref['model'].title()} L{ref['layer']} residual", ref["model"]
    row = db.query(UserExternalSAE).filter_by(user_id=user.id, sae_id=ref["sae_id"]).first()
    if row is None:
        raise HTTPException(status_code=404, detail="That SAE isn't in the user's list (the SAE page's load view adds one)")
    linked = model_for_hf_id(row.base_model) if row.base_model else None
    return key, ref, row.display_name or ref["sae_id"], linked


async def quote_analyze_sae(db: Session, user: User, ctx: ToolContext, args: dict) -> RunQuote:
    from services.credit_service import MIN_SAE_CHARGE, estimate_sae_cost

    key, ref, name, _ = _sae_target(db, user, args)
    text = _text(args, "text", "text")
    estimate = estimate_sae_cost(len(text), ref["model"] if ref["kind"] == "ozera" else "external", "analyze")
    return RunQuote(
        title="Analyze text through an SAE", details=[["SAE", name], ["Text", _cut(text)]],
        estimate=estimate, hold=estimate * 1.5,
        note=f"Charged by the tokens it reads, at least {_money(MIN_SAE_CHARGE)}", chip=f"Analyze · {_short(name)}",
    )


def _analysis_summary(response: dict, key: str, model: Optional[str]) -> dict:
    metrics = response.get("metrics") or {}
    width = metrics.get("total_features")
    l0 = metrics.get("avg_l0")
    sparse = (response.get("features") or {}).get("sparse_activations") or []
    counts: dict[int, int] = {}
    for token in sparse:
        for feature in token:
            counts[int(feature)] = counts.get(int(feature), 0) + 1
    tokens = response.get("tokens") or []
    strongest = sorted(response.get("top_features") or [], key=lambda f: -(f.get("activation") or 0))[:STRONGEST_FEATURES]
    common = sorted(counts.items(), key=lambda kv: -kv[1])[:10]
    return {
        "sae": key, "tokens": response.get("num_tokens"), "text_tokens": tokens[:80],
        "avg_l0": _round(l0, 1), "width": width,
        "density": _round(l0 / width, 4) if l0 is not None and width else None,
        "active_features": metrics.get("num_active_features"), "max_activation": _round(metrics.get("max_activation")),
        "strongest": [
            {"feature": f["feature_id"], "link": feature_link(key, f["feature_id"], _links_model(model)),
             "token": f.get("token"), "position": f.get("token_idx"), "activation": _round(f.get("activation"))}
            for f in strongest
        ],
        "most_frequent": [
            {"feature": f, "link": feature_link(key, f, _links_model(model)), "tokens_active": n} for f, n in common
        ],
        "note": "Opened in the SAE page's Analyze view. Activations are in this SAE's units; get_feature reads a feature's stored explanation",
    }


async def run_analyze_sae(db: Session, user: User, ctx: ToolContext, args: dict) -> ToolOutcome:
    from api.sae import ExternalSAEAnalyzeRequest, SAEAnalyzeRequest, analyze_external_sae, analyze_text

    key, ref, name, model = _sae_target(db, user, args)
    text = _text(args, "text", "text")
    if ref["kind"] == "ozera":
        response = await analyze_text(
            request=SAEAnalyzeRequest(model=ref["model"], layer=ref["layer"], activation_type="residual", text=text, top_k=50),
            db=db, user=user,
        )
    else:
        response = await analyze_external_sae(request=ExternalSAEAnalyzeRequest(sae_id=ref["sae_id"], text=text, top_k=50), db=db, user=user)
    return ToolOutcome(_analysis_summary(response, key, model), chip=f"Analyzed · {_short(name)}", result={
        "page": "sae", "data": {"sae": key, "text": text, "response": response},
    })


# Autointerp: one feature

async def _feature_atlas(db: Session, user: User, args: dict, explainable: bool = True):
    """(info, atlas, model, key, feature) for a single-feature autointerp run, checked as its route checks it."""
    from services import autointerp

    model, key = (args.get("model") or "").strip(), (args.get("sae") or "").strip()
    feature = _int(args, "feature", 0, None, "feature")
    info, atlas = await _atlas(db, user, model, key)
    if atlas is None:
        raise HTTPException(status_code=404, detail="This SAE has no atlas on this model yet (build_atlas builds one)")
    if explainable:
        state, reason = autointerp.state(atlas.meta)
        if state != "ready":
            _bad(reason)
    if feature >= atlas.width:
        _bad(f"This SAE has {atlas.width} features")
    return info, atlas, model, key, feature


def _models_note() -> str:
    from core.autointerp.budget import DEFAULT_EXPLAINER, DEFAULT_SCORER, MODELS

    return f"{MODELS[DEFAULT_EXPLAINER].name} explains, {MODELS[DEFAULT_SCORER].name} scores"


def _feature_details(info, atlas, model: str, feature: int) -> list[list[str]]:
    return [["Feature", str(feature)], ["SAE", info.name], ["Model", _model_name(model)], ["Corpus", atlas.corpus]]


async def quote_explain_feature(db: Session, user: User, ctx: ToolContext, args: dict) -> RunQuote:
    from api.autointerp import _explain_price
    from core.autointerp.budget import DEFAULT_EXPLAINER, DEFAULT_SCORER
    from services import autointerp
    from services.credit_service import MIN_AUTOINTERP_CHARGE

    info, atlas, model, key, feature = await _feature_atlas(db, user, args)
    chip = f"Explain {feature} · {_short(info.name)}"
    details = _feature_details(info, atlas, model, feature) + [["Models", _models_note()]]
    if autointerp.stored(db, atlas.directory, feature, DEFAULT_EXPLAINER, DEFAULT_SCORER) is not None:
        return RunQuote("Explain a feature", details, 0.0, note="Already explained with these models: the stored explanation, free", chip=chip)
    return RunQuote(
        "Explain a feature", details, _explain_price(),
        note=f"Charged its usage, at least {_money(MIN_AUTOINTERP_CHARGE)}; its exact upper bound is held while it runs (10–30 s)", chip=chip,
    )


def _ref(key: str):
    _, ref = _parse_sae(key)
    return ref


async def run_explain_feature(db: Session, user: User, ctx: ToolContext, args: dict) -> ToolOutcome:
    from api.autointerp import explain_feature_for
    from api.schemas.autointerp import ExplainRequest

    info, atlas, model, key, feature = await _feature_atlas(db, user, args)
    result = await explain_feature_for(db, user, ExplainRequest(model=model, sae=_ref(key), corpus=atlas.corpus, feature=feature))
    out = {
        "feature": feature, "link": feature_link(key, feature, model), "sae_name": info.name, "corpus": atlas.corpus,
        "explanation": _explanation_brief(result), "already_stored": result.get("cached"),
        "note": "Shared with everyone who has the SAE; the feature's chip shows it now",
    }
    return ToolOutcome(out, chip=f"Explained {feature} · {_short(info.name)}", result={"page": None, "labels": key, "data": None})


async def quote_test_hypothesis(db: Session, user: User, ctx: ToolContext, args: dict) -> RunQuote:
    from api.autointerp import _hypothesis_price
    from core.autointerp.budget import DEFAULT_SCORER, MAX_HYPOTHESIS_CHARS
    from core.autointerp.pipeline import clean_text
    from services import autointerp

    info, atlas, model, key, feature = await _feature_atlas(db, user, args)
    text = clean_text(_text(args, "hypothesis", "hypothesis", limit=1000))
    if not text:
        _bad("Write the hypothesis")
    if len(" ".join(args.get("hypothesis", "").split())) > MAX_HYPOTHESIS_CHARS:
        _bad(f"Keep the hypothesis under {MAX_HYPOTHESIS_CHARS} characters")
    details = _feature_details(info, atlas, model, feature) + [["Hypothesis", text]]
    chip = f"Hypothesis {feature} · {_short(info.name)}"
    same = [h for h in autointerp.hypotheses(db, user.id, atlas.directory, feature) if h.text == text and h.scorer_model == DEFAULT_SCORER]
    if same:
        return RunQuote("Score a hypothesis", details, 0.0, note="Already scored: the stored scores, free", chip=chip)
    return RunQuote("Score a hypothesis", details, _hypothesis_price(),
                    note="Charged its usage; its exact upper bound is held while it runs. Private to you", chip=chip)


async def run_test_hypothesis(db: Session, user: User, ctx: ToolContext, args: dict) -> ToolOutcome:
    from api.autointerp import score_hypothesis_for
    from api.schemas.autointerp import HypothesisRequest
    from core.autointerp import scoring
    from core.autointerp.budget import SCORE_PM

    info, atlas, model, key, feature = await _feature_atlas(db, user, args)
    text = " ".join(_text(args, "hypothesis", "hypothesis", limit=1000).split())
    result = await score_hypothesis_for(db, user, HypothesisRequest(model=model, sae=_ref(key), corpus=atlas.corpus, feature=feature, text=text))
    h = result["hypothesis"]
    out = {
        "feature": feature, "link": feature_link(key, feature, model), "hypothesis_id": h.get("id"), "text": h.get("text"),
        "status": h.get("status"), "note": h.get("note"),
        "score": _round(h.get("score")), "score_pm": SCORE_PM if h.get("score") is not None else None,
        "detection": _round(h.get("detection")), "detection_hard": _round(h.get("detection_hard")), "fuzzing": _round(h.get("fuzzing")),
        "token_baseline": _round(h.get("token_detection")),
        "token_feature": scoring.token_feature(h.get("detection"), h.get("token_detection")),
        "control_percentile": _round(h.get("control_percentile"), 1), "already_scored": result.get("cached"),
        "compare": "get_feature gives the stored explanation's scores on the same held-out set",
    }
    return ToolOutcome(out, chip=f"Hypothesis scored · {feature}", result={"page": None, "labels": key, "data": None})


def _test_source(db: Session, user: User, atlas, feature: int, hypothesis_id: int):
    """What a generation test tests: one of the user's hypotheses of the feature (by ID), or (0) its newest explanation."""
    from services import autointerp

    if hypothesis_id:
        source = autointerp.hypothesis_by_id(db, user.id, hypothesis_id)
        if source is None or source.atlas_key != atlas.directory or source.feature != feature:
            raise HTTPException(status_code=404, detail="No hypothesis of the user's with that ID for this feature (get_feature lists them)")
        return source
    versions = autointerp.versions(db, atlas.directory, feature)
    if not versions or not versions[0].explanation:
        raise HTTPException(status_code=404, detail="The feature has no explanation yet (explain_feature writes one)")
    return versions[0]


async def quote_generation_test(db: Session, user: User, ctx: ToolContext, args: dict) -> RunQuote:
    from api.autointerp import _atlas_specs, _generation_test_price
    from core.autointerp.budget import DEFAULT_EXPLAINER
    from services import autointerp

    info, atlas, model, key, feature = await _feature_atlas(db, user, args)
    source = _test_source(db, user, atlas, feature, _int(args, "hypothesis_id", 0, None, "hypothesis_id"))
    tested = source.text if isinstance(source, FeatureHypothesis) else source.explanation
    details = _feature_details(info, atlas, model, feature) + [["Tests", _cut(tested)]]
    chip = f"Generation test {feature} · {_short(info.name)}"
    if autointerp._stored_test(db, atlas, source, DEFAULT_EXPLAINER) is not None:
        return RunQuote("Generation test", details, 0.0, note="Already tested: the stored test, free", chip=chip)
    specs, _, _ = await _atlas_specs(db, user, model, _ref(key), atlas.corpus)
    return RunQuote("Generation test", details, _generation_test_price(atlas.model_id, specs, atlas.width),
                    note="Claude writes 15 texts and one GPU call reads them; charged their usage", chip=chip)


async def run_generation_test(db: Session, user: User, ctx: ToolContext, args: dict) -> ToolOutcome:
    from api.autointerp import generation_test_for
    from api.schemas.autointerp import GenerationTestRequest

    info, atlas, model, key, feature = await _feature_atlas(db, user, args)
    source = _test_source(db, user, atlas, feature, _int(args, "hypothesis_id", 0, None, "hypothesis_id"))
    is_hypothesis = isinstance(source, FeatureHypothesis)
    body = GenerationTestRequest(
        model=model, sae=_ref(key), corpus=atlas.corpus, feature=feature,
        explanation_id=None if is_hypothesis else source.id, hypothesis_id=source.id if is_hypothesis else None,
    )
    result = await generation_test_for(db, user, body)
    t = result["test"]
    texts = t.get("texts") or []

    def text_of(entry: dict) -> str:
        return _cut("".join(entry.get("tokens") or []), 160)

    out = {
        "feature": feature, "link": feature_link(key, feature, model), "tested": t.get("text"),
        "firing_rate": _round(t.get("firing_rate")), "near_miss_rate": _round(t.get("near_miss_rate")),
        "threshold": _round(t.get("threshold")), "corpus_maximum": _round(t.get("maximum")),
        "firing_texts": [{"text": text_of(e), "peak": _round(e.get("peak")), "fired": e.get("fired")} for e in texts if e.get("kind") == "firing"][:10],
        "near_misses": [{"text": text_of(e), "peak": _round(e.get("peak")), "fired": e.get("fired")} for e in texts if e.get("kind") == "near_miss"][:5],
        "already_tested": result.get("cached"),
        "note": "firing_rate: share of firing texts reaching half the feature's corpus maximum; near_miss_rate: share of near-misses staying below",
    }
    return ToolOutcome(out, chip=f"Generation test · {feature}", result={"page": None, "labels": key, "data": None})


# Autointerp jobs and atlases

async def _job_request(db: Session, user: User, args: dict):
    from api.schemas.autointerp import JobRequest, JobSelection
    from core.autointerp.budget import MAX_JOB_FEATURES

    model, key = (args.get("model") or "").strip(), (args.get("sae") or "").strip()
    features = _int(args, "features", 1, MAX_JOB_FEATURES, "features")
    info, atlas = await _atlas(db, user, model, key)
    if atlas is None:
        raise HTTPException(status_code=404, detail="This SAE has no atlas on this model yet (build_atlas builds one)")
    return info, atlas, JobRequest(model=model, sae=_ref(key), corpus=atlas.corpus, selection=JobSelection(kind="top", n=features))


async def quote_start_autointerp_job(db: Session, user: User, ctx: ToolContext, args: dict) -> RunQuote:
    from api.autointerp import estimate_job_for
    from services import autointerp

    if not autointerp.available():
        raise HTTPException(status_code=503, detail="Autointerp isn't configured on this server")
    info, atlas, body = await _job_request(db, user, args)
    plan = await estimate_job_for(db, user, body)
    if not plan["features"] and not plan["controls"]:
        _bad("Every selected feature is already explained with these models")
    return RunQuote(
        "Autointerp job",
        [["SAE", info.name], ["Model", _model_name(body.model)], ["Corpus", atlas.corpus],
         ["Features", f"{plan['features']} most frequent" + (f" ({plan['skipped']} already explained, skipped)" if plan["skipped"] else "")],
         ["Controls", f"{plan['controls']} random directions"], ["Models", _models_note() + " · Batch API"]],
        plan["estimated_cost"], plan["hold"],
        note="Holds about its upper bound when it starts; charged its usage when it ends (most within an hour)",
        chip=f"Autointerp job · {_short(info.name)}",
    )


async def run_start_autointerp_job(db: Session, user: User, ctx: ToolContext, args: dict) -> ToolOutcome:
    from api.autointerp import submit_job_for

    info, atlas, body = await _job_request(db, user, args)
    job = await submit_job_for(db, user, body)
    out = {
        "job": job.get("job_id"), "status": job.get("status"), "sae_name": info.name, "corpus": atlas.corpus,
        "features": job.get("features"), "note": "Runs in the background; list_jobs shows its progress, and the SAE page's Autointerp view lists it",
    }
    return ToolOutcome(out, chip=f"Job started · {_short(info.name)}")


def _atlas_request(args: dict):
    from api.schemas.geometry import AtlasJobRequest

    model, key = (args.get("model") or "").strip(), (args.get("sae") or "").strip()
    tokens = _int(args, "tokens", 1, None, "tokens")
    if tokens not in ATLAS_TOKENS:
        _bad("tokens is 1000000, 2000000 or 5000000")
    return AtlasJobRequest(model=model, sae=_ref(key), tokens=tokens)


async def quote_build_atlas(db: Session, user: User, ctx: ToolContext, args: dict) -> RunQuote:
    from api.geometry import estimate_atlas
    from core.geometry.atlas import CORPORA

    body = _atlas_request(args)
    info, existing = await _atlas(db, user, body.model, args.get("sae", ""))
    estimate = await estimate_atlas(body=body, current_user=user, db=db)
    if not estimate["within_limit"]:
        _bad("This atlas would run past the job's time limit; read fewer tokens")
    if not estimate["corpus_available"]:
        _bad(f"{CORPORA[body.corpus]['name']} isn't available on this server")
    details = [["SAE", info.name], ["Model", _model_name(body.model)], ["Corpus", f"{CORPORA[body.corpus]['name']}, {body.tokens:,} tokens"],
               ["Time", f"about {round(estimate['estimated_seconds'] / 60) or 1} min"]]
    if existing is not None and existing.corpus == body.corpus:
        details.append(["Replaces", f"its atlas from {str(existing.meta.get('created_at', ''))[:10]}"])
    return RunQuote("Build an atlas", details, estimate["estimated_cost"],
                    note="Charged its estimate when it starts; a failed job is refunded all but the GPU time it used",
                    chip=f"Atlas job · {_short(info.name)}")


async def run_build_atlas(db: Session, user: User, ctx: ToolContext, args: dict) -> ToolOutcome:
    from api.geometry import submit_atlas_job_for

    body = _atlas_request(args)
    job = await submit_atlas_job_for(db, user, body)
    out = {
        "job": job.get("job_id"), "status": job.get("status"), "sae_name": job.get("sae_name"), "tokens": job.get("tokens"),
        "minutes": round((job.get("estimated_seconds") or 0) / 60, 1),
        "note": "Runs in the background; list_jobs shows its progress, and the Geometry Lab's atlas view shows it when done",
    }
    return ToolOutcome(out, chip=f"Atlas job started · {_short(job.get('sae_name') or '')}")


# Probe Lab

def _probe_request(db: Session, user: User, args: dict):
    from api.schemas.probes import ProbeDatasetSpec, TrainProbesRequest
    from core.probes.datasets import builtin_datasets

    model = _check_model(db, user, args.get("model", ""))
    dataset = (args.get("dataset") or "").strip()
    if dataset not in builtin_datasets():
        raise HTTPException(status_code=404, detail="Unknown built-in dataset (the guide lists them)")
    fraction = round(_float(args, "test_fraction", 0.1, 0.4, "test_fraction"), 2)
    if fraction not in TEST_FRACTIONS:
        _bad("test_fraction is 0.1, 0.2, 0.3 or 0.4")
    return TrainProbesRequest(
        model=model, dataset=ProbeDatasetSpec(builtin_id=dataset, use_builtin_ood=True), chat_template=bool(args.get("chat_template")),
        read_span="text", test_fraction=fraction, seed=_int(args, "seed", 0, 2**31 - 1, "seed"),
    )


async def quote_train_probe(db: Session, user: User, ctx: ToolContext, args: dict) -> RunQuote:
    from api.probes import estimate_run
    from api.schemas.probes import EstimateProbesRequest
    from core.probes.datasets import builtin_datasets, dataset_summary

    body = _probe_request(db, user, args)
    summary = dataset_summary(builtin_datasets()[body.dataset.builtin_id])
    ood = summary.get("ood") or {}
    examples = summary["num_rows"] + (ood.get("num_rows") or 0)
    estimate = await estimate_run(
        body=EstimateProbesRequest(model=body.model, num_examples=examples, total_chars=summary["total_chars"] + (ood.get("total_chars") or 0),
                                   chat_template=body.chat_template),
        current_user=user, db=db,
    )
    if not estimate["within_limit"]:
        _bad("This run would take longer than the model's time limit")
    return RunQuote(
        "Train probes",
        [["Model", _model_name(body.model)], ["Dataset", f"{summary['name']} ({examples} examples)"],
         ["Split", f"{round(body.test_fraction * 100)}% held out, seed {body.seed}"], ["Chat template", "yes" if body.chat_template else "no"]],
        estimate["estimated_cost"], note="Charged by its size when it ends (token counts are estimated until then)",
        chip=f"Probes · {summary['name']}",
    )


def _sweep_summary(run: dict) -> dict:
    def at(values, i):
        return _round(values[i]) if values and i < len(values) and values[i] is not None else None

    best = []
    for pooling, entry in (run.get("poolings") or {}).items():
        for method, data in (entry.get("methods") or {}).items():
            metrics = data.get("metrics") or {}
            auroc = metrics.get("test_auroc") or []
            layer = max(range(1, len(auroc)), key=lambda i: auroc[i] if auroc[i] is not None else -1, default=None)
            if layer is None:
                continue
            best.append({
                "pooling": pooling, "method": method, "best_layer": layer, "test_auroc": at(auroc, layer),
                "control_auroc": at(metrics.get("control_test_auroc"), layer), "embeddings_auroc": at(auroc, 0),
                "test_acc": at(metrics.get("test_acc"), layer), "selectivity": at(metrics.get("selectivity"), layer),
                "ood_auroc": at(metrics.get("ood_auroc"), layer),
            })
    best.sort(key=lambda b: -(b["test_auroc"] or 0))
    dataset = run.get("dataset") or {}
    return {
        "model": run.get("model"), "layers": run.get("num_layers"), "dataset": dataset.get("name"),
        "labels": dataset.get("label_names"), "examples": {"train": dataset.get("n_train"), "test": dataset.get("n_test"), "ood": dataset.get("n_ood")},
        "ood_set": dataset.get("ood_name"), "majority_test_acc": _round((run.get("majority") or {}).get("test_acc")),
        "best_by_probe": best,
        "note": "Layer 0 is the embeddings (their AUROC is the tokens-alone baseline); control_auroc is the shuffled-label control task. Opened in the Probe Lab",
    }


async def run_train_probe(db: Session, user: User, ctx: ToolContext, args: dict) -> ToolOutcome:
    from api.probes import train_probes_for
    from api.schemas.probes import ProbeRunResponse

    body = _probe_request(db, user, args)
    run = ProbeRunResponse.model_validate(await train_probes_for(db, user, body)).model_dump(mode="json")
    return ToolOutcome(_sweep_summary(run), chip=f"Probes trained · {run['dataset']['name']}", result={
        "page": "probes", "data": {"run": run, "test_fraction": body.test_fraction, "dataset": body.dataset.builtin_id},
    })


# Probe Lab: a saved probe against an SAE

def _compare_request(db: Session, user: User, args: dict):
    """(the saved probe, the SAE's key and name, the request, the dataset's summary) for compare_probe_sae, on the probe's own data and split."""
    from api.schemas.probes import ProbeDatasetSpec, SaeProbeRequest
    from core.probes.datasets import builtin_datasets, dataset_summary
    from models.database import Probe

    probe_id = _int(args, "probe_id", 1, None, "probe_id")
    probe = db.query(Probe).filter(Probe.id == probe_id, Probe.user_id == user.id).first()
    if probe is None:
        raise HTTPException(status_code=404, detail="No saved probe of the user's with that ID (list_probes gives them)")
    dataset = probe.dataset or {}
    builtin = dataset.get("builtin_id")
    if not builtin or builtin not in builtin_datasets():
        _bad("This probe was trained on uploaded rows: compare it from the Probe Lab's SAE view")
    key, ref, name, _ = _sae_target(db, user, args)
    summary = dataset_summary(builtin_datasets()[builtin])
    with_ood = bool(dataset.get("n_ood"))
    fraction = min(0.5, max(0.1, float(dataset.get("test_fraction") or 0.2)))
    body = SaeProbeRequest(
        probe_id=probe.id, sae=_parse_sae(key)[1], dataset=ProbeDatasetSpec(builtin_id=builtin, use_builtin_ood=with_ood),
        test_fraction=fraction, seed=int(dataset.get("seed") or 0),
    )
    return probe, key, name, body, summary, with_ood


async def quote_compare_probe_sae(db: Session, user: User, ctx: ToolContext, args: dict) -> RunQuote:
    from api.probes import _model_specs, _resolve_sae, estimate_run
    from api.schemas.probes import EstimateProbesRequest

    probe, key, name, body, summary, with_ood = _compare_request(db, user, args)
    # The route's own check: the SAE reads the probe's model at the probe's layer
    _resolve_sae(db, user.id, probe.model_id, await _model_specs(db, user.id, probe.model_id), body.sae, probe.layer)
    ood = (summary.get("ood") or {}) if with_ood else {}
    examples = summary["num_rows"] + (ood.get("num_rows") or 0)
    estimate = await estimate_run(
        body=EstimateProbesRequest(model=probe.model_id, num_examples=examples, total_chars=summary["total_chars"] + (ood.get("total_chars") or 0),
                                   chat_template=probe.chat_template, kind="sae", sae=body.sae),
        current_user=user, db=db,
    )
    if not estimate["within_limit"]:
        _bad("This comparison would take longer than the model's time limit")
    return RunQuote(
        "Compare a probe with an SAE",
        [["Probe", f"{probe.name} · layer {probe.layer}, {probe.pooling}, {probe.method}"], ["Model", _model_name(probe.model_id)], ["SAE", name],
         ["Dataset", f"{summary['name']} ({examples} examples)"], ["Split", f"{round(body.test_fraction * 100)}% held out, seed {body.seed}"]],
        estimate["estimated_cost"], note="Charged by its size when it ends", chip=f"Probe vs {_short(name)}",
    )


def _comparison_summary(r: dict, key: str, probe) -> dict:
    alignment = r.get("alignment") or {}
    contributions = alignment.get("contributions") or {}

    def feature(f: dict) -> dict:
        return {"feature": f["feature"], "link": feature_link(key, f["feature"], r.get("model")), "cos": _round(f.get("cos")),
                "contribution": _round(f.get("contribution")), "mean_pos": _round(f.get("mean_pos")), "mean_neg": _round(f.get("mean_neg"))}

    controls = alignment.get("control_max_cos") or []
    sparse = r.get("sparse")
    sparse_probes = None
    if sparse:
        metrics = sparse.get("metrics") or {}

        def at(name: str, i: int):
            values = metrics.get(name) or []
            return _round(values[i]) if i < len(values) else None

        sparse_probes = [{"k": k, "test_auroc": at("test_auroc", i), "ood_auroc": at("ood_auroc", i)} for i, k in enumerate(sparse.get("ks") or [])]
    sae = r.get("sae") or {}
    dataset = r.get("dataset") or {}
    return {
        "probe": {"id": probe.id, "name": probe.name, "layer": r.get("layer"), "pooling": r.get("pooling"), "method": r.get("method")},
        "sae": key, "sae_name": (sae.get("info") or {}).get("name"), "width": (sae.get("info") or {}).get("width"),
        "l0": _round(sae.get("l0"), 1), "fvu": _round(sae.get("fvu")),
        "nearest": [feature(f) for f in (alignment.get("features") or [])[:8]],
        "random_max_cos": _round(alignment.get("random_max_cos")),
        "shuffled_max_cos": _round(sum(controls) / len(controls)) if controls else None,
        "captured": [{"k": c["k"], "fraction": _round(c["fraction"])} for c in (alignment.get("captured") or [])[:8]],
        "by_contribution": [feature(f) for f in (contributions.get("features") or [])[:8]],
        "class_gap": _round(contributions.get("gap")), "reconstructed_gap": _round(contributions.get("reconstructed")),
        "sparse_probes": sparse_probes, "dense_probe": {k: _round(v) for k, v in (r.get("dense") or {}).items()},
        "examples": {"train": dataset.get("n_train"), "test": dataset.get("n_test"), "ood": dataset.get("n_ood")},
        "note": "cos: |cosine| of a feature's decoder direction with the probe's; random_max_cos and shuffled_max_cos are the nearest feature's for "
                "random and shuffled-label directions. captured: share of the direction in the span of its k nearest features. contribution: the "
                "feature's part of the probe's class gap. Sparse probes use the k features that differ most between the classes, on the dense probe's split",
    }


async def run_compare_probe_sae(db: Session, user: User, ctx: ToolContext, args: dict) -> ToolOutcome:
    from api.probes import sae_probe_for
    from api.schemas.probes import SaeProbeResponse

    probe, key, name, body, _, _ = _compare_request(db, user, args)
    result = SaeProbeResponse.model_validate(await sae_probe_for(db, user, body)).model_dump(mode="json")
    return ToolOutcome(_comparison_summary(result, key, probe), chip=f"Probe vs {_short(name)}")


# Geometry Lab

def _geometry_request(db: Session, user: User, args: dict):
    from api.schemas.geometry import ConceptSpec, GeometryRunRequest
    from core.geometry.concepts import builtin_concepts

    model = _check_model(db, user, args.get("model", ""))
    concept = (args.get("concept") or "").strip()
    if concept not in builtin_concepts():
        raise HTTPException(status_code=404, detail="Unknown built-in concept (the guide lists them)")
    read = args.get("read")
    if read not in GEOMETRY_READS:
        _bad(f"read is one of: {', '.join(GEOMETRY_READS)}")
    return GeometryRunRequest(model=model, concept=ConceptSpec(builtin_id=concept), read=read,
                              chat_template=bool(args.get("chat_template")), seed=_int(args, "seed", 0, 2**31 - 1, "seed"))


async def quote_run_geometry(db: Session, user: User, ctx: ToolContext, args: dict) -> RunQuote:
    from api.geometry import estimate_run
    from api.schemas.geometry import EstimateGeometryRequest
    from core.geometry.concepts import builtin_concepts

    body = _geometry_request(db, user, args)
    estimate = await estimate_run(body=EstimateGeometryRequest(model=body.model, concept=body.concept, chat_template=body.chat_template),
                                  current_user=user, db=db)
    if not estimate["within_limit"]:
        _bad("This run would take longer than the model's time limit")
    name = builtin_concepts()[body.concept.builtin_id]["name"]
    return RunQuote(
        "Concept geometry",
        [["Model", _model_name(body.model)], ["Concept", f"{name} ({estimate['num_examples']} examples)"], ["Read at", body.read],
         ["Chat template", "yes" if body.chat_template else "no"], ["Seed", str(body.seed)]],
        estimate["estimated_cost"], note="Charged by its size when it ends", chip=f"Geometry · {name}",
    )


def _geometry_summary(run: dict) -> dict:
    shapes = run.get("shapes") or {}
    tested = shapes.get("dims") or []
    dims = 3 if 3 in tested else (tested[-1] if tested else None)
    d = tested.index(dims) if dims in tested else 0
    layers = run.get("num_layers") or 0
    hypotheses = []
    for h in shapes.get("hypotheses") or []:
        r2 = (h.get("r2") or [[]])[d] if h.get("r2") else []
        shuffled = (h.get("shuffled_max") or [[]])[d] if h.get("shuffled_max") else []
        margins = [(p, r2[p] - shuffled[p]) for p in range(1, min(len(r2), len(shuffled), layers + 1)) if r2[p] is not None and shuffled[p] is not None]
        if not margins:
            continue
        best = max(margins, key=lambda m: m[1])[0]
        hypotheses.append({
            "hypothesis": h.get("label"), "best_layer": best, "held_out_r2": _round(r2[best]), "shuffled_max_r2": _round(shuffled[best]),
            "embeddings_r2": _round(r2[0]) if r2 and r2[0] is not None else None,
        })
    hypotheses.sort(key=lambda h: -((h["held_out_r2"] or 0) - (h["shuffled_max_r2"] or 0)))
    best_layer = hypotheses[0]["best_layer"] if hypotheses else None

    def at(values, i):
        return _round(values[i], 2) if values and i is not None and i < len(values) and values[i] is not None else None

    concept = run.get("concept") or {}
    return {
        "model": run.get("model"), "concept": concept.get("name"), "values": len(concept.get("values") or []), "read": run.get("read"),
        "layers": layers, "subspace_dims": dims, "hypotheses": hypotheses,
        "at_best_layer": {"layer": best_layer, "participation_ratio": at(run.get("participation_ratio"), best_layer),
                          "components_90": (run.get("components_90") or [None] * (layers + 1))[best_layer] if best_layer is not None else None,
                          "neighbor_share": at(run.get("neighbors"), best_layer), "neighbor_chance": _round(run.get("neighbor_chance"), 3)},
        "note": "A hypothesis fits only where its held-out R² beats the max over shuffled value orders. Opened in the Geometry Lab",
    }


async def run_run_geometry(db: Session, user: User, ctx: ToolContext, args: dict) -> ToolOutcome:
    from api.geometry import run_geometry_for
    from api.schemas.geometry import GeometryRunResponse

    body = _geometry_request(db, user, args)
    run = GeometryRunResponse.model_validate(await run_geometry_for(db, user, body)).model_dump(mode="json")
    return ToolOutcome(_geometry_summary(run), chip=f"Geometry · {run['concept']['name']}", result={"page": "geometry", "data": {"run": run}})


# Activation Patching

async def _patching_request(db: Session, user: User, args: dict):
    from api.patching import _check_patches
    from api.schemas.patching import PatchSpec, RunPatchingRequest

    model = _check_model(db, user, args.get("model", ""))
    patches, problem = check_patches(args.get("patches"))
    if problem:
        _bad(problem)
    source = _text(args, "source_prompt", "source prompt", required=False)
    if any(p["intervention_type"] == "patch" for p in patches) and not source.strip():
        _bad("A patch intervention copies the source prompt's activations: give a source prompt, or use an ablation")
    specs = [PatchSpec(**p) for p in patches]
    await _check_patches(db, user.id, model, specs)
    return RunPatchingRequest(
        source_prompt=source if source.strip() else None, target_prompt=_text(args, "target_prompt", "target prompt"),
        model=model, patches=specs, max_tokens=_int(args, "max_tokens", 1, MAX_PATCHING_TOKENS, "max_tokens"),
        temperature=_float(args, "temperature", 0, 2, "temperature"),
    )


def _patch_text(p) -> str:
    where = f"L{p.layer} {p.patch_type}"
    if p.heads:
        where += f" heads {','.join(map(str, p.heads))}"
    if p.positions:
        where += f" positions {','.join(map(str, p.positions))}"
    blend = "" if p.blend_factor == 1 else f" ×{p.blend_factor:g}"
    return f"{where}: {p.intervention_type.replace('_', ' ')}{blend}"


async def quote_run_patching(db: Session, user: User, ctx: ToolContext, args: dict) -> RunQuote:
    from services.credit_service import estimate_patching_cost

    body = await _patching_request(db, user, args)
    estimate = estimate_patching_cost(
        source_prompt_length=len(body.source_prompt or ""), target_prompt_length=len(body.target_prompt),
        max_tokens=body.max_tokens, model_id=body.model, num_patches=len(body.patches),
    )
    details = [["Model", _model_name(body.model)]]
    if body.source_prompt:
        details.append(["Source", _cut(body.source_prompt)])
    details += [["Target", _cut(body.target_prompt)], ["Patches", "; ".join(_patch_text(p) for p in body.patches[:6]) + (" …" if len(body.patches) > 6 else "")],
                ["Tokens", f"{body.max_tokens}, temperature {body.temperature:g}"]]
    return RunQuote("Activation patching", details, estimate, note="Charged on the tokens it generates", chip=f"Patching · {_model_name(body.model)}")


async def run_run_patching(db: Session, user: User, ctx: ToolContext, args: dict) -> ToolOutcome:
    from api.patching import run_patching_experiment

    body = await _patching_request(db, user, args)
    result = (await run_patching_experiment(request=body, current_user=user, db=db)).model_dump(mode="json")
    effect = result.get("effect_summary") or {}
    out = {
        "model": body.model, "baseline_output": _cut(result.get("baseline_output"), 800), "patched_output": _cut(result.get("patched_output"), 800),
        "first_divergence_position": effect.get("first_divergence_position"), "token_changes": effect.get("token_changes"),
        "baseline_length": effect.get("baseline_length"), "patched_length": effect.get("patched_length"),
        "changed_tokens": (effect.get("changed_tokens") or [])[:10],
        "note": "Outputs are generated continuations of the target prompt. Opened on the Activation Patching page",
    }
    return ToolOutcome(out, chip=f"Patched · {_model_name(body.model)}", result={
        "page": "patching", "data": {"request": body.model_dump(mode="json"), "result": result},
    })


# Attention Analysis

async def _heads_target(db: Session, user: User, args: dict):
    """(model, prompt or None, the captured activations to reuse or None, specs)."""
    from core.patching import get_patching_engine

    activation_id = (args.get("activation_id") or "").strip()
    model = (args.get("model") or "").strip()
    if activation_id:
        captured = get_patching_engine().get_captured_activations(activation_id, user.id)
        if captured is None:
            raise HTTPException(status_code=404, detail="No capture with that ID (captures are kept in memory for a while)")
        if model and captured.model_id != model:
            _bad(f"That capture is from {captured.model_id}")
        return captured.model_id, None, captured, None
    model = _check_model(db, user, model)
    prompt = _text(args, "prompt", "prompt", limit=4000)
    return model, prompt, None, await _specs(db, user, model)


def _prompt_tokens(model: str, prompt: str) -> int:
    from core.open_source import OPEN_SOURCE_MODELS

    if model in OPEN_SOURCE_MODELS:
        return max(1, len(prompt) // 4)
    from core.tokenizer import get_tokenizer

    return max(1, len(get_tokenizer().encode(prompt)))


async def quote_classify_heads(db: Session, user: User, ctx: ToolContext, args: dict) -> RunQuote:
    from api.analysis import _num_heads
    from services.credit_service import estimate_analysis_cost, min_gpu_request_charge

    model, prompt, captured, specs = await _heads_target(db, user, args)
    if captured is not None:
        cost = estimate_analysis_cost(analysis_type="classify", num_layers=captured.num_layers, num_heads=_num_heads(captured),
                                      num_tokens=len(captured.tokens), model_id=model)
        return RunQuote("Classify attention heads", [["Model", _model_name(model)], ["Capture", _cut(captured.prompt)]],
                        cost, note="Charged on the CPU time it takes", chip=f"Classify heads · {_model_name(model)}")
    capture = min_gpu_request_charge(model)
    cost = estimate_analysis_cost(analysis_type="classify", num_layers=specs.layers or 1, num_heads=specs.heads or 1,
                                  num_tokens=_prompt_tokens(model, prompt), model_id=model)
    return RunQuote(
        "Capture and classify attention heads",
        [["Model", _model_name(model)], ["Prompt", _cut(prompt)], ["Charges", f"capture {_money(capture)} + classification {_money(cost)}"]],
        capture + cost, note="Two charges: the capture (one GPU request) and the classification", chip=f"Classify heads · {_model_name(model)}",
    )


def _heads_summary(result: dict) -> dict:
    by_type: dict[str, list] = {}
    for c in result.get("classifications") or []:
        by_type.setdefault(c["primary_type"], []).append(c)
    top = {
        kind: [{"layer": c["layer"], "head": c["head"], "confidence": _round(c.get("confidence")), "score": _round((c.get("scores") or {}).get(kind))}
               for c in sorted(items, key=lambda c: -(c.get("confidence") or 0))[:HEADS_PER_TYPE]]
        for kind, items in by_type.items() if kind not in ("mixed", "distributed")
    }
    return {
        "model": result.get("model_id"), "activation_id": result.get("activation_id"), "prompt": _cut(result.get("prompt"), 300),
        "tokens": len(result.get("tokens") or []), "layers": result.get("num_layers"), "heads_per_layer": result.get("num_heads"),
        "counts": {k: v for k, v in (result.get("summary") or {}).items() if v},
        "top_heads": top,
        "note": "Types come from thresholds on attention-pattern scores over this one prompt; a repeated sequence in the prompt is what shows induction heads. Opened on the Attention Analysis page",
    }


async def run_classify_heads(db: Session, user: User, ctx: ToolContext, args: dict) -> ToolOutcome:
    from api.analysis import ClassifyHeadsRequest, classify_heads
    from api.patching import capture_activations
    from api.schemas.patching import CaptureActivationsRequest

    model, prompt, captured, _ = await _heads_target(db, user, args)
    if captured is None:
        capture = await capture_activations(request=CaptureActivationsRequest(prompt=prompt, model=model), current_user=user, db=db)
        activation_id = capture.activation_id
    else:
        activation_id = captured.id
    response = await asyncio.to_thread(classify_heads, request=ClassifyHeadsRequest(activation_id=activation_id), current_user=user, db=db)
    result = response.model_dump(mode="json")
    return ToolOutcome(_heads_summary(result), chip=f"Heads classified · {_model_name(model)}", result={
        "page": "analysis", "data": {"model": model, "activation_id": activation_id, "result": result},
    })


Quote = Callable[[Session, User, ToolContext, dict], Awaitable[RunQuote]]
Run = Callable[[Session, User, ToolContext, dict], Awaitable[ToolOutcome]]

RUNS: dict[str, tuple[Quote, Run]] = {
    "analyze_sae": (quote_analyze_sae, run_analyze_sae),
    "build_atlas": (quote_build_atlas, run_build_atlas),
    "classify_heads": (quote_classify_heads, run_classify_heads),
    "compare_probe_sae": (quote_compare_probe_sae, run_compare_probe_sae),
    "explain_feature": (quote_explain_feature, run_explain_feature),
    "generate": (quote_generate, run_generate),
    "generation_test": (quote_generation_test, run_generation_test),
    "run_geometry": (quote_run_geometry, run_run_geometry),
    "run_patching": (quote_run_patching, run_run_patching),
    "start_autointerp_job": (quote_start_autointerp_job, run_start_autointerp_job),
    "test_hypothesis": (quote_test_hypothesis, run_test_hypothesis),
    "train_probe": (quote_train_probe, run_train_probe),
}


def _error(e: HTTPException) -> ToolOutcome:
    detail = str(e.detail)
    if e.status_code == 402 or "INSUFFICIENT_CREDITS" in detail or "Insufficient credits" in detail:
        detail = "The user's balance doesn't cover this run"
    return ToolOutcome({"error": detail}, error=True)


async def quote(name: str, ctx: ToolContext, args: dict) -> "RunQuote | ToolOutcome":
    """A run's price for the approval card, or the error that keeps it from running (nothing runs or is charged)."""
    if name not in RUNS:
        return ToolOutcome({"error": f"There's no run tool named {name}"}, error=True)
    db = SessionLocal()
    try:
        user = db.get(User, ctx.user_id)
        if user is None:
            return ToolOutcome({"error": "Not signed in"}, error=True)
        return await RUNS[name][0](db, user, ctx, args)
    except HTTPException as e:
        return _error(e)
    except (ValueError, TypeError) as e:
        logger.warning("Assistant run %s got bad input: %s", name, e)
        return ToolOutcome({"error": "Bad input to this run"}, error=True)
    finally:
        db.close()


async def run(name: str, ctx: ToolContext, args: dict) -> ToolOutcome:
    """Run an approved run as the user, through its page's route function; reports what its service charged."""
    from services.credit_service import recording_charges

    if name not in RUNS:
        return ToolOutcome({"error": f"There's no run tool named {name}"}, error=True)
    db = SessionLocal()
    with recording_charges() as charges:
        try:
            user = db.get(User, ctx.user_id)
            if user is None:
                outcome = ToolOutcome({"error": "Not signed in"}, error=True)
            else:
                outcome = await RUNS[name][1](db, user, ctx, args)
        except HTTPException as e:
            outcome = _error(e)
        except (ValueError, TypeError) as e:
            logger.warning("Assistant run %s got bad input: %s", name, e)
            outcome = ToolOutcome({"error": "Bad input to this run"}, error=True)
        finally:
            db.close()
    charged = round(-sum(charges), 4)
    outcome.charged = charged
    if isinstance(outcome.content, dict):
        outcome.content = {**outcome.content, "charged_usd": charged}
    return outcome
