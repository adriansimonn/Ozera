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
from core.assistant.pages import PAGES, check_labels, check_patches, check_rows
from core.assistant.tools import ATLAS_CORPORA, ATLAS_TOKENS, GEOMETRY_READS, METHODS, POOLINGS, STRUCTURES
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
    corpus = (args.get("corpus") or "openwebtext").strip()
    if corpus not in ATLAS_CORPORA:
        _bad(f"corpus is one of: {', '.join(ATLAS_CORPORA)}")
    return AtlasJobRequest(model=model, sae=_ref(key), corpus=corpus, tokens=tokens)


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
        "corpus": body.corpus,
        "note": "Runs in the background; list_jobs shows its progress, and the Geometry Lab's atlas view shows it when done",
    }
    return ToolOutcome(out, chip=f"Atlas job started · {_short(job.get('sae_name') or '')}", result={
        "page": "geometry", "data": {"model": body.model, "sae": args.get("sae", "").strip(), "corpus": body.corpus, "tokens": body.tokens},
    })


# Probe Lab

def _dataset_spec(args: dict, required: bool = True, use_ood: bool = True):
    """A run's dataset: a built-in one (args["dataset"]) or the user's rows, checked as the Probe Lab checks them; None if neither and not required."""
    from api.schemas.probes import ProbeDatasetSpec
    from core.probes.datasets import builtin_datasets

    dataset = (args.get("dataset") or "").strip()
    rows = args.get("rows") or []
    if dataset and rows:
        _bad("Give a built-in dataset or rows, not both")
    if dataset:
        if dataset not in builtin_datasets():
            raise HTTPException(status_code=404, detail="Unknown built-in dataset (the guide lists them)")
        return ProbeDatasetSpec(builtin_id=dataset, use_builtin_ood=use_ood)
    if not rows:
        if required:
            _bad("Give a built-in dataset ID, or rows of your own")
        return None
    checked, problem = check_rows(rows)
    if problem:
        _bad(problem)
    names = None
    if args.get("label_names"):
        names, problem = check_labels(args["label_names"])
        if problem:
            _bad(problem)
    name = " ".join(str(args.get("dataset_name") or "").split())[:255] or "Your rows"
    return ProbeDatasetSpec(rows=checked, name=name, label_names=names)


def _dataset_size(spec) -> tuple[int, int, str]:
    """(examples, characters, name) of a run's dataset, any OOD set included."""
    from core.probes.datasets import builtin_datasets, dataset_summary

    if spec.builtin_id:
        summary = dataset_summary(builtin_datasets()[spec.builtin_id])
        ood = (summary.get("ood") or {}) if spec.use_builtin_ood else {}
        return summary["num_rows"] + (ood.get("num_rows") or 0), summary["total_chars"] + (ood.get("total_chars") or 0), summary["name"]
    return len(spec.rows), sum(len(r.text) for r in spec.rows), spec.name or "Your rows"


def _rows_data(spec) -> Optional[dict]:
    """The user's rows as the Probe Lab's upload shows them (None for a built-in dataset)."""
    if spec.builtin_id:
        return None
    return {"filename": spec.name, "rows": [r.model_dump() for r in spec.rows], "label_names": spec.label_names or ["0", "1"], "warnings": []}


def _probe_request(db: Session, user: User, args: dict):
    from api.schemas.probes import TrainProbesRequest

    model = _check_model(db, user, args.get("model", ""))
    dataset = _dataset_spec(args)
    fraction = round(_float(args, "test_fraction", 0.1, 0.4, "test_fraction"), 2)
    if fraction not in TEST_FRACTIONS:
        _bad("test_fraction is 0.1, 0.2, 0.3 or 0.4")
    return TrainProbesRequest(
        model=model, dataset=dataset, chat_template=bool(args.get("chat_template")),
        read_span="text", test_fraction=fraction, seed=_int(args, "seed", 0, 2**31 - 1, "seed"),
    )


async def quote_train_probe(db: Session, user: User, ctx: ToolContext, args: dict) -> RunQuote:
    from api.probes import estimate_run
    from api.schemas.probes import EstimateProbesRequest

    body = _probe_request(db, user, args)
    examples, chars, name = _dataset_size(body.dataset)
    estimate = await estimate_run(
        body=EstimateProbesRequest(model=body.model, num_examples=examples, total_chars=chars, chat_template=body.chat_template),
        current_user=user, db=db,
    )
    if not estimate["within_limit"]:
        _bad("This run would take longer than the model's time limit")
    return RunQuote(
        "Train probes",
        [["Model", _model_name(body.model)], ["Dataset", f"{name} ({examples} examples)"],
         ["Split", f"{round(body.test_fraction * 100)}% held out, seed {body.seed}"], ["Chat template", "yes" if body.chat_template else "no"]],
        estimate["estimated_cost"], note="Charged by its size when it ends (token counts are estimated until then)",
        chip=f"Probes · {name}",
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
        "run_id": run.get("run_id"),
        "note": "Layer 0 is the embeddings (their AUROC is the tokens-alone baseline); control_auroc is the shuffled-label control task. "
                "save_probe saves one of its probes by run_id and layer. Opened in the Probe Lab",
    }


async def run_train_probe(db: Session, user: User, ctx: ToolContext, args: dict) -> ToolOutcome:
    from api.probes import train_probes_for
    from api.schemas.probes import ProbeRunResponse

    body = _probe_request(db, user, args)
    run = ProbeRunResponse.model_validate(await train_probes_for(db, user, body)).model_dump(mode="json")
    return ToolOutcome(_sweep_summary(run), chip=f"Probes trained · {run['dataset']['name']}", result={
        "page": "probes",
        "data": {"run": run, "test_fraction": body.test_fraction, "dataset": body.dataset.builtin_id, "upload": _rows_data(body.dataset)},
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
    summary = {**_comparison_summary(result, key, probe), "opened": "the Probe Lab's SAE view"}
    return ToolOutcome(summary, chip=f"Probe vs {_short(name)}", result={
        "page": "probes", "data": {"probe_id": probe.id, "sae": key, "response": result},
    })


# Geometry Lab

def _concept_spec(args: dict):
    """A run's concept: a built-in one (args["concept"]) or the user's own (args["custom"])."""
    from api.schemas.geometry import ConceptSpec, CustomConcept
    from core.geometry.concepts import builtin_concepts

    concept = (args.get("concept") or "").strip()
    custom = args.get("custom") if isinstance(args.get("custom"), dict) else {}
    values = [str(v).strip() for v in custom.get("values") or []]
    if concept:
        if values:
            _bad("Give a built-in concept or your own, not both")
        if concept not in builtin_concepts():
            raise HTTPException(status_code=404, detail="Unknown built-in concept (the guide lists them)")
        return ConceptSpec(builtin_id=concept)
    if not values:
        _bad("Give a built-in concept ID, or your own concept's values and templates")
    structure = custom.get("structure")
    if structure not in STRUCTURES:
        _bad(f"custom.structure is one of: {', '.join(STRUCTURES)}")
    groups = [str(g).strip() for g in custom.get("groups") or []]
    if groups and len(groups) != len(values):
        _bad(f"custom.groups has one group per value ({len(values)}), or none")
    try:
        return ConceptSpec(custom=CustomConcept(
            name=" ".join(str(custom.get("name") or "").split())[:100] or "Custom concept", structure=structure, values=values,
            templates=[str(t).strip() for t in custom.get("templates") or []],
            eval_templates=[str(t).strip() for t in custom.get("eval_templates") or []][:4],
            groups=groups or None,
        ))
    except ValueError as e:
        _bad(f"custom: {e}")


def _geometry_request(db: Session, user: User, args: dict):
    from api.geometry import _concept
    from api.schemas.geometry import GeometryRunRequest

    model = _check_model(db, user, args.get("model", ""))
    concept = _concept_spec(args)
    # The route's own checks of a custom concept (values, templates, sizes), before anything is priced
    _concept(concept)
    read = args.get("read")
    if read not in GEOMETRY_READS:
        _bad(f"read is one of: {', '.join(GEOMETRY_READS)}")
    return GeometryRunRequest(model=model, concept=concept, read=read,
                              chat_template=bool(args.get("chat_template")), seed=_int(args, "seed", 0, 2**31 - 1, "seed"))


def _concept_name(spec) -> str:
    from core.geometry.concepts import builtin_concepts

    return builtin_concepts()[spec.builtin_id]["name"] if spec.builtin_id else spec.custom.name


async def quote_run_geometry(db: Session, user: User, ctx: ToolContext, args: dict) -> RunQuote:
    from api.geometry import estimate_run
    from api.schemas.geometry import EstimateGeometryRequest

    body = _geometry_request(db, user, args)
    estimate = await estimate_run(body=EstimateGeometryRequest(model=body.model, concept=body.concept, chat_template=body.chat_template),
                                  current_user=user, db=db)
    if not estimate["within_limit"]:
        _bad("This run would take longer than the model's time limit")
    name = _concept_name(body.concept)
    details = [["Model", _model_name(body.model)], ["Concept", f"{name} ({estimate['num_examples']} examples)"]]
    if body.concept.custom:
        c = body.concept.custom
        details.append(["Values", _cut(", ".join(c.values), 160) + f" ({c.structure})"])
        details.append(["Templates", f"{len(c.templates)}, e.g. {_cut(c.templates[0], 100)}" if c.templates else "none"])
    details += [["Read at", body.read], ["Chat template", "yes" if body.chat_template else "no"], ["Seed", str(body.seed)]]
    return RunQuote("Concept geometry", details, estimate["estimated_cost"], note="Charged by its size when it ends", chip=f"Geometry · {name}")


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
    if best_layer is None:
        # No shape test applies (a categorical concept without groups): the layer whose variance is most between values
        between = run.get("between_fraction") or []
        scored = [(p, v) for p, v in enumerate(between) if p >= 1 and v is not None]
        best_layer = max(scored, key=lambda pv: pv[1])[0] if scored else None

    def at(values, i):
        return _round(values[i], 2) if values and i is not None and i < len(values) and values[i] is not None else None

    concept = run.get("concept") or {}
    return {
        "model": run.get("model"), "concept": concept.get("name"), "values": len(concept.get("values") or []), "read": run.get("read"),
        "layers": layers, "subspace_dims": dims, "hypotheses": hypotheses,
        "shape_tests": "none apply: a categorical concept without groups has no layout to test" if not shapes.get("hypotheses") else None,
        "at_best_layer": {"layer": best_layer, "participation_ratio": at(run.get("participation_ratio"), best_layer),
                          "between_fraction": at(run.get("between_fraction"), best_layer),
                          "components_90": (run.get("components_90") or [None] * (layers + 1))[best_layer] if best_layer is not None else None,
                          "neighbor_share": at(run.get("neighbors"), best_layer), "neighbor_chance": _round(run.get("neighbor_chance"), 3)},
        "run_id": run.get("run_id"),
        "note": "A hypothesis fits only where its held-out R² beats the max over shuffled value orders. compare_geometry, "
                "steer_geometry and geometry_sae take its run_id. Opened in the Geometry Lab",
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


# Probe Lab: saving a sweep's probe, and what saved probes do

def _kept_sweep(user: User, run_id: str) -> tuple[dict, dict]:
    from services import lab_runs

    kept = lab_runs.get(user.id, "sweep", run_id)
    if kept is None:
        raise HTTPException(status_code=404, detail="No kept sweep with that run_id: sweeps are kept on the server for a few hours "
                                                    "(train_probe trains one again, or the Probe Lab's Save button saves from the page)")
    return kept


def _plain(value):
    """A run's value as JSON data (the kept sweep holds the route's models)."""
    return value.model_dump(mode="json") if hasattr(value, "model_dump") else value


def _save_request(user: User, args: dict):
    """The save request for a kept sweep's probe, built as the Probe Lab's Save button builds it (probeUtils.saveRequest)."""
    import numpy as np

    from api.schemas.probes import SaveProbeRequest
    from core.tensor_codec import to_float32

    run, extra = _kept_sweep(user, (args.get("run_id") or "").strip())
    position = _int(args, "layer", 1, run["num_layers"], "layer")
    pooling, method = args.get("pooling"), args.get("method")
    if pooling not in POOLINGS or method not in METHODS:
        _bad(f"pooling is one of {', '.join(POOLINGS)}; method one of {', '.join(METHODS)}")
    entry = run["poolings"][pooling]
    result = entry["methods"][method]
    weights = to_float32(result["weights"])[position].astype(np.float64)
    mean_diff = to_float32(entry["methods"]["diff_means"]["weights"])[position].astype(np.float64)
    norm = float(np.linalg.norm(weights))
    class_gap = float(weights @ mean_diff / norm) if norm > 0 else None
    metrics = result.get("metrics") or {}

    def at(name: str):
        values = metrics.get(name) or []
        return values[position] if position < len(values) else None

    dataset = _plain(run["dataset"])
    short = {"logreg": "LR", "diff_means": "DiM"}[method]
    name = " ".join(str(args.get("name") or "").split())[:255] or f"{dataset['name']} · L{position - 1} {pooling} {short}"
    l2 = result.get("l2")
    return SaveProbeRequest(
        name=name, model=run["model"], layer=position - 1, pooling=pooling, method=method,
        weights=weights.tolist(), bias=float(result["biases"][position] or 0.0),
        chat_template=run["chat_template"], read_span=run["read_span"],
        normalization={"score_mean": at("score_mean"), "score_std": at("score_std"),
                       "act_norm": (entry.get("act_norms") or [None] * (position + 1))[position], "class_gap": class_gap},
        metrics={
            **{key: at(key) for key in ("test_auroc", "test_acc", "train_acc", "ood_auroc", "ood_acc", "control_test_auroc",
                                        "control_test_acc", "control_train_acc", "selectivity")},
            "embedding_test_auroc": (metrics.get("test_auroc") or [None])[0],
            "l2": l2[position] if l2 and position < len(l2) else None,
            "sweep": {"test_auroc": metrics.get("test_auroc") or [], "control_test_auroc": metrics.get("control_test_auroc") or [],
                      "ood_auroc": metrics.get("ood_auroc")},
        },
        dataset={**dataset, "seed": run["seed"], "test_fraction": extra.get("test_fraction", 0.2)},
    )


async def quote_save_probe(db: Session, user: User, ctx: ToolContext, args: dict) -> RunQuote:
    body = _save_request(user, args)
    auroc, control = body.metrics.get("test_auroc"), body.metrics.get("control_test_auroc")
    return RunQuote(
        "Save a probe",
        [["Name", body.name], ["Model", _model_name(body.model)], ["Probe", f"layer {body.layer}'s output, {body.pooling} pooling, {body.method}"],
         ["Test AUROC", f"{auroc:.3f} (control task {control:.3f})" if auroc is not None and control is not None else "–"]],
        0.0, note="Free; adds it to your saved probes", chip=f"Save probe · {_short(body.name)}",
    )


async def run_save_probe(db: Session, user: User, ctx: ToolContext, args: dict) -> ToolOutcome:
    from api.probes import save_probe

    body = _save_request(user, args)
    saved = (await save_probe(body=body, current_user=user, db=db)).model_dump(mode="json")
    out = {
        "probe_id": saved["id"], "name": saved["name"], "model": saved["model_id"], "layer": saved["layer"],
        "pooling": saved["pooling"], "method": saved["method"],
        "test_auroc": _round(body.metrics.get("test_auroc")), "control_auroc": _round(body.metrics.get("control_test_auroc")),
        "note": "layer is the decoder layer whose output it reads (the sweep's position minus 1). score_probe, steer_probe, ablate_probe and "
                "compare_probe_sae take its probe_id; the Generate page can score generations with it",
    }
    return ToolOutcome(out, chip=f"Saved probe · {_short(saved['name'])}", result={"page": "probes", "data": {"probe": saved}})


def _saved(db: Session, user: User, args: dict):
    from models.database import Probe

    probe_id = _int(args, "probe_id", 1, None, "probe_id")
    probe = db.query(Probe).filter(Probe.id == probe_id, Probe.user_id == user.id).first()
    if probe is None:
        raise HTTPException(status_code=404, detail="No saved probe of the user's with that ID (list_probes gives them)")
    return probe


def _labels_of(probe) -> list[str]:
    names = (probe.dataset or {}).get("label_names")
    return [str(n) for n in names] if isinstance(names, list) and len(names) == 2 else ["label 0", "label 1"]


def _probe_detail(probe) -> str:
    return f"{probe.name} · layer {probe.layer}, {probe.pooling}, {probe.method}"


def _score_texts(args: dict) -> list[str]:
    from api.schemas.probes import MAX_SCORE_TEXTS

    texts = args.get("texts")
    if not isinstance(texts, list) or not texts or not all(isinstance(t, str) and t.strip() for t in texts):
        _bad("texts is a list of 1 to 8 non-empty texts")
    if len(texts) > MAX_SCORE_TEXTS:
        _bad(f"At most {MAX_SCORE_TEXTS} texts")
    if any(len(t) > 2000 for t in texts):
        _bad("Each text is at most 2000 characters")
    return texts


async def quote_score_probe(db: Session, user: User, ctx: ToolContext, args: dict) -> RunQuote:
    from services.credit_service import min_gpu_request_charge

    probe = _saved(db, user, args)
    texts = _score_texts(args)
    return RunQuote(
        "Score texts with a probe",
        [["Probe", _probe_detail(probe)], ["Model", _model_name(probe.model_id)],
         ["Texts", f"{len(texts)}: {_cut(texts[0], 160)}" + (" …" if len(texts) > 1 else "")]],
        min_gpu_request_charge(probe.model_id), note="One GPU request", chip=f"Score · {_short(probe.name)}",
    )


async def run_score_probe(db: Session, user: User, ctx: ToolContext, args: dict) -> ToolOutcome:
    from api.probes import score_probe_for
    from api.schemas.probes import ScoreProbeRequest, ScoreProbeResponse

    probe = _saved(db, user, args)
    texts = _score_texts(args)
    result = ScoreProbeResponse.model_validate(
        await score_probe_for(db, user, ScoreProbeRequest(probe_id=probe.id, texts=texts))
    ).model_dump(mode="json")
    labels = _labels_of(probe)
    std = (probe.normalization or {}).get("score_std")

    def strongest(r: dict) -> list:
        start, end = r["span"]
        pooled = [(i, r["scores"][i]) for i in range(start, min(end, len(r["scores"])))]
        top = sorted(pooled, key=lambda p: -abs(p[1]))[:6]
        return [{"token": r["tokens"][i], "position": i, "score": _round(v)} for i, v in top]

    out = {
        "probe": {"id": probe.id, "name": probe.name, "layer": probe.layer, "labels": labels},
        "training_score_std": _round(std),
        "texts": [
            {"text": _cut(r["text"], 300), "score": _round(r["score"]), "reads_as": labels[1 if r["score"] > 0 else 0],
             "strongest_tokens": strongest(r)}
            for r in result["results"]
        ],
        "note": "A pooled score above 0 reads as the second label. Compare its size with the training scores' standard deviation. "
                "Opened in the Probe Lab's score view (the first text)",
    }
    return ToolOutcome(out, chip=f"Scored · {_short(probe.name)}", result={
        "page": "probes", "data": {
            "probe_id": probe.id, "texts": texts, "response": result, "labels": labels,
            "scale": 2 * std if std else None, "description": f"{probe.name} ({probe.model_id}, layer {probe.layer})",
        },
    })


def _steer_request(db: Session, user: User, args: dict):
    from api.schemas.probes import SteerProbeRequest
    from core.probes.budget import MAX_STEER_ALPHAS, MAX_STEER_TOKENS

    probe = _saved(db, user, args)
    gap = (probe.normalization or {}).get("class_gap")
    if gap is None:
        _bad("This probe was saved without its class gap, which steering is measured in; save it again from a sweep")
    if gap <= 0:
        _bad("The classes' means aren't apart along this probe's direction, so it can't steer")
    alphas = args.get("alphas")
    if not isinstance(alphas, list) or not all(isinstance(a, (int, float)) and not isinstance(a, bool) for a in alphas):
        _bad("alphas is a list of numbers")
    alphas = list(dict.fromkeys(float(a) for a in alphas if a != 0))
    if not alphas and not args.get("ablate"):
        _bad("Give a nonzero strength, or ablate the direction")
    if len(alphas) > MAX_STEER_ALPHAS or any(abs(a) > 64 for a in alphas):
        _bad(f"At most {MAX_STEER_ALPHAS} strengths, each at most ±64")
    return probe, SteerProbeRequest(
        # Only the ablation: a zero strength, which the route leaves out
        probe_id=probe.id, prompt=_text(args, "prompt", "prompt", limit=2000), alphas=alphas or [0.0],
        ablate=bool(args.get("ablate")), generated_only=bool(args.get("generated_only")),
        max_tokens=_int(args, "max_tokens", 1, MAX_STEER_TOKENS, "max_tokens"),
        temperature=_float(args, "temperature", 0, 2, "temperature"), seed=0,
    ), bool(alphas)


async def quote_steer_probe(db: Session, user: User, ctx: ToolContext, args: dict) -> RunQuote:
    from api.probes import estimate_steering
    from api.schemas.probes import EstimateSteeringRequest

    probe, body, steered = _steer_request(db, user, args)
    estimate = await estimate_steering(
        body=EstimateSteeringRequest(model=probe.model_id, prompt_chars=len(body.prompt), num_alphas=len(body.alphas) if steered else 0,
                                     ablate=body.ablate, max_tokens=body.max_tokens),
        current_user=user, db=db,
    )
    strengths = ", ".join(f"{a:+g}" for a in body.alphas) if steered else "none"
    return RunQuote(
        "Steer with a probe",
        [["Probe", _probe_detail(probe)], ["Model", _model_name(probe.model_id)], ["Prompt", _cut(body.prompt)],
         ["Strengths", strengths + (" · and ablated" if body.ablate else "")],
         ["Tokens", f"{body.max_tokens} each, temperature {body.temperature:g}" + (", generated tokens only" if body.generated_only else "")]],
        estimate["estimated_cost"], note=f"{estimate['generations']} generations; charged on the tokens they generate",
        chip=f"Steer · {_short(probe.name)}",
    )


async def run_steer_probe(db: Session, user: User, ctx: ToolContext, args: dict) -> ToolOutcome:
    from api.probes import steer_probe_for
    from api.schemas.probes import SteerProbeResponse

    probe, body, _ = _steer_request(db, user, args)
    result = SteerProbeResponse.model_validate(await steer_probe_for(db, user, body)).model_dump(mode="json")
    labels = _labels_of(probe)
    out = {
        "probe": {"id": probe.id, "name": probe.name, "layer": result["layer"], "labels": labels},
        "unit_norm": _round(result["unit_norm"]), "prompt_tokens": result["prompt_tokens"],
        "generations": [
            {"kind": g["kind"], "alpha": g["alpha"], "text": _cut(g["text"], 400), "probe_score": _round(g["probe_score"]),
             "perplexity": _round(g["perplexity"], 2), "first_divergence": g["first_divergence"], "tokens_changed": g["tokens_changed"]}
            for g in sorted(result["generations"], key=lambda g: (g["kind"] == "ablate", g["alpha"] or 0))
        ],
        "note": "probe_score and perplexity are read by the unsteered model; the baseline (alpha 0) is the comparison. A score that moves "
                "with alpha while perplexity stays near the baseline's means the model uses the direction. Opened in the Probe Lab's steer view",
    }
    return ToolOutcome(out, chip=f"Steered · {_short(probe.name)}", result={
        "page": "probes", "data": {"probe_id": probe.id, "request": body.model_dump(mode="json"), "response": result},
    })


def _ablate_request(db: Session, user: User, args: dict):
    from api.schemas.probes import AblateProbeRequest, ProbeDatasetSpec
    from core.probes.datasets import builtin_datasets

    probe = _saved(db, user, args)
    own = probe.dataset or {}
    spec = _dataset_spec(args, required=False, use_ood=False)
    if spec is None:
        builtin = own.get("builtin_id")
        if not builtin or builtin not in builtin_datasets():
            _bad("This probe was trained on uploaded rows: give a built-in dataset or rows to ablate on")
        spec = ProbeDatasetSpec(builtin_id=builtin, use_builtin_ood=False)
    fraction = args.get("test_fraction") or 0
    fraction = min(0.5, max(0.1, float(own.get("test_fraction") or 0.2))) if not fraction else _float(args, "test_fraction", 0.1, 0.5, "test_fraction")
    seed = _int(args, "seed", -1, 2**31 - 1, "seed")
    seed = int(own.get("seed") or 0) if seed < 0 else seed
    return probe, AblateProbeRequest(probe_id=probe.id, dataset=spec, test_fraction=fraction, seed=seed)


async def quote_ablate_probe(db: Session, user: User, ctx: ToolContext, args: dict) -> RunQuote:
    from api.probes import estimate_run
    from api.schemas.probes import EstimateProbesRequest

    probe, body = _ablate_request(db, user, args)
    examples, chars, name = _dataset_size(body.dataset)
    estimate = await estimate_run(
        body=EstimateProbesRequest(model=probe.model_id, num_examples=examples, total_chars=chars, chat_template=probe.chat_template,
                                   kind="ablate", method=probe.method),
        current_user=user, db=db,
    )
    if not estimate["within_limit"]:
        _bad("This ablation would take longer than the model's time limit")
    return RunQuote(
        "Ablate a probe's direction",
        [["Probe", _probe_detail(probe)], ["Model", _model_name(probe.model_id)], ["Dataset", f"{name} ({examples} examples)"],
         ["Split", f"{round(body.test_fraction * 100)}% held out, seed {body.seed}"]],
        estimate["estimated_cost"], note="Charged by its size when it ends", chip=f"Ablate · {_short(probe.name)}",
    )


async def run_ablate_probe(db: Session, user: User, ctx: ToolContext, args: dict) -> ToolOutcome:
    from api.probes import ablate_probe_for
    from api.schemas.probes import AblateProbeResponse

    probe, body = _ablate_request(db, user, args)
    result = AblateProbeResponse.model_validate(await ablate_probe_for(db, user, body)).model_dump(mode="json")
    curves = result["curves"]
    clean = curves.get("clean") or []
    best = max((p for p in range(1, len(clean)) if clean[p] is not None), key=lambda p: clean[p], default=None)

    def at(name: str, p: Optional[int]):
        values = curves.get(name) or []
        return _round(values[p]) if p is not None and p < len(values) else None

    shown = sorted({p for p in (result["layer"] + 1, best) if p is not None})
    out = {
        "probe": {"id": probe.id, "name": probe.name, "layer": result["layer"], "method": result["method"], "pooling": result["pooling"]},
        "dataset": result["dataset"]["name"], "examples": {"train": result["dataset"]["n_train"], "test": result["dataset"]["n_test"]},
        "behaviour": {side: {k: _round(v) if not isinstance(v, list) else [_round(x) for x in v] for k, v in b.items()}
                      for side, b in result["behaviour"].items()},
        "test_auroc": [
            {"position": p, "clean": at("clean", p), "probe_frozen": at("probe_frozen", p), "probe_retrained": at("probe_retrained", p),
             "control_frozen": at("control_frozen", p), "control_retrained": at("control_retrained", p)}
            for p in shown
        ],
        "note": "Positions are residual stream positions (the probe's layer + 1 is where it reads). frozen: the clean probes on ablated "
                "activations; retrained: probes refit on them. Control directions remove as much of the residual stream as the probe's: "
                "compare the probe's KL and AUROC drop with theirs. Opened in the Probe Lab's ablate view",
    }
    return ToolOutcome(out, chip=f"Ablated · {_short(probe.name)}", result={
        "page": "probes", "data": {"probe_id": probe.id, "response": result},
    })


def _generalize_request(db: Session, user: User, args: dict):
    from api.schemas.probes import GeneralizeRequest, MatrixDatasetSpec
    from core.open_source.registry import instruct_sibling
    from core.probes.budget import MAX_MATRIX_DATASETS, MIN_MATRIX_DATASETS
    from core.probes.datasets import builtin_datasets

    model = _check_model(db, user, args.get("model", ""))
    items = args.get("datasets")
    if not isinstance(items, list) or not MIN_MATRIX_DATASETS <= len(items) <= MAX_MATRIX_DATASETS:
        _bad(f"datasets lists {MIN_MATRIX_DATASETS} to {MAX_MATRIX_DATASETS} datasets")
    specs = []
    for i, item in enumerate(items):
        item = item if isinstance(item, dict) else {}
        dataset = (item.get("dataset") or "").strip()
        part = item.get("part") or "main"
        if dataset:
            if dataset not in builtin_datasets():
                raise HTTPException(status_code=404, detail=f"Dataset {i + 1}: unknown built-in dataset {dataset!r}")
            if part == "ood" and not builtin_datasets()[dataset].get("ood"):
                _bad(f"{builtin_datasets()[dataset]['name']} has no OOD set")
            specs.append(MatrixDatasetSpec(builtin_id=dataset, part=part))
            continue
        rows, problem = check_rows(item.get("rows") or [])
        if problem:
            _bad(f"Dataset {i + 1}: {problem}")
        names = None
        if item.get("label_names"):
            names, problem = check_labels(item["label_names"])
            if problem:
                _bad(f"Dataset {i + 1}: {problem}")
        specs.append(MatrixDatasetSpec(rows=rows, name=" ".join(str(item.get("name") or "").split())[:255] or None, label_names=names))
    transfer = None
    if args.get("transfer"):
        transfer = instruct_sibling(model)
        if transfer is None:
            _bad(f"{model} has no base/instruct sibling to transfer to")
    pooling = args.get("pooling")
    if pooling not in POOLINGS:
        _bad(f"pooling is one of: {', '.join(POOLINGS)}")
    return GeneralizeRequest(
        model=model, transfer_model=transfer, datasets=specs, pooling=pooling, chat_template=bool(args.get("chat_template")),
        read_span="text", test_fraction=round(_float(args, "test_fraction", 0.1, 0.5, "test_fraction"), 2),
        seed=_int(args, "seed", 0, 2**31 - 1, "seed"),
    )


def _matrix_size(spec) -> tuple[int, int, str]:
    from core.probes.datasets import builtin_datasets, dataset_summary

    if spec.builtin_id:
        summary = dataset_summary(builtin_datasets()[spec.builtin_id])
        if spec.part == "ood":
            ood = summary["ood"]
            return ood["num_rows"], ood["total_chars"], f"{summary['name']} · OOD"
        return summary["num_rows"], summary["total_chars"], summary["name"]
    return len(spec.rows), sum(len(r.text) for r in spec.rows), spec.name or "Your rows"


async def quote_generalize_probes(db: Session, user: User, ctx: ToolContext, args: dict) -> RunQuote:
    from api.probes import estimate_generalization
    from api.schemas.probes import EstimateGeneralizeRequest, MatrixDatasetSize

    body = _generalize_request(db, user, args)
    sizes = [_matrix_size(d) for d in body.datasets]
    estimate = await estimate_generalization(
        body=EstimateGeneralizeRequest(model=body.model, transfer_model=body.transfer_model,
                                       datasets=[MatrixDatasetSize(num_examples=n, total_chars=c) for n, c, _ in sizes],
                                       chat_template=body.chat_template, test_fraction=body.test_fraction),
        current_user=user, db=db,
    )
    if not estimate["within_limit"]:
        _bad("This run would take longer than the model's time limit; use fewer or smaller datasets")
    models = _model_name(body.model) + (f" and {_model_name(body.transfer_model)}" if body.transfer_model else "")
    return RunQuote(
        "Generalization matrix",
        [["Models", models], ["Datasets", "; ".join(f"{name} ({n})" for n, _, name in sizes)],
         ["Probes", f"{body.pooling} pooling, both methods, every layer"],
         ["Split", f"{round(body.test_fraction * 100)}% held out, seed {body.seed}"], ["Chat template", "yes" if body.chat_template else "no"]],
        estimate["estimated_cost"], note="Charged by its size when it ends", chip=f"Generalization · {_model_name(body.model)}",
    )


async def run_generalize_probes(db: Session, user: User, ctx: ToolContext, args: dict) -> ToolOutcome:
    from api.probes import generalize_probes_for
    from api.schemas.probes import GeneralizeResponse

    body = _generalize_request(db, user, args)
    result = GeneralizeResponse.model_validate(await generalize_probes_for(db, user, body)).model_dump(mode="json")
    names = [d["name"] for d in result["datasets"]]
    conditions = [f"{m} · {n}" if len(result["models"]) > 1 else n for m in result["models"] for n in names]

    def matrices(method: str) -> dict:
        auroc = result["auroc"][method]
        n = len(conditions)
        positions = len(auroc[0][0]) if n and auroc[0] else 0
        # The position where in-distribution (diagonal) AUROC is best on average
        diagonal = {p: [auroc[i][i][p] for i in range(n) if auroc[i][i][p] is not None] for p in range(1, positions)}
        best = max((p for p, v in diagonal.items() if v), key=lambda p: sum(diagonal[p]) / len(diagonal[p]), default=None)
        return {
            "position": best,
            "auroc_train_by_test": [[_round(auroc[i][j][best]) if best is not None else None for j in range(n)] for i in range(n)],
        }

    out = {
        "conditions": conditions,
        "by_method": {method: matrices(method) for method in result["auroc"]},
        "overlap": result["overlap"],
        "note": "Rows train, columns test, at the residual stream position where in-distribution AUROC is best on average; 0.5 is chance. "
                "overlap counts test texts also in another dataset's training split. Opened in the Probe Lab's generalize view",
    }
    return ToolOutcome(out, chip=f"Generalization · {_model_name(body.model)}", result={
        "page": "probes", "data": {"request": body.model_dump(mode="json"), "response": result},
    })


# Geometry Lab: what a kept run's concept does

def _kept_run(user: User, run_id: str) -> dict:
    from services import lab_runs

    kept = lab_runs.get(user.id, "geometry", (run_id or "").strip())
    if kept is None:
        raise HTTPException(status_code=404, detail="No kept run with that run_id: runs are kept on the server for a few hours "
                                                    "(run_geometry runs one again)")
    return kept[0]


def _run_concept(run: dict):
    """A kept run's concept as a request's (its built-in ID, or the whole custom concept)."""
    from api.schemas.geometry import ConceptSpec, CustomConcept

    c = run["concept"]
    if c.get("builtin_id"):
        return ConceptSpec(builtin_id=c["builtin_id"])
    return ConceptSpec(custom=CustomConcept(**{k: c[k] for k in (
        "name", "structure", "values", "templates", "eval_templates", "positions", "groups", "helix_periods")}))


def _value_index(values: list[str], value, what: str) -> int:
    text = str(value or "").strip()
    for i, v in enumerate(values):
        if v == text:
            return i
    lowered = [v.lower() for v in values]
    if text.lower() in lowered:
        return lowered.index(text.lower())
    _bad(f"{what} is one of the concept's values: {', '.join(values[:20])}" + (" …" if len(values) > 20 else ""))


def _geometry_steer_request(user: User, args: dict):
    from api.schemas.geometry import SteerRequest

    run_id = (args.get("run_id") or "").strip()
    run = _kept_run(user, run_id)
    c = run["concept"]
    if c["structure"] == "categorical":
        _bad("Steering follows the values' order; categorical concepts have none")
    values = c["values"]
    template = (args.get("eval_template") or "").strip()
    if not template:
        if not c.get("eval_templates"):
            _bad("This concept has no eval prompts: give eval_template")
        template = c["eval_templates"][0]
    return run_id, run, SteerRequest(
        model=run["model"], concept=_run_concept(run), read=run["read"], chat_template=run["chat_template"],
        position=_int(args, "layer", 0, run["num_layers"], "layer"), dims=_int(args, "dims", 1, 16, "dims"),
        source=_value_index(values, args.get("source"), "source"), target=_value_index(values, args.get("target"), "target"),
        steps=_int(args, "steps", 4, 32, "steps"), eval_template=template,
    )


def _layer_text(position: int) -> str:
    return "the embeddings" if position == 0 else f"layer {position - 1}'s output"


async def quote_steer_geometry(db: Session, user: User, ctx: ToolContext, args: dict) -> RunQuote:
    from api.geometry import estimate_steer

    run_id, run, body = _geometry_steer_request(user, args)
    estimate = await estimate_steer(body=body, current_user=user, db=db)
    if not estimate["within_limit"]:
        _bad("This run would take longer than the model's time limit; take fewer steps")
    values = run["concept"]["values"]
    return RunQuote(
        "Steer along a concept",
        [["Run", f"{run['concept']['name']} · {_model_name(run['model'])} · {run['read']}"],
         ["Walk", f"{values[body.source]} → {values[body.target]}, {body.steps} steps"],
         ["At", f"{_layer_text(body.position)}, top {body.dims} components"], ["Eval prompt", _cut(body.eval_template, 160)]],
        estimate["estimated_cost"], note=f"{estimate['eval_passes']} forward passes; charged by its size",
        chip=f"Steer · {values[body.source]} → {values[body.target]}",
    )


async def run_steer_geometry(db: Session, user: User, ctx: ToolContext, args: dict) -> ToolOutcome:
    from api.geometry import steer_for
    from api.schemas.geometry import SteerResponse

    run_id, run, body = _geometry_steer_request(user, args)
    result = SteerResponse.model_validate(await steer_for(db, user, body)).model_dump(mode="json")
    values = result["values"]

    def walk(method: str) -> dict:
        m = result["methods"][method]
        nearest = [values[i] for i in m["nearest"]]
        reached = next((step for step, i in enumerate(m["nearest"]) if i == body.target), None)
        return {
            "nearest_value_by_step": nearest, "reaches_target_at_step": reached,
            "hellinger_to_nearest_by_step": [_round(v) for v in m["hellinger"]],
            "final_to_target": _round(m["to_target"][-1]) if m["to_target"] else None,
            "final_mass": _round(m["mass"][-1]) if m["mass"] else None,
            "max_edit_size": _round(max((v for v in m["edit_size"] if v is not None), default=None)),
        }

    reference = result["reference"]
    out = {
        "walk": f"{values[body.source]} → {values[body.target]}", "at": _layer_text(body.position), "dims": body.dims, "steps": body.steps,
        "explained": _round(result["explained"]), "eval_decoding": _round(result["eval_decoding"]),
        "readable_values": [values[i] for i in result["readable"]],
        "target_nearest_other": _round(reference["nearest_other"][body.target]) if body.target < len(reference["nearest_other"]) else None,
        "manifold": walk("manifold"), "linear": walk("linear"),
        "note": "nearest_value_by_step: the value whose unsteered next-token distribution each step's is closest to (Hellinger). Steering "
                "lands on a value when its distance falls well below target_nearest_other, the distance between the target and its "
                "nearest other value. mass: probability on the values' tokens. Opened in the Geometry Lab's steer view",
    }
    return ToolOutcome(out, chip=f"Steered · {values[body.source]} → {values[body.target]}", result={
        "page": "geometry", "data": {"run_id": run_id, "response": result},
    })


def _geometry_sae_request(db: Session, user: User, args: dict):
    from api.schemas.geometry import GeometrySaeRequest

    run_id = (args.get("run_id") or "").strip()
    run = _kept_run(user, run_id)
    key, ref, name, _ = _sae_target(db, user, args)
    return run_id, run, key, name, GeometrySaeRequest(
        model=run["model"], concept=_run_concept(run), read=run["read"], chat_template=run["chat_template"],
        sae=_parse_sae(key)[1], seed=run["seed"],
    )


async def quote_geometry_sae(db: Session, user: User, ctx: ToolContext, args: dict) -> RunQuote:
    from api.geometry import estimate_sae

    run_id, run, key, name, body = _geometry_sae_request(db, user, args)
    estimate = await estimate_sae(body=body, current_user=user, db=db)
    if not estimate["within_limit"]:
        _bad("This run would take longer than the model's time limit")
    return RunQuote(
        "Read a concept through an SAE",
        [["Run", f"{run['concept']['name']} · {_model_name(run['model'])} · {run['read']}"], ["SAE", f"{name} (reads layer {estimate['layer']}'s output)"]],
        estimate["estimated_cost"], note="Charged by its size when it ends", chip=f"Geometry SAE · {_short(name)}",
    )


async def run_geometry_sae(db: Session, user: User, ctx: ToolContext, args: dict) -> ToolOutcome:
    from api.geometry import sae_geometry_for
    from api.schemas.geometry import GeometrySaeResponse

    run_id, run, key, name, body = _geometry_sae_request(db, user, args)
    result = GeometrySaeResponse.model_validate(await sae_geometry_for(db, user, body)).model_dump(mode="json")
    sae = result["sae"]
    width = (sae.get("info") or {}).get("width")
    out = {
        "sae": key, "sae_name": name, "layer": result["layer"], "width": width, "l0": _round(sae.get("l0"), 1),
        "density": _round(sae["l0"] / width, 4) if sae.get("l0") is not None and width else None, "fvu": _round(sae.get("fvu")),
        "active_features": result["active_features"], "varying_features": result["varying_features"],
        "fits": [
            {"dims": f["dims"] if f["dims"] is not None else "all", "verdict": f["verdict"], "variance": _round(f["variance"]),
             "ceiling_r2": _round(f["ceiling"]), "features_for_90": f["n90"], "compact_limit": f["compact"],
             "top_features": [{"feature": i, "link": feature_link(key, i, result["model"])} for i in f["order"][:8]],
             "random_order_r2_at_n90": _round((f["random"]["mean"] or [None])[f["n90"] - 1]) if f["n90"] and f["random"]["mean"] and f["n90"] <= len(f["random"]["mean"]) else None,
             "coactive": _round(f["fields"]["coactive"]), "contiguous_fields": _round(f["fields"]["contiguous"])}
            for f in result["fits"]
        ],
        "note": "Verdicts: capture (a compact group of features writes 90% of the class means' shape), shatter (the features writing 90% "
                "are nearly never active together), dilute (it takes many), missed (all features together write under 90%). fvu is the "
                "SAE's unexplained variance on these activations: high fvu makes every verdict unreliable. Opened in the Geometry Lab's SAE view",
    }
    return ToolOutcome(out, chip=f"Geometry SAE · {_short(name)}", result={
        "page": "geometry", "data": {"run_id": run_id, "sae": key, "response": result},
    })


# Attention Analysis: comparing two prompts, and mining one

async def _capture_side(db: Session, user: User, model: str, prompt_key: str, id_key: str, args: dict):
    """(model, prompt or None, capture or None) for one side: a capture reused by ID, or a prompt to capture."""
    return await _heads_target(db, user, {"model": model, "prompt": args.get(prompt_key, ""), "activation_id": args.get(id_key, "")})


def _capture_size(model: str, prompt: Optional[str], captured, specs) -> tuple[int, int, int]:
    """(layers, heads, tokens) of a side, captured or to capture."""
    from api.analysis import _num_heads

    if captured is not None:
        return captured.num_layers, _num_heads(captured), len(captured.tokens)
    return specs.layers or 1, specs.heads or 1, _prompt_tokens(model, prompt)


async def _compare_sides(db: Session, user: User, args: dict):
    model = (args.get("model") or "").strip()
    sides = [await _capture_side(db, user, model, f"prompt_{i}", f"activation_id_{i}", args) for i in (1, 2)]
    models = {m for m, _, _, _ in sides}
    if len(models) > 1:
        _bad(f"Both sides must be on one model (these are on {', '.join(sorted(models))})")
    return sides


async def quote_compare_attention(db: Session, user: User, ctx: ToolContext, args: dict) -> RunQuote:
    from services.credit_service import estimate_analysis_cost, min_gpu_request_charge

    sides = await _compare_sides(db, user, args)
    model = sides[0][0]
    sizes = [_capture_size(m, p, c, s) for m, p, c, s in sides]
    captures = sum(min_gpu_request_charge(model) for _, p, c, _ in sides if c is None)
    cost = estimate_analysis_cost(analysis_type="compare", num_layers=sizes[0][0], num_heads=sizes[0][1],
                                  num_tokens=sizes[0][2] + sizes[1][2], model_id=model)
    details = [["Model", _model_name(model)]]
    for i, (_, prompt, captured, _) in enumerate(sides, 1):
        details.append([f"Prompt {i}", _cut(captured.prompt if captured else prompt, 160) + (" (captured)" if captured else "")])
    if captures:
        details.append(["Charges", f"captures {_money(captures)} + comparison {_money(cost)}"])
    return RunQuote("Compare attention", details, captures + cost,
                    note="Captures are one GPU request each; the comparison is charged on its CPU time", chip=f"Compare attention · {_model_name(model)}")


async def run_compare_attention(db: Session, user: User, ctx: ToolContext, args: dict) -> ToolOutcome:
    from api.analysis import CompareAttentionRequest, compare_attention
    from api.patching import capture_activations
    from api.schemas.patching import CaptureActivationsRequest

    sides = await _compare_sides(db, user, args)
    model = sides[0][0]
    ids = []
    for _, prompt, captured, _ in sides:
        if captured is None:
            capture = await capture_activations(request=CaptureActivationsRequest(prompt=prompt, model=model), current_user=user, db=db)
            ids.append(capture.activation_id)
        else:
            ids.append(captured.id)
    response = await asyncio.to_thread(
        compare_attention, request=CompareAttentionRequest(activation_id_1=ids[0], activation_id_2=ids[1]), current_user=user, db=db,
    )
    result = response.model_dump(mode="json")
    out = {
        "model": model, "activation_ids": ids, "prompts": [_cut(result["prompt1"], 200), _cut(result["prompt2"], 200)],
        "layer_similarity": [{"layer": l["layer"], "mean": _round(l["mean_similarity"]), "min": _round(l["min_similarity"])}
                             for l in result["layer_similarities"]],
        "most_different_heads": [{"layer": h["layer"], "head": h["head"], "similarity": _round(h["similarity"])}
                                 for h in result["head_differences"][:10]],
        "common_patterns": result["common_patterns"][:6], "divergent_patterns": result["divergent_patterns"][:6],
        "note": "Similarity of the two prompts' attention patterns (1 is identical). One pair of prompts is one example. Opened on the Attention Analysis page",
    }
    return ToolOutcome(out, chip=f"Compared attention · {_model_name(model)}", result={
        "page": "analysis", "data": {"model": model, "activation_ids": ids, "result": result},
    })


async def quote_mine_patterns(db: Session, user: User, ctx: ToolContext, args: dict) -> RunQuote:
    from services.credit_service import estimate_analysis_cost, min_gpu_request_charge

    model, prompt, captured, specs = await _heads_target(db, user, args)
    layers, heads, tokens = _capture_size(model, prompt, captured, specs)
    cost = estimate_analysis_cost(analysis_type="mine", num_layers=layers, num_heads=heads, num_tokens=tokens, model_id=model)
    if captured is not None:
        return RunQuote("Mine attention patterns", [["Model", _model_name(model)], ["Capture", _cut(captured.prompt)]],
                        cost, note="Charged on the CPU time it takes", chip=f"Mine patterns · {_model_name(model)}")
    capture = min_gpu_request_charge(model)
    return RunQuote(
        "Capture and mine attention patterns",
        [["Model", _model_name(model)], ["Prompt", _cut(prompt)], ["Charges", f"capture {_money(capture)} + mining {_money(cost)}"]],
        capture + cost, note="Two charges: the capture (one GPU request) and the mining", chip=f"Mine patterns · {_model_name(model)}",
    )


async def run_mine_patterns(db: Session, user: User, ctx: ToolContext, args: dict) -> ToolOutcome:
    from api.analysis import MinePatternRequest, mine_patterns
    from api.patching import capture_activations
    from api.schemas.patching import CaptureActivationsRequest

    model, prompt, captured, _ = await _heads_target(db, user, args)
    if captured is None:
        capture = await capture_activations(request=CaptureActivationsRequest(prompt=prompt, model=model), current_user=user, db=db)
        activation_id = capture.activation_id
    else:
        activation_id = captured.id
    response = await asyncio.to_thread(mine_patterns, request=MinePatternRequest(activation_id=activation_id), current_user=user, db=db)
    result = response.model_dump(mode="json")
    out = {
        "model": model, "activation_id": activation_id,
        "counts": {k: v for k, v in (result.get("classification_summary") or {}).items() if v},
        "top_heads": [{"layer": t[0], "head": t[1], "importance": _round(t[2])} for t in (result.get("top_heads") or [])[:10]],
        "cross_layer_patterns": [{"type": p["pattern_type"], "description": _cut(p["description"], 200), "confidence": _round(p["confidence"]),
                                  "heads": p["involved_heads"][:8]} for p in (result.get("cross_layer_patterns") or [])[:8]],
        "circuit_candidates": [{"type": c["type"], "description": _cut(c["description"], 200), "confidence": _round(c["confidence"])}
                               for c in (result.get("circuit_candidates") or [])[:8]],
        "note": "Heuristics over one prompt's attention patterns: candidates to test (e.g. with run_patching), not findings. Opened on the Attention Analysis page",
    }
    return ToolOutcome(out, chip=f"Mined patterns · {_model_name(model)}", result={
        "page": "analysis", "data": {"model": model, "activation_id": activation_id, "result": result},
    })


Quote = Callable[[Session, User, ToolContext, dict], Awaitable[RunQuote]]
Run = Callable[[Session, User, ToolContext, dict], Awaitable[ToolOutcome]]

RUNS: dict[str, tuple[Quote, Run]] = {
    "ablate_probe": (quote_ablate_probe, run_ablate_probe),
    "analyze_sae": (quote_analyze_sae, run_analyze_sae),
    "build_atlas": (quote_build_atlas, run_build_atlas),
    "classify_heads": (quote_classify_heads, run_classify_heads),
    "compare_attention": (quote_compare_attention, run_compare_attention),
    "compare_probe_sae": (quote_compare_probe_sae, run_compare_probe_sae),
    "explain_feature": (quote_explain_feature, run_explain_feature),
    "generalize_probes": (quote_generalize_probes, run_generalize_probes),
    "generate": (quote_generate, run_generate),
    "generation_test": (quote_generation_test, run_generation_test),
    "geometry_sae": (quote_geometry_sae, run_geometry_sae),
    "mine_patterns": (quote_mine_patterns, run_mine_patterns),
    "run_geometry": (quote_run_geometry, run_run_geometry),
    "run_patching": (quote_run_patching, run_run_patching),
    "save_probe": (quote_save_probe, run_save_probe),
    "score_probe": (quote_score_probe, run_score_probe),
    "start_autointerp_job": (quote_start_autointerp_job, run_start_autointerp_job),
    "steer_geometry": (quote_steer_geometry, run_steer_geometry),
    "steer_probe": (quote_steer_probe, run_steer_probe),
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
