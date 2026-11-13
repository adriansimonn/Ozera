"""
The assistant's tool executors (core.assistant.tools lists the tools): each runs as the calling
user, through the same functions the pages' routes use, so access rules and stored results are
the pages' own. A failure the page would show as an HTTP error comes back as the tool's error.

Results are compact: the numbers Claude needs, each with its baseline, and a `link` for every
feature (the panel renders it as a chip with the feature's stored explanation). Nothing here
spends credits or writes anything but what reading does on the pages.
"""

import logging
from dataclasses import dataclass
from typing import Any, Awaitable, Callable, Optional

from fastapi import HTTPException
from sqlalchemy.orm import Session

from core.assistant.loop import ToolOutcome
from core.assistant.pages import PAGES, check_value, sae_key, sae_ref
from db import SessionLocal
from models.database import AtlasJob, AutointerpJob, Probe, User

logger = logging.getLogger(__name__)

# Items listed at most: features in a search, communities in an atlas, contexts of a feature
SEARCH_RESULTS = 20
ATLAS_COMMUNITIES = 15
COMMUNITY_FEATURES = 24
FEATURE_CONTEXTS = 5
CONTEXT_TOKENS = 12
JOBS_LISTED = 8
PROBES_LISTED = 40
ACTIVATION_POSITIONS = 64
TOP_PREDICTIONS = 5


@dataclass
class ToolContext:
    user_id: int
    # The page the user's message came from
    page: Optional[str] = None


def _round(value, digits: int = 3):
    return round(value, digits) if isinstance(value, float) else value


def feature_link(key: str, feature: int, model: Optional[str]) -> str:
    link = f"ozera:feature?sae={key}&n={feature}"
    return f"{link}&model={model}" if model else link


def _ref_model(ref: dict):
    from api.schemas.probes import ExternalSaeRef, OzeraSaeRef

    return OzeraSaeRef(**ref) if ref["kind"] == "ozera" else ExternalSaeRef(**ref)


def _parse_sae(key: str):
    ref = sae_ref(key or "")
    if ref is None:
        raise HTTPException(status_code=400, detail="Unknown SAE key; list_saes gives each SAE's key")
    return ref, _ref_model(ref)


def _short(name: str) -> str:
    return name if len(name) <= 40 else name[:39] + "…"


async def _atlas(db: Session, user: User, model: str, key: str):
    """(SAE info, the atlas as autointerp reads it, or None) for a model and SAE key, checked like the pages."""
    from api.autointerp import _atlas_specs

    _, ref = _parse_sae(key)
    _, info, atlas = await _atlas_specs(db, user, model, ref, None)
    return info, atlas


def _explanation_brief(e: Optional[dict]) -> Optional[dict]:
    if not e:
        return None
    return {
        "explanation": e.get("explanation"),
        "status": e.get("status"),
        "score": _round(e.get("score")),
        "score_pm": e.get("score_pm"),
        "detection": _round(e.get("detection")),
        "detection_hard": _round(e.get("detection_hard")),
        "fuzzing": _round(e.get("fuzzing")),
        "token_baseline": _round(e.get("token_detection")),
        "token_feature": e.get("token_feature"),
        "control_percentile": _round(e.get("control_percentile"), 1),
        "explainer": e.get("explainer"),
        "scorer": e.get("scorer"),
        "note": e.get("note"),
    }


# Read tools

async def list_models(db: Session, user: User, ctx: ToolContext, args: dict) -> ToolOutcome:
    from api.probes import list_models as probe_models

    models = await probe_models(current_user=user, db=db)
    return ToolOutcome({
        "models": [
            {
                "id": m["model_id"], "name": m["display_name"], "type": m["model_type"], "layers": m["num_layers"],
                "width": m["hidden_dim"], "parameters": m["parameters"], "instruct": m["is_instruct"],
                "gpu_tier": m["gpu_tier"], "sibling": m["sibling"],
            }
            for m in models
        ],
    })


async def list_saes(db: Session, user: User, ctx: ToolContext, args: dict) -> ToolOutcome:
    import asyncio

    from api.probes import list_probe_saes
    from core.probes.sae import sae_directory
    from services import atlas_store, autointerp

    model = args.get("model", "")
    saes = await list_probe_saes(model=model, current_user=user, db=db)

    async def entry(info) -> dict:
        ref = info.ref.model_dump()
        atlases = [m for m in await atlas_store.list_atlases(sae_directory(ref)) if m.get("model") == model]
        out = {
            "key": sae_key(ref), "name": info.name, "layer": info.layer, "width": info.width, "source": info.source,
            "trained_on": info.trained_on, "match": info.match,
        }
        if info.unusable_reason:
            out["unusable"] = info.unusable_reason
        out["atlases"] = []
        for meta in atlases:
            state, reason = autointerp.state(meta)
            l0, width = meta.get("l0"), meta.get("width") or info.width
            out["atlases"].append({
                "corpus": (meta.get("corpus") or {}).get("id"),
                "tokens": meta.get("tokens"),
                "l0": _round(l0, 1),
                "density": _round(l0 / width, 4) if l0 is not None and width else None,
                "explainable": state == "ready",
                "why_not": reason,
                "explained": autointerp.explained_count(db, meta["directory"]),
                "built": meta.get("created_at"),
            })
        return out

    entries = await asyncio.gather(*(entry(info) for info in saes))
    return ToolOutcome({"model": model, "saes": list(entries)}, chip=f"SAEs · {model}")


async def get_atlas(db: Session, user: User, ctx: ToolContext, args: dict) -> ToolOutcome:
    from services import atlas_store, autointerp

    model, key = args.get("model", ""), args.get("sae", "")
    info, atlas = await _atlas(db, user, model, key)
    chip = f"Atlas · {_short(info.name)}"
    if atlas is None:
        return ToolOutcome({"atlas": None, "note": "This SAE has no atlas on this model yet (the Geometry Lab's atlas view builds one)"}, chip=chip)
    summary = await atlas_store.read_json(f"{atlas.directory}/summary.json")
    names = autointerp.community_names(db, atlas)
    explained = autointerp.explained_features(db, atlas.directory)
    state, reason = autointerp.state(atlas.meta)
    by_id = {c["id"]: c for c in summary.get("communities", [])}
    ranked = [by_id[c] for c in summary.get("ranked", []) if c in by_id][:ATLAS_COMMUNITIES]
    width = summary.get("width") or info.width
    l0 = summary.get("l0")
    return ToolOutcome({
        "sae": key, "sae_name": info.name, "model": model, "corpus": atlas.corpus,
        "tokens": summary.get("tokens"), "built": summary.get("created_at"),
        "width": width, "alive": summary.get("alive"), "l0": _round(l0, 1),
        "density": _round(l0 / width, 4) if l0 is not None and width else None,
        "fvu": _round(summary.get("fvu"), 4),
        "autointerp": {"explainable": state == "ready", "why_not": reason, "explained": len(explained),
                       "controls_scored": len(autointerp.control_scores(db, atlas.directory))},
        "communities": len(summary.get("communities", [])),
        "best_ranked": [
            {
                "id": c["id"], "name": names[c["id"]].name if c["id"] in names else None, "size": c.get("size"),
                "explained": len(set(c.get("features", [])) & explained), "top_tokens": (c.get("top_words") or [])[:8],
                "geometry": {"score": _round((c.get("geometry") or {}).get("score")),
                             "note": "top-3 explained variance minus random frequency-matched groups'"} if c.get("geometry") else None,
            }
            for c in ranked
        ],
    }, chip=chip)


async def get_community(db: Session, user: User, ctx: ToolContext, args: dict) -> ToolOutcome:
    from services import atlas_store, autointerp

    model, key, community = args.get("model", ""), args.get("sae", ""), int(args.get("community", 0))
    info, atlas = await _atlas(db, user, model, key)
    chip = f"Community {community} · {_short(info.name)}"
    if atlas is None:
        return ToolOutcome({"error": "This SAE has no atlas on this model yet"}, error=True, chip=chip)
    try:
        detail = await atlas_store.read_json(f"{atlas.directory}/communities/{community}.json")
    except FileNotFoundError:
        return ToolOutcome({"error": f"The atlas has no community {community}"}, error=True, chip=chip)
    members = sorted(detail.get("features", []), key=lambda f: -(f.get("tokens") or 0))[:COMMUNITY_FEATURES]
    labels = {r.feature: r for r in autointerp.labels(db, atlas.directory, [f["id"] for f in members])}
    name = autointerp.community_names(db, atlas).get(community)
    geometry = detail.get("geometry") or {}
    return ToolOutcome({
        "community": community, "name": name.name if name else None, "size": len(detail.get("features", [])),
        "top_tokens": (detail.get("top_words") or [])[:10],
        "geometry": {"explained_top3": _round(sum((geometry.get("explained") or [])[:3])),
                     "random_groups_top3": _round(sum((geometry.get("baseline") or [])[:3])),
                     "score": _round(geometry.get("score"))} if geometry else None,
        "features": [
            {
                "feature": f["id"], "link": feature_link(key, f["id"], model), "tokens_active": f.get("tokens"),
                "max": _round(f.get("max")), "top_token": f.get("top"),
                "explanation": labels[f["id"]].explanation if f["id"] in labels else None,
                "score": _round(labels[f["id"]].score) if f["id"] in labels else None,
            }
            for f in members
        ],
        "contexts": [_context(e["tokens"], e["at"], e.get("act"), e.get("feature")) for e in (detail.get("examples") or [])[:6]],
    }, chip=chip)


def _context(tokens: list[str], at: int, act=None, feature=None) -> dict:
    """A short context around the token a feature fires on, that token in [[ ]]."""
    lo, hi = max(0, at - CONTEXT_TOKENS), min(len(tokens), at + 4)
    text = "".join(tokens[lo:at]) + "[[" + (tokens[at] if at < len(tokens) else "") + "]]" + "".join(tokens[at + 1:hi])
    out = {"text": text}
    if act is not None:
        out["act"] = _round(act)
    if feature is not None:
        out["feature"] = feature
    return out


async def search_features(db: Session, user: User, ctx: ToolContext, args: dict) -> ToolOutcome:
    from core.autointerp import scoring
    from models.database import FeatureExplanation
    from services import autointerp

    model, key, query = args.get("model", ""), args.get("sae", ""), (args.get("query") or "").strip()
    info, atlas = await _atlas(db, user, model, key)
    chip = f"Search “{query[:24]}” · {_short(info.name)}"
    if atlas is None:
        return ToolOutcome({"error": "This SAE has no atlas on this model yet, so none of its features are explained"}, error=True, chip=chip)
    if not query:
        return ToolOutcome({"error": "Give words to search for"}, error=True, chip=chip)
    rows = (
        db.query(FeatureExplanation.feature, FeatureExplanation.explanation, FeatureExplanation.score,
                 FeatureExplanation.detection, FeatureExplanation.token_detection)
        .filter(FeatureExplanation.id.in_(autointerp.newest_ids(db, atlas.directory)), autointerp.search_filter(db, query))
        .all()
    )
    rows.sort(key=lambda r: -(r.score if r.score is not None else -1))
    names = [
        {"community": n.community, "name": n.name}
        for n in autointerp.community_names(db, atlas).values()
        if all(word.lower() in n.name.lower() for word in query.split())
    ]
    return ToolOutcome({
        "query": query, "corpus": atlas.corpus, "matches": len(rows),
        "features": [
            {
                "feature": r.feature, "link": feature_link(key, r.feature, model), "explanation": r.explanation,
                "score": _round(r.score), "token_feature": scoring.token_feature(r.detection, r.token_detection),
            }
            for r in rows[:SEARCH_RESULTS]
        ],
        "communities": names[:10],
        "note": "Scores are ±0.076; 0.5 is chance",
    }, chip=chip)


async def get_feature(db: Session, user: User, ctx: ToolContext, args: dict) -> ToolOutcome:
    from api.autointerp import feature_detail
    from api.geometry import atlas_feature_data

    model, key, feature = args.get("model", ""), args.get("sae", ""), int(args.get("feature", 0))
    info, atlas = await _atlas(db, user, model, key)
    chip = f"Feature {feature} · {_short(info.name)}"
    link = feature_link(key, feature, model)
    if atlas is None:
        return ToolOutcome({"feature": feature, "link": link, "atlas": None,
                            "note": "This SAE has no atlas on this model, so nothing is stored about its features"}, chip=chip)
    _, ref = _parse_sae(key)
    detail = await feature_detail(db, user, model, ref, None, feature)
    stats = await atlas_feature_data(atlas.directory, feature)
    newest = detail.get("explanation")
    out: dict[str, Any] = {
        "feature": feature, "link": link, "sae_name": info.name, "corpus": detail.get("corpus"),
        "explanation": _explanation_brief(newest),
        "controls_scored": detail.get("controls"),
        "earlier_versions": [
            {"explanation": v.get("explanation"), "score": _round(v.get("score")), "explainer": v.get("explainer")}
            for v in detail.get("versions", [])[1:4]
        ],
        "corpus_stats": {
            "tokens_active": stats.get("tokens"), "frequency": _round(stats.get("freq"), 6),
            "mean": _round(stats.get("mean")), "max": _round(stats.get("max")), "community": stats.get("community"),
        },
        "top_tokens": [{"token": t["token"], "share": _round(t["value"])} for t in detail.get("top_tokens", [])[:10]],
        "promotes": [t["token"] for t in detail.get("promoted", [])[:8]],
        "suppresses": [t["token"] for t in detail.get("suppressed", [])[:8]],
        "contexts": [_context(e["tokens"], e["at"], e.get("act")) for e in stats.get("examples", [])[:FEATURE_CONTEXTS]],
        "coactive": [{"feature": n["id"], "link": feature_link(key, n["id"], model)} for n in stats.get("neighbors", [])[:5]],
    }
    if not detail.get("has_sets"):
        out["note"] = "The atlas has no autointerp example sets for it (a pre-autointerp atlas, a dense SAE, or a feature never active)"
    elif not detail.get("scorable"):
        out["unscored_note"] = detail.get("unscored_note")
    hypotheses = detail.get("hypotheses", [])
    if hypotheses:
        out["your_hypotheses"] = [
            {"id": h.get("id"), "text": h.get("text"), "score": _round(h.get("score")), "detection": _round(h.get("detection")),
             "token_baseline": _round(h.get("token_detection")), "control_percentile": _round(h.get("control_percentile"), 1)}
            for h in hypotheses[:5]
        ]
    tests = detail.get("tests", [])
    if tests:
        out["generation_tests"] = [
            {"explanation": t.get("text"), "firing_rate": _round(t.get("firing_rate")), "near_miss_rate": _round(t.get("near_miss_rate")),
             "texts": len(t.get("texts") or []), "generator": t.get("generator")}
            for t in tests[:5]
        ]
    return ToolOutcome(out, chip=chip)


def _probe_brief(p) -> dict:
    metrics = p.metrics or {}
    return {
        "id": p.id, "name": p.name, "model": p.model_id, "layer": p.layer, "pooling": p.pooling, "method": p.method,
        "dataset": (p.dataset or {}).get("name"),
        "test_auroc": _round(metrics.get("test_auroc")), "control_auroc": _round(metrics.get("control_test_auroc")),
        "embeddings_auroc": _round(metrics.get("embedding_test_auroc")), "ood_auroc": _round(metrics.get("ood_auroc")),
        "model_available": p.model_available,
    }


async def list_probes(db: Session, user: User, ctx: ToolContext, args: dict) -> ToolOutcome:
    from api.probes import list_probes as saved

    probes = await saved(model=None, current_user=user, db=db)
    return ToolOutcome({"probes": [_probe_brief(p) for p in probes[:PROBES_LISTED]], "total": len(probes)})


async def get_probe(db: Session, user: User, ctx: ToolContext, args: dict) -> ToolOutcome:
    from api.probes import get_probe as saved

    probe = await saved(probe_id=int(args.get("probe_id", 0)), include_weights=False, current_user=user, db=db)
    metrics = dict(probe.metrics or {})
    sweep = metrics.pop("sweep", None) or {}
    out = {
        **_probe_brief(probe),
        "metrics": {k: _round(v) for k, v in metrics.items() if not isinstance(v, (list, dict))},
        "chat_template": probe.chat_template, "read_span": probe.read_span,
        "dataset": probe.dataset, "created": probe.created_at.isoformat(), "model_changed": probe.model_changed,
    }
    if sweep.get("test_auroc"):
        out["sweep_test_auroc_by_layer"] = [_round(v) for v in sweep["test_auroc"]]
        out["sweep_control_auroc_by_layer"] = [_round(v) for v in sweep.get("control_test_auroc") or []]
    return ToolOutcome(out, chip=f"Probe · {_short(probe.name)}")


async def list_jobs(db: Session, user: User, ctx: ToolContext, args: dict) -> ToolOutcome:
    from services.autointerp_jobs import progress
    from services.job_orchestrator import get_user_jobs

    training = get_user_jobs(db, user.id, limit=JOBS_LISTED)
    atlases = db.query(AtlasJob).filter(AtlasJob.user_id == user.id).order_by(AtlasJob.created_at.desc()).limit(JOBS_LISTED).all()
    autointerp_jobs = (
        db.query(AutointerpJob).filter(AutointerpJob.user_id == user.id).order_by(AutointerpJob.created_at.desc()).limit(JOBS_LISTED).all()
    )

    def when(value) -> Optional[str]:
        return value.isoformat() if value else None

    return ToolOutcome({
        "training": [
            {"job": j.job_id, "model_name": j.model_name, "base": j.model_config, "status": j.status.value,
             "epoch": j.current_epoch, "epochs": j.total_epochs or j.epochs, "val_loss": _round(j.val_loss),
             "error": j.error_message, "created": when(j.created_at)}
            for j in training
        ],
        "atlases": [
            {"job": j.job_id, "sae_name": j.sae_name, "model": j.model_id, "corpus": j.corpus, "tokens": j.tokens,
             "status": j.status, "error": j.error_message, "created": when(j.created_at), "finished": when(j.finished_at)}
            for j in atlases
        ],
        "autointerp": [
            {"job": j.job_id, "sae_name": j.sae_name, "model": j.model_id, "status": j.status, "stage": j.stage,
             "features": j.features, "progress": progress(j, None), "error": j.error_message,
             "created": when(j.created_at), "finished": when(j.finished_at)}
            for j in autointerp_jobs
        ],
        "note": "Statuses as last recorded (pages and background checks update them)",
    })


async def get_activation(db: Session, user: User, ctx: ToolContext, args: dict) -> ToolOutcome:
    from core.open_source import OPEN_SOURCE_MODELS
    from inference.activation_store import get_activation_store

    activation_id = (args.get("activation_id") or "").strip()
    store = get_activation_store()
    summary = store.get_activation_summary(activation_id, user.id)
    if summary is None:
        return ToolOutcome({"error": "No stored generation with that ID (they're kept in memory for a while after generating)"}, error=True)
    meta = summary.get("metadata") or {}
    tokens = meta.get("decoded_tokens") or []
    if not tokens and summary.get("model") not in OPEN_SOURCE_MODELS:
        from core.tokenizer import get_tokenizer

        tokenizer = get_tokenizer()
        tokens = [tokenizer.decode([int(t)]) for t in summary.get("tokens", [])]
    top = store.get_top_k(activation_id, user.id, TOP_PREDICTIONS)
    positions = []
    if top is not None:
        for i in range(min(len(top["tokens"]), ACTIVATION_POSITIONS)):
            positions.append({
                "pos": i,
                "token": tokens[i] if i < len(tokens) else None,
                "next": tokens[i + 1] if i + 1 < len(tokens) else None,
                "top": [[t, _round(p, 3)] for t, p in zip(top["tokens"][i], top["probabilities"][i])],
            })
    return ToolOutcome({
        "activation_id": activation_id, "model": summary.get("model"), "prompt": summary.get("prompt"),
        "generated_text": meta.get("generated_text"), "prompt_tokens": meta.get("prompt_tokens"),
        "generated_tokens": meta.get("generated_tokens"), "layers": summary.get("num_layers"),
        "tokens": tokens[:200],
        "predictions": positions,
        "note": "predictions[i].top is the model's top next-token predictions after token i (the actual next token is next)",
    })


async def get_balance(db: Session, user: User, ctx: ToolContext, args: dict) -> ToolOutcome:
    from services.assistant import free_left
    from services.credit_service import get_credit_balance

    balance = get_credit_balance(db, user.id)
    return ToolOutcome({
        "available_usd": _round(float(balance.available_balance), 2) if balance else 0.0,
        "reserved_usd": _round(float(balance.reserved_usd), 2) if balance else 0.0,
        "free_messages_left_today": free_left(db, user.id),
    })


async def estimate(db: Session, user: User, ctx: ToolContext, args: dict) -> ToolOutcome:
    action = args.get("action", "")
    handler = ESTIMATES.get(action)
    if handler is None:
        return ToolOutcome({"error": f"Unknown action {action}"}, error=True)
    result = await handler(db, user, args)
    return ToolOutcome({"action": action, **result}, chip=f"Estimate · {action.replace('_', ' ')}")


# Estimates, by action: the same functions the pages' estimates call

async def _estimate_generate(db: Session, user: User, args: dict) -> dict:
    from core.open_source import OPEN_SOURCE_MODELS
    from services.credit_service import calculate_inference_cost
    from services.custom_models import resolve_model

    model, text = args.get("model", ""), args.get("text", "")
    resolve_model(db, user.id, model)
    max_tokens = int(args.get("count") or 100)
    if model in OPEN_SOURCE_MODELS:
        prompt_tokens = max(1, len(text) // 4)
    else:
        from core.tokenizer import get_tokenizer

        prompt_tokens = len(get_tokenizer().encode(text)) or 1
    cost = calculate_inference_cost(prompt_tokens, max_tokens, model_id=model)
    return {"estimate_usd": round(cost, 4), "detail": f"{prompt_tokens} prompt + up to {max_tokens} generated tokens; settled on the tokens used"}


async def _estimate_analyze_sae(db: Session, user: User, args: dict) -> dict:
    from services.credit_service import MIN_SAE_CHARGE, estimate_sae_cost

    ref, _ = _parse_sae(args.get("sae", ""))
    model = ref["model"] if ref["kind"] == "ozera" else "external"
    cost = estimate_sae_cost(len(args.get("text", "")), model, "analyze")
    return {"estimate_usd": round(cost, 4), "detail": f"charged by tokens read, at least ${MIN_SAE_CHARGE:.2f}"}


async def _estimate_explain(db: Session, user: User, args: dict) -> dict:
    from api.autointerp import _explain_price

    await _atlas(db, user, args.get("model", ""), args.get("sae", ""))
    return {"estimate_usd": _explain_price(), "detail": "Sonnet 5.5 explains, Haiku 4.5 scores; charged its usage"}


async def _estimate_hypothesis(db: Session, user: User, args: dict) -> dict:
    from api.autointerp import _hypothesis_price

    await _atlas(db, user, args.get("model", ""), args.get("sae", ""))
    return {"estimate_usd": _hypothesis_price(), "detail": "Haiku 4.5 scores it; charged its usage"}


async def _estimate_generation_test(db: Session, user: User, args: dict) -> dict:
    from api.autointerp import _atlas_specs, _generation_test_price

    _, ref = _parse_sae(args.get("sae", ""))
    specs, _, atlas = await _atlas_specs(db, user, args.get("model", ""), ref, None)
    if atlas is None:
        raise HTTPException(status_code=404, detail="No atlas for this SAE yet")
    return {"estimate_usd": _generation_test_price(atlas.model_id, specs, atlas.width), "detail": "Claude writes the texts; one GPU call reads them"}


async def _estimate_autointerp_job(db: Session, user: User, args: dict) -> dict:
    from core.autointerp.budget import DEFAULT_EXPLAINER, DEFAULT_SCORER, MAX_JOB_FEATURES
    from services import autointerp, autointerp_jobs
    from services.credit_service import estimate_autointerp_job

    info, atlas = await _atlas(db, user, args.get("model", ""), args.get("sae", ""))
    if atlas is None:
        raise HTTPException(status_code=404, detail="No atlas for this SAE yet")
    state, reason = autointerp.state(atlas.meta)
    if state != "ready":
        raise HTTPException(status_code=400, detail=reason)
    done = autointerp_jobs.explained_with(db, atlas.directory, DEFAULT_EXPLAINER, DEFAULT_SCORER, True)
    features = max(1, min(int(args.get("count") or 1000), MAX_JOB_FEATURES))
    controls = len(autointerp_jobs.controls_to_run(atlas.width, done))
    expected, hold = estimate_autointerp_job(DEFAULT_EXPLAINER, DEFAULT_SCORER, features, features, controls)
    return {"estimate_usd": round(expected, 2), "hold_usd": round(hold, 2),
            "detail": f"{features} features (most frequent first, already explained ones skipped) and {controls} controls, scored; through the Batch API"}


async def _estimate_build_atlas(db: Session, user: User, args: dict) -> dict:
    from api.geometry import estimate_atlas
    from api.schemas.geometry import AtlasJobRequest

    _, ref = _parse_sae(args.get("sae", ""))
    tokens = int(args.get("count") or 2_000_000)
    tokens = min((1_000_000, 2_000_000, 5_000_000), key=lambda t: abs(t - tokens))
    result = await estimate_atlas(body=AtlasJobRequest(model=args.get("model", ""), sae=ref, tokens=tokens), current_user=user, db=db)
    return {"estimate_usd": round(result["estimated_cost"], 2), "tokens": tokens,
            "minutes": round(result["estimated_seconds"] / 60, 1), "within_limit": result["within_limit"]}


async def _estimate_train_probe(db: Session, user: User, args: dict) -> dict:
    from api.probes import estimate_run
    from api.schemas.probes import EstimateProbesRequest
    from core.probes.datasets import builtin_datasets, dataset_summary

    dataset = builtin_datasets().get(args.get("item", ""))
    if dataset is None:
        raise HTTPException(status_code=404, detail="Unknown built-in dataset")
    summary = dataset_summary(dataset)
    ood = summary.get("ood") or {}
    body = EstimateProbesRequest(
        model=args.get("model", ""), num_examples=summary["num_rows"] + (ood.get("num_rows") or 0),
        total_chars=summary["total_chars"] + (ood.get("total_chars") or 0),
    )
    result = await estimate_run(body=body, current_user=user, db=db)
    return {"estimate_usd": round(result["estimated_cost"], 2), "examples": body.num_examples,
            "seconds": round(result["estimated_seconds"]), "within_limit": result["within_limit"]}


async def _estimate_run_geometry(db: Session, user: User, args: dict) -> dict:
    from api.geometry import estimate_run
    from api.schemas.geometry import ConceptSpec, EstimateGeometryRequest

    body = EstimateGeometryRequest(model=args.get("model", ""), concept=ConceptSpec(builtin_id=args.get("item", "")))
    result = await estimate_run(body=body, current_user=user, db=db)
    return {"estimate_usd": round(result["estimated_cost"], 2), "examples": result["num_examples"],
            "seconds": round(result["estimated_seconds"]), "within_limit": result["within_limit"]}


async def _estimate_classify_heads(db: Session, user: User, args: dict) -> dict:
    from services.assistant_runs import quote_classify_heads

    quoted = await quote_classify_heads(db, user, ToolContext(user.id), {"model": args.get("model", ""), "prompt": args.get("text", ""), "activation_id": ""})
    return {"estimate_usd": round(quoted.estimate, 4), "detail": quoted.note}


async def _estimate_compare_probe_sae(db: Session, user: User, args: dict) -> dict:
    from services.assistant_runs import quote_compare_probe_sae

    quoted = await quote_compare_probe_sae(db, user, ToolContext(user.id), {"probe_id": int(args.get("count") or 0), "sae": args.get("sae", "")})
    return {"estimate_usd": round(quoted.estimate, 4), "detail": quoted.note}


async def _estimate_patching(db: Session, user: User, args: dict) -> dict:
    from services.credit_service import estimate_patching_cost
    from services.custom_models import resolve_model

    model, text = args.get("model", ""), args.get("text", "")
    resolve_model(db, user.id, model)
    cost = estimate_patching_cost(len(text), len(text), int(args.get("count") or 20), model)
    return {"estimate_usd": round(cost, 4), "detail": "one patch; more patches cost a little more"}


ESTIMATES: dict[str, Callable[[Session, User, dict], Awaitable[dict]]] = {
    "analyze_sae": _estimate_analyze_sae,
    "autointerp_job": _estimate_autointerp_job,
    "build_atlas": _estimate_build_atlas,
    "classify_heads": _estimate_classify_heads,
    "compare_probe_sae": _estimate_compare_probe_sae,
    "explain_feature": _estimate_explain,
    "generate": _estimate_generate,
    "generation_test": _estimate_generation_test,
    "run_geometry": _estimate_run_geometry,
    "run_patching": _estimate_patching,
    "test_hypothesis": _estimate_hypothesis,
    "train_probe": _estimate_train_probe,
}


# UI tools: checked here, applied by the browser

async def navigate(db: Session, user: User, ctx: ToolContext, args: dict) -> ToolOutcome:
    page = PAGES.get(args.get("page", ""))
    if page is None:
        return ToolOutcome({"error": "Unknown page"}, error=True)
    return ToolOutcome(
        {"status": "sent", "page": page.id, "note": f"{page.title} opens in the user's browser"},
        chip=f"Opened {page.title}", ui={"type": "navigate", "page": page.id, "route": page.route},
    )


def _control_id(page_id: str, control: Any) -> str:
    """A control ID as given: bare (model) or as the guide writes it (probes.model)."""
    control = str(control or "").strip()
    return control[len(page_id) + 1:] if control.startswith(page_id + ".") else control


def _check_model(db: Session, user: User, model_id: str) -> Optional[str]:
    from services.custom_models import resolve_model

    try:
        resolve_model(db, user.id, model_id)
        return None
    except HTTPException:
        return f"There's no model {model_id} (list_models gives the IDs)"


async def set_inputs(db: Session, user: User, ctx: ToolContext, args: dict) -> ToolOutcome:
    from models.database import UserExternalSAE
    from services.model_specs import BASE_MODEL_CONFIGS

    page = PAGES.get(args.get("page", ""))
    if page is None or not page.bridged:
        return ToolOutcome({"error": "That page has no inputs the assistant can fill"}, error=True)
    fields, problems = {}, []
    for item in args.get("fields") or []:
        control = page.control(_control_id(page.id, item.get("control")))
        if control is None:
            problems.append(f"{page.id} has no control {item.get('control')!r}")
            continue
        value, problem = check_value(control, str(item.get("value", "")))
        if problem is None and control.id == "model":
            problem = _check_model(db, user, str(value).strip())
            value = str(value).strip()
        if problem is None and control.kind == "sae":
            ref = sae_ref(str(value))
            if ref["kind"] == "external" and db.query(UserExternalSAE).filter_by(user_id=user.id, sae_id=ref["sae_id"]).first() is None:
                problem = "That SAE isn't in the user's list"
            elif ref["kind"] == "ozera" and ref["layer"] >= BASE_MODEL_CONFIGS[ref["model"]].num_layers:
                problem = f"Ozera {ref['model']} has {BASE_MODEL_CONFIGS[ref['model']].num_layers} layers"
        if problem:
            problems.append(problem)
        else:
            fields[control.id] = value
    if problems or not fields:
        return ToolOutcome({"error": "; ".join(problems) or "Give at least one control to fill"}, error=True)
    note = "Filled on screen; nothing ran" + ("" if ctx.page == page.id else "; it fills when the page opens")
    return ToolOutcome(
        {"status": "sent", "page": page.id, "set": fields, "note": note},
        chip=f"Filled {', '.join(PAGES[page.id].control(c).label for c in fields)[:60]}",
        ui={"type": "set_inputs", "page": page.id, "fields": fields},
    )


async def highlight(db: Session, user: User, ctx: ToolContext, args: dict) -> ToolOutcome:
    page = PAGES.get(args.get("page", ""))
    control = page.control(_control_id(page.id, args.get("control"))) if page else None
    if page is None or control is None:
        return ToolOutcome({"error": f"{args.get('page')} has no control {args.get('control')!r}"}, error=True)
    return ToolOutcome(
        {"status": "sent", "page": page.id, "control": control.id},
        chip=f"Pointed at {control.label}", ui={"type": "highlight", "page": page.id, "control": control.id},
    )


EXECUTORS: dict[str, Callable[[Session, User, ToolContext, dict], Awaitable[ToolOutcome]]] = {
    "estimate": estimate,
    "get_activation": get_activation,
    "get_atlas": get_atlas,
    "get_balance": get_balance,
    "get_community": get_community,
    "get_feature": get_feature,
    "get_probe": get_probe,
    "highlight": highlight,
    "list_jobs": list_jobs,
    "list_models": list_models,
    "list_probes": list_probes,
    "list_saes": list_saes,
    "navigate": navigate,
    "search_features": search_features,
    "set_inputs": set_inputs,
}


async def execute(name: str, ctx: ToolContext, args: dict) -> ToolOutcome:
    """
    Run a tool as the context's user, in a session of its own; an HTTP error is the tool's
    error. Run tools (services/assistant_runs.py) come here only once the user approved them.
    """
    from core.assistant.tools import RUN, tool

    spec = tool(name)
    if spec is not None and spec.kind == RUN:
        from services import assistant_runs

        return await assistant_runs.run(name, ctx, args)
    executor = EXECUTORS.get(name)
    if executor is None:
        return ToolOutcome({"error": f"There's no tool named {name}"}, error=True)
    db = SessionLocal()
    try:
        user = db.get(User, ctx.user_id)
        if user is None:
            return ToolOutcome({"error": "Not signed in"}, error=True)
        return await executor(db, user, ctx, args)
    except HTTPException as e:
        return ToolOutcome({"error": str(e.detail)}, error=True)
    except (ValueError, TypeError) as e:
        logger.warning("Assistant tool %s got bad input: %s", name, e)
        return ToolOutcome({"error": "Bad input to this tool"}, error=True)
    finally:
        db.close()
