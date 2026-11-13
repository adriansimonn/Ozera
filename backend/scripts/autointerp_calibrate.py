"""
Autointerp calibration (AUTOINTERP_PLAN.md, Phase 1), before any UI ships: run core.autointerp
on one v2 atlas through the Message Batches API and measure what the plan's estimates assume.

Setup: Gemma 3 270M with Gemma Scope 2 (layer 9, 16k), on an OpenWebText atlas of 2M tokens.
Copy the atlas out of the SAE volume first, e.g.:

    modal volume get ozera-saes external/<sae_id>/atlas/gemma-3-270m--openwebtext calibration/atlas

Stages (each saves its batches and replies under --out, so an interrupted stage resumes):

    show    no requests: print a feature's prompts, to check highlights and escaping
    pilot   20 features + 8 controls with the default models (~$0.30): tokens per request
            against the estimates, parse failures, refusals, and the shuffled-explanation check
            (feature A's explanation scored on feature B's set, which should sit near 0.5)
    full    200 fixed features (seeded, stratified by frequency, saved in features.json) and
            the 64 controls (~$12):
              A  Sonnet 5.5 explains, Haiku 4.5 scores (kept as real labels)
              B  Sonnet 5.5 re-scores A's explanations (scorer agreement and shift)
              C  A again with reshuffled prompts (seed 1): the ± per score
            then report.json (measured token constants for core/autointerp/budget.py and a
            TOKEN_MARGIN) and hand_check.csv (50 explanations with their contexts to check by hand)
    import  store run A in Postgres (feature_explanations), so the SAE starts with labels

Usage:
    ANTHROPIC_API_KEY=... python backend/scripts/autointerp_calibrate.py pilot --atlas calibration/atlas --out calibration
"""

import argparse
import asyncio
import csv
import json
import math
import os
import random
import statistics
import sys
import time
from datetime import datetime
from typing import Optional

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from core.autointerp import budget  # noqa: E402
from core.autointerp import examples as ai  # noqa: E402
from core.autointerp.client import ClaudeClient, Reply, Usage  # noqa: E402
from core.autointerp.jobs import stratified  # noqa: E402
from core.autointerp.pipeline import (  # noqa: E402
    FeatureResult,
    Prompt,
    explain_example,
    explain_prompt,
    parse_explanation,
    score_replies,
    scoring_prompts,
)
from core.autointerp.prompts import PROMPT_VERSION  # noqa: E402
from core.geometry.atlas import CATALOG_COLUMNS  # noqa: E402

POLL_SECONDS = 60
FULL_FEATURES = 200
PILOT_FEATURES = 20
PILOT_CONTROLS = 8
HAND_CHECK = 50
SEED = 0


class LocalAtlas:
    """A v2 atlas copied from the SAE volume."""

    def __init__(self, directory: str):
        self.directory = directory
        with open(os.path.join(directory, "meta.json")) as f:
            self.meta = json.load(f)
        with open(os.path.join(directory, "summary.json")) as f:
            self.summary = json.load(f)
        if int(self.meta.get("version", 1)) < 2 or not self.meta.get("autointerp"):
            sys.exit("This atlas has no autointerp example sets (recompute it, or the SAE is too dense)")
        base = os.path.join(directory, ai.DIRECTORY)

        def read(name: str) -> bytes:
            with open(os.path.join(base, name), "rb") as f:
                return f.read()

        self.files = ai.load_files(read("tokens.bin"), read("read.bin"), read("vocabulary.json"), self.meta["seq_len"], self.meta["skip"])
        self.catalog = np.fromfile(os.path.join(directory, "catalog.bin"), dtype="<f4").reshape(-1, CATALOG_COLUMNS)
        self._shards: dict[int, ai.Shard] = {}
        self._read = read

    @property
    def width(self) -> int:
        return int(self.meta["width"])

    @property
    def controls(self) -> list[int]:
        info = self.summary["autointerp"]["controls"]
        return list(range(info["first_id"], info["first_id"] + info["count"]))

    def shard(self, feature: int) -> ai.Shard:
        n = feature // ai.SHARD
        if n not in self._shards:
            self._shards[n] = ai.Shard.from_bytes(self._read(f"{n}.npz"))
        return self._shards[n]

    def sets(self, feature: int) -> Optional[ai.FeatureSets]:
        return ai.feature_sets(self.files, self.shard(feature), feature)

    def scorable(self) -> list[tuple[int, int]]:
        """(feature, tokens it's active on) of every live feature that can be scored."""
        out = []
        for row in self.catalog:
            f = int(row[0])
            shard = self.shard(f)
            if f in shard and shard.arrays["unscored"][shard.index[f]] == ai.SCORABLE:
                out.append((f, int(row[1])))
        return out


# Batches, saved so a stage resumes

def _reply_to_json(reply: Reply) -> dict:
    return {"model": reply.model, "text": reply.text, "stop_reason": reply.stop_reason, "usage": reply.usage.to_dict(), "error": reply.error}


def _reply_from_json(data: dict) -> Reply:
    return Reply(data["model"], data["text"], data["stop_reason"], Usage.from_dict(data["usage"]), data["error"])


async def run_batch(client: ClaudeClient, out: str, name: str, requests: dict[str, dict]) -> dict[str, Reply]:
    """Submit requests (custom_id → params) as a batch, wait for it, and return the replies (cached on disk)."""
    done_path = os.path.join(out, f"{name}.replies.json")
    if os.path.exists(done_path):
        with open(done_path) as f:
            return {key: _reply_from_json(value) for key, value in json.load(f).items()}
    ids_path = os.path.join(out, f"{name}.batches.json")
    if os.path.exists(ids_path):
        with open(ids_path) as f:
            batch_ids = json.load(f)
    else:
        batch_ids = await client.submit_batches(list(requests.items()))
        with open(ids_path, "w") as f:
            json.dump(batch_ids, f)
        print(f"{name}: submitted {len(requests)} requests in {len(batch_ids)} batch(es)")
    started = time.time()
    while True:
        statuses = [await client.batch_status(b) for b in batch_ids]
        if all(s["status"] == "ended" for s in statuses):
            break
        left = sum(s["counts"]["processing"] for s in statuses)
        print(f"{name}: {left} requests processing ({(time.time() - started) / 60:.0f} min)")
        await asyncio.sleep(POLL_SECONDS)
    models = {key: params["model"] for key, params in requests.items()}
    replies: dict[str, Reply] = {}
    for b in batch_ids:
        replies.update(await client.batch_replies(b, models))
    with open(done_path, "w") as f:
        json.dump({key: _reply_to_json(r) for key, r in replies.items()}, f)
    return replies


# A run: explain, then score

def _usage_add(total: dict[str, Usage], reply: Reply) -> None:
    total[reply.model] = total.get(reply.model, Usage()) + reply.usage


async def explain_all(client, out, name, atlas: LocalAtlas, features: list[int], explainer: str, seed: int) -> tuple[dict[int, Optional[str]], dict]:
    prompts = {f"e-{f}": explain_prompt(atlas.sets(f), explainer, seed) for f in features}
    replies = await run_batch(client, out, f"{name}-explain", {key: p.params for key, p in prompts.items()})
    explanations, stats = {}, {"refused": [], "unparsed": [], "tokens": []}
    for f in features:
        reply = replies.get(f"e-{f}") or Reply(explainer, error="missing")
        stats["tokens"].append(reply.usage.to_dict())
        if reply.refused:
            stats["refused"].append(f)
        explanation = parse_explanation(reply)
        if explanation is None and not reply.refused:
            stats["unparsed"].append([f, reply.error or reply.text])
        explanations[f] = explanation
    return explanations, stats


async def score_all(
    client, out, name, atlas: LocalAtlas, explanations: dict[int, Optional[str]], explainer: str, scorer: str, seed: int,
) -> tuple[dict[int, FeatureResult], dict]:
    prompts: dict[int, list[Prompt]] = {}
    requests = {}
    for f, explanation in explanations.items():
        sets = atlas.sets(f)
        if explanation is None or not sets.scorable:
            continue
        prompts[f] = scoring_prompts(sets, explanation, scorer, seed)
        for i, p in enumerate(prompts[f]):
            requests[f"s-{f}-{i}"] = p.params
    replies = await run_batch(client, out, f"{name}-score", requests)
    results, tokens = {}, []
    for f, explanation in explanations.items():
        result = FeatureResult(f, explainer, scorer, explanation=explanation)
        if f in prompts:
            feature_replies = [replies.get(f"s-{f}-{i}") or Reply(scorer, error="missing") for i in range(len(prompts[f]))]
            for r in feature_replies:
                _usage_add(result.usage, r)
            score_replies(result, prompts[f], feature_replies)
            result.status = "scored" if result.detection is not None and result.fuzzing is not None else "refused"
            tokens.append(sum((r.usage for r in feature_replies), Usage()).to_dict())
        elif explanation is None:
            result.status = "refused"
        else:
            result.status, result.note = "unscored", atlas.sets(f).note
        results[f] = result
    return results, {"tokens": tokens}


async def shuffled_check(client, out, name, atlas, explanations: dict[int, Optional[str]], scorer: str, seed: int) -> Optional[float]:
    """Each feature's explanation scored by detection on another feature's set (a pipeline check: ≈0.5)."""
    real = [f for f, e in explanations.items() if e is not None and f < atlas.width and atlas.sets(f).scorable]
    if len(real) < 2:
        return None
    others = real[1:] + real[:1]
    requests, prompts = {}, {}
    for f, g in zip(real, others):
        detection = [p for p in scoring_prompts(atlas.sets(g), explanations[f], scorer, seed, baseline=False) if p.kind == "detection"]
        prompts[f] = (g, detection)
        for i, p in enumerate(detection):
            requests[f"x-{f}-{i}"] = p.params
    replies = await run_batch(client, out, f"{name}-shuffled", requests)
    scores = []
    for f, (g, detection) in prompts.items():
        result = FeatureResult(g, "", scorer)
        score_replies(result, detection, [replies.get(f"x-{f}-{i}") or Reply(scorer, error="missing") for i in range(len(detection))])
        if result.detection is not None:
            scores.append(result.detection)
    return statistics.mean(scores) if scores else None


# Reporting

def _mean(values) -> Optional[float]:
    values = [v for v in values if v is not None]
    return statistics.mean(values) if values else None


def _quantiles(values) -> Optional[dict]:
    values = sorted(v for v in values if v is not None)
    if not values:
        return None
    at = lambda q: values[min(len(values) - 1, int(q * len(values)))]  # noqa: E731
    return {"n": len(values), "mean": round(statistics.mean(values), 3), "p10": round(at(0.1), 3), "median": round(at(0.5), 3), "p90": round(at(0.9), 3)}


def _spearman(a: list[float], b: list[float]) -> Optional[float]:
    if len(a) < 3:
        return None
    def ranks(x):
        order = sorted(range(len(x)), key=lambda i: x[i])
        r = [0.0] * len(x)
        for rank, i in enumerate(order):
            r[i] = rank
        return r
    ra, rb = ranks(a), ranks(b)
    return float(np.corrcoef(ra, rb)[0, 1])


def token_report(explain_tokens: list[dict], score_tokens: list[dict]) -> dict:
    explain_in = _mean([t["input_tokens"] + t["cache_read_tokens"] + t["cache_write_tokens"] for t in explain_tokens])
    explain_out = _mean([t["output_tokens"] for t in explain_tokens])
    score_in = _mean([t["input_tokens"] + t["cache_read_tokens"] + t["cache_write_tokens"] for t in score_tokens])
    score_out = _mean([t["output_tokens"] for t in score_tokens])
    return {
        "measured": {"explain_input": explain_in, "explain_output": explain_out, "score_input": score_in, "score_output": score_out},
        "estimates": {
            "explain_input": budget.EXPLAIN_INPUT_TOKENS, "explain_output": budget.EXPLAIN_OUTPUT_TOKENS,
            "score_input": budget.SCORE_INPUT_TOKENS, "score_output": budget.SCORE_OUTPUT_TOKENS,
        },
    }


def _scores(results: dict[int, FeatureResult], ids: list[int], key: str) -> list[Optional[float]]:
    return [getattr(results[f], key) for f in ids if f in results]


# Where each kind of verdict row (core.autointerp.pipeline.score_replies) holds the verdict
VERDICT_COLUMN = {"detection": 3, "fuzzing": 2, "token": 3}


def compare_runs(a: dict[int, FeatureResult], b: dict[int, FeatureResult], ids: list[int], same_prompts: bool = False) -> dict:
    """Score agreement between two runs of the same features (verdict by verdict when they scored the same prompts)."""
    pairs = [(a[f], b[f]) for f in ids if f in a and f in b and a[f].score is not None and b[f].score is not None]
    if not pairs:
        return {}
    diff = [y.score - x.score for x, y in pairs]
    delta = [(y.detection - y.token_detection) - (x.detection - x.token_detection) for x, y in pairs if x.token_detection is not None and y.token_detection is not None]
    # Verdict by verdict, where both runs scored the same explanation on the same prompts (B vs A)
    agree = total = 0
    for x, y in pairs:
        if same_prompts and x.verdicts and y.verdicts:
            for kind, at in VERDICT_COLUMN.items():
                for vx, vy in zip(x.verdicts[kind], y.verdicts[kind]):
                    if vx[at] is not None and vy[at] is not None:
                        total += 1
                        agree += vx[at] == vy[at]
    sd = statistics.stdev(diff) if len(diff) > 1 else None
    sd_delta = statistics.stdev(delta) if len(delta) > 1 else None
    return {
        "features": len(pairs),
        "mean_shift": round(statistics.mean(diff), 4),
        "spearman": _spearman([x.score for x, _ in pairs], [y.score for _, y in pairs]),
        "verdict_agreement": round(agree / total, 4) if total else None,
        # One run's ± (95%): the difference of two runs has twice a run's variance
        "score_pm": round(1.96 * sd / math.sqrt(2), 4) if sd is not None else None,
        "token_delta_pm": round(1.96 * sd_delta / math.sqrt(2), 4) if sd_delta is not None else None,
    }


def summarize(results: dict[int, FeatureResult], features: list[int], controls: list[int]) -> dict:
    scored = [f for f in features if f in results and results[f].status == "scored"]
    return {
        "features": _quantiles(_scores(results, features, "score")),
        "controls": _quantiles(_scores(results, controls, "score")),
        "token_baseline": _quantiles(_scores(results, features, "token_detection")),
        "detection": _quantiles(_scores(results, features, "detection")),
        "detection_hard": _quantiles(_scores(results, features, "detection_hard")),
        "fuzzing": _quantiles(_scores(results, features, "fuzzing")),
        "token_features": sum(1 for f in scored if (results[f].detection or 0) - (results[f].token_detection or 0) < budget.TOKEN_MARGIN),
        "scored": len(scored),
        "statuses": {s: sum(1 for f in features + controls if f in results and results[f].status == s) for s in ("scored", "unscored", "refused", "failed")},
        "problems": sorted({p for f in results for p in results[f].problems})[:20],
    }


def cost_of(out: str, batches: list[str]) -> dict:
    """What every reply of these batches (run_batch names) cost: explain and checks included."""
    from services.credit_service import calculate_autointerp_cost, claude_cost

    totals: dict[str, Usage] = {}
    for name in batches:
        with open(os.path.join(out, f"{name}.replies.json")) as f:
            for value in json.load(f).values():
                reply = _reply_from_json(value)
                totals[reply.model] = totals.get(reply.model, Usage()) + reply.usage
    return {
        "anthropic": round(sum(claude_cost(m, u, batch=True) for m, u in totals.items()), 4),
        "charged_with_markup": round(calculate_autointerp_cost(totals, batch=True), 4),
    }


def save_results(out: str, name: str, results: dict[int, FeatureResult]) -> None:
    rows = {
        str(f): {
            "explanation": r.explanation, "status": r.status, "note": r.note, "detection": r.detection,
            "detection_hard": r.detection_hard, "fuzzing": r.fuzzing, "score": r.score, "token_detection": r.token_detection,
            "decile_recall": r.decile_recall, "details": r.details, "verdicts": r.verdicts, "usage": r.total_usage,
            "explainer": r.explainer, "scorer": r.scorer,
        }
        for f, r in results.items()
    }
    with open(os.path.join(out, f"{name}.results.json"), "w") as f:
        json.dump(rows, f)


def hand_check(out: str, atlas: LocalAtlas, results: dict[int, FeatureResult], features: list[int]) -> None:
    rng = random.Random(SEED)
    picked = rng.sample([f for f in features if results.get(f) and results[f].explanation], min(HAND_CHECK, len(features)))
    with open(os.path.join(out, "hand_check.csv"), "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["feature", "explanation", "score", "detection", "fuzzing", "token_detection", "contexts (top 4, then 4 from the deciles)", "accurate (y/n)", "too broad", "too narrow", "input/output confused", "notes"])
        for feature in sorted(picked):
            r = results[feature]
            sets = atlas.sets(feature)
            contexts = sets.explain[:4] + sets.explain[ai.EXPLAIN_TOP:ai.EXPLAIN_TOP + 4]
            writer.writerow([
                feature, r.explanation, r.score, r.detection, r.fuzzing, r.token_detection,
                "\n\n".join(explain_example(c) for c in contexts), "", "", "", "", "",
            ])


def confirm(what: str, cost: float, yes: bool) -> None:
    print(f"{what}: about ${cost:.2f} at batch prices (with Ozera's markup; Anthropic bills about ${cost / 1.3:.2f})")
    if not yes and input("Proceed? [y/N] ").strip().lower() != "y":
        sys.exit("Stopped")


# Stages

def show(atlas: LocalAtlas, feature: Optional[int]) -> None:
    feature = feature if feature is not None else stratified(atlas.scorable(), 1, SEED)[0]
    sets = atlas.sets(feature)
    print(f"Feature {feature}: {len(sets.explain)} explain, {len(sets.activating)} activating, {len(sets.random)} random, {len(sets.token)} token-matched; top tokens {sets.top_tokens[:5]}")
    explain = explain_prompt(sets, budget.DEFAULT_EXPLAINER, SEED)
    print("\n=== Explain system\n" + explain.params["system"] + "\n\n=== Explain message\n" + explain.params["messages"][0]["content"][:4000])
    for p in scoring_prompts(sets, "an example explanation", budget.DEFAULT_SCORER, SEED):
        if p.kind in ("detection", "fuzzing"):
            print(f"\n=== {p.kind} message\n" + p.params["messages"][0]["content"][:2500])
    print("\nPromoted:", sets.promoted, "\nSuppressed:", sets.suppressed)


async def pilot(client: ClaudeClient, atlas: LocalAtlas, out: str, yes: bool) -> dict:
    features = stratified(atlas.scorable(), PILOT_FEATURES, SEED)
    controls = atlas.controls[:PILOT_CONTROLS]
    ids = features + controls
    from services.credit_service import estimate_autointerp_cost

    # The shuffled check adds a detection pass (5 of the 13 scoring prompts) per feature
    confirm("Pilot", estimate_autointerp_cost(budget.DEFAULT_EXPLAINER, budget.DEFAULT_SCORER, len(ids), batch=True) * 1.4, yes)
    explanations, explain_stats = await explain_all(client, out, "pilot", atlas, ids, budget.DEFAULT_EXPLAINER, SEED)
    results, score_stats = await score_all(client, out, "pilot", atlas, explanations, budget.DEFAULT_EXPLAINER, budget.DEFAULT_SCORER, SEED)
    save_results(out, "pilot", results)
    shuffled = await shuffled_check(client, out, "pilot", atlas, explanations, budget.DEFAULT_SCORER, SEED)
    report = {
        "stage": "pilot",
        "features": features,
        "controls": controls,
        "tokens": token_report(explain_stats["tokens"], score_stats["tokens"]),
        "refused_explanations": explain_stats["refused"],
        "unparsed_explanations": explain_stats["unparsed"],
        "shuffled_detection": shuffled,
        "summary": summarize(results, features, controls),
        "explanations": {str(f): explanations[f] for f in ids},
        "cost": cost_of(out, ["pilot-explain", "pilot-score", "pilot-shuffled"]),
    }
    return report


async def full(client: ClaudeClient, atlas: LocalAtlas, out: str, yes: bool) -> dict:
    path = os.path.join(out, "features.json")
    if os.path.exists(path):
        with open(path) as f:
            features = json.load(f)["features"]
    else:
        features = stratified(atlas.scorable(), FULL_FEATURES, SEED)
        with open(path, "w") as f:
            json.dump({"features": features, "seed": SEED, "atlas": atlas.meta}, f)
    controls = atlas.controls
    ids = features + controls
    from services.credit_service import estimate_autointerp_cost

    confirm(
        "Full run (A, B, C)",
        estimate_autointerp_cost(budget.SONNET, budget.HAIKU, len(ids), batch=True) * 2
        + estimate_autointerp_cost(budget.SONNET, budget.SONNET, len(ids), batch=True)
        - estimate_autointerp_cost(budget.SONNET, budget.SONNET, len(ids), scored=False, batch=True),
        yes,
    )
    # A: Sonnet explains, Haiku scores
    explanations, explain_stats = await explain_all(client, out, "A", atlas, ids, budget.SONNET, SEED)
    a, a_stats = await score_all(client, out, "A", atlas, explanations, budget.SONNET, budget.HAIKU, SEED)
    save_results(out, "A", a)
    # B: Sonnet re-scores A's explanations
    b, _ = await score_all(client, out, "B", atlas, explanations, budget.SONNET, budget.SONNET, SEED)
    save_results(out, "B", b)
    # C: A again, reshuffled
    explanations_c, _ = await explain_all(client, out, "C", atlas, ids, budget.SONNET, SEED + 1)
    c, _ = await score_all(client, out, "C", atlas, explanations_c, budget.SONNET, budget.HAIKU, SEED + 1)
    save_results(out, "C", c)
    hand_check(out, atlas, a, features)

    rerun = compare_runs(a, c, features)
    margin = rerun.get("token_delta_pm")
    report = {
        "stage": "full",
        "features": features,
        "controls": controls,
        "tokens": token_report(explain_stats["tokens"], a_stats["tokens"]),
        "A": summarize(a, features, controls),
        "B_vs_A": {**summarize(b, features, controls), "agreement": compare_runs(a, b, features, same_prompts=True)},
        "C_vs_A": rerun,
        "suggested": {
            "TOKEN_MARGIN": round(max(0.02, margin), 3) if margin is not None else None,
            "score_pm": rerun.get("score_pm"),
        },
        "refused_explanations": explain_stats["refused"],
        "cost": cost_of(out, ["A-explain", "A-score", "B-score", "C-explain", "C-score"]),
    }
    return report


def import_run(atlas: LocalAtlas, out: str, name: str) -> None:
    """Store a run's results as explanations (labels) in Postgres."""
    from core.geometry.atlas import atlas_directory
    from core.probes.sae import sae_directory
    from db import SessionLocal
    from models.database import FeatureExplanation

    with open(os.path.join(out, f"{name}.results.json")) as f:
        rows = json.load(f)
    sae_key = sae_directory(atlas.meta["sae"])
    key = atlas_directory(sae_key, atlas.meta["model"], atlas.meta["corpus"]["id"])
    job_id = f"calibration-{name}"
    build = {k: atlas.meta.get(k) for k in ("created_at", "tokens", "seed", "version")}
    db = SessionLocal()
    try:
        if db.query(FeatureExplanation).filter(FeatureExplanation.atlas_key == key, FeatureExplanation.job_id == job_id).first():
            sys.exit(f"Run {name} is already imported for this atlas")
        for feature, r in rows.items():
            if r["status"] not in ("scored", "unscored", "refused"):
                continue
            db.add(FeatureExplanation(
                atlas_key=key, sae_key=sae_key, model_id=atlas.meta["model"], corpus=atlas.meta["corpus"]["id"], atlas=build,
                feature=int(feature), is_control=int(feature) >= atlas.width, explanation=r["explanation"],
                explainer_model=r["explainer"], scorer_model=r["scorer"], prompt_version=PROMPT_VERSION,
                examples_version=ai.EXAMPLES_VERSION, status=r["status"], note=r["note"], detection=r["detection"],
                detection_hard=r["detection_hard"], fuzzing=r["fuzzing"], score=r["score"], token_detection=r["token_detection"],
                decile_recall=r["decile_recall"], details=r["details"], verdicts=r["verdicts"], usage=r["usage"],
                job_id=job_id, created_at=datetime.utcnow(),
            ))
        db.commit()
        print(f"Imported {len(rows)} results of run {name} into {key}")
    finally:
        db.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("stage", choices=("show", "pilot", "full", "import"))
    parser.add_argument("--atlas", required=True, help="A v2 atlas directory copied from the SAE volume")
    parser.add_argument("--out", default="calibration", help="Where batches, replies and reports go")
    parser.add_argument("--feature", type=int, default=None, help="show: the feature to print")
    parser.add_argument("--run", default="A", help="import: the run to store (A by default)")
    parser.add_argument("--yes", action="store_true", help="Don't ask before spending")
    args = parser.parse_args()

    atlas = LocalAtlas(args.atlas)
    os.makedirs(args.out, exist_ok=True)
    if args.stage == "show":
        show(atlas, args.feature)
        return
    if args.stage == "import":
        import_run(atlas, args.out, args.run)
        return
    client = ClaudeClient()
    report = asyncio.run(pilot(client, atlas, args.out, args.yes) if args.stage == "pilot" else full(client, atlas, args.out, args.yes))
    path = os.path.join(args.out, f"{args.stage}.report.json")
    with open(path, "w") as f:
        json.dump(report, f, indent=1, default=str)
    print(json.dumps({k: v for k, v in report.items() if k not in ("explanations", "features", "controls")}, indent=1, default=str))
    print(f"Report: {path}")


if __name__ == "__main__":
    main()
