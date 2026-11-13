"""
Whole-SAE autointerp jobs (AUTOINTERP_PLAN.md, Phase 2): a selection of an atlas's features, and
its random-direction controls, explained and scored through the Message Batches API (half price;
most batches end within an hour, all within 24 h).

A job runs in stages, each a short CPU function on the SAE volume (services/modal_feature_stats.py).
The backend (services/autointerp_jobs.py) starts each one, waits on the batches between them
(asking Anthropic how they're doing is a cheap call, so no container waits), and charges for it:

  prepare          build every explain request, and count the input of every request the job may
                   send (scoring ones with room for any explanation): the job's upper bound, which
                   the backend holds before anything is sent
  submit           send the explain requests, as batches
  collect explain  read the explanations, then send the scoring requests as batches (or, with
                   nothing to score, write the results)
  collect score    score each explanation, and write the results

A job can instead name the atlas's communities (core.autointerp.communities), from explanations
the backend already stores: one request per community, built and counted in prepare, sent with
the explain batches, and read in collect explain (which writes names.json).

Files, in the job's directory (job_directory):

  plan.json          the job: the atlas build it reads, models, seed, its features and controls
                     (and which are scored), and its upper bound in tokens
  explain.jsonl      the explain requests ({"custom_id", "params"} a line), until they're read back
  batches.json       every batch sent, by stage and part ({stage: {part: [batch ID, requests]}}),
                     recorded as soon as it exists, so a stage that stopped part-way resumes
                     without sending a part twice (and the backend knows what was sent)
  explanations.json  each feature's explanation, or why it has none, and the tokens it used
  results/N.jsonl    each feature's result, RESULTS_PART a file, for the backend to import
  names.json         each community's name, or why it has none, for the backend to import

Batch requests are named e-<feature> (its explain request), s-<feature>-<i> (its scoring
prompt i, in scoring_prompts' order) and c-<community> (its naming request).
"""

import json
import os
import random
import shutil
from collections.abc import Callable, Iterator

from .budget import MAX_EXPLANATION_TOKENS, RESULTS_PART
from .client import ClaudeClient, Reply, Usage
from .communities import naming_prompt, parse_name, plan_names
from .examples import AtlasDirectory, FeatureSets
from .pipeline import FeatureResult, Prompt, apply_explanation, apply_scores, explain_prompt, scoring_prompts
from .prompts import PROMPT_VERSION

JOBS_DIRECTORY = "autointerp_jobs"
# A scoring sample (and calibration's fixed set) takes as many features from each band of frequency
FREQUENCY_BANDS = 5
# Features prepared (and their prompts counted) at a time
PREPARE_CHUNK = 512
SAMPLED_OUT = "Not in the job's scoring sample"

Progress = Callable[[str, float], None]


def _quiet(stage: str, fraction: float) -> None:
    pass


class JobError(Exception):
    """A job can't go on (shown to the user)."""


def stratified(features: list[tuple[int, int]], n: int, seed: int) -> list[int]:
    """n of (feature, tokens it's active on) pairs' features, sorted: as many from each band of frequency."""
    ordered = sorted(features, key=lambda x: x[1])
    rng = random.Random(seed)
    bands = [ordered[i * len(ordered) // FREQUENCY_BANDS:(i + 1) * len(ordered) // FREQUENCY_BANDS] for i in range(FREQUENCY_BANDS)]
    out = []
    for i, band in enumerate(bands):
        take = n // FREQUENCY_BANDS + (1 if i < n % FREQUENCY_BANDS else 0)
        out += [f for f, _ in rng.sample(band, min(take, len(band)))]
    return sorted(out)


def job_directory(sae_directory: str, job_id: str) -> str:
    """Where a job keeps its files: next to the SAE, outside its atlases (which a recompute replaces)."""
    return f"{sae_directory}/{JOBS_DIRECTORY}/{job_id}"


class JobFiles:
    """A job's directory. on_write runs after each file that must survive a crash is written (a volume commit)."""

    def __init__(self, directory: str, on_write: Callable[[], None] = lambda: None):
        self.directory = directory
        self.on_write = on_write
        os.makedirs(directory, exist_ok=True)

    def path(self, name: str) -> str:
        return os.path.join(self.directory, name)

    def read(self, name: str, default=None):
        try:
            with open(self.path(name)) as f:
                return json.load(f)
        except FileNotFoundError:
            if default is None:
                raise JobError(f"The job's {name} is missing")
            return default

    def write(self, name: str, data, commit: bool = True) -> None:
        tmp = self.path(name + ".tmp")
        with open(tmp, "w") as f:
            json.dump(data, f, ensure_ascii=False, separators=(",", ":"))
        os.replace(tmp, self.path(name))
        if commit:
            self.on_write()

    def batches(self, stage: str) -> dict[int, str]:
        """{part: batch ID} of the batches a stage has sent."""
        return {int(part): entry[0] for part, entry in self.read("batches.json", {}).get(stage, {}).items()}

    def record_batch(self, stage: str, part: int, batch_id: str, requests: int) -> None:
        batches = self.read("batches.json", {})
        batches.setdefault(stage, {})[str(part)] = [batch_id, requests]
        self.write("batches.json", batches)


def _atlas(atlas_dir: str, plan: dict) -> AtlasDirectory:
    atlas = AtlasDirectory(atlas_dir)
    if atlas.meta.get("created_at") != plan["atlas_created_at"]:
        raise JobError("The atlas was recomputed during the job; submit it again")
    return atlas


def _add(totals: dict[str, dict[str, int]], model_id: str, input_tokens: int, output_tokens: int) -> None:
    entry = totals.setdefault(model_id, {"input": 0, "output": 0})
    entry["input"] += input_tokens
    entry["output"] += output_tokens


def _usage_add(totals: dict[str, Usage], replies) -> None:
    for reply in replies:
        totals[reply.model] = totals.get(reply.model, Usage()) + reply.usage


def _usage_dict(totals: dict[str, Usage]) -> dict[str, dict]:
    return {model_id: usage.to_dict() for model_id, usage in totals.items()}


async def _bound(client: ClaudeClient, prompts: list[Prompt], overheads: dict, totals: dict) -> None:
    """Add prompts' upper bound to totals: their input counted (with room for any explanation where one goes) and max_tokens."""
    if not prompts:
        return
    for model_id, n in (await client.count_total([p.params for p in prompts], overheads)).items():
        _add(totals, model_id, n, 0)
    for p in prompts:
        extra = MAX_EXPLANATION_TOKENS if p.kind in ("detection", "fuzzing") else 0
        _add(totals, p.params["model"], extra, p.params["max_tokens"])


# Stages

async def prepare(job: JobFiles, atlas_dir: str, spec: dict, client: ClaudeClient, progress: Progress = _quiet) -> dict:
    """
    Build a job's explain requests and count its upper bound. spec: {"job_id", "explainer",
    "scorer", "seed", "atlas_created_at", "features", "scored", "controls"} (features to score
    are a subset of features; controls are always scored). Features never active over the corpus
    are dropped, and those that can't be scored are explained only.

    With spec["naming"] ({"explanations": {feature: text}, "done": {community: explained}}), a
    request is also built for every community to name from those explanations (plan_names).
    """
    atlas = _atlas(atlas_dir, spec)
    explainer, scorer, seed = spec["explainer"], spec["scorer"], int(spec["seed"])
    controls = set(spec["controls"])
    to_score = set(spec["scored"]) | controls
    ids = sorted(set(spec["features"]) | controls)  # in shard order
    kept, scored, dead = [], [], []
    bound: dict[str, dict] = {"explain": {}, "score": {}}
    requests = 0
    overheads: dict[str, int] = {}
    with open(job.path("explain.jsonl.tmp"), "w") as out:
        for c0 in range(0, len(ids), PREPARE_CHUNK):
            explains, scores = [], []
            for f in ids[c0:c0 + PREPARE_CHUNK]:
                sets = atlas.sets(f)
                if sets is None:
                    dead.append(f)
                    continue
                kept.append(f)
                prompt = explain_prompt(sets, explainer, seed)
                out.write(json.dumps({"custom_id": f"e-{f}", "params": prompt.params}, ensure_ascii=False) + "\n")
                explains.append(prompt)
                if f in to_score and sets.scorable:
                    scored.append(f)
                    scores += scoring_prompts(sets, "", scorer, seed)
            await _bound(client, explains, overheads, bound["explain"])
            await _bound(client, scores, overheads, bound["score"])
            requests += len(scores)
            progress("Counting tokens", min(1.0, (c0 + PREPARE_CHUNK) / max(len(ids), 1)))
        names = await _prepare_names(out, atlas_dir, spec.get("naming"), explainer, client, overheads, bound["explain"], progress)
    if not kept and not names:
        raise JobError("Nothing to explain or name: none of the selected features was active over the corpus")
    os.replace(job.path("explain.jsonl.tmp"), job.path("explain.jsonl"))
    plan = {
        **{key: spec[key] for key in ("job_id", "explainer", "scorer", "seed", "atlas_created_at")},
        "atlas": atlas_dir,
        "prompt_version": PROMPT_VERSION,
        "width": atlas.width,
        "features": [f for f in kept if f not in controls],
        "controls": [f for f in kept if f in controls],
        "scored": scored,
        "dead": dead,
        "names": names,
        "bound": bound,
    }
    job.write("plan.json", plan)
    return {
        "bound": bound,
        "features": len(plan["features"]),
        "controls": len(plan["controls"]),
        "scored": len(scored),
        "scored_features": sum(1 for f in scored if f not in controls),
        "dead": len(dead),
        "communities": len(names),
        "requests": {"explain": len(kept), "score": requests, "name": len(names)},
    }


async def _prepare_names(out, atlas_dir: str, naming, namer: str, client: ClaudeClient, overheads: dict, bound: dict, progress: Progress) -> list[dict]:
    """
    Write a naming request for every community to name (see plan_names) and add their upper
    bound: their text is final, so it's their exact input and max_tokens. Returns each one's
    {"community", "size", "explained"}.
    """
    if not naming:
        return []
    with open(os.path.join(atlas_dir, "summary.json")) as f:
        communities = json.load(f)["communities"]
    explanations = {int(f): text for f, text in naming["explanations"].items()}
    plans = plan_names(communities, explanations, {int(c): int(n) for c, n in (naming.get("done") or {}).items()})
    by_id = {c["id"]: c for c in communities}
    names = []
    for c0 in range(0, len(plans), PREPARE_CHUNK):
        prompts = []
        for plan in plans[c0:c0 + PREPARE_CHUNK]:
            try:
                with open(os.path.join(atlas_dir, "communities", f"{plan.community}.json")) as f:
                    contexts = json.load(f).get("examples", [])
            except FileNotFoundError:
                contexts = []
            prompt = naming_prompt(by_id[plan.community], plan, explanations, contexts, namer)
            out.write(json.dumps({"custom_id": f"c-{plan.community}", "params": prompt.params}, ensure_ascii=False) + "\n")
            prompts.append(prompt)
            names.append({"community": plan.community, "size": plan.size, "explained": plan.explained})
        await _bound(client, prompts, overheads, bound)
        progress("Counting tokens", min(1.0, (c0 + PREPARE_CHUNK) / len(plans)))
    return names


def _names(plan: dict, replies: dict[str, Reply], namer: str) -> tuple[list[dict], dict]:
    """Each planned community's name from its reply (named, refused or failed), and how many of each."""
    rows = []
    counts = {"named": 0, "refused": 0, "failed": 0}
    for entry in plan.get("names") or []:
        reply = replies.get(f"c-{entry['community']}") or Reply(namer, error="No reply")
        name = parse_name(reply)
        status = "named" if name else "refused" if reply.refused else "failed"
        counts[status] += 1
        rows.append({**entry, "name": name, "status": status, "usage": {namer: reply.usage.to_dict()}})
    return rows, counts


async def submit(job: JobFiles, client: ClaudeClient) -> dict:
    """Send the explain requests as batches (parts already sent by an earlier attempt aren't sent again)."""

    def requests() -> Iterator[tuple[str, dict]]:
        with open(job.path("explain.jsonl")) as f:
            for line in f:
                entry = json.loads(line)
                yield entry["custom_id"], entry["params"]

    ids = await client.submit_parts(requests(), job.batches("explain"), lambda part, batch_id, n: job.record_batch("explain", part, batch_id, n))
    return {"batches": ids}


async def _replies(client: ClaudeClient, job: JobFiles, stage: str, model_id: str) -> dict[str, Reply]:
    """Every reply of a stage's batches, by custom ID."""
    replies: dict[str, Reply] = {}
    for batch_id in job.batches(stage).values():
        replies.update(await client.batch_replies(batch_id, model_id))
    return replies


def _result_state(result: FeatureResult, score: bool) -> dict:
    return {
        "status": result.status, "explanation": result.explanation, "note": result.note,
        "problems": result.problems, "usage": result.total_usage, "score": score,
    }


def _result_from_state(feature: int, explainer: str, scorer: str, state: dict) -> FeatureResult:
    return FeatureResult(
        feature, explainer, scorer, status=state["status"], explanation=state["explanation"], note=state["note"],
        usage={model_id: Usage.from_dict(u) for model_id, u in state["usage"].items()}, problems=list(state["problems"]),
    )


def result_row(result: FeatureResult, is_control: bool, sampled_out: bool = False) -> dict:
    """A feature's line in the results (what the backend stores as its explanation)."""
    details = {**result.details, "problems": result.problems[:20]}
    if sampled_out:
        details["sampled_out"] = True
    return {
        "feature": result.feature,
        "is_control": is_control,
        "status": result.status,
        "explanation": result.explanation,
        "note": result.note,
        "detection": result.detection,
        "detection_hard": result.detection_hard,
        "fuzzing": result.fuzzing,
        "score": result.score,
        "token_detection": result.token_detection,
        "decile_recall": result.decile_recall,
        "details": details,
        "verdicts": result.verdicts,
        "usage": result.total_usage,
    }


def _write_results(job: JobFiles, rows: list[dict]) -> dict:
    """Write the results in parts (replacing any from an earlier attempt); returns their summary."""
    tmp = job.path("results.tmp")
    shutil.rmtree(tmp, ignore_errors=True)
    os.makedirs(tmp)
    parts = 0
    for p0 in range(0, len(rows), RESULTS_PART):
        with open(os.path.join(tmp, f"{parts}.jsonl"), "w") as f:
            for row in rows[p0:p0 + RESULTS_PART]:
                f.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
        parts += 1
    shutil.rmtree(job.path("results"), ignore_errors=True)
    os.replace(tmp, job.path("results"))
    statuses: dict[str, int] = {}
    for row in rows:
        statuses[row["status"]] = statuses.get(row["status"], 0) + 1
    job.on_write()
    return {"parts": parts, "statuses": statuses}


async def collect_explain(job: JobFiles, client: ClaudeClient, progress: Progress = _quiet) -> dict:
    """
    Read the explain batches' replies, then send the scoring requests of every explanation to
    score as batches. With nothing to score, the results are written instead.
    """
    plan = job.read("plan.json")
    atlas = _atlas(plan["atlas"], plan)
    explainer, scorer, seed = plan["explainer"], plan["scorer"], int(plan["seed"])
    progress("Reading explanations", 0.0)
    replies = await _replies(client, job, "explain", explainer)
    usage: dict[str, Usage] = {}
    _usage_add(usage, replies.values())

    controls = set(plan["controls"])
    to_score = set(plan["scored"])
    states: dict[str, dict] = {}
    scoring: list[int] = []
    ids = plan["features"] + plan["controls"]
    for i, f in enumerate(ids):
        sets = atlas.sets(f)
        result = FeatureResult(f, explainer, scorer)
        reply = replies.get(f"e-{f}") or Reply(explainer, error="No reply")
        score = apply_explanation(result, sets, reply) and f in to_score
        if score:
            scoring.append(f)
        elif result.explanation is not None and result.status == "failed":
            result.status, result.note = "unscored", SAMPLED_OUT
        states[str(f)] = _result_state(result, score)
        if i % 256 == 0:
            progress("Reading explanations", i / max(len(ids), 1))
    job.write("explanations.json", states, commit=False)
    names, named = _names(plan, replies, explainer)
    if names:
        job.write("names.json", names, commit=False)
    # The requests are in the batches now
    try:
        os.remove(job.path("explain.jsonl"))
    except FileNotFoundError:
        pass
    job.on_write()

    report = {
        "usage": _usage_dict(usage),
        "explained": sum(1 for s in states.values() if s["explanation"] is not None),
        "refused": sum(1 for s in states.values() if s["status"] == "refused"),
        "failed": sum(1 for s in states.values() if s["explanation"] is None and s["status"] == "failed"),
        "scoring": len(scoring),
    }
    if names:
        report["names"] = named
    if not scoring:
        rows = [
            result_row(_result_from_state(f, explainer, scorer, states[str(f)]), f in controls, states[str(f)]["note"] == SAMPLED_OUT)
            for f in ids
        ]
        report["results"] = _write_results(job, rows)
        return report

    def requests() -> Iterator[tuple[str, dict]]:
        for n, f in enumerate(scoring):
            for i, p in enumerate(scoring_prompts(atlas.sets(f), states[str(f)]["explanation"], scorer, seed)):
                yield f"s-{f}-{i}", p.params
            if n % 256 == 0:
                progress("Sending scoring requests", n / len(scoring))

    report["batches"] = await client.submit_parts(
        requests(), job.batches("score"), lambda part, batch_id, n: job.record_batch("score", part, batch_id, n),
    )
    return report


async def collect_score(job: JobFiles, client: ClaudeClient, progress: Progress = _quiet) -> dict:
    """Read the scoring batches' replies, score each explanation, and write the results."""
    plan = job.read("plan.json")
    atlas = _atlas(plan["atlas"], plan)
    explainer, scorer, seed = plan["explainer"], plan["scorer"], int(plan["seed"])
    states = job.read("explanations.json")
    progress("Reading scores", 0.0)
    replies = await _replies(client, job, "score", scorer)
    usage: dict[str, Usage] = {}
    _usage_add(usage, replies.values())

    controls = set(plan["controls"])
    ids = plan["features"] + plan["controls"]
    rows = []
    for n, f in enumerate(ids):
        state = states[str(f)]
        result = _result_from_state(f, explainer, scorer, state)
        if state["score"]:
            sets: FeatureSets = atlas.sets(f)
            prompts = scoring_prompts(sets, state["explanation"], scorer, seed)
            apply_scores(result, prompts, [replies.get(f"s-{f}-{i}") or Reply(scorer, error="No reply") for i in range(len(prompts))])
        rows.append(result_row(result, f in controls, state["note"] == SAMPLED_OUT))
        if n % 256 == 0:
            progress("Scoring", n / max(len(ids), 1))
    return {"usage": _usage_dict(usage), "results": _write_results(job, rows)}

