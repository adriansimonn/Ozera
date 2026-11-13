"""
The assistant's pilot evaluation (ASSISTANT_PLAN.md, Cost): scripted tasks on each model with
Claude for real, against local test data (an in-memory database, a small synthetic atlas of
nano's layer-1 SAE with explanations and community names, and a saved probe). A turn that calls
run tools pauses for approval, and the pilot always answers Skip (the turn then goes on), so
nothing charged on Ozera's GPUs runs.

For every call it records the hold, what the call cost and its usage, and checks that:
- each turn called the tools the task needs (and its tool chips); a run task's turn paused for
  approval with the run it needs, and after the Skip Claude didn't call it again;
- calls after a chat's first read the cache (and later chats read the shared prefix);
- every hold covered its call's cost;
- token counts, against the plan's per-turn estimates;
- refusals (reported, not expected).

Spends real money on the configured Anthropic key: about $1 for 10 tasks on 3 models; it stops
once Anthropic's cost passes --budget.

    cd backend && python scripts/assistant_pilot.py [--models haiku,sonnet,opus] [--tasks 1,3] [--budget 1.5] [--out pilot.json]
"""

import argparse
import asyncio
import json
import os
import sys
import tempfile
from datetime import datetime

BACKEND = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BACKEND)
os.chdir(BACKEND)

# Never the configured database: everything here is in memory
from dotenv import dotenv_values  # noqa: E402

KEY = os.getenv("ANTHROPIC_API_KEY") or dotenv_values(os.path.join(BACKEND, ".env")).get("ANTHROPIC_API_KEY")
os.environ["ANTHROPIC_API_KEY"] = KEY or ""
os.environ["DATABASE_URL"] = "postgresql://nobody@127.0.0.1:1/nothing"
os.environ["INFERENCE_MODE"] = "local"
for name in ("DATABASE_USER", "DATABASE_HOST"):
    os.environ.pop(name, None)

from sqlalchemy import create_engine  # noqa: E402
from sqlalchemy.orm import sessionmaker  # noqa: E402
from sqlalchemy.pool import StaticPool  # noqa: E402

import db as dbmod  # noqa: E402
from models.database import (  # noqa: E402
    AssistantChat, Base, CommunityName, CreditBalance, FeatureExplanation, Probe, Transaction, User,
)

engine = create_engine("sqlite://", poolclass=StaticPool, connect_args={"check_same_thread": False})
Base.metadata.create_all(engine)
Session = sessionmaker(bind=engine, autoflush=False)
dbmod.SessionLocal = Session

import services.credit_service as credit_service  # noqa: E402


def _balance_for_update(db, user_id):
    return db.query(CreditBalance).filter(CreditBalance.user_id == user_id).first()


credit_service.get_credit_balance_for_update = _balance_for_update  # SQLite has no lock_timeout
import services.assistant_billing as billing  # noqa: E402

billing.get_credit_balance_for_update = _balance_for_update
import services.assistant as assistant  # noqa: E402
import services.assistant_runs as assistant_runs  # noqa: E402
import services.assistant_tools as assistant_tools  # noqa: E402
from core.assistant import models as am  # noqa: E402
from services import atlas_store  # noqa: E402

assistant.SessionLocal = Session
assistant_tools.SessionLocal = Session
assistant_runs.SessionLocal = Session

ATLAS = "/saes/nano/layer_1_residual/atlas/nano--openwebtext"
CREATED = "2026-11-01T00:00:00Z"
MODEL_KEYS = {"haiku": am.HAIKU, "sonnet": am.SONNET, "opus": am.OPUS}


def write_atlas(root: str) -> None:
    """A small synthetic atlas of nano's layer-1 SAE (the files the read tools use)."""
    base = os.path.join(root, ATLAS.lstrip("/"))
    os.makedirs(os.path.join(base, "communities"))
    os.makedirs(os.path.join(base, "features"))
    meta = {
        "model": "nano", "sae": {"kind": "ozera", "model": "nano", "layer": 1}, "sae_name": "Ozera nano SAE · layer 1 residual",
        "layer": 1, "hidden_dim": 192, "version": 2, "corpus": {"id": "openwebtext", "name": "OpenWebText"},
        "tokens": 2_000_000, "sequences": 15_625, "seq_len": 128, "skip": 4, "requested_tokens": 2_000_000,
        "stopped_early": False, "width": 1536, "seed": 0, "l0": 24.0, "autointerp": True, "created_at": CREATED,
    }
    communities = [
        {"id": 0, "size": 3, "features": [3, 7, 9], "tokens": 41_000, "top_words": [" three", " two", " 1990", " seven"], "rank": 0,
         "geometry": {"tokens": 400, "explained": [0.41, 0.22, 0.1], "baseline": [0.33, 0.2, 0.1], "score": 0.1}, "concept": True},
        {"id": 1, "size": 2, "features": [12, 15], "tokens": 22_000, "top_words": [")", "]"], "rank": 1,
         "geometry": {"tokens": 300, "explained": [0.5, 0.2, 0.1], "baseline": [0.48, 0.21, 0.1], "score": 0.01}, "concept": False},
        {"id": 2, "size": 2, "features": [20, 22], "tokens": 9_000, "top_words": [" not", " never", " no"], "rank": 2, "geometry": None, "concept": False},
    ]
    summary = {**meta, "fvu": 0.081, "alive": 1210, "communities": communities, "ranked": [0, 1, 2], "graph": {}, "geometry": {}, "autointerp": {}, "map": {}}
    vocabulary = {"0": "The", "1": " year", "2": " was", "3": " three", "4": " people", "5": " did", "6": " not", "7": " come", "8": " (", "9": "see", "10": " above", "11": ")", "12": " 1990"}
    shard = {"vocabulary": vocabulary, "features": {
        "3": {"tokens": 18_000, "mean": 2.1, "max": 9.4, "community": 0, "examples": [
            {"ids": [0, 1, 2, 3, 4], "at": 3, "act": 9.4, "levels": [0, 0, 0, 100, 10]},
            {"ids": [4, 5, 3, 7], "at": 2, "act": 8.1, "levels": [0, 0, 86, 0]}], "neighbors": [[7, 0.4, 0.3], [9, 0.2, 0.1]]},
        "7": {"tokens": 15_000, "mean": 1.7, "max": 7.0, "community": 0, "examples": [
            {"ids": [0, 1, 2, 12], "at": 3, "act": 7.0, "levels": [0, 0, 0, 100]}], "neighbors": [[3, 0.4, 0.3]]},
        "12": {"tokens": 21_000, "mean": 3.0, "max": 11.2, "community": 1, "examples": [
            {"ids": [8, 9, 10, 11], "at": 3, "act": 11.2, "levels": [0, 0, 0, 100]}], "neighbors": [[15, 0.6, 0.5]]},
        "20": {"tokens": 6_000, "mean": 1.2, "max": 5.5, "community": 2, "examples": [
            {"ids": [4, 5, 6, 7], "at": 2, "act": 5.5, "levels": [0, 20, 100, 0]}], "neighbors": [[22, 0.5, 0.4]]},
    }}
    for name, data in (("meta.json", meta), ("summary.json", summary), ("features/0.json", shard)):
        with open(os.path.join(base, name), "w") as f:
            json.dump(data, f)
    for c in communities:
        detail = {
            "id": c["id"], "features": [{"id": f, "tokens": 5000 - 100 * f, "mean": 1.5, "max": 6.0, "top": c["top_words"][0]} for f in c["features"]],
            "top_words": c["top_words"], "concept": None, "concept_note": None, "geometry": c["geometry"],
            "examples": [{"feature": c["features"][0], "tokens": ["The", " year", " was", c["top_words"][0], " ago"], "at": 3, "act": 6.0, "levels": [0, 0, 0, 100, 0]}],
        }
        with open(os.path.join(base, "communities", f"{c['id']}.json"), "w") as f:
            json.dump(detail, f)

    async def read_raw(path: str) -> bytes:
        with open(os.path.join(root, path.lstrip("/")), "rb") as f:
            return f.read()

    async def list_raw(path: str) -> list[str]:
        full = os.path.join(root, path.lstrip("/"))
        return sorted(os.listdir(full)) if os.path.isdir(full) else []

    atlas_store._read_raw = read_raw
    atlas_store._list_raw = list_raw


def seed_data() -> int:
    """The pilot user (with credits, and today's free messages used, so every call is held), explanations and a probe."""
    db = Session()
    user = User(email="pilot@example.com", is_active=True)
    db.add(user)
    db.commit()
    db.add(CreditBalance(user_id=user.id, balance_usd=20, reserved_usd=0))
    common = dict(atlas_key=ATLAS, sae_key="/saes/nano/layer_1_residual", model_id="nano", corpus="openwebtext",
                  atlas={"created_at": CREATED}, explainer_model="claude-sonnet-5-5", scorer_model="claude-haiku-4-5",
                  prompt_version=2, examples_version=1, status="scored", created_at=datetime(2026, 11, 2))
    rows = [
        (3, "the word 'three' and other spelled-out numbers", 0.82, 0.86, 0.78, 0.84),
        (7, "years in dates, like 1990", 0.74, 0.79, 0.69, 0.62),
        (12, "closing parentheses after a reference", 0.69, 0.72, 0.66, 0.70),
        (20, "negation words like not and never", 0.71, 0.75, 0.67, 0.60),
    ]
    for feature, text, score, detection, fuzzing, token in rows:
        db.add(FeatureExplanation(**common, feature=feature, is_control=False, explanation=text, score=score, detection=detection,
                                  fuzzing=fuzzing, token_detection=token, detection_hard=detection - 0.1, details={}))
    for i, score in enumerate((0.52, 0.55, 0.58, 0.6, 0.61, 0.63, 0.64, 0.66, 0.7, 0.72)):
        db.add(FeatureExplanation(**common, feature=2000 + i, is_control=True, explanation="control", score=score, detection=score, fuzzing=score))
    db.add(CommunityName(atlas_key=ATLAS, sae_key="/saes/nano/layer_1_residual", model_id="nano", corpus="openwebtext",
                         atlas_created_at=CREATED, community=0, name="Numbers and years", namer_model="claude-sonnet-5-5",
                         prompt_version=1, explained=2, size=3, created_at=datetime(2026, 11, 3)))
    db.add(Probe(user_id=user.id, name="sentiment L4 mean", model_id="nano", layer=4, pooling="mean", method="logreg", hidden_dim=192,
                 weights=b"\x00" * 768, bias=0.1, normalization={}, dataset={"name": "Sentiment", "builtin_id": "sentiment"},
                 metrics={"test_auroc": 0.71, "control_test_auroc": 0.62, "embedding_test_auroc": 0.66, "test_acc": 0.66,
                          "control_test_acc": 0.58, "selectivity": 0.08}))
    db.commit()
    uid = user.id
    for _ in range(am.FREE_MESSAGES_PER_DAY):
        assistant.claim_free(db, uid)
    db.close()
    return uid


SAE_PAGE = {"route": "/sae", "page": "sae", "mode": "browse", "sae": "ozera:nano:1", "text": "", "analysis": None, "selected_feature": None,
            "corpus_atlas": {"model": "nano", "corpus": "openwebtext", "tokens": 2_000_000}}
PROBES_PAGE = {"route": "/probes", "page": "probes", "view": "sweep", "model": "nano", "dataset": "truth", "run": None, "saved_probes": 1}

TASKS = [
    ("What models can I use here?", {"route": "/"}, {"list_models"}),
    ("Which SAEs can I read on nano, and which have explained features?", {"route": "/sae"}, {"list_saes"}),
    ("What does feature 3 of this SAE do? How good is its explanation?", SAE_PAGE, {"get_feature"}),
    ("Find features about negation in this SAE.", SAE_PAGE, {"search_features"}),
    ("Summarize this SAE's atlas.", SAE_PAGE, {"get_atlas"}),
    ("What's in community 0 of this atlas?", SAE_PAGE, {"get_community"}),
    ("How much would explaining one feature of this SAE cost?", SAE_PAGE, {"estimate"}),
    ("Set up the Probe Lab for the sentiment dataset on Ozera mini, and show me where to start training.", {"route": "/sae", "page": "sae"}, {"navigate", "set_inputs"}),
    ("Is my saved probe any good?", PROBES_PAGE, {"list_probes"}),
    ("What's my balance, and how many free messages do I have left?", {"route": "/settings"}, {"get_balance"}),
    # Runs (Phase 2): each pauses for approval, and is skipped
    ("Run 'The cat sat on the mat because it was tired' through this SAE.", SAE_PAGE, {"analyze_sae"}),
    ("Feature 9 has no explanation yet. Explain it.", SAE_PAGE, {"explain_feature"}),
    ("Score my hypothesis that feature 20 fires on negations at the start of a clause.", SAE_PAGE, {"test_hypothesis"}),
    ("Train probes on the sentiment dataset on nano.", PROBES_PAGE, {"train_probe"}),
]
RUN_TASKS = {11, 12, 13, 14}


class Recorder:
    """Wraps a turn's hold and settle to record each call."""

    def __init__(self, turn):
        self.calls = []
        hold, settle, keep, release = turn.hold, turn.settle, turn.keep, turn.release

        def held(bound):
            h = hold(bound)
            self.calls.append({"bound_tokens": bound, "hold": h.amount})
            return h

        def settled(h, attempts):
            charged = settle(h, attempts)
            cost, charge = billing.attempts_cost(attempts)
            self.calls[-1].update(charged=charged, charge=charge, anthropic=cost,
                                  attempts=[{"model": m, **u.to_dict()} for m, u in attempts])
            return charged

        def kept(h, why):
            self.calls[-1].update(kept=why)
            return keep(h, why)

        def released(h):
            self.calls[-1].update(released=True)
            return release(h)

        turn.hold, turn.settle, turn.keep, turn.release = held, settled, kept, released


async def run_task(client, uid: int, model_id: str, text: str, snapshot: dict) -> dict:
    db = Session()
    chat = assistant.create_chat(db, uid, model_id)
    assistant.claim_turn(db, chat)
    turn = assistant.start_turn(db, chat, uid, text, snapshot)
    db.close()
    recorder = Recorder(turn)
    events: asyncio.Queue = asyncio.Queue()
    await assistant.run(turn, events, client=client)
    out = []
    while not events.empty():
        out.append(events.get_nowait())
    paused = None
    db = Session()
    chat = db.get(AssistantChat, chat.id)
    if chat.status == "awaiting_approval" and chat.pending:
        # Skip every run, and let the turn go on (nothing charged runs)
        paused = [r["name"] for r in chat.pending["runs"]]
        pending, decisions, errors = await assistant.check_approval(db, chat, uid, {r["id"]: "skip" for r in chat.pending["runs"]})
        assistant.claim_approval(db, chat)
        resumed = assistant.resume(db, chat, uid, pending)
        db.close()
        resumed_recorder = Recorder(resumed)
        await assistant.run(resumed, events, client=client, approval=(pending, decisions, errors))
        recorder.calls += resumed_recorder.calls
        while not events.empty():
            out.append(events.get_nowait())
        db = Session()
        chat = db.get(AssistantChat, chat.id)
    detail = assistant.turns(db, chat)[-1]
    db.close()
    return {"events": out, "turn": detail, "calls": recorder.calls, "paused": paused}


async def main(args) -> None:
    import anthropic

    if not KEY:
        sys.exit("No ANTHROPIC_API_KEY (environment or backend/.env)")
    client = anthropic.AsyncAnthropic(api_key=KEY, max_retries=0, timeout=am.CALL_TIMEOUT_SECONDS)
    with tempfile.TemporaryDirectory() as root:
        write_atlas(root)
        uid = seed_data()
        from core.assistant.loop import static_tokens

        statics = {m: await static_tokens(client, am.MODELS[m]) for m in (MODEL_KEYS[k] for k in args.models)}
        print("Static prefix (counted, with margin):", statics)
        spent, results = 0.0, []
        tasks = [(i, TASKS[i - 1]) for i in args.tasks]
        for key in args.models:
            model_id = MODEL_KEYS[key]
            for n, (text, snapshot, expected) in tasks:
                if spent > args.budget:
                    print(f"Stopping: ${spent:.3f} spent, over the ${args.budget} budget")
                    break
                r = await run_task(client, uid, model_id, text, snapshot)
                spent += sum(c.get("anthropic", 0) for c in r["calls"])
                called = {t["name"] for t in r["turn"]["tools"]}
                if n in RUN_TASKS:
                    # Paused with the run, and it wasn't called again after the Skip
                    names = [t["name"] for t in r["turn"]["tools"]]
                    retried = any(names.count(x) > 1 for x in expected)
                    called = called if (r["paused"] and expected <= set(r["paused"]) and not retried) else set()
                reads = [c["attempts"][0]["cache_read_tokens"] for c in r["calls"] if c.get("attempts")]
                covered = all(c.get("charged", c["hold"]) <= c["hold"] + 1e-9 and c.get("charge", 0) <= c["hold"] + 1e-9 for c in r["calls"])
                row = {
                    "model": key, "task": n, "status": r["turn"]["status"], "tools_ok": expected <= called, "called": sorted(called),
                    "chips": [t["chip"] for t in r["turn"]["tools"]], "calls": len(r["calls"]), "cache_reads": reads,
                    "holds_cover": covered, "cost": r["turn"]["cost"], "anthropic": round(sum(c.get("anthropic", 0) for c in r["calls"]), 5),
                    "input": sum(a["input_tokens"] + a["cache_write_tokens"] + a["cache_write_1h_tokens"] + a["cache_read_tokens"] for c in r["calls"] for a in c.get("attempts", [])),
                    "output": sum(a["output_tokens"] for c in r["calls"] for a in c.get("attempts", [])),
                    "max_hold_use": round(max((c.get("charge", 0) / c["hold"] for c in r["calls"] if c["hold"]), default=0), 3),
                    "detail": r["turn"]["detail"], "reply": r["turn"]["reply"], "raw_calls": r["calls"],
                }
                results.append(row)
                print(f"{key:6} {n:2} {row['status']:9} tools {'ok ' if row['tools_ok'] else 'MISS'} {row['called']} calls {row['calls']} "
                      f"reads {reads} cover {row['holds_cover']} cost ${row['cost']:.4f} in {row['input']} out {row['output']} hold use {row['max_hold_use']}")
                print("        " + row["reply"].replace("\n", " ")[:260])
        db = Session()
        ledger = sum(float(t.amount_usd) for t in db.query(Transaction).filter_by(user_id=uid))
        balance = float(db.query(CreditBalance).filter_by(user_id=uid).first().balance_usd)
        db.close()
        print(f"\nAnthropic cost ${spent:.4f}; charged ${-ledger:.4f} (ledger) = ${20 - balance:.4f} (balance change)")
        if args.out:
            with open(args.out, "w") as f:
                json.dump({"statics": statics, "spent": spent, "results": results}, f, indent=1, default=str)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", default="haiku,sonnet,opus", type=lambda s: s.split(","))
    parser.add_argument("--tasks", default=",".join(str(i) for i in range(1, len(TASKS) + 1)), type=lambda s: [int(x) for x in s.split(",")])
    parser.add_argument("--budget", default=1.5, type=float)
    parser.add_argument("--out", default=None)
    asyncio.run(main(parser.parse_args()))
