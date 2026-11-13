"""
Names for an atlas's communities (AUTOINTERP_PLAN.md, Phase 3).

A community (core.geometry.atlas) is a group of features that are often active on the same
tokens. One request per community names what its features share, from their explanations (the
most active explained ones), the tokens the community fires on most, and contexts its features
are active in; the Atlas view shows the name in place of the community's number.

A community is named once COMMUNITY_MIN_SHARE of its features (at least
COMMUNITY_MIN_EXPLAINED) have explanations: a name written from a few of them would stand for
the rest. Community IDs belong to one atlas build (a recomputed atlas groups features afresh),
so a name is kept with the build it was made from.
"""

import json
import math
from collections.abc import Container, Mapping, Sequence
from dataclasses import dataclass
from typing import Optional

from .budget import (
    COMMUNITY_CONTEXTS,
    COMMUNITY_MEMBERS,
    COMMUNITY_MIN_EXPLAINED,
    COMMUNITY_MIN_SHARE,
    COMMUNITY_TOP_WORDS,
    MAX_COMMUNITY_NAME_CHARS,
    MODELS,
)
from .client import Reply, request_params
from .examples import escape, highlighted
from .pipeline import Prompt
from .prompts import NAME_SCHEMA, NAME_SYSTEM, naming_message


@dataclass
class NamePlan:
    """A community to name."""

    community: int
    size: int
    members: list[int]  # The explained features its prompt shows, most active first
    explained: int  # Its features with an explanation


def needed(size: int) -> int:
    """Features of a community of this size that must have explanations before it's named."""
    return max(COMMUNITY_MIN_EXPLAINED, math.ceil(COMMUNITY_MIN_SHARE * size))


def plan_names(
    communities: Sequence[dict],
    explained_features: Container[int],
    done: Optional[Mapping[int, int]] = None,
) -> list[NamePlan]:
    """
    The communities to name, in the atlas's order: those with enough explained features, less
    those already named from as many (done: {community: explained features when it was named}).
    communities: an atlas summary's entries ({"id", "size", "features" most active first, ...});
    explained_features: the features with an explanation (a set, or explanations by feature).
    """
    done = done or {}
    plans = []
    for community in communities:
        explained = [f for f in community["features"] if f in explained_features]
        if len(explained) < needed(community["size"]) or done.get(community["id"], -1) >= len(explained):
            continue
        plans.append(NamePlan(community["id"], community["size"], explained[:COMMUNITY_MEMBERS], len(explained)))
    return plans


def naming_prompt(community: dict, plan: NamePlan, explanations: Mapping[int, str], contexts: Sequence[dict], namer: str) -> Prompt:
    """
    The request naming a community. contexts: its stored contexts ({"tokens", "levels"}, a
    feature's activation on each token), the first COMMUNITY_CONTEXTS of which are shown.
    """
    descriptions = [escape(explanations[f]) for f in plan.members]
    tokens = [
        json.dumps(escape(word.strip()), ensure_ascii=False)
        for word, _ in community.get("top_words", [])[:COMMUNITY_TOP_WORDS] if word.strip()
    ]
    examples = [highlighted(c["tokens"], [level > 0 for level in c["levels"]]) for c in contexts[:COMMUNITY_CONTEXTS]]
    user = naming_message(descriptions, tokens, examples)
    return Prompt("name", request_params(namer, NAME_SYSTEM, user, NAME_SCHEMA, MODELS[namer].name_max_tokens))


def parse_name(reply: Reply) -> Optional[str]:
    """The name, whitespace collapsed and capped, with the prompts' lookalikes for < and > read back."""
    data = reply.json()
    text = data.get("name") if data else None
    if not isinstance(text, str):
        return None
    text = " ".join(text.split()).replace("‹", "<").replace("›", ">").strip(" \"'.")
    return text[:MAX_COMMUNITY_NAME_CHARS].strip() or None
