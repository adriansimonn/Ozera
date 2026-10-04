"""
Splitting probe datasets without leaking between the parts.

Examples in the same group always land on the same side: minimal pairs (both versions of a
sentence) and exact duplicate texts. Otherwise a probe could be tested on the other half of
a pair it was trained on.
"""

import random
from typing import Hashable, Optional, Sequence


def group_ids(texts: Sequence[str], groups: Optional[Sequence[Optional[Hashable]]] = None) -> list[int]:
    """
    One group number per example: examples share one if they share a given group or a text.

    Args:
        texts: The examples' texts
        groups: Optional group per example (None for an example in no group)
    """
    parent = list(range(len(texts)))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    first_seen: dict = {}
    for i, text in enumerate(texts):
        keys = [("text", text)]
        if groups is not None and groups[i] is not None:
            keys.append(("group", groups[i]))
        for key in keys:
            if key in first_seen:
                parent[find(i)] = find(first_seen[key])
            else:
                first_seen[key] = i

    numbers: dict[int, int] = {}
    return [numbers.setdefault(find(i), len(numbers)) for i in range(len(texts))]


def split_indices(
    labels: Sequence[int],
    groups: Sequence[int],
    fraction: float,
    seed: int,
) -> tuple[list[int], list[int]]:
    """
    Split examples into two parts, keeping groups together and stratifying by label.

    Groups are stratified by their mix of labels (single positives, single negatives, pairs
    with one of each, ...), and about `fraction` of each stratum goes to the second part.

    Returns:
        (first part's indices, second part's indices), each sorted

    Raises:
        ValueError: either part would be missing a class
    """
    members: dict[int, list[int]] = {}
    for i, group in enumerate(groups):
        members.setdefault(group, []).append(i)

    strata: dict[tuple, list[int]] = {}
    for group, idx in members.items():
        strata.setdefault(tuple(sorted({labels[i] for i in idx})), []).append(group)

    rng = random.Random(seed)
    second: list[int] = []
    for key in sorted(strata):
        stratum = strata[key]
        rng.shuffle(stratum)
        take = round(len(stratum) * fraction)
        if len(stratum) >= 2:
            take = min(max(take, 1), len(stratum) - 1)
        for group in stratum[:take]:
            second.extend(members[group])

    second_set = set(second)
    first = [i for i in range(len(labels)) if i not in second_set]
    for part in (first, second):
        if len({labels[i] for i in part}) < 2:
            raise ValueError("Not enough examples of each class to split this dataset; add more examples")
    return first, sorted(second)
