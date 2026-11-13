"""
Autointerp: Claude explains SAE features from the contexts a v2 atlas keeps for each one, and
scores the explanations on held-out contexts next to baselines (see AUTOINTERP_PLAN.md).

No torch here: core.geometry.atlas builds the example sets on the GPU; this package reads them,
builds prompts, scores replies, and talks to the Anthropic SDK.
"""
