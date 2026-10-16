"""
Tokenizing a concept's examples and finding where each one is read: the value's tokens, or
the template's last token.

Open-source targets map tokens to characters with offset mappings, so a value's tokens are
those overlapping its characters. Ozera targets (tiktoken, no offsets) tokenize the prefix and
the prefix plus value separately; a template is refused when filling it changes how the text
around the value tokenizes (the value would share a token with it, as "{x}th" can).

A value's leading space belongs to its first token in every tokenizer used here (GPT-2 and
Llama-style BPE attach it to the next word, SentencePiece writes it as ▁), so templates put a
space before {x} and the space reads as part of the value.
"""

from dataclasses import dataclass
from typing import Sequence

from core.probes.runner import EncodedText, ProbeTarget

from .budget import GeometryInputError
from .concepts import PLACEHOLDER, READS, fill


@dataclass
class ExampleTokens:
    """An example's token IDs, its value's tokens [value_start, value_end), and where its text ends."""

    ids: list[int]
    value_start: int
    value_end: int
    text_end: int  # One past the template text's last token (chat template tokens follow it)


def _merge_error(template: str, value: str) -> GeometryInputError:
    return GeometryInputError(
        f'In "{fill(template, value)}", "{value}" shares a token with the text around it; '
        f"put a space or punctuation on both sides of {PLACEHOLDER}"
    )


def _ozera_example(tokenizer, template: str, value: str) -> ExampleTokens:
    prefix, suffix = template.split(PLACEHOLDER)
    # A trailing space would tokenize on its own; next to the value it joins the value's first token
    ids_prefix = tokenizer.encode(prefix.rstrip(" ")) if prefix.strip() else []
    ids_value = tokenizer.encode(prefix + value)
    ids = tokenizer.encode(prefix + value + suffix)
    if (
        ids_value[:len(ids_prefix)] != ids_prefix
        or ids[:len(ids_value)] != ids_value
        or len(ids_value) <= len(ids_prefix)
    ):
        raise _merge_error(template, value)
    return ExampleTokens(ids, len(ids_prefix), len(ids_value), len(ids))


def _open_source_example(target, template: str, value: str, chat_template: bool) -> ExampleTokens:
    tokenizer = target.tokenizer
    text = fill(template, value)
    if chat_template:
        rendered = tokenizer.apply_chat_template(
            target.loader.chat_messages(text),
            add_generation_prompt=True,
            enable_thinking=False,
            tokenize=False,
        )
        base = rendered.rfind(text)
        encoding = tokenizer(rendered, add_special_tokens=False, return_offsets_mapping=True)
    else:
        rendered, base = text, 0
        encoding = tokenizer(text, return_offsets_mapping=True)

    ids = list(encoding["input_ids"])
    offsets = encoding["offset_mapping"]
    prefix = template.split(PLACEHOLDER)[0]
    a = base + len(prefix)
    b = a + len(value)
    in_value = [i for i, (s, e) in enumerate(offsets) if e > s and s < b and e > a]
    in_text = [i for i, (s, e) in enumerate(offsets) if e > s and s < base + len(text) and e > base]
    if not in_value or not in_text:
        raise GeometryInputError(f'Couldn\'t find "{value}" in the tokens of "{text}"')
    first_start = offsets[in_value[0]][0]
    if offsets[in_value[-1]][1] > b or rendered[first_start:a].strip():
        raise _merge_error(template, value)
    return ExampleTokens(ids, in_value[0], in_value[-1] + 1, in_text[-1] + 1)


def encode_example(target: ProbeTarget, template: str, value: str, chat_template: bool = False) -> ExampleTokens:
    """
    One example's tokens and spans.

    Raises:
        GeometryInputError: the value shares a token with the template's text, or the model
            has no chat template
    """
    if target.model_type == "ozera":
        if chat_template:
            raise GeometryInputError("Ozera models have no chat template")
        return _ozera_example(target.tokenizer, template, value)
    if chat_template and not target.loader.config.is_instruct:
        raise GeometryInputError(f"{target.loader.config.display_name} has no chat template")
    return _open_source_example(target, template, value, chat_template)


def read_span(example: ExampleTokens, read: str) -> tuple[int, int]:
    """The positions [start, end) an example is read at (pooled by their mean)."""
    if read == "value_last":
        return example.value_end - 1, example.value_end
    if read == "value_mean":
        return example.value_start, example.value_end
    if read == "template_last":
        return example.text_end - 1, example.text_end
    raise GeometryInputError(f"Unknown read position: {read}")


def encode_concept(
    target: ProbeTarget,
    values: Sequence[str],
    templates: Sequence[str],
    read: str,
    chat_template: bool = False,
) -> tuple[list[EncodedText], list[dict]]:
    """
    Tokenize every example (template by template) with the span it's read at.

    Returns:
        (examples for core.probes.runner.collect_features with "mean" pooling,
         per value {"tokens": its tokens in the first template, "counts": the distinct numbers
         of tokens it splits into over the templates})

    Raises:
        GeometryInputError: a template doesn't tokenize cleanly around a value, or an example
            is longer than the model can read
    """
    if read not in READS:
        raise GeometryInputError(f"Unknown read position: {read}")
    encoded: list[EncodedText] = []
    info = [{"tokens": [], "counts": set()} for _ in values]
    for t, template in enumerate(templates):
        for v, value in enumerate(values):
            example = encode_example(target, template, value, chat_template)
            if len(example.ids) > target.max_tokens:
                raise GeometryInputError(f"Template {t + 1} is longer than {target.max_tokens} tokens")
            start, end = read_span(example, read)
            encoded.append(EncodedText(example.ids, start, end))
            info[v]["counts"].add(example.value_end - example.value_start)
            if t == 0:
                info[v]["tokens"] = [target.token_text(i) for i in example.ids[example.value_start:example.value_end]]
    return encoded, [{"tokens": entry["tokens"], "counts": sorted(entry["counts"])} for entry in info]
