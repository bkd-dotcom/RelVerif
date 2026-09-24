"""LLM relational extraction: prose -> typed-predicate AST.

The system under test. Given a single normative sentence, the model must emit a
RelationalRule (premises -> conclusions) over the controlled vocabulary in
``schemas_relational.GLOSSARY``. This is the genuinely-harder task the
crypto claims rest on: not a boolean atom but *typed predicates with roles*.

The extractor is given the vocabulary as a role-hint block (retrieval: glossary +
role hints) plus one worked example, and is
asked for strict JSON. It sees ONLY the source sentence — never the gold rule — so
scoring is a real held-out extraction measurement, not a reconstruction.

Determinism/reproducibility mirrors ``gold_annotator``: temperature-0 (client's
responsibility) + on-disk cache keyed by (model, source). A transient parse failure
is never cached. The client is injected (anything exposing ``generate(prompt) ->
(text, meta)``) so tests run offline with a fake.
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any, Protocol

from pipeline.schemas_relational import (
    GLOSSARY,
    RelationalRule,
    glossary_prompt_block,
)


class Generator(Protocol):
    def generate(self, prompt: str) -> tuple[str, Any]: ...


# one worked example anchors the output shape (uses only vocabulary predicates)
_EXAMPLE = """[EXAMPLE]
SENTENCE: "A UE shall accept the authentication only if it has verified the MAC."
OUTPUT:
{"premises": [{"name": "accepts", "args": ["s", "u"], "sorts": ["Session", "UE"], "negated": false}],
 "conclusions": [{"name": "mac_verified", "args": ["s", "u"], "sorts": ["Session", "UE"], "negated": false}]}"""

_PROMPT = """[TASK]
You extract a typed relational rule from one sentence of a security standard. Emit
predicates from the controlled vocabulary below — do NOT invent predicate names or
sorts. Use IF (premises) THEN (conclusions) structure that matches the sentence's
conditional/causal meaning. Mark a predicate "negated": true for a negated fact
(e.g. "shall not", "fails to"). Keep the premise as the trigger/condition and the
conclusion as the required outcome; do not swap them.

[VOCABULARY]
{glossary}

{example}

[SENTENCE]
{source}

[OUTPUT]
Return ONLY a JSON object, no prose:
{{"premises": [{{"name": "...", "args": ["..."], "sorts": ["..."], "negated": false}}],
 "conclusions": [{{"name": "...", "args": ["..."], "sorts": ["..."], "negated": false}}]}}
An empty premise list means the conclusion holds unconditionally.
"""


def build_prompt(source_text: str) -> str:
    return _PROMPT.format(
        glossary=glossary_prompt_block(),
        example=_EXAMPLE,
        source=str(source_text).strip(),
    )


def parse_rule(text: str, rule_id: str, source_text: str) -> RelationalRule | None:
    """Parse the model's JSON into a RelationalRule; None on unparseable output."""
    match = re.search(r"\{.*\}", str(text).strip(), flags=re.DOTALL)
    if not match:
        return None
    try:
        obj = json.loads(match.group(0))
    except json.JSONDecodeError:
        return None
    if not isinstance(obj, dict):
        return None
    d = {
        "rule_id": rule_id,
        "source_text": source_text,
        "premises": obj.get("premises", []) or [],
        "conclusions": obj.get("conclusions", []) or [],
    }
    try:
        rule = RelationalRule.from_dict(d)
    except (TypeError, ValueError, AttributeError):
        return None
    # an all-empty parse is treated as a failed extraction (predicted_empty)
    if not rule.premises and not rule.conclusions:
        return None
    return rule


class RelationalExtractor:
    """One LLM relational extractor, cached for reproducibility."""

    def __init__(self, client: Generator, model_id: str, cache_dir: Path | None = None) -> None:
        self._client = client
        self.model_id = model_id
        self._cache_dir = Path(cache_dir) if cache_dir else None
        if self._cache_dir:
            self._cache_dir.mkdir(parents=True, exist_ok=True)

    def _cache_path(self, source_text: str) -> Path | None:
        if not self._cache_dir:
            return None
        raw = f"{self.model_id}|{source_text}"
        key = hashlib.sha1(raw.encode("utf-8"), usedforsecurity=False).hexdigest()
        return self._cache_dir / f"{key}.json"

    def extract(self, source_text: str, rule_id: str) -> RelationalRule | None:
        path = self._cache_path(source_text)
        if path and path.exists():
            try:
                return RelationalRule.from_dict(json.loads(path.read_text(encoding="utf-8")))
            except (json.JSONDecodeError, KeyError):
                pass
        text, _meta = self._client.generate(build_prompt(source_text))
        rule = parse_rule(text, rule_id, source_text)
        if path and rule is not None:  # never cache a transient failure
            path.write_text(json.dumps(rule.to_dict(), ensure_ascii=False), encoding="utf-8")
        return rule


def known_vocabulary() -> set[str]:
    return set(GLOSSARY.keys())
