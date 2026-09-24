"""The relational extractor prompt + parse + cache (offline, fake client)."""

from __future__ import annotations

from pipeline.relational_extractor import RelationalExtractor, build_prompt, parse_rule
from pipeline.schemas_relational import GLOSSARY


class _FakeClient:
    def __init__(self, response: str):
        self.response = response
        self.calls = 0

    def generate(self, prompt: str):
        self.calls += 1
        return self.response, {"provider": "fake"}


def test_prompt_carries_vocabulary_and_source_only():
    prompt = build_prompt("A UE shall accept only if it verified the MAC.")
    # role hints present
    for name in list(GLOSSARY)[:3]:
        assert name in prompt
    # the source sentence is present; no gold rule leaks in (blinded extraction)
    assert "A UE shall accept" in prompt
    assert "IF " not in prompt.split("[SENTENCE]")[1]  # no pre-formed rule after the sentence marker


def test_parse_valid_json():
    text = ('{"premises": [{"name": "accepts", "args": ["s","u"], "sorts": ["Session","UE"], '
            '"negated": false}], "conclusions": [{"name": "mac_verified", "args": ["s","u"], '
            '"sorts": ["Session","UE"], "negated": false}]}')
    rule = parse_rule(text, "R1", "src")
    assert rule is not None
    assert len(rule.premises) == 1
    assert len(rule.conclusions) == 1
    assert rule.premises[0].signature[0] == "accepts"


def test_parse_failure_returns_none():
    assert parse_rule("sorry, I cannot help", "R1", "src") is None
    assert parse_rule("", "R1", "src") is None


def test_empty_parse_is_treated_as_failed_extraction():
    assert parse_rule('{"premises": [], "conclusions": []}', "R1", "src") is None


def test_cache_round_trip(tmp_path):
    response = ('{"premises": [], "conclusions": [{"name": "conceals_supi", "args": ["s","u"], '
                '"sorts": ["Session","UE"], "negated": false}]}')
    client = _FakeClient(response)
    ex = RelationalExtractor(client, "fake-model", cache_dir=tmp_path)
    r1 = ex.extract("The UE conceals its SUPI.", "R1")
    r2 = ex.extract("The UE conceals its SUPI.", "R1")  # served from cache
    assert r1 is not None
    assert r2 is not None
    assert client.calls == 1  # second call hit the cache
    assert r2.conclusions[0].signature[0] == "conceals_supi"
    # a cache file was written
    assert any(p.suffix == ".json" for p in tmp_path.iterdir())
