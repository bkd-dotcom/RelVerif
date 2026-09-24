"""Typed relational AST + controlled vocabulary for 5G-AKA extraction.

The propositional pipeline emits boolean IF-THEN rules over opaque atoms; the relational representation's harder
task is to emit **typed predicates with roles** — ``key_bound_to(session, sn)`` rather
than a flat symbol ``KeyBoundToSN``. This module defines:

  * the sort system (``SORTS``) — the entity types a relational rule may quantify over;
  * a controlled predicate vocabulary (``GLOSSARY``) — the allowed relations, each with
    an ordered list of argument sorts and a one-line gloss. This doubles as the
    retrieval/role hint the extractor is given (the mitigation for the harder
    extraction task) and as the normalization target for scoring;
  * ``Predicate`` / ``RelationalRule`` dataclasses with a JSON round-trip and a
    canonical signature used for gold-vs-extraction matching.

Design choices that make scoring meaningful:
  * A predicate's *signature* is ``(normalized_name, sorts, negated)`` — argument
    *variable names* are free (alpha-renaming), but the relation and its role types
    are what must be recovered. This is what predicate-level P/R/F1 scores.
  * Premise vs conclusion position is preserved so directionality (premise/conclusion
    swap) is a measurable, separate error mode — same discipline as the propositional
    directionality-catch metric, now at the relational level.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field

# --- sorts: the entity types (Party / Session / Message / Key / Nonce) --------------
SORTS: tuple[str, ...] = (
    "UE",        # user equipment / subscriber
    "SN",        # serving network (SEAF/AMF)
    "HN",        # home network (AUSF/UDM/ARPF)
    "Session",   # one authentication run
    "Key",       # a derived key (KAUSF, KSEAF, KAMF, ...)
    "Nonce",     # RAND / freshness material
    "SQN",       # sequence number
    "Message",   # a protocol message / identifier (SUCI, RES*, MAC, ...)
    "Failure",   # a failure-message type (MAC_FAILURE / SYNC_FAILURE)
)

# --- controlled predicate vocabulary: name -> (arg sorts, gloss) --------------------
# This is the extractor's role-hint glossary AND the scorer's normalization target.
GLOSSARY: dict[str, tuple[tuple[str, ...], str]] = {
    "conceals_supi":       (("Session", "UE"), "the subscriber's SUPI is concealed (SUCI used) in the session"),
    "authenticated_with":  (("Session", "UE", "SN"), "the UE ran authentication with serving network sn"),
    "authorized_by_hn":    (("SN", "HN"), "the serving network is authorized by the home network"),
    "derives_key":         (("Session", "SN", "Key"), "sn derives the anchor key in the session"),
    "key_bound_to":        (("Session", "Key", "SN"), "the derived key is cryptographically bound to sn"),
    "uses_snn_in_kdf":     (("Session", "Key"), "the serving network name is an input to the key derivation"),
    "mac_verified":        (("Session", "UE"), "the UE verified the network-authentication MAC"),
    "sqn_in_range":        (("Session", "UE"), "the received sequence number is in the acceptable range"),
    "computes_res":        (("Session", "UE", "Message"), "the UE computes the authentication response RES*"),
    "res_matches_xres":    (("Session", "UE"), "the UE's RES* matches the expected XRES* at the network"),
    "accepts":             (("Session", "UE"), "the UE accepts / completes the authentication"),
    "sends_failure":       (("Session", "UE", "Failure"), "the UE returns a failure message of the given type"),
    "failure_indistinct":  (("Session",), "failure messages are indistinguishable to an observer"),
    "freshness_checked":   (("Session", "Nonce"), "the freshness of the challenge is checked"),
    "identifies":          (("Message", "UE"), "the message reveals/identifies the subscriber"),
    "encrypted_with_hn_key": (("Message", "HN"), "the message is encrypted with the home network public key"),
    "derives_kausf":       (("Session", "HN", "Key"), "the home network derives the KAUSF for the session"),
    "forwards_res":        (("Session", "SN", "Message"), "the serving network forwards the response RES* to the home network"),
    "nas_security_active": (("Session", "UE"), "NAS security has been activated for the UE in the session"),
}

# tolerant alias map: extractor variants -> canonical vocabulary name. Kept small and
# conservative; extending it is a deliberate, reviewable act (it inflates recall).
NAME_ALIASES: dict[str, str] = {
    "conceal_supi": "conceals_supi",
    "supi_concealed": "conceals_supi",
    "suci_used": "conceals_supi",
    "authenticates_with": "authenticated_with",
    "authed_with": "authenticated_with",
    "ran_aka_with": "authenticated_with",
    "authorized": "authorized_by_hn",
    "sn_authorized": "authorized_by_hn",
    "derive_key": "derives_key",
    "derives_anchor_key": "derives_key",
    "bound_to": "key_bound_to",
    "key_bound": "key_bound_to",
    "kseaf_bound_to": "key_bound_to",
    "snn_in_kdf": "uses_snn_in_kdf",
    "uses_snn": "uses_snn_in_kdf",
    "mac_ok": "mac_verified",
    "verifies_mac": "mac_verified",
    "sqn_fresh": "sqn_in_range",
    "sqn_acceptable": "sqn_in_range",
    "compute_res": "computes_res",
    "computes_response": "computes_res",
    "res_equals_xres": "res_matches_xres",
    "res_matches": "res_matches_xres",
    "accept": "accepts",
    "completes": "accepts",
    "send_failure": "sends_failure",
    "returns_failure": "sends_failure",
    "unified_failure": "failure_indistinct",
    "failure_unlinkable": "failure_indistinct",
    "check_freshness": "freshness_checked",
    "identify": "identifies",
    "reveals": "identifies",
    "encrypted_with_hn_pubkey": "encrypted_with_hn_key",
}


def normalize_name(name: str) -> str:
    """Lowercase + strip + alias-map a predicate name to the controlled vocabulary."""
    key = str(name).strip().lower().replace(" ", "_").replace("-", "_")
    return NAME_ALIASES.get(key, key)


@dataclass(frozen=True)
class Predicate:
    name: str
    args: tuple[str, ...] = ()           # variable names (free / alpha-renamable)
    sorts: tuple[str, ...] = ()          # sorts aligned to args
    negated: bool = False

    @property
    def signature(self) -> tuple[str, tuple[str, ...], bool]:
        """Match key: normalized name + arg sorts + negation. Variable names excluded."""
        return (normalize_name(self.name), tuple(self.sorts), self.negated)

    def to_dict(self) -> dict:
        return {"name": self.name, "args": list(self.args), "sorts": list(self.sorts), "negated": self.negated}

    @classmethod
    def from_dict(cls, d: dict) -> Predicate:
        return cls(
            name=str(d.get("name", "")),
            args=tuple(d.get("args", []) or []),
            sorts=tuple(d.get("sorts", []) or []),
            negated=bool(d.get("negated", False)),
        )

    def __str__(self) -> str:
        neg = "!" if self.negated else ""
        inner = ", ".join(f"{a}:{s}" for a, s in zip(self.args, self.sorts, strict=False))
        return f"{neg}{normalize_name(self.name)}({inner})"


@dataclass
class RelationalRule:
    rule_id: str
    source_text: str
    premises: tuple[Predicate, ...] = ()
    conclusions: tuple[Predicate, ...] = ()
    notes: str = ""
    meta: dict = field(default_factory=dict)

    # --- signatures used by the scorer ---
    def premise_sigs(self) -> set:
        return {p.signature for p in self.premises}

    def conclusion_sigs(self) -> set:
        return {p.signature for p in self.conclusions}

    def all_sigs(self) -> set:
        return self.premise_sigs() | self.conclusion_sigs()

    def to_dict(self) -> dict:
        return {
            "rule_id": self.rule_id,
            "source_text": self.source_text,
            "premises": [p.to_dict() for p in self.premises],
            "conclusions": [p.to_dict() for p in self.conclusions],
            "notes": self.notes,
            "meta": self.meta,
        }

    @classmethod
    def from_dict(cls, d: dict) -> RelationalRule:
        return cls(
            rule_id=str(d.get("rule_id", "")),
            source_text=str(d.get("source_text", "")),
            premises=tuple(Predicate.from_dict(p) for p in d.get("premises", []) or []),
            conclusions=tuple(Predicate.from_dict(p) for p in d.get("conclusions", []) or []),
            notes=str(d.get("notes", "")),
            meta=dict(d.get("meta", {}) or {}),
        )

    def __str__(self) -> str:
        prem = " AND ".join(str(p) for p in self.premises) or "TRUE"
        conc = " AND ".join(str(p) for p in self.conclusions) or "TRUE"
        return f"IF {prem} THEN {conc}"


def load_gold(path: str) -> list[RelationalRule]:
    """Load a JSONL gold file (one RelationalRule dict per line)."""
    rules: list[RelationalRule] = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rules.append(RelationalRule.from_dict(json.loads(line)))
    return rules


def dump_gold(rules: list[RelationalRule], path: str) -> None:
    with open(path, "w", encoding="utf-8") as fh:
        fh.writelines(json.dumps(r.to_dict(), ensure_ascii=False) + "\n" for r in rules)


def glossary_prompt_block() -> str:
    """Render the controlled vocabulary as a role-hint block for the extractor prompt."""
    lines = ["SORTS: " + ", ".join(SORTS), "", "PREDICATES (name(argSorts) — meaning):"]
    for name, (sorts, gloss) in GLOSSARY.items():
        lines.append(f"  {name}({', '.join(sorts)}) — {gloss}")
    return "\n".join(lines)
