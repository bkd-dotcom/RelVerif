"""Compile relational rules (the typed AST) into Z3 over a bounded universe.

The hand-written slice (`relational_aka.py`) wrote its Z3 constraints directly; the
extractor (`relational_extractor.py`)
produces `RelationalRule` ASTs from prose. This module is the bridge: a **general compiler**
from a list of `RelationalRule` — *extracted or gold* — into Z3, so the discover ->
fix -> re-verify loop can run on rules that came out of the LLM rather than being
hand-written. This is what makes the relational pipeline end-to-end.

Each controlled-vocabulary predicate becomes one Z3 function over the bounded sorts.
A rule ``IF p1 AND p2 THEN c1 AND c2`` compiles to the Horn reading
``ForAll(vars, Implies(And(premises), And(conclusions)))`` — variables shared across
literals are the same Z3 constant, which also gives a free **well-typedness gate**:
if the same variable is used at two incompatible sorts, or a predicate name is not in
the vocabulary (a hallucinated relation), the rule is rejected as unencodable rather
than silently mis-compiled.

Scope (honest): bounded, small finite sorts. A conclusion-only ("free") variable is
universally quantified over the (small) sort — sound within the bound, and we keep
the relevant sorts singleton where that matters. Not an unbounded proof.
"""

from __future__ import annotations

from dataclasses import dataclass

from z3 import And, BoolSort, BoolVal, Const, EnumSort, ExprRef, Function, Implies, Not

from pipeline.schemas_relational import GLOSSARY, SORTS, RelationalRule, normalize_name


class RelationalCompileError(ValueError):
    """Raised when a rule cannot be well-typed against the vocabulary/universe."""


# default bounded universe sizes. Sorts that appear as a conclusion-only variable in
# the 5G-AKA rules (Key/Nonce/Message/HN/Session) are kept singleton so universal
# quantification over them is exact; UE/SN/Failure need >=2 to express confusion.
DEFAULT_SIZES: dict[str, int] = {
    "UE": 2, "SN": 2, "HN": 1, "Session": 1,
    "Key": 1, "Nonce": 1, "SQN": 1, "Message": 1, "Failure": 2,
}

_UNIVERSE_COUNTER = [0]


@dataclass
class CompileReport:
    compiled: int
    rejected: list[tuple[str, str]]  # (rule_id, reason)


class RelationalUniverse:
    """Bounded sorts + one Z3 function per vocabulary predicate (shared by all rules)."""

    def __init__(self, sizes: dict[str, int] | None = None) -> None:
        _UNIVERSE_COUNTER[0] += 1
        tag = _UNIVERSE_COUNTER[0]
        sizes = {**DEFAULT_SIZES, **(sizes or {})}
        # sort/function names are prefixed uniquely per instance ("ru{tag}_") so they
        # never collide in Z3's global context with the hand-written encoders (relational.py /
        # relational_aka.py), which declare their own sorts in the same process.
        pfx = f"ru{tag}_"
        self._sorts = {}
        self._consts = {}
        for name in SORTS:
            n = max(1, sizes.get(name, 1))
            sort, consts = EnumSort(f"{pfx}{name}", [f"{pfx}{name.lower()}{i}" for i in range(n)])
            self._sorts[name] = sort
            self._consts[name] = list(consts)
        self._funcs = {}
        for pname, (arg_sorts, _gloss) in GLOSSARY.items():
            domain = [self._sorts[s] for s in arg_sorts]
            self._funcs[pname] = Function(f"{pfx}{pname}", *domain, BoolSort())

    def sort(self, name: str):
        return self._sorts[name]

    def const(self, sort_name: str, i: int = 0):
        return self._consts[sort_name][i]

    def consts(self, sort_name: str) -> list:
        return self._consts[sort_name]

    def func(self, pred_name: str):
        return self._funcs[normalize_name(pred_name)]

    def known_predicate(self, pred_name: str) -> bool:
        return normalize_name(pred_name) in self._funcs


class RelationalRuleCompiler:
    """Compile RelationalRule ASTs into Z3 constraints over a shared universe."""

    def __init__(self, universe: RelationalUniverse) -> None:
        self.u = universe

    def _var_sorts(self, rule: RelationalRule) -> dict[str, str]:
        """Assign a sort to every variable; raise on OOV predicate or sort conflict."""
        var_sorts: dict[str, str] = {}
        for pred in (*rule.premises, *rule.conclusions):
            cname = normalize_name(pred.name)
            if cname not in GLOSSARY:
                msg = f"unknown predicate '{pred.name}'"
                raise RelationalCompileError(msg)
            arg_sorts = GLOSSARY[cname][0]
            if len(pred.args) != len(arg_sorts):
                msg = f"arity mismatch for '{cname}': {len(pred.args)} args vs {len(arg_sorts)} sorts"
                raise RelationalCompileError(msg)
            for var, srt in zip(pred.args, arg_sorts, strict=True):
                if var in var_sorts and var_sorts[var] != srt:
                    msg = f"variable '{var}' used at both {var_sorts[var]} and {srt}"
                    raise RelationalCompileError(msg)
                var_sorts[var] = srt
        return var_sorts

    def _literal(self, pred, consts: dict[str, ExprRef]) -> ExprRef:
        fn = self.u.func(pred.name)
        expr = fn(*[consts[v] for v in pred.args])
        return Not(expr) if pred.negated else expr

    @staticmethod
    def _conj(exprs: list[ExprRef]) -> ExprRef:
        if not exprs:
            return BoolVal(True)  # noqa: FBT003 — z3 BoolVal takes the value positionally
        if len(exprs) == 1:
            return exprs[0]
        return And(*exprs)

    def compile_rule(self, rule: RelationalRule) -> ExprRef:
        """Return the Z3 constraint for one rule (raises RelationalCompileError if unencodable)."""
        var_sorts = self._var_sorts(rule)
        consts = {v: Const(v, self.u.sort(s)) for v, s in var_sorts.items()}
        prem = self._conj([self._literal(p, consts) for p in rule.premises])
        conc = self._conj([self._literal(c, consts) for c in rule.conclusions])
        body = Implies(prem, conc) if rule.premises else conc
        if consts:
            from z3 import ForAll
            return ForAll(list(consts.values()), body)
        return body

    def compile_all(self, rules: list[RelationalRule]) -> tuple[list[ExprRef], CompileReport]:
        """Compile a rule set; skip (and report) the ones that fail the well-typedness gate."""
        compiled: list[ExprRef] = []
        rejected: list[tuple[str, str]] = []
        for r in rules:
            try:
                compiled.append(self.compile_rule(r))
            except RelationalCompileError as e:
                rejected.append((r.rule_id, str(e)))
        return compiled, CompileReport(compiled=len(compiled), rejected=rejected)
