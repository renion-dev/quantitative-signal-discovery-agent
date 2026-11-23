# SPDX-FileCopyrightText: Copyright (c) 2023-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.  # noqa
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Skeptic agent: pre-backtest review of generated signals.

The skeptic catches classic quant pitfalls BEFORE a signal is backtested, so
a lookahead-biased or leaky candidate never pollutes the optimization loop
with an inflated IC. Two layers:

  - ``detect_static_risks`` — deterministic, LLM-free scan of the signal
    description (and optional generated code) for known pitfalls.
  - ``build_skeptic_prompt`` / ``parse_skeptic_review`` — an optional LLM
    review returning a verdict: approve | revise | reject.

Every risk string returned by ``detect_static_risks`` is prefixed with a
category tag so callers can distinguish hard violations (``lookahead:``,
``leakage:``, ``degenerate:``, ``operator:`` — blocking) from advisory ones
(``lookback:``, ``loop:``, ``field:``, ``formula:``, ``suspicious:``,
``operator_case:``).

``operator_case:`` is a spelling warning: the formula calls an operator that
EXISTS in the calculator catalogue but with a different letter case (e.g.
``TS_ZScore`` vs the catalogue's ``TS_Zscore``). Such a candidate is valid and
must not be vetoed, so this category is deliberately advisory.
"""

import builtins
import json
import logging
import re
from pathlib import Path
from typing import NamedTuple

from .llm_utils import sanitize_unicode

logger = logging.getLogger(__name__)

TEMPLATE_DIR = Path(__file__).parent / "template"
SKEPTIC_TEMPLATE_JSON = TEMPLATE_DIR / "skeptic_output_template.json"

DEFAULT_REVIEW_TEMPLATE = {
    "verdict": "<approve|revise|reject>",
    "issues": ["<list_of_specific_issues>"],
    "rationale": "<short_explanation>",
}
DEFAULT_ALLOWED_VERDICTS = ["approve", "revise", "reject"]

VERDICT_MEANINGS = {
    "approve": "the signal is sound: no lookahead bias, leakage, or degenerate logic",
    "revise": "the signal is promising but has fixable issues; describe exactly what to change",
    "reject": "the signal is unsound (lookahead, leakage, or degenerate logic) and must not be backtested",
}

# Risk categories that justify an automatic veto.
BLOCKING_RISK_PREFIXES = ("lookahead:", "leakage:", "degenerate:", "operator:")
# Advisory categories: reported to the reviewer, never auto-veto.
# ``operator_case:`` = the operator exists in the catalogue but is spelled with a
# different letter case (a valid candidate, only a spelling warning).
NON_BLOCKING_RISK_PREFIXES = (
    "lookback:", "loop:", "field:", "formula:", "suspicious:", "operator_case:",
)

OHLCV_FIELDS = {"open", "close", "high", "low", "volume"}
# Aliases the code generator already normalizes to canonical OHLCV names.
OHLCV_FIELD_ALIASES = {
    "adj_close", "adjclose", "adjusted_close", "adjustedclose",
    "close_price", "closing_price", "price",
    "open_price", "opening_price",
    "high_price", "low_price",
    "trading_volume", "volume_traded",
}

_MAX_LOOKBACK = 252  # one trading year
_MIN_LOOKBACK = 2
_MAX_LOOP_BOUND = 10000

# Forward-looking constructs: negative shifts pull FUTURE bars into the present.
_LOOKAHEAD_SHIFT_RE = re.compile(r"\b(shift|pct_change)\s*\(\s*-", re.IGNORECASE)
_LEAD_RE = re.compile(r"\blead\s*\(", re.IGNORECASE)
# Identifiers that name future data (future_close, next_day_return, ...).
_FUTURE_IDENTIFIER_RE = re.compile(r"\b\w*(future|next_)\w*\b", re.IGNORECASE)
# Fields that only exist after the fact (a computed forward return, a label).
_LEAK_FIELD_RE = re.compile(
    r"\b(forward_return\w*|future_return\w*|next_close\w*|target\w*|label\w*)\b", re.IGNORECASE
)

# A formula without any time-series/lag operator shares its bar with the
# forward-return computation, so the IC can be mechanically inflated.
_TEMPORAL_CALL_RE = re.compile(
    r"\b(TS_\w+|EMA\w*|Decay\w*|shift|rolling|pct_change|diff|cummax|cumsum|expanding)\s*\(",
    re.IGNORECASE,
)
_PRICE_FIELD_RE = re.compile(r"\b(Open|High|Low|Close|Volume)\b")

# Numeric second argument of TS_*/EMA*/Decay* calls = a lookback literal.
_LOOKBACK_ARG_RE = re.compile(r"\b(?:TS_\w+|EMA\w*|Decay\w*)\s*\(\s*[^,()]+,\s*(\d+)\s*\)")

_HYPE_RE = re.compile(
    r"\b(guaranteed|risk[-\s]?free|100%\s*(?:accuracy|win(?:ning)?|correct)|always\s+wins?|"
    r"cannot\s+lose|no\s+risk)\b",
    re.IGNORECASE,
)

# --- unknown-operator check -------------------------------------------------
# Call names in the ``formula`` field are checked against the operator
# catalogue in ``template/calculator.json``. A formula calling an operator that
# does not exist cannot execute (NameError at backtest time), so the finding is
# BLOCKING. False-positive control: a name is only reported when it is NOT a
# Python builtin/stdlib helper, NOT a module reference, NOT already a valid
# catalogue operator, AND looks like an operator (family name or family prefix).
OPERATOR_CATALOGUE_PATH = TEMPLATE_DIR / "calculator.json"

# Explicit operator naming conventions; extended with prefixes derived from the
# catalogue's own names (e.g. ``TS_Corr`` -> ``TS_``). Matched case-insensitively.
_OPERATOR_FAMILY_PREFIXES = ("ts_", "tm_", "ema_", "sma_", "wma_", "rank_", "decay_", "roc_")

# Bare operator-style names (matched case-insensitively): the catalogue's own
# names, their stems without a ``TS_``/``CS_``-style prefix, and these families.
_OPERATOR_BARE_FAMILIES = frozenset({
    "Rank", "Rank_Mul", "Rank_Add", "Rank_Sub", "Rank_Div", "Div", "Add", "Sub", "Mul",
    "Decay_Linear", "Power", "Sign", "Abs", "Max", "Min", "Std", "Var", "Corr", "Cov",
    "Skew", "Kurt", "Median", "ZScore", "Log", "Exp", "Sqrt", "If", "Greater", "Less",
    "EMA", "SMA", "WMA", "ROC", "Sharpe", "Drawdown", "Beta", "Residual",
})

# Python builtins / stdlib helpers — never reported as unknown operators.
_PYTHON_HELPER_NAMES = frozenset(
    name for name in dir(builtins) if not name.startswith("_")
) | frozenset({
    "abs", "min", "max", "sum", "len", "range", "round", "int", "float", "str", "bool",
    "list", "dict", "set", "tuple", "zip", "enumerate", "sorted", "any", "all", "pow", "print",
})

# Module references / import aliases — never reported as unknown operators.
_MODULE_REFERENCE_NAMES = frozenset({
    "pd", "np", "math", "statistics", "pandas", "numpy", "scipy", "sklearn", "os", "sys",
    "json", "re", "plt", "sns", "datetime", "collections", "itertools", "functools",
    "random", "warnings", "logging",
})

# ``name(`` — a call. Dotted attribute calls (``Close.shift``, ``math.log``)
# are excluded at use time: methods and module functions are not operators.
_CALL_NAME_RE = re.compile(r"\b([A-Za-z_][A-Za-z0-9_]*)\s*\(")


class _OperatorCatalogue(NamedTuple):
    """Operator catalogue plus the operator-like-name sets derived from it."""

    valid: frozenset[str]  # catalogue operator names (exact case)
    like_names: frozenset[str]  # operator-like names (lowercased)
    like_prefixes: tuple[str, ...]  # operator-like name prefixes (lowercased)


# Cached per catalogue path; failures are cached as ``None`` so an unreadable
# catalogue neither raises nor spams the log on every candidate.
_OPERATOR_CATALOGUE_CACHE: dict[str, "_OperatorCatalogue | None"] = {}


def _load_operator_catalogue() -> "_OperatorCatalogue | None":
    """
    Load the operator catalogue from ``template/calculator.json`` lazily.

    Deliberately does NOT import ``signal_evaluator`` (heavy and a potential
    import cycle): the catalogue is read straight from disk instead. Returns
    ``None`` when the file is missing or unreadable — callers must degrade
    silently in that case (no risk, no exception).
    """
    key = str(OPERATOR_CATALOGUE_PATH)
    if key in _OPERATOR_CATALOGUE_CACHE:
        return _OPERATOR_CATALOGUE_CACHE[key]
    catalogue: _OperatorCatalogue | None
    try:
        with open(OPERATOR_CATALOGUE_PATH, "r") as f:
            data = json.load(f)
        if not isinstance(data, list):
            raise ValueError("operator catalogue must be a JSON list")
        valid = frozenset(
            str(entry["name"]).strip()
            for entry in data
            if isinstance(entry, dict) and str(entry.get("name") or "").strip()
        )
        if not valid:
            raise ValueError("operator catalogue is empty")
        like_names = {name.lower() for name in valid} | {
            name.lower() for name in _OPERATOR_BARE_FAMILIES
        }
        prefixes = set(_OPERATOR_FAMILY_PREFIXES)
        for name in valid:
            if "_" in name:
                head, _, tail = name.partition("_")
                prefixes.add(head.lower() + "_")
                like_names.add(tail.lower())
        catalogue = _OperatorCatalogue(
            valid,
            frozenset(like_names),
            tuple(sorted(prefixes)),
        )
    except (OSError, ValueError, TypeError, KeyError) as exc:
        logger.warning(f"Operator catalogue unavailable ({exc}); operator-name check disabled")
        catalogue = None
    _OPERATOR_CATALOGUE_CACHE[key] = catalogue
    return catalogue


def load_skeptic_template() -> dict:
    """Load the skeptic review schema from ``template/skeptic_output_template.json``."""
    if not SKEPTIC_TEMPLATE_JSON.exists():
        logger.warning(f"Skeptic template not found at {SKEPTIC_TEMPLATE_JSON}")
        return {}
    try:
        with open(SKEPTIC_TEMPLATE_JSON, "r") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError) as exc:
        logger.warning(f"Failed to load skeptic template: {exc}")
        return {}


def detect_static_risks(signal_description: dict, code: str | None = None) -> list[str]:
    """
    Deterministically scan a signal description (and optional code) for pitfalls.

    Checks (each risk string is prefixed with its category tag):
      - ``lookahead:``  forward-looking ``shift(-n)`` / ``pct_change(-n)`` /
        ``lead(...)``, identifiers containing ``future``/``next_``, and
        same-bar lookahead (price fields with no time-series operator).
      - ``leakage:``    future-only fields (forward returns, targets, labels).
      - ``degenerate:`` constant formulas (no operators or data fields).
      - ``operator:``  formula calls an operator name that does not exist in
        the calculator catalogue — the signal cannot execute (blocking).
      - ``operator_case:`` formula calls an operator that DOES exist in the
        catalogue but with a different letter case (``TS_ZScore`` vs
        ``TS_Zscore``) — a spelling warning only, never a veto.
      - ``lookback:``   absurd lookbacks (> 252 or < 2), from declared
        ``lookback_periods`` and numeric second args of TS_*/EMA*/Decay* calls.
      - ``loop:``       ``while True`` or ``range(...)`` bounds > 10000 in code.
      - ``field:``      declared data fields outside the OHLCV set (+ aliases).
      - ``suspicious:`` hype claims in ``meaning``.

    Returns a (possibly empty) list of human-readable risk strings. Never raises.
    """
    risks: list[str] = []
    if not isinstance(signal_description, dict):
        return risks

    formula = str(signal_description.get("formula") or "").strip()
    meaning = str(signal_description.get("meaning") or "").strip()
    fields = [f for f in (signal_description.get("data_fields_used") or []) if isinstance(f, str)]
    lookbacks = signal_description.get("lookback_periods") or []
    code = code if isinstance(code, str) else ""

    # Identifier-like contexts only: formula, code, and declared field names.
    # Prose (meaning) is excluded so English words like "future" don't false-positive.
    scanned = "\n".join(part for part in (formula, code, *fields) if part)

    # --- lookahead: forward-looking shifts and future-only identifiers -------
    if _LOOKAHEAD_SHIFT_RE.search(scanned):
        risks.append("lookahead: forward-looking shift(-n)/pct_change(-n) pulls future values into the present")
    if _LEAD_RE.search(scanned):
        risks.append("lookahead: lead(...) references future bars")
    future_hits = sorted({m.group(0) for m in _FUTURE_IDENTIFIER_RE.finditer(scanned)})
    if future_hits:
        risks.append(f"lookahead: identifier(s) referencing future data: {', '.join(future_hits[:5])}")

    # --- lookahead: same-bar lookahead ---------------------------------------
    if formula and _PRICE_FIELD_RE.search(formula) and not _TEMPORAL_CALL_RE.search(formula):
        risks.append(
            "lookahead: formula uses same-bar prices with no time-series/lag operator — "
            "forward returns derive from the same bar's close, so the IC can be mechanically inflated"
        )

    # --- degenerate: constant / content-free formulas -------------------------
    if formula:
        if re.fullmatch(r"[\s\d.eE+\-*/(),]+", formula):
            risks.append(
                "degenerate: formula contains no data fields or operators — constant signal "
                "with no cross-sectional variance"
            )
        elif not _PRICE_FIELD_RE.search(formula) and not re.search(r"\b[A-Za-z_]\w*\s*\(", formula):
            risks.append("degenerate: formula references no data field and no operator call")

    # --- operator: hallucinated operator names in the formula -----------------
    catalogue = _load_operator_catalogue() if formula else None
    if catalogue is not None:
        valid, like_names, like_prefixes = catalogue
        # lowercased catalogue name -> canonical spelling. Built from the sorted
        # names so the canonical pick is deterministic even if two catalogue
        # entries ever collide when lowercased.
        canonical_by_lower = {name.lower(): name for name in sorted(valid)}
        called_names = sorted({
            match.group(1)
            for match in _CALL_NAME_RE.finditer(formula)
            # Dotted calls (Close.shift, math.log) are attributes, not operators.
            if match.start() == 0 or formula[match.start() - 1] != "."
        })
        for name in called_names:
            if name in valid or name in _PYTHON_HELPER_NAMES or name in _MODULE_REFERENCE_NAMES:
                continue
            lowered = name.lower()
            if lowered not in like_names and not lowered.startswith(like_prefixes):
                continue
            canonical = canonical_by_lower.get(lowered)
            if canonical is not None:
                # The operator EXISTS in the catalogue — only the letter case
                # differs (e.g. ``TS_ZScore`` vs the catalogue's ``TS_Zscore``).
                # That is a spelling warning, not a hallucination: the candidate
                # is a VALID hypothesis and must never be vetoed for casing.
                # Safety: a case-mismatched name resolves either to the
                # candidate's own self-contained helper (correct result) or to a
                # missing name -> explicit NameError at backtest time. It can
                # never silently yield a wrong metric.
                risks.append(
                    f"operator_case: operator name {name!r} differs in case from the "
                    f"catalogue entry {canonical!r}"
                )
                continue
            risks.append(
                f"operator: formula calls unknown operator {name!r} "
                f"(not in the {len(valid)}-operator catalogue)"
            )

    # --- lookback: absurd window sizes ----------------------------------------
    seen_lookbacks: set[int] = set()
    for value in lookbacks:
        try:
            seen_lookbacks.add(int(value))
        except (TypeError, ValueError):
            continue
    for match in _LOOKBACK_ARG_RE.finditer(scanned):
        seen_lookbacks.add(int(match.group(1)))
    for n in sorted(seen_lookbacks):
        if n > _MAX_LOOKBACK:
            risks.append(f"lookback: lookback of {n} periods exceeds {_MAX_LOOKBACK} trading days")
        elif n < _MIN_LOOKBACK:
            risks.append(f"lookback: lookback of {n} period(s) is below the minimum of {_MIN_LOOKBACK}")

    # --- loop: unbounded / huge loops in generated code ------------------------
    if code:
        if re.search(r"\bwhile\s+True\b", code):
            risks.append("loop: 'while True' has no static bound — potential infinite loop")
        for match in re.finditer(r"\brange\s*\(\s*(\d+)\s*(?:,\s*(\d+)\s*)?\)", code):
            stop = int(match.group(2)) if match.group(2) else int(match.group(1))
            if stop > _MAX_LOOP_BOUND:
                risks.append(f"loop: range(...) iterates up to {stop} times — unbounded/huge loop")

    # --- field / leakage: data-field sanity ------------------------------------
    allowed_fields = OHLCV_FIELDS | OHLCV_FIELD_ALIASES
    for field in fields:
        if field.strip().lower() not in allowed_fields:
            risks.append(f"field: declares data field {field!r} outside the OHLCV set")
    leak_hits = sorted({m.group(0) for m in _LEAK_FIELD_RE.finditer(formula)} | {
        m.group(0) for m in _LEAK_FIELD_RE.finditer("\n".join(fields))
    })
    if leak_hits:
        risks.append(f"leakage: references future-only field(s): {', '.join(leak_hits[:5])}")

    # --- suspicious: hype in the rationale -------------------------------------
    hype = _HYPE_RE.search(meaning)
    if hype:
        risks.append(
            f"suspicious: rationale claim {hype.group(0)!r} looks too good to be true — verify the economics"
        )

    return risks


def has_blocking_risk(risks: list[str]) -> bool:
    """Whether any risk string carries a blocking category prefix."""
    for risk in risks or []:
        if isinstance(risk, str) and risk.startswith(BLOCKING_RISK_PREFIXES):
            return True
    return False


def build_skeptic_prompt(
    signal_description: dict,
    code: str | None = None,
    static_risks: list[str] | None = None,
) -> str:
    """
    Build the prompt for the skeptic LLM review of one candidate signal.

    Asks the model to act as a skeptical senior quant and return a verdict
    (approve | revise | reject) with ``issues`` and ``rationale``, as a JSON
    block matching the schema in ``template/skeptic_output_template.json``.
    """
    template = load_skeptic_template()
    review_template = template.get("review_template") or DEFAULT_REVIEW_TEMPLATE
    allowed = template.get("allowed_verdicts") or DEFAULT_ALLOWED_VERDICTS

    desc_block = json.dumps(signal_description, indent=2, default=str)
    code_section = (
        f"\nGENERATED CODE (self-contained module; the signal functions are at the end):\n"
        f"```python\n{code}\n```\n"
        if code
        else "\nGENERATED CODE: (not yet generated)\n"
    )
    risks = [r for r in (static_risks or []) if isinstance(r, str) and r.strip()]
    risks_section = (
        "STATIC RISK SCAN (deterministic pre-checks; treat as hints to verify, not final truth):\n"
        + "\n".join(f"- {r}" for r in risks)
        if risks
        else "STATIC RISK SCAN (deterministic pre-checks): no classic pitfalls detected by static analysis."
    )
    verdict_lines = "\n".join(f"- {v}: {VERDICT_MEANINGS.get(str(v), '')}" for v in allowed)
    allowed_block = ", ".join(str(v) for v in allowed)

    return f"""You are a skeptical senior quantitative researcher reviewing a candidate stock-selection signal
BEFORE it is backtested. Your job is to catch flaws that would make any backtest result meaningless:
lookahead bias (using data not available at decision time), data leakage (future information entering
the signal), degenerate logic (constants, trivially mechanical correlations), and implausible claims.

CANDIDATE SIGNAL DESCRIPTION:
{desc_block}
{code_section}
{risks_section}

NOTE: a finding prefixed ``operator_case:`` is only a spelling warning (that operator exists in
the catalogue with a different letter case) — it is NOT a hard violation; never reject a candidate
solely because of it.

REVIEW CHECKLIST:
- Does the formula or code use future bars (negative shifts, leads, forward returns, next-period fields)?
- Are the signal values and the forward returns derived from the same bar without any lag?
- Are lookbacks economically plausible (not a single bar, not more than one trading year)?
- Is the signal a constant or near-constant, or justified by an implausible 'too good to be true' rationale?
- Would the signal have been knowable at the time a position would have been taken?

VERDICT RULES — return exactly one of:
{verdict_lines}

OUTPUT FORMAT — at the END of your reply, inside a ```json block, return exactly this schema:

```json
{json.dumps(review_template, indent=2)}
```

`verdict` MUST be one of: {allowed_block}.
`issues` must list concrete, specific problems (empty list when approving).
`rationale` must be a short explanation grounded in the checklist above."""


_VERDICT_ALIASES = {
    "approve": "approve", "approved": "approve", "accept": "approve", "pass": "approve", "ok": "approve",
    "reject": "reject", "rejected": "reject", "veto": "reject", "decline": "reject",
    "revise": "revise", "revision": "revise", "needs_revision": "revise", "revise_required": "revise",
}


def _normalize_verdict(value) -> str | None:
    """Map a raw verdict (and common aliases) to approve|revise|reject, or None."""
    if not isinstance(value, str):
        return None
    key = value.strip().lower().replace("-", "_").replace(" ", "_")
    return _VERDICT_ALIASES.get(key)


def _coerce_issues(value) -> list[str]:
    """Coerce an issues field of any shape into a list of non-empty strings."""
    if value is None:
        return []
    if isinstance(value, str):
        return [value.strip()] if value.strip() else []
    if isinstance(value, (list, tuple, set)):
        return [str(item).strip() for item in value if str(item).strip()]
    text = str(value).strip()
    return [text] if text else []


def _bullet_lines(text: str) -> list[str]:
    """Extract bullet lines (``-``, ``*``, ``•``) as heuristic issues."""
    return [
        line.lstrip("-*• ").strip()
        for line in text.splitlines()
        if line.strip().startswith(("-", "*", "•"))
    ]


def _iter_json_candidates(text: str):
    """Yield candidate JSON substrings: fenced blocks first, then balanced {...}."""
    for fence in re.findall(r"```(?:json|JSON)?\s*([\[\{].*?[\]\}])\s*```", text, re.DOTALL):
        yield fence

    start = None
    depth = 0
    in_string = False
    escape = False
    for i, ch in enumerate(text):
        if in_string:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
        elif ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}":
            if depth > 0:
                depth -= 1
                if depth == 0 and start is not None:
                    yield text[start : i + 1]
                    start = None


def parse_skeptic_review(text: str) -> dict:
    """
    Parse a skeptic review into ``{"verdict", "issues", "rationale"}``.

    JSON is preferred (whole-text, fenced, or embedded); when nothing parseable
    is found, simple heuristics apply: text containing "reject"/"veto" maps to
    ``reject``, then "revise" maps to ``revise``; anything else defaults to
    ``approve``. The verdict is always normalized to approve|revise|reject.

    Never raises and never returns None: empty/None/garbage input yields
    ``{"verdict": "approve", "issues": [], "rationale": ""}``.
    """
    if not isinstance(text, str) or not text.strip():
        return {"verdict": "approve", "issues": [], "rationale": ""}

    cleaned = sanitize_unicode(re.sub(r"<think\b[^>]*>.*?</think\s*>", "", text, flags=re.DOTALL)).strip()

    parsed_obj = None
    for candidate_text in [cleaned, *_iter_json_candidates(cleaned)]:
        try:
            parsed = json.loads(candidate_text)
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            parsed_obj = parsed
            break
        if isinstance(parsed, list):
            parsed_obj = next((item for item in parsed if isinstance(item, dict)), None)
            if parsed_obj is not None:
                break

    if isinstance(parsed_obj, dict):
        verdict = _normalize_verdict(
            parsed_obj.get("verdict") or parsed_obj.get("decision") or parsed_obj.get("review")
        )
        issues = _coerce_issues(parsed_obj.get("issues"))
        rationale = str(parsed_obj.get("rationale") or parsed_obj.get("reason") or "").strip()
        if verdict or issues or rationale:
            return {"verdict": verdict or "approve", "issues": issues, "rationale": rationale[:500]}

    lowered = cleaned.lower()
    if "reject" in lowered or "veto" in lowered:
        return {"verdict": "reject", "issues": _bullet_lines(cleaned), "rationale": cleaned[:500]}
    if "revise" in lowered:
        return {"verdict": "revise", "issues": _bullet_lines(cleaned), "rationale": cleaned[:500]}
    return {"verdict": "approve", "issues": [], "rationale": ""}


def extract_function_source(module_code: str, function_name: str) -> str | None:
    """
    Extract one top-level ``def <name>(...)`` block from a generated module.

    The block spans from the ``def`` line to just before the next top-level
    ``def`` (indented / nested functions do not terminate it). Returns None
    when the module or function is missing.
    """
    if not isinstance(module_code, str) or not isinstance(function_name, str) or not function_name.strip():
        return None
    pattern = rf"(?ms)^def\s+{re.escape(function_name.strip())}\s*\(.*?(?=^def\s+\w+\s*\(|\Z)"
    match = re.search(pattern, module_code)
    return match.group(0) if match else None
