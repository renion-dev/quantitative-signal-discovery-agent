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
Genetic crossover helpers for the signal discovery workflow.

When ``genetic_crossover`` is enabled, the two best-evaluated signals (by
|mean IC|) are "bred" into ONE new candidate hypothesis. This module holds the
pure, testable pieces of that step:

  - ``select_parents``           pick the top-k evaluated signals
  - ``build_crossover_prompt``   prompt for the signal-generator LLM
  - ``parse_crossover_response`` extract the combined hypothesis from the reply

The LLM call itself stays in the workflow so this module has no async/LLM
dependencies and is fully unit-testable.
"""

import json
import logging
import math
import re

from .llm_utils import extract_json_array, sanitize_unicode
from .signal_generator import build_signal_template, format_operators_for_prompt, load_output_template

logger = logging.getLogger(__name__)

# Fields a crossover candidate must provide so the existing signal parser
# (``parse_signal_specs_with_errors``) can consume it unchanged — same
# required fields as ``signal_output_template.json::validation_rules``.
REQUIRED_SIGNAL_FIELDS = ("name", "formula", "meaning")

# Inline copy of the generator's signal template, used only when
# ``signal_output_template.json`` cannot be loaded.
_FALLBACK_SIGNAL_TEMPLATE = {
    "name": "<signal_name>",
    "formula": "<formula_using_operators>",
    "meaning": "<economic_intuition>",
    "category": "<momentum|volatility|volume|reversal|quality|other>",
    "data_fields_used": ["<list_of_data_fields>"],
    "operators_used": ["<list_of_operators>"],
    "lookback_periods": ["<list_of_lookback_days>"],
}


def _as_valid_ic(value) -> float | None:
    """Return the IC as a finite float, or ``None`` when missing/invalid."""
    try:
        ic = float(value)
    except (TypeError, ValueError):
        return None
    return ic if math.isfinite(ic) else None


def select_parents(signals: list[dict], metrics: list[dict], k: int = 2) -> list[dict]:
    """
    Pick the top-k signals by |mean IC| for genetic crossover.

    ``signals`` and ``metrics`` are parallel lists: ``metrics[i]`` is the
    evaluation result (e.g. an IC results dict) for ``signals[i]``. Entries
    whose metric is missing, not a dict, or has a missing/None/non-finite
    ``mean_ic`` are ignored. Ties on |mean_ic| are broken in favor of the
    earlier signal.

    Returns the selected signal dicts in descending |mean_ic| order, or ``[]``
    when fewer than two valid candidates exist (crossover needs two parents).
    """
    if k < 1:
        return []

    scored: list[tuple[int, float, dict]] = []
    for index, (signal, metric) in enumerate(zip(signals, metrics)):
        if not isinstance(signal, dict):
            continue
        metric_dict = metric if isinstance(metric, dict) else {}
        ic = _as_valid_ic(metric_dict.get("mean_ic"))
        if ic is None:
            continue
        scored.append((index, abs(ic), signal))

    if len(scored) < 2:
        return []

    scored.sort(key=lambda item: (-item[1], item[0]))
    return [signal for _, _, signal in scored[:k]]


def _signal_template_block() -> str:
    """Render the generator's own JSON template; fall back to an inline copy."""
    try:
        return build_signal_template(1, load_output_template())
    except Exception as exc:
        logger.warning(f"Could not load signal output template ({exc}); using inline fallback")
        item = json.dumps(_FALLBACK_SIGNAL_TEMPLATE, indent=2)
        return "```json\n[\n" + item + "\n]\n```"


def build_crossover_prompt(
    parent_a: dict,
    parent_b: dict,
    operator_catalog: list | str | None = None,
    feedback: str | None = None,
) -> str:
    """
    Build the prompt asking the signal-generator LLM to breed two parents.

    The prompt embeds both parent formulas verbatim, states that the output
    must be EXACTLY ONE combined hypothesis, and requests the SAME JSON schema
    the signal generator uses (rendered from ``signal_output_template.json``)
    so the existing signal parser can consume the reply unchanged.

    ``operator_catalog`` may be the operator list from
    ``load_calculator_operators()`` (formatted with
    ``format_operators_for_prompt``), a pre-formatted string (passed through
    unchanged), or None (a generic instruction is emitted).
    """
    if isinstance(operator_catalog, str):
        operators_block = operator_catalog.strip() or "(no operators supplied)"
    elif operator_catalog:
        operators_block = format_operators_for_prompt(operator_catalog)
    else:
        operators_block = "(operator catalogue unavailable — reuse only the operators appearing in the parents)"

    name_a = parent_a.get("name", "unnamed parent A")
    name_b = parent_b.get("name", "unnamed parent B")
    feedback_section = f"\nADDITIONAL CONTEXT:\n{feedback}\n" if feedback else ""
    signal_template = _signal_template_block()

    return f"""You are a senior quantitative researcher performing a genetic crossover of two stock-selection signals.

Combine the TWO parent signals below into EXACTLY ONE new, economically sensible hypothesis that
inherits the strengths of both parents. Do NOT simply copy either parent; breed them into a single
combined formula.

PARENT A ({name_a}):
formula: {parent_a.get('formula', '?')}
meaning: {parent_a.get('meaning', '?')}

PARENT B ({name_b}):
formula: {parent_b.get('formula', '?')}
meaning: {parent_b.get('meaning', '?')}
{feedback_section}
OPERATORS:
{operators_block}

STRICT RULES:
- Output EXACTLY ONE signal object (one combined hypothesis) — never a list of variants.
- Every operator call MUST match its signature exactly — same number of arguments, same order.
- Use only the exact operator names shown above (case-sensitive).
- The combined formula must be economically meaningful; explain the intuition in `meaning`.

Fill in this exact template at the END of your reply (inside a ```json block):

{signal_template}

Generate the single crossover signal now."""


def _is_valid_signal(candidate: dict) -> bool:
    """Whether the dict carries non-empty values for every required field."""
    return all(str(candidate.get(field, "") or "").strip() for field in REQUIRED_SIGNAL_FIELDS)


def parse_crossover_response(text: str) -> dict | None:
    """
    Extract the first valid crossover signal object from an LLM reply.

    Reuses the shared JSON-extraction helpers (``extract_json_array`` /
    ``sanitize_unicode``) so the reply may be a fenced JSON block, a bare JSON
    object or array, or JSON embedded in prose. A JSON array is scanned for
    its first valid entry.

    Returns the signal dict when it has non-empty ``name``/``formula``/
    ``meaning``, otherwise None. Never raises.
    """
    if not isinstance(text, str) or not text.strip():
        return None

    cleaned = sanitize_unicode(re.sub(r"<think\b[^>]*>.*?</think\s*>", "", text, flags=re.DOTALL))

    candidates: list = []
    try:
        candidates.append(json.loads(cleaned))
    except json.JSONDecodeError:
        candidates.extend(extract_json_array(cleaned))
        for fence in re.findall(r"```(?:json|JSON)?\s*([\[\{].*?[\]\}])\s*```", cleaned, re.DOTALL):
            try:
                candidates.append(json.loads(fence))
            except json.JSONDecodeError:
                continue

    for candidate in candidates:
        if isinstance(candidate, list):
            candidate = next(
                (item for item in candidate if isinstance(item, dict) and _is_valid_signal(item)),
                None,
            )
        if isinstance(candidate, dict) and _is_valid_signal(candidate):
            return candidate
    return None
