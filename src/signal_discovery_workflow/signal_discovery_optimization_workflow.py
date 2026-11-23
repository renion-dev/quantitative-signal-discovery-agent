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
Signal Discovery Optimization Workflow.

Orchestrates the loop:
    signal_generator -> signal_code_generator -> signal_evaluator
                ^                                           |
                |__________ optimization advisor ___________|

The agents themselves live in their own modules; this file only handles
iteration, scoring, feedback, and persisting accepted/best results.
"""

import asyncio
import json
import logging
import math
import uuid
import warnings
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
from langchain_core.messages import HumanMessage, SystemMessage
from nat.builder.builder import Builder
from nat.builder.framework_enum import LLMFrameworkEnum
from nat.builder.function_info import FunctionInfo
from nat.cli.register_workflow import register_function
from nat.data_models.function import FunctionBaseConfig
from pydantic import Field

from .llm_utils import NO_THINK_INSTRUCTION, extract_json_array, extract_response_text
from .signal_code_generator import _python_function_name, generate_signal_code
from .signal_evaluator import (
    compute_directional_ic,
    compute_forward_returns,
    compute_rank_ic,
    execute_signal_code,
    extract_code_from_response,
    load_stock_data,
)
from .signal_generator import generate_signal_json, load_calculator_operators
from .signal_genetic import build_crossover_prompt, parse_crossover_response, select_parents
from .signal_skeptic import (
    build_skeptic_prompt,
    detect_static_risks,
    extract_function_source,
    has_blocking_risk,
    parse_skeptic_review,
)

logger = logging.getLogger(__name__)

OUTPUT_DIR = Path(__file__).parent / "output"


_STATUS_HEADLINE = {
    "accepted": "Signal accepted",
    "best_effort": "Best-effort result (IC threshold not met)",
    "failed": "No valid signals generated",
}


def _eligible_crossover_parent_count(history_entries: list[tuple[int, str, dict, dict]]) -> int:
    """Count candidates that :func:`select_parents` can use as parents."""
    count = 0
    for _, _, signal, metric in history_entries:
        if not isinstance(signal, dict) or not isinstance(metric, dict):
            continue
        try:
            mean_ic = float(metric.get("mean_ic"))
        except (TypeError, ValueError):
            continue
        if math.isfinite(mean_ic):
            count += 1
    return count


def _directional_inversion_advice(ic_results: dict) -> str | None:
    """Return deterministic sign-inversion advice when warranted.

    The directional-IC payload makes the decision from the in-sample split
    only.  A missing, inconclusive, or non-significant payload produces no
    advice, so a weak signal never receives a misleading inversion request.
    """
    directional = ic_results.get("directional_ic")
    if not isinstance(directional, dict) or not directional.get("should_invert"):
        return None

    train_mean_ic = directional.get("train_mean_ic")
    train_p_value = directional.get("train_p_value")
    try:
        ic_text = f"{float(train_mean_ic):.4f}"
    except (TypeError, ValueError):
        ic_text = "N/A"
    try:
        p_value_text = f"{float(train_p_value):.4f}"
    except (TypeError, ValueError):
        p_value_text = "N/A"

    return (
        "DIRECTIONAL IC CHECK: the in-sample IC is negative "
        f"(mean IC {ic_text}, p-value {p_value_text}) and is statistically significant. "
        "INVERT THE SIGNAL SIGN (multiply the signal by -1) so higher signal values "
        "predict higher forward returns."
    )


def _compose_feedback(
    advice: str,
    best_result: dict | None,
    best_ic: float | None,
) -> str:
    """
    Bundle the advisor's advice with the best-known result so the next
    iteration anchors on what already worked instead of wandering.

    Without this, every iteration only sees the *latest* advice and tends to
    drift — sometimes regressing past a good signal it found earlier.
    """
    if not best_result or best_ic is None:
        return advice

    try:
        best_signals = extract_json_array(best_result["signal_json"])
        items = best_signals if best_signals else [json.loads(best_result["signal_json"])]
        best_summary = "\n".join(
            f"- {f.get('name', '?')}: {f.get('formula', '?')}"
            for f in items if isinstance(f, dict)
        )
    except (json.JSONDecodeError, KeyError, TypeError):
        best_summary = "(unable to summarize)"

    return (
        f"BEST SIGNAL(S) SO FAR (iteration {best_result.get('iteration', '?')}, "
        f"|IC| = {abs(best_ic):.4f}):\n{best_summary}\n\n"
        f"ADVICE FROM LAST ITERATION:\n{advice}\n\n"
        "Try to BEAT the best |IC| above. Build on what worked rather than "
        "starting from scratch."
    )


def _parse_request(raw: str) -> tuple[str, str | None]:
    """
    Decode the workflow input.

    Accepts either:
      - A plain signal request string: ``"momentum signals"``
      - A JSON object with a ``request`` field and optional ``seed_feedback``:
        ``{"request": "momentum signals", "seed_feedback": "- try ..."}``

    Returns ``(request_text, seed_feedback_or_None)``. The JSON form is
    opt-in so the standard ``nat run --input "..."`` interface is unchanged.
    """
    text = (raw or "").strip()
    if text.startswith("{"):
        try:
            obj = json.loads(text)
            if isinstance(obj, dict) and "request" in obj:
                return str(obj["request"]), obj.get("seed_feedback")
        except json.JSONDecodeError:
            pass
    return raw, None


# Allowed values for the per-iteration lineage ``source`` tag.
_LINEAGE_SOURCES = ("seed", "resumed", "optimization_feedback", "genetic_crossover")


def _new_run_id() -> str:
    """Short unique id for one workflow invocation (shared by all its files)."""
    return uuid.uuid4().hex[:12]


def _build_lineage(
    iteration: int,
    run_id: str,
    seed_feedback: str | None,
    parent_signal: str | None,
    crossover_merged: bool,
) -> dict:
    """
    Build the per-iteration lineage record for the dashboard's evolution tree.

    - Iteration 1: the run starts from the user's request — parents are None
      and the source is ``"seed"``, or ``"resumed"`` when the run was started
      with ``seed_feedback`` from a prior run's ``last_feedback``.
    - Iteration > 1: the parent is the previous iteration's selected signal
      within the same run (``parent_run_id`` = this run's id). The source is
      ``"genetic_crossover"`` when the candidate batch was seeded by the
      crossover step, otherwise ``"optimization_feedback"``.
    """
    if iteration <= 1:
        source = "resumed" if seed_feedback else "seed"
        return {"parent_signal": None, "parent_run_id": None, "source": source}
    source = "genetic_crossover" if crossover_merged else "optimization_feedback"
    return {"parent_signal": parent_signal, "parent_run_id": run_id, "source": source}


async def generate_signals_concurrently(
    llm,
    request_text: str,
    num_signals: int,
    operators: list[dict],
    feedback: str | None = None,
    max_concurrency: int = 4,
) -> str:
    """
    Generate hypotheses via concurrent one-signal LLM calls.

    Issues ``num_signals`` calls to the existing ``generate_signal_json``
    helper (same prompt building and SystemMessage/HumanMessage pattern as the
    sequential path), each asking for ONE signal, bounded by an
    ``asyncio.Semaphore(max_concurrency)``. Results are combined into a single
    JSON array of signal objects — the same shape the sequential path
    produces — so the downstream parser is unchanged.

    Guarantees:
      - the returned array preserves the original request order regardless of
        completion order (``asyncio.gather`` returns results in input order);
      - one failing call never aborts the batch: it is logged and skipped;
      - if every call fails, ``"[]"`` is returned.
    """
    semaphore = asyncio.Semaphore(max(1, int(max_concurrency)))

    async def generate_one(index: int) -> list[dict]:
        async with semaphore:
            raw = await generate_signal_json(llm, request_text, 1, operators, feedback)
        parsed = extract_json_array(raw)
        valid = [item for item in parsed if isinstance(item, dict)]
        if not valid:
            raise ValueError(f"no valid JSON signal object in response ({len(raw)} chars)")
        return valid

    tasks = [generate_one(index) for index in range(max(0, int(num_signals)))]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    hypotheses: list[dict] = []
    failed = 0
    for index, result in enumerate(results):
        if isinstance(result, BaseException):
            failed += 1
            logger.warning(f"Parallel hypothesis {index + 1}/{num_signals} failed, skipping: {result}")
            continue
        hypotheses.extend(result)

    logger.info(
        f"Parallel generation: requested {num_signals} hypothesis call(s), "
        f"produced {len(hypotheses)} signal(s), {failed} call(s) failed"
    )
    return json.dumps(hypotheses, indent=2)


def _format_workflow_result(
    status: str,
    request: str,
    iteration: int,
    total_iterations: int,
    signal_json: str,
    ic_results: dict,
    saved_path: str | None,
    config,
    last_feedback: str | None = None,
    run_id: str | None = None,
    lineage: dict | None = None,
    skeptic_reviews: list[dict] | None = None,
) -> str:
    """Format the workflow's final result as a structured, human-readable JSON string.

    ``last_feedback`` is the optimization advice from the most recent failed
    iteration. Including it in the result lets a caller resume the loop later
    by passing it back in as the seed feedback for a new run.

    ``run_id`` / ``lineage`` / ``skeptic_reviews`` are ADDITIVE: every
    existing top-level key is unchanged, and the new keys (``run_id``,
    ``lineage``, ``skeptic``, ``overfit_risk``, ``oos_validation``) appear
    only when the caller supplies the corresponding data.
    """
    review_by_name: dict[str, dict] = {}
    for review in skeptic_reviews or []:
        if isinstance(review, dict) and review.get("name"):
            review_by_name[review["name"]] = {
                key: review[key]
                for key in ("verdict", "issues", "static_risks", "vetoed")
                if key in review
            }

    signals_summary = []
    try:
        data = extract_json_array(signal_json) if signal_json else []
        if not data and signal_json:
            parsed = json.loads(signal_json)
            data = parsed if isinstance(parsed, list) else [parsed]
        items = data if isinstance(data, list) else [data]
        for item in items:
            if not isinstance(item, dict):
                continue
            summary = {
                "name": item.get("name"),
                "formula": item.get("formula"),
                "category": item.get("category"),
                "data_fields_used": item.get("data_fields_used"),
                "lookback_periods": item.get("lookback_periods"),
            }
            review = review_by_name.get(item.get("name"))
            if review:
                summary["skeptic"] = review
            signals_summary.append(summary)
    except json.JSONDecodeError:
        pass

    metrics = {
        k: ic_results.get(k)
        for k in ("mean_ic", "ic_std", "ic_ir", "t_stat", "p_value", "num_periods", "positive_ic_ratio")
        if ic_results.get(k) is not None
    }

    payload = {
        "status": status,
        "headline": _STATUS_HEADLINE.get(status, status),
        "request": request,
        "iteration": iteration,
        "total_iterations": total_iterations,
        "selected_signal": ic_results.get("selected_signal"),
        "thresholds": {
            "ic_threshold": config.ic_threshold,
            "p_value_threshold": config.p_value_threshold,
        },
        "metrics": metrics,
        "signals": signals_summary,
        "saved_path": saved_path,
    }

    if ic_results.get("error"):
        payload["error"] = ic_results["error"]

    # Directional IC is additive and present only when its config flag was
    # enabled during evaluation.  Keep the legacy payload key set otherwise.
    directional_ic = ic_results.get("directional_ic")
    if isinstance(directional_ic, dict):
        payload["directional_ic"] = directional_ic

    if last_feedback:
        payload["last_feedback"] = last_feedback

    # Additive lineage metadata: only present when the workflow supplies it.
    if run_id:
        payload["run_id"] = run_id
    if lineage is not None:
        payload["lineage"] = lineage

    # Skeptic review summary: omitted entirely when the feature is off.
    if skeptic_reviews:
        vetoed_reviews = [review for review in skeptic_reviews if review.get("vetoed")]
        payload["skeptic"] = {
            "reviewed": len(skeptic_reviews),
            "vetoed": len(vetoed_reviews),
            "rejections": [
                {"name": review.get("name"), "reasons": review.get("reasons", [])}
                for review in vetoed_reviews
            ],
        }

    # Out-of-sample validation markers: omitted when the feature is off.
    if ic_results.get("overfit_risk") is not None:
        payload["overfit_risk"] = bool(ic_results["overfit_risk"])
    if ic_results.get("oos_validation") is not None:
        payload["oos_validation"] = ic_results["oos_validation"]

    return json.dumps(payload, indent=2, default=str)


def save_signal_results(
    signal_json: str,
    signal_code: str,
    ic_results: dict,
    iteration: int,
    run_id: str | None = None,
    lineage: dict | None = None,
    request: str | None = None,
    save_evolution: bool = True,
    run_id_for_filename: str | None = None,
    skeptic_reviews: list[dict] | None = None,
) -> str:
    """
    Save signal results to ``output/signal_<run_id>_iter<n>.json``.

    Legacy key set (unchanged): ``timestamp``, ``iteration``,
    ``selected_signal``, ``signal_description``, ``signal_code``,
    ``evaluation_metrics``.

    When ``save_evolution`` is True, the payload additionally carries
    ``run_id``, ``lineage``, and ``request`` (each only when not None), plus
    ``skeptic_reviews`` (the per-candidate pre-backtest review verdicts) when
    a non-empty list is supplied — so a skeptic veto stays auditable from the
    saved artifacts. Each review dict is expected to already be
    JSON-serialisable with keys ``name``, ``verdict`` ("approve"/"reject"),
    ``issues``, ``static_risks``, ``vetoed``, and ``reasons``. The list is
    stored as a shallow copy and never mutated or reordered. When
    ``save_evolution`` is False, the payload keeps EXACTLY the legacy key set
    — ``skeptic_reviews`` is ignored in that mode, even if passed. The key is
    also omitted when the list is None or empty, keeping payloads lean. The
    filename embeds the run id whenever one is supplied
    (``run_id_for_filename`` takes precedence over the ``run_id`` metadata);
    with no run id at all the legacy ``signal_<timestamp>_iter<n>.json`` form
    is kept, so old files and any ``*.json`` globbing reader keep working.
    """
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename_run_id = run_id_for_filename or run_id
    if filename_run_id:
        filepath = OUTPUT_DIR / f"signal_{filename_run_id}_iter{iteration}.json"
    else:
        filepath = OUTPUT_DIR / f"signal_{timestamp}_iter{iteration}.json"

    metrics = {k: v for k, v in ic_results.items() if k != "selected_signal"}
    payload = {
        "timestamp": timestamp,
        "iteration": iteration,
        "selected_signal": ic_results.get("selected_signal"),
        "signal_description": signal_json,
        "signal_code": signal_code,
        "evaluation_metrics": metrics,
    }
    if save_evolution:
        if run_id is not None:
            payload["run_id"] = run_id
        if lineage is not None:
            payload["lineage"] = lineage
        if request is not None:
            payload["request"] = request
        if skeptic_reviews:
            # Copy so later caller-side mutation cannot corrupt the saved file.
            payload["skeptic_reviews"] = list(skeptic_reviews)
    with open(filepath, "w") as f:
        json.dump(payload, f, indent=2)
    logger.info(f"Saved signal results to {filepath}")
    return str(filepath)


class SignalOptimizerConfig(FunctionBaseConfig, name="signal_optimizer"):
    """Iteratively generate, evaluate, and refine quantitative signals."""

    signal_generator_llm: str | None = Field(default=None, description="LLM for signal generation")
    code_generator_llm: str | None = Field(default=None, description="LLM for code generation")
    optimization_advisor_llm: str | None = Field(default=None, description="LLM for optimization feedback")
    llm_name: str | None = Field(default=None, description="Fallback LLM if specific ones not set")
    ic_threshold: float = Field(default=0.03, description="Minimum |IC| to accept")
    p_value_threshold: float = Field(default=0.05, description="Maximum p-value")
    max_iterations: int = Field(default=3, description="Max optimization attempts")
    num_signals: int = Field(default=5, description="Signals per iteration")
    forward_periods: int = Field(default=5, description="Forward return periods")
    save_results: bool = Field(default=True, description="Save results to disk")
    parallel_generation: bool = Field(
        default=False, description="Generate the hypotheses via concurrent one-signal LLM calls"
    )
    max_concurrency: int = Field(
        default=4, description="Max simultaneous hypothesis LLM calls when parallel_generation is on"
    )
    genetic_crossover: bool = Field(
        default=False, description="Breed the two best signals into an extra crossover candidate"
    )
    genetic_parents: int = Field(default=2, description="Top-|IC| parents feeding the crossover prompt")
    crossover_fallback_enabled: bool = Field(
        default=False,
        description="Use earlier same-run parents when the current iteration has fewer than two parents",
    )
    skeptic_review: bool = Field(
        default=False, description="Run a pre-backtest skeptic review of each candidate"
    )
    skeptic_llm: str | None = Field(
        default=None,
        description="Optional LLM for the skeptic review; None runs deterministic static checks only",
    )
    directional_ic_enabled: bool = Field(
        default=False,
        description="Detect statistically significant signals oriented in the wrong direction",
    )
    oos_validation: bool = Field(
        default=False, description="In-sample/out-of-sample IC split to detect overfitting"
    )
    oos_train_ratio: float = Field(default=0.7, description="Share of history used for the in-sample split")
    oos_reject_overfit: bool = Field(
        default=False, description="Refuse to accept signals flagged as overfit by the OOS check"
    )
    save_evolution: bool = Field(
        default=True, description="Persist run_id/lineage/request metadata in saved result files"
    )


@register_function(config_type=SignalOptimizerConfig, framework_wrappers=[LLMFrameworkEnum.LANGCHAIN])
async def signal_optimizer_function(config: SignalOptimizerConfig, builder: Builder):
    """Compose the signal / code / evaluator agents into an optimization loop."""

    signal_llm_name = config.signal_generator_llm or config.llm_name
    code_llm_name = config.code_generator_llm or config.llm_name
    advisor_llm_name = config.optimization_advisor_llm or config.llm_name
    if not signal_llm_name:
        raise ValueError("Must specify llm_name or signal_generator_llm")

    signal_llm = await builder.get_llm(signal_llm_name, wrapper_type=LLMFrameworkEnum.LANGCHAIN)
    code_llm = await builder.get_llm(code_llm_name or signal_llm_name, wrapper_type=LLMFrameworkEnum.LANGCHAIN)
    advisor_llm = await builder.get_llm(advisor_llm_name or signal_llm_name, wrapper_type=LLMFrameworkEnum.LANGCHAIN)
    # The skeptic LLM is opt-in: when unset, only the deterministic static
    # risk scan runs (a broken/absent LLM must never gate the pipeline alone).
    skeptic_llm = None
    if config.skeptic_review and config.skeptic_llm:
        skeptic_llm = await builder.get_llm(config.skeptic_llm, wrapper_type=LLMFrameworkEnum.LANGCHAIN)
    logger.info(
        f"LLMs: signal={signal_llm_name}, code={code_llm_name}, advisor={advisor_llm_name}, "
        f"skeptic={config.skeptic_llm if config.skeptic_review else '(disabled)'}"
    )

    operators = load_calculator_operators()
    stock_data = load_stock_data()

    # ---- per-step helpers (close over the LLMs and shared state) ----

    def evaluate_ic(signal_code: str, oos_data_out: dict | None = None) -> dict[str, Any]:
        """Run the signal code and compute its rank IC against forward returns.

        ``oos_data_out`` is an optional out-parameter: when provided it
        receives the evaluated ``signal_values`` and ``forward_returns``
        DataFrames so the caller can run the out-of-sample validation gate
        without re-executing the signal code. The return contract is unchanged.
        """
        if not stock_data:
            return {"error": "No stock data", "mean_ic": None}

        clean_code = extract_code_from_response(signal_code)
        exec_result = execute_signal_code(
            clean_code,
            stock_data,
            selection_periods=config.forward_periods,
        )
        if exec_result is None:
            return {"error": "Code execution failed", "mean_ic": None}
        signal_values, selected_signal = exec_result

        close = stock_data.get("Close")
        if close is None:
            return {"error": "No Close data", "mean_ic": None}

        # Suppress numpy/pandas RuntimeWarnings from rolling-window operators
        # over windows that contain NaN — the result is correctly NaN.
        with np.errstate(invalid="ignore", divide="ignore"), warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=RuntimeWarning)
            forward_returns = compute_forward_returns(close, periods=config.forward_periods)
            if oos_data_out is not None:
                oos_data_out["signal_values"] = signal_values
                oos_data_out["forward_returns"] = forward_returns
            ic_results = compute_rank_ic(signal_values, forward_returns)

            # Sign-aware directional IC is opt-in.  Its orientation decision
            # uses the in-sample split only; the same sign is then applied to
            # the out-of-sample split by ``compute_directional_ic``.
            if config.directional_ic_enabled:
                try:
                    ic_results["directional_ic"] = compute_directional_ic(
                        signal_values,
                        forward_returns,
                        train_ratio=config.oos_train_ratio,
                        ic_threshold=config.ic_threshold,
                        p_value_threshold=config.p_value_threshold,
                    )
                except Exception as exc:
                    logger.warning(f"Directional IC unavailable; continuing without a decision: {exc}")
                    ic_results["directional_ic"] = {
                        "direct_mean_ic": None,
                        "inverse_mean_ic": None,
                        "absolute_ic": None,
                        "train_mean_ic": None,
                        "train_p_value": None,
                        "train_num_periods": 0,
                        "inverse_train_mean_ic": None,
                        "test_mean_ic": None,
                        "test_p_value": None,
                        "test_num_periods": 0,
                        "inverse_test_mean_ic": None,
                        "oriented_test_mean_ic": None,
                        "oriented_test_p_value": None,
                        "detected_sign": 0,
                        "decision": "none",
                        "decision_reason": "directional_ic_computation_failed",
                        "is_significant": False,
                        "should_invert": False,
                        "oos_reason": "directional_ic_computation_failed",
                        "split_date": None,
                        "error": str(exc),
                    }

            # Portfolio metrics (Sharpe, Sortino, drawdowns, turnover, ...).
            # The workflow consumes DAILY bars (`forward_periods` is in days),
            # so periods_per_year=252. Lazy import: quant_metrics may be absent,
            # and any failure here must never kill the run. Metrics are merged
            # additively — keys already present in the rank-IC results (notably
            # `num_periods`, the usable-IC-date count) are NOT clobbered by the
            # portfolio's own `num_periods` (portfolio return periods).
            try:
                from signal_discovery_workflow.quant_metrics import (
                    build_portfolio_weights,
                    compute_portfolio_metrics,
                    compute_turnover,
                    portfolio_returns_from_weights,
                )

                weights = build_portfolio_weights(signal_values, top_quantile=0.2, bottom_quantile=0.2)
                portfolio_returns = portfolio_returns_from_weights(weights, forward_returns)
                portfolio_metrics = compute_portfolio_metrics(portfolio_returns, periods_per_year=252)
                turnover = compute_turnover(weights, periods_per_year=252)
                ic_results.update(
                    {
                        key: value
                        for key, value in {**portfolio_metrics, "turnover": turnover}.items()
                        if key not in ic_results
                    }
                )
            except Exception as exc:
                logger.warning(f"Portfolio metrics unavailable; continuing with rank-IC metrics only: {exc}")
        ic_results["selected_signal"] = selected_signal
        return ic_results

    def is_acceptable(ic_results: dict) -> bool:
        mean_ic = ic_results.get("mean_ic")
        p_value = ic_results.get("p_value")
        if mean_ic is None or abs(mean_ic) < config.ic_threshold:
            return False
        if p_value is not None and p_value > config.p_value_threshold:
            return False
        directional = ic_results.get("directional_ic")
        if isinstance(directional, dict) and directional.get("should_invert"):
            # The magnitude is significant, but the IS-derived orientation is
            # wrong.  Ask the advisor to invert it instead of accepting it.
            logger.info(
                "Signal acceptance deferred: the significant in-sample IC has a negative "
                "orientation; the advisor will be asked to invert the signal sign"
            )
            return False
        return True

    async def generate_feedback(signal_json: str, ic_results: dict, iteration: int) -> str:
        """
        Ask the advisor LLM for compact, actionable feedback.

        The full signal JSON is sent so the advisor sees every field
        (name, formula, meaning, category, lookbacks). Output is constrained
        to a few short bullets so it fits comfortably in the next iteration's
        signal-generator prompt.
        """
        mean_ic = ic_results.get("mean_ic")
        p_value = ic_results.get("p_value")
        mean_ic_str = f"{mean_ic:.4f}" if mean_ic is not None else "N/A"
        p_value_str = f"{p_value:.4f}" if p_value is not None else "N/A"
        error = ic_results.get("error", "")
        error_line = f"- Error: {error}\n" if error else ""

        prompt = f"""Iteration {iteration} results:
- Mean IC: {mean_ic_str} (target >= {config.ic_threshold})
- P-value: {p_value_str} (target <= {config.p_value_threshold})
{error_line}
SIGNALS TRIED:
{signal_json}

Output exactly 3 bullet points, max 15 words each, suggesting concrete
changes (different operator, different lookback, different data field).
No prose, no preamble. Format:
- <change 1>
- <change 2>
- <change 3>"""

        response = await advisor_llm.ainvoke(
            [
                SystemMessage(content=NO_THINK_INSTRUCTION),
                SystemMessage(content="You are a senior quant providing concise signal optimization advice."),
                HumanMessage(content=prompt),
            ]
        )
        feedback = extract_response_text(response).strip()
        # Hard cap so a chatty model can't blow up the next iteration's prompt.
        if len(feedback) > 500:
            feedback = feedback[:500].rsplit("\n", 1)[0]
        return feedback

    def run_oos_validation(signal_values, forward_returns, train_ratio: float) -> dict | None:
        """
        Chronological in-sample/out-of-sample IC split for one candidate.

        The ``quant_metrics`` import is deliberately lazy: the module may be
        absent or mid-refactor, in which case the OOS check is skipped with a
        warning instead of crashing the run.
        """
        try:
            from signal_discovery_workflow.quant_metrics import cross_validate_signal
        except ImportError as exc:
            logger.warning(f"OOS validation skipped: quant_metrics unavailable ({exc})")
            return None
        except Exception as exc:
            logger.warning(f"OOS validation skipped: quant_metrics import failed ({exc})")
            return None
        try:
            return cross_validate_signal(signal_values, forward_returns, train_ratio=train_ratio)
        except Exception as exc:
            logger.warning(f"OOS validation failed: {exc}")
            return None

    async def run_skeptic_gate(
        signal_json: str,
        signal_code: str,
        codegen_errors: list[str],
    ) -> tuple[str | None, str, list[dict]]:
        """
        Review every candidate BEFORE the backtest; veto flawed ones.

        For each candidate the deterministic static scan runs first; when a
        skeptic LLM is configured its verdict is consulted as well. A candidate
        is vetoed when it carries a blocking static risk (lookahead / leakage /
        degenerate) or the LLM verdict is ``reject``.

        Returns ``(filtered_signal_json, code_to_evaluate, reviews)``. A None
        ``filtered_signal_json`` means EVERY candidate was vetoed and the
        backtest must be skipped for this iteration. Surviving candidates are
        re-generated through the normal code-generation path so vetoed signal
        functions cannot win the in-module best-signal selection.
        """
        candidates = [item for item in extract_json_array(signal_json) if isinstance(item, dict)]
        if not candidates:
            # Nothing parseable to review — let the structural-error path handle it.
            return signal_json, signal_code, []

        reviews: list[dict] = []
        kept: list[dict] = []
        for index, description in enumerate(candidates):
            function_name = _python_function_name(description.get("name"), index)
            code_slice = extract_function_source(signal_code, function_name) or signal_code
            risks = detect_static_risks(description, code=code_slice)

            verdict, issues = "approve", []
            if skeptic_llm is not None:
                prompt = build_skeptic_prompt(description, code=code_slice, static_risks=risks)
                try:
                    response = await skeptic_llm.ainvoke(
                        [
                            SystemMessage(content=NO_THINK_INSTRUCTION),
                            HumanMessage(content=prompt),
                        ]
                    )
                    review = parse_skeptic_review(extract_response_text(response))
                    verdict, issues = review["verdict"], review["issues"]
                except Exception as exc:
                    # Fail open: a broken skeptic LLM must not kill the run.
                    logger.warning(f"Skeptic LLM review failed for '{description.get('name', '?')}': {exc}")

            blocking = has_blocking_risk(risks)
            vetoed = blocking or verdict == "reject"
            reasons = list(risks) if blocking else list(issues)
            reviews.append(
                {
                    "name": description.get("name"),
                    "verdict": "reject" if vetoed else verdict,
                    "issues": issues,
                    "static_risks": risks,
                    "vetoed": vetoed,
                    "reasons": reasons,
                }
            )
            if vetoed:
                logger.info(f"Skeptic vetoed candidate '{description.get('name', '?')}': {reasons}")
            else:
                kept.append(description)

        if not kept:
            return None, signal_code, reviews
        if len(kept) == len(candidates):
            return signal_json, signal_code, reviews

        filtered_json = json.dumps(kept, indent=2)
        filtered_code = await generate_signal_code(
            code_llm, filtered_json, operators, errors_out=codegen_errors
        )
        return filtered_json, filtered_code, reviews

    # ---- main optimization loop ----

    async def run_optimization(request: str) -> str:
        """
        Run the closed-loop signal discovery optimization.

        Args:
            request: Either a plain signal request string (e.g.,
                ``"momentum signals"``) or a JSON object with the shape
                ``{"request": "momentum signals", "seed_feedback": "..."}``
                to resume from a prior run's ``last_feedback``.

                The JSON form is opt-in so the CLI-friendly single-string
                interface still works (NAT's input schema is single-arg).
        """
        request_text, seed_feedback = _parse_request(request)

        # One stable id per workflow invocation, shared by every iteration's
        # saved file so a single run's artifacts are contiguous on disk.
        # Feature flags are read defensively so old config files keep working.
        run_id = _new_run_id()
        save_evolution = bool(getattr(config, "save_evolution", True))
        parallel_generation = bool(getattr(config, "parallel_generation", False))
        max_concurrency = int(getattr(config, "max_concurrency", 4) or 4)
        skeptic_enabled = bool(getattr(config, "skeptic_review", False))
        genetic_enabled = bool(getattr(config, "genetic_crossover", False))
        genetic_parents_k = max(2, int(getattr(config, "genetic_parents", 2)))
        crossover_fallback_enabled = bool(getattr(config, "crossover_fallback_enabled", False))
        oos_validation_enabled = bool(getattr(config, "oos_validation", False))
        oos_reject_overfit = bool(getattr(config, "oos_reject_overfit", False))
        oos_train_ratio = float(getattr(config, "oos_train_ratio", 0.7))

        best_result: dict | None = None
        best_ic: float | None = None
        feedback: str | None = seed_feedback
        last_error: str | None = None
        prev_selected_signal: str | None = None
        iteration_saved_paths: dict[int, str] = {}
        # Genetic crossover state: candidates bred at the end of iteration N
        # join iteration N+1's batch, and the evaluated-candidate history the
        # parents are selected from (capped to the most recent entries).
        carryover: list[dict] = []
        candidate_history: list[tuple[dict, dict]] = []
        # Parallel metadata used only by the opt-in earlier-iteration fallback.
        # Each entry is ``(iteration, run_id, signal, metrics)`` so a fallback
        # can prove that every parent belongs to this run and an earlier
        # iteration of it.
        candidate_history_context: list[tuple[int, str, dict, dict]] = []
        last_skeptic_reviews: list[dict] = []

        async def generate_crossover_candidate(current_iteration: int) -> dict | None:
            """Breed the two best evaluated candidates into ONE new hypothesis."""
            history_signals = [signal for signal, _ in candidate_history]
            history_metrics = [metric for _, metric in candidate_history]

            if not crossover_fallback_enabled:
                # Legacy path: select from the complete per-run history exactly
                # as before.  The fallback flag must not alter this behavior.
                parents = select_parents(history_signals, history_metrics, k=genetic_parents_k)
            else:
                current_entries = [
                    entry
                    for entry in candidate_history_context
                    if entry[0] == current_iteration and entry[1] == run_id
                ]
                current_count = _eligible_crossover_parent_count(current_entries)

                if current_count >= 2:
                    # The current iteration already supplies enough parents, so
                    # retain the existing selection path unchanged.
                    parents = select_parents(history_signals, history_metrics, k=genetic_parents_k)
                else:
                    earlier_entries = [
                        entry
                        for entry in candidate_history_context
                        if entry[0] < current_iteration and entry[1] == run_id
                    ]
                    earlier_count = _eligible_crossover_parent_count(earlier_entries)
                    earlier_signals = [entry[2] for entry in earlier_entries]
                    earlier_metrics = [entry[3] for entry in earlier_entries]
                    parents = select_parents(earlier_signals, earlier_metrics, k=genetic_parents_k)
                    if len(parents) < 2:
                        if current_count + earlier_count < 2:
                            logger.warning(
                                "Genetic crossover fallback skipped: fewer than two evaluated parents "
                                f"exist across run {run_id} (current iteration={current_count}, "
                                f"earlier iterations={earlier_count})"
                            )
                        else:
                            logger.warning(
                                "Genetic crossover fallback skipped: earlier iterations of run "
                                f"{run_id} contain fewer than two eligible parents "
                                f"(current iteration={current_count}, earlier iterations={earlier_count})"
                            )
                        return None

                    selected_parent_ids = {id(parent) for parent in parents}
                    source_iterations = sorted(
                        {
                            entry[0]
                            for entry in earlier_entries
                            if id(entry[2]) in selected_parent_ids
                        }
                    )
                    logger.info(
                        "Genetic crossover fallback: using "
                        f"{len(parents)} parent(s) from earlier iteration(s) "
                        f"{source_iterations} of run {run_id}"
                    )

            if len(parents) < 2:
                logger.info("Genetic crossover skipped: fewer than two evaluated parents")
                return None
            extra_note = ""
            if len(parents) > 2:
                extra_note = "Additional high-|IC| parents for inspiration:\n" + "\n".join(
                    f"- {parent.get('name', '?')}: {parent.get('formula', '?')}" for parent in parents[2:]
                )
            prompt = build_crossover_prompt(
                parents[0], parents[1], operator_catalog=operators, feedback=extra_note or None
            )
            try:
                response = await signal_llm.ainvoke(
                    [
                        SystemMessage(content=NO_THINK_INSTRUCTION),
                        HumanMessage(content=prompt),
                    ]
                )
                candidate = parse_crossover_response(extract_response_text(response))
            except Exception as exc:
                logger.warning(f"Genetic crossover LLM call failed: {exc}")
                return None
            if candidate is None:
                logger.warning("Genetic crossover response could not be parsed; skipping crossover candidate")
                return None
            logger.info(f"Genetic crossover produced candidate '{candidate.get('name', '?')}'")
            return candidate

        if seed_feedback:
            logger.info(f"Resuming with seed feedback ({len(seed_feedback)} chars)")
        logger.info(f"Run {run_id} started (parallel_generation={parallel_generation})")

        for iteration in range(1, config.max_iterations + 1):
            logger.info(f"=== Iteration {iteration}/{config.max_iterations} ===")

            logger.info("Generating signals...")
            if parallel_generation and config.num_signals > 1:
                signal_json = await generate_signals_concurrently(
                    signal_llm, request_text, config.num_signals, operators, feedback, max_concurrency
                )
            else:
                signal_json = await generate_signal_json(
                    signal_llm, request_text, config.num_signals, operators, feedback
                )

            # Crossover candidates bred last iteration join this iteration's
            # batch BEFORE code generation, so they flow through the skeptic
            # gate, code generation, and the backtest like any LLM candidate.
            crossover_merged = False
            if carryover:
                generated = [item for item in extract_json_array(signal_json) if isinstance(item, dict)]
                signal_json = json.dumps([*generated, *carryover], indent=2)
                carryover = []
                crossover_merged = True

            # ``crossover_merged`` selects the lineage ``source`` for this iteration.
            current_lineage = _build_lineage(
                iteration, run_id, seed_feedback, prev_selected_signal, crossover_merged
            )

            logger.info("Generating code...")
            codegen_errors: list[str] = []
            signal_code = await generate_signal_code(
                code_llm, signal_json, operators, errors_out=codegen_errors
            )

            # ---- skeptic gate (pre-backtest veto) ----
            iteration_skeptic_reviews: list[dict] = []
            if skeptic_enabled:
                gate_json, gate_code, gate_reviews = await run_skeptic_gate(
                    signal_json, signal_code, codegen_errors
                )
                if gate_json is None:
                    # Every candidate was vetoed before the backtest: skip
                    # evaluation for this iteration (same pattern as the
                    # structural parse-error path) and let the rejection
                    # reasons drive the next iteration's feedback.
                    rejection_reasons = [
                        f"{review.get('name', '?')}: "
                        + ("; ".join(review.get("reasons") or []) or review.get("verdict", "rejected"))
                        for review in gate_reviews
                    ]
                    last_error = "skeptic review rejected all candidates"
                    logger.info(f"Skeptic review vetoed all {len(gate_reviews)} candidate(s); skipping backtest")
                    if config.save_results and save_evolution:
                        iteration_saved_paths[iteration] = save_signal_results(
                            signal_json,
                            signal_code,
                            {"error": last_error, "mean_ic": None},
                            iteration,
                            run_id=run_id,
                            lineage=current_lineage,
                            request=request_text,
                            save_evolution=True,
                            skeptic_reviews=gate_reviews,
                        )
                    last_skeptic_reviews = gate_reviews
                    advice = (
                        "All candidates were rejected by the skeptic review "
                        "(lookahead bias, leakage, or degenerate logic). Rewrite them using only "
                        "past data and standard operators. Specific rejections:\n"
                        + "\n".join(f"- {reason}" for reason in rejection_reasons)
                    )
                    feedback = _compose_feedback(advice, best_result, best_ic)
                    continue
                signal_json, signal_code, iteration_skeptic_reviews = gate_json, gate_code, gate_reviews

            logger.info("Evaluating IC...")
            oos_bundle: dict | None = {} if oos_validation_enabled else None
            ic_results = evaluate_ic(signal_code, oos_data_out=oos_bundle)
            mean_ic = ic_results.get("mean_ic")
            if ic_results.get("error"):
                last_error = str(ic_results["error"])
            prev_selected_signal = ic_results.get("selected_signal")

            # ---- out-of-sample gate (overfitting detection) ----
            overfit_rejected = False
            if oos_bundle and mean_ic is not None and "signal_values" in oos_bundle:
                oos_result = run_oos_validation(
                    oos_bundle["signal_values"], oos_bundle["forward_returns"], oos_train_ratio
                )
                if oos_result is not None:
                    ic_results["oos_validation"] = oos_result
                    ic_results["overfit_risk"] = bool(oos_result.get("is_overfit"))
                    if ic_results["overfit_risk"] and oos_reject_overfit:
                        overfit_rejected = True
                        logger.info("Candidate failed out-of-sample validation; refusing acceptance")

            # ---- genetic crossover: remember evaluated candidates ----
            if genetic_enabled:
                for item in extract_json_array(signal_json):
                    if isinstance(item, dict):
                        candidate_history.append((item, ic_results))
                        candidate_history_context.append((iteration, run_id, item, ic_results))
                if len(candidate_history) > 50:
                    candidate_history = candidate_history[-50:]
                    candidate_history_context = candidate_history_context[-50:]

            # With evolution tracking on, persist EVERY iteration so the
            # dashboard can render idea -> iteration -> final signal trees.
            # With it off, keep the legacy cadence (accepted / best-effort only).
            if config.save_results and save_evolution:
                iteration_saved_paths[iteration] = save_signal_results(
                    signal_json,
                    signal_code,
                    ic_results,
                    iteration,
                    run_id=run_id,
                    lineage=current_lineage,
                    request=request_text,
                    save_evolution=True,
                    skeptic_reviews=iteration_skeptic_reviews,
                )

            if mean_ic is not None:
                logger.info(f"Mean IC: {mean_ic:.4f}, p-value: {ic_results.get('p_value', 'N/A')}")
                if best_ic is None or abs(mean_ic) > abs(best_ic):
                    best_ic = mean_ic
                    best_result = {
                        "signal_json": signal_json,
                        "signal_code": signal_code,
                        "ic_results": ic_results,
                        "iteration": iteration,
                        "lineage": current_lineage,
                        "skeptic_reviews": iteration_skeptic_reviews,
                    }

            if is_acceptable(ic_results) and not overfit_rejected:
                logger.info("Signal accepted!")
                if config.save_results and not save_evolution:
                    saved_path = save_signal_results(
                        signal_json,
                        signal_code,
                        ic_results,
                        iteration,
                        run_id_for_filename=run_id,
                        save_evolution=False,
                    )
                else:
                    # Already persisted per-iteration above.
                    saved_path = iteration_saved_paths.get(iteration)
                return _format_workflow_result(
                    status="accepted",
                    request=request_text,
                    iteration=iteration,
                    total_iterations=config.max_iterations,
                    signal_json=signal_json,
                    ic_results=ic_results,
                    saved_path=saved_path,
                    config=config,
                    last_feedback=feedback,
                    run_id=run_id,
                    lineage=current_lineage if save_evolution else None,
                    skeptic_reviews=iteration_skeptic_reviews,
                )

            # When every signal was rejected at parse/arity time the advisor
            # has nothing to say (no IC numbers exist). Skip the LLM round-trip
            # and feed the structural errors directly to the next iteration so
            # the generator can self-correct.
            if codegen_errors and mean_ic is None:
                last_error = "; ".join(codegen_errors)
                advice = (
                    "Your previous signals were ALL REJECTED before evaluation. "
                    "Re-read each operator signature carefully and match the argument "
                    "count exactly. Specific failures:\n"
                    + "\n".join(f"- {e}" for e in codegen_errors)
                )
                logger.info(f"Skipping advisor: {len(codegen_errors)} structural errors will drive next iteration")
            else:
                logger.info("Generating optimization feedback...")
                advice = await generate_feedback(signal_json, ic_results, iteration)

            # An overfit-rejected candidate feeds a targeted note into the
            # next iteration's optimization feedback.
            if overfit_rejected:
                oos_summary = ic_results.get("oos_validation") or {}
                advice = (
                    "OUT-OF-SAMPLE CHECK FAILED: in-sample IC "
                    f"{mean_ic:.4f} did not generalize (out-of-sample IC "
                    f"{oos_summary.get('test_mean_ic')}). "
                    "Prefer simpler, more stable signals."
                    f"\n{advice}"
                )

            # A significant negative IS orientation gets a deterministic note
            # in addition to the advisor's bullets.  This advice is part of the
            # composed feedback passed to the next iteration.
            inversion_advice = _directional_inversion_advice(ic_results)
            if inversion_advice:
                advice = f"{inversion_advice}\n{advice}"

            # Compose the feedback shown to the next iteration: anchor it on
            # the best-known result so the model has a concrete target to
            # beat, then append the advisor's bullets (or arity errors).
            feedback = _compose_feedback(advice, best_result, best_ic)

            # ---- genetic crossover: breed the best candidates ----
            if genetic_enabled:
                crossover_candidate = await generate_crossover_candidate(iteration)
                if crossover_candidate is not None:
                    carryover.append(crossover_candidate)

            last_skeptic_reviews = iteration_skeptic_reviews

        if best_result:
            if config.save_results and not save_evolution:
                saved_path = save_signal_results(
                    best_result["signal_json"],
                    best_result["signal_code"],
                    best_result["ic_results"],
                    best_result["iteration"],
                    run_id_for_filename=run_id,
                    save_evolution=False,
                )
            else:
                saved_path = iteration_saved_paths.get(best_result["iteration"])
            return _format_workflow_result(
                status="best_effort",
                request=request_text,
                iteration=best_result["iteration"],
                total_iterations=config.max_iterations,
                signal_json=best_result["signal_json"],
                ic_results=best_result["ic_results"],
                saved_path=saved_path,
                config=config,
                last_feedback=feedback,
                run_id=run_id,
                lineage=best_result.get("lineage") if save_evolution else None,
                skeptic_reviews=best_result.get("skeptic_reviews"),
            )

        return _format_workflow_result(
            status="failed",
            request=request_text,
            iteration=0,
            total_iterations=config.max_iterations,
            signal_json="",
            ic_results={"error": last_error} if last_error else {},
            saved_path=None,
            config=config,
            last_feedback=feedback,
            run_id=run_id,
            skeptic_reviews=last_skeptic_reviews or None,
        )

    yield FunctionInfo.from_fn(
        run_optimization,
        description="Run signal discovery optimization loop: generate -> code -> evaluate -> feedback.",
    )
