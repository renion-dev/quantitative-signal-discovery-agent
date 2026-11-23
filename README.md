cd /home/ihor/projects/nvidia-quant-signal && pkill -f "uvicorn dashboard.app" 2>/dev/null; sleep 1; .venv/bin/python -m uvicorn dashboard.app:app --host 0.0.0.0 --port 8090
# Quantitative Signal Discovery Agent developer example

An end-to-end signal discovery workflow for quantitative finance using NVIDIA NeMo Agent Toolkit. This workflow demonstrates how to leverage LLMs to automatically generate, code, and evaluate alpha signals.

## Overview

Signal discovery is the process of finding quantitative signals — also known as alpha signals — that have predictive power for future stock returns. This workflow automates the traditional labor-intensive process using LLMs.

### Workflow Architecture

![Workflow Architecture](notebooks/images/workflow-architecture.png)

The workflow uses a **closed-loop optimization** approach:
1. Generate signal ideas using an LLM
2. Convert ideas to executable Python code
3. Evaluate the signal's predictive power (Rank IC)
4. If IC meets threshold → Accept and save
5. If IC is poor → Generate optimization advice and retry

## Getting Started

### Prerequisites

- **Platform:** Linux, macOS, or Windows
- **Python:** version 3.11, 3.12, or 3.13
- **Package manager:** pip or uv

### API Keys

You will need an NVIDIA API key. Get yours from [build.nvidia.com](https://build.nvidia.com/settings/api-keys).

```bash
export NVIDIA_API_KEY="your-api-key-here"
```

### Installing Dependencies

```bash
uv sync --locked
```

(All commands below use `uv run ...` so you don't need to activate the venv. If you prefer to activate it once and drop the `uv run` prefix, run `source .venv/bin/activate` first.)

### Download Data

The workflow requires S&P 500 price-volume data (Open, Close, High, Low, Volume). Use the included script to download fresh data via [yfinance](https://github.com/ranaroussi/yfinance):

```bash
uv run python -m signal_discovery_workflow.download_data
```

You can customize the date range:

```bash
uv run python -m signal_discovery_workflow.download_data --start 2015-01-01 --end 2025-12-31
```

> **Disclaimer:** Each user is responsible for checking the content of datasets and the applicable licenses and determining if suitable for the intended use.

## Deployment Options

This workflow can be deployed in three ways:

### Deploy on Brev

Spin up a one-click GPU instance on [brev.nvidia.com](https://brev.nvidia.com/) and open [`brev/launchable-setup.ipynb`](brev/launchable-setup.ipynb). It clones the repo, installs `uv`, syncs dependencies, and registers a Jupyter kernel. When it finishes, follow the inline link to jump into [`brev/signal-discovery-workflow.ipynb`](brev/signal-discovery-workflow.ipynb), which uses [`configs/config-optimization-no-telemetry.yml`](configs/config-optimization-no-telemetry.yml) so the cloud notebook does not require a separate Phoenix server.

### Option 1: Interactive Notebook Deployment

Best for exploration, experimentation, and learning. The notebook provides step-by-step execution with inline documentation.

```bash
uv run jupyter notebook notebooks/signal-discovery-workflow.ipynb
```

The notebook includes:
- API key setup
- Configuration exploration
- Step-by-step workflow execution
- Interactive result visualization
- Ability to modify parameters on-the-fly

### Option 2: CLI Deployment

Best for production, automation, and scripting. Run the workflow directly from the command line.

#### Basic Usage

```bash
# Run the signal discovery workflow
uv run nat run --config_file configs/config-optimization.yml --input "momentum signals"
```

#### With Phoenix Tracing (Recommended)

For full observability with LLM tracing, run Phoenix in a separate terminal first:

**Terminal 1 - Start Phoenix Server:**
```bash
uv run phoenix serve
```

Phoenix will start at http://localhost:6006

**Terminal 2 - Run the Workflow:**
```bash
export NVIDIA_API_KEY="your-api-key-here"
uv run nat run --config_file configs/config-optimization.yml --input "momentum signals"
```

View traces at http://localhost:6006 to see:
- LLM calls and responses
- Token usage
- Latency metrics
- Full execution trace

#### Running Different Signal Types

```bash
# Generate volatility signals
uv run nat run --config_file configs/config-optimization.yml --input "volatility signals"

# Generate mean reversion signals
uv run nat run --config_file configs/config-optimization.yml --input "mean reversion signals"

# Generate volume-based signals
uv run nat run --config_file configs/config-optimization.yml --input "volume price divergence signals"
```

## Components

| Component | Description |
|-----------|-------------|
| **Signal Agent** | Uses an LLM to generate signal expressions based on price-volume data and operators |
| **Code Agent** | Wraps each signal formula in a Python function via an LLM, and inlines the required operator implementations from `calculator.json` to produce a self-contained executable module |
| **Eval Agent** | Performs backtesting via Rank IC and generates optimization suggestions |
| **Data Download Script** | Fetches S&P 500 price-volume data from Yahoo Finance via `yfinance` |

## Configuration

The workflow configuration is defined in `configs/config-optimization.yml`:

> **Note:** The `base_url` for the LLMs depends on your API key. Set it to either:
> - `https://integrate.api.nvidia.com/v1/` — for keys from [build.nvidia.com](https://build.nvidia.com)
> - `https://inference-api.nvidia.com/v1/` — for NVIDIA internal or enterprise API keys

| Parameter | Description |
|-----------|-------------|
| `signal_generator_llm` | Reference to the LLM block used for signal ideation (typically higher temperature for creativity) |
| `code_generator_llm` | Reference to the LLM block used for translating formulas into Python (low temperature for determinism) |
| `optimization_advisor_llm` | Reference to the LLM block used to produce iteration feedback (balanced temperature) |
| `ic_threshold` | Minimum absolute IC value to accept a signal (e.g., 0.02 = 2%) |
| `p_value_threshold` | Maximum p-value for statistical significance (e.g., 0.05 = 5%) |
| `max_iterations` | Maximum number of optimization iterations before returning the best result |
| `num_signals` | Number of signals to generate per iteration |
| `forward_periods` | Number of days for forward return calculation (e.g., 5 = weekly) |
| `save_results` | Whether to save accepted/best-effort signals to `output/` |

You can use the same model for all three agents, or assign separate models to the Signal, Code, and Advisor agents by editing the `llms` blocks in the YAML.

## Evaluation Metrics

The workflow uses two key metrics to decide whether to accept or reject a generated signal:

| Metric | Description | Acceptance Criteria |
|--------|-------------|---------------------|
| **Mean IC** | Average Spearman rank correlation between signal values and forward returns, computed across all time periods | \|IC\| ≥ `ic_threshold` (default: 0.02) |
| **P-value** | Statistical significance of the mean IC being different from zero | ≤ `p_value_threshold` (default: 0.05) |

A signal is accepted when both criteria are met. Otherwise, the Eval Agent generates optimization suggestions and the workflow retries.

## Workflow Result Format

Each run returns a structured JSON result containing the outcome, metrics, the signals that were tried, and (if the loop did not accept on the first iteration) the optimization advice produced for the next attempt. Example:

```json
{
  "status": "best_effort",
  "headline": "Best-effort result (IC threshold not met)",
  "request": "momentum signals",
  "iteration": 2,
  "total_iterations": 3,
  "selected_signal": "signal_volume_decayed_momentum",
  "thresholds": {
    "ic_threshold": 0.02,
    "p_value_threshold": 0.05
  },
  "metrics": {
    "mean_ic": 0.0103,
    "ic_std": 0.21,
    "ic_ir": 0.049,
    "t_stat": 2.91,
    "p_value": 0.0036,
    "num_periods": 3494,
    "positive_ic_ratio": 0.529
  },
  "signals": [
    {
      "name": "Volume-Decayed Momentum",
      "formula": "Mul(TS_Return(Close, 20), Decay_Linear(Volume, 20))",
      "category": "momentum",
      "data_fields_used": ["Close", "Volume"],
      "lookback_periods": [20]
    }
  ],
  "saved_path": "src/signal_discovery_workflow/output/signal_xxx.json",
  "last_feedback": "- Try TS_Std instead of TS_Var for cleaner volatility signal\n- Use 60-day lookback instead of 20\n- Replace Volume with Close*Volume for dollar-volume weighting"
}
```

| Field | Description |
|-------|-------------|
| `status` | `"accepted"`, `"best_effort"`, or `"failed"` |
| `headline` | Human-readable summary |
| `iteration` / `total_iterations` | Which iteration produced this result, out of the max allowed |
| `selected_signal` | Python function name of the signal whose IC was reported (the evaluator picks the best when multiple signals are returned) |
| `metrics` | All non-null IC statistics |
| `signals` | Compact summary of every signal that was generated |
| `saved_path` | Where the full signal JSON + code was persisted |
| `last_feedback` | Optimization advice from the last failed iteration; pass it back to resume the loop |

### Resuming an Optimization Loop

The workflow input accepts either a plain string or a JSON object that bundles `seed_feedback` from a prior run. Pass the `last_feedback` field from a prior result to start a new loop with the previous advice already applied — useful when you want more iterations than `max_iterations` allows, or want to switch models mid-run.

The shell snippet below uses `jq` to read `last_feedback` from the prior result and pack it into the JSON input shape (install with `brew install jq` on macOS or `apt-get install jq` on Debian/Ubuntu):

```bash
# First run - best effort, did not converge
uv run nat run --config_file configs/config-optimization.yml --input "momentum signals" > result1.json

# Extract last_feedback and resume
SEED=$(jq -r '.last_feedback' result1.json)
uv run nat run --config_file configs/config-optimization.yml \
  --input "$(jq -nc --arg req 'momentum signals' --arg seed "$SEED" \
              '{request: $req, seed_feedback: $seed}')"
```

Or programmatically, passing the same JSON shape as the workflow's input string:

```python
import json

result1 = json.loads(await runner.ainvoke("momentum signals"))

resume_input = json.dumps({
    "request": "momentum signals",
    "seed_feedback": result1["last_feedback"],
})
result2 = json.loads(await runner.ainvoke(resume_input))
```

The `last_feedback` field can be persisted to disk and re-loaded later — there's no in-memory state required to resume.

## Development

Tests cover the agent helpers, prompt builders, JSON parsing, module assembly, and end-to-end execution:

```bash
uv sync --locked --extra test
uv run --locked pytest tests/
```

The repo also ships a GitHub Actions workflow (`.github/workflows/ci.yml`) with three jobs on every pull request to `main`: `lint` (`ruff check src tests scripts`), `test` (pytest on Python 3.12), and `verify` (six hermetic verification scripts, each run as its own step). The scripts in `scripts/` are offline and hermetic: they require no network access or credentials, do not depend on local data artifacts or `dashboard/paper.json`, and import no dashboard runtime dependency. `scripts/verify_dashboard_static.py` statically analyzes `dashboard/index.html` for HTML sinks and template interpolation sites and fails closed.

## Project Structure

```
quantitative-signal-discovery-agent/
├── .github/workflows/ci.yml             # PR-level CI: ruff lint + pytest + hermetic verification
├── brev/
│   └── launchable-setup.ipynb           # One-shot environment setup for Brev launchables
├── configs/
│   ├── config-optimization.yml          # Workflow + LLM config with Phoenix tracing
│   └── config-optimization-no-telemetry.yml # Brev/cloud config, no Phoenix tracing
├── notebooks/
│   ├── signal-discovery-workflow.ipynb  # Interactive walkthrough
│   └── images/workflow-architecture.png
├── pyproject.toml                       # Dependencies, ruff/pytest config, NAT entry point
├── scripts/                             # Offline/hermetic verification scripts
│   ├── verify_sandbox.py                # Sandbox verification
│   ├── verify_quant_metrics.py          # Quantitative metric verification
│   ├── verify_operator_cache.py         # Operator cache verification
│   ├── verify_skeptic_gate.py           # Skeptic gate verification
│   ├── verify_dashboard_static.py       # Fail-closed static dashboard analysis
│   └── verify_data_sources.py           # Data-source verification
├── uv.lock                              # Pinned dependency resolution
├── README.md
├── src/signal_discovery_workflow/
│   ├── __init__.py
│   ├── register.py                                 # NAT function registration
│   ├── signal_generator.py                         # Signal agent: generates JSON signal descriptions
│   ├── signal_code_generator.py                    # Code agent: turns JSON into executable Python
│   ├── signal_evaluator.py                         # Eval agent: runs signal code, computes Rank IC
│   ├── signal_discovery_optimization_workflow.py   # Orchestrator (closed-loop generate/code/eval/feedback)
│   ├── llm_utils.py                                # Shared LLM-output helpers (parse, sanitize, normalize)
│   ├── download_data.py                            # Fetches S&P 500 data via yfinance
│   ├── data/sp500/                                 # OHLCV CSVs (gitignored)
│   ├── output/                                     # Saved signal results (gitignored)
│   └── template/
│       ├── calculator.json                         # Operator catalogue (name, signature, code)
│       └── signal_output_template.json             # JSON schema the signal agent fills in
└── tests/                                          # pytest suite
```

## Extended Capabilities

The workflow now includes optional performance, safety, research, and observability features. Optional acceleration and validation switches are off by default; `save_evolution` is enabled in the shipped configurations.

| Capability | Implemented behavior |
|------------|----------------------|
| Multi-asset data | Registers S&P 500, Nasdaq-100, sector ETFs, major forex, spot crypto, Binance perpetual futures, fundamentals, and sentiment presets. The generic downloader writes evaluator-compatible field CSVs plus a manifest. |
| Portfolio analytics | Builds a date-by-stock, equal-weight long/short portfolio from each signal and adds Sharpe, Sortino, drawdown, CAGR, volatility, and turnover metrics to the evaluation result. |
| Out-of-sample validation | Splits aligned dates chronologically into in-sample and out-of-sample sets, reports both IC results, and can prevent acceptance when the result is classified as overfit. |
| Sign-aware directional IC | Computes the direct and inverse rank IC, splits aligned dates chronologically, decides the orientation on the in-sample split only, and applies that already-fixed sign to the out-of-sample split. Because `rho(-x, y) == -rho(x, y)` exactly, the inverse mean IC is the negated direct mean IC and the absolute IC is unchanged. The feature catches a statistically significant signal oriented the wrong way and asks the advisor to invert it; it requests no inversion when the in-sample `|IC|` or p-value does not meet the configured thresholds. Insufficient in-sample or out-of-sample data is reported through dedicated `decision_reason` and `oos_reason` fields. |
| Parallel generation | Issues one hypothesis-generation LLM call per requested signal, bounded by a concurrency semaphore, and combines the results in request order. |
| Genetic crossover | Selects high-`|IC|` evaluated parents, asks the signal LLM for one combined hypothesis, and carries that candidate into the next iteration. |
| Crossover fallback | When the current iteration yields fewer than two eligible parents, lets crossover draw from earlier iterations of the same run only, never across runs, and logs the number of parents and the source iterations. If fewer than two eligible parents exist across the whole run, it skips gracefully with an explicit warning. |
| Skeptic gate | Reviews candidates before backtesting. Deterministic lookahead, leakage, degenerate-logic, and unknown-operator findings can veto a candidate; an optional LLM can also issue a `reject` verdict. Case-insensitive operator matching yields an advisory `operator_case:` for case-only mismatches, while genuinely unknown operators still veto. |
| Evolution tracking | Assigns a run ID and records the request, iteration, parent signal, and lineage source in every saved iteration so the dashboard can build a run tree. |
| Sandboxed execution | Optionally validates generated Python with AST rules and executes it in a restricted namespace. A subprocess API with CPU, memory, and wall-clock limits is also available. |
| GPU Rank IC | Optionally delegates rank-IC computation to CuPy or PyTorch, with automatic CPU fallback when no usable accelerator is present. On the repository's real S&P 500 grid, the verified CuPy path matched CPU mean IC within `8.7e-19` and measured about `15x` faster for one full-grid signal and about `50x` faster for eight; see [Measured Performance](#measured-performance). |
| Operator memoization | Optionally reuses identical operator results across candidate modules with content- and implementation-aware cache keys. |

Measured with the production `extract_json_array` extraction path over the 9,374 saved artifacts (18,665 candidate occurrences with a formula), the reconstructed pre-change gate blocked 1,223 candidates (6.55%); the current gate blocks 930 (4.98%), recovering 293. The change reclassified 334 case-only operator findings (across 321 candidate occurrences) from blocking to advisory — 28 of those candidates stay blocked for another reason — while all 956 genuine unknown-operator findings (across 928 candidate occurrences) remain blocking, with none lost. The rate is per candidate occurrence across repeated optimization runs, not per distinct signal concept: deduplicating by name (3,250 distinct names) would conflate 245 names that received different verdicts in different iterations. Case mismatches resolve to the candidate's self-contained helper (correct result) or raise an explicit backtest-time `NameError`; they cannot silently produce a wrong metric.

## Measured Performance

### GPU Rank IC on the S&P 500 Grid

The real CUDA path was exercised on an NVIDIA GeForce RTX 2050 with CUDA 12.9 and CuPy 14.2.0 (`cupy-cuda12x[ctk]`). CuPy needs the CUDA toolkit headers to compile its runtime kernels: install the matching extra (`cupy-cuda12x[ctk]`) or point `CUDA_PATH` at a local toolkit. Without the headers, the device calls raise at kernel-compile time and the module falls back to CPU with a warning; results stay correct, but the acceleration is lost.

On the repository's real S&P 500 grid (3,519 dates × 381 tickers, NaN-only inputs), `backend="cupy"` produced `|GPU mean_ic - CPU mean_ic| = 8.7e-19` (relative `2.2e-16`). The `ic_std`, `ic_ir`, `t_stat`, `p_value`, `num_periods`, and `positive_ic_ratio` metrics agreed to the same precision.

| Workload | GPU | CPU | Speedup |
|----------|-----|-----|---------|
| One signal over the full grid | 0.21 s | 3.21 s | ~15x |
| Eight full-grid signals | 0.64 s | 32.0 s | ~50x |

In this verification, `compute_rank_ic_batch` returned results identical to the per-signal calls, but it is not systematically faster. It pads every signal to the full returns grid (missing cells become masked NaN), while a per-signal call processes only the intersection of dates and tickers. For eight signals that already covered the whole grid, the batch call measured 0.89x of the per-signal loop, so treat it as a convenience API, not a speed-up. The device path masks `±inf`; the CPU path retains `±inf` through `dropna()` semantics.

### A/B Benchmark: Full Optional Bundle (N=5 per arm)

This benchmark executed five control runs and five treatment runs against live NVIDIA NIM using `nvidia/nemotron-3.5-lightning-30b-a3b`. Both arms used the identical request (`momentum signals`), the same model, and the same thresholds (`ic_threshold: 0.02`, `p_value_threshold: 0.05`); only the optional switches differed. GPU acceleration was off in both arms and was not a variable.

The control arm turned off the benchmarked optional capabilities: parallel generation, genetic crossover, skeptic review, OOS validation, directional IC, crossover fallback, and operator cache. The treatment arm turned on all of them.

All 10 runs exited with status 0 and produced evaluable signals. All 10 ended as `best_effort`: the best `|IC|` never reached the 0.02 acceptance threshold, although every run passed the p-value gate. The workflow ranks candidates by `|IC|` and does not invert the signal sign, so a statistically significant negative IC still counts as a strong candidate under that ranking.

**Efficiency.** Treatment was slower in every paired repetition. Control runtimes ranged from 31.5 to 48.1 s (mean 39.3 s), and treatment runtimes ranged from 77.3 to 109.4 s (mean 89.2 s); even the slowest treatment run was slower than the fastest control run. The mean paired difference was **+49.9 s (+127%)** (paired t = 9.82, p = 0.0006, Cohen's dz = 4.39); an unpaired Welch check gives p = 0.00025. The minimum detectable difference at 80% power is about 19–21 s, so the observed slowdown is much larger than the measured noise.

**Signal quality.** Treatment's best `|IC|` was on average about 10% lower. However, the paired 95% CI was [-0.0046, +0.0020] and includes zero, and the observed difference (0.0013) is below the approximately 0.0045 difference detectable at 80% power. The quality result is therefore statistically indistinguishable at N=5; it is not evidence of degradation.

**Directional IC.** All 10 treatment iteration artifacts contained a `directional_ic` payload. The in-sample orientation was negative in 10/10 (in-sample IC between -0.008 and -0.014, in-sample p < 0.05), but the magnitude never met the configured 0.02 significance threshold. The feature therefore correctly requested no inversion, and no directional-IC advisor advice was emitted. Control artifacts contained no `directional_ic` payload at all; the legacy key set is unchanged when the flag is off.

**Crossover and skeptic activity.** The treatment runs produced nine crossover candidates, of which only four of nine reached a subsequent evaluated iteration. Four candidates were produced after the final iteration, and one crossover response was unparseable and skipped. The new fallback branch was never exercised: no run had fewer than two eligible parents. One skeptic veto occurred in one of the five treatment runs.

**Convergence.** Control converged to one identical formula (`Div(TS_Return(Close, 10), Decay_Linear(Volume, 10))`) in 2 of 5 runs, with identical mean IC to all printed digits. Treatment produced five distinct selected formulas in 5 of 5 runs. The benchmark therefore partly measures where the model converges, not 10 independent quality draws.

#### Benchmark Limitations

- N=5 per arm provides low statistical power.
- The two arms ran sequentially rather than randomized/interleaved, so arm is confounded with time and possible live-API drift.
- The treatment changes six capabilities at once, so no individual feature can be attributed.
- The directional-IC inversion path and crossover fallback path were not productively exercised in this benchmark.

#### Conclusion

On this evidence, the full optional bundle roughly doubles latency without a demonstrated quality or acceptance benefit, so it should **not** be enabled by default. The features remain off by default and independently configurable. Enable the features individually and run a larger, randomized, factorial/ablation benchmark before adopting the bundle.

## Advanced Configuration

### Workflow Feature Flags

The following values are present in both [`configs/config-optimization.yml`](configs/config-optimization.yml) and [`configs/config-optimization-no-telemetry.yml`](configs/config-optimization-no-telemetry.yml):

| Parameter | Value | Description |
|-----------|-------|-------------|
| `parallel_generation` | `false` | Generate hypothesis LLM calls concurrently when more than one signal is requested. |
| `max_concurrency` | `4` | Maximum simultaneous hypothesis LLM calls when `parallel_generation` is enabled. |
| `genetic_crossover` | `false` | Create one crossover candidate from the two best evaluated parents and carry it into the next iteration. |
| `genetic_parents` | `2` | Number of top-`|IC|` parents considered; the first two are combined and any additional parents are supplied as context. |
| `crossover_fallback_enabled` | `false` | Allow crossover to draw eligible parents from earlier iterations of the same run when the current iteration has fewer than two. |
| `skeptic_review` | `false` | Run the pre-backtest candidate review. |
| `skeptic_llm` | `null` | Optional LLM block for the review; `null` runs deterministic static checks only. |
| `directional_ic_enabled` | `false` | Compute sign-aware directional IC and request a sign inversion only for a statistically significant negative in-sample orientation. |
| `oos_validation` | `false` | Add the chronological in-sample/out-of-sample IC check. |
| `oos_train_ratio` | `0.7` | Fraction of aligned dates assigned to the in-sample split. |
| `oos_reject_overfit` | `false` | Prevent a candidate classified as overfit from satisfying the acceptance gate. |
| `save_evolution` | `true` | Persist every iteration with `run_id`, `lineage`, and request metadata. |

### Runtime Evaluation Switches

The boolean and backend switches are read at evaluation time. The cache-size override is read when the shared operator cache is initialized:

| Variable | Default | Description |
|----------|---------|-------------|
| `SIGNAL_SANDBOX_ENABLED` | `false` | Enables the evaluator's restricted in-process sandbox path. `1`, `true`, `yes`, and `on` are treated as enabled. |
| `SIGNAL_USE_GPU` | `false` | Enables GPU delegation for Rank IC computation. |
| `SIGNAL_GPU_BACKEND` | `auto` | Selects `auto`, `cupy`, `torch`, or `cpu`; an unusable explicit GPU backend falls back to CPU. |
| `SIGNAL_USE_OPERATOR_CACHE` | `false` | Enables memoization of generated operator calls. |
| `SIGNAL_OPERATOR_CACHE_MAX_ENTRIES` | `512` | Maximum number of entries in the shared LRU operator cache when it is initialized; invalid or non-positive values fall back to the default. |

### Dataset Presets

| Preset | Source | Default universe | Fields |
|--------|--------|------------------|--------|
| `sp500` | yfinance | 392 static symbols | Open, High, Low, Close, Volume |
| `nasdaq100` | yfinance | 99 static symbols | Open, High, Low, Close, Volume |
| `sectors` | yfinance | 11 US sector SPDR ETFs | Open, High, Low, Close, Volume |
| `forex_major` | yfinance | 7 major USD pairs | Open, High, Low, Close, Volume |
| `crypto_spot` | yfinance | 6 USD-quoted spot pairs | Open, High, Low, Close, Volume |
| `crypto_futures` | Binance | 7 registry examples; dedicated CLI defaults to the top 60 | Open, High, Low, Close, Volume |
| `fundamentals` | yfinance | 10 symbols | Seven current point-in-time snapshot fields |
| `sentiment` | Injected provider | 5 symbols | `news_sentiment_score`, `article_count` |

Each OHLCV file uses dates as rows and symbols as columns. Datasets handled by the generic downloader also receive a `manifest.json` with the dataset, symbols, source, fetch time, and file dimensions.

## Portfolio Metrics and Out-of-Sample Validation

### Portfolio Construction

For each date, the workflow selects the top 20% and bottom 20% of valid signal values. Each bucket is equal-weighted; the long bucket totals `+0.5`, the short bucket totals `-0.5`, gross exposure is `1.0`, and net exposure is `0.0`. If the two buckets cannot be filled with distinct names, the date remains uninvested. A portfolio return is valid only when every held position has a finite forward return.

### Metric Formulas

The workflow uses `periods_per_year = 252`. In the formulas below, `r` is a valid portfolio return, `N` is the number of valid returns, `w` is a portfolio weight row, and `rf` is the annual risk-free rate.

| Metric | Formula | Convention |
|--------|---------|------------|
| `annual_return` | `mean(r) × periods_per_year` | Arithmetic annualization. |
| `cagr` | `prod(1 + r) ** (periods_per_year / N) - 1` | Geometric annualization; zero when final equity is not positive. |
| `annual_volatility` | `std(r, ddof=1) × sqrt(periods_per_year)` | Sample standard deviation. |
| `sharpe_ratio` | `(mean(r) - rf / periods_per_year) / std(r, ddof=1) × sqrt(periods_per_year)` | Uses sample standard deviation (`ddof=1`). |
| `sortino_ratio` | `(mean(r) - rf / periods_per_year) / sqrt(mean(min(0, r - rf / periods_per_year) ** 2)) × sqrt(periods_per_year)` | Downside deviation uses the full sample. |
| `max_drawdown` | `min(equity / running_max(equity) - 1)` | Initial equity `1.0` is included in the running maximum. |
| `calmar_ratio` | `annual_return / abs(max_drawdown)` | Uses arithmetic annual return. |
| `total_return` | `prod(1 + r) - 1` | Compounded period return. |
| `turnover` | `mean(sum(abs(w[t] - w[t-1]))) × periods_per_year` | Full absolute-weight-change, two-way convention. |
| `best_day`, `worst_day` | `max(r)`, `min(r)` | Raw, unannualized period returns. |

Missing and non-finite returns are excluded. Undefined ratios and sample-risk statistics are reported as `0.0`; a series is treated as riskless when its dispersion is at or below `max(1e-12, abs(mean(r)) × 1e-9)`. Portfolio metrics are merged additively into the Rank IC result and do not replace its `num_periods`, which remains the number of usable IC dates.

### Out-of-Sample Rules

The validation is one chronological split, not walk-forward cross-validation: aligned dates are sorted and divided at `floor(len(dates) × oos_train_ratio)`. A candidate is classified as overfit when any of these conditions holds:

- In-sample and out-of-sample mean IC have opposite signs.
- `abs(test_mean_ic) < 0.5 × abs(train_mean_ic)`.
- Either split has fewer than 30 usable IC dates.
- Either split is missing an IC value.

With `oos_reject_overfit: true`, a flagged candidate cannot satisfy the acceptance gate. The workflow continues with targeted feedback and can still return that candidate as a `best_effort` result.

## Quick Start: Multi-Asset Data

List the registered presets without contacting a data provider:

```bash
.venv/bin/python -m signal_discovery_workflow.download_market_data --all-presets
```

Inspect a download plan without fetching data:

```bash
.venv/bin/python -m signal_discovery_workflow.download_market_data --dataset crypto_spot --dry-run
```

Download a preset into the evaluator-compatible layout:

```bash
.venv/bin/python -m signal_discovery_workflow.download_market_data \
  --dataset sectors \
  --start 2015-01-01 \
  --end 2025-12-31
```

Downloads default to `src/signal_discovery_workflow/data/<dataset>/`. The workflow evaluator currently reads `src/signal_discovery_workflow/data/sp500/`, so place the selected `Open.csv`, `High.csv`, `Low.csv`, `Close.csv`, and `Volume.csv` files in that directory before running the evaluator. Binance perpetual futures use the dedicated downloader, which selects symbols by quote volume:

```bash
.venv/bin/python -m signal_discovery_workflow.download_binance_futures --symbols 60 --interval 1d
```

## Quick Start: Dashboard

Start the local dashboard:

```bash
.venv/bin/python -m uvicorn dashboard.app:app --host 0.0.0.0 --port 8090
```

Open http://localhost:8090. With `save_evolution: true`, saved files use the `signal_<run_id>_iter<n>.json` naming pattern under `src/signal_discovery_workflow/output/`. These new-format artifacts may include a top-level `skeptic_reviews` array of per-iteration reviews containing `name`, `verdict` (`approve`/`reject`), `issues`, `static_risks`, `vetoed`, and `reasons`; `save_evolution: false` keeps legacy artifacts byte-compatible without it. The evolution and metrics APIs can also be queried directly:

```bash
curl -s http://127.0.0.1:8090/api/evolution
curl -s http://127.0.0.1:8090/api/evolution/YOUR_RUN_ID
curl -s http://127.0.0.1:8090/api/metrics/summary
```

`GET /api/evolution` returns run summaries, lineage trees, and aggregate evolution statistics. `GET /api/evolution/{run_id}` adds the saved formulas, generated code, and optimization feedback for one run. `GET /api/metrics/summary` returns metric availability, minima, maxima, and overfit flags. The EVOLUTION tab shows `SKEPTIC · N rejected · M approved` per node and one reason line per rejected candidate. These three endpoints parse saved JSON only; they do not execute the saved signal code.

## Limitations and Caveats

- **Sandbox boundary:** the in-process sandbox is AST validation plus restricted execution, not a security boundary. Subprocess mode adds resource limits and is stronger, but it is still not an operating-system container. Use container isolation for hostile code.
- **Sandbox mode:** `SIGNAL_SANDBOX_ENABLED=1` selects the evaluator's in-process restricted mode. Subprocess isolation is available only through `execute_signal_code_sandboxed(..., allow_subprocess=True)`.
- **GPU verification:** The real CUDA path has been executed and verified on an NVIDIA GeForce RTX 2050 with CUDA 12.9 and CuPy 14.2.0 (`cupy-cuda12x[ctk]`). CuPy needs the CUDA toolkit headers to compile its runtime kernels: install the matching extra (`cupy-cuda12x[ctk]`) or point `CUDA_PATH` at a local toolkit. Without the headers, the device calls raise at kernel-compile time and the module falls back to CPU with a warning; results stay correct, but the acceleration is lost. CuPy and PyTorch wheels remain optional, not default dependencies.
- **Non-finite GPU inputs:** the device path masks `±inf` while the CPU path retains it through `dropna()` semantics. Device and CPU results can therefore differ for `±inf`; NaN-only inputs match.
- **Sentiment provider:** sentiment requires an injected provider. The default fetcher makes no network request, returns empty frames, and writes no sentiment data file.
- **Static universes:** the Nasdaq-100 preset is a static, approximately 100-name list (99 symbols in the current source), not a live constituent index. Other preset universes are also snapshots rather than continuously refreshed memberships.
- **Cache economics:** benefit depends on operator cost and workload mix. With the fingerprint memo, operator-level caching measured approximately `4.8x-5.9x` faster; end-to-end through `signal_evaluator.execute_signal_code`, gains ranged from about `1.0x` for backtest-dominated workloads to roughly `9x` for operator-dominated workloads, depending on operator time versus downstream backtest time. The cache is off by default; there is no universal speedup.
- **Cache ownership:** the generic operator-cache API returns cached objects by reference unless `copy_on_return=True`; the evaluator integration uses shallow copies. Sandboxed execution runs in its own namespace and is outside the evaluator's operator cache.
- **OOS scope:** the OOS check is a single chronological split, with no walk-forward folds, purge/embargo window, or multiple-testing correction.
- **Costs and capacity:** portfolio metrics do not deduct transaction costs, slippage, borrow costs, financing, taxes, or market impact; turnover is a diagnostic rather than a modeled cost.
- **Annualization:** workflow portfolio metrics are hard-coded to 252 periods per year even though `forward_periods` is configurable. Adjust the integration if the portfolio return period is not one daily period.
- **Fundamental snapshots:** the default fundamentals fetcher reads current `Ticker.info` snapshots; the optional combine helper can broadcast them across the sample. It does not provide historically point-in-time fundamentals, so using them in a backtest can introduce lookahead bias.
- **Dashboard scope:** only the evolution and metrics-summary endpoints documented above are passive JSON readers. Other dashboard analysis endpoints may evaluate saved signal code.

## Additional Resources

- [NeMo Agent Toolkit Documentation](https://docs.nvidia.com/nemo-agent-toolkit/)
- [Arize Phoenix Documentation](https://arize.com/docs/phoenix)
- [NeMo Fine-tuning Guide](https://docs.nvidia.com/nemo-framework/user-guide/latest/sft_peft/index.html) — to specialize Nemotron on your signal history
- [Sandbox Security Model](SANDBOXING.md) — AST validation, restricted execution and subprocess isolation for LLM-generated signal code

## License

See [LICENSE.txt](LICENSE.txt) and [LICENSE-3rd-party.txt](LICENSE-3rd-party.txt) for details.
