# Sandboxing LLM-Generated Signal Code

This document describes the sandbox layer for executing untrusted,
LLM-generated signal code (`src/signal_discovery_workflow/sandbox.py`), its
threat model, what it does and does **not** protect against, and how to
integrate it with `signal_evaluator.execute_signal_code`.

> **Honest summary:** this sandbox raises the bar for accidental and casual
> malicious behavior. **In-process mode is not a security boundary.** Subprocess
> mode with rlimits is substantially stronger but still not equivalent to an OS
> container. Treat generated code as hostile regardless of the verdict of the
> static analyzer.

---

## 1. Threat model

LLM-generated signal code is **untrusted input**. Before this layer, generated
code was executed with a plain `exec(code, namespace)` in the host process,
which could:

| Attack | Effect |
| --- | --- |
| `import os` / `os.system(...)` | Arbitrary command execution with the host process's privileges |
| `import socket` | Outbound network access, data exfiltration |
| `open()` / `pathlib` / `shutil` | Read secrets, delete or corrupt data files |
| `eval` / `exec` / `compile` / `__import__` | Dynamic execution and dynamic imports (escape primitives) |
| `().__class__.__bases__[0].__subclasses__()` | Classic Python sandbox escape via dunder traversal |
| `while True:` / `range(10**9)` | Hang the host process forever (DoS) |
| Huge `np.ones((10**6, 10**6))`-style allocations | Memory exhaustion of the host (DoS) |

Adversary model: the *generator* (an LLM prompted directly or via injected
content) may produce arbitrary Python. The code is untrusted even when the
prompt is benign (prompt injection via retrieved market news, poisoned
context, or plain hallucination).

## 2. Layers provided

### Layer 1 — Static AST analysis (`classify_code_violations`)

Parses the source and reports findings of the form
`{"type": str, "detail": str, "line": str}`. Detects:

- **Disallowed imports** — anything outside `ALLOWED_IMPORTS`
  (`pandas`, `numpy`, `math`, `statistics`), including `from os import ...`,
  relative imports, and star imports.
- **Forbidden builtin calls** — `exec`, `eval`, `compile`, `open`,
  `__import__`, `input`, `globals`, `locals`, `vars`, `getattr`, `setattr`,
  `delattr`, `breakpoint` (also when reached as attributes, e.g. `df.eval`).
- **Dunder attribute access** — `__class__`, `__subclasses__`, `__globals__`,
  `__builtins__`, ... and bare dunder names (except `__name__`).
- **Private attribute access** — any attribute starting with `_`.
- **Dangerous module calls/attributes** — call or attribute chains rooted in
  `os`, `subprocess`, `sys`, `socket`, `shutil`, `pathlib`, `ctypes`,
  `importlib`, including through aliases (`import os as o`) and `from os import
  system`, and even without a local import (`os.system` alone is flagged
  because the sandbox never binds such names).
- **Risky scientific-stack access** — `pd.io`, `pd.core`, `np.ctypeslib`,
  `np.testing`, etc., and `from pandas import io`.
- **Resource-exhaustion loops** — `while True` without a reachable `break`
  (breaks inside nested loops/scopes correctly attributed), and loops with
  huge literal bounds (`range(10**9)`, `while i < 10**9`, comprehensions) via
  constant folding; threshold `HUGE_LOOP_BOUND = 10**8`.
- **`str.format` / `format_map` calls** — format strings can traverse object
  attributes (`"{0.__class__}"`), bypassing AST-level dunder checks.
- **Syntax errors / null bytes / non-string input** — reported as findings so
  validation fails closed.

**Frame/introspection attributes (audit finding SB-01).** Attribute access to
interpreter frame/introspection members — `gi_frame`, `gi_code`, `f_back`,
`f_globals`, `f_locals`, `f_builtins`, `f_trace`, `tb_frame`, `tb_next`,
`cr_frame`, `cr_code`, `ag_frame`, `ag_code` — is statically rejected with
violation type `frame_introspection`, closing the frame-walking escape surface
(e.g. `gen.gi_frame.f_back.f_globals`). The dunder escape primitives
(`__code__`, `__globals__`, `__mro__`, `__subclasses__`, `__bases__`,
`__reduce__`, `__getattribute__`, ...) remain rejected under the existing
dunder-attribute rule. This is defence in depth only: in-process mode remains
a robustness guard, NOT a security boundary (see Residual Risks below).

### Layer 2 — Fail-closed validation (`validate_signal_code`)

Raises `SandboxViolation` (subclass of `SandboxError`) with a readable,
aggregated message when any finding exists, or when the source exceeds
`MAX_CODE_LENGTH` (200,000 chars) or contains null bytes. Returns `None` for
clean code. Both execution modes validate **before** anything runs; the
subprocess child **re-validates** independently (defense in depth).

### Layer 3 — Restricted execution

#### In-process restricted mode (default)

After validation, the code is executed with:

- **No `__import__`, `open`, `eval`, `exec` builtins.** Allowlisted imports are
  parsed out of the AST and re-bound by the host to *restricted module
  proxies*; the import statements themselves become no-ops. Disallowed imports
  (which validation already rejects) would fail at runtime with `ImportError`.
- **Explicit builtin allowlist** (`_SAFE_BUILTIN_NAMES` + exception classes):
  `len`, `range`, `min`, `max`, `sum`, `abs`, `round`, `sorted`, `enumerate`,
  `zip`, `list`, `dict`, `tuple`, `set`, `bool`, `float`, `int`, `str`,
  `print` (redirected to an in-memory buffer), `isinstance`, `hasattr`,
  and `getattr` restricted to non-underscore names, plus a few safe extras
  (`map`, `filter`, `any`, `all`, ...) and common exception types.
- **Restricted module proxies** for `pandas`/`numpy`/`math`/`statistics`:
  attribute access starting with `_` and denylisted public submodules
  (`pd.io`, `np.ctypeslib`, ...) raises `SandboxViolation` at runtime.
- **Safely loaded operator catalogue**: `template/calculator.json` snippets are
  themselves statically validated and executed under the same restricted
  builtins (unlike the evaluator's plain `exec`); operators that fail are
  skipped, never silently executed with full privileges.
- `print` output is captured (capped at 100 lines × 200 chars), never written
  to the host's stdout.

#### Subprocess isolation mode (`allow_subprocess=True`)

The code runs in a separate `sys.executable` child process:

- The child applies `resource.setrlimit` **before** importing anything:
  `RLIMIT_CPU = timeout + 2 s`, `RLIMIT_AS = max_memory_mb × 1 MiB`
  (skipped when `max_memory_mb=None`), `RLIMIT_NOFILE` capped at 256.
- The parent enforces a wall-clock timeout (`timeout_seconds + 1 s` grace) and
  kills the entire child process group on expiry, returning `None`.
- The child re-validates the code, executes it under the same restricted
  namespace, and pickles the result `(DataFrame, name)` to a file created with
  `0600` permissions inside a `0700` temp directory; the parent reads it back
  and the directory is removed in a `finally` block.
- On platforms without `resource` (Windows), the call degrades to in-process
  mode with a logged warning.
- Temporary files are created with `O_EXCL`/`0600`; the full code is never
  logged (only violation types, line numbers, and <200-char snippets).

## 3. What this sandbox CANNOT block (residual risks)

Be explicit about these when reasoning about generated code:

1. **In-process mode is not a security boundary.** With no `__import__` and no
   dunder access in the *source*, casual escapes fail — but determined escapes
   exist that static analysis and the restricted namespace cannot see, e.g.:
   - format-string attribute traversal if `format` usage is added back,
   - C-level attribute access hidden inside allowed callables,
   - any future refactor that re-exposes `getattr`/dunder access.
   Treat in-process mode as robustness hardening only.
2. **Resource exhaustion inside allowed operations.** `np.ones((50000, 50000))`,
   huge `pd.concat` chains, or pathological string growth are all legal
   syntax. In-process mode has **no** CPU, memory, or wall-clock control at
   all (Python cannot safely kill a thread); a hang or OOM in-process takes
   the host down with it. Subprocess mode limits this (RLIMIT_AS, RLIMIT_CPU,
   wall-clock kill) but is still not a hard guarantee against kernel-level
   pressure.
3. **Runtime-legal abuse of pandas/numpy APIs.** The proxies block private and
   known-risky attributes, but the public surface of pandas/numpy is enormous;
   anything reachable through public attributes is allowed. In-process mode
   also lets generated code mutate host-global state (e.g. `pd.set_option`),
   which persists after the call.
4. **DoS via large allocations** is only *narrowed* by the loop-bound checks;
   a single huge allocation is not detected statically.
5. **Pandas object injection.** `DataFrame.attrs`, object-dtype columns, and
   `eval`-style pandas features can smuggle behavior; object dtype columns in
   results flow back into the host process unpickled.
6. **Pickle trust boundary (subprocess mode).** The parent unpickles the
   result file written by the child. The child is our own trusted runner, but
   if the runner or interpreter were compromised, unpickling its output would
   execute code in the parent. Results are also re-validated only
   structurally (tuple of DataFrame + str), not semantically.
7. **Side channels** (timing, memory pressure) and **host-level attacks**
   (e.g. exhausting disk via temp files within rlimits) are out of scope.
8. **Static analysis is syntactic.** Obfuscation that defeats the AST checks
   (e.g. via unicode homoglyphs inside identifiers, dynamic attribute names
   built at runtime through allowed primitives) is only caught if it happens
   to trip a runtime proxy check.
9. **No network/filesystem deny-by-default at the OS level.** The child
   process can still open sockets/files at the OS level; `RLIMIT_NOFILE`
   limits descriptor count, not destinations. Code that passes validation
   cannot *reference* `socket`/`open` (so it cannot easily reach them), but the
   process itself is not confined by the OS.

## 4. Recommended production deployment

- **Always use `allow_subprocess=True`** for untrusted code, with an explicit
  `timeout_seconds` and `max_memory_mb`. In-process mode is acceptable only
  for first-party, reviewed code.
- Note `max_memory_mb=2048` is close to the import footprint of
  pandas+numpy+scipy (≈1.4 GB peak VSZ measured); raise it rather than
  disabling it if imports fail.
- **Best: run the whole evaluator (or just the sandbox child) inside a
  locked-down container**: no network (`--network none`), read-only rootfs,
  non-root user, dropped capabilities, seccomp/AppArmor profile, CPU and
  memory cgroup limits, and a dedicated tmpfs for exchange files. rlimits
  complement — but do not replace — container isolation. Consider gVisor/
  nsjail/bubblewrap for stronger kernel isolation.
- Keep generated code, data, and credentials in separate trust zones: the
  sandbox child should never have API keys in its environment.
- Log and alert on `SandboxViolation` events; they indicate the generator (or
  its context) is producing hostile output.

## 5. Integrating with the evaluator

`signal_evaluator.execute_signal_code` currently does `exec(code, namespace)`
and is owned by another engineer; the sandbox is a drop-in front door that
does **not** modify it:

```python
from signal_discovery_workflow.sandbox import (
    execute_signal_code_sandboxed,
    sandbox_report,
    validate_signal_code,
)

def evaluate_generated_code(code: str, stock_data, selection_periods: int = 5):
    # 1. Static gate (optional direct call; the executor validates anyway)
    validate_signal_code(code)          # raises SandboxViolation -> reject

    # 2. Skeptic summary for the review agent
    report = sandbox_report(code)       # {"safe": ..., "estimated_risk": ...}
    if not report["safe"]:
        raise ValueError(f"Rejected: {report['recommendation']}")

    # 3. Execute with isolation (preferred in production)
    return execute_signal_code_sandboxed(
        code,
        stock_data,
        selection_periods=selection_periods,
        timeout_seconds=60.0,
        max_memory_mb=4096,
        allow_subprocess=True,
    )
```

The sandbox returns the same contract as the evaluator —
`(signal_values_df, selected_signal_name)` for the highest-|IC| signal
function, or `None` on any failure — and mirrors the evaluator's
argument-resolution and helper-detection semantics so results match. A parity
test (`test_contract_matches_evaluator_frame`) pins this.

## 6. Configuration knobs

| Knob | Default | Meaning |
| --- | --- | --- |
| `timeout_seconds` | `30.0` | Wall-clock budget (subprocess mode: parent kill + RLIMIT_CPU headroom). Not enforceable in-process. |
| `max_memory_mb` | `2048` | `RLIMIT_AS` for the child; `None` disables the memory limit. Ignored in-process. |
| `allow_subprocess` | `False` | Use subprocess isolation. Auto-degrades to in-process where `resource` is unavailable. |
| `MAX_CODE_LENGTH` | `200_000` | Max accepted source characters (module constant). |
| `HUGE_LOOP_BOUND` | `10**8` | Literal loop bounds above this are flagged statically. |
| `ALLOWED_IMPORTS` | `pandas, numpy, math, statistics` | The only importable top-level modules. |

## 7. Test coverage

`tests/test_sandbox.py` covers: required malicious-pattern detections, clean
legitimate signal code, all 66 operator-catalogue snippets staying clean,
fail-closed validation (length, null bytes, syntax), in-process execution
contract (DataFrame + highest-|IC| name, parity with the evaluator), the
`os.system` file-creation attack (no file appears), runtime proxy fail-closed
behavior, subprocess round-trip parity, the infinite-loop wall-clock kill
(finishes well under `timeout + 5s`), and `sandbox_report` verdicts.

Run with:

```bash
.venv/bin/python -m pytest tests/test_sandbox.py -q
```
