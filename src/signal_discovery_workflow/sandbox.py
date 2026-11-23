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
Sandbox layer for executing untrusted, LLM-generated signal code.

LLM-generated signal code is UNTRUSTED INPUT. This module provides a
defense-in-depth layer that can be placed in front of
``signal_evaluator.execute_signal_code``:

1. ``classify_code_violations`` — static AST analysis of the source.
2. ``validate_signal_code``       — fail-closed gate (raises ``SandboxViolation``).
3. ``execute_signal_code_sandboxed`` — executes the code under one of two modes:

   * **In-process restricted mode** (default): the code runs with a minimal
     builtin allowlist (no ``__import__``, ``open``, ``eval``, ``exec``), only
     allowlisted modules bound as restricted proxies, and the operator
     catalogue from ``template/calculator.json`` loaded safely.
   * **Subprocess isolation mode** (``allow_subprocess=True``): the code runs
     in a separate ``python`` child process guarded by ``resource.setrlimit``
     (RLIMIT_CPU, RLIMIT_AS, RLIMIT_NOFILE) and a parent-enforced wall-clock
     timeout. This is the only mode that approaches a real security boundary.

4. ``sandbox_report`` — a machine-readable summary for a "skeptic" reviewer.

.. warning::
   The in-process mode is a robustness guard, NOT a security boundary. See
   ``SANDBOXING.md`` for the honest list of residual risks. For
   untrusted code in production, always use subprocess mode (ideally inside
   a locked-down container).

Logging policy: the full generated code is NEVER logged (it may contain user
data). Only violation types, line numbers, and snippets truncated to <200
characters are logged.
"""

import ast
import builtins as _builtins_mod
import importlib
import inspect
import json
import logging
import os
import pickle
import re
import shutil
import signal
import subprocess
import sys
import tempfile
import warnings
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd
from scipy import stats

logger = logging.getLogger(__name__)

try:  # pragma: no cover - trivial branch
    import resource

    RESOURCE_LIMITS_AVAILABLE = True
except ImportError:  # Windows
    resource = None  # type: ignore[assignment]
    RESOURCE_LIMITS_AVAILABLE = False

# ---------------------------------------------------------------------------
# Public constants
# ---------------------------------------------------------------------------

#: The only modules that LLM-generated signal code may import. Common aliases
#: (``pd`` for pandas, ``np`` for numpy) are supported.
ALLOWED_IMPORTS: frozenset[str] = frozenset({"pandas", "numpy", "math", "statistics"})

#: Maximum accepted source length, in characters.
MAX_CODE_LENGTH: int = 200_000

#: Loop bounds above this literal value are flagged as resource-exhaustion risks.
HUGE_LOOP_BOUND: int = 100_000_000

#: Modules whose very presence in generated code is treated as an escape attempt.
DANGEROUS_MODULE_ROOTS: frozenset[str] = frozenset(
    {"os", "subprocess", "sys", "socket", "shutil", "pathlib", "ctypes", "importlib"}
)

#: Names pre-seeded into every sandbox namespace: alias -> allowlisted module.
#: Sandbox-executed code can use these without importing, so the static
#: analyzer must treat them as already-bound names.
_SEEDED_ALIASES: tuple[tuple[str, str], ...] = (
    ("pd", "pandas"), ("pandas", "pandas"),
    ("np", "numpy"), ("numpy", "numpy"),
    ("math", "math"), ("statistics", "statistics"),
)

#: Builtins that must never appear as calls in untrusted code.
FORBIDDEN_BUILTINS: frozenset[str] = frozenset(
    {
        "exec", "eval", "compile", "open", "__import__", "input",
        "globals", "locals", "vars", "getattr", "setattr", "delattr", "breakpoint",
    }
)

#: Risky public submodules of the allowed scientific stack, blocked on the
#: restricted module proxies (e.g. ``np.ctypeslib``, ``pd.io``).
_MODULE_DENYLISTS: dict[str, frozenset[str]] = {
    "pandas": frozenset({"io", "core", "util", "compat", "testing"}),
    "numpy": frozenset({"ctypeslib", "testing", "f2py", "distutils"}),
}

#: Attribute names that expose interpreter frames and code objects (generators,
#: coroutines, async generators, tracebacks). Reaching ``f_globals`` /
#: ``f_builtins`` through a frame is a classic sandbox-escape primitive, so
#: static access is rejected with type ``frame_introspection`` regardless of
#: the object it is accessed on (SB-01 hardening).
FRAME_INTROSPECTION_ATTRIBUTES: frozenset[str] = frozenset(
    {
        "gi_frame", "gi_code",                      # generator
        "f_back", "f_globals", "f_locals", "f_builtins", "f_trace",  # frame
        "tb_frame", "tb_next",                      # traceback
        "cr_frame", "cr_code",                      # coroutine
        "ag_frame", "ag_code",                      # async generator
    }
)

#: Dunder members that expose type objects, MROs, subclasses or pickling /
#: descriptor hooks. All of these are already rejected by the blanket dunder
#: rule below; they are listed explicitly so the blocked security surface is
#: documented and survives future refactors of the rule ordering.
DUNDER_ESCAPE_ATTRIBUTES: frozenset[str] = frozenset(
    {
        "__code__", "__globals__", "__builtins__", "__mro__", "__subclasses__",
        "__bases__", "__reduce__", "__reduce_ex__", "__getattribute__",
        "__class__", "__subclasshook__", "__init_subclass__", "__set_name__",
        "__get__", "__set__", "__delete__", "__new__",
    }
)

# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------


class SandboxError(Exception):
    """Base class for sandbox failures (execution or configuration problems)."""


class SandboxViolation(SandboxError):
    """Raised when untrusted code attempts a disallowed capability."""


# ---------------------------------------------------------------------------
# Smart-quote normalization (mirrors signal_evaluator so both accept the same
# LLM output; single-char -> single-char translation preserves line numbers)
# ---------------------------------------------------------------------------

_SMART_UNICODE = str.maketrans(
    {
        "‘": "'", "’": "'",
        "“": '"', "”": '"',
        "–": "-", "—": "-",
        " ": " ",
    }
)

# ---------------------------------------------------------------------------
# Restricted module proxies
# ---------------------------------------------------------------------------


class _RestrictedModule:
    """Read-only proxy around an allowlisted module.

    Blocks attribute names starting with ``_`` and a denylist of risky public
    submodules (``pd.io``, ``np.ctypeslib``, ...). This is best-effort
    hardening for the in-process mode; it is not a security boundary.
    """

    __slots__ = ("_sb_target", "_sb_denied", "_sb_name")

    def __init__(self, target: Any, denied: frozenset[str] = frozenset()):
        object.__setattr__(self, "_sb_target", target)
        object.__setattr__(self, "_sb_denied", frozenset(denied))
        object.__setattr__(self, "_sb_name", getattr(target, "__name__", "module"))

    def __getattr__(self, name: str) -> Any:
        target = object.__getattribute__(self, "_sb_target")
        denied = object.__getattribute__(self, "_sb_denied")
        if name.startswith("_") or name in denied:
            raise SandboxViolation(
                f"access to '{object.__getattribute__(self, '_sb_name')}."
                f"{name}' is not allowed in the sandbox"
            )
        try:
            return getattr(target, name)
        except AttributeError:
            raise AttributeError(
                f"module '{object.__getattribute__(self, '_sb_name')}' has no attribute '{name}'"
            ) from None

    def __setattr__(self, name: str, value: Any) -> None:
        raise SandboxViolation("cannot set attributes on sandboxed modules")

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return f"<sandboxed module '{object.__getattribute__(self, '_sb_name')}'>"


_MODULE_PROXY_CACHE: dict[str, _RestrictedModule] = {}


def _module_proxy(root: str) -> _RestrictedModule:
    """Return (a cached) restricted proxy for an allowlisted top-level module."""
    proxy = _MODULE_PROXY_CACHE.get(root)
    if proxy is None:
        target = importlib.import_module(root)
        proxy = _RestrictedModule(target, _MODULE_DENYLISTS.get(root, frozenset()))
        _MODULE_PROXY_CACHE[root] = proxy
    return proxy


# ---------------------------------------------------------------------------
# Safe builtins namespace
# ---------------------------------------------------------------------------

_MISSING = object()

#: Allowlisted, side-effect-free builtins (superset of the required minimum).
_SAFE_BUILTIN_NAMES: tuple[str, ...] = (
    "len", "range", "min", "max", "sum", "abs", "round", "sorted",
    "enumerate", "zip", "list", "dict", "tuple", "set", "bool", "float",
    "int", "str", "isinstance", "issubclass", "hasattr", "hash", "repr",
    "all", "any", "divmod", "filter", "frozenset", "map", "iter", "next",
    "pow", "reversed",
)

#: Exception types sandboxed code is allowed to reference in except clauses.
_SAFE_EXCEPTION_NAMES: tuple[str, ...] = (
    "Exception", "ArithmeticError", "AssertionError", "AttributeError",
    "FloatingPointError", "IndexError", "KeyError", "LookupError",
    "MemoryError", "NameError", "NotImplementedError", "OverflowError",
    "RecursionError", "RuntimeError", "RuntimeWarning", "StopIteration",
    "TypeError", "UnicodeError", "ValueError", "Warning", "ZeroDivisionError",
)

_MAX_PRINT_LINES = 100
_MAX_PRINT_WIDTH = 200


def _restricted_getattr(obj: Any, name: str, default: Any = _MISSING) -> Any:
    """``getattr`` restricted to non-underscore attribute names."""
    if not isinstance(name, str) or name.startswith("_"):
        raise SandboxViolation("getattr on private/dunder attribute names is not allowed")
    try:
        return getattr(obj, name)
    except AttributeError:
        if default is _MISSING:
            raise
        return default


def _restricted_hasattr(obj: Any, name: str) -> bool:
    """``hasattr`` restricted to non-underscore attribute names."""
    try:
        _restricted_getattr(obj, name)
    except SandboxViolation:
        return False
    except AttributeError:
        return False
    return True


def _make_print(print_buffer: list[str]) -> Callable[..., None]:
    """Build a ``print`` replacement that captures output instead of writing
    to the host's stdout (the sandbox may run inside a server process)."""

    def _sandboxed_print(*args: Any, sep: str = " ", end: str = "\n", file: Any = None, flush: bool = False) -> None:
        if file is not None:
            raise SandboxViolation("print(file=...) is not allowed in sandboxed code")
        text = sep.join(str(a) for a in args)
        if len(print_buffer) < _MAX_PRINT_LINES:
            print_buffer.append(text[:_MAX_PRINT_WIDTH])
        return None

    return _sandboxed_print


def _build_safe_builtins(print_buffer: list[str]) -> dict[str, Any]:
    """Build the explicit builtins allowlist used as ``__builtins__``."""
    safe: dict[str, Any] = {}
    for name in _SAFE_BUILTIN_NAMES:
        safe[name] = getattr(_builtins_mod, name)
    for name in _SAFE_EXCEPTION_NAMES:
        safe[name] = getattr(_builtins_mod, name)
    safe["getattr"] = _restricted_getattr
    safe["hasattr"] = _restricted_hasattr
    safe["print"] = _make_print(print_buffer)
    # Required for `class` definitions in generated code; harmless without
    # dunder attribute access (which static validation rejects).
    safe["__build_class__"] = _builtins_mod.__build_class__
    return safe


# ---------------------------------------------------------------------------
# Static AST analysis
# ---------------------------------------------------------------------------


def _root_name(node: ast.AST) -> str | None:
    """Return the left-most identifier of a Name/Attribute chain, else None."""
    while isinstance(node, ast.Attribute):
        node = node.value
    if isinstance(node, ast.Name):
        return node.id
    return None


def _const_eval(node: ast.AST) -> float | int | None:
    """Best-effort constant folding of numeric literals and simple binops."""
    if isinstance(node, ast.Constant):
        value = node.value
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            return value
        return None
    if isinstance(node, ast.UnaryOp):
        inner = _const_eval(node.operand)
        if inner is None:
            return None
        if isinstance(node.op, ast.USub):
            return -inner
        if isinstance(node.op, ast.UAdd):
            return +inner
        return None
    if isinstance(node, ast.BinOp):
        left, right = _const_eval(node.left), _const_eval(node.right)
        if left is None or right is None:
            return None
        try:
            if isinstance(node.op, ast.Pow):
                # Guard against absurd exponents stalling the analyzer itself.
                if abs(right) > 1024 or abs(left) > 10 ** 8:
                    return None
                return left**right
            if isinstance(node.op, ast.Add):
                return left + right
            if isinstance(node.op, ast.Sub):
                return left - right
            if isinstance(node.op, ast.Mult):
                return left * right
            if isinstance(node.op, ast.Div):
                return left / right
            if isinstance(node.op, ast.FloorDiv):
                return left // right
            if isinstance(node.op, ast.Mod):
                return left % right
        except Exception:
            return None
    return None


def _loop_has_break(loop_node: ast.While | ast.For | ast.AsyncFor) -> bool:
    """Whether the loop body (or its else clause) contains a `break` that
    binds to *this* loop. Nested loops and nested scopes own their breaks."""
    stack: list[ast.AST] = [*loop_node.body, *loop_node.orelse]
    while stack:
        node = stack.pop()
        if isinstance(node, ast.Break):
            return True
        if isinstance(
            node, (ast.While, ast.For, ast.AsyncFor, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)
        ):
            continue
        stack.extend(ast.iter_child_nodes(node))
    return False


def _is_truthy_constant(node: ast.AST) -> bool:
    return isinstance(node, ast.Constant) and bool(node.value) is True


def _check_range_bound(call_node: ast.Call, report: Callable[..., None]) -> None:
    """Flag `range(...)` calls whose stop bound is a huge numeric literal."""
    if not (isinstance(call_node.func, ast.Name) and call_node.func.id == "range"):
        return
    if not call_node.args:
        return
    stop_expr = call_node.args[0] if len(call_node.args) == 1 else call_node.args[1]
    bound = _const_eval(stop_expr)
    if bound is not None and bound > HUGE_LOOP_BOUND:
        report(
            call_node,
            "huge_loop_bound",
            f"loop bound ~{bound:.3g} exceeds limit {HUGE_LOOP_BOUND}; "
            "iterations must be bounded to prevent resource exhaustion",
        )


def _snippet(code: str, node: ast.AST, limit: int = 120) -> str:
    """Short, single-line source snippet for a finding (never the full code)."""
    try:
        segment = ast.get_source_segment(code, node) or ""
    except Exception:
        segment = ""
    return " ".join(segment.split())[:limit]


def classify_code_violations(code: str) -> list[dict[str, str]]:
    """
    Statically analyze untrusted signal code and report sandbox violations.

    Returns a list of findings, each ``{"type": str, "detail": str, "line": str}``
    (``line`` may be ``"?"``). An empty list means no *statically detectable*
    violation — it does not prove the code is safe.
    """
    if not isinstance(code, str):
        return [{"type": "invalid_input", "detail": "code must be a string", "line": "?"}]

    findings: list[dict[str, str]] = []

    def report(node: ast.AST | None, violation_type: str, detail: str) -> None:
        snippet = _snippet(code, node) if node is not None else ""
        full_detail = detail + (f" (`{snippet}`)" if snippet else "")
        findings.append(
            {
                "type": violation_type,
                "detail": full_detail[:200],
                "line": str(getattr(node, "lineno", "?") or "?"),
            }
        )

    try:
        tree = ast.parse(code.translate(_SMART_UNICODE))
    except (SyntaxError, ValueError) as exc:
        lineno = getattr(exc, "lineno", None)
        return [
            {
                "type": "syntax_error",
                "detail": f"code could not be parsed: {type(exc).__name__}: {str(exc)[:120]}",
                "line": str(lineno) if lineno else "?",
            }
        ]

    # alias -> root module of the object it names (e.g. `import os as o` -> {"o": "os"}).
    # Seeded with the sandbox's own pre-bound aliases (pd/np/...) and the known
    # dangerous module roots, so `pd.io` or `os.system` are flagged even when the
    # corresponding import is missing from the module.
    alias_roots: dict[str, str] = dict(_SEEDED_ALIASES)
    for dangerous_root in DANGEROUS_MODULE_ROOTS:
        alias_roots.setdefault(dangerous_root, dangerous_root)

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                root = alias.name.split(".")[0]
                alias_roots[alias.asname or root] = root
                if root not in ALLOWED_IMPORTS:
                    report(
                        node,
                        "disallowed_import",
                        f"import of module '{alias.name}' is not allowed "
                        f"(allowed: {', '.join(sorted(ALLOWED_IMPORTS))})",
                    )

        elif isinstance(node, ast.ImportFrom):
            root = (node.module or "").split(".")[0]
            if node.level:
                report(node, "relative_import", "relative imports are not allowed")
                continue
            for alias in node.names:
                if alias.name == "*":
                    report(node, "star_import", "star imports are not allowed")
                    continue
                if root:
                    alias_roots[alias.asname or alias.name] = root
                if root not in ALLOWED_IMPORTS:
                    report(
                        node,
                        "disallowed_import",
                        f"import of module '{root}' is not allowed "
                        f"(allowed: {', '.join(sorted(ALLOWED_IMPORTS))})",
                    )
                elif root in _MODULE_DENYLISTS and (
                    alias.name in _MODULE_DENYLISTS[root] or alias.name.startswith("_")
                ):
                    report(
                        node,
                        "restricted_import_symbol",
                        f"importing '{root}.{alias.name}' is not allowed",
                    )

        elif isinstance(node, ast.Call):
            func = node.func
            if isinstance(func, ast.Name) and func.id in FORBIDDEN_BUILTINS:
                report(node, "forbidden_builtin_call", f"call to builtin '{func.id}' is not allowed")
            elif isinstance(func, ast.Attribute) and func.attr in FORBIDDEN_BUILTINS:
                report(node, "forbidden_builtin_call", f"call to '.{func.attr}(...)' is not allowed")
            elif isinstance(func, ast.Attribute) and func.attr in ("format", "format_map"):
                report(
                    node,
                    "unsafe_format",
                    "str.format/format_map can traverse object attributes from format "
                    "strings (e.g. '{0.__class__}'); use f-strings instead",
                )
            else:
                root = _root_name(func)
                if root and alias_roots.get(root) in DANGEROUS_MODULE_ROOTS:
                    report(
                        node,
                        "dangerous_call",
                        f"call through restricted module '{alias_roots[root]}' is not allowed",
                    )
            _check_range_bound(node, report)

        elif isinstance(node, ast.Attribute):
            attr = node.attr
            if attr in FRAME_INTROSPECTION_ATTRIBUTES:
                report(
                    node,
                    "frame_introspection",
                    f"access to frame/introspection attribute '{attr}' is not allowed",
                )
            elif attr in DUNDER_ESCAPE_ATTRIBUTES or attr.startswith("__"):
                report(node, "dunder_attribute", f"access to dunder attribute '{attr}' is not allowed")
            elif attr.startswith("_"):
                report(node, "underscore_attribute", f"access to private attribute '{attr}' is not allowed")
            else:
                base_root = _root_name(node.value)
                module_root = alias_roots.get(base_root) if base_root else None
                if module_root in DANGEROUS_MODULE_ROOTS:
                    report(
                        node,
                        "dangerous_attribute",
                        f"attribute access on restricted module '{module_root}' is not allowed",
                    )
                elif module_root in _MODULE_DENYLISTS and attr in _MODULE_DENYLISTS[module_root]:
                    report(
                        node,
                        "restricted_module_access",
                        f"access to '{module_root}.{attr}' is not allowed",
                    )

        elif isinstance(node, ast.Name):
            # Bare dunder names (`__builtins__`, `__subclasses__`, ...) are classic
            # escape primitives. `__name__` is allowed (e.g. main-guards).
            if node.id.startswith("__") and node.id != "__name__":
                report(node, "dunder_name", f"reference to dunder name '{node.id}' is not allowed")

        elif isinstance(node, ast.While):
            if _is_truthy_constant(node.test) and not _loop_has_break(node):
                report(
                    node,
                    "infinite_loop_risk",
                    "'while' loop with constant-true condition and no break may hang execution",
                )
            elif isinstance(node.test, ast.Compare):
                operands = [node.test.left, *node.test.comparators]
                bounds = [v for v in (_const_eval(op) for op in operands) if v is not None]
                if bounds and max(bounds) > HUGE_LOOP_BOUND:
                    report(
                        node,
                        "huge_loop_bound",
                        f"loop bound ~{max(bounds):.3g} exceeds limit {HUGE_LOOP_BOUND}",
                    )

        elif isinstance(node, ast.For):
            if isinstance(node.iter, ast.Call):
                _check_range_bound(node.iter, report)

        elif isinstance(node, ast.comprehension):
            if isinstance(node.iter, ast.Call):
                _check_range_bound(node.iter, report)

    return findings


# ---------------------------------------------------------------------------
# Validation (fail-closed gate)
# ---------------------------------------------------------------------------


def _format_violations(violations: list[dict[str, str]], max_listed: int = 20) -> str:
    lines = [f"  - [{v['type']}] line {v['line']}: {v['detail']}" for v in violations[:max_listed]]
    if len(violations) > max_listed:
        lines.append(f"  - ... and {len(violations) - max_listed} more violation(s)")
    return f"{len(violations)} sandbox violation(s):\n" + "\n".join(lines)


def validate_signal_code(code: str) -> None:
    """
    Validate untrusted signal code. Raises :class:`SandboxViolation` when the
    code violates the sandbox policy (fail closed); returns ``None`` when clean.

    Enforces: maximum source length (``MAX_CODE_LENGTH``), absence of null
    bytes, parseability, and every finding from :func:`classify_code_violations`.
    """
    if not isinstance(code, str):
        raise SandboxViolation("signal code must be a string")
    if len(code) > MAX_CODE_LENGTH:
        raise SandboxViolation(
            f"signal code too long: {len(code)} characters (max {MAX_CODE_LENGTH})"
        )
    if "\x00" in code:
        raise SandboxViolation("signal code contains null bytes")
    violations = classify_code_violations(code)
    if violations:
        raise SandboxViolation(_format_violations(violations))


# ---------------------------------------------------------------------------
# AST preparation: rewrite allowlisted imports into pre-bound namespace entries
# ---------------------------------------------------------------------------


class _ImportNeutralizer(ast.NodeTransformer):
    """Rewrite allowlisted import statements to ``pass`` and record the names
    they would bind, resolved *by the host* against restricted module proxies.

    Sandbox-executed code has no ``__import__`` builtin, so import statements
    cannot run by themselves; only modules in :data:`ALLOWED_IMPORTS` are
    pre-bound here. Non-allowlisted imports (which validation already rejects)
    are left untouched and fail at runtime with ``ImportError`` — fail closed.
    """

    def __init__(self, bindings: dict[str, Any]):
        self._bindings = bindings

    @staticmethod
    def _as_pass(node: ast.stmt) -> ast.Pass:
        replacement = ast.Pass()
        return ast.copy_location(replacement, node)

    def visit_Import(self, node: ast.Import) -> ast.stmt:
        allowed = True
        for alias in node.names:
            root = alias.name.split(".")[0]
            if root not in ALLOWED_IMPORTS:
                allowed = False
                continue
            self._bindings[alias.asname or root] = _module_proxy(root)
        return self._as_pass(node) if allowed else node

    def visit_ImportFrom(self, node: ast.ImportFrom) -> ast.stmt:
        root = (node.module or "").split(".")[0]
        if node.level or root not in ALLOWED_IMPORTS:
            return node if not node.level else self._as_pass(node)
        proxy = _module_proxy(root)
        denylist = _MODULE_DENYLISTS.get(root, frozenset())
        for alias in node.names:
            if alias.name == "*":
                continue  # statically rejected; bind nothing
            if alias.name in denylist or alias.name.startswith("_"):
                continue  # statically rejected; bind nothing (fail closed)
            try:
                self._bindings[alias.asname or alias.name] = _restricted_getattr(proxy, alias.name)
            except SandboxViolation:
                continue
        return self._as_pass(node)


def _prepare_ast(code: str) -> tuple[ast.Module, dict[str, Any]]:
    """Parse (with the evaluator's smart-quote normalization) and neutralize
    allowlisted imports. Raises SandboxViolation on parse failure."""
    try:
        tree = ast.parse(code.translate(_SMART_UNICODE))
    except (SyntaxError, ValueError) as exc:
        raise SandboxViolation(f"sandbox could not parse signal code: {exc}") from None
    bindings: dict[str, Any] = {}
    tree = _ImportNeutralizer(bindings).visit(tree)
    ast.fix_missing_locations(tree)
    return tree, bindings


# ---------------------------------------------------------------------------
# Safe operator-catalogue loading (first-party code, still executed restricted)
# ---------------------------------------------------------------------------

CALCULATOR_PATH = Path(__file__).parent / "template" / "calculator.json"

_OPERATOR_FUNCTIONS: dict[str, Callable[..., Any]] | None = None


def _load_operators_safely() -> dict[str, Callable[..., Any]]:
    """Load operator functions from ``calculator.json`` with safe loading.

    Unlike ``signal_evaluator._load_all_operators`` (plain ``exec`` with full
    builtins), each operator snippet is statically validated and then compiled
    and executed with the sandbox's restricted builtins and module proxies.
    Operators that fail validation are skipped (fail closed). Results are cached.
    """
    global _OPERATOR_FUNCTIONS
    if _OPERATOR_FUNCTIONS is not None:
        return _OPERATOR_FUNCTIONS

    loaded: dict[str, Callable[..., Any]] = {}
    if CALCULATOR_PATH.exists():
        with open(CALCULATOR_PATH, "r") as f:
            operators = json.load(f)
        for op in operators:
            name, op_code = op.get("name"), op.get("code")
            if not isinstance(name, str) or not isinstance(op_code, str):
                continue
            try:
                if classify_code_violations(op_code):
                    logger.warning("sandbox: operator '%s' rejected by static validation; skipping", name)
                    continue
                tree, _ = _prepare_ast(op_code)
                ns: dict[str, Any] = {
                    "__name__": "sandbox_operator",
                    "__builtins__": _build_safe_builtins([]),
                }
                for alias, root in (
                    ("pd", "pandas"), ("pandas", "pandas"),
                    ("np", "numpy"), ("numpy", "numpy"),
                    ("math", "math"), ("statistics", "statistics"),
                ):
                    ns[alias] = _module_proxy(root)
                exec(compile(tree, "<sandbox_operator>", "exec"), ns)
                fn = ns.get(name)
                if callable(fn):
                    loaded[name] = fn
                else:
                    logger.warning("sandbox: operator '%s' did not define a callable; skipping", name)
            except Exception as exc:
                logger.warning("sandbox: failed to load operator '%s': %s", name, str(exc)[:200])
    else:
        logger.warning("sandbox: operator catalogue not found at %s", CALCULATOR_PATH)

    _OPERATOR_FUNCTIONS = loaded
    return loaded


# ---------------------------------------------------------------------------
# Restricted execution namespace
# ---------------------------------------------------------------------------


def _execute_restricted(
    code: str,
    stock_data: dict[str, pd.DataFrame],
    selection_periods: int,
) -> tuple[pd.DataFrame, str] | None:
    """Validate, then execute untrusted code under the restricted namespace and
    select the best signal. May raise SandboxError/SandboxViolation; callers
    (public API, child runner) are responsible for converting that to None."""
    validate_signal_code(code)

    operators = _load_operators_safely()
    tree, bindings = _prepare_ast(code)

    print_buffer: list[str] = []
    namespace: dict[str, Any] = {
        "__name__": "sandboxed_signal",
        "__builtins__": _build_safe_builtins(print_buffer),
    }
    for alias, root in _SEEDED_ALIASES:
        namespace[alias] = _module_proxy(root)
    namespace.update(operators)  # catalogue operators available (old contract)
    namespace.update(bindings)   # names bound by the code's own allowlisted imports

    try:
        exec(compile(tree, "<sandboxed_signal>", "exec"), namespace)
    except SandboxViolation:
        raise
    except Exception as exc:
        raise SandboxError(
            f"signal code failed to execute: {type(exc).__name__}: {str(exc)[:200]}"
        ) from exc

    if print_buffer:
        logger.debug("sandbox: captured %d print line(s) from generated code", len(print_buffer))

    return _select_best_signal(namespace, stock_data, selection_periods)


# ---------------------------------------------------------------------------
# Signal selection (mirrors signal_evaluator.execute_signal_code contract)
# ---------------------------------------------------------------------------

#: Standard data field names; mirrors signal_evaluator.STANDARD_FIELDS.
STANDARD_FIELDS = ["Close", "Volume", "High", "Low", "Open"]


def _is_dataframe_param(p: inspect.Parameter) -> bool:
    """Mirror of signal_evaluator._is_dataframe_param."""
    ann = p.annotation
    if ann is pd.DataFrame:
        return True
    if isinstance(ann, str) and "DataFrame" in ann:
        return True
    if ann is inspect.Parameter.empty:
        if p.default is inspect.Parameter.empty:
            return True
        if p.default is None or isinstance(p.default, pd.DataFrame):
            return True
        return False
    return False


def _resolve_signal_args(
    sig: inspect.Signature,
    stock_data: dict[str, pd.DataFrame],
) -> dict[str, pd.DataFrame]:
    """Mirror of signal_evaluator._resolve_signal_args (kept behavior-identical)."""
    available_fields = [f for f in STANDARD_FIELDS if f in stock_data]
    df_params = [name for name, p in sig.parameters.items() if _is_dataframe_param(p)]

    kwargs: dict[str, pd.DataFrame] = {}
    used_fields: set[str] = set()
    unmatched: list[str] = []

    for param in df_params:
        match: str | None = None
        p_lower = param.lower()
        for field in available_fields:
            f_lower = field.lower()
            if p_lower == f_lower or f_lower in p_lower or p_lower in f_lower:
                match = field
                break
        if match and match not in used_fields:
            kwargs[param] = stock_data[match]
            used_fields.add(match)
        else:
            unmatched.append(param)

    fallback_pool = [f for f in available_fields if f not in used_fields]
    for param in unmatched:
        if not fallback_pool:
            break
        kwargs[param] = stock_data[fallback_pool.pop(0)]

    return kwargs


def _detect_helpers(candidates: list[tuple[str, Any]]) -> set[str]:
    """Mirror of signal_evaluator._detect_helpers."""
    helpers: set[str] = set()
    names = {name for name, _ in candidates}
    for _, fn in candidates:
        try:
            src = inspect.getsource(fn)
        except (OSError, TypeError):
            continue
        for name in names:
            if name != fn.__name__ and re.search(rf"\b{re.escape(name)}\s*\(", src):
                helpers.add(name)
    return helpers


def _select_best_signal(
    namespace: dict[str, Any],
    stock_data: dict[str, pd.DataFrame],
    selection_periods: int,
) -> tuple[pd.DataFrame, str] | None:
    """Execute candidate signal functions and return (values, name) for the
    signal with the highest |IC| — same contract as the evaluator."""
    operator_names = set(_load_operators_safely().keys())
    candidates = [
        (name, obj)
        for name, obj in namespace.items()
        if not name.startswith("_")
        and callable(obj)
        and hasattr(obj, "__code__")
        and name not in operator_names
    ]

    if not candidates:
        logger.warning("sandbox: no signal functions found in the code")
        return None

    helpers = _detect_helpers(candidates)
    signal_functions = [(n, f) for n, f in candidates if n not in helpers] or candidates

    logger.info(
        "sandbox: found %d signal function(s): %s",
        len(signal_functions),
        [n for n, _ in signal_functions],
    )

    best_result: pd.DataFrame | None = None
    best_ic: float | None = None
    best_name: str | None = None

    with np.errstate(invalid="ignore", divide="ignore"), warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=RuntimeWarning)

        for func_name, signal_func in signal_functions:
            try:
                sig = inspect.signature(signal_func)
                kwargs = _resolve_signal_args(sig, stock_data)
                df_param_count = sum(
                    1 for p in sig.parameters.values() if _is_dataframe_param(p)
                )

                if len(kwargs) == df_param_count:
                    result = signal_func(**kwargs)
                elif len(sig.parameters) == 0:
                    result = signal_func()
                else:
                    logger.warning(
                        "sandbox: cannot determine args for %s (params=%s), skipping",
                        func_name,
                        list(sig.parameters),
                    )
                    continue

                if isinstance(result, pd.Series):
                    result = result.to_frame()

                if isinstance(result, pd.DataFrame):
                    forward_ret = (
                        stock_data["Close"].shift(-selection_periods) / stock_data["Close"] - 1
                    )
                    valid_dates = result.dropna(how="all").index
                    sample_ics: list[float] = []

                    for date in valid_dates:
                        if date not in forward_ret.index:
                            continue
                        fr = result.loc[date].dropna()
                        rr = forward_ret.loc[date].dropna()
                        common = fr.index.intersection(rr.index)
                        if len(common) < 10:
                            continue
                        f_vals = fr[common].values
                        r_vals = rr[common].values
                        if np.std(f_vals) < 1e-10 or np.std(r_vals) < 1e-10:
                            continue
                        try:
                            corr, _ = stats.spearmanr(f_vals, r_vals)
                            if not np.isnan(corr):
                                sample_ics.append(corr)
                        except Exception:
                            pass

                    if sample_ics:
                        mean_ic = abs(float(np.mean(sample_ics)))
                        logger.info("sandbox: %s: |IC| = %.4f", func_name, mean_ic)
                        if best_ic is None or mean_ic > best_ic:
                            best_ic, best_result, best_name = mean_ic, result, func_name
                    elif best_result is None:
                        best_result, best_name = result, func_name

            except SandboxViolation:
                logger.error(
                    "sandbox: runtime sandbox violation while executing '%s'; aborting",
                    func_name,
                )
                return None
            except Exception as exc:
                logger.warning("sandbox: error executing %s: %s", func_name, str(exc)[:200])
                continue

    if best_result is not None:
        logger.info("sandbox: selected best signal: %s", best_name)
        return best_result, best_name

    return None


# ---------------------------------------------------------------------------
# Subprocess isolation mode
# ---------------------------------------------------------------------------

#: Trusted runner executed in the child process. argv:
#: <payload_path> <result_path> <package_parent> <cpu_seconds> <memory_mb>
#: It applies rlimits FIRST, re-validates the code, executes it under the same
#: restricted namespace as in-process mode, and pickles the result with
#: owner-only permissions.
_CHILD_RUNNER = '''\
"""Trusted sandbox runner (child process). Not user-editable."""
import os
import pickle
import sys


def _apply_rlimits(cpu_seconds, memory_mb):
    import resource

    cpu = max(1, int(cpu_seconds))
    try:
        resource.setrlimit(resource.RLIMIT_CPU, (cpu, cpu + 1))
    except Exception:
        pass
    if memory_mb and int(memory_mb) > 0:
        mem_bytes = max(256, int(memory_mb)) * 1024 * 1024
        try:
            resource.setrlimit(resource.RLIMIT_AS, (mem_bytes, mem_bytes))
        except Exception:
            pass
    try:
        soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
        target = 256 if hard == resource.RLIM_INFINITY else min(hard, 256)
        if soft > target:
            resource.setrlimit(resource.RLIMIT_NOFILE, (target, hard))
    except Exception:
        pass


def main():
    payload_path, result_path, pkg_parent = sys.argv[1], sys.argv[2], sys.argv[3]
    cpu_seconds, memory_mb = float(sys.argv[4]), int(sys.argv[5])
    _apply_rlimits(cpu_seconds, memory_mb)
    if pkg_parent and pkg_parent not in sys.path:
        sys.path.insert(0, pkg_parent)
    from signal_discovery_workflow import sandbox as sb

    with open(payload_path, "rb") as f:
        payload = pickle.load(f)

    sb.validate_signal_code(payload["code"])  # re-validate in the child (fail closed)
    result = sb._execute_restricted(
        payload["code"], payload["stock_data"], payload["selection_periods"]
    )

    tmp_path = result_path + ".tmp"
    fd = os.open(tmp_path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "wb") as f:
        pickle.dump(result, f, protocol=4)
    os.replace(tmp_path, result_path)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        sys.stderr.write(f"sandbox runner failed: {type(exc).__name__}: {str(exc)[:500]}\\n")
        sys.exit(3)
'''


def _kill_process_tree(proc: subprocess.Popen) -> None:
    """Best-effort kill of the child (and its process group) without raising."""
    try:
        if hasattr(os, "killpg"):
            os.killpg(proc.pid, signal.SIGKILL)
        else:
            proc.kill()
    except Exception:
        pass
    try:
        proc.communicate(timeout=5)
    except Exception:
        pass


def _execute_in_subprocess(
    code: str,
    stock_data: dict[str, pd.DataFrame],
    selection_periods: int,
    timeout_seconds: float,
    max_memory_mb: int | None,
) -> tuple[pd.DataFrame, str] | None:
    """Run the code in an isolated child process with rlimits + wall-clock timeout."""
    timeout = max(1.0, float(timeout_seconds))
    pkg_parent = str(Path(__file__).resolve().parent.parent)
    tmpdir = tempfile.mkdtemp(prefix="signal_sandbox_")
    payload_path = os.path.join(tmpdir, "payload.pkl")
    result_path = os.path.join(tmpdir, "result.pkl")
    proc: subprocess.Popen | None = None
    try:
        os.chmod(tmpdir, 0o700)  # restrictive permissions on the temp dir
        fd = os.open(payload_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(fd, "wb") as f:
            pickle.dump(
                {"code": code, "stock_data": stock_data, "selection_periods": selection_periods},
                f,
                protocol=4,
            )

        proc = subprocess.Popen(
            [
                sys.executable, "-c", _CHILD_RUNNER,
                payload_path, result_path, pkg_parent,
                str(int(timeout) + 2),              # RLIMIT_CPU headroom vs wall clock
                str(int(max_memory_mb) if max_memory_mb else 0),
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            cwd=tmpdir,
            env=os.environ.copy(),
            close_fds=True,
            start_new_session=True,  # own process group -> killpg kills descendants
        )
        try:
            _stdout, stderr = proc.communicate(timeout=timeout + 1.0)
        except subprocess.TimeoutExpired:
            logger.warning("sandbox: child process exceeded %.1fs wall clock; killed", timeout)
            _kill_process_tree(proc)
            return None

        if proc.returncode != 0:
            err_tail = (stderr or b"")[-500:].decode("utf-8", errors="replace")
            logger.warning(
                "sandbox: child exited with code %s: %s", proc.returncode, err_tail
            )
            return None

        if not os.path.exists(result_path):
            logger.warning("sandbox: child produced no result file")
            return None

        with open(result_path, "rb") as f:
            result = pickle.load(f)

        if (
            isinstance(result, tuple)
            and len(result) == 2
            and isinstance(result[0], pd.DataFrame)
            and isinstance(result[1], str)
        ):
            return result
        logger.warning("sandbox: child returned unexpected result type %s", type(result).__name__)
        return None

    except SandboxViolation as exc:
        logger.warning("sandbox: violation in subprocess mode: %s", exc)
        return None
    except Exception as exc:
        logger.error(
            "sandbox: subprocess execution failed: %s: %s", type(exc).__name__, str(exc)[:200]
        )
        return None
    finally:
        if proc is not None and proc.poll() is None:
            _kill_process_tree(proc)
        shutil.rmtree(tmpdir, ignore_errors=True)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def execute_signal_code_sandboxed(
    code: str,
    stock_data: dict[str, pd.DataFrame],
    selection_periods: int = 5,
    timeout_seconds: float = 30.0,
    max_memory_mb: int | None = 2048,
    allow_subprocess: bool = False,
) -> tuple[pd.DataFrame, str] | None:
    """
    Execute untrusted, LLM-generated signal code under sandbox controls.

    Same contract as ``signal_evaluator.execute_signal_code``: returns
    ``(signal_values_df, selected_signal_name)`` for the signal function with
    the highest |IC| defined in ``code``, or ``None`` on ANY failure. This
    function never raises and never logs the full generated code.

    Args:
        code: Untrusted Python source defining signal functions.
        stock_data: ``{field: DataFrame}`` price-volume data (mirrors evaluator).
        selection_periods: Forward-return horizon used to rank signals by |IC|.
        timeout_seconds: Wall-clock budget. Enforced by killing the child in
            subprocess mode; NOT enforceable in in-process mode (documented
            residual risk).
        max_memory_mb: RLIMIT_AS for the child process. ``None`` disables the
            memory limit. Ignored (with a debug note) in in-process mode.
        allow_subprocess: Use subprocess isolation mode. Falls back to
            in-process mode with a warning where ``resource`` is unavailable
            (Windows).

    Returns:
        ``(signal_values_df, selected_signal_name)`` or ``None``.
    """
    use_subprocess = bool(allow_subprocess)
    if use_subprocess and not RESOURCE_LIMITS_AVAILABLE:
        logger.warning(
            "sandbox: resource limits unavailable on this platform; "
            "degrading to in-process restricted mode"
        )
        use_subprocess = False

    # Fail closed: static validation gate before any execution path.
    try:
        validate_signal_code(code)
    except SandboxViolation as exc:
        logger.warning("sandbox: rejected signal code: %s", exc)
        return None
    except Exception as exc:
        logger.error(
            "sandbox: unexpected validation failure: %s: %s", type(exc).__name__, str(exc)[:200]
        )
        return None

    try:
        if use_subprocess:
            return _execute_in_subprocess(
                code, stock_data, selection_periods, timeout_seconds, max_memory_mb
            )
        if timeout_seconds and float(timeout_seconds) > 0:
            logger.debug(
                "sandbox: in-process mode cannot enforce timeout_seconds=%.1f or "
                "max_memory_mb=%s (no wall-clock/memory control without process isolation)",
                float(timeout_seconds),
                max_memory_mb,
            )
        return _execute_restricted(code, stock_data, selection_periods)
    except SandboxViolation as exc:
        logger.warning("sandbox: violation during sandboxed execution: %s", exc)
        return None
    except SandboxError as exc:
        logger.error("sandbox: execution failed: %s", str(exc)[:300])
        return None
    except Exception as exc:
        logger.error(
            "sandbox: unexpected execution failure: %s: %s", type(exc).__name__, str(exc)[:200]
        )
        return None


_HIGH_RISK_TYPES = frozenset(
    {
        "disallowed_import", "relative_import", "star_import",
        "restricted_import_symbol", "forbidden_builtin_call",
        "dunder_name", "dunder_attribute", "underscore_attribute",
        "frame_introspection",
        "dangerous_call", "dangerous_attribute", "restricted_module_access",
        "unsafe_format", "invalid_input",
    }
)

_MEDIUM_RISK_TYPES = frozenset({"infinite_loop_risk", "huge_loop_bound"})


def sandbox_report(code: str) -> dict[str, Any]:
    """
    Summarize the sandbox verdict on a piece of untrusted code for a
    "skeptic" reviewer agent.

    Returns ``{"safe": bool, "violations": [...], "violation_count": int,
    "estimated_risk": "low"|"medium"|"high", "recommendation": str}``.

    ``safe=True`` means *statically clean* — not a guarantee of safety.
    """
    try:
        violations = classify_code_violations(code)
    except SandboxError as exc:
        violations = [{"type": "invalid_input", "detail": str(exc)[:200], "line": "?"}]
    except Exception as exc:  # analyzer itself failed — fail closed
        violations = [
            {"type": "invalid_input", "detail": f"classifier failure: {str(exc)[:150]}", "line": "?"}
        ]

    count = len(violations)
    types_present = {v.get("type", "") for v in violations}

    if count == 0:
        return {
            "safe": True,
            "violations": [],
            "violation_count": 0,
            "estimated_risk": "low",
            "recommendation": (
                "Code passed static sandbox validation. Prefer executing via "
                "execute_signal_code_sandboxed(..., allow_subprocess=True) so CPU, memory "
                "and wall-clock limits apply; static analysis cannot prove safety."
            ),
        }

    if types_present & _HIGH_RISK_TYPES:
        risk = "high"
        recommendation = (
            "Do not execute this code: it attempts restricted capabilities "
            "(disallowed imports/calls, dunder access, or dynamic execution). "
            "Reject it or regenerate the signal."
        )
    elif types_present & _MEDIUM_RISK_TYPES:
        risk = "medium"
        recommendation = (
            "Do not execute as-is: potential resource exhaustion (unbounded or "
            "huge loop bounds). Require bounded iterations, then re-validate."
        )
    else:  # e.g. syntax_error — the code could not be analyzed
        risk = "medium"
        recommendation = (
            "Do not execute: the code could not be statically validated. "
            "Fix or regenerate the signal code first."
        )

    return {
        "safe": False,
        "violations": violations,
        "violation_count": count,
        "estimated_risk": risk,
        "recommendation": recommendation,
    }
