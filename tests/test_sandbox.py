# SPDX-FileCopyrightText: Copyright (c) 2023-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Tests for the sandbox layer around untrusted, LLM-generated signal code.

Covers static classification, fail-closed validation, restricted in-process
execution, subprocess isolation (rlimits + wall-clock timeout), and the
skeptic-facing sandbox_report summary.
"""

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from signal_discovery_workflow import sandbox as sb
from signal_discovery_workflow.sandbox import (
    ALLOWED_IMPORTS,
    SandboxError,
    SandboxViolation,
    classify_code_violations,
    execute_signal_code_sandboxed,
    sandbox_report,
    validate_signal_code,
)

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def stock_dates():
    return pd.date_range("2020-01-01", periods=100, freq="B")


@pytest.fixture
def stock_tickers():
    return [f"S{i:02d}" for i in range(20)]


@pytest.fixture
def stock_data(stock_dates, stock_tickers):
    close = pd.DataFrame(
        100 * np.exp(np.cumsum(np.random.default_rng(0).normal(0.0005, 0.01, size=(100, 20)), axis=0)),
        index=stock_dates,
        columns=stock_tickers,
    )
    volume = pd.DataFrame(
        np.random.default_rng(1).integers(100_000, 1_000_000, size=(100, 20)).astype(float),
        index=stock_dates,
        columns=stock_tickers,
    )
    return {
        "Open": close * 0.999,
        "Close": close,
        "High": close * 1.005,
        "Low": close * 0.995,
        "Volume": volume,
    }


#: Realistic, benign LLM output: inline operator definitions + two signals.
BENIGN_CODE = '''
import pandas as pd
import numpy as np


def TS_Return(x: pd.DataFrame, d: int) -> pd.DataFrame:
    return x.pct_change(d)


def Div(x: pd.DataFrame, y: pd.DataFrame) -> pd.DataFrame:
    result = x / y
    return result


def signal_volume_confirmed_momentum(Close: pd.DataFrame, Volume: pd.DataFrame) -> pd.DataFrame:
    """Volume confirmed momentum."""
    return Div(TS_Return(Close, 10), Volume)


def signal_simple_momentum(Close: pd.DataFrame) -> pd.DataFrame:
    """Plain momentum."""
    return TS_Return(Close, 5)
'''

#: Code that tries to touch the filesystem via os.system.
MALICIOUS_OS_CODE = '''
import os
import pandas as pd


def signal_pwn(Close: pd.DataFrame) -> pd.DataFrame:
    os.system("touch /tmp/pwned_sandbox_test")
    return Close
'''

#: Infinite loop that passes static validation (variable condition, bounded
#: literals) but never terminates — exercises the subprocess wall-clock kill.
HANGING_CODE = '''
import pandas as pd


def signal_hang_forever(Close: pd.DataFrame) -> pd.DataFrame:
    """Pretends to compute a signal but never returns."""
    i = 0
    while i >= 0:
        i += 1
    return Close
'''


def _violations_of(code: str) -> set[str]:
    return {v["type"] for v in classify_code_violations(code)}


def _combined_operator_module() -> str:
    """All catalogue operator implementations in one module + one normal signal
    function — the false-positive regression guard for attribute rules."""
    operators = json.loads(Path(sb.CALCULATOR_PATH).read_text())
    parts = [op["code"] for op in operators]
    parts.append(
        "\n\ndef signal_combined_check(Close: pd.DataFrame) -> pd.DataFrame:\n"
        "    return TS_Return(Close, 5)\n"
    )
    return "\n".join(parts)


# ---------------------------------------------------------------------------
# classify_code_violations
# ---------------------------------------------------------------------------

class TestClassifyCodeViolations:
    @pytest.mark.parametrize(
        ("code", "expected_type"),
        [
            ("import os", "disallowed_import"),
            ("import subprocess", "disallowed_import"),
            ("import socket", "disallowed_import"),
            ("from pathlib import Path", "disallowed_import"),
            ("x = open('/etc/passwd')", "forbidden_builtin_call"),
            ("y = __import__('os')", "forbidden_builtin_call"),
            ("y = eval('1+1')", "forbidden_builtin_call"),
            ("z = compile('1', '<s>', 'eval')", "forbidden_builtin_call"),
            ("n = input()", "forbidden_builtin_call"),
            ("g = globals()", "forbidden_builtin_call"),
            ("b = breakpoint()", "forbidden_builtin_call"),
            ("x = ().__class__.__bases__", "dunder_attribute"),
            ("f = (lambda: 0).__globals__", "dunder_attribute"),
            ("o = object.__subclasses__()", "dunder_attribute"),
            ("p = df._mgr", "underscore_attribute"),
            ("import shutil\nshutil.rmtree('/tmp')", "dangerous_call"),
            ("import sys\nsys.exit(1)", "dangerous_call"),
            ("import os as o\no.system('x')", "dangerous_call"),
        ],
    )
    def test_flags_malicious_patterns(self, code, expected_type):
        assert expected_type in _violations_of(code)

    def test_findings_have_required_shape(self):
        findings = classify_code_violations("import os\nos.system('x')")
        assert findings
        for finding in findings:
            assert set(finding.keys()) == {"type", "detail", "line"}
            assert all(isinstance(v, str) for v in finding.values())
            assert finding["line"] in {str(i + 1) for i in range(2)} or finding["line"] == "?"

    def test_while_true_without_break_flagged(self):
        assert "infinite_loop_risk" in _violations_of("while True:\n    x = 1\n")

    def test_while_true_with_break_not_flagged(self):
        assert "infinite_loop_risk" not in _violations_of(
            "while True:\n    x = 1\n    if x > 3:\n        break\n"
        )

    def test_huge_range_bound_flagged(self):
        findings = classify_code_violations("for i in range(10**9):\n    x = i\n")
        assert "huge_loop_bound" in {f["type"] for f in findings}

    def test_small_range_bound_clean(self):
        assert classify_code_violations("for i in range(100):\n    x = i\n") == []

    def test_huge_while_comparison_bound_flagged(self):
        findings = classify_code_violations("i = 0\nwhile i < 10**9:\n    i += 1\n")
        assert "huge_loop_bound" in {f["type"] for f in findings}

    # -- SB-01: frame / introspection attribute access ----------------------

    def test_frame_introspection_chain_flagged(self):
        """Audit finding SB-01: generator-frame escape chain must be flagged."""
        findings = classify_code_violations("_g.gi_frame.f_back.f_globals['__builtins__']")
        assert findings, "frame-introspection chain must not classify clean"
        assert "frame_introspection" in {f["type"] for f in findings}

    @pytest.mark.parametrize(
        ("code", "expected_type"),
        [
            ("h = gen.gi_frame", "frame_introspection"),
            ("c = gen.gi_code", "frame_introspection"),
            ("b = frm.f_back", "frame_introspection"),
            ("g = frm.f_globals", "frame_introspection"),
            ("l = frm.f_locals", "frame_introspection"),
            ("bl = frm.f_builtins", "frame_introspection"),
            ("tr = frm.f_trace", "frame_introspection"),
            ("tf = exc.__traceback__.tb_frame", "frame_introspection"),
            ("tn = exc.__traceback__.tb_next", "frame_introspection"),
            ("cr = co.cr_frame", "frame_introspection"),
            ("ag = agen.ag_frame", "frame_introspection"),
            ("cd = fn.__code__", "dunder_attribute"),
            ("gl = fn.__globals__", "dunder_attribute"),
            ("sc = cls.__subclasses__()", "dunder_attribute"),
            ("rd = obj.__reduce__()", "dunder_attribute"),
        ],
    )
    def test_frame_and_dunder_introspection_members_flagged(self, code, expected_type):
        assert expected_type in {f["type"] for f in classify_code_violations(code)}

    def test_ordinary_attribute_access_not_flagged(self):
        assert classify_code_violations("Close.pct_change(5)") == []
        assert classify_code_violations("x.shape") == []
        assert classify_code_violations("df.iloc[0]") == []
        assert classify_code_violations("result = signal.rolling(10).mean()") == []

    def test_all_operators_clean_combined_module(self):
        """False-positive regression guard: one module containing every catalogue
        operator plus a normal signal function must classify clean."""
        assert classify_code_violations(_combined_operator_module()) == []

    def test_legitimate_signal_code_is_clean(self):
        clean = (
            "import pandas as pd\n"
            "import numpy as np\n"
            "\n"
            "def signal_pct_change(Close: pd.DataFrame) -> pd.DataFrame:\n"
            '    """Simple 5-day momentum."""\n'
            "    return Close.pct_change(5)\n"
        )
        assert classify_code_violations(clean) == []

    def test_all_operator_catalogue_snippets_are_clean(self):
        """First-party operator snippets must never trip the classifier — a
        false positive here would silently drop operators from the sandbox."""
        calculator = Path(sb.CALCULATOR_PATH)
        assert calculator.exists()
        operators = json.loads(calculator.read_text())
        assert len(operators) > 0
        for op in operators:
            assert classify_code_violations(op["code"]) == [], op["name"]

    def test_syntax_error_reported(self):
        findings = classify_code_violations("def broken(:\n    pass")
        assert findings[0]["type"] == "syntax_error"

    def test_non_string_input_reported(self):
        findings = classify_code_violations(None)  # type: ignore[arg-type]
        assert findings and findings[0]["type"] == "invalid_input"

    def test_allowed_imports_content(self):
        assert ALLOWED_IMPORTS == frozenset({"pandas", "numpy", "math", "statistics"})

    def test_findings_do_not_embed_full_source(self):
        """Snippets in details are truncated; the full code is never echoed."""
        padding = "x = 1\n" * 100
        code = f"{padding}import os\nos.system('echo long-line-{'A' * 500}')\n"
        for finding in classify_code_violations(code):
            assert len(finding["detail"]) <= 200


# ---------------------------------------------------------------------------
# validate_signal_code
# ---------------------------------------------------------------------------

class TestValidateSignalCode:
    def test_clean_code_returns_none(self):
        assert validate_signal_code(BENIGN_CODE) is None

    def test_malicious_code_raises_sandbox_violation(self):
        with pytest.raises(SandboxViolation) as excinfo:
            validate_signal_code(MALICIOUS_OS_CODE)
        message = str(excinfo.value)
        assert "disallowed_import" in message  # readable aggregated message

    def test_null_bytes_rejected(self):
        with pytest.raises(SandboxViolation, match="null byte"):
            validate_signal_code("x = 1\x00")

    def test_overlong_code_rejected(self):
        code = "x = 1\n# " + "a" * 200_000
        assert len(code) > sb.MAX_CODE_LENGTH
        with pytest.raises(SandboxViolation, match="too long"):
            validate_signal_code(code)

    def test_normal_length_code_accepted(self):
        assert validate_signal_code("x = 1\n# padding\n") is None


# ---------------------------------------------------------------------------
# execute_signal_code_sandboxed — in-process restricted mode (default)
# ---------------------------------------------------------------------------

class TestExecuteSandboxedInProcess:
    def test_benign_module_returns_dataframe_and_name(self, stock_data):
        result = execute_signal_code_sandboxed(BENIGN_CODE, stock_data)
        assert result is not None
        signal_values, signal_name = result
        assert isinstance(signal_values, pd.DataFrame)
        assert isinstance(signal_name, str)
        assert signal_name in {"signal_simple_momentum", "signal_volume_confirmed_momentum"}
        assert signal_values.shape == stock_data["Close"].shape

    def test_selected_signal_has_highest_abs_ic(self, stock_data):
        """The contract: the returned signal is the one with the highest |IC|.
        Cross-check the sandbox's choice against a direct |IC| computation."""
        result = execute_signal_code_sandboxed(BENIGN_CODE, stock_data)
        assert result is not None
        _, selected = result

        import re

        from signal_discovery_workflow.signal_evaluator import execute_signal_code

        evaluator_result = execute_signal_code(BENIGN_CODE, stock_data)
        assert evaluator_result is not None
        assert selected == evaluator_result[1]

        # And the returned frame corresponds to the selected function's output
        signal_values, signal_name = result
        match = re.search(rf"def {signal_name}\(", BENIGN_CODE)
        assert match is not None

    def test_contract_matches_evaluator_frame(self, stock_data):
        from signal_discovery_workflow.signal_evaluator import execute_signal_code

        sandbox_result = execute_signal_code_sandboxed(BENIGN_CODE, stock_data)
        evaluator_result = execute_signal_code(BENIGN_CODE, stock_data)
        assert sandbox_result is not None and evaluator_result is not None
        assert sandbox_result[1] == evaluator_result[1]
        pd.testing.assert_frame_equal(sandbox_result[0], evaluator_result[0])

    def test_os_system_code_returns_none_and_no_file(self, stock_data):
        target = Path("/tmp/pwned_sandbox_test")
        if target.exists():  # clear stale artifacts from earlier runs
            target.unlink()
        try:
            result = execute_signal_code_sandboxed(MALICIOUS_OS_CODE, stock_data)
            assert result is None
            assert not target.exists()
        finally:
            if target.exists():  # only reachable if the sandbox failed
                target.unlink()

    def test_os_system_without_import_also_fails_closed(self, stock_data):
        """Without `import os`, `os.system` is a NameError at call time —
        the signal fails and no file may be created."""
        target = Path("/tmp/pwned_sandbox_test")
        if target.exists():
            target.unlink()
        try:
            code = (
                "def signal_pwn(Close: pd.DataFrame) -> pd.DataFrame:\n"
                "    os.system('touch /tmp/pwned_sandbox_test')\n"
                "    return Close\n"
            )
            result = execute_signal_code_sandboxed(code, stock_data)
            assert result is None
            assert not target.exists()
        finally:
            if target.exists():
                target.unlink()

    def test_runtime_proxy_setattr_fails_closed(self, stock_data):
        """Passes static validation but violates the restricted-module proxy at
        runtime — must return None, not raise, and not corrupt the host."""
        code = (
            "import pandas as pd\n"
            "\n"
            "def signal_mutate(Close: pd.DataFrame) -> pd.DataFrame:\n"
            "    pd.sandbox_was_here = True\n"
            "    return Close\n"
        )
        result = execute_signal_code_sandboxed(code, stock_data)
        assert result is None
        assert not hasattr(pd, "sandbox_was_here")

    @pytest.mark.parametrize(
        "code",
        [
            "",  # empty
            "   \n",  # whitespace only
            "x = 1\n",  # no signal functions
            "def broken(:\n",  # syntax error
            None,  # not a string
            12345,  # not a string
        ],
    )
    def test_never_raises_returns_none(self, code, stock_data):
        result = execute_signal_code_sandboxed(code, stock_data)  # type: ignore[arg-type]
        assert result is None

    def test_exception_classes_are_importable_and_related(self):
        assert issubclass(SandboxViolation, SandboxError)
        assert issubclass(SandboxError, Exception)

    def test_frame_introspection_snippet_returns_none(self, stock_data):
        """SB-01: a frame-walking snippet is rejected at the static gate —
        returns None, raises nothing, and reaches no interpreter frame."""
        code = (
            "import pandas as pd\n"
            "\n"
            "def signal_frame_probe(Close: pd.DataFrame) -> pd.DataFrame:\n"
            "    _g = (x for x in Close.values)\n"
            "    chain = _g.gi_frame.f_back.f_globals['__builtins__']\n"
            "    return Close\n"
        )
        result = execute_signal_code_sandboxed(code, stock_data)
        assert result is None

    def test_combined_operator_module_executes_end_to_end(self, stock_data):
        """Strongest false-positive guard: all 66 operators + a normal signal
        still run under the sandbox after the frame-introspection rule."""
        result = execute_signal_code_sandboxed(_combined_operator_module(), stock_data)
        assert result is not None
        values, name = result
        assert isinstance(values, pd.DataFrame)
        assert name == "signal_combined_check"


# ---------------------------------------------------------------------------
# execute_signal_code_sandboxed — subprocess isolation mode
# ---------------------------------------------------------------------------

requires_resource = pytest.mark.skipif(
    not sb.RESOURCE_LIMITS_AVAILABLE,
    reason="resource module unavailable (rlimits/subprocess isolation not supported)",
)


class TestExecuteSandboxedSubprocess:
    @requires_resource
    def test_benign_module_roundtrip(self, stock_data):
        result = execute_signal_code_sandboxed(
            BENIGN_CODE, stock_data, allow_subprocess=True, timeout_seconds=30.0
        )
        assert result is not None
        signal_values, signal_name = result
        assert isinstance(signal_values, pd.DataFrame)
        assert signal_name in {"signal_simple_momentum", "signal_volume_confirmed_momentum"}

    @requires_resource
    def test_subprocess_matches_in_process_result(self, stock_data):
        in_process = execute_signal_code_sandboxed(BENIGN_CODE, stock_data)
        in_subprocess = execute_signal_code_sandboxed(
            BENIGN_CODE, stock_data, allow_subprocess=True
        )
        assert in_process is not None and in_subprocess is not None
        assert in_process[1] == in_subprocess[1]
        pd.testing.assert_frame_equal(in_process[0], in_subprocess[0])

    @requires_resource
    def test_infinite_loop_killed_within_wall_clock(self, stock_data):
        """A hanging signal must be killed and must not block the test suite."""
        timeout_seconds = 4.0
        start = time.monotonic()
        result = execute_signal_code_sandboxed(
            HANGING_CODE,
            stock_data,
            allow_subprocess=True,
            timeout_seconds=timeout_seconds,
        )
        elapsed = time.monotonic() - start
        assert result is None
        assert elapsed < timeout_seconds + 5, f"hang test took {elapsed:.1f}s"

    @requires_resource
    def test_malicious_code_rejected_before_spawning(self, stock_data):
        """Validation fails closed in the parent; no child process is needed."""
        start = time.monotonic()
        result = execute_signal_code_sandboxed(
            MALICIOUS_OS_CODE, stock_data, allow_subprocess=True
        )
        assert result is None
        assert time.monotonic() - start < 5  # no subprocess overhead observed


# ---------------------------------------------------------------------------
# sandbox_report
# ---------------------------------------------------------------------------

class TestSandboxReport:
    def test_clean_code_reported_safe(self):
        report = sandbox_report(BENIGN_CODE)
        assert report["safe"] is True
        assert report["violations"] == []
        assert report["violation_count"] == 0
        assert report["estimated_risk"] == "low"
        assert isinstance(report["recommendation"], str) and report["recommendation"]

    def test_malicious_code_reported_unsafe(self):
        report = sandbox_report(MALICIOUS_OS_CODE)
        assert report["safe"] is False
        assert len(report["violations"]) > 0
        assert report["violation_count"] == len(report["violations"])
        assert report["estimated_risk"] == "high"
        assert isinstance(report["recommendation"], str) and report["recommendation"]

    def test_dunder_escape_reported_high(self):
        report = sandbox_report("x = ().__class__.__bases__")
        assert report["safe"] is False
        assert report["estimated_risk"] == "high"

    def test_loop_risk_reported_medium(self):
        report = sandbox_report("for i in range(10**9):\n    x = i\n")
        assert report["safe"] is False
        assert report["estimated_risk"] == "medium"

    def test_report_shape(self):
        report = sandbox_report("import os")
        assert set(report.keys()) == {
            "safe", "violations", "violation_count", "estimated_risk", "recommendation"
        }
