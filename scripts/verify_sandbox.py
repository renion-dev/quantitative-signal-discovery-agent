#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2023-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Verification script for sandbox AST gate functionality.
# Tests that benign code passes and hostile code is caught with correct violation types.

"""Sandbox verification: test that the sandbox correctly classifies code."""

import sys
from pathlib import Path

# Robust import handling
try:
    from signal_discovery_workflow.sandbox import classify_code_violations
except ImportError:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
    from signal_discovery_workflow.sandbox import classify_code_violations


def test_benign_modules():
    """Test that allowed modules classify clean."""
    # Code using only allowlisted imports and safe builtins
    benign_code = '''
import pandas as pd
import numpy as np
import math

def compute_signal(Close):
    returns = pd.DataFrame(np.log(Close / Close.shift(1)), columns=Close.columns)
    ma = Close.rolling(20).mean()
    signal = (Close > ma).astype(float)
    return signal
'''
    findings = classify_code_violations(benign_code)
    violations = [f for f in findings if f.get("type") not in ("syntax_error",)]

    if violations:
        print(f"FAIL: Benign code should have no violations, got: {violations}")
        return False
    print("PASS: Benign module code (pandas, numpy, math) classifies clean")
    return True


def test_forbidden_builtins():
    """Test that forbidden builtins are caught."""
    test_cases = [
        ("open", "open('/etc/passwd')", "forbidden_builtin_call"),
        ("__import__", "__import__('os')", "forbidden_builtin_call"),
        ("eval", "eval('1+1')", "forbidden_builtin_call"),
        ("exec", "exec('print(1)')", "forbidden_builtin_call"),
        ("getattr", "getattr(str, 'upper')", "forbidden_builtin_call"),
        ("globals", "globals()", "forbidden_builtin_call"),
        ("locals", "locals()", "forbidden_builtin_call"),
    ]

    all_passed = True
    for name, code, expected_type in test_cases:
        findings = classify_code_violations(code)
        if not findings:
            print(f"FAIL: {name} should be flagged as {expected_type}")
            all_passed = False
            continue
        violation_types = [f["type"] for f in findings]
        if expected_type not in violation_types:
            print(f"FAIL: {name} flagged with {violation_types}, expected {expected_type}")
            all_passed = False
        else:
            print(f"PASS: {name} correctly flagged as {expected_type}")

    return all_passed


def test_dangerous_imports():
    """Test that dangerous module imports are caught."""
    test_cases = [
        ("os", "import os", "disallowed_import"),
        ("subprocess", "import subprocess", "disallowed_import"),
        ("sys", "import sys", "disallowed_import"),
        ("socket", "import socket", "disallowed_import"),
        ("pathlib", "import pathlib", "disallowed_import"),
        ("ctypes", "import ctypes", "disallowed_import"),
        ("importlib", "import importlib", "disallowed_import"),
    ]

    all_passed = True
    for name, code, expected_type in test_cases:
        findings = classify_code_violations(code)
        if not findings:
            print(f"FAIL: {name} import should be flagged as {expected_type}")
            all_passed = False
            continue
        violation_types = [f["type"] for f in findings]
        if expected_type not in violation_types:
            print(f"FAIL: {name} import flagged with {violation_types}, expected {expected_type}")
            all_passed = False
        else:
            print(f"PASS: {name} import correctly flagged as {expected_type}")

    return all_passed


def test_dangerous_calls():
    """Test that dangerous module calls are caught."""
    test_cases = [
        ("os.system", "os.system('ls')", "dangerous_call"),
        ("subprocess.run", "subprocess.run(['ls'])", "dangerous_call"),
        ("sys.exit", "sys.exit(0)", "dangerous_call"),
    ]

    all_passed = True
    for name, code, expected_type in test_cases:
        findings = classify_code_violations(code)
        if not findings:
            print(f"FAIL: {name} call should be flagged as {expected_type}")
            all_passed = False
            continue
        violation_types = [f["type"] for f in findings]
        if expected_type not in violation_types:
            print(f"FAIL: {name} call flagged with {violation_types}, expected {expected_type}")
            all_passed = False
        else:
            print(f"PASS: {name} call correctly flagged as {expected_type}")

    return all_passed


def test_frame_introspection():
    """Test that frame introspection attributes are caught."""
    test_cases = [
        ("f_globals", "frame.f_globals", "frame_introspection"),
        ("f_back", "frame.f_back", "frame_introspection"),
        ("gi_frame", "gen.gi_frame", "frame_introspection"),
        ("tb_frame", "tb.tb_frame", "frame_introspection"),
        ("f_locals", "frame.f_locals", "frame_introspection"),
        ("f_builtins", "frame.f_builtins", "frame_introspection"),
        ("cr_frame", "coro.cr_frame", "frame_introspection"),
        ("ag_frame", "agen.ag_frame", "frame_introspection"),
    ]

    all_passed = True
    for name, code, expected_type in test_cases:
        findings = classify_code_violations(code)
        if not findings:
            print(f"FAIL: {name} access should be flagged as {expected_type}")
            all_passed = False
            continue
        violation_types = [f["type"] for f in findings]
        if expected_type not in violation_types:
            print(f"FAIL: {name} access flagged with {violation_types}, expected {expected_type}")
            all_passed = False
        else:
            print(f"PASS: {name} access correctly flagged as {expected_type}")

    return all_passed


def test_dunder_attributes():
    """Test that dunder attributes are caught."""
    test_cases = [
        ("__globals__", "fn.__globals__", "dunder_attribute"),
        ("__code__", "fn.__code__", "dunder_attribute"),
        ("__builtins__", "mod.__builtins__", "dunder_attribute"),
        ("__subclasses__", "cls.__subclasses__()", "dunder_attribute"),
        ("__mro__", "cls.__mro__", "dunder_attribute"),
    ]

    all_passed = True
    for name, code, expected_type in test_cases:
        findings = classify_code_violations(code)
        if not findings:
            print(f"FAIL: {name} access should be flagged as {expected_type}")
            all_passed = False
            continue
        violation_types = [f["type"] for f in findings]
        if expected_type not in violation_types:
            print(f"FAIL: {name} access flagged with {violation_types}, expected {expected_type}")
            all_passed = False
        else:
            print(f"PASS: {name} access correctly flagged as {expected_type}")

    return all_passed


def test_operator_catalogue_clean():
    """Test that a minimal 66-operator-like catalogue code classifies clean."""
    # Simulate code that might be generated with operator definitions
    operator_catalogue_code = '''
import pandas as pd
import numpy as np

def Add(x, y):
    return x + y

def Sub(x, y):
    return x - y

def RollMean(x, d=20):
    return x.rolling(d).mean()

def Delay(x, d=1):
    return x.shift(d)

def ZScore(x, d=20):
    return (x - x.rolling(d).mean()) / x.rolling(d).std()

def signal_strategy(Close):
    ma = RollMean(Close)
    z = ZScore(Close)
    return Close / ma + z
'''
    findings = classify_code_violations(operator_catalogue_code)
    violations = [f for f in findings if f.get("type") not in ("syntax_error",)]

    if violations:
        print(f"FAIL: Operator catalogue code should be clean, got: {violations}")
        return False
    print("PASS: 66-operator catalogue code (with Add, Sub, etc.) classifies clean")
    return True


def main():
    """Run all sandbox verification tests."""
    print("=" * 60)
    print("Sandbox Verification Tests")
    print("=" * 60)

    results = [
        test_benign_modules(),
        test_forbidden_builtins(),
        test_dangerous_imports(),
        test_dangerous_calls(),
        test_frame_introspection(),
        test_dunder_attributes(),
        test_operator_catalogue_clean(),
    ]

    print("=" * 60)
    if all(results):
        print("PASS: All sandbox verification tests passed")
        return 0
    else:
        print("FAIL: Some sandbox verification tests failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())
