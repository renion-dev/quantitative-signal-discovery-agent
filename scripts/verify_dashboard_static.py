#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2023-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# STATIC verification script for dashboard/index.html.
# Uses pure text analysis - does not import the dashboard module.
# Verifies esc/escJs helpers exist and template interpolations are safe.

"""Conservative static verification of dashboard/index.html.

This standalone CI tool uses stdlib-only text analysis and does not import the dashboard runtime,
use the network, or depend on dashboard data artifacts. It is not a complete JavaScript parser and
is not a sole XSS security guarantee. It detects detectable parse/bracket errors and unsafe
template-interpolation/direct-sink sites, but does not prove the absence of every JavaScript grammar
or semantic error. It is a guardrail against regressions, not a security guarantee, and is
deliberately conservative: it may flag unreachable or unused code.

Function-valued event-handler property assignments are accepted only when the right-hand side is
syntactically an arrow function, function expression, or bound method reference. A bare identifier
is deliberately rejected because it could refer to an untrusted string.

The following constructs are not currently detected:
* logical-assignment sinks such as ``el.innerHTML ||= x``;
* ``Object.assign(el, {innerHTML: x})``;
* computed member assignment with a template-literal key such as ``el[`innerHTML`] = x``;
* bracket-call sinks such as ``el['insertAdjacentHTML']('beforeend', x)``.
"""

import argparse
import re
import sys
from dataclasses import dataclass
from pathlib import Path


class TemplateScanError(ValueError):
    """Raised when the inline JavaScript cannot be scanned conservatively."""


@dataclass(frozen=True)
class HtmlSink:
    """A direct HTML sink and the expressions assigned to it."""

    name: str
    expressions: tuple[str, ...]
    line: int


REGEX_PREFIX_CHARS = frozenset("=(:,[!&|?{};")
REGEX_PREFIX_KEYWORDS = frozenset({"case", "delete", "in", "new", "of", "return", "throw", "typeof", "void"})

# Raw values are explicitly allow-listed only when the dashboard contract guarantees that they are
# numeric, booleans, counts, or locally generated static fragments. Strings rendered as HTML must
# use esc() (or escJs() in an inline event handler).
SAFE_RAW_INTERPOLATIONS = frozenset(
    {
        "i",
        "p.entry",
        "p.mark||'—'",
        "paper.last_pnl!==undefined?paper.last_pnl:'—'",
        "s.pos||0",
        "s.neg||0",
        "r.long/max*45",
        "r.short/max*45",
        "r.long",
        "r.short",
        "r.net",
        "r.opened",
        "r.closed",
        "r.total_pnl",
        "p.price||''",
        "p.change24h||''",
        "st.last_pnl!==undefined?st.last_pnl:'—'",
        "cl.length",
        "lp.price",
        "c.conf",
        "c.change24",
        "c.trend7",
        "c.sma20",
        "c.high20",
        "c.low20",
        "c.avgvol",
        "c.pos",
        "avail[k].count",
        "tot",
        "s.total_runs??0",
        "s.accepted_count??0",
        "s.overfit_count??0",
        "r.iterations.length",
        "c.rej",
        "c.app",
        "c.total",
        "b[0]",
        "b[1]",
        "r[0]",
        "n.iteration",
        "n.code_len||0",
    }
)

# These expressions produce a fixed CSS class, fixed text, or a value selected from fixed literals.
SAFE_STATIC_INTERPOLATIONS = frozenset(
    {
        "(paper.total_pnl||0)>=0?'text-green-400':'text-red-400'",
        "p.side==='long'?'text-green-400':'text-red-400'",
        "p.side==='long'?'bg-green-900/40 text-green-400':'bg-red-900/40 text-red-400'",
        "r.net>0?'text-green-400':r.net<0?'text-red-400':'text-gray-400'",
        "stateColor",
        "c.side==='long'?'text-green-400':'text-red-400'",
        "closedPnl>=0?'text-green-400':'text-red-400'",
        "lp.price>c.sma20?'text-green-400':'text-red-400'",
        "lp.price>c.sma20?'LONG':'SHORT'",
        "lp.price>c.sma20?'above':'below'",
        "lp.price>c.sma20?'momentum up, enter long':'mean-reversion down, enter short'",
        "c.change24>=0?'text-green-400':'text-red-400'",
        "c.trend7>=0?'text-green-400':'text-red-400'",
        "(prices[r.symbol]?.change24h||0)>=0?'text-green-400':'text-red-400'",
        "(c.pnl||0)>=0?'text-green-400':'text-red-400'",
        "(p.pnl||0)>=0?'text-green-400':'text-red-400'",
        "(s.overfit_count||0)>0?'#ff4444':'#6b7280'",
        "acc?'#00ff88':'#f59e0b'",
        "acc?'ACCEPTED':'BEST'",
        "_evoRun===r.run_id?'sel':''",
        "_evoSel===n.id?'sel':''",
        "kids.length>1?'evo-branch':''",
        "kids.length>1?'<div class=\"evo-link\" style=\"width:12px\"></div>':''",
        "n.code?'open':''",
        "r[2]?evoIcText(r[1]):'text-gray-200'",
        "chips||'<span class=\"text-gray-600 text-xs\">no portfolio metrics yet — appear after new runs</span>'",
        "prices[r.symbol]?prices[r.symbol].price.toFixed(4):'—'",
    }
)

# Local helpers are allow-listed only when their return values are numeric formatting, fixed CSS,
# or HTML fragments whose own dynamic values are checked elsewhere by this scanner.
SAFE_HELPER_INTERPOLATIONS = frozenset(
    {
        "fmtP(p.sl,p.entry)",
        "fmtP(p.tp,p.entry)",
        "fnum(b.mean_ic,4)",
        "fnum(n.ic_ir,4)",
        "fnum(n.mean_ic,4)",
        "fnum(n.p_value,4)",
        "fnum(n.t_stat,2)",
        "fnum(r[1],4)",
        "fnum(s.best_ic,4)",
        "evoIcText(b.mean_ic)",
        "evoIcText(n.mean_ic)",
        "evoIcText(s.best_ic)",
        "evoBadge(n)",
        "evoOverfitBadge(n)",
        "evoAcceptBadge(n)",
        "evoSkepticCard(n)",
        "evoSkepticBlock(n)",
        "evoNodeCard(n)",
        "evoRenderSubtree(c)",
        "evoRenderSubtree(k)",
        "parts.join(' · ')",
        "mrows",
        "extra",
        "cls",
    }
)

# The conditional form has one escaped output branch and one empty branch.
SAFE_ESCAPED_CONDITIONALS = frozenset({"s.best_signal?esc(s.best_signal):''"})

# Direct HTML sinks may only use these local renderers when their output is not escaped at the sink.
SAFE_SINK_HELPERS = frozenset({"evoRenderSubtree(root)"})

# Property assignments that render HTML or install executable content.
DIRECT_PROPERTY_SINKS = frozenset({"innerHTML", "outerHTML", "srcdoc"})

# window.innerHTML / window['innerHTML'] is not an element-rendering sink.
WINDOW_PROPERTY_SINKS = frozenset({"innerhtml", "outerhtml", "srcdoc"})

EVENT_HANDLER_ATTRIBUTES = frozenset(
    {
        "onabort",
        "onanimationcancel",
        "onanimationend",
        "onanimationiteration",
        "onanimationstart",
        "onbeforeinput",
        "onbeforeinputcapture",
        "onbeforetoggle",
        "onblur",
        "oncancel",
        "oncanplay",
        "oncanplaythrough",
        "onchange",
        "onclick",
        "onclose",
        "oncompositionend",
        "oncompositionstart",
        "oncompositionupdate",
        "oncontextmenu",
        "oncopy",
        "oncut",
        "ondblclick",
        "ondrag",
        "ondragend",
        "ondragenter",
        "ondragleave",
        "ondragover",
        "ondragstart",
        "ondrop",
        "ondurationchange",
        "onemptied",
        "onended",
        "onerror",
        "onfocus",
        "onfocusin",
        "onfocusout",
        "onformdata",
        "oninput",
        "oninvalid",
        "onkeydown",
        "onkeypress",
        "onkeyup",
        "onload",
        "onloadeddata",
        "onloadedmetadata",
        "onloadstart",
        "onmousedown",
        "onmouseenter",
        "onmouseleave",
        "onmousemove",
        "onmouseout",
        "onmouseover",
        "onmouseup",
        "onoffline",
        "ononline",
        "onpagehide",
        "onpageshow",
        "onpaste",
        "onpause",
        "onplay",
        "onplaying",
        "onpointercancel",
        "onpointerdown",
        "onpointerenter",
        "onpointerleave",
        "onpointermove",
        "onpointerout",
        "onpointerover",
        "onpointerrawupdate",
        "onpointerup",
        "onprogress",
        "onratechange",
        "onreset",
        "onresize",
        "onscroll",
        "onscrollend",
        "onsearch",
        "onsecuritypolicyviolation",
        "onseeked",
        "onseeking",
        "onselect",
        "onselectionchange",
        "onselectstart",
        "onslotchange",
        "onstalled",
        "onsubmit",
        "onsuspend",
        "ontimeupdate",
        "ontoggle",
        "ontouchcancel",
        "ontouchend",
        "ontouchmove",
        "ontouchstart",
        "ontransitioncancel",
        "ontransitionend",
        "ontransitionrun",
        "ontransitionstart",
        "onvolumechange",
        "onwaiting",
        "onwebkitanimationend",
        "onwebkitanimationiteration",
        "onwebkitanimationstart",
        "onwebkittransitionend",
        "onwheel",
    }
)

STRING_LITERAL = re.compile(r"^(?:'(?:[^'\\]|\\.)*'|\"(?:[^\"\\]|\\.)*\")$")
STATIC_STRING_VALUE = re.compile(r"^(?:'((?:[^'\\]|\\.)*)'|\"((?:[^\"\\]|\\.)*)\")$")
JS_IDENTIFIER = re.compile(r"^[A-Za-z_$][\w$]*$")
JS_MEMBER_TARGET = re.compile(
    r"^[A-Za-z_$][\w$]*"
    r"(?:(?:\s*\.\s*[A-Za-z_$][\w$]*)|(?:\s*\[\s*(?:'[^'\\]*'|\"[^\"\\]*\")\s*\]))*$"
)
FORMATTED_NUMBER = re.compile(r"^(?:Number\([^()]+\)|[^`]+)\.toFixed\(\d+\)$")
ESCAPED_CALL = re.compile(r"^(?:esc|escJs)\(")
SAFE_TEMPLATE_MAP = re.compile(
    r"^(?:\([^()]+\)|[A-Za-z_$][\w$]*(?:\.[A-Za-z_$][\w$]*)*)"
    r"\.map\([A-Za-z_$][\w$]*=>`.*`\)\.join\(['\"]{1,2}\)$",
    re.DOTALL,
)
SAFE_CONDITIONAL_TEMPLATE = re.compile(r"^[^?]+\?`.*`:(?:''|\"\"|'[^']*')$", re.DOTALL)
SAFE_CONDITIONAL_TEMPLATE_MAP = re.compile(
    r"^[^?]+\?[A-Za-z_$][\w$]*(?:\.[A-Za-z_$][\w$]*)*"
    r"\.map\([A-Za-z_$][\w$]*=>`.*`\)\.join\(['\"]{1,2}\):'[^']*'$",
    re.DOTALL,
)
SAFE_ESCAPED_MAP = re.compile(
    r"^(?:longs|shorts)\.map\(x=>'(?:[^'\\]|\\.)*'\+escJs\(x\)\+'(?:[^'\\]|\\.)*'"
    r"\+esc\(x\)\+'(?:[^'\\]|\\.)*'\)\.join\(''\)$"
)

REQUIRED_SKEPTIC_SITES = frozenset(
    {
        "esc(String((v&&v.name)||'unknown'))",
        "esc(evoSkepticReason(v))||'<span class=\"text-gray-600\">no reason given</span>'",
    }
)
REQUIRED_SIGNAL_CARD_SITES = frozenset(
    {
        "esc(d.selected_signal||s.file)",
        "esc(d.status||'—')",
        "esc(d.headline||'')",
    }
)
REQUIRED_NUMERIC_SITES = frozenset(
    {
        "(m.mean_ic??0).toFixed(4)",
        "(paper.total_pnl||0).toFixed(4)",
        "Number(metrics.mean_ic).toFixed(4)",
    }
)


def find_dashboard_index(dashboard_path=None):
    """Find dashboard/index.html relative to repo root, or use an explicit path."""
    if dashboard_path is not None:
        return Path(dashboard_path)
    script_dir = Path(__file__).resolve().parent
    repo_root = script_dir.parent
    return repo_root / "dashboard" / "index.html"


def _skip_quoted(text, index, quote):
    """Skip a JavaScript single- or double-quoted string."""
    index += 1
    while index < len(text):
        char = text[index]
        if char == "\\":
            index += 2
            continue
        if char == quote:
            return index + 1
        index += 1
    raise TemplateScanError(f"Unterminated {quote} string at offset {index}")


def _previous_non_space(text, index):
    """Return the previous non-whitespace character, or an empty string."""
    index -= 1
    while index >= 0 and text[index].isspace():
        index -= 1
    return text[index] if index >= 0 else ""


def _is_regex_start(text, index):
    """Conservatively identify a regex literal rather than a division operator."""
    previous = _previous_non_space(text, index)
    if previous in REGEX_PREFIX_CHARS:
        return True
    if previous == ">":
        before_arrow = _previous_non_space(text, index - 1)
        if before_arrow == "=":
            return True

    word_end = index - 1
    while word_end >= 0 and text[word_end].isspace():
        word_end -= 1
    word_start = word_end
    while word_start >= 0 and (text[word_start].isalpha() or text[word_start] == "_"):
        word_start -= 1
    keyword = text[word_start + 1 : word_end + 1]
    return keyword in REGEX_PREFIX_KEYWORDS


def _skip_regex_literal(text, index):
    """Skip a JavaScript regex literal, including a character class."""
    index += 1
    in_character_class = False
    while index < len(text):
        char = text[index]
        if char == "\\":
            index += 2
            continue
        if char == "[":
            in_character_class = True
        elif char == "]":
            in_character_class = False
        elif char == "/" and not in_character_class:
            index += 1
            while index < len(text) and text[index].isalpha():
                index += 1
            return index
        elif char in "\r\n":
            raise TemplateScanError(f"Newline in regex literal at offset {index}")
        index += 1
    raise TemplateScanError(f"Unterminated regex literal at offset {index}")


def _scan_template(text, index, interpolations, template_count, interpolation_ranges=None):
    """Scan one template literal and return the offset after its closing backtick."""
    template_start = index
    index += 1
    template_count[0] += 1
    while index < len(text):
        char = text[index]
        if char == "\\":
            index += 2
            continue
        if char == "`":
            return index + 1
        if char == "$" and index + 1 < len(text) and text[index + 1] == "{":
            expression, next_index = _scan_interpolation(
                text,
                index + 2,
                interpolations,
                template_count,
                interpolation_ranges,
            )
            line = text.count("\n", 0, index) + 1
            context = text[max(0, template_start - 256) : index]
            interpolations.append((line, expression, context))
            index = next_index
            continue
        index += 1
    raise TemplateScanError(f"Unterminated template literal at offset {template_start}")


def _scan_interpolation(text, index, interpolations, template_count, interpolation_ranges=None):
    """Scan a template interpolation and return its expression plus the next offset."""
    expression_start = index
    depth = 1
    while index < len(text):
        char = text[index]
        if char in "'\"":
            index = _skip_quoted(text, index, char)
            continue
        if char == "`":
            index = _scan_template(text, index, interpolations, template_count, interpolation_ranges)
            continue
        if char == "/" and index + 1 < len(text) and text[index + 1] == "/":
            newline = text.find("\n", index)
            index = len(text) if newline == -1 else newline
            continue
        if char == "/" and index + 1 < len(text) and text[index + 1] == "*":
            comment_end = text.find("*/", index + 2)
            if comment_end == -1:
                raise TemplateScanError(f"Unterminated block comment at offset {index}")
            index = comment_end + 2
            continue
        if char == "/" and _is_regex_start(text, index):
            # A slash in expression position is a regex literal. If it is malformed, failing is
            # required: silently treating it as division could skip executable template content.
            index = _skip_regex_literal(text, index)
            continue
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                if interpolation_ranges is not None:
                    interpolation_ranges.append((expression_start, index + 1))
                return text[expression_start:index].strip(), index + 1
        index += 1
    raise TemplateScanError(f"Unterminated interpolation at offset {expression_start}")


def _find_tag_end(content, start):
    """Return the offset of the closing > for a tag, respecting quoted attributes."""
    quote = None
    index = start
    while index < len(content):
        char = content[index]
        if quote:
            if char == quote:
                quote = None
        elif char in "'\"":
            quote = char
        elif char == ">":
            return index
        index += 1
    raise TemplateScanError("Unterminated script opening tag")


def _parse_script_attributes(attribute_text):
    """Parse script attributes with exact names so data-src/x-src are not treated as src."""
    attributes = {}
    index = 0
    while index < len(attribute_text):
        while index < len(attribute_text) and attribute_text[index].isspace():
            index += 1
        if index >= len(attribute_text) or attribute_text[index] == "/":
            break

        name_start = index
        while index < len(attribute_text) and not attribute_text[index].isspace() and attribute_text[index] not in "=/>":
            index += 1
        if index == name_start:
            raise TemplateScanError(f"Malformed script attribute near offset {index}")
        name = attribute_text[name_start:index].lower()

        while index < len(attribute_text) and attribute_text[index].isspace():
            index += 1
        value = None
        if index < len(attribute_text) and attribute_text[index] == "=":
            index += 1
            while index < len(attribute_text) and attribute_text[index].isspace():
                index += 1
            if index >= len(attribute_text):
                raise TemplateScanError(f"Missing value for script attribute {name!r}")
            if attribute_text[index] in "'\"":
                quote = attribute_text[index]
                index += 1
                value_start = index
                while index < len(attribute_text) and attribute_text[index] != quote:
                    index += 1
                if index >= len(attribute_text):
                    raise TemplateScanError(f"Unterminated value for script attribute {name!r}")
                value = attribute_text[value_start:index]
                index += 1
            else:
                value_start = index
                while index < len(attribute_text) and not attribute_text[index].isspace() and attribute_text[index] != ">":
                    index += 1
                value = attribute_text[value_start:index]

        if name in attributes and attributes[name] != value:
            raise TemplateScanError(f"Conflicting duplicate script attribute {name!r}")
        attributes[name] = value

    return attributes


def _html_comment_ranges(content):
    """Return HTML comment ranges, including an unterminated final comment."""
    ranges = []
    cursor = 0
    while True:
        start = content.find("<!--", cursor)
        if start == -1:
            break
        end = content.find("-->", start + 4)
        if end == -1:
            ranges.append((start, len(content)))
            break
        ranges.append((start, end + 3))
        cursor = end + 3
    return ranges


def _extract_inline_script_sources(content):
    """Return executable inline script bodies, excluding only proven external-only scripts."""
    sources = []
    cursor = 0
    comment_ranges = _html_comment_ranges(content)
    script_start = re.compile(r"<script(?=[\s/>])", re.IGNORECASE)
    script_end = re.compile(r"</script\s*>", re.IGNORECASE)

    while True:
        match = script_start.search(content, cursor)
        if match is None:
            break
        containing_comment = next(
            (
                (comment_start, comment_end)
                for comment_start, comment_end in comment_ranges
                if comment_start <= match.start() < comment_end
            ),
            None,
        )
        if containing_comment is not None:
            # Script-looking text inside an HTML comment is inert and must not be analysed.
            cursor = containing_comment[1]
            continue
        opening_end = _find_tag_end(content, match.end())
        attributes = _parse_script_attributes(content[match.end() : opening_end])
        closing = script_end.search(content, opening_end + 1)
        if closing is None:
            raise TemplateScanError(f"Unclosed <script> block at offset {match.start()}")

        body = content[opening_end + 1 : closing.start()]
        source_value = attributes.get("src")
        proven_external = isinstance(source_value, str) and bool(source_value.strip()) and not body.strip()
        if not proven_external:
            sources.append(body)
        cursor = closing.end()

    if not sources:
        raise TemplateScanError("No inline <script> block found in dashboard/index.html")
    return sources


def _scan_javascript_interpolations(source):
    """Scan JavaScript source for template interpolations."""
    interpolations = []
    template_count = [0]
    index = 0
    while index < len(source):
        char = source[index]
        if char in "'\"":
            index = _skip_quoted(source, index, char)
            continue
        if char == "`":
            index = _scan_template(source, index, interpolations, template_count)
            continue
        if char == "/" and index + 1 < len(source) and source[index + 1] == "/":
            newline = source.find("\n", index)
            index = len(source) if newline == -1 else newline
            continue
        if char == "/" and index + 1 < len(source) and source[index + 1] == "*":
            comment_end = source.find("*/", index + 2)
            if comment_end == -1:
                raise TemplateScanError(f"Unterminated block comment at offset {index}")
            index = comment_end + 2
            continue
        if char == "/" and _is_regex_start(source, index):
            # Do not swallow malformed regex literals. The scanner cannot prove such a construct is
            # benign, so it must fail closed rather than skip possible executable content.
            index = _skip_regex_literal(source, index)
            continue
        index += 1
    return interpolations, template_count[0]


def _validate_balanced_template(source, index):
    """Validate a template literal, including all nested interpolation expressions."""
    template_start = index
    index += 1
    while index < len(source):
        char = source[index]
        if char == "\\":
            index += 2
            continue
        if char == "`":
            return index + 1
        if char == "$" and index + 1 < len(source) and source[index + 1] == "{":
            index = _validate_balanced_interpolation(source, index + 2)
            continue
        index += 1
    raise TemplateScanError(f"Unterminated template literal at offset {template_start}")


def _validate_balanced_interpolation(source, index):
    """Validate the JavaScript expression inside a template interpolation."""
    pairs = {"(": ")", "[": "]", "{": "}"}
    expected_closers = ["}"]
    while index < len(source):
        char = source[index]
        if char in "'\"":
            index = _skip_quoted(source, index, char)
            continue
        if char == "`":
            index = _validate_balanced_template(source, index)
            continue
        if char == "/" and index + 1 < len(source) and source[index + 1] in "/*":
            index = _skip_comment(source, index)
            continue
        if char == "/" and _is_regex_start(source, index):
            index = _skip_regex_literal(source, index)
            continue
        if char in pairs:
            expected_closers.append(pairs[char])
        elif char in ")]}":
            if not expected_closers:
                raise TemplateScanError(f"Unexpected closing '{char}' at offset {index}")
            expected = expected_closers.pop()
            if expected != char:
                raise TemplateScanError(
                    f"Mismatched closing '{char}' at offset {index}; expected '{expected}'"
                )
            if not expected_closers:
                return index + 1
        index += 1
    raise TemplateScanError(f"Unclosed template interpolation at offset {index}")


def _validate_balanced_javascript(source):
    """Reject unbalanced or mismatched JavaScript brackets."""
    pairs = {"(": ")", "[": "]", "{": "}"}
    stack = []
    index = 0
    while index < len(source):
        char = source[index]
        if char in "'\"":
            index = _skip_quoted(source, index, char)
            continue
        if char == "`":
            index = _validate_balanced_template(source, index)
            continue
        if char == "/" and index + 1 < len(source) and source[index + 1] in "/*":
            index = _skip_comment(source, index)
            continue
        if char == "/" and _is_regex_start(source, index):
            index = _skip_regex_literal(source, index)
            continue
        if char in pairs:
            stack.append((char, index))
        elif char in ")]}":
            if not stack:
                raise TemplateScanError(f"Unexpected closing '{char}' at offset {index}")
            opening, opening_offset = stack.pop()
            if pairs[opening] != char:
                raise TemplateScanError(
                    f"Mismatched closing '{char}' at offset {index}; expected '{pairs[opening]}' "
                    f"for '{opening}' at offset {opening_offset}"
                )
        index += 1

    if stack:
        opening, opening_offset = stack[-1]
        raise TemplateScanError(f"Unclosed '{opening}' at offset {opening_offset}")


def scan_template_interpolations(content):
    """Return all inline-JavaScript template interpolations and the template count.

    The scanner is intentionally conservative: malformed JavaScript raises an error instead of
    silently skipping content. It handles strings, comments, regex literals, nested templates,
    and balanced interpolation braces without importing the dashboard or adding dependencies.
    """
    sources = _extract_inline_script_sources(content)
    for script_source in sources:
        _validate_balanced_javascript(script_source)
    source = "\n".join(sources)
    return _scan_javascript_interpolations(source)


def _matching_parenthesis(text, opening):
    """Return the closing parenthesis matching an opening parenthesis."""
    depth = 0
    quote = None
    index = opening
    while index < len(text):
        char = text[index]
        if quote:
            if char == "\\":
                index += 2
                continue
            if char == quote:
                quote = None
            index += 1
            continue
        if char in "'\"":
            quote = char
        elif char == "(":
            depth += 1
        elif char == ")":
            depth -= 1
            if depth == 0:
                return index
        index += 1
    raise TemplateScanError("Unbalanced parentheses")


def _is_escaped_expression(expression):
    """Return whether an expression is wholly escaped, optionally with a static fallback."""
    match = ESCAPED_CALL.match(expression)
    if not match:
        return False
    opening = expression.index("(", match.start())
    closing = _matching_parenthesis(expression, opening)
    trailing = expression[closing + 1 :].strip()
    if not trailing:
        return True
    return trailing.startswith("||") and bool(STRING_LITERAL.fullmatch(trailing[2:].strip()))


def _is_safe_interpolation(expression, context):
    """Return whether an interpolation is escaped or explicitly allow-listed."""
    if expression in SAFE_RAW_INTERPOLATIONS:
        return True
    if expression in SAFE_STATIC_INTERPOLATIONS:
        return True
    if expression in SAFE_HELPER_INTERPOLATIONS:
        return True
    if expression in SAFE_ESCAPED_CONDITIONALS:
        return True
    if _is_escaped_expression(expression):
        return True
    if FORMATTED_NUMBER.fullmatch(expression):
        return True
    if SAFE_TEMPLATE_MAP.fullmatch(expression):
        return True
    if SAFE_CONDITIONAL_TEMPLATE.fullmatch(expression):
        return True
    if SAFE_CONDITIONAL_TEMPLATE_MAP.fullmatch(expression):
        return True
    if SAFE_ESCAPED_MAP.fullmatch(expression):
        return True

    # r.symbol is text, but it is safe in the CSV template and must not become globally allow-listed.
    return expression == "r.symbol" and "symbol,long,short,net,price,change24h" in context


def _skip_comment(text, index):
    """Skip a JavaScript line or block comment."""
    if text[index : index + 2] == "//":
        newline = text.find("\n", index)
        return len(text) if newline == -1 else newline
    comment_end = text.find("*/", index + 2)
    if comment_end == -1:
        raise TemplateScanError(f"Unterminated block comment at offset {index}")
    return comment_end + 2


def _skip_template(text, index):
    """Skip a complete template literal, including nested interpolations."""
    return _scan_template(text, index, [], [0])


def _extract_balanced_region(text, opening):
    """Return the contents and end offset of a balanced bracket-delimited region."""
    pairs = {"(": ")", "[": "]", "{": "}"}
    expected = [pairs[text[opening]]]
    index = opening + 1
    while index < len(text):
        char = text[index]
        if char in "'\"":
            index = _skip_quoted(text, index, char)
            continue
        if char == "`":
            index = _skip_template(text, index)
            continue
        if char == "/" and index + 1 < len(text) and text[index + 1] in "/*":
            index = _skip_comment(text, index)
            continue
        if char == "/" and _is_regex_start(text, index):
            index = _skip_regex_literal(text, index)
            continue
        if char in pairs:
            expected.append(pairs[char])
        elif char in ")]}":
            if not expected or char != expected[-1]:
                raise TemplateScanError(f"Mismatched bracket at offset {index}")
            expected.pop()
            if not expected:
                return text[opening + 1 : index], index + 1
        index += 1
    raise TemplateScanError(f"Unbalanced bracket region at offset {opening}")


def _split_top_level(text, delimiter):
    """Split an expression at top-level commas, plus signs, or logical ORs."""
    parts = []
    start = 0
    stack = []
    index = 0
    while index < len(text):
        char = text[index]
        if char in "'\"":
            index = _skip_quoted(text, index, char)
            continue
        if char == "`":
            index = _skip_template(text, index)
            continue
        if char == "/" and index + 1 < len(text) and text[index + 1] in "/*":
            index = _skip_comment(text, index)
            continue
        if char == "/" and _is_regex_start(text, index):
            index = _skip_regex_literal(text, index)
            continue
        if char in "([{":
            stack.append(char)
        elif char in ")]}":
            if stack:
                stack.pop()
        elif not stack:
            if delimiter == "," and char == ",":
                parts.append(text[start:index].strip())
                start = index + 1
            elif delimiter == "+" and char == "+":
                parts.append(text[start:index].strip())
                start = index + 1
            elif delimiter == "||" and char == "|" and index + 1 < len(text) and text[index + 1] == "|":
                parts.append(text[start:index].strip())
                start = index + 2
                index += 1
        index += 1
    parts.append(text[start:].strip())
    return parts


def _find_top_level_arrow(text):
    """Return the offset of a top-level => in an arrow callback."""
    stack = []
    index = 0
    while index + 1 < len(text):
        char = text[index]
        if char in "'\"":
            index = _skip_quoted(text, index, char)
            continue
        if char == "`":
            index = _skip_template(text, index)
            continue
        if char == "/" and index + 1 < len(text) and text[index + 1] in "/*":
            index = _skip_comment(text, index)
            continue
        if char == "/" and _is_regex_start(text, index):
            index = _skip_regex_literal(text, index)
            continue
        if char in "([{":
            stack.append(char)
        elif char in ")]}":
            if stack:
                stack.pop()
        elif not stack and char == "=" and text[index + 1] == ">":
            return index
        index += 1
    return -1


def _extract_statement_expression(text, start, end=None):
    """Extract an expression through the next top-level semicolon."""
    stack = []
    index = start
    limit = len(text) if end is None else end
    while index < limit:
        char = text[index]
        if char in "'\"":
            index = _skip_quoted(text, index, char)
            continue
        if char == "`":
            index = _skip_template(text, index)
            continue
        if char == "/" and index + 1 < len(text) and text[index + 1] in "/*":
            index = _skip_comment(text, index)
            continue
        if char == "/" and _is_regex_start(text, index):
            index = _skip_regex_literal(text, index)
            continue
        if char in "([{":
            stack.append(char)
        elif char in ")]}":
            if stack:
                stack.pop()
            elif char == "}":
                break
        elif char == ";" and not stack:
            break
        index += 1
    return text[start:index].strip(), index


def _extract_return_expression(block):
    """Extract the expression returned by a callback block."""
    stack = []
    index = 0
    while index < len(block):
        char = block[index]
        if char in "'\"":
            index = _skip_quoted(block, index, char)
            continue
        if char == "`":
            index = _skip_template(block, index)
            continue
        if char == "/" and index + 1 < len(block) and block[index + 1] in "/*":
            index = _skip_comment(block, index)
            continue
        if char == "/" and _is_regex_start(block, index):
            index = _skip_regex_literal(block, index)
            continue
        if char in "([{":
            stack.append(char)
        elif char in ")]}":
            if stack:
                stack.pop()
        elif (
            not stack
            and block.startswith("return", index)
            and (index + 6 == len(block) or not (block[index + 6].isalnum() or block[index + 6] in "_$"))
        ):
            return _extract_statement_expression(block, index + 6)[0]
        index += 1
    return None


def _extract_call_arguments(text, opening):
    """Return call arguments and the offset after the closing parenthesis."""
    arguments_text, end = _extract_balanced_region(text, opening)
    if not arguments_text.strip():
        return [], end
    return _split_top_level(arguments_text, ","), end


def _template_expression_is_safe(expression):
    """Return whether every interpolation in a template expression is safe."""
    interpolations, template_count = _scan_javascript_interpolations(expression)
    if template_count < 1:
        return False
    return all(_is_safe_interpolation(item[1], item[2]) for item in interpolations)


def _map_chain_is_safe(expression):
    """Return whether a map/join chain renders only validated template output."""
    marker = ".map("
    marker_offset = expression.rfind(marker)
    if marker_offset < 0:
        return False

    opening = marker_offset + len(marker) - 1
    callback, end = _extract_balanced_region(expression, opening)
    if not re.fullmatch(r"\.join\(['\"]{1,2}\)", expression[end:].strip()):
        return False

    arrow = _find_top_level_arrow(callback)
    if arrow < 0:
        return False
    body = callback[arrow + 2 :].strip()
    if body.startswith("{"):
        block, _ = _extract_balanced_region(body, 0)
        result = _extract_return_expression(block)
        if result is None:
            return False
    else:
        result = body

    if result in SAFE_SINK_HELPERS:
        return True
    if _is_escaped_expression(result):
        return True
    return _template_expression_is_safe(result)


def _is_function_expression(expression):
    """Return whether an expression is syntactically a function expression."""
    candidate = expression.strip()
    if candidate.startswith("async") and (
        len(candidate) == len("async") or not (candidate[len("async")].isalnum() or candidate[len("async")] in "_$")
    ):
        candidate = candidate[len("async") :].lstrip()
        if not candidate.startswith("function"):
            return False

    if not candidate.startswith("function"):
        return False

    cursor = len("function")
    while cursor < len(candidate) and candidate[cursor].isspace():
        cursor += 1
    if cursor < len(candidate) and candidate[cursor] == "*":
        cursor += 1
        while cursor < len(candidate) and candidate[cursor].isspace():
            cursor += 1
    if cursor < len(candidate) and (candidate[cursor].isalpha() or candidate[cursor] in "_$"):
        while cursor < len(candidate) and (candidate[cursor].isalnum() or candidate[cursor] in "_$"):
            cursor += 1
        while cursor < len(candidate) and candidate[cursor].isspace():
            cursor += 1
    if cursor >= len(candidate) or candidate[cursor] != "(":
        return False

    try:
        _, parameters_end = _extract_balanced_region(candidate, cursor)
    except TemplateScanError:
        return False
    body_start = parameters_end
    while body_start < len(candidate) and candidate[body_start].isspace():
        body_start += 1
    if body_start >= len(candidate) or candidate[body_start] != "{":
        return False

    try:
        _, body_end = _extract_balanced_region(candidate, body_start)
    except TemplateScanError:
        return False
    return not candidate[body_end:].strip()


def _is_arrow_function_expression(expression):
    """Return whether an expression is syntactically an arrow function."""
    arrow = _find_top_level_arrow(expression)
    if arrow < 0:
        return False

    parameters = expression[:arrow].strip()
    if parameters.startswith("async") and (
        len(parameters) == len("async")
        or not (parameters[len("async")].isalnum() or parameters[len("async")] in "_$")
    ):
        parameters = parameters[len("async") :].lstrip()

    if parameters.startswith("("):
        try:
            _, parameters_end = _extract_balanced_region(parameters, 0)
        except TemplateScanError:
            return False
        if parameters[parameters_end:].strip():
            return False
    elif not JS_IDENTIFIER.fullmatch(parameters):
        return False

    return bool(expression[arrow + 2 :].strip())


def _is_bound_method_expression(expression):
    """Return whether an expression is a direct bound method reference."""
    marker = ".bind"
    marker_offset = expression.rfind(marker)
    if marker_offset < 0:
        return False

    target = expression[:marker_offset].strip()
    if not JS_MEMBER_TARGET.fullmatch(target):
        return False

    opening = marker_offset + len(marker)
    while opening < len(expression) and expression[opening].isspace():
        opening += 1
    if opening >= len(expression) or expression[opening] != "(":
        return False

    try:
        _, call_end = _extract_call_arguments(expression, opening)
    except TemplateScanError:
        return False
    return not expression[call_end:].strip()


def _unwrap_parenthesized_expression(expression):
    """Remove balanced outer parentheses when they enclose the whole expression."""
    candidate = expression.strip()
    while candidate.startswith("("):
        try:
            contents, end = _extract_balanced_region(candidate, 0)
        except TemplateScanError:
            break
        if candidate[end:].strip():
            break
        candidate = contents.strip()
    return candidate


def _is_function_value_expression(expression):
    """Return whether an expression is syntactically a function value."""
    expression = _unwrap_parenthesized_expression(expression)
    return (
        _is_function_expression(expression)
        or _is_arrow_function_expression(expression)
        or _is_bound_method_expression(expression)
    )


def _is_safe_event_handler_expression(expression):
    """Return whether an event-handler property receives a safe value."""
    expression = expression.strip()
    if _is_function_value_expression(expression):
        return True

    # Bare identifiers, template literals, concatenations, property accesses, and ordinary call
    # results remain rejected: an identifier can hold an untrusted string, and the other forms can
    # carry one. A bare identifier is deliberately conservative even if it currently names a function.
    if expression.startswith("`") and expression.endswith("`"):
        return False

    alternatives = _split_top_level(expression, "||")
    if len(alternatives) > 1:
        return all(_is_safe_event_handler_expression(item) for item in alternatives)

    operands = _split_top_level(expression, "+")
    if len(operands) > 1:
        return False

    if _is_escaped_expression(expression):
        return True
    if STRING_LITERAL.fullmatch(expression):
        return True
    return bool(re.fullmatch(r"(?:-?\d+(?:\.\d+)?|null|true|false)", expression))


def _is_safe_sink_expression(expression):
    """Return whether a direct HTML sink expression is static, escaped, or known-safe."""
    expression = expression.strip()
    if not expression:
        return False
    if expression in SAFE_SINK_HELPERS:
        return True
    if _is_escaped_expression(expression):
        return True
    if STRING_LITERAL.fullmatch(expression):
        return True
    if re.fullmatch(r"(?:-?\d+(?:\.\d+)?|null|true|false)", expression):
        return True

    alternatives = _split_top_level(expression, "||")
    if len(alternatives) > 1:
        return all(_is_safe_sink_expression(item) for item in alternatives)

    operands = _split_top_level(expression, "+")
    if len(operands) > 1:
        return all(_is_safe_sink_expression(item) for item in operands)

    if expression.startswith("`") and expression.endswith("`"):
        return _template_expression_is_safe(expression)

    return _map_chain_is_safe(expression)


def _is_safe_sink_expression_for_name(sink_name, expression):
    """Return whether a sink value is safe for the specific sink kind."""
    if sink_name.lower() in EVENT_HANDLER_ATTRIBUTES:
        return _is_safe_event_handler_expression(expression)
    return _is_safe_sink_expression(expression)


def _base_identifier_before(source, index):
    """Return the identifier immediately before an index, if any."""
    cursor = index - 1
    while cursor >= 0 and source[cursor].isspace():
        cursor -= 1
    if cursor < 0 or not (source[cursor].isalnum() or source[cursor] in "_$"):
        return ""
    end = cursor
    while cursor >= 0 and (source[cursor].isalnum() or source[cursor] in "_$"):
        cursor -= 1
    return source[cursor + 1 : end + 1]


def _base_identifier_before_dot(source, index):
    """Return the identifier before a dot immediately preceding an index."""
    cursor = index - 1
    while cursor >= 0 and source[cursor].isspace():
        cursor -= 1
    if cursor < 0 or source[cursor] != ".":
        return ""
    cursor -= 1
    while cursor >= 0 and source[cursor].isspace():
        cursor -= 1
    if cursor < 0 or not (source[cursor].isalnum() or source[cursor] in "_$"):
        return ""
    end = cursor
    while cursor >= 0 and (source[cursor].isalnum() or source[cursor] in "_$"):
        cursor -= 1
    return source[cursor + 1 : end + 1]


def _scan_bracket_property_sink(source, index, end, sinks):
    """Detect el['innerHTML'] / el['outerHTML'] / el['srcdoc'] assignments."""
    if index >= end or source[index] != "[":
        return None

    cursor = index + 1
    while cursor < end and source[cursor].isspace():
        cursor += 1
    if cursor >= end or source[cursor] not in "'\"":
        return None

    quote = source[cursor]
    value_start = cursor + 1
    cursor = value_start
    while cursor < end:
        if source[cursor] == "\\":
            cursor += 2
            continue
        if source[cursor] == quote:
            break
        cursor += 1
    if cursor >= end:
        raise TemplateScanError(f"Unterminated bracket property name at offset {index}")

    property_name = source[value_start:cursor].lower()
    canonical_names = {"innerhtml": "innerHTML", "outerhtml": "outerHTML", "srcdoc": "srcdoc"}
    canonical_name = canonical_names.get(property_name)
    if canonical_name is None and property_name in EVENT_HANDLER_ATTRIBUTES:
        canonical_name = property_name
    if canonical_name is None:
        return None

    cursor += 1
    while cursor < end and source[cursor].isspace():
        cursor += 1
    if cursor >= end or source[cursor] != "]":
        return None
    cursor += 1
    while cursor < end and source[cursor].isspace():
        cursor += 1

    if source.startswith("+=", cursor):
        expression_start = cursor + 2
    elif cursor < end and source[cursor] == "=" and not (
        cursor + 1 < end and source[cursor + 1] in "=><"
    ):
        expression_start = cursor + 1
    else:
        return None

    base = _base_identifier_before(source, index)
    if base.lower() == "window" and property_name in WINDOW_PROPERTY_SINKS:
        return None

    expression, _ = _extract_statement_expression(source, expression_start, end)
    line = source.count("\n", 0, index) + 1
    sinks.append(HtmlSink(canonical_name, (expression,), line))
    return expression_start


def _set_attribute_sink(arguments):
    """Return the dangerous setAttribute value and sink name, if applicable."""
    if len(arguments) < 2:
        return None, None

    attribute_expression = arguments[0].strip()
    static_match = STATIC_STRING_VALUE.fullmatch(attribute_expression)
    if static_match:
        attribute_name = (static_match.group(1) if static_match.group(1) is not None else static_match.group(2)).lower()
        if attribute_name == "innerhtml":
            # setAttribute('innerHTML', value) does not render HTML as an element property.
            return None, None
        if attribute_name in EVENT_HANDLER_ATTRIBUTES or attribute_name == "srcdoc":
            return arguments[1].strip(), f"setAttribute('{attribute_name}', ...)"
        return None, None

    # A dynamic attribute name could resolve to an event handler, so fail closed.
    return arguments[1].strip(), "setAttribute(dynamicAttribute, ...)"


def _scan_code_for_sinks(source, start, end, sinks, scanned_ranges):
    """Scan JavaScript code recursively for direct HTML sink assignments and calls."""
    index = start
    while index < end:
        char = source[index]
        if char == "[":
            expression_start = _scan_bracket_property_sink(source, index, end, sinks)
            if expression_start is not None:
                # Continue through the assigned expression so nested callback bodies are scanned.
                index = expression_start
            else:
                index += 1
            continue
        if char in "'\"":
            index = _skip_quoted(source, index, char)
            continue
        if char == "`":
            interpolation_ranges = []
            index = _scan_template(source, index, [], [0], interpolation_ranges)
            for range_start, range_end in sorted(interpolation_ranges, key=lambda item: item[0] - item[1]):
                if any(existing_start <= range_start and range_end <= existing_end for existing_start, existing_end in scanned_ranges):
                    continue
                scanned_ranges.add((range_start, range_end))
                _scan_code_for_sinks(source, range_start, range_end, sinks, scanned_ranges)
            continue
        if char == "/" and index + 1 < end and source[index + 1] in "/*":
            index = _skip_comment(source, index)
            continue
        if char == "/" and _is_regex_start(source, index):
            index = _skip_regex_literal(source, index)
            continue
        if char.isalpha() or char in "_$":
            name_start = index
            while index < end and (source[index].isalnum() or source[index] in "_$"):
                index += 1
            name = source[name_start:index]
            lookahead = index
            while lookahead < end and source[lookahead].isspace():
                lookahead += 1
            line = source.count("\n", 0, name_start) + 1

            property_sink_name = None
            if name in DIRECT_PROPERTY_SINKS:
                property_sink_name = name
            elif name.lower() in EVENT_HANDLER_ATTRIBUTES:
                property_sink_name = name.lower()

            if property_sink_name is not None and lookahead < end:
                base = _base_identifier_before_dot(source, name_start)
                if base.lower() == "window" and property_sink_name.lower() in WINDOW_PROPERTY_SINKS:
                    expression_start = None
                elif source.startswith("+=", lookahead):
                    expression_start = lookahead + 2
                elif source[lookahead] == "=" and not (
                    lookahead + 1 < end and source[lookahead + 1] in "=><"
                ):
                    expression_start = lookahead + 1
                else:
                    expression_start = None
                if expression_start is not None:
                    expression, _ = _extract_statement_expression(source, expression_start, end)
                    sinks.append(HtmlSink(property_sink_name, (expression,), line))
                    # Do not skip the expression: nested callbacks, templates, and sinks must be scanned.
                    index = expression_start
                    continue

            if name == "insertAdjacentHTML" and lookahead < end and source[lookahead] == "(":
                arguments, _ = _extract_call_arguments(source, lookahead)
                expression = arguments[1] if len(arguments) > 1 else "<missing content argument>"
                sinks.append(HtmlSink(name, (expression,), line))
                # Scan the arguments recursively instead of jumping past them.
                index = lookahead + 1
                continue

            if name == "setAttribute" and lookahead < end and source[lookahead] == "(":
                arguments, _ = _extract_call_arguments(source, lookahead)
                expression, sink_name = _set_attribute_sink(arguments)
                if sink_name is not None:
                    sinks.append(HtmlSink(sink_name, (expression,), line))
                # Scan all arguments recursively; a non-sink attribute may still contain a nested sink.
                index = lookahead + 1
                continue

            if name == "document" and lookahead + 1 < end and source[lookahead] == ".":
                write_offset = lookahead + 1
                while write_offset < end and source[write_offset].isspace():
                    write_offset += 1
                if source.startswith("write", write_offset):
                    opening = write_offset + len("write")
                    while opening < end and source[opening].isspace():
                        opening += 1
                    if opening < end and source[opening] == "(":
                        arguments, _ = _extract_call_arguments(source, opening)
                        expressions = tuple(arguments) if arguments else ("<missing content argument>",)
                        sinks.append(HtmlSink("document.write", expressions, line))
                        # Scan the arguments recursively instead of jumping past them.
                        index = opening + 1
                        continue
            continue
        index += 1


def scan_html_sinks(content):
    """Return direct HTML sinks found in executable inline JavaScript."""
    sources = _extract_inline_script_sources(content)
    for script_source in sources:
        _validate_balanced_javascript(script_source)
    source = "\n".join(sources)
    sinks = []
    _scan_code_for_sinks(source, 0, len(source), sinks, set())
    return sinks


def analyze_dashboard_content(content):
    """Return all unsafe template interpolation and direct HTML sink findings."""
    issues = []
    interpolations, _ = scan_template_interpolations(content)
    for line, expression, context in interpolations:
        if not _is_safe_interpolation(expression, context):
            issues.append(f"unsafe template interpolation at line {line}: {expression}")

    for sink in scan_html_sinks(content):
        for expression in sink.expressions:
            if not _is_safe_sink_expression_for_name(sink.name, expression):
                issues.append(f"unsafe HTML sink {sink.name} at line {sink.line}: {expression}")
    return issues


def _read_dashboard(dashboard_path=None):
    """Read dashboard/index.html or return an error string."""
    index_path = find_dashboard_index(dashboard_path)
    if not index_path.exists():
        return None, f"dashboard/index.html not found at {index_path}"
    return index_path.read_text(), None


def test_esc_function_exists(dashboard_path=None):
    """Test that the esc function exists in the dashboard."""
    content, error = _read_dashboard(dashboard_path)
    if error:
        print(f"FAIL: {error}")
        return False

    esc_pattern = r'function\s+esc\s*\('
    if not re.search(esc_pattern, content):
        print("FAIL: esc() helper function not found in dashboard")
        return False

    print("PASS: esc() helper function exists in dashboard")
    return True


def test_escjs_function_exists(dashboard_path=None):
    """Test that the escJs function exists in the dashboard."""
    content, error = _read_dashboard(dashboard_path)
    if error:
        print(f"FAIL: {error}")
        return False

    escjs_pattern = r'function\s+escJs\s*\('
    if not re.search(escjs_pattern, content):
        print("FAIL: escJs() helper function not found in dashboard")
        return False

    print("PASS: escJs() helper function exists in dashboard")
    return True


def test_esc_escjs_definitions(dashboard_path=None):
    """Test that both esc and escJs handle their documented dangerous characters."""
    content, error = _read_dashboard(dashboard_path)
    if error:
        print(f"FAIL: {error}")
        return False

    esc_match = re.search(r'function\s+esc\s*\(\s*[^)]+\s*\)\s*\{[^}]*return[^}]*\}', content, re.DOTALL)
    if not esc_match:
        print("FAIL: esc() function definition with return not found")
        return False

    esc_def = esc_match.group(0)
    required_entities = ["&", "<", ">", '"', "'"]
    for entity in required_entities:
        if entity not in esc_def:
            print(f"FAIL: esc() missing handling for {entity!r}")
            return False

    escjs_match = re.search(r'function\s+escJs\s*\([^)]+\)\s*\{[^}]*return[^}]*\}', content, re.DOTALL)
    if not escjs_match:
        print("FAIL: escJs() function definition with return not found")
        return False

    escjs_def = escjs_match.group(0)
    required_js_characters = ["\\", "'", '"', "`", "\\n", "\\r", "\\t", "<", ">", "&"]
    for character in required_js_characters:
        if character not in escjs_def:
            print(f"FAIL: escJs() missing handling for {character!r}")
            return False

    print("PASS: esc() and escJs() have proper implementations")
    return True


def test_skeptic_reviews_length_check(dashboard_path=None):
    """Test that legacy and evolution paths are guarded by skeptic review length checks."""
    content, error = _read_dashboard(dashboard_path)
    if error:
        print(f"FAIL: {error}")
        return False

    required_guards = (
        "!n.skeptic_reviews||!n.skeptic_reviews.length",
        "n.skeptic_reviews&&n.skeptic_reviews.length",
    )
    for guard in required_guards:
        if guard not in content:
            print(f"FAIL: skeptic_reviews length guard missing: {guard}")
            return False

    print("PASS: skeptic_reviews length check guards present")
    return True


def test_skeptic_values_esc_wrapped(dashboard_path=None):
    """Test the actual skeptic-derived interpolation sites use esc()."""
    content, error = _read_dashboard(dashboard_path)
    if error:
        print(f"FAIL: {error}")
        return False

    try:
        interpolations, _ = scan_template_interpolations(content)
    except TemplateScanError as exc:
        print(f"FAIL: could not verify skeptic interpolation sites: {exc}")
        return False

    expressions = {expression for _, expression, _ in interpolations}
    missing = REQUIRED_SKEPTIC_SITES - expressions
    if missing:
        print(f"FAIL: skeptic interpolation sites are not esc() wrapped: {sorted(missing)}")
        return False

    print("PASS: skeptic-derived interpolation sites are esc() wrapped")
    return True


def test_numeric_values_not_esc_wrapped(dashboard_path=None):
    """Test known numeric interpolation sites are deliberately not HTML-escaped."""
    content, error = _read_dashboard(dashboard_path)
    if error:
        print(f"FAIL: {error}")
        return False

    try:
        interpolations, _ = scan_template_interpolations(content)
    except TemplateScanError as exc:
        print(f"FAIL: could not verify numeric interpolation sites: {exc}")
        return False

    expressions = {expression for _, expression, _ in interpolations}
    missing = REQUIRED_NUMERIC_SITES - expressions
    if missing:
        print(f"FAIL: expected numeric interpolation sites missing: {sorted(missing)}")
        return False

    print("PASS: known numeric interpolation sites use native formatting")
    return True


def test_template_literal_safety(dashboard_path=None):
    """Fail loudly unless every template interpolation is escaped or allow-listed."""
    content, error = _read_dashboard(dashboard_path)
    if error:
        print(f"FAIL: {error}")
        return False

    try:
        interpolations, template_count = scan_template_interpolations(content)
    except TemplateScanError as exc:
        print(f"FAIL: could not verify template literal safety: {exc}")
        return False

    if template_count < 20 or len(interpolations) < 50:
        print(
            "FAIL: template literal scan found too few sites to verify "
            f"({template_count} templates, {len(interpolations)} interpolations)"
        )
        return False

    unsafe = []
    for line, expression, context in interpolations:
        if not _is_safe_interpolation(expression, context):
            unsafe.append((line, expression))

    if unsafe:
        for line, expression in unsafe:
            print(f"FAIL: unescaped or non-allow-listed interpolation at line {line}: {expression}")
        return False

    print(
        "PASS: template literal safety verified at every interpolation site "
        f"({len(interpolations)} interpolations across {template_count} templates)"
    )
    return True


def test_html_sink_safety(dashboard_path=None):
    """Fail loudly unless every direct HTML sink expression is escaped or known-safe."""
    content, error = _read_dashboard(dashboard_path)
    if error:
        print(f"FAIL: {error}")
        return False

    try:
        sinks = scan_html_sinks(content)
    except TemplateScanError as exc:
        print(f"FAIL: could not verify direct HTML sinks: {exc}")
        return False

    unsafe = []
    for sink in sinks:
        for expression in sink.expressions:
            if not _is_safe_sink_expression_for_name(sink.name, expression):
                unsafe.append((sink.name, sink.line, expression))

    if unsafe:
        for name, line, expression in unsafe:
            print(f"FAIL: unescaped or non-allow-listed HTML sink {name} at line {line}: {expression}")
        return False

    print(f"PASS: direct HTML sink safety verified ({len(sinks)} sinks)")
    return True


def test_detector_regressions():
    """Prove the scanner rejects reviewed false-negative and malformed-input mutations."""
    direct_sink = """
    <script>
    function esc(value) { return value; }
    const d = {headline: '<img src=x onerror=alert(1)>'};
    function refresh() {
      const el = {innerHTML: ''};
      el.innerHTML = d.headline;
    }
    </script>
    """
    issues = analyze_dashboard_content(direct_sink)
    if not any("HTML sink innerHTML" in issue and "d.headline" in issue for issue in issues):
        print(f"FAIL: direct innerHTML regression was not detected: {issues}")
        return False

    sink_matrix = (
        ("innerHTML", "el.innerHTML = d.value;"),
        ("outerHTML", "el.outerHTML = d.value;"),
        ("insertAdjacentHTML", "el.insertAdjacentHTML('beforeend', d.value);"),
        ("document.write", "document.write(d.value);"),
    )
    for sink_name, statement in sink_matrix:
        synthetic = f"<script>const d={{value:'unsafe'}}; {statement}</script>"
        issues = analyze_dashboard_content(synthetic)
        if not any(f"HTML sink {sink_name}" in issue and "d.value" in issue for issue in issues):
            print(f"FAIL: direct {sink_name} regression was not detected: {issues}")
            return False

    nested_sink = """
    <script>
    const data = {title: '<img src=x onerror=alert(1)>'};
    const el = {innerHTML: ''};
    el.innerHTML = [1].map(item => {
      el.innerHTML = data.title;
      return `<p>${esc(item)}</p>`;
    }).join('');
    </script>
    """
    issues = analyze_dashboard_content(nested_sink)
    if not any("HTML sink innerHTML" in issue and "data.title" in issue for issue in issues):
        print(f"FAIL: nested sink inside an outer sink callback was not detected: {issues}")
        return False

    nested_interpolation = """
    <script>
    const data = {title: '<img src=x onerror=alert(1)>'};
    const el = {innerHTML: ''};
    el.innerHTML = [1].map(item => {
      return `<p>${data.title}</p>`;
    }).join('');
    </script>
    """
    issues = analyze_dashboard_content(nested_interpolation)
    if not any("template interpolation" in issue and "data.title" in issue for issue in issues):
        print(f"FAIL: unsafe interpolation inside an outer sink callback was not detected: {issues}")
        return False

    bracket_sinks = (
        ("innerHTML", "el['innerHTML'] = data.title;"),
        ("outerHTML", 'el["outerHTML"] = data.title;'),
    )
    for sink_name, statement in bracket_sinks:
        synthetic = f"<script>const data={{title:'unsafe'}}; {statement}</script>"
        issues = analyze_dashboard_content(synthetic)
        if not any(f"HTML sink {sink_name}" in issue and "data.title" in issue for issue in issues):
            print(f"FAIL: bracket-notation {sink_name} regression was not detected: {issues}")
            return False

    for event_handler in ("onclick", "onerror", "onload", "onmouseover"):
        sink_name = f"setAttribute('{event_handler}', ...)"
        statement = f"el.setAttribute('{event_handler}', data.title);"
        synthetic = f"<script>const data={{title:'unsafe'}}; {statement}</script>"
        issues = analyze_dashboard_content(synthetic)
        if not any(f"HTML sink {sink_name}" in issue and "data.title" in issue for issue in issues):
            print(f"FAIL: {sink_name} regression was not detected: {issues}")
            return False

    extra_sinks = (
        ("srcdoc", "iframe.srcdoc = data.title;"),
        ("srcdoc", "iframe['srcdoc'] = data.title;"),
        ("onclick", "el.onclick = data.title;"),
        ("onerror", "el['onerror'] = data.title;"),
    )
    for sink_name, statement in extra_sinks:
        synthetic = f"<script>const data={{title:'unsafe'}}; {statement}</script>"
        issues = analyze_dashboard_content(synthetic)
        if not any(f"HTML sink {sink_name}" in issue and "data.title" in issue for issue in issues):
            print(f"FAIL: {sink_name} regression was not detected: {issues}")
            return False

    accepted_function_handlers = (
        "el.onclick = () => doSomething();",
        "el.onclick = function(){ doSomething(); };",
        "el.onclick = doSomething.bind(this);",
    )
    for statement in accepted_function_handlers:
        synthetic = f"<script>function doSomething() {{}} {statement}</script>"
        issues = analyze_dashboard_content(synthetic)
        if issues:
            print(f"FAIL: function-valued event handler was incorrectly rejected: {statement}: {issues}")
            return False

    rejected_event_handlers = (
        ("bare identifier", "el.onclick = doSomething;"),
        ("template literal", "el.onclick = `${data.title}`;"),
        ("concatenation", "el.onclick = 'handler:' + data.title;"),
        ("property access", "el.onclick = handlers.onclick;"),
        ("call result", "el.onclick = getHandler();"),
    )
    for description, statement in rejected_event_handlers:
        synthetic = f"<script>const data={{title:'unsafe'}}; {statement}</script>"
        issues = analyze_dashboard_content(synthetic)
        if not any("HTML sink onclick" in issue for issue in issues):
            print(f"FAIL: {description} event handler was not rejected: {issues}")
            return False

    excluded_sinks = """
    <script>
    const data = {title: 'not-rendered'};
    window['innerHTML'] = data.title;
    el.setAttribute('innerHTML', data.title);
    </script>
    """
    issues = analyze_dashboard_content(excluded_sinks)
    if issues:
        print(f"FAIL: known non-rendering sink forms were incorrectly flagged: {issues}")
        return False

    commented_script = """
    <!-- <script>el.innerHTML = data.title; function inert() {</script> -->
    <script>const safe = true;</script>
    """
    issues = analyze_dashboard_content(commented_script)
    if issues:
        print(f"FAIL: script inside an HTML comment was incorrectly analysed: {issues}")
        return False

    unbalanced_javascript = (
        "<script>function broken() { const value = {</script>",
        "<script>call(</script>",
        "<script>const values = [</script>",
        "<script>const value = `${call(}</script>`</script>",
        "<script>function broken() {</script><script>}</script>",
    )
    for malformed_javascript in unbalanced_javascript:
        try:
            analyze_dashboard_content(malformed_javascript)
        except TemplateScanError as exc:
            if not any(
                marker in str(exc)
                for marker in ("Unclosed", "Mismatched", "Unexpected closing")
            ):
                print(f"FAIL: unbalanced JavaScript failed for the wrong reason: {exc}")
                return False
        else:
            print(f"FAIL: unbalanced JavaScript was silently accepted: {malformed_javascript}")
            return False

    executable_data_src = """
    <script data-src="not-an-external-source.js">
    const d = {headline: '<img src=x onerror=alert(1)>'};
    const el = {innerHTML: ''};
    el.innerHTML = `${d.headline||''}`;
    </script>
    """
    sources = _extract_inline_script_sources(executable_data_src)
    if len(sources) != 1 or "d.headline" not in sources[0]:
        print(f"FAIL: executable data-src script was skipped: {sources}")
        return False
    issues = analyze_dashboard_content(executable_data_src)
    if not any("template interpolation" in issue for issue in issues):
        print(f"FAIL: data-src template regression was not detected: {issues}")
        return False
    if not any("HTML sink innerHTML" in issue for issue in issues):
        print(f"FAIL: data-src sink regression was not detected: {issues}")
        return False

    attribute_boundary = """
    <script data-srcset="one.js">const first = 1;</script>
    <script x-src="two.js">const second = 2;</script>
    <script src="external.js"></script>
    <script src="external-with-inline.js">const third = 3;</script>
    """
    sources = _extract_inline_script_sources(attribute_boundary)
    if (
        len(sources) != 3
        or "first" not in sources[0]
        or "second" not in sources[1]
        or "third" not in sources[2]
    ):
        print(f"FAIL: script src attribute boundary was not respected: {sources}")
        return False

    malformed_regex = "<script>const broken = /abc</script>"
    try:
        analyze_dashboard_content(malformed_regex)
    except TemplateScanError as exc:
        if "regex" not in str(exc).lower():
            print(f"FAIL: malformed regex failed for the wrong reason: {exc}")
            return False
    else:
        print("FAIL: malformed regex literal was silently accepted")
        return False

    print(
        "PASS: direct/nested/bracket/extra-sink/function-handler, executable-script, comment, balance, "
        "and regex regressions detected"
    )
    return True


def test_signal_card_esc_wrapped(dashboard_path=None):
    """Test the actual signal-card interpolation sites use esc()."""
    content, error = _read_dashboard(dashboard_path)
    if error:
        print(f"FAIL: {error}")
        return False

    try:
        interpolations, _ = scan_template_interpolations(content)
    except TemplateScanError as exc:
        print(f"FAIL: could not verify signal-card interpolation sites: {exc}")
        return False

    expressions = {expression for _, expression, _ in interpolations}
    missing = REQUIRED_SIGNAL_CARD_SITES - expressions
    if missing:
        print(f"FAIL: signal-card interpolation sites are not esc() wrapped: {sorted(missing)}")
        return False

    print("PASS: signal-card interpolation sites are esc() wrapped")
    return True


def main(argv=None):
    """Run all dashboard static verification tests."""
    parser = argparse.ArgumentParser(description="Statically verify dashboard HTML escaping")
    parser.add_argument(
        "--dashboard",
        type=Path,
        help="Optional dashboard HTML path (defaults to the repository dashboard/index.html)",
    )
    args = parser.parse_args(argv)
    dashboard_path = args.dashboard

    print("=" * 60)
    print("Dashboard Static Verification Tests")
    print("=" * 60)

    results = [
        test_detector_regressions(),
        test_esc_function_exists(dashboard_path),
        test_escjs_function_exists(dashboard_path),
        test_esc_escjs_definitions(dashboard_path),
        test_skeptic_reviews_length_check(dashboard_path),
        test_skeptic_values_esc_wrapped(dashboard_path),
        test_numeric_values_not_esc_wrapped(dashboard_path),
        test_template_literal_safety(dashboard_path),
        test_html_sink_safety(dashboard_path),
        test_signal_card_esc_wrapped(dashboard_path),
    ]

    print("=" * 60)
    if all(results):
        print("PASS: All dashboard static verification tests passed")
        return 0
    else:
        print("FAIL: Some dashboard static verification tests failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())
