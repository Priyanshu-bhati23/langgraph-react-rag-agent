"""Behavioural evaluation: did the agent call the right tools and stay grounded?

Each case checks four independent things, because an agent can get the right
answer for the wrong reason (no retrieval, or a made-up citation):
  tools     - the expected tools were called, in order
  citation  - the answer cites at least one of the passages that hold the fact
  answer    - the answer contains the expected strings
  grounded  - the graph's own grounding check passed
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

from .graph import AgentResult, run


@dataclass
class Case:
    id: str
    question: str
    expected_tools: list[str]
    must_cite: list[str]
    must_contain: list[str]


@dataclass
class CaseResult:
    case: Case
    result: AgentResult
    tools: bool
    citation: bool
    answer: bool

    @property
    def passed(self) -> bool:
        return self.tools and self.citation and self.answer and self.result.grounded


def load_cases(path: str | Path) -> list[Case]:
    lines = Path(path).read_text(encoding="utf-8").splitlines()
    return [Case(**json.loads(line)) for line in lines if line.strip()]


def normalise(text: str) -> str:
    """Lower-case, unify Unicode spaces, and drop thousands separators so "9,000" matches "9000"."""
    text = text.replace("\u202f", " ").replace("\u00a0", " ").lower()
    return re.sub(r"(?<=\d)[,\u2009 ](?=\d{3}\b)", "", text)


def is_subsequence(expected: list[str], actual: list[str]) -> bool:
    remaining = iter(actual)
    return all(name in remaining for name in expected)


def evaluate(graph, cases: list[Case]) -> list[CaseResult]:
    results = []
    for case in cases:
        result = run(graph, case.question)
        called = [name for name, _ in result.tool_calls]
        results.append(
            CaseResult(
                case=case,
                result=result,
                tools=is_subsequence(case.expected_tools, called),
                citation=not case.must_cite or bool(set(case.must_cite) & set(result.citations)),
                answer=all(normalise(s) in normalise(result.answer) for s in case.must_contain),
            )
        )
    return results


def format_report(results: list[CaseResult]) -> str:
    def mark(ok: bool) -> str:
        return "pass" if ok else "FAIL"

    lines = [f"{'case':<18}{'tools':<7}{'cite':<7}{'answer':<8}{'grounded':<10}steps"]
    for r in results:
        lines.append(
            f"{r.case.id:<18}{mark(r.tools):<7}{mark(r.citation):<7}{mark(r.answer):<8}"
            f"{mark(r.result.grounded):<10}{r.result.steps}"
        )
    passed = sum(r.passed for r in results)
    lines.append(f"\n{passed}/{len(results)} cases passed ({passed / max(len(results), 1):.0%})")
    return "\n".join(lines)
