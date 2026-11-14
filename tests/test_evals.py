from pathlib import Path

from agent.evals import Case, evaluate, format_report, is_subsequence, load_cases, normalise
from agent.graph import build_graph
from agent.models import OfflinePolicyModel

from .fakes import ScriptedChatModel, ai, call

DATASET = Path(__file__).parent.parent / "evals" / "dataset.jsonl"


def test_is_subsequence():
    assert is_subsequence(["a", "c"], ["a", "b", "c"])
    assert is_subsequence([], ["a"])
    assert not is_subsequence(["c", "a"], ["a", "b", "c"])
    assert not is_subsequence(["a", "a"], ["a"])


def test_bundled_dataset_passes_end_to_end_with_real_retrieval(tools):
    results = evaluate(build_graph(OfflinePolicyModel(), tools), load_cases(DATASET))
    assert len(results) == 6
    assert all(r.passed for r in results), format_report(results)


def test_eval_catches_a_right_answer_reached_without_retrieval(tools):
    """The answer string is correct, but no tool was called and nothing is cited."""
    case = Case("sandbox", "Sandbox rate limit?", ["search_knowledge_base"], ["rate-limits#2"], ["60"])
    graph = build_graph(ScriptedChatModel(script=[ai("It is 60 requests per minute.")]), tools)
    (result,) = evaluate(graph, [case])
    assert result.answer and not result.tools and not result.citation and not result.passed
    assert "FAIL" in format_report([result])


def test_eval_catches_wrong_tool_order(tools):
    case = Case("math", "q", ["search_knowledge_base", "calculator"], [], ["9000"])
    model = ScriptedChatModel(
        script=[
            ai("", call("calculator", expression="600 * 15")),
            ai("", call("search_knowledge_base", query="production rate limit")),
            ai("9000 [rate-limits#3]"),
        ]
    )
    (result,) = evaluate(build_graph(model, tools), [case])
    assert result.answer and not result.tools


def test_answer_matching_ignores_number_formatting():
    assert "9000" in normalise("That is 9,000\u202frequests")
    assert "180 days" in normalise("up to\u202f180\u202fdays")
    assert normalise("versions 1,2") == "versions 1,2"  # not a thousands separator
