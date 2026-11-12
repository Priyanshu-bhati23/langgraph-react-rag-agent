import pytest

from agent.tools import safe_eval


@pytest.mark.parametrize(
    ("expression", "expected"),
    [("600 * 15", 9000), ("(2 + 3) * 4", 20), ("-7 // 2", -4), ("2 ** 10", 1024), ("10 / 4", 2.5), ("17 % 5", 2)],
)
def test_safe_eval_arithmetic(expression, expected):
    assert safe_eval(expression) == expected


@pytest.mark.parametrize(
    "expression",
    [
        "__import__('os').system('id')",
        "open('/etc/passwd').read()",
        "x + 1",
        "(1).__class__",
        "[1, 2][0]",
        "9 ** 9 ** 9",
        "True + 1",
        "1 +",
        "1" * 300,
    ],
)
def test_safe_eval_rejects_everything_else(expression):
    with pytest.raises(ValueError):
        safe_eval(expression)


def test_calculator_tool_formats_results_and_reports_errors(tools):
    calculator = next(t for t in tools if t.name == "calculator")
    assert calculator.invoke({"expression": "600 * 15"}) == "9000"
    assert calculator.invoke({"expression": "10 / 4"}) == "2.5"
    assert calculator.invoke({"expression": "1 / 0"}).startswith("error:")
    assert calculator.invoke({"expression": "import os"}).startswith("error:")


def test_search_tool_returns_passages_and_ids_as_artifact(tools):
    search = next(t for t in tools if t.name == "search_knowledge_base")
    message = search.invoke({"name": search.name, "args": {"query": "refund window"}, "id": "t1", "type": "tool_call"})
    assert "[refunds#1]" in message.content
    assert message.artifact[0] == "refunds#1"


def test_search_tool_says_when_nothing_matches(tools):
    search = next(t for t in tools if t.name == "search_knowledge_base")
    message = search.invoke({"name": search.name, "args": {"query": "zzzz"}, "id": "t2", "type": "tool_call"})
    assert message.content == "No relevant passages found." and message.artifact == []
