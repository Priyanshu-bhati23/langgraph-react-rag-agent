"""Behavioural tests for the ReAct graph: which tools get called, in what order,
and whether the final answer is grounded in what was retrieved."""

from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage

from agent.graph import build_graph, grounding_problem, run

from .fakes import ScriptedChatModel, ai, call


def agent(tools, *script, **kw):
    repeat_last = kw.pop("repeat_last", False)
    model = ScriptedChatModel(script=list(script), repeat_last=repeat_last)
    return model, build_graph(model, tools, **kw)


# --- expected tool calls -------------------------------------------------------------


def test_documentation_question_triggers_retrieval_then_a_cited_answer(tools):
    model, graph = agent(
        tools,
        ai("", call("search_knowledge_base", query="refund window")),
        ai("Payments can be refunded for up to 180 days after capture [refunds#1]."),
    )
    result = run(graph, "How long do I have to refund a payment?")

    assert result.tool_calls == [("search_knowledge_base", {"query": "refund window"})]
    assert result.citations == ["refunds#1"] and result.grounded
    assert result.steps == 2
    # The model's second turn really did see the retrieved passage.
    tool_message = next(m for m in model.seen[1] if isinstance(m, ToolMessage))
    assert "180 days" in tool_message.content


def test_multi_step_question_calls_search_then_calculator_in_order(tools):
    model, graph = agent(
        tools,
        ai("", call("search_knowledge_base", query="production rate limit")),
        ai("", call("calculator", expression="600 * 15")),
        ai("Production allows 600 requests per minute [rate-limits#3], so 9000 in 15 minutes."),
    )
    result = run(graph, "How many requests can a production key send in 15 minutes?")

    assert [name for name, _ in result.tool_calls] == ["search_knowledge_base", "calculator"]
    assert result.tool_calls[1][1] == {"expression": "600 * 15"}
    assert "9000" in next(m.content for m in model.seen[2] if isinstance(m, ToolMessage) and m.name == "calculator")
    assert result.grounded


def test_parallel_tool_calls_in_one_turn_are_all_executed(tools):
    model, graph = agent(
        tools,
        ai(
            "",
            call("search_knowledge_base", "a", query="sandbox rate limit"),
            call("calculator", "b", expression="60 * 60"),
        ),
        ai("Sandbox keys get 60 requests per minute [rate-limits#2], which is 3600 per hour."),
    )
    result = run(graph, "Sandbox requests per hour?")
    replies = [m for m in model.seen[1] if isinstance(m, ToolMessage)]
    assert [m.name for m in replies] == ["search_knowledge_base", "calculator"]
    assert replies[1].content == "3600" and result.grounded


def test_pure_arithmetic_needs_no_retrieval_and_no_citation(tools):
    _, graph = agent(tools, ai("", call("calculator", expression="2 ** 10")), ai("That is 1024."))
    result = run(graph, "What is 2 to the power of 10?")
    assert [n for n, _ in result.tool_calls] == ["calculator"]
    assert result.retrieved == [] and result.citations == [] and result.grounded


def test_system_prompt_and_tools_are_supplied_on_every_turn(tools):
    model, graph = agent(tools, ai("", call("calculator", expression="1 + 1")), ai("2"))
    run(graph, "1 + 1?")
    assert model.bound_tool_names == ["search_knowledge_base", "calculator"]
    assert all(isinstance(turn[0], SystemMessage) for turn in model.seen)
    assert "search_knowledge_base" in model.seen[0][0].content


# --- grounded answers ----------------------------------------------------------------


def test_fabricated_citation_is_rejected_and_repaired(tools):
    model, graph = agent(
        tools,
        ai("", call("search_knowledge_base", query="refund fees")),
        ai("Fees are returned on refund [pricing#9]."),
        ai("The processing fee is not returned when a payment is refunded [refunds#3]."),
    )
    result = run(graph, "Do I get the fee back on a refund?")

    nudge = next(m for m in model.seen[2] if isinstance(m, HumanMessage) and "rejected" in m.content)
    assert "pricing#9" in nudge.content and "refunds#3" in nudge.content
    assert result.answer.endswith("[refunds#3].") and result.grounded
    assert result.citations == ["refunds#3"]


def test_answer_without_citation_after_retrieval_is_sent_back(tools):
    model, graph = agent(
        tools,
        ai("", call("search_knowledge_base", query="webhook retries")),
        ai("Deliveries are retried 8 times."),
        ai("Failed deliveries are retried up to 8 times over 72 hours [webhooks#1]."),
    )
    result = run(graph, "How often are webhooks retried?")
    assert len(model.seen) == 3 and result.grounded and result.citations == ["webhooks#1"]


def test_answer_that_stays_ungrounded_is_flagged_not_hidden(tools):
    _, graph = agent(
        tools,
        ai("", call("search_knowledge_base", query="refund fees")),
        ai("Fees are returned [pricing#9]."),
        ai("Trust me, fees are returned [pricing#9]."),
    )
    result = run(graph, "Do I get the fee back?")
    assert result.grounded is False
    assert result.answer.startswith("Trust me")  # caller gets the answer and the flag, and decides


def test_grounding_problem_rules():
    assert grounding_problem("60 per minute [rate-limits#2].", ["rate-limits#2"]) is None
    assert grounding_problem("It is 1024.", []) is None
    assert grounding_problem("60 per minute\u3010rate-limits#2\u3011.", ["rate-limits#2"]) is None  # gpt-oss style
    assert "never retrieved" in grounding_problem("See [made-up#1].", ["rate-limits#2"])
    assert "cites no passage" in grounding_problem("60 per minute.", ["rate-limits#2"])


# --- failure handling ----------------------------------------------------------------


def test_unknown_tool_name_returns_an_error_the_model_can_recover_from(tools):
    model, graph = agent(
        tools,
        ai("", call("web_search", query="lumen pay refunds")),
        ai("", call("search_knowledge_base", query="refund window")),
        ai("Up to 180 days [refunds#1]."),
    )
    result = run(graph, "Refund window?")
    error = next(m for m in model.seen[1] if isinstance(m, ToolMessage))
    assert error.status == "error" and "unknown tool 'web_search'" in error.content
    assert "search_knowledge_base" in error.content  # tells the model what does exist
    assert result.grounded


def test_bad_tool_arguments_do_not_crash_the_run(tools):
    model, graph = agent(
        tools,
        ai("", call("calculator", formula="2+2")),  # wrong argument name
        ai("", call("calculator", expression="2 + 2")),
        ai("4"),
    )
    result = run(graph, "2 + 2?")
    assert next(m for m in model.seen[1] if isinstance(m, ToolMessage)).status == "error"
    assert result.answer == "4"


def test_step_budget_stops_a_model_that_never_answers(tools):
    model, graph = agent(tools, ai("", call("search_knowledge_base", query="refund")), repeat_last=True, max_steps=4)
    result = run(graph, "Loop forever")
    assert len(model.seen) == 4 and result.steps == 4
    assert "4-step budget" in result.answer and result.grounded is False
    assert len(result.tool_calls) == 4  # the fourth request was refused, not executed
    assert sum(isinstance(m, ToolMessage) for m in result.messages) == 3
