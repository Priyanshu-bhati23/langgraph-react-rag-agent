"""The ReAct loop as an explicit LangGraph state machine.

    agent ──tool calls──▶ tools ──▶ agent ...
      │ final answer
      ▼
    grounding check ──ok──▶ END
      │ bad/missing citations (one repair attempt)
      └────────────────────▶ agent

Built by hand rather than with `create_react_agent` so the two things a
production agent needs are visible in the graph: a hard step budget and a
check that the answer only cites passages that were actually retrieved.
"""

from __future__ import annotations

import operator
import re
from dataclasses import dataclass, field
from typing import Annotated, Literal, TypedDict

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, AnyMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import BaseTool
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages

SYSTEM_PROMPT = """You are a support engineer answering questions about the product documentation.

Work in a loop of reasoning and tool use:
- For anything about the product, call `search_knowledge_base` first. Do not answer from memory.
- For any arithmetic, call `calculator`. Do not compute in your head.
- When you have enough information, give a short final answer.

Grounding rules for the final answer:
- Every factual claim taken from the documentation must be followed by the id of the passage
  it came from, in square brackets, exactly as shown in the search results, e.g. [rate-limits#2].
- Only cite ids that appeared in search results during this conversation.
- If the documentation does not contain the answer, say so plainly instead of guessing."""

# Square brackets, or the full-width 【】 that some models (e.g. gpt-oss) use for citations.
CITATION = re.compile(r"[\[\u3010]([a-z0-9][a-z0-9_-]*#\d+)[\]\u3011]")


class AgentState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]
    retrieved: Annotated[list[str], operator.add]  # passage ids returned by search so far
    steps: int  # model turns taken
    repairs: int  # grounding repair attempts used
    grounded: bool | None


@dataclass
class AgentResult:
    answer: str
    tool_calls: list[tuple[str, dict]]
    citations: list[str]
    retrieved: list[str]
    grounded: bool
    steps: int
    messages: list[AnyMessage] = field(repr=False, default_factory=list)


def grounding_problem(answer: str, retrieved: list[str]) -> str | None:
    """Return a description of why `answer` is not grounded, or None if it is."""
    cited = CITATION.findall(answer)
    unknown = sorted(set(cited) - set(retrieved))
    if unknown:
        return f"it cites passages that were never retrieved: {', '.join(unknown)}"
    if retrieved and not cited:
        return "it uses documentation search results but cites no passage id"
    return None


def build_graph(model: BaseChatModel, tools: list[BaseTool], max_steps: int = 6, max_repairs: int = 1):
    by_name = {t.name: t for t in tools}
    bound = model.bind_tools(tools)

    def agent(state: AgentState) -> dict:
        reply = bound.invoke([SystemMessage(SYSTEM_PROMPT), *state["messages"]])
        return {"messages": [reply], "steps": state["steps"] + 1}

    def run_tools(state: AgentState) -> dict:
        results, retrieved = [], []
        for call in state["messages"][-1].tool_calls:
            tool = by_name.get(call["name"])
            if tool is None:
                # Models do hallucinate tool names; tell it what exists instead of crashing.
                content = f"error: unknown tool {call['name']!r}. Available: {', '.join(by_name)}"
                results.append(ToolMessage(content, tool_call_id=call["id"], name=call["name"], status="error"))
                continue
            try:
                message = tool.invoke(call)
            except Exception as exc:  # bad arguments, tool bug - recoverable by the model
                message = ToolMessage(f"error: {exc}", tool_call_id=call["id"], name=call["name"], status="error")
            if isinstance(message.artifact, list):
                retrieved.extend(message.artifact)
            results.append(message)
        return {"messages": results, "retrieved": retrieved}

    def check_grounding(state: AgentState) -> dict:
        problem = grounding_problem(str(state["messages"][-1].content), state["retrieved"])
        if problem is None:
            return {"grounded": True}
        if state["repairs"] >= max_repairs:
            return {"grounded": False}
        allowed = ", ".join(dict.fromkeys(state["retrieved"])) or "none - search first"
        nudge = (
            f"Your answer was rejected because {problem}. Rewrite it so that every documentation "
            f"claim cites one of these ids: {allowed}."
        )
        return {"messages": [HumanMessage(nudge)], "repairs": state["repairs"] + 1, "grounded": None}

    def out_of_budget(state: AgentState) -> dict:
        text = f"I could not finish within the {max_steps}-step budget, so I am not giving an answer I cannot support."
        return {"messages": [AIMessage(text)], "grounded": False}

    def after_agent(state: AgentState) -> Literal["tools", "check", "budget"]:
        if not state["messages"][-1].tool_calls:
            return "check"
        return "budget" if state["steps"] >= max_steps else "tools"

    def after_check(state: AgentState) -> Literal["agent", "__end__"]:
        if state["grounded"] is None and state["steps"] < max_steps:
            return "agent"
        return END

    graph = StateGraph(AgentState)
    graph.add_node("agent", agent)
    graph.add_node("tools", run_tools)
    graph.add_node("check", check_grounding)
    graph.add_node("budget", out_of_budget)
    graph.add_edge(START, "agent")
    graph.add_conditional_edges("agent", after_agent, {"tools": "tools", "check": "check", "budget": "budget"})
    graph.add_edge("tools", "agent")
    graph.add_conditional_edges("check", after_check, {"agent": "agent", END: END})
    graph.add_edge("budget", END)
    return graph.compile()


def run(graph, question: str) -> AgentResult:
    initial = {"messages": [HumanMessage(question)], "retrieved": [], "steps": 0, "repairs": 0, "grounded": None}
    # Each model turn is at most 3 graph steps (agent, tools/check, repair); leave headroom.
    final = graph.invoke(initial, {"recursion_limit": 60})
    messages = final["messages"]
    answer = next((str(m.content) for m in reversed(messages) if isinstance(m, AIMessage) and not m.tool_calls), "")
    calls = [(c["name"], c["args"]) for m in messages if isinstance(m, AIMessage) for c in m.tool_calls]
    return AgentResult(
        answer=answer,
        tool_calls=calls,
        citations=list(dict.fromkeys(CITATION.findall(answer))),
        retrieved=list(dict.fromkeys(final["retrieved"])),
        grounded=bool(final["grounded"]),
        steps=final["steps"],
        messages=messages,
    )
