"""Chat model loading, plus a deterministic stand-in for offline runs."""

from __future__ import annotations

import os
import re
from typing import Any

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatResult

_MATH = re.compile(r"\d[\d,.]*(?:\s*[-+*/x×]\s*\d[\d,.]*)+")


def load_chat_model() -> BaseChatModel:
    """Pick the model from LLM_PROVIDER / LLM_MODEL. Falls back to the offline policy."""
    provider, model = os.getenv("LLM_PROVIDER", "").lower(), os.getenv("LLM_MODEL")
    if provider == "anthropic" and model:
        from langchain_anthropic import ChatAnthropic

        return ChatAnthropic(model=model, temperature=0, max_tokens=1024)
    if provider == "openai" and model:
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(model=model, temperature=0, base_url=os.getenv("OPENAI_BASE_URL"))
    return OfflinePolicyModel()


class OfflinePolicyModel(BaseChatModel):
    """NOT a language model. A fixed search-then-answer policy that lets the graph, the
    tools and the eval harness run end to end without an API key (demos and CI).

    Policy: search once with the user's question; if the question contains an arithmetic
    expression, calculate it; then answer by quoting the top two passages with their citations.
    """

    @property
    def _llm_type(self) -> str:
        return "offline-policy"

    def bind_tools(self, tools: Any, **kwargs: Any) -> OfflinePolicyModel:
        return self

    def _generate(self, messages: list[BaseMessage], stop=None, run_manager=None, **kwargs: Any) -> ChatResult:
        question = next(str(m.content) for m in messages if isinstance(m, HumanMessage))
        tool_results = [m for m in messages if isinstance(m, ToolMessage)]
        done = {m.name for m in tool_results}
        math = _MATH.search(question)

        if "search_knowledge_base" not in done:
            reply = AIMessage(
                "", tool_calls=[{"name": "search_knowledge_base", "args": {"query": question}, "id": "s1"}]
            )
        elif math and "calculator" not in done:
            expression = math.group(0).replace(",", "").replace("x", "*").replace("×", "*")
            reply = AIMessage("", tool_calls=[{"name": "calculator", "args": {"expression": expression}, "id": "c1"}])
        else:
            search = next(m for m in tool_results if m.name == "search_knowledge_base")
            ids = search.artifact or []
            if not ids:
                text = "The documentation does not cover this."
            else:
                # Quote the two best passages, each with its citation id.
                quoted = []
                for block in str(search.content).split("\n\n")[:2]:
                    tag, _, rest = block.partition("] ")
                    quoted.append(f"{rest.split(') ', 1)[-1]} {tag}]")
                text = " ".join(quoted)
            calc = next((m for m in tool_results if m.name == "calculator"), None)
            if calc is not None:
                text += f" Calculation result: {calc.content}."
            reply = AIMessage(text)
        return ChatResult(generations=[ChatGeneration(message=reply)])
