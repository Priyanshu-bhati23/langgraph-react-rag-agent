"""A chat model that replays a script, so graph behaviour can be asserted exactly."""

from __future__ import annotations

from typing import Any

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from pydantic import Field


def call(name: str, call_id: str = "call_1", **args: Any) -> dict:
    return {"name": name, "args": args, "id": call_id}


def ai(text: str = "", *tool_calls: dict) -> AIMessage:
    return AIMessage(text, tool_calls=list(tool_calls))


class ScriptedChatModel(BaseChatModel):
    script: list[AIMessage]
    repeat_last: bool = False
    seen: list[list[BaseMessage]] = Field(default_factory=list)
    bound_tool_names: list[str] = Field(default_factory=list)

    @property
    def _llm_type(self) -> str:
        return "scripted"

    def bind_tools(self, tools: Any, **kwargs: Any) -> ScriptedChatModel:
        self.bound_tool_names.extend(t.name for t in tools)
        return self

    def _generate(self, messages: list[BaseMessage], stop=None, run_manager=None, **kwargs: Any) -> ChatResult:
        self.seen.append(list(messages))
        turn = len(self.seen) - 1
        if turn >= len(self.script):
            if not self.repeat_last:
                raise AssertionError(f"model was called {turn + 1} times but the script has {len(self.script)} turns")
            turn = len(self.script) - 1
        # Fresh copy so each turn gets its own message id (LangGraph de-duplicates by id).
        template = self.script[turn]
        reply = AIMessage(
            template.content, tool_calls=[dict(c, id=f"{c['id']}_{len(self.seen)}") for c in template.tool_calls]
        )
        return ChatResult(generations=[ChatGeneration(message=reply)])
