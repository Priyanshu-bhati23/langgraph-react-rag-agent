"""The tools the agent may call."""

from __future__ import annotations

import ast
import operator
from collections.abc import Callable

from langchain_core.tools import BaseTool, tool

from .index import VectorIndex

_BINARY: dict[type, Callable] = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}
_UNARY: dict[type, Callable] = {ast.USub: operator.neg, ast.UAdd: operator.pos}


def safe_eval(expression: str) -> float:
    """Evaluate arithmetic by walking the AST. No names, calls or attributes,
    so model-written input can never reach `eval`."""
    if len(expression) > 200:
        raise ValueError("expression too long")

    def walk(node: ast.AST) -> float:
        if isinstance(node, ast.Expression):
            return walk(node.body)
        if isinstance(node, ast.Constant) and isinstance(node.value, int | float) and not isinstance(node.value, bool):
            return node.value
        if isinstance(node, ast.BinOp) and type(node.op) in _BINARY:
            left, right = walk(node.left), walk(node.right)
            if isinstance(node.op, ast.Pow) and (abs(right) > 100 or abs(left) > 1e6):
                raise ValueError("exponent too large")
            return _BINARY[type(node.op)](left, right)
        if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARY:
            return _UNARY[type(node.op)](walk(node.operand))
        raise ValueError("only numbers and + - * / // % ** ( ) are allowed")

    try:
        tree = ast.parse(expression.strip(), mode="eval")
    except SyntaxError as exc:
        raise ValueError(f"not a valid arithmetic expression: {expression!r}") from exc
    return walk(tree)


def make_tools(index: VectorIndex, k: int = 4) -> list[BaseTool]:
    @tool(response_format="content_and_artifact")
    def search_knowledge_base(query: str) -> tuple[str, list[str]]:
        """Search the documentation for passages relevant to `query`.

        Use this for any question about the product, its limits, pricing or
        behaviour. Returns passages prefixed with a citation id like [rate-limits#2].
        """
        hits = index.search(query, k=k)
        if not hits:
            return "No relevant passages found.", []
        body = "\n\n".join(f"[{h.chunk.id}] ({h.chunk.heading}) {h.chunk.text}" for h in hits)
        # The artifact carries the ids actually retrieved; the grounding check reads it.
        return body, [h.chunk.id for h in hits]

    @tool
    def calculator(expression: str) -> str:
        """Evaluate an arithmetic expression, e.g. "120 * 60 * 8". Use it for every calculation."""
        try:
            result = safe_eval(expression)
        except (ValueError, ZeroDivisionError, OverflowError) as exc:
            return f"error: {exc}"
        return str(int(result)) if float(result).is_integer() else f"{result:.6g}"

    return [search_knowledge_base, calculator]
