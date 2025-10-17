from typing import List
import ast, operator as op
from duckduckgo_search import DDGS

# Calculator (safe eval)
_ALLOWED = {
    ast.Add: op.add, ast.Sub: op.sub, ast.Mult: op.mul,
    ast.Div: op.truediv, ast.Pow: op.pow, ast.USub: op.neg,
    ast.Mod: op.mod, ast.FloorDiv: op.floordiv
}

def _eval(node):
    if isinstance(node, ast.Num):
        return node.n
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    if isinstance(node, ast.BinOp):
        left = _eval(node.left)
        right = _eval(node.right)
        if type(node.op) not in _ALLOWED:
            raise ValueError("Operation not allowed")
        return _ALLOWED[type(node.op)](left, right)
    if isinstance(node, ast.UnaryOp):
        if type(node.op) not in _ALLOWED:
            raise ValueError("Unary operation not allowed")
        return _ALLOWED[type(node.op)](_eval(node.operand))
    raise ValueError("Unsupported expression")

def calculator(expr: str) -> str:
    try:
        node = ast.parse(expr, mode="eval").body
        result = _eval(node)
        return f"{expr} = {result}"
    except Exception as e:
        return f"Calculator error: {e}"

#  Web search (DuckDuckGo)
def web_search(query: str, max_results: int = 5) -> str:
    try:
        results = []
        with DDGS() as ddgs:
            for r in ddgs.text(query, max_results=max_results):
                results.append(f"- {r.get('title')}: {r.get('href')}\n  {r.get('body')}")
        if not results:
            return "No results."
        return "Top results:\n" + "\n".join(results)
    except Exception as e:
        return f"Search error: {e}"

# Registry
TOOLS = {
    "calculator": calculator,
    "web_search": web_search,
}

def list_tools() -> List[str]:
    return sorted(TOOLS.keys())

def run_tool(name: str, tool_input: str) -> str:
    func = TOOLS.get(name)
    if not func:
        return f"Unknown tool: {name}"
    return func(tool_input)
