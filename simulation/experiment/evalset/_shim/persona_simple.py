"""Stand-in for persona_simple, which sanitychecking2/trajectories.py imports from
the main simulation's agents/ directory. Only
STIGMA_TEMPLATE_BY_TIER is needed; it is read as a literal from the frozen
agent code (../agents/persona_simple.py) without importing that
file, whose own imports would collide with sanitychecking2's llm_client.
Verified 2026-10-11: all 4 intended_register strings of the paper's gold
reducing item are verbatim in this dict."""
import ast
from pathlib import Path

_SRC = Path(__file__).resolve().parents[2] / "agents" / "persona_simple.py"
for _node in ast.parse(_SRC.read_text(encoding="utf-8")).body:
    if isinstance(_node, ast.Assign) and any(getattr(t, "id", None) == "STIGMA_TEMPLATE_BY_TIER"
                                             for t in _node.targets):
        STIGMA_TEMPLATE_BY_TIER = ast.literal_eval(_node.value)
