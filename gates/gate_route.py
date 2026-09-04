"""H3GateRoute: run one of two output branches, from a single switch.

H3GateAB picks which of two *upstream* chains gets evaluated, and that is all
it can do -- it has one output, so whatever consumes it runs either way. That
is not enough when each branch ends in its own OUTPUT_NODE (a save node, a
preview, a video writer). ComfyUI builds its execution roots from every
OUTPUT_NODE in the prompt and walks backward from each one, so two save nodes
are two independent roots: both branches run, no matter what a gate placed
*after* them selects. Selecting between their results is a display choice,
not an execution one.

This node is the fork placed *before* both roots. It has two outputs, and it
combines both halves of gate.py's mechanism on a single switch:

- the unselected side's input is never requested (lazy), so nothing upstream
  of it runs;
- the unselected side's output is an ``ExecutionBlocker``, so the OUTPUT_NODE
  below it -- and everything between -- is skipped.

Two H3Gate nodes with opposite toggles do the same job; this exists so one
switch cannot be left half-flipped, running both branches or neither.
"""

from .any_type import ANY

try:
    from comfy_execution.graph import ExecutionBlocker
except ImportError:
    ExecutionBlocker = None

_UNSET = object()  # marks a lazy input ComfyUI never forwarded because nothing is wired to it


class H3GateRoute:
    """Pass one of two branches through to its own output; block the other one."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "use_a": ("BOOLEAN", {"default": True, "label_on": "a", "label_off": "b"}),
            },
            "optional": {
                "a": (ANY, {"lazy": True}),
                "b": (ANY, {"lazy": True}),
            },
        }

    RETURN_TYPES = (ANY, ANY)
    RETURN_NAMES = ("a", "b")
    FUNCTION = "route"
    CATEGORY = "latent/minimax_h3"
    DESCRIPTION = (
        "One switch, two whole pipelines: the selected side passes through to its own "
        "output and the other side is blocked at both ends -- its upstream chain is "
        "never evaluated and its downstream nodes are skipped. Use this when each "
        "branch ends in its own save/preview node, which ComfyUI would otherwise run "
        "unconditionally; H3GateAB cannot stop that, because it has only one output."
    )

    def check_lazy_status(self, use_a, a=_UNSET, b=_UNSET):
        if use_a:
            if a is None:
                return ["a"]
            return []
        if b is None:
            return ["b"]
        return []

    def route(self, use_a, a=_UNSET, b=_UNSET):
        selected, name = (a, "a") if use_a else (b, "b")

        if selected is _UNSET or selected is None:
            raise ValueError(
                f"H3GateRoute: use_a selects branch '{name}', but its input is not connected"
            )

        if ExecutionBlocker is None:
            raise RuntimeError(
                "H3GateRoute: this ComfyUI build has no comfy_execution.graph.ExecutionBlocker, "
                "so the unselected branch cannot be blocked. Mute that branch's output node "
                "instead (Ctrl+M), or update ComfyUI."
            )

        print(f"[Minimax-H3-Nodes] H3GateRoute: routing to branch {name}, other branch skipped")
        blocker = ExecutionBlocker(None)
        return (selected, blocker) if use_a else (blocker, selected)
