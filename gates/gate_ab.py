"""H3GateAB: sample exactly one of two branches, never both.

Same lazy-input mechanism as gate.py's H3Gate (see its module docstring for
why lazy alone is not enough), applied to a two-way choice instead of an
on/off toggle: ``check_lazy_status`` requests only the selected side's input
name, so the unselected side's whole upstream chain is never run. There is
no ExecutionBlocker case here -- exactly one side is always chosen, so the
output is always that side's value, never a blocked branch.

Built for clip chaining: ``a`` is a fresh Ref2VA latent for a chain's first
clip, ``b`` is H3MotionContext's stitched output for every clip after it --
only one gets sampled per run.
"""

from .any_type import ANY

_UNSET = object()  # marks a lazy input ComfyUI never forwarded because nothing is wired to it


class H3GateAB:
    """Evaluate and pass through exactly one of two branches, chosen by use_a."""

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

    RETURN_TYPES = (ANY,)
    RETURN_NAMES = ("value",)
    FUNCTION = "select"
    CATEGORY = "latent/minimax_h3"
    DESCRIPTION = (
        "Pick one of two branches and evaluate only that one -- the unselected branch's "
        "upstream chain never runs. Built for clip chaining: 'a' a fresh Ref2VA latent "
        "for the first clip, 'b' a later clip's H3MotionContext output; only the selected "
        "one is ever sampled. Raises if the selected side isn't connected, rather than "
        "silently falling back to the other one."
    )

    def check_lazy_status(self, use_a, a=_UNSET, b=_UNSET):
        if use_a:
            if a is None:
                return ["a"]
            return []
        if b is None:
            return ["b"]
        return []

    def select(self, use_a, a=_UNSET, b=_UNSET):
        if use_a:
            if a is _UNSET or a is None:
                raise ValueError(
                    "H3GateAB: use_a selects branch 'a', but its input is not connected"
                )
            print("[Minimax-H3-Nodes] H3GateAB: selected branch a")
            return (a,)

        if b is _UNSET or b is None:
            raise ValueError(
                "H3GateAB: use_a selects branch 'b', but its input is not connected"
            )
        print("[Minimax-H3-Nodes] H3GateAB: selected branch b")
        return (b,)
