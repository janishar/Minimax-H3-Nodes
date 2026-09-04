"""H3Gate: remove a branch from the run, not just mute its output.

Two independent mechanisms are needed and neither works alone:

1. **Prune upstream.** ``value`` is declared lazy (``{"lazy": True}``).
   ComfyUI only evaluates a lazy input when ``check_lazy_status`` asks for it
   by name; returning ``[]`` here means nothing upstream of ``value`` is ever
   requested, let alone run. Lazy alone is not enough on its own -- it stops
   the upstream chain, but anything wired to this node's output still runs
   on whatever it receives.
2. **Prune downstream.** When disabled, this node returns a
   ``comfy_execution.graph.ExecutionBlocker`` instead of ``None``. The
   executor skips any node that receives an ``ExecutionBlocker`` as an input,
   rather than running it on a bare ``None`` (or erroring on the type
   mismatch). A blocker alone is not enough either -- without the lazy input,
   the expensive upstream branch would already have run before its result
   got thrown away here.

See gates/README.md for how this differs from ComfyUI's own bypass (Ctrl+B,
which still executes the upstream chain) and mute (Ctrl+M, which prunes one
node the user has to select by hand).
"""

from .any_type import ANY

try:
    from comfy_execution.graph import ExecutionBlocker
except ImportError:
    ExecutionBlocker = None

_UNSET = object()  # marks a lazy input ComfyUI never forwarded because nothing is wired to it


class H3Gate:
    """Pass a value through when enabled; remove its whole upstream branch when not."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "enabled": ("BOOLEAN", {"default": True, "label_on": "on", "label_off": "off"}),
            },
            "optional": {
                "value": (ANY, {"lazy": True}),
            },
        }

    RETURN_TYPES = (ANY,)
    RETURN_NAMES = ("value",)
    FUNCTION = "gate"
    CATEGORY = "latent/minimax_h3"
    DESCRIPTION = (
        "Cut a branch out of the graph rather than muting it by hand. Enabled: value "
        "passes through unchanged and nothing else about it is touched. Disabled: value "
        "is never evaluated -- everything upstream of it is skipped -- and this node "
        "returns an ExecutionBlocker so everything downstream is skipped too. Unlike "
        "bypass (Ctrl+B), which still executes the upstream chain."
    )

    def check_lazy_status(self, enabled, value=_UNSET):
        if enabled and value is None:
            return ["value"]
        return []

    def gate(self, enabled, value=_UNSET):
        if value is _UNSET:
            value = None

        if enabled:
            return (value,)

        print("[Minimax-H3-Nodes] H3Gate: disabled, branch skipped")

        if ExecutionBlocker is None:
            raise RuntimeError(
                "H3Gate: this ComfyUI build has no comfy_execution.graph.ExecutionBlocker, "
                "so a disabled gate cannot block the downstream branch. Mute the downstream "
                "output node instead (Ctrl+M), or update ComfyUI."
            )
        return (ExecutionBlocker(None),)
