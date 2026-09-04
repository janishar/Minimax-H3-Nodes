"""H3GateSwitch: sample exactly one of N branches, chosen by index.

Same lazy-input mechanism as gate.py's H3Gate and gate_ab.py's H3GateAB (see
gate.py's module docstring for why lazy alone is not enough), generalized
from a fixed on/off or two-way choice to any number of branches:
``select`` picks a 1-based index, and ``check_lazy_status`` requests only
that index's input name -- every other branch's whole upstream chain never
runs.

The paired web/gate_switch.js extension grows a fresh empty "value_N" input
every time the last one gets a connection (and prunes back down to
MIN_INPUTS empty trailing slots on disconnect), mirroring
latent_io/combine_latents.py's paired JS -- see that file's module
docstring. This file only ever sees whichever value_N keys arrive connected,
via ``**kwargs``; MIN_INPUTS/MAX_INPUTS below must match the JS file's
constants of the same name.
"""

from .any_type import ANY

_UNSET = object()  # marks a lazy input ComfyUI never forwarded because nothing is wired to it

MIN_INPUTS = 2
MAX_INPUTS = 32


class H3GateSwitch:
    """Evaluate and pass through exactly one of N branches, chosen by select."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "select": ("INT", {"default": 1, "min": 1, "max": MAX_INPUTS}),
            },
            "optional": {
                "value_1": (ANY, {"lazy": True}),
                "value_2": (ANY, {"lazy": True}),
            },
        }

    RETURN_TYPES = (ANY,)
    RETURN_NAMES = ("value",)
    FUNCTION = "select_branch"
    CATEGORY = "latent/minimax_h3"
    DESCRIPTION = (
        "Pick one of any number of branches and evaluate only that one -- every "
        "unselected branch's upstream chain never runs. Connect the last value_N "
        "input to grow another, like H3CombineLatents' sockets. 'select' is a "
        "1-based index into however many are wired up. Raises if the selected "
        "branch isn't connected, rather than silently falling back to another one."
    )

    @staticmethod
    def _key(select):
        return f"value_{select}"

    def check_lazy_status(self, select, **kwargs):
        key = self._key(select)
        if key in kwargs and kwargs[key] is None:
            return [key]
        return []

    def select_branch(self, select, **kwargs):
        key = self._key(select)
        value = kwargs.get(key, _UNSET)
        if value is _UNSET or value is None:
            raise ValueError(
                f"H3GateSwitch: select={select} chooses '{key}', but its input is not connected"
            )
        print(f"[Minimax-H3-Nodes] H3GateSwitch: selected branch {key}")
        return (value,)
