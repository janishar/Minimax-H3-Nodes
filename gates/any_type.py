"""Wildcard socket type shared by gate.py and gate_ab.py.

ComfyUI checks whether a link can connect by comparing the origin socket's
declared type against the destination socket's declared type with `!=`. A
string subclass whose `__ne__` always reports "not different" therefore
accepts a link from any type -- LATENT, IMAGE, MODEL, CONDITIONING, or
anything else -- without the gate nodes needing to enumerate them. This is
the standard wildcard idiom used across the ComfyUI ecosystem (predating
core's own `comfy.comfy_types.IO.ANY`), kept local here so this pack doesn't
depend on where that lives in a given ComfyUI version.
"""


class AnyType(str):
    """A str that compares equal to any other value for ComfyUI's type check."""

    def __ne__(self, other):
        return False


ANY = AnyType("*")
