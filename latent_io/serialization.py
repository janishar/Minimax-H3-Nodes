"""Serialisation helpers for MiniMax H3 AV latents.

H3's latent carries video and audio in a single custom ``NestedTensor`` container
class. It is not a ``torch.Tensor`` subclass and has no ``.contiguous()``, which
is why core ``SaveLatent`` raises AttributeError on it. It also cannot be rebuilt
with ``torch.nested.nested_tensor``, since that requires every component to share
a dimension count and the two streams are 5D and 4D.

``torch.save`` handles it: pickle records the class by reference and rebuilds it
on load from the same ComfyUI install.
"""

import torch

FORMAT = "h3latent"
VERSION = 1
EXT = ".h3latent"


def describe(samples):
    """Human-readable summary of a latent, for console logging."""
    inner = samples.get("samples") if isinstance(samples, dict) else samples
    name = type(inner).__name__

    if isinstance(inner, torch.Tensor):
        if getattr(inner, "is_nested", False):
            shapes = [tuple(p.shape) for p in inner.unbind()]
            return f"{name}(nested, parts={shapes})"
        return f"{name}{tuple(inner.shape)} {inner.dtype}"

    streams = [
        f"{key}{tuple(value.shape)}"
        for key, value in getattr(inner, "__dict__", {}).items()
        if isinstance(value, torch.Tensor)
    ]
    return f"{name}({', '.join(streams) if streams else 'opaque'})"


def to_cpu(obj):
    """Recursively move tensors to CPU so files are portable and compact."""
    if isinstance(obj, torch.Tensor):
        return obj.detach().cpu()

    if isinstance(obj, dict):
        return {key: to_cpu(value) for key, value in obj.items()}

    if isinstance(obj, (list, tuple)):
        moved = [to_cpu(value) for value in obj]
        return tuple(moved) if isinstance(obj, tuple) else moved

    if hasattr(obj, "__dict__"):
        for key, value in list(vars(obj).items()):
            try:
                setattr(obj, key, to_cpu(value))
            except Exception:
                pass  # read-only attribute; leave it where it is
        return obj

    return obj


def build_payload(samples):
    return {"format": FORMAT, "version": VERSION, "latent": to_cpu(samples)}


def read_payload(payload, source=""):
    if not isinstance(payload, dict) or payload.get("format") != FORMAT:
        raise ValueError(f"{source or 'file'} is not an H3 latent file")
    if payload.get("version", 0) > VERSION:
        raise ValueError(
            f"{source or 'file'} was written by a newer H3-LatentIO "
            f"(v{payload['version']}, this build reads v{VERSION})"
        )
    return payload["latent"]
