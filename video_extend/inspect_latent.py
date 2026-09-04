"""H3InspectLatent: structural dump of a MiniMax H3 AV latent.

Exact-continuation chaining (H3MotionContext) needs certainty about the
container's attribute names, each stream's tensor rank, and which axis is
time -- MiniMax H3's latent is a comfy.nested_tensor.NestedTensor, not a
torch.Tensor and not torch.nested, and its only attribute is a plain
``.tensors`` list with no per-stream names. This node exists so those facts
get read off a real latent and eyeballed before anything downstream assumes
them.
"""

from ..latent_io.serialization import tensor_fields
from .common import audio_latent_t, video_latent_t


class H3InspectLatent:
    """Pass-through structural dump of a LATENT, for MiniMax H3 development."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "samples": ("LATENT",),
                "video_frames": (
                    "INT",
                    {
                        "default": 39,
                        "min": 1,
                        "max": 100000,
                        "tooltip": "Pixel frame count (24 fps) to test each tensor axis against as a candidate time axis.",
                    },
                ),
            }
        }

    RETURN_TYPES = ("LATENT",)
    FUNCTION = "inspect"
    CATEGORY = "latent/minimax_h3"
    DESCRIPTION = (
        "Print a full structural dump of a MiniMax H3 AV latent -- container class, tensor "
        "fields with shape/dtype/device, and candidate time axes -- and pass it through "
        "unchanged. Run this before wiring H3SliceTail/H3MotionContext: their slicing assumes "
        "facts about the latent's layout that are not documented anywhere else."
    )

    def inspect(self, samples, video_frames):
        inner = samples.get("samples") if isinstance(samples, dict) else samples
        lines = [f"[Minimax-H3-Nodes] H3InspectLatent: {type(inner).__module__}.{type(inner).__name__}"]

        slots = getattr(type(inner), "__slots__", None)
        lines.append(f"  __slots__: {slots!r}" if slots else "  __slots__: (none -- uses __dict__)")
        lines.append(f"  is_nested: {getattr(inner, 'is_nested', '(no such attribute)')}")

        fields = list(tensor_fields(inner))
        if not fields:
            lines.append("  no tensor fields found")

        expected_video_t = video_latent_t(video_frames)
        expected_audio_t = audio_latent_t(video_frames)
        lines.append(
            f"  candidate time axes for video_frames={video_frames}: "
            f"video_latent_t={expected_video_t}, audio_latent_t={expected_audio_t}"
        )

        for label, tensor in fields:
            lines.append(f"  {label}: shape={tuple(tensor.shape)} dtype={tensor.dtype} device={tensor.device}")
            for axis, size in enumerate(tensor.shape):
                if size == expected_video_t:
                    lines.append(f"    axis {axis} (size {size}) matches expected video_latent_t")
                if size == expected_audio_t:
                    lines.append(f"    axis {axis} (size {size}) matches expected audio_latent_t")

        if isinstance(samples, dict):
            other_keys = [k for k in samples.keys() if k != "samples"]
            lines.append(f"  other LATENT dict keys: {other_keys}")

        print("\n".join(lines))
        return (samples,)
