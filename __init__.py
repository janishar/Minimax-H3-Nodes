"""ComfyUI-Minimax_H3

Core SaveLatent assumes the latent is a plain torch.Tensor and calls
.contiguous() on it. MiniMax H3's latent is a custom NestedTensor container
holding the video and audio streams together, so that call fails. These nodes
serialise the whole latent object instead, preserving both streams and any
extra keys.

video_extend/ chains MiniMax H3 clips in latent space: it slices the tail off
one clip's sampled latent, overwrites the front of the next clip's target
latent with it, and marks those positions as never-denoised so motion and
audio continue across the join with no VAE decode/re-encode in between.
"""

from .gates import H3Gate, H3GateAB, H3GateSwitch
from .latent_io import H3CombineLatents, H3LoadLatent, H3SaveLatent
from .video_extend import H3InspectLatent, H3MotionContext, H3SliceTail, H3TrimLeading

NODE_CLASS_MAPPINGS = {
    "H3SaveLatent": H3SaveLatent,
    "H3LoadLatent": H3LoadLatent,
    "H3CombineLatents": H3CombineLatents,
    "H3InspectLatent": H3InspectLatent,
    "H3SliceTail": H3SliceTail,
    "H3MotionContext": H3MotionContext,
    "H3TrimLeading": H3TrimLeading,
    "H3Gate": H3Gate,
    "H3GateAB": H3GateAB,
    "H3GateSwitch": H3GateSwitch,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "H3SaveLatent": "Save MiniMax H3 AV Latent",
    "H3LoadLatent": "Load MiniMax H3 AV Latent",
    "H3CombineLatents": "Combine MiniMax H3 AV Latents",
    "H3InspectLatent": "Inspect MiniMax H3 AV Latent",
    "H3SliceTail": "Slice MiniMax H3 AV Latent Tail",
    "H3MotionContext": "MiniMax H3 Motion Context",
    "H3TrimLeading": "Trim Leading Frames (MiniMax H3)",
    "H3Gate": "Gate (skip branch)",
    "H3GateAB": "Gate A/B (pick branch)",
    "H3GateSwitch": "Gate Switch (pick branch)",
}

WEB_DIRECTORY = "./web"

__version__ = "1.0.0"
__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY"]
