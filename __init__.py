"""ComfyUI-Minimax_H3

Core SaveLatent assumes the latent is a plain torch.Tensor and calls
.contiguous() on it. H3's latent is a custom NestedTensor container holding the
video and audio streams together, so that call fails. These nodes serialise the
whole latent object instead, preserving both streams and any extra keys.
"""

from .latent_io import H3LoadLatent, H3SaveLatent

NODE_CLASS_MAPPINGS = {
    "H3SaveLatent": H3SaveLatent,
    "H3LoadLatent": H3LoadLatent,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "H3SaveLatent": "Save H3 AV Latent",
    "H3LoadLatent": "Load H3 AV Latent",
}

__version__ = "1.0.0"
__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
