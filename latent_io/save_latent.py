import os

import torch

import folder_paths

from .serialization import EXT, build_payload, describe


class H3SaveLatent:
    """Write a MiniMax H3 AV latent (video + audio) to disk."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "samples": ("LATENT",),
                "filename_prefix": ("STRING", {"default": "latent/minimax_h3"}),
            }
        }

    RETURN_TYPES = ()
    FUNCTION = "save"
    OUTPUT_NODE = True
    CATEGORY = "latent/minimax_h3"
    DESCRIPTION = (
        "Save a MiniMax H3 AV latent, keeping the video and audio streams intact. "
        "Use instead of core SaveLatent, which cannot handle MiniMax H3's nested latent."
    )

    def save(self, samples, filename_prefix):
        output_dir = folder_paths.get_output_directory()
        full_dir, base, counter, subfolder, _ = folder_paths.get_save_image_path(
            filename_prefix, output_dir
        )
        os.makedirs(full_dir, exist_ok=True)

        filename = f"{base}_{counter:05}_{EXT}"
        path = os.path.join(full_dir, filename)

        torch.save(build_payload(samples), path)

        megabytes = os.path.getsize(path) / 1e6
        print(
            f"[MiniMax H3 LatentIO] saved {path} ({megabytes:.2f} MB) {describe(samples)}"
        )
        return {"ui": {"text": [os.path.join(subfolder, filename)]}}
