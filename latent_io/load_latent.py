import os

import torch

import folder_paths

from .serialization import EXT, describe, read_payload


def _find_latent_files():
    output_dir = folder_paths.get_output_directory()
    found = []
    for root, _, names in os.walk(output_dir):
        for name in names:
            if name.endswith(EXT):
                found.append(os.path.relpath(os.path.join(root, name), output_dir))
    return sorted(found)


class H3LoadLatent:
    """Read back a latent written by Save MiniMax H3 AV Latent."""

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"latent_file": (_find_latent_files(),)}}

    RETURN_TYPES = ("LATENT",)
    FUNCTION = "load"
    CATEGORY = "latent/minimax_h3"
    DESCRIPTION = (
        "Load a MiniMax H3 AV latent from disk. Feed straight into a sampler or "
        "the MMH3UltimateUpscale latent input."
    )

    def load(self, latent_file):
        path = os.path.join(folder_paths.get_output_directory(), latent_file)
        payload = torch.load(path, map_location="cpu", weights_only=False)
        latent = read_payload(payload, latent_file)
        print(f"[MiniMax H3 LatentIO] loaded {path} {describe(latent)}")
        return (latent,)

    @classmethod
    def IS_CHANGED(cls, latent_file):
        path = os.path.join(folder_paths.get_output_directory(), latent_file)
        return os.path.getmtime(path) if os.path.exists(path) else float("nan")
