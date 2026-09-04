# Minimax-H3-Nodes

Save and load MiniMax H3 AV latents.

## Why

Core `SaveLatent` does `samples["samples"].contiguous()`. MiniMax H3's latent is
a custom `NestedTensor` container holding the video stream (`[B,24,T,H,W]`) and
the audio stream (`[B,32,2,T]`) together — not a `torch.Tensor` subclass, and
with no `.contiguous()`. The save raises `AttributeError`.

It also cannot be rebuilt with `torch.nested.nested_tensor`, which requires every
component to share a dimension count; these two are 5D and 4D.

These nodes serialise the whole latent object with `torch.save`, preserving both
streams bit-identically along with any extra dict keys.

## Install

    cd ComfyUI/custom_nodes/
    git clone <this repo> Minimax-H3-Nodes

No dependencies beyond ComfyUI itself. Restart ComfyUI.

## Nodes

| Node | Category | In / Out |
|---|---|---|
| Save MiniMax H3 AV Latent | `latent/minimax_h3` | `LATENT` in, writes `.minimaxh3latent` |
| Load MiniMax H3 AV Latent | `latent/minimax_h3` | picks a file, `LATENT` out |

Files are written under the ComfyUI output directory, in the `latent/` subfolder,
using `filename_prefix` (default `latent/minimax_h3`), auto-numbered — e.g.
`output/latent/minimax_h3_00001_.minimaxh3latent`. Roughly 0.5 MB per 124-frame
864x480 clip.

## Typical use

Decouple generation from upscaling so peak memory stays at one clip:

    Sampler -> denoised_output -> Save MiniMax H3 AV Latent      (generation run)
    Load MiniMax H3 AV Latent -> MMH3UltimateUpscale -> decode   (upscale run, later)

Chain clips at draft resolution and save *those* latents — a motion-context tail
must come from the draft, not an upscaled one, or the guide encode shape
mismatches.

Each save logs the detected class name and stream shapes to console, which is
the quickest way to confirm what your build's latent actually contains.

## Caveat

Files are pickles and reference the MiniMax H3 latent class by name. If the
MiniMax H3 nodes are updated and that class moves or is renamed, older files
stop loading. Keep the decoded video for anything you need long term.

## Layout

```
Minimax-H3-Nodes/
├── __init__.py              # NODE_CLASS_MAPPINGS / NODE_DISPLAY_NAME_MAPPINGS
├── latent_io/
│   ├── __init__.py          # re-exports the node classes
│   ├── save_latent.py       # H3SaveLatent
│   ├── load_latent.py       # H3LoadLatent
│   └── serialization.py     # shared encode/describe helpers
├── pyproject.toml           # [tool.comfy] block, no dependencies
├── README.md
└── .gitignore
```
