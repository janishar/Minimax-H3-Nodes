"""H3CombineLatents: batch any number of MiniMax H3 AV latents together.

Core ComfyUI's LatentBatch concatenates two plain tensors along the batch
dimension; it cannot touch MiniMax H3's NestedTensor container (no
``.contiguous()``, see latent_io/serialization.py), and it is hard-wired to
exactly two inputs. This node duck-types the same video+audio streams the
rest of the pack does (``.tensors``, as in video_extend/common.py's
``validate_av_latent``) and concatenates each stream independently before
repacking them into the same container type -- and, since a plain
torch.Tensor is just a single stream, it doubles as an N-way LatentBatch for
ordinary latents too.

The paired web/combine_latents.js extension grows a fresh empty "latent_N"
input every time the last one gets a connection (and prunes back down to one
empty trailing slot on disconnect), so the socket count on the node tracks
however many latents are actually wired in. This file only sees whichever
``latent_*`` keys arrive connected -- see ``**kwargs`` in ``combine`` below.
"""

import torch


def _streams(inner):
    """Return a latent's component tensors: NestedTensor's ``.tensors``, or itself."""
    tensors = getattr(inner, "tensors", None)
    if tensors is not None:
        return list(tensors)
    if isinstance(inner, torch.Tensor):
        return [inner]
    raise ValueError(f"H3CombineLatents: unrecognised latent samples type {type(inner).__name__}")


class H3CombineLatents:
    """Concatenate any number of MiniMax H3 AV latents into one batched latent."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "latent_1": ("LATENT",),
                "latent_2": ("LATENT",),
            }
        }

    RETURN_TYPES = ("LATENT",)
    FUNCTION = "combine"
    CATEGORY = "latent/minimax_h3"
    DESCRIPTION = (
        "Batch any number of MiniMax H3 AV latents together along the batch dimension "
        "-- connect the last latent_N input to grow another. Video and audio streams "
        "are concatenated separately, so every input must share the same frame count, "
        "height, and width. Plain single-tensor latents are batched the same way, like "
        "core LatentBatch but for N inputs."
    )

    def combine(self, latent_1, latent_2, **kwargs):
        named = {"latent_1": latent_1, "latent_2": latent_2}
        named.update({k: v for k, v in kwargs.items() if k.startswith("latent_") and v is not None})
        ordered_keys = sorted(named, key=lambda name: int(name.rsplit("_", 1)[-1]))
        latents = [named[key] for key in ordered_keys]

        stream_lists = [_streams(latent["samples"]) for latent in latents]

        stream_count = len(stream_lists[0])
        for key, streams in zip(ordered_keys[1:], stream_lists[1:]):
            if len(streams) != stream_count:
                raise ValueError(
                    f"H3CombineLatents: {key} has {len(streams)} latent stream(s), "
                    f"{ordered_keys[0]} has {stream_count}; cannot combine"
                )

        combined_streams = []
        for i in range(stream_count):
            parts = [streams[i] for streams in stream_lists]
            shape_tail = tuple(parts[0].shape[1:])
            for key, part in zip(ordered_keys[1:], parts[1:]):
                if tuple(part.shape[1:]) != shape_tail:
                    raise ValueError(
                        f"H3CombineLatents: stream {i} shape mismatch -- {ordered_keys[0]} is "
                        f"{tuple(parts[0].shape)}, {key} is {tuple(part.shape)}"
                    )
            combined_streams.append(torch.cat(parts, dim=0))

        first_inner = latents[0]["samples"]
        if hasattr(first_inner, "tensors"):
            combined_inner = type(first_inner)(tuple(combined_streams))
        else:
            combined_inner = combined_streams[0]

        out = dict(latents[0])
        out.pop("noise_mask", None)
        out.pop("batch_index", None)
        out["samples"] = combined_inner

        print(
            f"[MiniMax H3 LatentIO] H3CombineLatents: combined {len(latents)} latents "
            f"({', '.join(ordered_keys)}) -> batch {combined_streams[0].shape[0]}"
        )

        return (out,)
