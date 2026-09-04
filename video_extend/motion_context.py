"""H3MotionContext: stitch clip N's tail into clip N+1's target latent.

Copies clip N's sampled AV latent tail into the front of clip N+1's Ref2VA
latent (a copy of it -- Ref2VA's own latent is never rebuilt from scratch,
since it may carry structure this pack does not know about) and marks those
positions as never-denoised via the LATENT dict's ``noise_mask`` key.
MiniMax H3's model (comfy/model_base.py's MiniMaxH3._denoise_mask_conds /
scale_latent_inpaint) reads that key with 0 = preserve, 1 = generate, so the
sampler treats the copied-in positions as given context and only generates
the rest -- continuing motion, and optionally audio, across the join.
"""

import torch

from .common import (
    CONTEXT_LENGTHS_ALL,
    CONTEXT_LENGTHS_AUDIO,
    assert_video_block_boundary,
    audio_latent_t,
    validate_av_latent,
    video_latent_t,
)


class H3MotionContext:
    """Overwrite the front of clip N+1's target latent with clip N's tail."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prev_latent": ("LATENT", {"tooltip": "Clip N's sampled (denoised) AV latent."}),
                "target_latent": (
                    "LATENT",
                    {"tooltip": "Clip N+1's Ref2VA LATENT output. A copy of this is modified and returned."},
                ),
                "context_length": ([str(n) for n in CONTEXT_LENGTHS_ALL], {"default": "39"}),
                "audio_continuity": (
                    "BOOLEAN",
                    {
                        "default": True,
                        "tooltip": (
                            "Also stitch the audio stream. Requires context_length to be one of "
                            f"{CONTEXT_LENGTHS_AUDIO} (frames * 40/24 must be a whole number). "
                            "Off stitches video only and allows the shorter video-only lengths."
                        ),
                    },
                ),
            }
        }

    RETURN_TYPES = ("LATENT", "INT")
    RETURN_NAMES = ("latent", "overlap_frames")
    FUNCTION = "stitch"
    CATEGORY = "latent/minimax_h3"
    DESCRIPTION = (
        "Overwrite the front of clip N+1's target latent with the tail of clip N's sampled "
        "latent and mark those positions as never-denoised, so the sampler continues motion "
        "(and, optionally, audio) across the join instead of starting fresh. target_latent "
        "must be Ref2VA's LATENT output for clip N+1, run before this node."
    )

    def stitch(self, prev_latent, target_latent, context_length, audio_continuity):
        context_frames = int(context_length)
        if audio_continuity and context_frames not in CONTEXT_LENGTHS_AUDIO:
            raise ValueError(
                f"H3MotionContext: context_length={context_frames} is not audio-aligned "
                f"(frames * 40/24 is not a whole number); use audio_continuity=False or one "
                f"of {CONTEXT_LENGTHS_AUDIO}"
            )

        prev_video, prev_audio = validate_av_latent(prev_latent["samples"], "prev_latent")
        target_video, target_audio = validate_av_latent(target_latent["samples"], "target_latent")
        target_inner = target_latent["samples"]

        if tuple(prev_video.shape[-2:]) != tuple(target_video.shape[-2:]):
            raise ValueError(
                f"H3MotionContext: prev_latent and target_latent are different spatial sizes "
                f"({tuple(prev_video.shape[-2:])} vs {tuple(target_video.shape[-2:])}); chain "
                f"clips at the same resolution and upscale after."
            )

        context_video_t = video_latent_t(context_frames)
        prev_video_start = assert_video_block_boundary(prev_video.shape[2], context_video_t, "H3MotionContext (prev_latent)")
        if context_video_t > target_video.shape[2]:
            raise ValueError(
                f"H3MotionContext: context_length {context_frames} frames needs {context_video_t} "
                f"video latent steps, more than target_latent's {target_video.shape[2]}"
            )

        video_device, video_dtype = target_video.device, target_video.dtype
        tail_video = prev_video[:, :, prev_video_start:].to(device=video_device, dtype=video_dtype)

        new_target_video = target_video.clone()
        new_target_video[:, :, :context_video_t] = tail_video

        video_mask = torch.ones((1, 1) + tuple(target_video.shape[2:]), device=video_device, dtype=torch.float32)
        video_mask[:, :, :context_video_t] = 0.0

        audio_device, audio_dtype = target_audio.device, target_audio.dtype
        new_target_audio = target_audio.clone()
        audio_mask = torch.ones((1, 1) + tuple(target_audio.shape[2:]), device=audio_device, dtype=torch.float32)

        if audio_continuity:
            context_audio_t = audio_latent_t(context_frames)
            prev_audio_start = prev_audio.shape[-1] - context_audio_t
            if prev_audio_start < 0:
                raise ValueError(
                    f"H3MotionContext: context needs {context_audio_t} audio latent steps but "
                    f"prev_latent only has {prev_audio.shape[-1]}"
                )
            if context_audio_t > target_audio.shape[-1]:
                raise ValueError(
                    f"H3MotionContext: context needs {context_audio_t} audio latent steps, more "
                    f"than target_latent's {target_audio.shape[-1]}"
                )
            tail_audio = prev_audio[:, :, :, prev_audio_start:].to(device=audio_device, dtype=audio_dtype)
            new_target_audio[:, :, :, :context_audio_t] = tail_audio
            audio_mask[:, :, :, :context_audio_t] = 0.0

        stitched = type(target_inner)((new_target_video, new_target_audio))
        mask = type(target_inner)((video_mask, audio_mask))

        out = dict(target_latent)
        out["samples"] = stitched
        out["noise_mask"] = mask

        print(
            f"[Minimax-H3-Nodes] H3MotionContext: context_length={context_frames} frames, "
            f"video preserved steps 0:{context_video_t}/{target_video.shape[2]}, "
            f"audio_continuity={audio_continuity}"
            + (
                f", audio preserved steps 0:{audio_latent_t(context_frames)}/{target_audio.shape[-1]}"
                if audio_continuity
                else ""
            )
        )

        return (out, context_frames)
