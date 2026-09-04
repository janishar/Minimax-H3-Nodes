"""H3SliceTail: extract just the tail of a MiniMax H3 AV latent.

Exists to be decoded and eyeballed on its own, before H3MotionContext relies
on the same slicing math for a real join: if the tail plays back as exactly
the clip's last context_length/24 seconds (video and audio in sync), the
index math and block-boundary assumption are confirmed cheaply.
"""

from .common import (
    CONTEXT_LENGTHS_ALL,
    assert_video_block_boundary,
    audio_latent_t,
    validate_av_latent,
    video_latent_t,
)


class H3SliceTail:
    """Return only the tail of a MiniMax H3 AV latent, as a standalone latent."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "samples": ("LATENT",),
                "context_length": ([str(n) for n in CONTEXT_LENGTHS_ALL], {"default": "39"}),
            }
        }

    RETURN_TYPES = ("LATENT",)
    FUNCTION = "slice_tail"
    CATEGORY = "latent/minimax_h3"
    DESCRIPTION = (
        "Extract only the last context_length pixel frames of a MiniMax H3 AV latent, as a "
        "standalone latent. Decode it and play it back to confirm the slice lands exactly on "
        "the clip's tail before trusting the same math in H3MotionContext."
    )

    def slice_tail(self, samples, context_length):
        context_frames = int(context_length)
        inner = samples["samples"]
        video, audio = validate_av_latent(inner, "samples")

        total_video_t = video.shape[2]
        context_video_t = video_latent_t(context_frames)
        video_start = assert_video_block_boundary(total_video_t, context_video_t, "H3SliceTail")

        total_audio_t = audio.shape[-1]
        context_audio_t = audio_latent_t(context_frames)
        audio_start = total_audio_t - context_audio_t
        if audio_start < 0:
            raise ValueError(
                f"H3SliceTail: context needs {context_audio_t} audio latent steps but the clip only has {total_audio_t}"
            )

        tail_video = video[:, :, video_start:].clone()
        tail_audio = audio[:, :, :, audio_start:].clone()

        print(
            f"[Minimax-H3-Nodes] H3SliceTail: context_length={context_frames} frames -> "
            f"video steps {video_start}:{total_video_t} ({context_video_t} kept), "
            f"audio steps {audio_start}:{total_audio_t} ({context_audio_t} kept)"
        )

        tail = type(inner)((tail_video, tail_audio))
        out = dict(samples)
        out["samples"] = tail
        return (out,)
