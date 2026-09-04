"""Shared frame math and latent validation for MiniMax H3 clip chaining.

MiniMax H3's video stream is causal-VAE compressed on a repeating 5-token
cycle (token weights 1,4,4,4,4 pixel-frames), which only decodes correctly
from a tensor's own index 0 -- see comfy/ldm/minimax/model.py's
FRAME_PER_TOKEN and comfy_extras/nodes_minimax_h3.py's align_frame_count /
video_latent_t. Reimplemented here (verified against that build) rather than
imported, since those are node-file internals, not a public API, and rather
than copied from either GPL reference implementation named in the task.

Valid clip lengths sit on the video phase grid frames = 17k + 5. Of those,
the ones where frames * 40 / 24 (24fps video, 40Hz audio latent) is a whole
number let the audio stream join exactly at the same boundary; the rest are
video-only.
"""

FPS = 24
AUDIO_FPS = 40

VIDEO_CHANNELS = 24
AUDIO_CHANNELS = 32

CONTEXT_LENGTHS_AUDIO = (39, 90, 141, 192, 243)
CONTEXT_LENGTHS_VIDEO_ONLY = (5, 22, 56)
CONTEXT_LENGTHS_ALL = tuple(sorted(set(CONTEXT_LENGTHS_AUDIO) | set(CONTEXT_LENGTHS_VIDEO_ONLY)))


def video_latent_t(frame_count):
    """Video-token count for a pixel-frame count on the 17k+5 grid."""
    return 2 if frame_count <= 5 else ((frame_count - 5) // 17) * 5 + 2


def audio_latent_t(frame_count):
    """Audio-latent step count (40 Hz) for a pixel-frame count (24 fps)."""
    return round(frame_count * AUDIO_FPS / FPS)


def assert_video_block_boundary(total_t, context_t, label):
    """The tail start index must be a multiple of 5 video latent tokens.

    Each token's pixel-frame weight depends on its index mod 5 (see module
    docstring), and that indexing always restarts at 0 for whatever tensor
    the token lives in. Slicing at a non-multiple-of-5 offset would carry a
    token into a new tensor where it gets reinterpreted as a different
    pixel-frame weight than it actually has.
    """
    start = total_t - context_t
    if start < 0:
        raise ValueError(
            f"{label}: context needs {context_t} video latent steps but the clip only has {total_t}"
        )
    if start % 5 != 0:
        raise ValueError(
            f"{label}: tail start index {start} (= {total_t} - {context_t}) is not a MiniMax H3 "
            f"block boundary (must be a multiple of 5 video latent tokens)"
        )
    return start


def validate_av_latent(nested, label):
    """Confirm the NestedTensor duck-types as a 2-stream MiniMax H3 AV latent and return (video, audio)."""
    tensors = getattr(nested, "tensors", None)
    if not getattr(nested, "is_nested", False) or tensors is None or len(tensors) != 2:
        raise ValueError(f"{label} is not a MiniMax H3 AV latent (expected a 2-stream NestedTensor)")
    video, audio = tensors
    if video.ndim != 5 or video.shape[1] != VIDEO_CHANNELS:
        raise ValueError(f"{label}: expected video stream [B,{VIDEO_CHANNELS},T,H,W], got {tuple(video.shape)}")
    if audio.ndim != 4 or audio.shape[1] != AUDIO_CHANNELS:
        raise ValueError(f"{label}: expected audio stream [B,{AUDIO_CHANNELS},2,T], got {tuple(audio.shape)}")
    return video, audio
