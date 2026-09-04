"""H3TrimLeading: drop leading frames from a decoded IMAGE batch.

Used after VAE-decoding a clip that was joined with H3MotionContext, to drop
the frames that were copied in from the previous clip's tail, before
concatenating clips into one continuous video.
"""


class H3TrimLeading:
    """Drop the first frame_count frames of an IMAGE batch."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "frame_count": (
                    "INT",
                    {
                        "default": 0,
                        "min": 0,
                        "max": 100000,
                        "tooltip": "Leading frames to drop -- wire in H3MotionContext's overlap_frames output.",
                    },
                ),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "trim"
    CATEGORY = "latent/minimax_h3"
    DESCRIPTION = (
        "Drop the first frame_count frames of a decoded MiniMax H3 clip -- the ones copied in "
        "from the previous clip's tail by H3MotionContext -- before concatenating clips into "
        "one continuous video."
    )

    def trim(self, images, frame_count):
        if frame_count >= images.shape[0]:
            raise ValueError(
                f"H3TrimLeading: frame_count {frame_count} would drop the entire {images.shape[0]}-frame batch"
            )
        return (images[frame_count:],)
