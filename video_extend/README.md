# video_extend

Chains MiniMax H3 clips in latent space: the tail of one clip's sampled
latent is copied into the front of the next clip's target latent and marked
as never-denoised, so motion (and, optionally, audio) continue across the
join with no VAE decode/re-encode in between.

## The two constraints

A context length has to satisfy both of these at once:

- **Video phase grid.** MiniMax H3's causal VAE compresses video on a
  repeating 5-token cycle (pixel-frame weights 1, 4, 4, 4, 4), so only
  `frames = 17k + 5` pixel-frame counts land on a token boundary. Slicing a
  tail at any other offset would carry a token into a new tensor where it
  gets reinterpreted as a different pixel-frame weight than it actually has
  — `H3SliceTail` and `H3MotionContext` both assert this (`assert_video_block_boundary`
  in `common.py`) rather than trusting the caller.
- **Audio alignment.** Video runs at 24 fps and the audio latent at 40 Hz, so
  a context length only joins the audio stream exactly when
  `frames * 40 / 24` is a whole number.

Solving both gives exactly: **39, 90, 141, 192, 243, 294 …** (39 frames =
65 audio steps, exactly).

| audio_continuity | valid context_length (frames) |
|---|---|
| `True` | 39, 90, 141, 192, 243 |
| `False` | also 5, 22, 56 (on the video grid, fractional in audio, so video-only) |

`H3MotionContext` validates this at run time rather than filtering the combo
box: pick `audio_continuity=False` before using 5, 22, or 56.

## Wiring order

```
clip N:   ... sampler -> denoised LATENT ────────────┐
                                                       │
clip N+1: Ref2VA -> LATENT ──────────────┬── H3MotionContext ── latent ──▶ sampler (latent_image)
          (must run before               │                └── overlap_frames ──▶ H3TrimLeading
           H3MotionContext --            │
           its LATENT is what            │
           gets written into)            │
                                          prev_latent = clip N's denoised LATENT
```

1. Run clip N's sampler to get its denoised `LATENT`.
2. Run clip N+1's `MiniMax H3 Reference to Video` (Ref2VA) node to get *its*
   `LATENT` — this must happen before `H3MotionContext`, since that latent is
   what gets copied into, not replaced. `H3MotionContext` never constructs an
   empty latent itself, since Ref2VA's may carry structure this pack doesn't
   know about.
3. Feed both into `H3MotionContext` → `latent` goes to clip N+1's sampler as
   `latent_image`; `overlap_frames` goes to `H3TrimLeading` after that clip
   is decoded, to drop the frames that were only there for continuity before
   concatenating clips into one video.

Before wiring any of this on a real generation (each of which costs minutes
on this machine), use `H3InspectLatent` to dump a real latent's structure and
`H3SliceTail` to decode-and-eyeball that the slicing math lands exactly on
the clip's tail.

## Masking

`H3MotionContext` writes the copied-in positions into `target_latent`'s
video/audio tensors *and* sets `latent["noise_mask"]` to a matching
`NestedTensor` of per-stream masks (`0 = preserve, 1 = generate`). This is
MiniMax H3's own per-token denoise-mask contract
(`comfy/model_base.py`'s `MiniMaxH3._denoise_mask_conds` /
`scale_latent_inpaint`) — the same `noise_mask` key core inpainting nodes use,
just built for H3's two-stream latent instead of a single tensor.

## Device and dtype

All four nodes are device-agnostic: they operate on whatever device the
input latents arrive on, and only move the sliced prev-clip tail to
target_latent's device/dtype immediately before writing it in (a no-op
unless the two clips happen to be on different devices). Nothing here
selects or forces a device.
