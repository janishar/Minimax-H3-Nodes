"""Standalone tests for video_extend -- no ComfyUI or model required.

Exercises slicing, overwriting, index math and the phase assertions against a
stand-in container class shaped like comfy.nested_tensor.NestedTensor
(.tensors list + .is_nested flag), with realistic MiniMax H3 shapes
(video [B,24,T,H,W], audio [B,32,2,T]).

folder_paths only resolves inside a running ComfyUI. latent_io/__init__.py
imports it eagerly (H3SaveLatent/H3LoadLatent need it), and video_extend
reuses latent_io.serialization's tensor introspection, so importing this
node pack at all pulls that in transitively. Stub it before importing
anything -- nothing here calls into save/load, so an empty module is enough.
"""

import importlib
import sys
import types
from pathlib import Path

import torch

if "folder_paths" not in sys.modules:
    sys.modules["folder_paths"] = types.ModuleType("folder_paths")

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT.parent) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT.parent))

PKG = REPO_ROOT.name  # "Minimax-H3-Nodes" -- importlib accepts this even though it's not an identifier

common = importlib.import_module(f"{PKG}.video_extend.common")
slice_tail_mod = importlib.import_module(f"{PKG}.video_extend.slice_tail")
motion_context_mod = importlib.import_module(f"{PKG}.video_extend.motion_context")
trim_leading_mod = importlib.import_module(f"{PKG}.video_extend.trim_leading")

H3SliceTail = slice_tail_mod.H3SliceTail
H3MotionContext = motion_context_mod.H3MotionContext
H3TrimLeading = trim_leading_mod.H3TrimLeading


class FakeNestedTensor:
    """Stand-in for comfy.nested_tensor.NestedTensor: same shape the nodes rely on."""

    def __init__(self, tensors):
        self.tensors = list(tensors)
        self.is_nested = True


def make_av_latent(frame_count, height=64, width=96, batch=1, device="cpu", dtype=torch.float32, fill="arange"):
    """Build a fake AV latent with distinguishable per-frame content for a given pixel frame_count."""
    video_t = common.video_latent_t(frame_count)
    audio_t = common.audio_latent_t(frame_count)
    lat_h, lat_w = height // 16, width // 16

    video = torch.zeros((batch, common.VIDEO_CHANNELS, video_t, lat_h, lat_w), device=device, dtype=dtype)
    audio = torch.zeros((batch, common.AUDIO_CHANNELS, 2, audio_t), device=device, dtype=dtype)
    if fill == "arange":
        # each time step gets a distinct scalar value so slices can be checked by content, not just shape
        for t in range(video_t):
            video[:, :, t] = t
        for t in range(audio_t):
            audio[:, :, :, t] = 1000 + t
    else:
        video[:] = fill
        audio[:] = fill

    return {"samples": FakeNestedTensor((video, audio))}, video_t, audio_t


passed = 0
failed = 0


def check(name, condition):
    global passed, failed
    if condition:
        passed += 1
        print(f"PASS {name}")
    else:
        failed += 1
        print(f"FAIL {name}")


def check_raises(name, fn, exc_type=ValueError):
    global passed, failed
    try:
        fn()
    except exc_type:
        passed += 1
        print(f"PASS {name}")
    else:
        failed += 1
        print(f"FAIL {name} (expected {exc_type.__name__}, nothing raised)")


# ---------------------------------------------------------------------------
# Frame math / phase grid assertions
# ---------------------------------------------------------------------------

def test_frame_math(device="cpu"):
    tag = f" [{device}]" if device != "cpu" else ""

    # 39 frames = 65 audio steps exactly, per the task's derived table.
    check(f"audio_latent_t(39) == 65{tag}", common.audio_latent_t(39) == 65)

    for n in common.CONTEXT_LENGTHS_AUDIO:
        check(f"frames={n} is on the 17k+5 grid{tag}", (n - 5) % 17 == 0)
        check(f"frames={n} is audio-aligned (n*40 % 24 == 0){tag}", (n * 40) % 24 == 0)

    for n in common.CONTEXT_LENGTHS_VIDEO_ONLY:
        check(f"frames={n} is on the 17k+5 grid{tag}", (n - 5) % 17 == 0)
        check(f"frames={n} is fractional in audio (n*40 % 24 != 0){tag}", (n * 40) % 24 != 0)

    # video_latent_t always lands on 5k+2, so any two grid lengths differ by a multiple of 5.
    for n in common.CONTEXT_LENGTHS_ALL:
        check(f"video_latent_t({n}) % 5 == 2{tag}", common.video_latent_t(n) % 5 == 2)

    check_raises(
        f"assert_video_block_boundary rejects a non-block-boundary offset{tag}",
        lambda: common.assert_video_block_boundary(total_t=10, context_t=7, label="test"),
    )
    check_raises(
        f"assert_video_block_boundary rejects a too-long context{tag}",
        lambda: common.assert_video_block_boundary(total_t=10, context_t=20, label="test"),
    )
    start = common.assert_video_block_boundary(total_t=12, context_t=12, label="test")
    check(f"assert_video_block_boundary accepts an exact-length context{tag}", start == 0)


# ---------------------------------------------------------------------------
# H3SliceTail
# ---------------------------------------------------------------------------

def test_slice_tail(device="cpu", dtype=torch.float32):
    tag = f" [{device}, {dtype}]" if device != "cpu" or dtype != torch.float32 else ""

    total_frames = 90
    context_frames = 39
    latent, total_video_t, total_audio_t = make_av_latent(total_frames, device=device, dtype=dtype)
    video_before = latent["samples"].tensors[0].clone()
    audio_before = latent["samples"].tensors[1].clone()

    out, = H3SliceTail().slice_tail(latent, str(context_frames))
    tail_video, tail_audio = out["samples"].tensors

    expected_video_t = common.video_latent_t(context_frames)
    expected_audio_t = common.audio_latent_t(context_frames)
    check(f"H3SliceTail video shape{tag}", tail_video.shape[2] == expected_video_t)
    check(f"H3SliceTail audio shape{tag}", tail_audio.shape[-1] == expected_audio_t)

    # content check: the tail must equal the *last* steps of the source, not some other slice
    check(
        f"H3SliceTail video content matches source tail{tag}",
        torch.equal(tail_video, video_before[:, :, total_video_t - expected_video_t:]),
    )
    check(
        f"H3SliceTail audio content matches source tail{tag}",
        torch.equal(tail_audio, audio_before[:, :, :, total_audio_t - expected_audio_t:]),
    )

    # input must not be mutated
    check(f"H3SliceTail does not mutate its input (video){tag}", torch.equal(latent["samples"].tensors[0], video_before))
    check(f"H3SliceTail does not mutate its input (audio){tag}", torch.equal(latent["samples"].tensors[1], audio_before))

    check(f"H3SliceTail preserves device{tag}", str(tail_video.device).startswith(device))
    check(f"H3SliceTail preserves dtype{tag}", tail_video.dtype == dtype)


# ---------------------------------------------------------------------------
# H3MotionContext
# ---------------------------------------------------------------------------

def test_motion_context(device="cpu", dtype=torch.float32):
    tag = f" [{device}, {dtype}]" if device != "cpu" or dtype != torch.float32 else ""

    prev_frames = 90
    target_frames = 141
    context_frames = 39

    prev_latent, prev_video_t, prev_audio_t = make_av_latent(prev_frames, device=device, dtype=dtype, fill="arange")
    target_latent, target_video_t, target_audio_t = make_av_latent(target_frames, device=device, dtype=dtype, fill=-1.0)

    prev_video_before = prev_latent["samples"].tensors[0].clone()
    prev_audio_before = prev_latent["samples"].tensors[1].clone()
    target_video_before = target_latent["samples"].tensors[0].clone()
    target_audio_before = target_latent["samples"].tensors[1].clone()

    out, overlap_frames = H3MotionContext().stitch(prev_latent, target_latent, str(context_frames), True)

    check(f"H3MotionContext overlap_frames == context_length{tag}", overlap_frames == context_frames)

    new_video, new_audio = out["samples"].tensors
    context_video_t = common.video_latent_t(context_frames)
    context_audio_t = common.audio_latent_t(context_frames)

    expected_video_tail = prev_video_before[:, :, prev_video_t - context_video_t:]
    expected_audio_tail = prev_audio_before[:, :, :, prev_audio_t - context_audio_t:]

    check(
        f"H3MotionContext video front == prev tail{tag}",
        torch.equal(new_video[:, :, :context_video_t].to(expected_video_tail.dtype), expected_video_tail.to(expected_video_tail.dtype)),
    )
    check(
        f"H3MotionContext video remainder unchanged{tag}",
        torch.equal(new_video[:, :, context_video_t:], target_video_before[:, :, context_video_t:]),
    )
    check(
        f"H3MotionContext audio front == prev tail{tag}",
        torch.equal(new_audio[:, :, :, :context_audio_t].to(expected_audio_tail.dtype), expected_audio_tail.to(expected_audio_tail.dtype)),
    )
    check(
        f"H3MotionContext audio remainder unchanged{tag}",
        torch.equal(new_audio[:, :, :, context_audio_t:], target_audio_before[:, :, :, context_audio_t:]),
    )

    video_mask, audio_mask = out["noise_mask"].tensors
    check(f"H3MotionContext video mask preserves front (0){tag}", torch.all(video_mask[:, :, :context_video_t] == 0.0).item())
    check(f"H3MotionContext video mask generates rest (1){tag}", torch.all(video_mask[:, :, context_video_t:] == 1.0).item())
    check(f"H3MotionContext audio mask preserves front (0){tag}", torch.all(audio_mask[:, :, :, :context_audio_t] == 0.0).item())
    check(f"H3MotionContext audio mask generates rest (1){tag}", torch.all(audio_mask[:, :, :, context_audio_t:] == 1.0).item())

    # neither input may be mutated
    check(f"H3MotionContext does not mutate prev_latent (video){tag}", torch.equal(prev_latent["samples"].tensors[0], prev_video_before))
    check(f"H3MotionContext does not mutate prev_latent (audio){tag}", torch.equal(prev_latent["samples"].tensors[1], prev_audio_before))
    check(f"H3MotionContext does not mutate target_latent (video){tag}", torch.equal(target_latent["samples"].tensors[0], target_video_before))
    check(f"H3MotionContext does not mutate target_latent (audio){tag}", torch.equal(target_latent["samples"].tensors[1], target_audio_before))

    check(f"H3MotionContext output device follows target{tag}", str(new_video.device).startswith(device))
    check(f"H3MotionContext output dtype follows target{tag}", new_video.dtype == dtype)

    # audio_continuity=False: video stitched, audio left fully generate, no audio copy
    target_latent2, _, target_audio_t2 = make_av_latent(target_frames, device=device, dtype=dtype, fill=-1.0)
    target_audio2_before = target_latent2["samples"].tensors[1].clone()
    out2, _ = H3MotionContext().stitch(prev_latent, target_latent2, str(context_frames), False)
    _, new_audio2 = out2["samples"].tensors
    audio_mask2 = out2["noise_mask"].tensors[1]
    check(f"H3MotionContext audio_continuity=False leaves audio untouched{tag}", torch.equal(new_audio2, target_audio2_before))
    check(f"H3MotionContext audio_continuity=False generates all audio (mask==1){tag}", torch.all(audio_mask2 == 1.0).item())

    # video-only context length must be rejected when audio_continuity=True
    check_raises(
        f"H3MotionContext rejects a non-audio-aligned context_length with audio_continuity=True{tag}",
        lambda: H3MotionContext().stitch(prev_latent, target_latent, "22", True),
    )

    # spatial size mismatch must be rejected
    mismatched_target, _, _ = make_av_latent(target_frames, height=64, width=128, device=device, dtype=dtype)
    check_raises(
        f"H3MotionContext rejects mismatched spatial size{tag}",
        lambda: H3MotionContext().stitch(prev_latent, mismatched_target, str(context_frames), True),
    )


def test_motion_context_cross_dtype():
    # prev in a different dtype than target: output must follow target's dtype, never cast target down
    prev_latent, prev_video_t, prev_audio_t = make_av_latent(90, dtype=torch.float64, fill="arange")
    target_latent, target_video_t, target_audio_t = make_av_latent(141, dtype=torch.float32, fill=-1.0)

    out, _ = H3MotionContext().stitch(prev_latent, target_latent, "39", True)
    new_video, new_audio = out["samples"].tensors
    check("H3MotionContext cross-dtype: output video dtype follows target", new_video.dtype == torch.float32)
    check("H3MotionContext cross-dtype: output audio dtype follows target", new_audio.dtype == torch.float32)
    check("H3MotionContext cross-dtype: prev_latent dtype untouched", prev_latent["samples"].tensors[0].dtype == torch.float64)


# ---------------------------------------------------------------------------
# H3TrimLeading
# ---------------------------------------------------------------------------

def test_trim_leading():
    images = torch.arange(10 * 4 * 4 * 3, dtype=torch.float32).reshape(10, 4, 4, 3)
    out, = H3TrimLeading().trim(images, 3)
    check("H3TrimLeading drops the requested count", out.shape[0] == 7)
    check("H3TrimLeading keeps the tail content", torch.equal(out, images[3:]))
    check_raises("H3TrimLeading rejects dropping the whole batch", lambda: H3TrimLeading().trim(images, 10))


# ---------------------------------------------------------------------------

def run_all(device, dtype=torch.float32):
    test_frame_math(device)
    test_slice_tail(device, dtype)
    test_motion_context(device, dtype)


if __name__ == "__main__":
    run_all("cpu")
    test_motion_context_cross_dtype()
    test_trim_leading()

    if torch.cuda.is_available():
        print("\n--- repeating shape/index/device tests on cuda ---")
        run_all("cuda")

    if torch.backends.mps.is_available():
        print("\n--- repeating shape/index/device tests on mps ---")
        run_all("mps")

    print(f"\n{passed} passed, {failed} failed")
    sys.exit(1 if failed else 0)
