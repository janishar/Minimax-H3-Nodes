"""Standalone tests for latent_io.combine_latents -- no ComfyUI or model required.

See tests/test_video_extend.py's module docstring for why folder_paths needs
stubbing before importing anything from this pack.
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

combine_latents_mod = importlib.import_module(f"{PKG}.latent_io.combine_latents")
H3CombineLatents = combine_latents_mod.H3CombineLatents


class FakeNestedTensor:
    """Stand-in for comfy.nested_tensor.NestedTensor: same shape the nodes rely on."""

    def __init__(self, tensors):
        self.tensors = list(tensors)
        self.is_nested = True


def make_av_latent(batch=1, video_t=3, audio_t=5, h=2, w=3, fill=0.0, extra=None):
    video = torch.full((batch, 24, video_t, h, w), float(fill))
    audio = torch.full((batch, 32, 2, audio_t), float(fill))
    out = {"samples": FakeNestedTensor((video, audio))}
    if extra:
        out.update(extra)
    return out


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


def test_combine_two():
    a = make_av_latent(batch=1, fill=1.0)
    b = make_av_latent(batch=2, fill=2.0)
    out, = H3CombineLatents().combine(a, b)
    video, audio = out["samples"].tensors
    check("combine_two: video batch summed", video.shape[0] == 3)
    check("combine_two: audio batch summed", audio.shape[0] == 3)
    check("combine_two: latent_1 content preserved", torch.equal(video[0], a["samples"].tensors[0][0]))
    check("combine_two: latent_2 content preserved", torch.equal(video[1:], b["samples"].tensors[0]))


def test_combine_dynamic_extra_inputs():
    a = make_av_latent(batch=1, fill=1.0)
    b = make_av_latent(batch=1, fill=2.0)
    c = make_av_latent(batch=1, fill=3.0)
    d = make_av_latent(batch=1, fill=4.0)
    out, = H3CombineLatents().combine(a, b, latent_3=c, latent_4=d)
    video, _ = out["samples"].tensors
    check("dynamic: all four batched", video.shape[0] == 4)
    check("dynamic: order follows numeric suffix", torch.equal(video[2], c["samples"].tensors[0][0]))
    check("dynamic: order follows numeric suffix (4th)", torch.equal(video[3], d["samples"].tensors[0][0]))


def test_combine_ignores_none_kwargs():
    a = make_av_latent(fill=1.0)
    b = make_av_latent(fill=2.0)
    out, = H3CombineLatents().combine(a, b, latent_3=None)
    check("None dynamic input is skipped, not treated as a latent", out["samples"].tensors[0].shape[0] == 2)


def test_combine_plain_tensor_latents():
    a = {"samples": torch.full((1, 4, 8, 8), 1.0)}
    b = {"samples": torch.full((2, 4, 8, 8), 2.0)}
    out, = H3CombineLatents().combine(a, b)
    check("plain tensor: batched like LatentBatch", out["samples"].shape[0] == 3)
    check("plain tensor: no NestedTensor wrapping", isinstance(out["samples"], torch.Tensor))


def test_combine_shape_mismatch_raises():
    a = make_av_latent(h=2, w=3)
    b = make_av_latent(h=2, w=4)
    check_raises("mismatched spatial size raises", lambda: H3CombineLatents().combine(a, b))


def test_combine_stream_count_mismatch_raises():
    a = make_av_latent()
    b = {"samples": FakeNestedTensor((torch.zeros(1, 24, 3, 2, 3),))}  # only one stream
    check_raises("mismatched stream count raises", lambda: H3CombineLatents().combine(a, b))


def test_combine_drops_noise_mask_and_batch_index():
    a = make_av_latent(extra={"noise_mask": "stale", "batch_index": [0]})
    b = make_av_latent()
    out, = H3CombineLatents().combine(a, b)
    check("noise_mask dropped", "noise_mask" not in out)
    check("batch_index dropped", "batch_index" not in out)


def test_combine_preserves_other_keys_from_first():
    a = make_av_latent(extra={"custom_key": "keep-me"})
    b = make_av_latent()
    out, = H3CombineLatents().combine(a, b)
    check("unrelated dict keys carried from latent_1", out.get("custom_key") == "keep-me")


if __name__ == "__main__":
    test_combine_two()
    test_combine_dynamic_extra_inputs()
    test_combine_ignores_none_kwargs()
    test_combine_plain_tensor_latents()
    test_combine_shape_mismatch_raises()
    test_combine_stream_count_mismatch_raises()
    test_combine_drops_noise_mask_and_batch_index()
    test_combine_preserves_other_keys_from_first()

    print(f"\n{passed} passed, {failed} failed")
    sys.exit(1 if failed else 0)
