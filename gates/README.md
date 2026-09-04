# gates

Stop a branch of the graph from running at all -- not a bypass that still
executes the upstream chain, and not a mute the user has to apply by hand to
every node in the branch.

## Why this exists

Generation on this machine costs minutes per clip, and a typical workflow has
several parallel branches: a draft preview, an upscale pass, several chained
clips. Turning one of those off should mean *nothing in it runs* -- not "runs
and gets discarded."

## The mechanism

Two independent pieces, both required:

1. **Prune upstream.** The value input is declared lazy
   (`{"lazy": True}`) and the node implements `check_lazy_status`. ComfyUI
   only executes the nodes an output actually needs; a lazy input that's
   never requested means nothing upstream of it is evaluated at all.
   Returning `[]` from `check_lazy_status` is what turns the branch off.

2. **Prune downstream.** The node returns
   `comfy_execution.graph.ExecutionBlocker(None)`. Anything wired to that
   output is then *skipped* by the executor, rather than running on (or
   erroring against) a bare `None`.

Neither piece works alone:

- Lazy without the blocker stops the upstream branch, but whatever consumes
  this node's output still runs -- on `None`, which is usually a crash
  further downstream rather than a clean skip.
- The blocker without lazy still evaluates the entire upstream branch (paying
  the real cost you were trying to avoid) before throwing the result away
  here.

`ExecutionBlocker` moved modules between ComfyUI versions, so the import is
guarded with `try/except ImportError`. If it's unavailable, a disabled
`H3Gate` raises a `RuntimeError` naming the situation and suggesting muting
the downstream output node instead (Ctrl+M) -- rather than silently running
the branch anyway or failing with an unrelated `TypeError` further down the
graph.

## Nodes

### H3Gate -- "Gate (skip branch)"

An on/off switch for one branch.

- **Enabled:** `value` passes through completely unchanged (no copy, no
  cast, no device move -- whatever arrives is what leaves).
- **Disabled:** `value` is never evaluated, the console logs that the branch
  was skipped, and the node returns an `ExecutionBlocker`.

### H3GateAB -- "Gate A/B (pick branch)"

A two-way switch: `use_a` picks `a` or `b`, and only the selected side is
ever evaluated -- the other side's whole upstream chain never runs. If the
selected side isn't connected, it raises a `ValueError` naming which side was
selected. It will not silently fall back to the other input, because that
would hide a wiring mistake behind a plausible-looking result.

This is the more useful of the two for clip chaining: `a` is a fresh Ref2VA
latent for a chain's first clip, `b` is `H3MotionContext`'s stitched output
for every clip after it. Wire both, toggle `use_a`, and only one branch of
the chain is ever sampled per run.

## Wildcard typing

Both nodes accept and return `ANY` (`gates/any_type.py`) -- a string
subclass whose `__ne__` always reports "not different," so it matches
`LATENT`, `IMAGE`, `MODEL`, `CONDITIONING`, or anything else ComfyUI can pass
through a socket. One gate works for any branch, not just latents.

## vs. ComfyUI's own bypass and mute

| | What happens |
|---|---|
| **Bypass** (Ctrl+B) | The node is skipped, but everything upstream of it still runs -- bypass only removes the one bypassed node from the chain, not the cost behind it. |
| **Mute** (Ctrl+M) | Prunes execution, but has to be applied to the specific node(s) you want removed, by hand, every time -- there's no single widget that removes a whole branch based on a condition. |
| **H3Gate / H3GateAB** | One widget, one place: toggling it removes the entire upstream branch from that run, with no per-node bookkeeping. |

## A blocked branch can look like nothing happened

If the branch you disable contains the *only* output node in the workflow
(a lone `SaveImage`, `VHS_VideoCombine`, etc.), disabling it means the run
produces no output at all -- no file written, no error shown, no obvious sign
anything happened. This is the intended behavior (the branch was correctly
skipped), but it can look exactly like a silent failure if you forget which
branch fed your only output node. Keep at least one always-on output
somewhere while testing a gated workflow, or check the console for the
"branch skipped" log line.
