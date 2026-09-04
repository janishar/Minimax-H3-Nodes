"""Standalone tests for the gates package -- no ComfyUI or torch required.

Stubs comfy_execution.graph with a fake ExecutionBlocker before importing
gate.py, the same way test_video_extend.py stubs folder_paths. Both stubs
are needed here even though gates/ itself touches neither: importing
`PKG.gates.any_type` first runs the top-level package's __init__.py, which
imports latent_io (needs folder_paths) -- see that file's module docstring.
"""

import importlib
import sys
import types
from pathlib import Path

if "folder_paths" not in sys.modules:
    sys.modules["folder_paths"] = types.ModuleType("folder_paths")

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT.parent) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT.parent))

PKG = REPO_ROOT.name  # "Minimax-H3-Nodes" -- importlib accepts this even though it's not an identifier


class FakeExecutionBlocker:
    def __init__(self, message):
        self.message = message


def _stub_comfy_execution_graph():
    graph_mod = types.ModuleType("comfy_execution.graph")
    graph_mod.ExecutionBlocker = FakeExecutionBlocker
    package_mod = types.ModuleType("comfy_execution")
    package_mod.graph = graph_mod
    sys.modules["comfy_execution"] = package_mod
    sys.modules["comfy_execution.graph"] = graph_mod


_stub_comfy_execution_graph()

any_type_mod = importlib.import_module(f"{PKG}.gates.any_type")
gate_mod = importlib.import_module(f"{PKG}.gates.gate")
gate_ab_mod = importlib.import_module(f"{PKG}.gates.gate_ab")
gate_switch_mod = importlib.import_module(f"{PKG}.gates.gate_switch")
gate_route_mod = importlib.import_module(f"{PKG}.gates.gate_route")

ANY = any_type_mod.ANY
H3Gate = gate_mod.H3Gate
H3GateAB = gate_ab_mod.H3GateAB
H3GateSwitch = gate_switch_mod.H3GateSwitch
H3GateRoute = gate_route_mod.H3GateRoute


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
# any_type
# ---------------------------------------------------------------------------

def test_any_type():
    check("ANY != 'LATENT' is False (wildcard accepts LATENT)", not (ANY != "LATENT"))
    check("ANY != 'IMAGE' is False (wildcard accepts IMAGE)", not (ANY != "IMAGE"))
    check("ANY != 123 is False (wildcard accepts non-string values too)", not (ANY != 123))
    check("ANY != None is False (wildcard compares unequal to nothing)", not (ANY != None))


# ---------------------------------------------------------------------------
# H3Gate
# ---------------------------------------------------------------------------

def test_gate_check_lazy_status():
    gate = H3Gate()
    check(
        "H3Gate.check_lazy_status requests value when enabled",
        gate.check_lazy_status(enabled=True, value=None) == ["value"],
    )
    check(
        "H3Gate.check_lazy_status requests nothing when disabled",
        gate.check_lazy_status(enabled=False, value=None) == [],
    )
    check(
        "H3Gate.check_lazy_status requests nothing once value is resolved",
        gate.check_lazy_status(enabled=True, value="already-resolved") == [],
    )


def test_gate_enabled_passthrough_identity():
    sentinel = object()
    out, = H3Gate().gate(enabled=True, value=sentinel)
    check("H3Gate enabled path returns the identical object (is, not ==)", out is sentinel)


def test_gate_disabled_returns_blocker():
    out, = H3Gate().gate(enabled=False, value=None)
    check("H3Gate disabled path returns an ExecutionBlocker", isinstance(out, FakeExecutionBlocker))
    check("H3Gate disabled path blocks silently (message is None)", out.message is None)


def test_gate_disabled_never_touches_an_unconnected_value():
    # value omitted entirely, as ComfyUI would for an unconnected optional lazy input
    out, = H3Gate().gate(enabled=False)
    check("H3Gate disabled with no value connected still returns a blocker", isinstance(out, FakeExecutionBlocker))


def test_gate_missing_execution_blocker_raises():
    original = gate_mod.ExecutionBlocker
    gate_mod.ExecutionBlocker = None
    try:
        check_raises(
            "H3Gate disabled without ExecutionBlocker available raises a clear error",
            lambda: H3Gate().gate(enabled=False, value=None),
            RuntimeError,
        )
    finally:
        gate_mod.ExecutionBlocker = original


# ---------------------------------------------------------------------------
# H3GateAB
# ---------------------------------------------------------------------------

def test_gate_ab_check_lazy_status_requests_only_selected_side():
    gate_ab = H3GateAB()
    check(
        "H3GateAB requests only 'a' when use_a is True",
        gate_ab.check_lazy_status(use_a=True, a=None, b=None) == ["a"],
    )
    check(
        "H3GateAB requests only 'b' when use_a is False",
        gate_ab.check_lazy_status(use_a=False, a=None, b=None) == ["b"],
    )
    check(
        "H3GateAB requests nothing once the selected side is resolved",
        gate_ab.check_lazy_status(use_a=True, a="resolved", b=None) == [],
    )


def test_gate_ab_selects_a():
    sentinel = object()
    out, = H3GateAB().select(use_a=True, a=sentinel, b=object())
    check("H3GateAB use_a=True returns 'a' unchanged (is, not ==)", out is sentinel)


def test_gate_ab_selects_b():
    sentinel = object()
    out, = H3GateAB().select(use_a=False, a=object(), b=sentinel)
    check("H3GateAB use_a=False returns 'b' unchanged (is, not ==)", out is sentinel)


def test_gate_ab_raises_on_unconnected_selection():
    check_raises(
        "H3GateAB raises when 'a' is selected but not connected (explicit None)",
        lambda: H3GateAB().select(use_a=True, a=None, b=object()),
    )
    check_raises(
        "H3GateAB raises when 'a' is selected but never wired at all (omitted)",
        lambda: H3GateAB().select(use_a=True, b=object()),
    )
    check_raises(
        "H3GateAB raises when 'b' is selected but not connected",
        lambda: H3GateAB().select(use_a=False, a=object(), b=None),
    )


def test_gate_ab_does_not_fall_back_to_the_other_branch():
    # 'b' is a perfectly good value, but use_a selects 'a', which isn't connected --
    # this must raise, never silently return b.
    check_raises(
        "H3GateAB does not silently fall back to the unselected branch",
        lambda: H3GateAB().select(use_a=True, a=None, b="a real value"),
    )


# ---------------------------------------------------------------------------
# H3GateSwitch
# ---------------------------------------------------------------------------

def test_gate_switch_check_lazy_status_requests_only_selected_index():
    switch = H3GateSwitch()
    check(
        "H3GateSwitch requests only 'value_1' when select=1 and it's unresolved",
        switch.check_lazy_status(select=1, value_1=None, value_2=None) == ["value_1"],
    )
    check(
        "H3GateSwitch requests only 'value_3' when select=3 and it's unresolved",
        switch.check_lazy_status(select=3, value_1=None, value_3=None) == ["value_3"],
    )
    check(
        "H3GateSwitch requests nothing once the selected index is resolved",
        switch.check_lazy_status(select=1, value_1="resolved", value_2=None) == [],
    )
    check(
        "H3GateSwitch requests nothing for a select index with no matching key at all",
        switch.check_lazy_status(select=9, value_1=None, value_2=None) == [],
    )


def test_gate_switch_selects_by_index():
    sentinel = object()
    out, = H3GateSwitch().select_branch(select=1, value_1=sentinel, value_2=object())
    check("H3GateSwitch select=1 returns 'value_1' unchanged (is, not ==)", out is sentinel)

    out, = H3GateSwitch().select_branch(select=3, value_1=object(), value_2=object(), value_3=sentinel)
    check("H3GateSwitch select=3 returns 'value_3' unchanged (is, not ==)", out is sentinel)


def test_gate_switch_raises_on_unconnected_selection():
    check_raises(
        "H3GateSwitch raises when the selected index isn't connected (explicit None)",
        lambda: H3GateSwitch().select_branch(select=2, value_1=object(), value_2=None),
    )
    check_raises(
        "H3GateSwitch raises when the selected index was never wired at all (omitted)",
        lambda: H3GateSwitch().select_branch(select=1, value_2=object()),
    )
    check_raises(
        "H3GateSwitch raises when select points past every wired index",
        lambda: H3GateSwitch().select_branch(select=9, value_1=object(), value_2=object()),
    )


def test_gate_switch_does_not_fall_back_to_another_branch():
    check_raises(
        "H3GateSwitch does not silently fall back to an unselected branch",
        lambda: H3GateSwitch().select_branch(select=1, value_1=None, value_2="a real value"),
    )


# ---------------------------------------------------------------------------
# H3GateRoute
# ---------------------------------------------------------------------------

def test_gate_route_check_lazy_status_requests_only_selected_side():
    route = H3GateRoute()
    check(
        "H3GateRoute requests only 'a' when use_a is True",
        route.check_lazy_status(use_a=True, a=None, b=None) == ["a"],
    )
    check(
        "H3GateRoute requests only 'b' when use_a is False",
        route.check_lazy_status(use_a=False, a=None, b=None) == ["b"],
    )
    check(
        "H3GateRoute requests nothing once the selected side is resolved",
        route.check_lazy_status(use_a=False, a=None, b="resolved") == [],
    )


def test_gate_route_blocks_the_unselected_output():
    sentinel = object()
    out_a, out_b = H3GateRoute().route(use_a=True, a=sentinel, b=None)
    check("H3GateRoute use_a=True passes 'a' through unchanged (is, not ==)", out_a is sentinel)
    check("H3GateRoute use_a=True blocks output 'b'", isinstance(out_b, FakeExecutionBlocker))

    out_a, out_b = H3GateRoute().route(use_a=False, a=None, b=sentinel)
    check("H3GateRoute use_a=False passes 'b' through unchanged (is, not ==)", out_b is sentinel)
    check("H3GateRoute use_a=False blocks output 'a'", isinstance(out_a, FakeExecutionBlocker))


def test_gate_route_blocks_silently():
    _, out_b = H3GateRoute().route(use_a=True, a=object(), b=None)
    check("H3GateRoute blocks silently (message is None)", out_b.message is None)


def test_gate_route_raises_on_unconnected_selection():
    check_raises(
        "H3GateRoute raises when 'a' is selected but not connected (explicit None)",
        lambda: H3GateRoute().route(use_a=True, a=None, b=object()),
    )
    check_raises(
        "H3GateRoute raises when 'b' is selected but never wired at all (omitted)",
        lambda: H3GateRoute().route(use_a=False, a=object()),
    )


def test_gate_route_does_not_fall_back_to_the_other_branch():
    check_raises(
        "H3GateRoute does not silently route the unselected branch instead",
        lambda: H3GateRoute().route(use_a=True, a=None, b="a real value"),
    )


def test_gate_route_missing_execution_blocker_raises():
    original = gate_route_mod.ExecutionBlocker
    gate_route_mod.ExecutionBlocker = None
    try:
        check_raises(
            "H3GateRoute without ExecutionBlocker available raises a clear error",
            lambda: H3GateRoute().route(use_a=True, a=object(), b=None),
            RuntimeError,
        )
    finally:
        gate_route_mod.ExecutionBlocker = original


if __name__ == "__main__":
    test_any_type()
    test_gate_check_lazy_status()
    test_gate_enabled_passthrough_identity()
    test_gate_disabled_returns_blocker()
    test_gate_disabled_never_touches_an_unconnected_value()
    test_gate_missing_execution_blocker_raises()
    test_gate_ab_check_lazy_status_requests_only_selected_side()
    test_gate_ab_selects_a()
    test_gate_ab_selects_b()
    test_gate_ab_raises_on_unconnected_selection()
    test_gate_ab_does_not_fall_back_to_the_other_branch()
    test_gate_switch_check_lazy_status_requests_only_selected_index()
    test_gate_switch_selects_by_index()
    test_gate_switch_raises_on_unconnected_selection()
    test_gate_switch_does_not_fall_back_to_another_branch()
    test_gate_route_check_lazy_status_requests_only_selected_side()
    test_gate_route_blocks_the_unselected_output()
    test_gate_route_blocks_silently()
    test_gate_route_raises_on_unconnected_selection()
    test_gate_route_does_not_fall_back_to_the_other_branch()
    test_gate_route_missing_execution_blocker_raises()

    print(f"\n{passed} passed, {failed} failed")
    sys.exit(1 if failed else 0)
