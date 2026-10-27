"""The elemental polymorph ladders the binary and ternary phase builders take from the registry.

For each builder this pins (a) the per-component step -- which polymorphs of
``component_data[c].polymorphs`` it takes, as ``(name, spacegroup_number, enthalpy, entropy)``
in the order they appear in its output -- and (b) its full output order.

Builders:
  * ``binary.build_phases_from_chull`` on the Hf-Zr hull (every polymorph kept, the hcp
    ground states excluded, every polymorph excluded) and on Cu-Mg. Each ladder phase is a
    fresh copy of the registry's polymorph with empty ``points``.
  * ``TernaryLiquidInterpolation.get_ternary_form_en`` on Hf-Ti-Zr (every corner carries a bcc
    polymorph) and Al-Mg-Si (none does), with solid solutions off and, for Hf-Ti-Zr, on.
  * The polymorph name set the ternary colour map is built with (Hf-Ti-Zr, solid solutions
    off).

The hull editor's no-op ``apply`` is compared with ``from_cache`` in
``test_hull_editor_seeding.py``.
"""

from __future__ import annotations

import pytest

import gliquid.ternary as ternary_mod
from gliquid.binary import BinaryLiquid, build_phases_from_chull
from gliquid.ternary import TernaryLiquidInterpolation

# The package names under test, one line each.
BUILD_PHASES = build_phases_from_chull
TERNARY_FORM_EN = TernaryLiquidInterpolation.get_ternary_form_en
COLOR_MAP_BUILDER = "build_phase_color_map"

# The liquid parameters test_imputation_bit_identity.py pins.
CU_MG_PARAMS = [-41302.26, -5.1627, -8148.05, 0.0]
SS_KWARGS = {"ref_mode": "from_unary_db"}
ZERO = [0.0, 0.0, 0.0, 0.0]
TERNARY_SYSTEMS = {
    "Hf-Ti-Zr": (["Hf", "Ti", "Zr"], {"Hf-Ti": ZERO, "Ti-Zr": ZERO, "Zr-Hf": ZERO}),
    "Al-Mg-Si": (["Al", "Mg", "Si"], {"Al-Mg": ZERO, "Mg-Si": ZERO, "Si-Al": ZERO}),
}

BINARY_ORDER = {
    ("Hf-Zr", None): [
        ("beta-Hf (bcc)", 0.0),
        ("alpha-Hf (hcp)", 0.0),
        ("alpha-Zr (hcp)", 1.0),
        ("beta-Zr (bcc)", 1.0),
        ("L", None),
    ],
    ("Hf-Zr", (194,)): [("beta-Hf (bcc)", 0.0), ("beta-Zr (bcc)", 1.0), ("L", None)],
    ("Hf-Zr", (194, 229)): [("Hf", 0.0), ("Zr", 1.0), ("L", None)],
    ("Cu-Mg", None): [
        ("Cu (fcc)", 0.0),
        ("MgCu2", 0.3333333333333333),
        ("Mg2Cu", 0.6666666666666666),
        ("Mg (hcp)", 1.0),
        ("L", None),
    ],
}
BINARY_STEP = {
    ("Hf-Zr", None): {
        "Hf": [("beta-Hf (bcc)", 229, 5896, 2.9246), ("alpha-Hf (hcp)", 194, 0.0, 0.0)],
        "Zr": [("alpha-Zr (hcp)", 194, 0.0, 0.0), ("beta-Zr (bcc)", 229, 4156, 3.6585)],
    },
    ("Hf-Zr", (194,)): {
        "Hf": [("beta-Hf (bcc)", 229, 5896, 2.9246)],
        "Zr": [("beta-Zr (bcc)", 229, 4156, 3.6585)],
    },
    ("Hf-Zr", (194, 229)): {"Hf": [], "Zr": []},
    ("Cu-Mg", None): {
        "Cu": [("Cu (fcc)", 225, 0.0, 0.0)],
        "Mg": [("Mg (hcp)", 194, 0.0, 0.0)],
    },
}
BINARY_CASES = list(BINARY_ORDER)

# Ternary rows (x0, x1, S, H, "Phase Name") with solid solutions off, in output order.
TERNARY_ROWS = {
    "Hf-Ti-Zr": [
        (0.0, 0.0, 0.0, 0.0, "alpha-Hf (hcp)"),
        (1.0, 0.0, 0.0, 0.0, "alpha-Ti (hcp)"),
        (0.0, 1.0, 0.0, 0.0, "alpha-Zr (hcp)"),
        (0.0, 0.0, 2.9246, 5896.0, "beta-Hf (bcc)"),
        (1.0, 0.0, 8.9887, 10381.94, "beta-Ti (bcc)"),
        (0.0, 1.0, 3.6585, 4156.0, "beta-Zr (bcc)"),
    ],
}
TERNARY_ORDER = {
    "Hf-Ti-Zr": [
        "alpha-Hf (hcp)",
        "alpha-Ti (hcp)",
        "alpha-Zr (hcp)",
        "beta-Hf (bcc)",
        "beta-Ti (bcc)",
        "beta-Zr (bcc)",
    ],
    "Al-Mg-Si": ["Al (fcc)", "Mg (hcp)", "Mg17Al12", "Mg2Si", "MgAl2", "Si (diamond cubic)"],
}
TERNARY_STEP = {
    "Hf-Ti-Zr": {
        "Hf": [("alpha-Hf (hcp)", 0.0, 0.0), ("beta-Hf (bcc)", 5896.0, 2.9246)],
        "Ti": [("alpha-Ti (hcp)", 0.0, 0.0), ("beta-Ti (bcc)", 10381.94, 8.9887)],
        "Zr": [("alpha-Zr (hcp)", 0.0, 0.0), ("beta-Zr (bcc)", 4156.0, 3.6585)],
    },
    "Al-Mg-Si": {
        "Al": [("Al (fcc)", 0.0, 0.0)],
        "Mg": [("Mg (hcp)", 0.0, 0.0)],
        "Si": [("Si (diamond cubic)", 0.0, 0.0)],
    },
}
# Hf-Ti-Zr with solid solutions on: the phase names at each corner.
TERNARY_SS_ON_MODELS = ["BCC", "FCC", "HCP"]
TERNARY_SS_ON_CORNERS = {
    "Hf": {"ladder": [], "dft_entry": ["Hf"]},
    "Ti": {"ladder": [], "dft_entry": ["Ti"]},
    "Zr": {"ladder": [], "dft_entry": ["Zr"]},
}
COLOR_MAP_POLYMORPH_NAMES = {
    "Hf-Ti-Zr": [
        "alpha-Hf (hcp)",
        "alpha-Ti (hcp)",
        "alpha-Zr (hcp)",
        "beta-Hf (bcc)",
        "beta-Ti (bcc)",
        "beta-Zr (bcc)",
    ],
}


@pytest.fixture(scope="module")
def binaries() -> dict[str, BinaryLiquid]:
    return {
        "Hf-Zr": BinaryLiquid.from_cache(
            "Hf-Zr", pd_ind=0, solid_solutions=False, ss_kwargs=dict(SS_KWARGS)
        ),
        "Cu-Mg": BinaryLiquid.from_cache("Cu-Mg", param_format="comb-exp", params=CU_MG_PARAMS),
    }


def _binary_phases(bl, exclude):
    return BUILD_PHASES(
        bl.dft_ch, bl.components, bl.component_data, set(exclude) if exclude else None
    )


def _registry_polymorph(bl, phase):
    """The registry polymorph a ladder phase was copied from, or ``None``."""
    for comp in bl.components:
        for poly in bl.component_data[comp].polymorphs:
            if poly.name == phase.name and poly.spacegroup_number == phase.spacegroup_number:
                return comp, poly
    return None


class TestBuildPhasesFromChull:
    @pytest.mark.parametrize("case", BINARY_CASES, ids=[f"{s}-{e}" for s, e in BINARY_CASES])
    def test_output_order(self, binaries, case):
        system, exclude = case
        bl = binaries[system]
        phases = _binary_phases(bl, exclude)
        got = [
            (p.name, None if p.composition is None else float(p.fraction_in(bl.components)[0]))
            for p in phases
        ]
        assert got == BINARY_ORDER[case]

    @pytest.mark.parametrize("case", BINARY_CASES, ids=[f"{s}-{e}" for s, e in BINARY_CASES])
    def test_per_component_step(self, binaries, case):
        system, exclude = case
        bl = binaries[system]
        phases = _binary_phases(bl, exclude)
        got = {comp: [] for comp in bl.components}
        for phase in phases:
            found = _registry_polymorph(bl, phase)
            if found is not None:
                got[found[0]].append(
                    (phase.name, phase.spacegroup_number, phase.enthalpy, phase.entropy)
                )
        assert got == BINARY_STEP[case]

    def test_each_ladder_phase_is_a_fresh_copy(self, binaries):
        bl = binaries["Hf-Zr"]
        phases = _binary_phases(bl, None)
        ladder = [(p, _registry_polymorph(bl, p)) for p in phases]
        ladder = [(p, found[1]) for p, found in ladder if found is not None]
        assert len(ladder) == 4
        for phase, poly in ladder:
            assert phase is not poly
            assert phase.points == []
            phase.points.append([0.5, 1000.0])
            assert poly.points == []

    def test_every_polymorph_excluded_leaves_the_dft_entries_at_the_endpoints(self, binaries):
        bl = binaries["Hf-Zr"]
        phases = _binary_phases(bl, (194, 229))
        endpoints = [p for p in phases if p.composition is not None]
        assert [p.name for p in endpoints] == ["Hf", "Zr"]
        assert all(p.spacegroup_number is None for p in endpoints)


def _ternary(components, xs_mix, **kwargs) -> TernaryLiquidInterpolation:
    return TernaryLiquidInterpolation(
        list(components),
        delta=0.25,
        interp_scheme="linear",
        param_format="linear",
        xs_mix=xs_mix,
        T_incr=10,
        order="given",
        **kwargs,
    )


@pytest.fixture(scope="module")
def ternaries() -> dict:
    out = {}
    for name, (components, xs_mix) in TERNARY_SYSTEMS.items():
        ti = _ternary(components, xs_mix)
        ti.interpolate()
        out[(name, "ss_off")] = ti
    components, xs_mix = TERNARY_SYSTEMS["Hf-Ti-Zr"]
    ti = _ternary(components, xs_mix, solid_solutions=True)
    ti.interpolate()
    out[("Hf-Ti-Zr", "ss_on")] = ti
    return out


def _rows(ti) -> list[tuple]:
    frame = TERNARY_FORM_EN(ti, ti.components)
    return [
        (float(r["x0"]), float(r["x1"]), float(r["S"]), float(r["H"]), str(r["Phase Name"]))
        for _i, r in frame.iterrows()
    ]


def _corner(components, comp) -> tuple[float, float]:
    i = list(components).index(comp)
    return tuple(1.0 if j == i else 0.0 for j in range(1, len(components)))


def _ladder_names(comp) -> set[str]:
    from gliquid.phase import UNARY

    return {p.name for p in UNARY[comp].polymorphs}


class TestTernaryFormEnergies:
    @pytest.mark.parametrize("system", list(TERNARY_SYSTEMS))
    def test_output_order(self, ternaries, system):
        assert [r[4] for r in _rows(ternaries[(system, "ss_off")])] == TERNARY_ORDER[system]

    def test_hf_ti_zr_rows(self, ternaries):
        assert _rows(ternaries[("Hf-Ti-Zr", "ss_off")]) == TERNARY_ROWS["Hf-Ti-Zr"]

    @pytest.mark.parametrize("system", list(TERNARY_SYSTEMS))
    def test_per_component_step(self, ternaries, system):
        ti = ternaries[(system, "ss_off")]
        rows = _rows(ti)
        got = {
            comp: [
                (r[4], r[3], r[2])
                for r in rows
                if (r[0], r[1]) == _corner(ti.components, comp) and r[4] in _ladder_names(comp)
            ]
            for comp in ti.components
        }
        assert got == TERNARY_STEP[system]

    @pytest.mark.parametrize("system", list(TERNARY_SYSTEMS))
    def test_no_dft_elemental_row_at_a_claimed_corner(self, ternaries, system):
        ti = ternaries[(system, "ss_off")]
        rows = _rows(ti)
        for comp in ti.components:
            at = [r[4] for r in rows if (r[0], r[1]) == _corner(ti.components, comp)]
            assert at, comp
            assert set(at) <= _ladder_names(comp), comp

    def test_solid_solutions_on_corners(self, ternaries):
        ti = ternaries[("Hf-Ti-Zr", "ss_on")]
        assert sorted(ti.ss_models) == TERNARY_SS_ON_MODELS
        rows = _rows(ti)
        got = {}
        for comp in ti.components:
            at = [r[4] for r in rows if (r[0], r[1]) == _corner(ti.components, comp)]
            got[comp] = {
                "ladder": [n for n in at if n in _ladder_names(comp)],
                "dft_entry": [n for n in at if n not in _ladder_names(comp)],
            }
        assert got == TERNARY_SS_ON_CORNERS


def test_color_map_polymorph_names(monkeypatch):
    """The polymorph names ``process_data`` hands the colour-map builder."""
    captured = []
    original = getattr(ternary_mod, COLOR_MAP_BUILDER)

    def spy(*args, **kwargs):
        captured.append(sorted(kwargs.get("polymorph_names") or ()))
        return original(*args, **kwargs)

    monkeypatch.setattr(ternary_mod, COLOR_MAP_BUILDER, spy)
    components, xs_mix = TERNARY_SYSTEMS["Hf-Ti-Zr"]
    ti = _ternary(components, xs_mix)
    ti.interpolate()
    ti.process_data()
    assert captured, "process_data never built a colour map"
    assert captured[0] == COLOR_MAP_POLYMORPH_NAMES["Hf-Ti-Zr"]
