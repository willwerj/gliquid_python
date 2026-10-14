"""Unit tests for the dimension-agnostic imputation core (``gliquid.imputation.core``).

Envelope values are checked against an independent oracle: a linear program over the same
points (the envelope is ``min sum(w g)`` subject to ``sum(w) = 1``, ``sum(w x) = x_q``,
``w >= 0``), and in one dimension against ``np.interp`` over a monotone-chain lower hull.
Runs offline against the shipped Cu-Mg and Al-Mg-Si data fixtures.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from pymatgen.analysis.phase_diagram import PhaseDiagram
from pymatgen.core import Composition
from scipy.optimize import linprog

import gliquid.api as api
import gliquid.imputation.binary_imputation as imp_binary
import gliquid.imputation.core as imp_core
from gliquid.binary import BinaryLiquid
from gliquid.phase import EV_ATOM_TO_J_MOL

R = 8.314


def lp_envelope(points, g, x_query) -> float:
    pts = np.atleast_2d(np.asarray(points, dtype=float))
    if pts.shape[0] == 1:
        pts = pts.T
    a_eq = np.vstack([np.ones(len(g)), pts.T])
    b_eq = np.append(1.0, np.atleast_1d(x_query))
    res = linprog(
        np.asarray(g, dtype=float), A_eq=a_eq, b_eq=b_eq, bounds=(0, None), method="highs"
    )
    assert res.success
    return float(res.fun)


def chain_lower_hull(x, g):
    order = np.argsort(x)
    hull = []
    for i in order:
        while len(hull) >= 2:
            (x1, g1), (x2, g2) = hull[-2], hull[-1]
            if (x2 - x1) * (g[i] - g1) - (g2 - g1) * (x[i] - x1) <= 0:
                hull.pop()
            else:
                break
        hull.append((x[i], g[i]))
    return np.array(hull)


def liquid_curve(x, temp=1000.0):
    xi = np.clip(x, 1e-12, 1 - 1e-12)
    return (
        -20000 * x * (1 - x) + R * temp * (xi * np.log(xi) + (1 - xi) * np.log(1 - xi)) + 3000 * x
    )


# -- Compositions --


class TestNormalizeFractions:
    COMPONENTS = ["La", "Mn", "Si"]

    @pytest.mark.parametrize(
        "spec",
        [
            "LaMnSi2",
            Composition("LaMnSi2"),
            {"La": 1, "Mn": 1, "Si": 2},
            {"La": 0.25, "Mn": 0.25, "Si": 0.5},
            [1, 1, 2],
            [0.25, 0.5],
        ],
    )
    def test_every_form_gives_the_same_fractions(self, spec):
        got = imp_core.normalize_fractions(spec, self.COMPONENTS)
        assert list(got) == self.COMPONENTS
        assert got == pytest.approx({"La": 0.25, "Mn": 0.25, "Si": 0.5})

    def test_missing_elements_are_zero(self):
        assert imp_core.normalize_fractions("LaSi", self.COMPONENTS)["Mn"] == 0.0

    def test_binary_scalar_is_the_second_component(self):
        got = imp_core.normalize_fractions(0.3, ["Cu", "Mg"])
        assert got == {"Cu": 1.0 - 0.3, "Mg": 0.3}

    @pytest.mark.parametrize(
        "spec, components",
        [
            ("LaFeSi", ["La", "Mn", "Si"]),
            ({"La": -1, "Mn": 1, "Si": 1}, ["La", "Mn", "Si"]),
            ([0, 0, 0], ["La", "Mn", "Si"]),
            ([0.1], ["La", "Mn", "Si"]),
            (0.3, ["La", "Mn", "Si"]),
        ],
    )
    def test_invalid_input_raises(self, spec, components):
        with pytest.raises(ValueError):
            imp_core.normalize_fractions(spec, components)

    def test_independent_coords(self):
        fr = imp_core.normalize_fractions("LaMnSi2", self.COMPONENTS)
        assert imp_core.independent_coords(fr, self.COMPONENTS).tolist() == [0.25, 0.5]


# -- Envelope --


class TestLowerEnvelope:
    def test_one_dimension_matches_interp_over_the_lower_hull(self):
        x = np.linspace(0.0, 1.0, 101)
        g = liquid_curve(x)
        hull = chain_lower_hull(x, g)
        for xq in (0.0, 0.123, 0.5, 0.987, 1.0):
            env = imp_core.lower_envelope_at(x, g, [xq])
            assert env.value == pytest.approx(np.interp(xq, hull[:, 0], hull[:, 1]), abs=1e-6)

    def test_one_dimension_matches_the_lp_oracle(self):
        rng = np.random.default_rng(3)
        x = np.sort(rng.random(40))
        x[0], x[-1] = 0.0, 1.0
        g = rng.normal(0.0, 5000.0, 40)
        for xq in (0.05, 0.4, 0.77):
            assert imp_core.lower_envelope_at(x, g, xq).value == pytest.approx(
                lp_envelope(x, g, xq), abs=1e-6
            )

    def test_two_dimensions_match_the_lp_oracle(self):
        rng = np.random.default_rng(7)
        corners = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
        interior = rng.dirichlet([1, 1, 1], 60)[:, 1:]
        pts = np.vstack([corners, interior])
        g = rng.normal(-10000.0, 8000.0, len(pts))
        for xq in ([0.2, 0.3], [0.45, 0.45], [0.05, 0.9]):
            assert imp_core.lower_envelope_at(pts, g, xq).value == pytest.approx(
                lp_envelope(pts, g, xq), abs=1e-5
            )

    def test_support_reproduces_the_query(self):
        rng = np.random.default_rng(11)
        pts = np.vstack([[[0, 0], [1, 0], [0, 1]], rng.dirichlet([1, 1, 1], 30)[:, 1:]])
        g = rng.normal(0.0, 1000.0, len(pts))
        xq = np.array([0.3, 0.25])
        env = imp_core.lower_envelope_at(pts, g, xq)
        w = np.array(env.weights)
        assert w.min() >= -1e-9
        assert w.sum() == pytest.approx(1.0)
        assert w @ pts[list(env.rows)] == pytest.approx(xq)
        assert w @ g[list(env.rows)] == pytest.approx(env.value, abs=1e-6)

    def test_too_few_points_raise(self):
        with pytest.raises(ValueError):
            imp_core.lower_envelope_at([[0.0, 0.0], [1.0, 0.0]], [0.0, 0.0], [0.5, 0.0])


class TestEnvelopeEnergy:
    def table(self):
        x = np.linspace(0.0, 1.0, 51)
        liquid = pd.DataFrame(
            {"X": x, "S": 5.0 + 0.0 * x, "H": -5000 * x * (1 - x), "Phase Name": "L"}
        )
        solid = pd.DataFrame({"X": [0.5], "S": [0.0], "H": [-3000.0], "Phase Name": ["AB"]})
        return pd.concat([liquid, solid], ignore_index=True)

    def test_uses_g_equals_h_minus_ts(self):
        table = self.table()
        env = imp_core.envelope_energy(table, [0.25], 1000.0, comp_cols=["X"], exclude=["AB"])
        g = table["H"] - 1000.0 * table["S"]
        assert env.value == pytest.approx(lp_envelope(table["X"][:-1], g[:-1], 0.25), abs=1e-6)
        assert set(env.phases) == {"L"}
        assert env.entropy == pytest.approx(5.0)

    def test_exclude_changes_the_support(self):
        table = self.table()
        with_solid = imp_core.envelope_energy(table, [0.5], 100.0, comp_cols=["X"])
        without = imp_core.envelope_energy(table, [0.5], 100.0, comp_cols=["X"], exclude=["AB"])
        assert "AB" in with_solid.phases
        assert with_solid.value == pytest.approx(-3000.0)
        assert without.value > with_solid.value


# -- Tangent --


class TestTangentValue:
    def test_congruent_returns_the_liquid(self):
        g = lambda x: float(np.sum(x**2)) * 1000.0  # noqa: E731
        assert imp_core.tangent_value(g, [0.3, 0.3], [0.3, 0.3]) == g(np.array([0.3, 0.3]))

    def test_exact_on_a_linear_liquid(self):
        slope = np.array([1200.0, -800.0])
        g = lambda x: 500.0 + float(slope @ x)  # noqa: E731
        got = imp_core.tangent_value(g, [0.1, 0.6], [0.4, 0.2])
        assert got == pytest.approx(g(np.array([0.1, 0.6])), abs=1e-6)

    def test_one_dimension_is_the_binary_difference(self):
        g = lambda x: float(liquid_curve(np.asarray(x))[0])  # noqa: E731
        x_inv, x_s, dx = 0.0004, 0.2, 1e-3
        xl = min(max(x_inv - dx, 1e-6), 1.0 - 1e-6)
        xr = min(max(x_inv + dx, 1e-6), 1.0 - 1e-6)
        expected = g([x_inv]) + (g([xr]) - g([xl])) / (xr - xl) * (x_s - x_inv)
        assert imp_core.tangent_value(g, [x_s], [x_inv], dx) == expected

    def test_anchor_on_an_edge_steps_inward(self):
        g = lambda x: float(np.sum(x**2))  # noqa: E731
        got = imp_core.tangent_value(g, [0.3, 0.3], [0.5, 0.5])  # anchor on the B-C edge
        assert np.isfinite(got)

    def test_anchor_outside_the_simplex_raises(self):
        with pytest.raises(ValueError):
            imp_core.tangent_value(lambda x: 0.0, [0.3, 0.3], [0.7, 0.7])


# -- Entries and hulls --


@pytest.fixture(scope="module")
def cu_mg():
    bl = BinaryLiquid.from_cache("Cu-Mg", param_format="comb-exp")
    if bl.init_error:
        pytest.skip("Cu-Mg: init error on load")
    return bl


@pytest.fixture(scope="module")
def al_mg_si_hull():
    hull, _ = api.get_dft_convexhull(["Al", "Mg", "Si"])
    return hull


class TestEntriesOnAHull:
    def test_entry_at_enthalpy_round_trips_through_reanchor(self, al_mg_si_hull):
        fractions = {"Al": 0.2, "Mg": 0.4, "Si": 0.4}
        entry = imp_core.tagged_entry_dict(
            Composition(fractions), 0.0, "Probe", {"imputed": True, "fractions": fractions}
        )
        moved = imp_core.entry_at_enthalpy(al_mg_si_hull, entry, -12345.0)
        (rebuilt,) = imp_core.reanchor_imputed_entries(al_mg_si_hull, [moved])
        form = al_mg_si_hull.get_form_energy_per_atom(rebuilt) * EV_ATOM_TO_J_MOL
        assert form == pytest.approx(-12345.0, abs=1e-6)
        assert moved["entry_id"] == "imputed:Probe"

    def test_phase_diagram_enthalpy_matches_the_phase_list_enthalpy(self, cu_mg):
        for x in (0.1, 0.4, 0.8):
            fr = {"Cu": 1.0 - x, "Mg": x}
            assert imp_core.phase_diagram_enthalpy_at(cu_mg.dft_ch, fr) == pytest.approx(
                imp_binary._phase_list_enthalpy_at(cu_mg, x), abs=1e-6
            )

    def test_protected_clearance_without_protected_compounds_is_infinite(self, cu_mg):
        assert imp_core.protected_clearance(cu_mg.dft_ch, set()) == float("inf")

    def test_augmented_hull_keeps_the_element_order(self, cu_mg):
        hull = imp_core.augmented_hull(cu_mg.dft_ch, [])
        assert list(hull.elements) == list(cu_mg.dft_ch.elements)
        assert isinstance(hull, PhaseDiagram)

    def test_hull_with_reanchored_entries_matches_the_inline_rebuild(self, cu_mg):
        hull = cu_mg.dft_ch
        a, b = cu_mg.components
        dicts = [
            imp_core.tagged_entry_dict(
                Composition({a: 1.0 - x, b: x}),
                imp_core.hull_ref_energy_per_atom(hull, {a: 1.0 - x, b: x}) + h / EV_ATOM_TO_J_MOL,
                name,
                {"imputed": True, "label": name, "x_b": x, "h_form_j_per_mol": h},
            )
            for name, x, h in (("P2", 0.6, -40000.0), ("P1", 0.2, -30000.0))
        ]
        rebuilt, entries = imp_core.hull_with_reanchored_entries(hull, dicts)
        inline = PhaseDiagram(
            list(hull.all_entries) + imp_core.reanchor_imputed_entries(hull, dicts),
            elements=list(hull.elements),
        )
        assert list(rebuilt.elements) == list(hull.elements)
        assert [e.entry_id for e in entries] == ["imputed:P2", "imputed:P1"]
        assert [e.entry_id for e in rebuilt.all_entries] == [e.entry_id for e in inline.all_entries]
        assert [e.energy for e in rebuilt.all_entries] == [e.energy for e in inline.all_entries]
        assert {e.entry_id for e in rebuilt.stable_entries} == {
            e.entry_id for e in inline.stable_entries
        }


class TestAcceptGrowing:
    def ops(self, stable_sets):
        """Stub ops: the hull for ``n`` accumulated entries carries ``stable_sets[n]``."""

        class Hull:
            def __init__(self, ids):
                self.stable_entries = [type("E", (), {"entry_id": i})() for i in ids]

        return imp_core.HullOps(
            augmented=lambda dicts: Hull(stable_sets[len(dicts)]),
            hull_enthalpy_at=lambda ed: 0.0,
            entry_at_enthalpy=lambda ed, h: ed,
            protected_depth=lambda hull, protected: float("inf"),
        )

    def cand(self, name, margin):
        return {"entry_dict": {"entry_id": f"imputed:{name}"}, "name": name, "margin_j": margin}

    def test_strongest_first_and_displacement(self):
        ops = self.ops({1: {"imputed:B", "keep"}, 2: {"imputed:B", "imputed:A"}})
        accepted, rejected = imp_core.accept_growing(
            ops,
            [self.cand("A", 1.0), self.cand("B", 5.0)],
            {"keep"},
            protect=True,
            clamp=lambda ed, acc, prot: None,
            vertex_noun="protected vertex/vertices",
        )
        assert [e["entry_id"] for e in accepted] == ["imputed:B"]
        assert rejected == [
            (
                "A",
                "would displace 1 protected vertex/vertices and no clamped energy clears the hull: keep",
            )
        ]

    def test_unprotected_admits_the_displacer(self):
        ops = self.ops({1: {"imputed:A"}})
        accepted, _ = imp_core.accept_growing(
            ops, [self.cand("A", 1.0)], {"keep"}, protect=False, clamp=lambda *a: None
        )
        assert [e["entry_id"] for e in accepted] == ["imputed:A"]


def test_provenance_header():
    header = imp_core.provenance_header("x")
    assert list(header.items()) == [
        ("calibration", "x"),
        ("gliquid_version", imp_core.package_version()),
    ]


def test_params_moved():
    assert imp_core.params_moved([1.0, 2.0], [1.0, 2.0], 1e-9) is False
    assert imp_core.params_moved([1.0, 2.0], [1.0, 2.1], 1e-9) is True
    assert imp_core.params_moved([1.0], [1.0, 2.0], 1e-9) is None
