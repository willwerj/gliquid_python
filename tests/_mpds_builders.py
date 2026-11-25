"""Builders shared by the test modules: synthetic MPDS records, shapes and phase dicts, small
hulls, figure captures, the constraint-equation readers and the shared fixtures.

Import from a module in ``tests/`` with ``from ._mpds_builders import ...``; a module loaded by
file path puts ``tests/`` on ``sys.path`` first and imports ``_mpds_builders``. The fixtures
reach the tests through ``tests/conftest.py``.
"""

from __future__ import annotations

import contextlib
import os
from functools import partial
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
import sympy as sp
from pymatgen.analysis.phase_diagram import PhaseDiagram
from pymatgen.core import Composition, Element
from pymatgen.entries.computed_entries import ComputedEntry

import gliquid.api as api
import gliquid.cache as cache
import gliquid.config as config
import gliquid.imputation.binary_imputation as imp_binary
import gliquid.mpds as mpds
from gliquid.binary import BinaryLiquid, build_phases_from_chull
from gliquid.solid_coverage import SolidCoverageReport
from gliquid.solution import l_sym, x_vals
from gliquid.ternary import TernaryLiquidInterpolation

# -- MPDS records and shapes --
#: Synthetic MPDS diagrams written by make_dummy_mpds.py; the suite reads the MPDS records of
#: the repository ``cache/`` from here.
MPDS_DUMMY_DIR = Path(__file__).resolve().parent / "fixtures" / "mpds_dummy"
REPO_CACHE_DIR = Path(__file__).resolve().parents[1] / "cache"


@contextlib.contextmanager
def dummy_mpds_store():
    """Serve every MPDS record of the repository ``cache/`` from :data:`MPDS_DUMMY_DIR`.

    While active, the configured directory store answers MPDS reads, existence checks and
    variant listings for ``cache/`` from the dummy directory alone, so an MPDS file in
    ``cache/`` is never read; an MPDS write there raises. Other kinds of record, and every
    other store, are untouched.
    """
    backend = cache._GLOBAL_DIRECTORY_BACKEND
    dummy = cache.DirectoryBackend(MPDS_DUMMY_DIR)
    repo = os.path.normcase(str(REPO_CACHE_DIR.resolve()))
    resolved: dict[str, bool] = {}
    path_for, variants, write_json = backend.path_for, backend.variants, backend.write_json

    def routed(kind):
        if kind != cache.KIND_MPDS or config.cache_dir is None:
            return False
        root = str(config.cache_dir)
        if root not in resolved:
            resolved[root] = os.path.normcase(str(Path(root).resolve())) == repo
        return resolved[root]

    def routed_path_for(key):
        return dummy.path_for(key) if routed(key.kind) else path_for(key)

    def routed_variants(sys_name, kind):
        return dummy.variants(sys_name, kind) if routed(kind) else variants(sys_name, kind)

    def routed_write_json(key, payload):
        if routed(key.kind):
            raise AssertionError(f"a test wrote the MPDS record {key} into the repository cache")
        return write_json(key, payload)

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(backend, "path_for", routed_path_for)
        patch.setattr(backend, "variants", routed_variants)
        patch.setattr(backend, "write_json", routed_write_json)
        yield


def svgpath(points):
    """``[[x_pct, T_celsius], ...]`` -> MPDS-style svgpath string."""
    return "M " + " L ".join(f"{x},{t}" for x, t in points)


def solid_shape(label, path, kind, phase=None, with_phase=True):
    """A one-phase solid shape; ``with_phase=False`` leaves out the ``phase`` key."""
    shape = {"nphases": 1, "is_solid": True, "kind": kind, "label": label}
    if with_phase:
        shape["phase"] = phase
    shape["svgpath"] = path
    return shape


def line_shape(label, x, t_lo, t_hi, phase=None, with_phase=True):
    """A line compound: a vertical at ``x`` at.% from ``t_lo`` to ``t_hi`` degC."""
    return solid_shape(label, f"M {x},{t_lo} L {x},{t_hi}", "compound", phase, with_phase)


def field_shape(label, points, phase=None, with_phase=True):
    """A one-phase solid field (``kind='phase'``) bounded by ``points``."""
    return solid_shape(label, svgpath(points), "phase", phase, with_phase)


bare_line_shape = partial(line_shape, with_phase=False)
bare_field_shape = partial(field_shape, with_phase=False)


def liquid_shape(points, label="L", kind="phase", **extra):
    """A liquid shape through ``points``; ``extra`` adds keys."""
    return {
        "label": label,
        "nphases": 1,
        "is_solid": False,
        "kind": kind,
        "svgpath": svgpath(points),
        **extra,
    }


def mpds_record(shapes, elements, temp, labels=(), reference="synthetic", comp_range=(0.0, 100.0)):
    """A synthetic MPDS record holding ``shapes`` and ``labels`` on a ``temp`` degC frame."""
    return {
        "reference": {"entry": reference},
        "chemical_elements": list(elements),
        "temp": list(temp),
        "comp_range": list(comp_range),
        "shapes": list(shapes),
        "labels": [list(entry) for entry in labels],
    }


def liquidus_record(shapes, elements=("Hf", "Zr"), temp=(0, 2600)):
    """:func:`mpds_record` with the ``synthetic-test-json`` reference and integer comp range."""
    return mpds_record(shapes, elements, temp, reference="synthetic-test-json", comp_range=(0, 100))


def stub_mpds_load(monkeypatch, record):
    """Make ``mpds.load_mpds_data`` return ``record`` and its extracted liquidus."""
    monkeypatch.setattr(
        mpds,
        "load_mpds_data",
        lambda system, pd_ind=None: (record, mpds.extract_digitized_liquidus(record)),
    )


def from_record(monkeypatch, record, system="Hf-Zr", **kwargs):
    """``BinaryLiquid.from_cache(system, **kwargs)`` over a synthetic MPDS ``record``."""
    stub_mpds_load(monkeypatch, record)
    return BinaryLiquid.from_cache(system, **kwargs)


def collapsed_phases(record, **kwargs):
    """``mpds.collapse_polymorphs`` over ``record``'s phases; ``kwargs`` go to it."""
    return mpds.collapse_polymorphs(
        mpds.identify_mpds_phases(record, with_polymorph=True), record, **kwargs
    )


def unpaired_ht_forms(record, **kwargs):
    """``mpds.mark_unpaired_ht_forms`` over ``record``'s phases; ``kwargs`` go to it."""
    return mpds.mark_unpaired_ht_forms(
        mpds.identify_mpds_phases(record, with_polymorph=True),
        record["chemical_elements"],
        mpds_json=record,
        **kwargs,
    )


def lc_phase(name, comp, t_lo=300.0, t_hi=1500.0, spacegroup=None, pearson=None):
    """A digitized line-compound phase dict at ``comp``."""
    return {
        "type": "lc",
        "name": name,
        "comp": comp,
        "tbounds": [[comp, t_lo], [comp, t_hi]],
        "structure": {"prototype": name, "spacegroup": spacegroup, "pearson": pearson},
    }


def ss_phase(name, lo, hi, comp=None, t_lo=300.0, t_hi=1500.0, spacegroup=None):
    """A digitized solid-solution phase dict spanning ``lo`` to ``hi``."""
    comp = 0.5 * (lo + hi) if comp is None else comp
    return {
        "type": "ss",
        "name": name,
        "comp": comp,
        "tbounds": [[comp, t_lo], [comp, t_hi]],
        "cbounds": [[lo, t_lo], [hi, t_hi]],
        "structure": {"prototype": name, "spacegroup": spacegroup, "pearson": None},
    }


def melting_inv(kind, liq_comp, phase_name, phase_comp):
    """A ``'cmp'`` or ``'per'`` invariant record naming the phase it melts."""
    return {
        "type": kind,
        "comp": liq_comp,
        "temp": 1200.0,
        "phases": [phase_name],
        "phase_comps": [phase_comp],
    }


def eutectic_inv(comp, phases=(None, None), phase_comps=(None, None)):
    """A eutectic record, which names its two ADJACENT phases rather than one it melts."""
    return {
        "type": "eut",
        "comp": comp,
        "temp": 1000.0,
        "phases": list(phases),
        "phase_comps": list(phase_comps),
    }


def coverage_report(ranges, floor, fraction=0.0, ignored=None, span=(0.0, 1.0), slope=0.0):
    """A ``SolidCoverageReport`` with ``ranges`` unsupported, ``min_supported_total=floor`` and
    ``ignored_floor_slope=slope``."""
    return SolidCoverageReport(
        span=span,
        unsupported_fraction=fraction,
        unsupported_ranges=[list(r) for r in ranges],
        n_compounds=0,
        n_missing_compounds=0,
        phases=[],
        ss_models=(),
        thresholds={"min_supported_total": floor, "ignored_floor_slope": slope},
        ignored_ranges=None if ignored is None else [list(r) for r in ignored],
    )


# -- hulls and binary models --
def compound_hull(endpoints, compound, energy=-0.3, elements=None):
    """A two-element hull: both ``endpoints`` at 0 eV plus ``compound`` at ``energy``.

    ``elements`` sets the hull's element order (default ``endpoints``).
    """
    entries = [
        ComputedEntry(Composition(endpoints[0]), 0.0),
        ComputedEntry(Composition(endpoints[1]), 0.0),
        ComputedEntry(Composition(compound), energy),
    ]
    return PhaseDiagram(entries, elements=[Element(el) for el in elements or endpoints])


def strip_interior(bl: BinaryLiquid) -> BinaryLiquid:
    """Remove every interior DFT compound from ``bl``'s hull and phase list, in place."""
    keep = [e for e in bl.dft_ch.all_entries if len(e.composition.elements) == 1]
    bl.dft_ch = PhaseDiagram(keep, elements=list(bl.dft_ch.elements))
    bl.phases = build_phases_from_chull(bl.dft_ch, bl.components, bl.component_data)
    pts = imp_binary._hull_points(bl)
    bl.eqs["h_hull_interp"] = np.interp(x_vals[1:-1], pts[:, 0], pts[:, 1])
    bl.invariants = None
    bl.low_t_exp_phases = None
    bl.hsx = None
    return bl


def imputed_entry(x_b=0.66, energy=-0.3, label="Mg2Cu"):
    """A tagged imputed Cu-Mg ``ComputedEntry`` at ``x_b`` with ``energy`` eV/atom."""
    comp = Composition({"Cu": 1 - x_b, "Mg": x_b})
    return ComputedEntry(
        comp,
        energy * comp.num_atoms,
        entry_id=f"imputed:{label}",
        data={"imputed": True, "label": label, "x_b": x_b},
    )


def cu_mg_entry_pool():
    """Cu, Mg and CuMg2 entry dicts without structures, enough to build a hull."""
    return [
        ComputedEntry(Composition(f), e, entry_id=f"s-{f}").as_dict()
        for f, e in (("Cu", 0.0), ("Mg", 0.0), ("CuMg2", -0.9))
    ]


def without_module_paths(value):
    """``value`` with every ``@module`` key dropped; pymatgen versions differ in the module
    path they serialize for the same class."""
    if isinstance(value, dict):
        return {k: without_module_paths(v) for k, v in value.items() if k != "@module"}
    if isinstance(value, list):
        return [without_module_paths(v) for v in value]
    return value


def _pin_float(value):
    """The float a pinned leaf holds (a float, or a ``"float:<repr>"`` string), else ``None``."""
    if isinstance(value, float):
        return value
    if isinstance(value, str) and value.startswith(("float:", "float64:")):
        return float(value.split(":", 1)[1])
    return None


def matches_pin(actual, pinned, rel=1e-9):
    """Whether ``actual`` equals ``pinned``: floats to relative ``rel``, every other leaf, key
    and length exactly."""
    if isinstance(actual, dict) and isinstance(pinned, dict):
        return actual.keys() == pinned.keys() and all(
            matches_pin(actual[k], pinned[k], rel) for k in actual
        )
    if isinstance(actual, list) and isinstance(pinned, list):
        return len(actual) == len(pinned) and all(
            matches_pin(a, p, rel) for a, p in zip(actual, pinned)
        )
    a, p = _pin_float(actual), _pin_float(pinned)
    if a is not None and p is not None:
        return a == p or abs(a - p) <= rel * max(abs(a), abs(p))
    return actual == pinned


_IDEAL = [0.0, 0.0, 0.0, 0.0]
IDEAL_MIX = {
    "Hf-Ti-Zr": {"Hf-Ti": _IDEAL, "Ti-Zr": _IDEAL, "Zr-Hf": _IDEAL},
    "Al-Mg-Si": {"Al-Mg": _IDEAL, "Mg-Si": _IDEAL, "Si-Al": _IDEAL},
}


def ideal_ternary(sys_name, hull_mode="gtx", **kwargs):
    """An ideal-solution ``TernaryLiquidInterpolation`` of a shipped ternary at delta 0.25."""
    return TernaryLiquidInterpolation(
        sys_name.split("-"),
        delta=0.25,
        interp_scheme="linear",
        param_format="linear",
        xs_mix=IDEAL_MIX[sys_name],
        T_incr=100,
        hull_mode=hull_mode,
        order="given",
        **kwargs,
    )


# -- constraint equations --
R = 8.314
PARAM_SYMS = [l_sym(k) for k in ("L0_a", "L0_b", "L1_a", "L1_b")]


def recovered_L0_a(entry, params):
    """``L0_a`` solved from the entry's equation with the other three parameters fixed."""
    L0_a, L0_b, L1_a, L1_b = PARAM_SYMS
    equation = entry[3].subs({L0_b: params[1], L1_a: params[2], L1_b: params[3]})
    (value,) = sp.solve(equation, L0_a)
    return float(value)


def residual_rt(entry, params):
    """|lhs - rhs| of the entry's equation at ``params``, in units of RT at its temperature."""
    eq = entry[3]
    return abs(float((eq.lhs - eq.rhs).subs(dict(zip(PARAM_SYMS, params))))) / (R * entry[2])


def by_order(entries):
    """Constraint entries keyed by their order."""
    return {entry[1]: entry for entry in entries}


# -- T-x figures --
def facet_rows(t, *vertices):
    """One hull facet as the three ``df_tx`` rows that share its temperature."""
    return [[x / 100.0, float(t), label] for x, label in vertices]


class FakeHsx:
    """The attributes ``plot_tx`` reads, over a hand-built ``df_tx`` in fractions and C."""

    def __init__(self, rows, phases, inv_points, conds=(0.0, 1500.0)):
        self.df_tx = pd.DataFrame(rows, columns=["x", "t", "label"])
        # every solid at H = 0 (on the 0 K hull), the liquid above it
        first = self.df_tx.drop_duplicates("label")
        self.df = pd.DataFrame(
            {
                "X [Fraction]": first["x"].to_numpy(dtype=float),
                "H [J/mol]": (first["label"] == "L").astype(float).to_numpy(),
                "Phase": first["label"].to_numpy(),
            }
        )
        self.phases = list(phases)
        self.comps = ["A", "B"]
        self.conds = list(conds)
        self._inv = inv_points

    def liquidus_invariants(self):
        return self._inv, [], {}


def band_bounds(collection):
    """``(x0, x1, y0, y1)`` of a ``fill_betweenx`` band's path, in data units."""
    vertices = collection.get_paths()[0].vertices
    return (
        float(vertices[:, 0].min()),
        float(vertices[:, 0].max()),
        float(vertices[:, 1].min()),
        float(vertices[:, 1].max()),
    )


def trace_row(tr):
    """``[type, mode, name, line color, line dash, len(x)]`` of one plotly trace."""
    line = getattr(tr, "line", None)
    x = getattr(tr, "x", None)
    return [
        tr.type,
        getattr(tr, "mode", None),
        getattr(tr, "name", None),
        getattr(line, "color", None) if line is not None else None,
        getattr(line, "dash", None) if line is not None else None,
        (len(x) if x is not None else None),
    ]


def capture_figure(fig):
    """Trace rows, sorted annotation texts and y-range of a plotly figure."""
    anns = sorted(a.text for a in (fig.layout.annotations or ()) if a.text is not None)
    yr = fig.layout.yaxis.range
    return {
        "n_traces": len(fig.data),
        "traces": [trace_row(tr) for tr in fig.data],
        "annotations": anns,
        "yaxis_range": list(yr) if yr is not None else None,
    }


# -- environment --
class Tripwire:
    """Stands in for a network client: counts the calls and raises on each one."""

    def __init__(self):
        self.calls = 0

    def __call__(self, *args, **kwargs):
        self.calls += 1
        raise AssertionError("a network client was constructed")


class FakeMPRester:
    """A context-managed ``MPRester`` whose ``get_entries_in_chemsys`` returns ``fetch(...)``.

    ``fetch`` is called with ``(elements, additional_criteria)``.
    """

    def __init__(self, fetch):
        self._fetch = fetch

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def get_entries_in_chemsys(self, elements, additional_criteria=None):
        return self._fetch(elements, additional_criteria)


def install_fake_mprester(monkeypatch, fetch):
    """Route ``api.mp_rester`` to a ``FakeMPRester`` over ``fetch``, with a dummy API key."""
    monkeypatch.setattr(api, "mp_rester", lambda api_key=None: FakeMPRester(fetch))
    monkeypatch.setattr(api, "get_api_key", lambda name: "0" * 32)


@contextlib.contextmanager
def hermetic_mpds():
    """Online mode with no API key and tripwired clients, restored on exit: an uncached MPDS
    record takes ``load_mpds_data``'s no-key path, and nothing is fetched or written."""
    saved = (config.offline, api.get_api_key, api.get_mpds_client, api.mp_rester)
    config.set_offline(False)
    api.get_api_key = lambda name: None
    api.get_mpds_client = Tripwire()
    api.mp_rester = Tripwire()
    try:
        yield
    finally:
        config.set_offline(saved[0])
        api.get_api_key, api.get_mpds_client, api.mp_rester = saved[1:]


def notebooks_dir() -> Path | None:
    """``notebooks/`` under the repository root, or None where it does not ship."""
    for parent in Path(__file__).resolve().parents:
        if parent.name == "gliquid_python":
            candidate = parent / "notebooks"
            return candidate if candidate.is_dir() else None
    return None


def shipped_notebooks() -> list[Path]:
    """The shipped ``*.ipynb`` files, sorted; empty where ``notebooks/`` does not ship."""
    nb_dir = notebooks_dir()
    return sorted(nb_dir.glob("*.ipynb")) if nb_dir else []


def code_cells(notebook: dict) -> list[str]:
    """The source of every code cell of a parsed notebook, one string per cell."""
    return [
        "".join(cell.get("source", ""))
        for cell in notebook.get("cells", [])
        if cell.get("cell_type") == "code"
    ]


# -- fixtures --
@pytest.fixture(scope="session", autouse=True)
def mpds_dummy_store():
    """The session's MPDS reads of the repository ``cache/`` come from :data:`MPDS_DUMMY_DIR`."""
    with dummy_mpds_store():
        yield MPDS_DUMMY_DIR


@pytest.fixture(autouse=True)
def close_figures():
    """Close every matplotlib figure after each test of a module that imports this."""
    yield
    plt.close("all")


SMOKE_FIT_KWARGS = {"n_opts": 1, "max_iter": 64}


@pytest.fixture(scope="session")
def cu_mg_fit():
    """Cu-Mg (comb-exp) and the fit list of one ``fit_parameters(**SMOKE_FIT_KWARGS)``.

    Returns ``(bl, fits)``. Modules share the one object: read it, or deep-copy it first.
    """
    bl = BinaryLiquid.from_cache("Cu-Mg", param_format="comb-exp")
    if bl.init_error:
        pytest.skip("Cu-Mg: init error on load")
    return bl, bl.fit_parameters(**SMOKE_FIT_KWARGS)


@pytest.fixture(scope="session")
def cu_mg_full(cu_mg_fit):
    """The fitted Cu-Mg model of ``cu_mg_fit``; skips when the fit returned nothing."""
    bl, fits = cu_mg_fit
    if not fits or bl.init_error:
        pytest.skip("Cu-Mg: baseline could not be fit")
    return bl


@pytest.fixture(scope="module")
def ss_from_cache():
    """Hf-Zr from the cache with solid solutions, references from the unary database."""
    return BinaryLiquid.from_cache(
        "Hf-Zr", pd_ind=0, solid_solutions=True, ss_kwargs={"ref_mode": "from_unary_db"}
    )


@pytest.fixture
def no_persist():
    """Ternary imputed phases are not persisted for the test."""
    before = config.persist_ternary_imputed
    config.set_persist_ternary_imputed(False)
    yield
    config.set_persist_ternary_imputed(before)
