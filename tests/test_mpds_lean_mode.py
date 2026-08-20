"""Lean MPDS records, and the contract that stops one from silently answering wrong.

A LEAN record (``mpds.lean_record``) keeps an MPDS diagram's header plus its already-
stitched liquidus and throws ``shapes`` away — 93-97% of the file, and nothing a fit or a
liquidus plot reads. Two decisions carry the whole design, and both are pinned here.

**1. The stored curve is the PRE-fill one.** ``extract_digitized_liquidus`` densifies every
in-region composition gap wider than 0.06 before anyone downstream sees the curve, so its
output cannot tell a digitized point from synthetic fill. Store THAT and
``mpds.liquidus_coverage`` reports a dense, well-sampled curve for every system in the
corpus and the interior-sparsity gate in ``BinaryLiquid.from_cache`` never fires again —
silently, corpus-wide, with the metrics still looking plausible.
``TestPreFillCurveIsWhatIsStored`` asserts the equivalence AND, as its negative control,
builds the post-fill record the mistake would produce and shows the gate going blind:
measured on the real corpus, Pu-Ti's widest gap reads 0.824 pre-fill and 0.029 post-fill.

**2. Lean mode RAISES; it never auto-degrades.** This is the higher-value half.
``identify_mpds_phases`` reads ``mpds_json.get("shapes", [])`` and would return ``[]`` on a
lean record. An empty phase list makes ``assess_solid_coverage`` report ZERO reported
compounds — which reads as "nothing unsupported" — and **the coverage gate passes**. That
is a silent wrong answer of exactly the class the gate was built to prevent, so every such
consumer raises instead. ``TestRaiseNotDegrade`` therefore asserts ``raises``, never
"returns empty", and each raise is paired with a POSITIVE CONTROL showing the same call on
the full record returns something non-empty — otherwise the raise would be proving nothing
about a call that was going to come back empty anyway.

Corpus-scale evidence lives outside pytest, in the shipped CLI: ``python -m gliquid.cache
verify`` against a ``--mpds-mode lean`` store re-derives ``extract_digitized_liquidus`` and
``liquidus_coverage`` on both sides of every record and fails if a lean store's MPDS half
was compared zero times.
"""

import json
from contextlib import contextmanager
from pathlib import Path

import pytest

import gliquid.cache as cache
import gliquid.config as config
import gliquid.mpds as mpds
from gliquid.binary import BLPlotter, BinaryLiquid

# ---------------------------------------------------------------------------------------
# Synthetic diagrams. Frame is Hf-Zr throughout so the real cached DFT hull can be used.
# ---------------------------------------------------------------------------------------


def _svgpath(points):
    """[[x_pct, T_celsius], ...] -> MPDS-style svgpath string."""
    return "M " + " L ".join(f"{x},{t}" for x, t in points)


def _liquid(points):
    return {
        "label": "L",
        "nphases": 1,
        "is_solid": False,
        "kind": "phase",
        "svgpath": _svgpath(points),
    }


def _line_compound(comp_pct, t_lo=200, t_hi=1600, label="HfZr2"):
    """A digitized line compound — a boundary line, which identify_mpds_phases types 'lc'."""
    return {
        "label": label,
        "nphases": 1,
        "is_solid": True,
        "kind": "compound",
        "phase": f"{label}/139/tI6",
        "svgpath": _svgpath([[comp_pct, t_lo], [comp_pct, t_hi]]),
    }


def _make_full(liquid_regions, *, elements=("Hf", "Zr"), temp=(0, 2600), solids=True):
    shapes = [_liquid(r) for r in liquid_regions]
    if solids:
        shapes.append(_line_compound(33.3))
    record = {
        "reference": {"entry": "https://mpds.io/entry/C-synthetic"},
        "comp_range": [0, 100],
        "temp": list(temp),
        "labels": [["HfZr<sub>2</sub>", [33.3, 1600]]],
        "entry": "C-synthetic",
        "jcode": "0000",
        "year": "1991",
        "shapes": shapes,
    }
    if elements is not None:
        record["chemical_elements"] = list(elements)
    return record


def _make_frameless():
    """A cached json carrying NO ``chemical_elements`` block — ~13% of the corpus.

    Such a record also has no digitized liquid field, and that is not a convenience: the
    endpoint anchoring in ``_shape_liquidus`` indexes ``mpds_json['chemical_elements']``
    directly, so a frameless json with an 'L' shape cannot be extracted in FULL mode
    either. The pairing keeps this fixture a record the package can actually meet.
    """
    return _make_full([], elements=None)


# Ordinary, well-sampled V liquidus at 2 at.% spacing — one contiguous region.
DENSE = [[[x, 2233 - (2233 - 1400) * x / 50] for x in range(0, 50, 2)]
         + [[x, 1400 + (1855 - 1400) * (x - 50) / 50] for x in range(50, 101, 2)]]

# Bi-Si class: two disjoint wedges near the pure ends, an ~85 at.% UNDIGITIZED hole between
# them. The class the interior-sparsity gate exists for.
BI_SI_CLASS = [
    [[0, 2233], [2, 2100], [4, 1950], [5.5, 1800], [7, 1700]],
    [[92, 1500], [94, 1600], [96, 1700], [98, 1780], [100, 1855]],
]

# Pu-Ti class: ONE contiguous liquid region, but sampled with a huge interior hole. This is
# the case the fill step erases completely — post-fill its widest gap is 0.03.
SPARSE_ONE_REGION = [[[0, 2233], [3, 2100], [6, 1980], [78, 1500], [82, 1600], [100, 1855]]]

ALL_SHAPES = {
    "dense": DENSE,
    "bi_si_class_two_regions": BI_SI_CLASS,
    "sparse_one_region": SPARSE_ONE_REGION,
}


@pytest.fixture(scope="module")
def hf_zr_hull():
    """The real cached Hf-Zr DFT hull — offline, from the package's own cache/."""
    import gliquid.api as api

    ch, _ = api.get_dft_convexhull(["Hf", "Zr"], "GGA")
    return ch


# ---------------------------------------------------------------------------------------


class TestRecordMode:
    def test_a_full_json_is_full(self):
        assert mpds.record_mode(_make_full(DENSE)) == "full"

    def test_the_reduction_is_lean(self):
        assert mpds.record_mode(mpds.lean_record(_make_full(DENSE))) == "lean"

    def test_the_no_diagram_placeholder_is_empty(self):
        assert mpds.record_mode({"reference": None}) == "empty"

    def test_a_reduced_placeholder_is_still_empty_not_lean(self):
        """Parity: a system MPDS has no diagram for must behave identically in both stores.

        Classifying it 'lean' would make every consumer start RAISING on a record that
        carries no less information than it ever did — the reduction removed nothing.
        """
        assert mpds.record_mode(mpds.lean_record({"reference": None})) == "empty"

    def test_a_non_dict_is_empty_rather_than_an_attribute_error(self):
        assert mpds.record_mode(None) == "empty"
        assert mpds.record_mode([]) == "empty"


class TestTheReductionDropsShapesAndKeepsTheRest:
    def test_shapes_is_gone(self):
        assert "shapes" not in mpds.lean_record(_make_full(DENSE))

    @pytest.mark.parametrize(
        "field", ["reference", "chemical_elements", "temp", "comp_range", "labels", "entry"]
    )
    def test_every_field_a_consumer_reads_survives(self, field):
        full = _make_full(DENSE)
        assert mpds.lean_record(full)[field] == full[field]

    @pytest.mark.parametrize("name", ["Hf-Zr_MPDS_PD_0.json", "Al-Cu.json", "C-Nb.json"])
    def test_the_reduction_is_smaller_on_real_diagrams(self, package_corpus, name):
        """Measured on real cached diagrams, not on a synthetic one.

        ``shapes`` is 93-97% of an MPDS json, but the reduction does NOT save 93-97%: the
        liquidus points inside the 'L' shapes have to be kept, and they come back as
        full-precision JSON floats (``[0.524, 1507.65]``) where the source held a compact
        svgpath token (``52.4,1234.5``). Rounding them would shrink this further and is
        deliberately not done — exact equivalence with the full record is the whole claim.
        Real per-file ratios run 1.6x (Cu-Mg, a liquidus-dominated diagram) to 3.6x
        (Hf-Zr); over the 6,689-record matrix_data corpus the store's MPDS half goes
        26.0 MB -> 9.8 MB. 1.5x is the floor this pins, not the expectation.
        """
        source = json.loads((package_corpus / name).read_text())
        full_bytes = len(json.dumps(source))
        lean_bytes = len(json.dumps(mpds.lean_record(source)))
        assert lean_bytes * 1.5 < full_bytes, f"{name}: {full_bytes} -> {lean_bytes}"


class TestPreFillCurveIsWhatIsStored:
    """THE decision. Storing the post-fill curve retires the interior-sparsity gate."""

    @pytest.mark.parametrize("name", sorted(ALL_SHAPES))
    def test_liquidus_is_identical(self, name):
        full = _make_full(ALL_SHAPES[name])
        assert mpds.extract_digitized_liquidus(full) == mpds.extract_digitized_liquidus(
            mpds.lean_record(full)
        )

    @pytest.mark.parametrize("name", sorted(ALL_SHAPES))
    def test_coverage_dict_is_identical(self, name):
        full = _make_full(ALL_SHAPES[name])
        assert mpds.liquidus_coverage(full) == mpds.liquidus_coverage(mpds.lean_record(full))

    def test_the_bi_si_hole_survives_the_reduction(self):
        """Bi-Si is the named case: wedges at the ends, an undigitized interior."""
        cov = mpds.liquidus_coverage(mpds.lean_record(_make_full(BI_SI_CLASS)))
        assert cov["max_gap"] == pytest.approx(0.85)
        assert cov["covered_fraction"] == pytest.approx(0.15)
        assert len(cov["holes"]) == 1
        assert cov["holes"][0] == pytest.approx([0.07, 0.92])

    def test_negative_control_post_fill_storage_blinds_the_gate(self):
        """The mistake this design exists to prevent, built and measured.

        A lean record holding ``extract_digitized_liquidus``'s output instead of the
        pre-fill curve reports a densely-sampled liquidus for a system whose interior is
        one enormous hole — and ``from_cache`` then admits it. Without this control the
        equality tests above would pass just as happily on the broken reduction, because
        for a contiguous region the two curves agree everywhere the fill did not fire.
        """
        full = _make_full(SPARSE_ONE_REGION)
        honest = mpds.liquidus_coverage(mpds.lean_record(full))

        post_fill_curve, _ = mpds.extract_digitized_liquidus(full)
        broken = dict(mpds.lean_record(full))
        broken["_gliquid"] = dict(broken["_gliquid"], stitched=post_fill_curve)
        blind = mpds.liquidus_coverage(broken)

        assert honest["max_gap"] > config.liquidus_max_gap, "the gate should fire here"
        assert blind["max_gap"] <= 0.06, "post-fill, no gap wider than the fill step survives"
        assert blind["covered_fraction"] == pytest.approx(1.0)
        assert honest["covered_fraction"] < 0.5


class TestFrameIsStoredExplicitly:
    """``mpds_frame_matches`` treats an ABSENT frame as MATCHING — so it must be explicit."""

    def test_a_reversed_frame_still_does_not_match(self):
        lean = mpds.lean_record(_make_full(DENSE, elements=("Hf", "Zr")))
        assert mpds.mpds_frame_matches(lean, ["Zr", "Hf"]) is False

    def test_positive_control_its_own_frame_matches(self):
        lean = mpds.lean_record(_make_full(DENSE, elements=("Hf", "Zr")))
        assert mpds.mpds_frame_matches(lean, ["Hf", "Zr"]) is True

    def test_an_unknown_frame_is_written_as_an_explicit_null(self):
        """~13% of cached jsons carry no frame block. The key is still present, as null,
        so the reduction can never be mistaken for 'the frame was dropped in transit'."""
        lean = mpds.lean_record(_make_frameless())
        assert "chemical_elements" in lean
        assert lean["chemical_elements"] is None

    def test_an_unknown_frame_keeps_full_mode_behaviour(self):
        full = _make_frameless()
        lean = mpds.lean_record(full)
        for components in (["Hf", "Zr"], ["Zr", "Hf"]):
            assert mpds.mpds_frame_matches(lean, components) == mpds.mpds_frame_matches(
                full, components
            )


class TestRaiseNotDegrade:
    """Every guarded consumer RAISES on a lean record. Never 'returns empty'."""

    @pytest.fixture
    def full(self):
        return _make_full(DENSE)

    @pytest.fixture
    def lean(self, full):
        return mpds.lean_record(full)

    def test_liquid_shape_paths_raises(self, full, lean):
        assert mpds.liquid_shape_paths(full), "POSITIVE CONTROL: the full record has an 'L'"
        with pytest.raises(config.CacheModeError, match="shapes"):
            mpds.liquid_shape_paths(lean)

    def test_identify_mpds_phases_raises(self, full, lean):
        assert mpds.identify_mpds_phases(full), "POSITIVE CONTROL: the full record has phases"
        with pytest.raises(config.CacheModeError) as exc:
            mpds.identify_mpds_phases(lean)
        assert "identify_mpds_phases" in str(exc.value)

    def test_identify_invariant_points_raises(self, full, lean):
        from gliquid.phase import UNARY

        args = (["Hf", "Zr"], mpds.extract_digitized_liquidus(full)[0],
                UNARY.component_data(["Hf", "Zr"]), [300, 3000])
        # POSITIVE CONTROL: the same call on the full record completes.
        assert mpds.identify_invariant_points(full, *args) is not None
        with pytest.raises(config.CacheModeError) as exc:
            mpds.identify_invariant_points(lean, *args)
        assert "disable_inv_constrs" in str(exc.value), "the message must name the escape"

    def test_get_low_temp_phase_data_raises(self, full, lean, hf_zr_hull):
        mpds_side, _ = mpds.get_low_temp_phase_data(full, hf_zr_hull)
        assert mpds_side[0] or mpds_side[1], "POSITIVE CONTROL: the full record tables a phase"
        with pytest.raises(config.CacheModeError) as exc:
            mpds.get_low_temp_phase_data(lean, hf_zr_hull)
        assert "get_low_temp_phase_data" in str(exc.value)

    def test_an_empty_record_does_not_raise_in_either_mode(self):
        """Parity again: 'no diagram' is not 'evidence withheld'. These must stay quiet."""
        placeholder = {"reference": None}
        assert mpds.identify_mpds_phases(placeholder) == []
        assert mpds.identify_mpds_phases(mpds.lean_record(placeholder)) == []

    def test_the_error_is_a_config_error(self):
        assert issubclass(config.CacheModeError, config.ConfigError)
        assert cache.CacheModeError is config.CacheModeError


# ---------------------------------------------------------------------------------------
# Through BinaryLiquid — where the silent pass would actually have happened.
# ---------------------------------------------------------------------------------------


def _bl_on(monkeypatch, record):
    monkeypatch.setattr(
        mpds,
        "load_mpds_data",
        lambda input, pd_ind=None: (record, mpds.extract_digitized_liquidus(record)),
    )
    return BinaryLiquid.from_cache("Hf-Zr")


class TestBinaryLiquidOnALeanRecord:
    def test_from_cache_succeeds_and_measures_the_same_coverage(self, monkeypatch):
        """A lean record must still BUILD — fitting from params and plotting is the point."""
        full = _make_full(DENSE)
        bl_full = _bl_on(monkeypatch, full)
        bl_lean = _bl_on(monkeypatch, mpds.lean_record(full))
        assert bl_lean.digitized_liq == bl_full.digitized_liq
        assert bl_lean.liq_coverage == bl_full.liq_coverage
        assert bl_lean.init_error is bl_full.init_error

    def test_the_sparsity_gate_still_fires_on_a_lean_record(self, monkeypatch, caplog):
        """The gate the pre-fill decision protects, exercised end to end through a lean
        record. If the reduction had stored the post-fill curve this would come back
        ``init_error is False`` and nothing anywhere would say so."""
        bl = _bl_on(monkeypatch, mpds.lean_record(_make_full(BI_SI_CLASS)))
        assert bl.init_error is True
        assert bl.liq_coverage["max_gap"] == pytest.approx(0.85)
        assert "interior-sparse" in caplog.text

    def test_assess_solid_coverage_raises(self, monkeypatch):
        """**The anti-silent-pass test.** Not 'returns an empty report' — RAISES.

        With no ``shapes`` the phase list is empty, every reported-compound count is zero,
        ``unsupported_fraction`` is 0.0, and ``is_insufficient()`` says False: the gate
        passes a system it measured nothing about.
        """
        full = _make_full(DENSE)
        report = _bl_on(monkeypatch, full).assess_solid_coverage()
        assert report.phases, "POSITIVE CONTROL: the full record scores real phases"

        bl = _bl_on(monkeypatch, mpds.lean_record(full))
        with pytest.raises(config.CacheModeError) as exc:
            bl.assess_solid_coverage()
        assert "assess_solid_coverage" in str(exc.value)

    def test_find_invariant_points_raises(self, monkeypatch):
        bl = _bl_on(monkeypatch, mpds.lean_record(_make_full(DENSE)))
        with pytest.raises(config.CacheModeError):
            bl.find_invariant_points()

    def test_fit_parameters_raises_up_front(self, monkeypatch):
        bl = _bl_on(monkeypatch, mpds.lean_record(_make_full(DENSE)))
        with pytest.raises(config.CacheModeError) as exc:
            bl.fit_parameters()
        message = str(exc.value)
        assert "disable_inv_constrs=True" in message, "must name the first escape kwarg"
        assert "check_solid_coverage=False" in message, "must name the second escape kwarg"
        assert "--mpds-mode full" in message, "must name the REAL fix: a full store"
        assert bl.invariants is None, "nothing may be mutated on the way to the raise"

    def test_fit_parameters_does_not_raise_on_a_full_record(self, monkeypatch):
        """POSITIVE CONTROL: the raise is about the record, not about this system."""
        bl = _bl_on(monkeypatch, _make_full(DENSE))
        assert isinstance(bl.fit_parameters(n_opts=1, max_iter=2), list)

    def test_both_escapes_together_are_honoured(self, monkeypatch):
        """Asking explicitly for the degraded fit works. Only the SILENT one is refused."""
        bl = _bl_on(monkeypatch, mpds.lean_record(_make_full(DENSE)))
        result = bl.fit_parameters(
            disable_inv_constrs=True, check_solid_coverage=False, n_opts=1, max_iter=2
        )
        assert isinstance(result, list)

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"disable_inv_constrs": True},
            {"check_solid_coverage": False},
            {},
        ],
    )
    def test_one_escape_alone_is_not_enough(self, monkeypatch, kwargs):
        """Half-escaped is still silently unconstrained in the other half."""
        bl = _bl_on(monkeypatch, mpds.lean_record(_make_full(DENSE)))
        with pytest.raises(config.CacheModeError):
            bl.fit_parameters(**kwargs)


# ---------------------------------------------------------------------------------------
# A real lean single-file store, end to end.
# ---------------------------------------------------------------------------------------


@contextmanager
def _using_cache(path):
    """Point gliquid at ``path``, then put back the values that were ACTUALLY there.

    Restoring to a hardcoded path instead would silently un-swap the rest of the session
    for every test that runs after this one — a real defect found during spec 04.
    """
    saved = (config.cache_dir, config.cache_mode, config.dir_structure)
    try:
        config.set_cache_dir(path)
        yield
    finally:
        config.set_cache_dir(saved[0])
        config.set_cache_mode(saved[1])
        config.dir_structure = saved[2]
        cache.close_sqlite_backends()


@pytest.fixture(scope="module")
def package_corpus():
    root = config.cache_dir
    if root is None or not Path(root).is_dir():
        pytest.skip("no directory cache corpus configured")
    return Path(root)


@pytest.fixture(scope="module")
def lean_store(package_corpus, tmp_path_factory):
    """The package's own ``cache/`` corpus, migrated with ``--mpds-mode lean``."""
    dest = tmp_path_factory.mktemp("lean") / "lean.sqlite"
    assert (
        cache.main(
            ["migrate", "--from", str(package_corpus), "--to", str(dest), "--mpds-mode", "lean"]
        )
        == 0
    )
    yield dest
    cache.close_sqlite_backends()


class TestLeanStore:
    def test_records_round_trip_through_the_store(self, package_corpus, lean_store):
        backend = cache.SqliteBackend(lean_store)
        try:
            source = json.loads((package_corpus / "Hf-Zr_MPDS_PD_0.json").read_text())
            key = cache.CacheKey("Hf-Zr", cache.KIND_MPDS, "0")
            assert backend.read_json(key) == mpds.lean_record(source)
        finally:
            backend.close()

    def test_no_lean_row_carries_a_full_payload(self, lean_store):
        """The one state the contract must make impossible: a row labelled full with no
        shapes in it, or a lean row still carrying the blob it was meant to drop."""
        backend = cache.SqliteBackend(lean_store)
        try:
            rows = backend._conn().execute(
                "SELECT COUNT(*), COUNT(payload), COUNT(DISTINCT mode) FROM mpds_diagrams"
            ).fetchone()
            assert rows[0] > 0, "a store with no MPDS rows would make this vacuous"
            assert rows[1] == 0, "lean rows must not carry a payload blob"
            assert backend._conn().execute(
                "SELECT DISTINCT mode FROM mpds_diagrams"
            ).fetchall() == [("lean",)]
        finally:
            backend.close()

    def test_verify_checks_equivalence_rather_than_object_equality(
        self, package_corpus, lean_store, capsys
    ):
        assert (
            cache.main(
                ["verify", "--directory", str(package_corpus), "--sqlite", str(lean_store)]
            )
            == 0
        )
        out = capsys.readouterr().out
        assert "mpds mode            : lean" in out
        assert "liquidus divergences : 0" in out
        assert "coverage divergences : 0" in out
        # A comparison over zero records is not a pass, and verify says so itself.
        compared = int(
            next(line for line in out.splitlines() if line.startswith("lean mpds compared"))
            .split(":")[1]
            .strip()
            .replace(",", "")
        )
        assert compared > 0, "0 lean MPDS comparisons would make this verify vacuous"

    def test_info_reports_the_mode_and_the_lean_columns(self, lean_store, capsys):
        assert cache.main(["info", str(lean_store)]) == 0
        out = capsys.readouterr().out
        assert "mode lean" in out
        assert "stitched liquidus" in out
        assert "MPDS TOTAL" in out

    def test_an_unknown_mpds_mode_is_refused_by_name(self, package_corpus, tmp_path, capsys):
        dest = tmp_path / "nope.sqlite"
        args = ["--from", str(package_corpus), "--to", str(dest), "--mpds-mode", "svelte"]
        assert cache.main(["migrate", *args]) == 2
        assert "not a known mode" in capsys.readouterr().out
        assert not dest.exists()


def _trace_signature(fig):
    """Everything a reader would see: type, name, mode and the plotted points."""
    return [
        (
            trace.type,
            trace.name,
            getattr(trace, "mode", None),
            None if trace.x is None else tuple(trace.x),
            None if trace.y is None else tuple(trace.y),
        )
        for trace in fig.data
    ]


class TestPlotParity:
    """The call chain a lean store exists to serve: from_cache(params=...) -> plot_tx.

    ``plot_tx`` never touches ``mpds_json``; it reads the digitized liquidus and the hull.
    Both are identical under the reduction, so the figures must be too.
    """

    def _render(self):
        """(record mode actually loaded, trace signature) for the pinned system."""
        bl = BinaryLiquid.from_cache("Hf-Zr", pd_ind=0, params=[-15000, 0, 0, 0])
        bl.update_phase_points()
        return mpds.record_mode(bl.mpds_json), _trace_signature(BLPlotter(bl).get_plot("fit+liq"))

    def test_traces_are_identical_between_lean_and_full(self, package_corpus, lean_store):
        with _using_cache(package_corpus):
            full_mode, expected = self._render()
        with _using_cache(lean_store):
            lean_mode, actual = self._render()

        # Anti-vacuity: if set_cache_dir had not taken, BOTH sides would have read the
        # directory corpus and the comparison would be a full record against itself.
        assert full_mode == "full", "the control side must be a full record"
        assert lean_mode == "lean", "the lean side must actually have come from the store"
        assert expected, "an empty figure would make this vacuous"
        assert actual == expected
