"""``mpds.load_mpds_data``, ``mpds.from_mpds_download`` and ``mpds.has_cached_diagram``.

A diagram downloaded from an MPDS entry page (the JSON button) names its entry as an
``mpds://`` reference, keeps compound names only in its ``labels`` list, and may leave a
one-phase field unlabelled; :func:`_download` builds a synthetic one. Store tests write into a
flat store under ``tmp_path`` with an MPDS key set and a tripwire client, so a record that
failed to load from the store would fetch, and fail.
"""

import copy
import json

import pytest

import gliquid.api as api
import gliquid.config as config
import gliquid.mpds as mpds

from ._mpds_builders import MPDS_DUMMY_DIR, Tripwire

ENTRY_URL = "https://mpds.io/entry/{}"


def _dummy(system):
    return json.loads((MPDS_DUMMY_DIR / f"{system}.json").read_text())


def _field(svgpath, label=None, nphases=1):
    shape = {"kind": "phase", "nphases": nphases, "is_solid": True, "svgpath": svgpath}
    if label is not None:
        shape["label"] = label
    return shape


def _download():
    """A synthetic Cu-Mg diagram as an MPDS entry page saves it.

    MgCu2 (x = 33.3) is named by its label; the compound at x = 66.7 carries only the system
    name. Three one-phase fields are unlabelled: (Cu) with its label outside it, (Mg) with its
    label inside, and a 'Laves ht' field reached only through a leader line. The vertical side
    at x = 85 belongs to one two-phase field and is not a compound, though a formula label
    sits beside it.
    """
    liquid = {
        "kind": "phase",
        "nphases": 1,
        "is_solid": False,
        "label": "L",
        "svgpath": "M 0,1084 L 20,900 L 33.3,820 L 45,700 L 66.7,570 L 80,500 L 100,650 "
        "L 100,1100 L 0,1100",
    }
    shapes = [
        liquid,
        _field("M 0,400 L 6,400 L 6,700 L 0,1084"),
        _field("M 96,400 L 100,400 L 100,640 L 97,600"),
        _field("M 70,500 L 74,500 L 74,540 L 70,540"),
        _field("M 6,400 L 33.3,400 L 33.3,700 L 6,700", nphases=2),
        _field("M 33.3,400 L 66.7,400 L 66.7,550 L 33.3,550", nphases=2),
        _field("M 33.3,700 L 45,700 L 33.3,820", nphases=2),
        _field("M 66.7,400 L 96,400 L 96,480 L 66.7,480", nphases=2),
        _field("M 80,500 L 85,500 L 85,600 L 80,560", nphases=2),
        {"kind": "drawing", "svgpath": "M 40.5,995 L 72,520"},
    ]
    return {
        "temp": [400.0, 1100.0],
        "arity": 2,
        "entry": "C000101",
        "naxes": 2,
        "labels": [
            ["L", [50.0, 950.0], None],
            ["MgCu<sub>2</sub>", [31.9, 443.0], 1],
            ["Cu-Mg", [65.9, 442.0], None],
            ["(Cu)", [8.0, 430.0], 2],
            ["(Mg)", [98.9, 441.0], 3],
            ["Laves ht", [40.0, 1000.0], None],
            ["CuMg<sub>6</sub>", [86.0, 520.0], None],
        ],
        "shapes": shapes,
        "diatype": "binary",
        "title_a": "Cu",
        "title_b": "Mg",
        "warning": None,
        "comp_end": {"Cu": 0, "Mg": 1},
        "comp_range": [0.0, 100.0],
        "comp_start": {"Cu": 1, "Mg": 0},
        "chemical_elements": ["Cu", "Mg"],
        "$schema": "mpds://schema.json#/definitions/phase_diagram",
        "version": "1.2.5",
        "object_type": "C",
        "use_visavis_type": "pd",
        "reference": {"entry": "mpds://entry/C000101", "release": "2026"},
    }


@pytest.fixture
def store(tmp_path, monkeypatch):
    """A flat directory store under ``tmp_path``; the MPDS client counts and refuses calls."""
    client = Tripwire()
    monkeypatch.setattr(api, "get_mpds_client", client)
    monkeypatch.setattr(api, "get_api_key", lambda name: "0" * 32)
    saved = (config.cache_dir, config.cache_mode, config.dir_structure)
    config.set_cache_dir(tmp_path)
    config.set_dir_structure("flat")
    try:
        yield tmp_path, client
    finally:
        config.set_cache_dir(saved[0])
        config.set_cache_mode(saved[1])
        config.set_dir_structure(saved[2])


def _write(path, record):
    path.write_text(json.dumps(record))


def _lines(record):
    """``{label: (x, top T, bottom T)}`` of the record's compound lines."""
    out = {}
    for shape in record["shapes"]:
        if shape.get("kind") == "compound":
            (x1, t1), (x2, t2) = mpds._svg_points(shape["svgpath"])
            assert x1 == x2
            out[shape["label"]] = (x1, max(t1, t2), min(t1, t2))
    return out


class TestFromMpdsDownload:
    @pytest.fixture(scope="class")
    def converted(self):
        return mpds.from_mpds_download(_download())

    def test_it_is_recognized_as_a_download(self):
        assert mpds.is_mpds_download(_download())
        assert not mpds.is_mpds_download(_dummy("Hf-Zr"))
        assert not mpds.is_mpds_download({"reference": None})

    def test_the_reference_names_the_entry_page(self, converted):
        assert converted["reference"] == {"entry": ENTRY_URL.format("C000101")}

    def test_leader_lines_and_the_warning_are_dropped(self, converted):
        assert all(s.get("kind") != "drawing" for s in converted["shapes"])
        assert "warning" not in converted

    def test_each_unlabelled_field_takes_the_label_that_names_it(self, converted):
        fields = [s for s in converted["shapes"] if s.get("nphases") == 1 and s["is_solid"]]
        assert sorted(s["label"] for s in fields if s["kind"] == "phase") == [
            "(Cu)",
            "(Mg)",
            "Laves ht",
        ]

    def test_line_compounds_stand_on_the_shared_two_phase_edges(self, converted):
        assert _lines(converted) == {
            "MgCu2": (33.3, 820.0, 400.0),
            "Mg2Cu": (66.7, 550.0, 400.0),
        }

    def test_the_phases_read_from_it(self, converted):
        phases = mpds.identify_mpds_phases(converted)
        assert [(p["type"], p["name"]) for p in phases] == [
            ("ss", "(Cu)"),
            ("lc", "MgCu2"),
            ("lc", "Mg2Cu"),
            ("ss", "Laves ht"),
            ("ss", "(Mg)"),
        ]

    def test_the_input_is_not_modified(self):
        record = _download()
        before = copy.deepcopy(record)
        mpds.from_mpds_download(record)
        assert record == before

    def test_a_converted_record_is_returned_unchanged(self, converted):
        assert mpds.from_mpds_download(converted) is converted

    @pytest.mark.parametrize(
        "record",
        [{"reference": None}, None, ["C000101"]],
        ids=["placeholder", "none", "list"],
    )
    def test_a_record_that_is_not_a_download_is_returned_unchanged(self, record):
        assert mpds.from_mpds_download(record) is record

    def test_a_download_without_a_reference_gains_one(self):
        record = _dummy("Cu-Mg")
        del record["reference"]
        out = mpds.from_mpds_download(record)
        assert out["reference"] == {"entry": ENTRY_URL.format(record["entry"])}
        assert out["shapes"] == record["shapes"]

    def test_the_lean_reduction_reads_the_converted_record(self):
        lean = mpds.lean_record(_download())
        assert lean["reference"] == {"entry": ENTRY_URL.format("C000101")}
        assert mpds.record_mode(lean) == "lean"


class TestHasCachedDiagram:
    def test_a_download_under_any_name_is_written_as_the_system_record(self, store):
        root, client = store
        _write(root / "c.json", _download())
        before = (root / "c.json").read_text()

        assert mpds.has_cached_diagram(["Mg", "Cu"])

        assert json.loads((root / "Cu-Mg.json").read_text()) == mpds.from_mpds_download(_download())
        assert (root / "c.json").read_text() == before
        assert client.calls == 0

    def test_other_json_files_are_not_taken(self, store):
        root, client = store
        _write(root / "notes.json", {"shapes": []})
        other = _download()
        other["comp_end"] = {"Ag": 0, "V": 1}
        _write(root / "c.json", other)

        assert not mpds.has_cached_diagram("Cu-Mg")
        assert not (root / "Cu-Mg.json").exists()
        assert client.calls == 0


class TestLoadMpdsData:
    def test_a_download_is_found_converted_and_loaded(self, store):
        root, client = store
        _write(root / "c (1).json", _download())

        mpds_json, (liquidus, is_partial) = mpds.load_mpds_data("Cu-Mg")

        assert mpds_json == mpds.from_mpds_download(_download())
        assert (liquidus, is_partial) == mpds.extract_digitized_liquidus(mpds_json)
        assert liquidus
        assert (root / "Cu-Mg.json").exists()
        assert client.calls == 0

    def test_a_download_saved_under_the_system_name_is_converted_on_read(self, store):
        root, client = store
        _write(root / "Cu-Mg.json", _download())

        mpds_json, _ = mpds.load_mpds_data("Cu-Mg")

        assert _lines(mpds_json) == _lines(mpds.from_mpds_download(_download()))
        assert client.calls == 0

    def test_an_api_record_loads_as_stored(self, store):
        root, client = store
        record = _dummy("Hf-Zr")
        _write(root / "Hf-Zr.json", record)

        mpds_json, (liquidus, _) = mpds.load_mpds_data(["Zr", "Hf"])

        assert mpds_json == record
        assert liquidus
        assert client.calls == 0

    def test_a_cached_placeholder_is_a_hit_with_a_key_set(self, store):
        root, client = store
        _write(root / "Ga-Ru.json", _dummy("Ga-Ru"))

        assert mpds.load_mpds_data("Ga-Ru") == ({"reference": None}, (None, False))
        assert client.calls == 0
