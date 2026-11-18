"""The digitized MPDS diagrams shipped in ``cache/`` carry only allowlisted fields.

A shipped record's top-level keys are among ``ALLOWED_TOP`` and its ``reference`` holds only
``entry``.
"""

import json
from pathlib import Path

import pytest

CACHE = Path(__file__).resolve().parents[1] / "cache"
ALLOWED_TOP = frozenset(
    {
        "$schema",
        "arity",
        "chemical_elements",
        "comp_end",
        "comp_range",
        "comp_start",
        "diatype",
        "entry",
        "labels",
        "naxes",
        "object_repr",
        "object_type",
        "reference",
        "shapes",
        "temp",
        "title_a",
        "title_b",
        "unlocked",
        "use_visavis_type",
        "version",
    }
)
ALLOWED_REFERENCE = frozenset({"entry"})


def unexpected_fields(record):
    """The field names in an MPDS record outside the allowlists, sorted."""
    found = sorted(set(record) - ALLOWED_TOP)
    reference = record.get("reference")
    if isinstance(reference, dict):
        found += sorted(f"reference.{key}" for key in set(reference) - ALLOWED_REFERENCE)
    return found


def _shipped_mpds_records():
    for path in sorted(CACHE.glob("*.json")):
        if "_ENTRIES_" in path.name:
            continue
        record = json.loads(path.read_text())
        if isinstance(record, dict) and "shapes" in record:
            yield path.name, record


SHIPPED = list(_shipped_mpds_records())


def test_the_check_names_every_field_outside_the_allowlists():
    record = {
        "shapes": [],
        "temp": [0, 1000],
        "extra": 1,
        "reference": {"entry": "https://mpds.io/entry/C000000", "note": "n"},
    }
    assert unexpected_fields(record) == ["extra", "reference.note"]


def test_the_cache_ships_mpds_records():
    assert len(SHIPPED) >= 5


@pytest.mark.parametrize("name,record", SHIPPED, ids=[name for name, _ in SHIPPED])
def test_a_shipped_record_carries_only_allowlisted_fields(name, record):
    assert unexpected_fields(record) == []
