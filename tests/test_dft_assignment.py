"""Unit tests for solid_coverage.assign_dft_phases: which DFT hull phase stands behind which MPDS phase."""

import pytest

from gliquid import solid_coverage

# (system, MPDS phase, drawn lo, drawn hi, comp, MPDS space group, hull x, hull space group,
#  same phase). Compositions are fractions of the system's second component.
RULINGS = [
    ("Ag-Sn", "Ag0.8Sn0.2", 0.1208, 0.2266, 0.1285, 194, 0.25, 59, False),
    ("Al-Ir", "IrAl2.75", 0.2655, 0.2806, 0.2728, 195, 0.25, 194, False),
    ("Al-Re", "Re14Al54.75", 0.2042, 0.212, 0.212, 2, 0.2667, 2, False),
    ("Bi-Tl", "Tl0.8Bi0.2", 0.5997, 0.9521, 0.887, 225, 0.75, 194, True),
    ("Ca-Ga", "CaGa2.25", 0.6836, 0.6969, 0.6857, 191, 0.6667, 194, False),
    ("Ca-Zn", "Ca3.33Zn10.11", 0.7394, 0.7514, 0.7459, 194, 0.6667, 74, False),
    ("Co-Nb", "Nb6.5Co6.5", 0.5044, 0.5265, 0.5147, 166, 0.5385, 166, True),
    ("Co-Ta", "Ta0.8Co2.2", 0.2754, 0.2905, 0.2852, 194, 0.25, 166, False),
    ("Co-Ti", "Ti0.9Co2.1", 0.281, 0.3134, 0.3134, 194, 0.25, 221, False),
    ("Cr-Ir", "Cr0.5Ir0.5", 0.3208, 0.6996, 0.6996, 194, 0.75, 221, False),
    ("Cu-Ga", "Cu0.78Ga0.22 rt", 0.2156, 0.2306, 0.2259, None, 0.25, 221, False),
    ("Cu-Si", "Cu4.15Si0.85 rt", 0.1708, 0.1805, 0.1718, 213, 0.2105, 220, False),
    ("Cu-Y", "Cu5.4Y0.8", 0.1258, 0.1439, 0.1439, 191, 0.1667, 191, True),
    ("Cu-Y", "Cu5Y1.25", 0.1828, 0.2, 0.2, 10, 0.1667, 191, False),
    ("Cu-Zn", "Cu0.2Zn0.8", 0.78, 0.8795, 0.78, 194, 0.75, 194, True),
    ("Fe-W", "W5.1Fe7.9", 0.3949, 0.4665, 0.4091, 166, 0.3333, 194, False),
    ("Ga-Mn", "Mn1.25Ga3.75", 0.2236, 0.24, 0.2337, 229, 0.2, 229, True),
    ("Gd-Ge", "GdGe1.6 rt", 0.5896, 0.61, 0.5993, 191, 0.5, 63, False),
    ("Gd-Si", "GdSi1.5 ht", 0.5997, 0.6201, 0.6098, None, 0.6667, 141, False),
    ("Ge-Mn", "Mn0.77Ge0.23 rt", 0.7642, 0.7857, 0.7659, 139, 0.75, 225, False),
    ("Ge-Tm", "TmGe1.9", 0.3505, 0.3645, 0.3574, 51, 0.3333, 63, False),
    ("Hf-Rh", "Hf0.9Rh1.1 rt", 0.5227, 0.544, 0.544, 123, 0.5, 221, False),
    ("In-Sc", "Sc1.1In0.9", 0.5463, 0.5555, 0.5509, 221, 0.5, 123, False),
    ("Ir-Nb", "Nb0.63Ir0.37", 0.5846, 0.6697, 0.6695, 136, 0.75, 223, False),
    ("Ir-Nb", "Nb0.85Ir1.15", 0.39, 0.4552, 0.39, 51, 0.5, 123, False),
    ("Ir-Nb", "Nb0.95Ir1.05", 0.4701, 0.4798, 0.4701, 123, 0.5, 123, True),
    ("Ir-Ta", "Ta0.82Ir1.18 rt", 0.411, 0.4224, 0.4117, 123, 0.5, 51, False),
    ("Ir-Ta", "Ta0.86Ir1.14", 0.3906, 0.4958, 0.3906, 51, 0.5, 51, True),
    ("Ir-V", "V0.96Ir1.04", 0.4051, 0.48, 0.4051, 123, 0.5, 65, False),
    ("La-Si", "LaSi1.67", 0.6245, 0.6403, 0.6344, 74, 0.6667, 141, False),
    ("Li-Zn", "Li2.8Zn4.2 rt", 0.5797, 0.6094, 0.5997, 166, 0.5, 227, False),
    ("Mn-Si", "Mn0.85Si0.15 rt", 0.1214, 0.1581, 0.154, 148, 0.25, 225, False),
    ("Mn-Si", "Mn25.2Si5.8", 0.163, 0.1894, 0.174, 71, 0.25, 225, False),
    ("Mn-Si", "(Mn) ht1", 0.0004, 0.1673, 0.0515, 213, 0.25, 225, False),
    ("Na-Pb", "Na1.2Pb2.8", 0.6554, 0.7265, 0.7002, 221, 0.75, 221, True),
    ("Os-Ta", "Ta0.5Os0.5", 0.3612, 0.5198, 0.43, 217, 0.6, 35, False),
    ("Rh-Sn", "Rh1.5Sn", 0.3772, 0.4072, 0.4003, 194, 0.3333, 62, False),
    ("Rh-V", "V0.88Rh1.12", 0.405, 0.48, 0.4123, 123, 0.5, 65, False),
    ("Si-Tb", "TbSi1.67", 0.3671, 0.3795, 0.3795, 191, 0.4, 63, False),
]


def field(name, lo, hi, comp, spacegroup=None):
    return {
        "type": "ss",
        "name": name,
        "comp": comp,
        "cbounds": [[lo, 1000.0], [hi, 1000.0]],
        "structure": {"spacegroup": spacegroup},
    }


def line(name, comp, spacegroup=None):
    return {"type": "lc", "name": name, "comp": comp, "structure": {"spacegroup": spacegroup}}


@pytest.mark.parametrize(
    "system,name,lo,hi,comp,sg_mpds,x,sg_dft,same",
    RULINGS,
    ids=[f"{r[0]}:{r[1]}" for r in RULINGS],
)
def test_the_match_agrees_with_the_ruling(system, name, lo, hi, comp, sg_mpds, x, sg_dft, same):
    got = solid_coverage.assign_dft_phases(
        [field(name, lo, hi, comp, sg_mpds)],
        system.split("-"),
        [x],
        {x: sg_dft},
        rescuable=frozenset({0}),
    )
    assert bool(got) is same


def test_the_rulings_are_not_separable_by_distance_alone():
    def dist(r):
        return max(r[2] - r[6], r[6] - r[3], 0.0)

    nearest_distinct = min(dist(r) for r in RULINGS if not r[8] and dist(r) > 0)
    farthest_same = max(dist(r) for r in RULINGS if r[8])
    assert nearest_distinct < farthest_same


class TestOneToOne:
    def test_one_hull_phase_serves_one_compound(self):
        phases = [field("A0.3B0.7", 0.70, 0.745, 0.70), field("A0.2B0.8", 0.755, 0.90, 0.755)]
        got = solid_coverage.assign_dft_phases(phases, ["A", "B"], [0.75])
        assert len(got) == 1

    def test_the_assignment_covers_more_than_nearest_first_would(self):
        """Nearest-first gives 0.50 to the first field and leaves the second uncovered."""
        phases = [field("A0.5B0.5", 0.45, 0.555, 0.50), field("A0.45B0.55", 0.545, 0.60, 0.55)]
        got = solid_coverage.assign_dft_phases(phases, ["A", "B"], [0.55, 0.46])
        assert {i: m.hull for i, m in got.items()} == {0: 0.46, 1: 0.55}

    def test_among_equal_coverage_the_residual_is_minimised(self):
        phases = [line("X", 0.50), line("Y", 0.60)]
        got = solid_coverage.assign_dft_phases(phases, [], [0.505, 0.598])
        assert {i: m.hull for i, m in got.items()} == {0: 0.505, 1: 0.598}

    def test_polymorphs_share_their_compound_s_hull_phase(self):
        phases = [line("Mg3Bi2 rt", 0.60), line("Mg3Bi2 ht", 0.60)]
        got = solid_coverage.assign_dft_phases(phases, ["Bi", "Mg"], [0.60])
        assert {i: m.hull for i, m in got.items()} == {0: 0.60, 1: 0.60}


class TestSpellings:
    """Two spellings of one formula draw on ONE hull entry."""

    @staticmethod
    def _assign(names, elements=("Bi", "Mg"), hull=(0.6,)):
        phases = [line(name, 0.6) for name in names]
        assigned = solid_coverage.assign_dft_phases(phases, list(elements), list(hull))
        return {i: m.hull for i, m in assigned.items()}

    def test_integer_spellings_share_the_entry(self):
        assert self._assign(["Mg3Bi2", "Bi2Mg3"]) == {0: 0.6, 1: 0.6}

    def test_a_fractional_spelling_shares_the_entry(self):
        assert self._assign(["Mg3Bi2", "Mg1.5Bi"]) == {0: 0.6, 1: 0.6}

    def test_without_elements_the_spellings_compete(self):
        assert len(self._assign(["Mg3Bi2", "Bi2Mg3"], elements=())) == 1

    def test_two_compounds_keep_their_own_entries(self):
        phases = [line("DyAl2", 1 / 3), line("AlDy2", 2 / 3)]
        got = solid_coverage.assign_dft_phases(phases, ["Al", "Dy"], [1 / 3, 2 / 3])
        assert {i: m.hull for i, m in got.items()} == {
            0: pytest.approx(1 / 3),
            1: pytest.approx(2 / 3),
        }


class TestJointCoverage:
    def test_further_hull_phases_on_the_range_join_the_match(self):
        got = solid_coverage.assign_dft_phases(
            [field("A0.5B0.5", 0.40, 0.60, 0.50)], ["A", "B"], [0.45, 0.55]
        )
        assert got[0].hull == 0.55 or got[0].hull == 0.45
        assert got[0].comps == tuple([got[0].hull, *got[0].extra])
        assert sorted(got[0].comps) == [0.45, 0.55]

    def test_a_hull_phase_off_the_range_does_not_join(self):
        phases = [field("A0.5B0.5", 0.40, 0.60, 0.50, spacegroup=71)]
        got = solid_coverage.assign_dft_phases(phases, ["A", "B"], [0.50, 0.63], {0.63: 71})
        assert got[0].hull == 0.50 and got[0].extra == ()

    def test_a_joining_phase_goes_to_one_compound(self):
        phases = [field("A0.5B0.5", 0.40, 0.60, 0.45), field("A0.4B0.6", 0.55, 0.70, 0.65)]
        got = solid_coverage.assign_dft_phases(phases, ["A", "B"], [0.45, 0.65, 0.58])
        joined = [x for m in got.values() for x in m.extra]
        assert joined == [0.58]


class TestTheRecord:
    def test_an_exact_match_records_a_structure_mismatch(self):
        got = solid_coverage.assign_dft_phases(
            [line("CrIr3", 0.75, spacegroup=194)], ["Cr", "Ir"], [0.75], {0.75: 221}
        )
        assert (got[0].via, got[0].structure, got[0].residual) == ("exact", "mismatch", 0.0)

    def test_an_unknown_structure_reads_unknown(self):
        got = solid_coverage.assign_dft_phases(
            [line("CrIr3", 0.75, spacegroup=194)], ["Cr", "Ir"], [0.75]
        )
        assert got[0].structure == "unknown"

    def test_a_structure_match_records_its_residual(self):
        phases = [field("A0.3B0.7", 0.60, 0.70, 0.60, spacegroup=191)]
        got = solid_coverage.assign_dft_phases(phases, ["A", "B"], [0.73], {0.73: 191})
        assert got[0].via == "structure"
        assert got[0].residual == pytest.approx(0.03)

    def test_space_groups_one_and_two_never_match(self):
        for sg in (1, 2):
            phases = [field("A0.3B0.7", 0.60, 0.70, 0.60, spacegroup=sg)]
            assert solid_coverage.assign_dft_phases(phases, ["A", "B"], [0.73], {0.73: sg}) == {}

    def test_a_component_phase_never_claims(self):
        phases = [{"type": "comp", "name": "(A)", "comp": 0.02}]
        assert solid_coverage.assign_dft_phases(phases, [], [0.02]) == {}

    def test_an_unpaired_high_temperature_form_never_claims(self):
        phases = [line("AB ht", 0.50)]
        assert solid_coverage.assign_dft_phases(phases, [], [0.50])
        assert (
            solid_coverage.assign_dft_phases(phases, [], [0.50], unpaired_ht={("AB", 0.50)}) == {}
        )
