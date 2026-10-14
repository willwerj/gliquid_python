"""The phase-comparison MPDS panel draws one segment per thermal form.

``get_low_temp_phase_data`` reports each tabled compound's forms alongside the two tables,
and ``render_phase_comparison`` stacks one fill per form, coloured by that form's own
decomposition type and all sharing the compound's own composition range, with a divider
line marking each internal transition. Fixtures are synthetic; element symbols are real so
``standard_formula`` can parse them.
"""

import warnings
from types import SimpleNamespace

import matplotlib
import pytest

matplotlib.use("Agg")

from matplotlib.colors import to_rgba  # noqa: E402

from gliquid import mpds  # noqa: E402
from gliquid.plotting import binary_figs  # noqa: E402
from gliquid.plotting.binary_figs import (  # noqa: E402
    THERMAL_FORM_COLORS,
    THERMAL_FORM_DIVIDER_COLOR,
)

from ._mpds_builders import band_bounds as _bounds
from ._mpds_builders import bare_field_shape as _dome
from ._mpds_builders import bare_line_shape as _line
from ._mpds_builders import close_figures  # noqa: F401
from ._mpds_builders import compound_hull as _hull
from ._mpds_builders import mpds_record as _json

ALPHA = 0.5
MIN_WIDTH = 0.026
K = 273.15


def _plateau_liquidus(t_top, x_lo, x_hi, slope, step=2):
    """A liquid field flat at ``t_top`` C over ``[x_lo, x_hi]`` at.%, sloping down outside."""
    points = [
        (x, t_top - slope * max(0, x_lo - x) - slope * max(0, x - x_hi))
        for x in range(0, 101, step)
    ]
    return {
        "label": "L",
        "nphases": 1,
        "svgpath": "M " + " L ".join(f"{x},{t}" for x, t in points),
    }


def _sloped_liquidus(t_at_zero, slope, step=2):
    """A liquid field rising linearly from ``t_at_zero`` C, so no compound top lies on it."""
    points = [(x, t_at_zero + slope * x) for x in range(0, 101, step)]
    return {
        "label": "L",
        "nphases": 1,
        "svgpath": "M " + " L ".join(f"{x},{t}" for x, t in points),
    }


def _mpds_side(data, hull):
    (congruent, incongruent, max_phase_temp, forms), _ = mpds.get_low_temp_phase_data(data, hull)
    return congruent, incongruent, max_phase_temp, forms


def _render(data, hull):
    return binary_figs.render_phase_comparison(SimpleNamespace(mpds_json=data, dft_ch=hull))


def _mpds_bands(fig):
    """The MPDS-panel bands, bottom-most first."""
    return sorted(fig.axes[0].collections, key=lambda band: _bounds(band)[2])


def _mpds_dividers(fig):
    """The MPDS-panel divider lines, lowest temperature first: ``(x_lo, x_hi, t)``."""
    lines = sorted(fig.axes[0].lines, key=lambda line: line.get_ydata()[0])
    return [(min(line.get_xdata()), max(line.get_xdata()), line.get_ydata()[0]) for line in lines]


def _colour_of(collection):
    return tuple(float(v) for v in collection.get_facecolor()[0])


def _expected_colour(decomp):
    return pytest.approx(to_rgba(THERMAL_FORM_COLORS[decomp], ALPHA))


class TestMoBLike:
    """One compound, an ``rt`` line and a hotter ``ht`` line that melts congruently."""

    ELEMENTS = ("B", "Mo")
    AXIS = (0.0, 2300.0)
    T_TRANSITION = 1500.0
    T_MELT = 2500.0

    def _data(self):
        return _json(
            [
                _plateau_liquidus(self.T_MELT - K, 45, 55, 5.0, step=5),
                _line("MoB rt", 50.0, 0.0, self.T_TRANSITION - K),
                _line("MoB ht", 50.0, self.T_TRANSITION - K, self.T_MELT - K),
            ],
            self.ELEMENTS,
            self.AXIS,
        )

    def _hull(self):
        return _hull(self.ELEMENTS, "MoB")

    def test_the_tables_hold_the_merged_compound(self):
        congruent, incongruent, max_phase_temp, _ = _mpds_side(self._data(), self._hull())
        assert congruent == {"MoB": ((0.5, 0.5), pytest.approx(self.T_MELT))}
        assert incongruent == {}
        assert max_phase_temp == pytest.approx(self.T_MELT)

    def test_two_records_a_transition_under_a_congruent_form(self):
        _, _, _, forms = _mpds_side(self._data(), self._hull())
        assert list(forms) == ["MoB"]
        assert forms["MoB"] == [
            {
                "form": "rt",
                "comp_range": (0.5, 0.5),
                "t_lo": 0.0,
                "t_hi": pytest.approx(self.T_TRANSITION),
                "decomp": "transition",
            },
            {
                "form": "ht",
                "comp_range": (0.5, 0.5),
                "t_lo": pytest.approx(self.T_TRANSITION),
                "t_hi": pytest.approx(self.T_MELT),
                "decomp": "congruent",
            },
        ]

    def test_the_panel_stacks_a_purple_segment_under_a_blue_one(self):
        fig = _render(self._data(), self._hull())
        assert len(fig.axes[0].collections) == 2
        lower, upper = _mpds_bands(fig)
        assert _bounds(lower) == pytest.approx((0.487, 0.513, 0.0, self.T_TRANSITION))
        assert _bounds(upper) == pytest.approx((0.487, 0.513, self.T_TRANSITION, self.T_MELT))
        assert _bounds(lower)[3] == _bounds(upper)[2]
        assert _colour_of(lower) == _expected_colour("transition")
        assert _colour_of(upper) == _expected_colour("congruent")

    def test_a_white_divider_marks_the_transition(self):
        fig = _render(self._data(), self._hull())
        (divider,) = fig.axes[0].lines
        assert (divider.get_xdata()[0], divider.get_xdata()[1]) == pytest.approx((0.487, 0.513))
        assert divider.get_ydata()[0] == divider.get_ydata()[1] == pytest.approx(self.T_TRANSITION)
        assert to_rgba(divider.get_color()) == pytest.approx(to_rgba(THERMAL_FORM_DIVIDER_COLOR))

    def test_the_y_range_still_follows_max_phase_temp(self):
        fig = _render(self._data(), self._hull())
        assert fig.axes[0].get_ylim() == pytest.approx((0.0, 1.1 * self.T_MELT))


class TestLiZnLike:
    """Two low-temperature fields at slightly different x and one ``ht`` field spelled as a
    fractional formula, all one compound within ``polymorph_pair_tol``."""

    ELEMENTS = ("Li", "Zn")
    AXIS = (0.0, 800.0)
    LT = (0.74, 0.78, 200.0 + K)
    RT = (0.76, 0.81, 300.0 + K)
    HT = (0.735, 0.80, 500.0 + K)
    #: Envelope shared by every form: the widest lb (HT's) and the widest ub (RT's).
    ENVELOPE = (0.735, 0.81)

    def _data(self):
        return _json(
            [
                _plateau_liquidus(500.0, 70, 84, 5.0),
                _dome("LiZn4 lt", [(74, 0), (78, 0), (76, 200)]),
                _dome("LiZn4 rt", [(76, 0), (81, 0), (78.5, 300)]),
                _dome("Li0.2Zn0.8 ht", [(73.5, 250), (80, 250), (77, 500)]),
            ],
            self.ELEMENTS,
            self.AXIS,
        )

    def _hull(self):
        return _hull(self.ELEMENTS, "LiZn4")

    def test_three_records_share_one_compound_wide_range(self):
        congruent, _, _, forms = _mpds_side(self._data(), self._hull())
        assert list(congruent) == ["LiZn4"]
        (lt, rt, ht) = forms["LiZn4"]
        assert [r["form"] for r in (lt, rt, ht)] == ["lt", "rt", "ht"]
        assert [r["decomp"] for r in (lt, rt, ht)] == ["transition", "transition", "congruent"]
        assert lt["comp_range"] == pytest.approx(self.ENVELOPE)
        assert rt["comp_range"] == pytest.approx(self.ENVELOPE)
        assert ht["comp_range"] == pytest.approx(self.ENVELOPE)
        assert (lt["t_lo"], lt["t_hi"]) == (0.0, pytest.approx(self.LT[2]))
        assert (rt["t_lo"], rt["t_hi"]) == (pytest.approx(self.LT[2]), pytest.approx(self.RT[2]))
        assert (ht["t_lo"], ht["t_hi"]) == (pytest.approx(self.RT[2]), pytest.approx(self.HT[2]))

    def test_the_ht_segment_starts_where_the_higher_low_form_ends(self):
        _, _, _, forms = _mpds_side(self._data(), self._hull())
        _, rt, ht = forms["LiZn4"]
        assert ht["t_lo"] == rt["t_hi"]

    def test_the_panel_draws_purple_purple_blue_at_one_shared_width(self):
        fig = _render(self._data(), self._hull())
        assert len(fig.axes[0].collections) == 3
        lt, rt, ht = _mpds_bands(fig)
        assert _bounds(lt) == pytest.approx((*self.ENVELOPE, 0.0, self.LT[2]))
        assert _bounds(rt) == pytest.approx((*self.ENVELOPE, self.LT[2], self.RT[2]))
        assert _bounds(ht) == pytest.approx((*self.ENVELOPE, self.RT[2], self.HT[2]))
        assert _colour_of(lt) == _expected_colour("transition")
        assert _colour_of(rt) == _expected_colour("transition")
        assert _colour_of(ht) == _expected_colour("congruent")

    def test_two_dividers_mark_the_two_transitions(self):
        fig = _render(self._data(), self._hull())
        dividers = _mpds_dividers(fig)
        assert len(dividers) == 2
        assert dividers[0] == pytest.approx((*self.ENVELOPE, self.LT[2]))
        assert dividers[1] == pytest.approx((*self.ENVELOPE, self.RT[2]))
        for line in fig.axes[0].lines:
            assert to_rgba(line.get_color()) == pytest.approx(to_rgba(THERMAL_FORM_DIVIDER_COLOR))


class TestSingleFormCompound:
    """One digitized shape: one record from 0 to ``T_max``, one band of the table's colour."""

    ELEMENTS = ("Al", "Cu")
    AXIS = (0.0, 1100.0)
    X = 0.3
    TOP = 600.0 + K

    def _data(self, congruent):
        liquidus = (
            _plateau_liquidus(600.0, 25, 35, 6.0, step=5)
            if congruent
            else _sloped_liquidus(700.0, 3.0, step=5)
        )
        return _json([liquidus, _line("Al2Cu", 100 * self.X, 0.0, 600.0)], self.ELEMENTS, self.AXIS)

    def _hull(self):
        return _hull(self.ELEMENTS, "Al2Cu")

    @pytest.mark.parametrize("congruent", [True, False])
    def test_one_record_spanning_zero_to_t_max(self, congruent):
        cong, incong, max_phase_temp, forms = _mpds_side(self._data(congruent), self._hull())
        table, decomp = (cong, "congruent") if congruent else (incong, "incongruent")
        assert list(table) == ["Al2Cu"]
        assert max_phase_temp == pytest.approx(self.TOP)
        assert forms == {
            "Al2Cu": [
                {
                    "form": "",
                    "comp_range": (self.X, self.X),
                    "t_lo": 0.0,
                    "t_hi": pytest.approx(self.TOP),
                    "decomp": decomp,
                }
            ]
        }

    @pytest.mark.parametrize("congruent", [True, False])
    def test_the_figure_is_one_band_in_the_tables_colour(self, congruent):
        """A single-form compound draws one band, hanging from the baseline to ``T_max``,
        blue when congruent and purple when incongruent, widened to the labelling minimum."""
        data, hull = self._data(congruent), self._hull()
        cong, incong, max_phase_temp, _ = _mpds_side(data, hull)
        fig = _render(data, hull)
        (band,) = fig.axes[0].collections
        assert len(cong) + len(incong) == 1
        assert _bounds(band) == pytest.approx(
            (self.X - MIN_WIDTH / 2, self.X + MIN_WIDTH / 2, 0.0, max_phase_temp)
        )
        assert _colour_of(band) == _expected_colour("congruent" if congruent else "incongruent")
        assert len(fig.axes[0].lines) == 0, "a single-form compound draws no divider"


class TestOverlappingDigitization:
    """Both forms drawn full height, the hotter one topping out lower: the hotter form spans
    nothing, so it gets no record and the drawn form carries the melting verdict."""

    ELEMENTS = ("C", "Ce")
    AXIS = (0.0, 2600.0)
    RT_TOP = 2249.65
    HT_TOP = 2239.95

    def _data(self, extra=()):
        return _json(
            [
                _plateau_liquidus(self.RT_TOP, 28, 38, 10.0),
                _line("CeC2 rt", 33.3, 100.0, self.RT_TOP),
                _line("CeC2 ht", 33.3, 90.0, self.HT_TOP),
                *extra,
            ],
            self.ELEMENTS,
            self.AXIS,
        )

    def _hull(self):
        return _hull(self.ELEMENTS, "CeC2")

    def test_the_hotter_form_gets_no_record(self):
        congruent, _, max_phase_temp, forms = _mpds_side(self._data(), self._hull())
        assert list(congruent) == ["CeC2"]
        (rt,) = forms["CeC2"]
        assert rt["form"] == "rt"
        assert rt["decomp"] == "congruent"
        assert rt["t_lo"] == 0.0
        assert rt["t_hi"] == pytest.approx(max_phase_temp)
        assert rt["t_hi"] == pytest.approx(self.RT_TOP + K)

    def test_only_the_coolest_form_survives_when_every_hotter_one_spans_nothing(self):
        """POSITIVE CONTROL for the hotter forms above: an even cooler form drawn highest
        takes the verdict and both forms above it drop out."""
        lt = _line("CeC2 lt", 33.3, 80.0, self.RT_TOP + 5.0)
        _, _, max_phase_temp, forms = _mpds_side(self._data([lt]), self._hull())
        (only,) = forms["CeC2"]
        assert only["form"] == "lt"
        assert only["decomp"] == "congruent"
        assert (only["t_lo"], only["t_hi"]) == (0.0, pytest.approx(max_phase_temp))

    def test_the_compound_draws_one_band_in_the_melting_colour(self):
        fig = _render(self._data(), self._hull())
        (band,) = fig.axes[0].collections
        assert _bounds(band)[2] == 0.0
        assert _bounds(band)[3] == pytest.approx(self.RT_TOP + K)
        assert _colour_of(band) == _expected_colour("congruent")
        assert len(fig.axes[0].lines) == 0, "the dropped hotter form leaves no transition to mark"


class TestUnpairedHtForm:
    """A high-temperature form with no cooler sibling is in no table and draws no segment."""

    ELEMENTS = ("Al", "Cu")
    AXIS = (0.0, 1100.0)

    def _data(self):
        return _json(
            [
                _sloped_liquidus(700.0, 3.0, step=5),
                _line("Al2Cu", 30.0, 0.0, 600.0),
                _line("AlCu3 ht", 75.0, 0.0, 1000.0),
            ],
            self.ELEMENTS,
            self.AXIS,
        )

    def _hull(self):
        return _hull(self.ELEMENTS, "Al2Cu")

    def test_it_is_absent_from_every_table(self):
        cong, incong, _, forms = _mpds_side(self._data(), self._hull())
        names = set(cong) | set(incong)
        assert "Al2Cu" in names, "POSITIVE CONTROL: the paired compound is tabled"
        assert "AlCu3" not in names
        assert set(forms) == names

    def test_it_draws_no_segment(self):
        fig = _render(self._data(), self._hull())
        (band,) = fig.axes[0].collections
        assert _bounds(band)[:2] == pytest.approx((0.3 - MIN_WIDTH / 2, 0.3 + MIN_WIDTH / 2))


def _draw_order(ax):
    """``ax``'s bands and lines in the order matplotlib paints them, bottom-most first."""
    artists = [a for a in ax.get_children() if a in ax.collections or a in ax.lines]
    return sorted(artists, key=lambda artist: artist.get_zorder())


class TestNarrowSegmentsDrawOnTop:
    """A wide two-form field overlapping a narrow line compound: the narrow band is painted
    after every fill and divider of the wide one, so it stays visible where they overlap."""

    ELEMENTS = ("Al", "Cu")
    AXIS = (0.0, 1100.0)
    WIDE = (0.30, 0.60)
    NARROW_X = 0.40

    def _data(self):
        return _json(
            [
                _sloped_liquidus(700.0, 3.0, step=5),
                _line("Al3Cu2", 100 * self.NARROW_X, 0.0, 450.0),
                _dome("AlCu rt", [(30, 0), (60, 0), (45, 300)]),
                _dome("AlCu ht", [(30, 300), (60, 300), (45, 500)]),
            ],
            self.ELEMENTS,
            self.AXIS,
        )

    def _hull(self):
        return _hull(self.ELEMENTS, "AlCu")

    def test_the_fixture_overlaps(self):
        """POSITIVE CONTROL: both compounds are tabled and the narrow one lies inside the
        wide one's composition range, so paint order decides what is visible."""
        _, _, _, forms = _mpds_side(self._data(), self._hull())
        assert set(forms) == {"Al3Cu2", "AlCu"}
        assert forms["AlCu"][0]["comp_range"] == pytest.approx(self.WIDE)
        assert len(forms["AlCu"]) == 2

    def test_the_narrow_band_is_painted_last(self):
        fig = _render(self._data(), self._hull())
        order = _draw_order(fig.axes[0])
        narrow = [
            i
            for i, artist in enumerate(order)
            if artist in fig.axes[0].collections
            and _bounds(artist)[1] - _bounds(artist)[0] < self.WIDE[1] - self.WIDE[0]
        ]
        assert len(narrow) == 1
        assert narrow[0] == len(order) - 1, "a wide fill or divider paints over the narrow band"

    def test_a_divider_still_paints_over_its_own_fills(self):
        fig = _render(self._data(), self._hull())
        order = _draw_order(fig.axes[0])
        (divider,) = fig.axes[0].lines
        wide = [
            i
            for i, artist in enumerate(order)
            if artist in fig.axes[0].collections and _bounds(artist)[:2] == pytest.approx(self.WIDE)
        ]
        assert len(wide) == 2
        assert order.index(divider) > max(wide)


class TestTheColourTable:
    def test_only_a_congruent_form_is_blue(self):
        assert THERMAL_FORM_COLORS == {
            "congruent": "blue",
            "incongruent": "purple",
            "transition": "purple",
        }


class TestCompositionAxis:
    def test_the_percent_labels_sit_on_fixed_ticks_without_a_warning(self):
        data, hull = TestMoBLike()._data(), TestMoBLike()._hull()
        with warnings.catch_warnings():
            warnings.simplefilter("error", UserWarning)
            fig = _render(data, hull)
        ax2 = fig.axes[1]
        assert list(ax2.get_xticks()) == pytest.approx([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
        assert [label.get_text() for label in ax2.get_xticklabels()] == [
            "0",
            "20",
            "40",
            "60",
            "80",
            "100",
        ]
