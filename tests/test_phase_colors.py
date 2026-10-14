"""build_phase_color_map: the PHASE_PALETTE contract + SS collision guard.

The palette scheme (the max-contrast PHASE_PALETTE) lives in
gliquid.plotting.style.build_phase_color_map. Two contracts:
1. ASSIGNMENT — line phases cycle PHASE_PALETTE in ``phase_names`` order and 'L' is
   cornflowerblue; the live-vs-frozen dict check runs in
   test_decomposition_pins.py (the ``color_map`` figure summaries).
2. COLLISION-FREE — no line-phase or liquid color may equal any of the 3 reserved
   SS_FIXED_COLORS, compared numerically (hex vs 'rgb(r, g, b)' vs named colors).
"""

import re

from gliquid.plotting.style import PHASE_PALETTE, SS_FIXED_COLORS, build_phase_color_map

_NAMED = {"cornflowerblue": (100, 149, 237)}


def _rgb(color: str) -> tuple[int, int, int]:
    color = color.strip().lower()
    if color.startswith("#"):
        return tuple(int(color[i : i + 2], 16) for i in (1, 3, 5))
    m = re.fullmatch(r"rgb\((\d+),\s*(\d+),\s*(\d+)\)", color)
    if m:
        return tuple(int(g) for g in m.groups())
    return _NAMED[color]


class TestPaletteAssignment:
    def test_static_non_ss_map(self):
        got = build_phase_color_map(["A", "B", "L", "C"])
        assert got == {
            "L": "cornflowerblue",
            "A": PHASE_PALETTE[0],
            "B": PHASE_PALETTE[1],
            "C": PHASE_PALETTE[2],
        }

    def test_palette_cycles(self):
        phases = [f"P{i}" for i in range(len(PHASE_PALETTE) + 2)]
        got = build_phase_color_map(phases + ["L"])
        assert got[f"P{len(PHASE_PALETTE)}"] == PHASE_PALETTE[0]
        assert got[f"P{len(PHASE_PALETTE) + 1}"] == PHASE_PALETTE[1]

    def test_inserting_imputed_phase_preserves_other_colors(self):
        """Inserting an imputed phase at an interior composition must not shift the
        palette index -- and therefore the color -- of any pre-existing phase."""
        phase_names = ["A", "B", "C", "D", "L"]
        polymorph_names = {"A", "D"}
        before = build_phase_color_map(phase_names, polymorph_names=polymorph_names)

        with_imputed = ["A", "B", "Imp", "C", "D", "L"]
        after = build_phase_color_map(
            with_imputed, polymorph_names=polymorph_names, imputed_names={"Imp"}
        )

        for phase, color in before.items():
            assert after[phase] == color
        assert after["Imp"] not in set(before.values())


class TestSsCollisionGuard:
    def test_palette_numerically_disjoint_from_reserved(self):
        reserved = {_rgb(c) for c in SS_FIXED_COLORS.values()} | {_rgb("cornflowerblue")}
        palette = {_rgb(c) for c in PHASE_PALETTE}
        assert not (palette & reserved)

    def test_ss_colors_never_collide_with_line_or_liquid(self):
        # enough line phases to cycle the whole palette, plus all three SS phases
        line_phases = [f"P{i}" for i in range(len(PHASE_PALETTE) + 3)]
        ss = list(SS_FIXED_COLORS)
        cmap = build_phase_color_map(["L", *line_phases, *ss], ss_names=ss)

        ss_rgb = {name: _rgb(cmap[name]) for name in ss}
        assert ss_rgb == {name: _rgb(c) for name, c in SS_FIXED_COLORS.items()}

        other_rgb = {p: _rgb(cmap[p]) for p in ["L", *line_phases]}
        clashes = {p: c for p, c in other_rgb.items() if c in set(ss_rgb.values())}
        assert not clashes

    def test_ss_fixed_colors_mutually_distinct(self):
        vals = [_rgb(c) for c in SS_FIXED_COLORS.values()]
        assert len(set(vals)) == len(vals)
        assert _rgb("cornflowerblue") not in vals

    def test_ss_fallback_when_reserved_color_taken(self):
        """If a ternary override or duplicate claims a reserved color, SS names fall
        back deterministically and still avoid collisions."""
        ss = ["BCC", "BCC2"]
        cmap = build_phase_color_map(["L", "A", *ss], ss_names=ss)
        assert cmap["BCC"] == SS_FIXED_COLORS["BCC"]
        assert cmap["BCC2"] == "#6c5ce7"  # first fallback (BCC2 has no reserved color)
        assert len(set(cmap.values())) == len(cmap)
