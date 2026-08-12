"""Shared plotting style: the phase palette, reserved colors, and display-name rules.

Single source of truth for the colors used across the gliquid plotting stack (the
gliquid.plotting modules and the plotter classes in gliquid.binary / gliquid.ternary).
Leaf module -- imports no gliquid code, so both the model and plotting layers may
depend on it without cycles.
"""

from __future__ import annotations

# Fixed colors for solid-solution phases; reserved and never reused for hull phases.
SS_FIXED_COLORS = {
    "BCC": "#d7263d",
    "FCC": "#1b9aaa",
    "HCP": "#f4a259",
}

# Single-home shared colors (values must not drift between binary and ternary figures).
LIQUID_COLOR = "cornflowerblue"  # the liquid phase 'L' + fitted-liquidus legend proxy
ASSESSED_LIQUIDUS_COLOR = "#B82E2E"  # digitized/assessed liquidus curve + legend proxy
PREDICTED_LIQUIDUS_COLOR = "#117733"  # prediction-mode liquidus recolor + legend proxy

# The unified line-phase palette, shared by the binary and ternary figures: greedy
# max-min CIELAB dE76 selection from the project seed list, hard-excluded around the
# reserved colors above (min pairwise dE 23.5; min dE to the reserved set 23.1),
# rotation start chosen at review. Assignment ORDER matters -- build_phase_color_map
# cycles this list, so the first phases in a system take the earliest entries.
PHASE_PALETTE = [
    "#9400D3",
    "#66A61E",
    "#483D8B",
    "#DA70D6",
    "#FFD700",
    "#8B4513",
    "#C71585",
    "#8FBC8F",
    "#FF7F00",
    "#DEB887",
    "#708090",
    "#556B2F",
    "#A6761D",
    "#F781BF",
    "#BC8F8F",
    "#ADFF2F",
    "#5C4033",
    "#800080",
    "#FF4500",
    "#FB9A99",
]


def build_phase_color_map(phase_names, ss_names=()) -> dict[str, str]:
    """Deterministic phase -> color map shared by the binary AND ternary plot stacks.

    Line (non-liquid, non-SS) phases cycle PHASE_PALETTE in ``phase_names`` order and
    'L' is cornflowerblue. Solid-solution names take their reserved SS_FIXED_COLORS
    (deterministic fallbacks if exhausted), and the reserved colors are excluded from
    the line-phase palette so SS and line phases can never collide.
    """
    ss_names = list(ss_names)
    reserved = set(SS_FIXED_COLORS.values())
    reserved.add(LIQUID_COLOR)
    base_palette = [c for c in PHASE_PALETTE if c not in reserved]
    if not base_palette:
        base_palette = list(PHASE_PALETTE)

    phase_map: dict[str, str] = {"L": LIQUID_COLOR}
    line_phases = [p for p in phase_names if p != "L" and p not in ss_names]
    for idx, phase in enumerate(line_phases):
        phase_map[phase] = base_palette[idx % len(base_palette)]

    fallback_ss_palette = ["#6c5ce7", "#00a896", "#ef476f", "#ffd166"]
    used_colors = set(phase_map.values())
    fallback_idx = 0
    for ss_name in ss_names:
        fixed = SS_FIXED_COLORS.get(ss_name)
        if fixed is not None and fixed not in used_colors:
            phase_map[ss_name] = fixed
            used_colors.add(fixed)
            continue
        while (
            fallback_idx < len(fallback_ss_palette)
            and fallback_ss_palette[fallback_idx] in used_colors
        ):
            fallback_idx += 1
        if fallback_idx < len(fallback_ss_palette):
            phase_map[ss_name] = fallback_ss_palette[fallback_idx]
            used_colors.add(fallback_ss_palette[fallback_idx])
            fallback_idx += 1
        else:
            phase_map[ss_name] = "#3a86ff"
    return phase_map


def format_phase_display_name(phase_label: str, ss_names, comps) -> str:
    """Legend name: SS phases carry their component pair, others their raw label."""
    if phase_label in ss_names:
        return f"{phase_label} ({comps[0]}, {comps[1]})"
    return phase_label
