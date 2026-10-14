"""Shared plotting style: the phase palette, reserved colors, and display-name rules.

Single source of truth for the colors used across the gliquid plotting stack (the
gliquid.plotting modules and the plotter classes in gliquid.binary / gliquid.ternary).
Leaf module -- imports no gliquid code, so both the model and plotting layers may
depend on it without cycles.
"""

from __future__ import annotations

import math
import re
from fractions import Fraction
from functools import reduce

from pymatgen.core import Composition

# Fixed colors for solid-solution phases, reserved for them.
SS_FIXED_COLORS = {
    "BCC": "#d7263d",
    "FCC": "#1b9aaa",
    "HCP": "#f4a259",
}

# Colors shared by the binary and ternary figures.
LIQUID_COLOR = "cornflowerblue"  # the liquid phase 'L' + fitted-liquidus legend proxy
ASSESSED_LIQUIDUS_COLOR = "#B82E2E"  # digitized/assessed liquidus curve + legend proxy
# The liquidus hue lightened toward white, for the stretches of assessed liquidus with no solid
# free-energy reference; no legend entry of its own.
ASSESSED_LIQUIDUS_UNSUPPORTED_COLOR = "#D48282"
PREDICTED_LIQUIDUS_COLOR = "#117733"  # prediction-mode liquidus recolor + legend proxy

# Imputed phases: solid vertices whose energy was read off the fitted liquidus.
IMPUTED_EDGE_COLOR = "#3b3b3b"
IMPUTED_MARKER_COLOR = "#7b2d8e"
IMPUTED_LEGEND_LABEL = "Imputed"
# A user-defined phase (added or edited by hand) is drawn in the same style under its own word.
USER_DEFINED_LEGEND_LABEL = "User-defined"
IMPUTED_LINE_DASH = "4px,2px"  # Tx line of an imputed or user-defined phase, and its legend key
IMPUTED_BORDER_DASHES = (0, (1, 1.2))  # dotted outline of a marked band, and its legend proxy
IMPUTED_BORDER_WIDTH = 0.9
IMPUTED_BAND_TINT = 0.7  # share of the un-imputed ink a marked band keeps
# Side of the legend proxy patch, in font-size units, passed as both handlelength and handleheight.
IMPUTED_LEGEND_HANDLE_SIZE = 0.7
IMPUTED_LEGEND_HANDLE_TEXT_PAD = 0.4  # patch-to-label gap of the key, in font-size units

# A highlighted ternary liquidus: the rest of the surface blends each colorscale stop this far
# toward its own grey.
DIMMED_SURFACE_DESATURATION = 0.7
DIMMED_PHASE_COLOR = "#c8c8c8"
FIELD_OUTLINE_COLOR = "#1a1a1a"
FIELD_OUTLINE_WIDTH = 5
# The boundary of the two-liquid field, drawn in every view.
GAP_OUTLINE_COLOR = "#7f7f7f"
GAP_OUTLINE_DASH = "dot"


def non_dft_legend_label(kinds) -> str:
    """The legend entry for phases drawn in the imputed style.

    Args:
        kinds: the non-DFT kinds present, ``"imputed"`` and/or ``"user"``.

    Returns:
        str: :data:`USER_DEFINED_LEGEND_LABEL` when every phase is user-defined, both words
        joined when both kinds are present, else :data:`IMPUTED_LEGEND_LABEL`.
    """
    kinds = set(kinds)
    if "user" not in kinds:
        return IMPUTED_LEGEND_LABEL
    if "imputed" not in kinds:
        return USER_DEFINED_LEGEND_LABEL
    return f"{IMPUTED_LEGEND_LABEL} / {USER_DEFINED_LEGEND_LABEL.lower()}"


def imputed_band_alphas(alpha: float, tint: float = IMPUTED_BAND_TINT) -> tuple[float, float]:
    """Fill alphas for the two stacked layers of a marked phase-comparison band.

    Args:
        alpha: the fill alpha each of the two layers uses on an un-imputed band.
        tint: share of the un-imputed ink the marked band keeps.

    Returns:
        tuple: ``(under, over)``, the fill alphas of the lower and upper layer. Compositing
        those two over a white ground gives ``white + tint * (baseline - white)``, where
        ``baseline`` is the same two layers drawn at ``alpha`` -- so the overlap keeps the
        un-imputed hue exactly and only its ink drops.
    """
    over = tint * alpha
    under = tint * alpha * (1 - alpha) / (1 - over)
    return under, over


# Liquidus MAPE (percent) at which mape_color reaches the red end of its scale.
MAPE_COLOR_MAX = 40.0

# (MAPE %, fraction of the way to the red end) knots mape_color interpolates linearly between.
MAPE_COLOR_KNOTS = ((0.0, 0.0), (5.0, 0.08), (20.0, 0.70), (MAPE_COLOR_MAX, 1.0))


def mape_color(mape: float) -> str:
    """The color of a liquidus MAPE on the reversed RdYlGn scale, as ``"rgb(r,g,b)"``.

    0 % is the green end and ``MAPE_COLOR_MAX`` or more the red end, piecewise linear through
    ``MAPE_COLOR_KNOTS`` so the color changes fastest between 5 and 20 %; channels are the
    colormap's floats times 255, truncated to integers.
    """
    import numpy as np
    from matplotlib import colormaps

    values, fractions = zip(*MAPE_COLOR_KNOTS)
    frac = 1.0 - float(np.interp(mape, values, fractions))
    r, g, b, _ = colormaps["RdYlGn"](frac)
    return f"rgb({int(r * 255)},{int(g * 255)},{int(b * 255)})"


def contrast_text(rgb: str) -> str:
    """``"black"`` or ``"white"``, whichever has the higher WCAG contrast on ``"rgb(r,g,b)"``."""
    channels = [int(c) / 255 for c in rgb[rgb.index("(") + 1 : rgb.index(")")].split(",")]
    linear = [c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4 for c in channels]
    luminance = 0.2126 * linear[0] + 0.7152 * linear[1] + 0.0722 * linear[2]
    return "black" if (luminance + 0.05) / 0.05 >= 1.05 / (luminance + 0.05) else "white"


# The line-phase palette shared by the binary and ternary figures, in assignment order
# (build_phase_color_map cycles it).
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


def build_phase_color_map(
    phase_names, ss_names=(), polymorph_names=(), imputed_names=()
) -> dict[str, str]:
    """Deterministic phase -> color map shared by the binary AND ternary plot stacks.

    Line (non-liquid, non-SS) phases cycle PHASE_PALETTE grouped elemental polymorphs
    first, then remaining phases, then imputed phases -- composition order (their
    relative order in ``phase_names``) preserved within each group -- and 'L' is
    cornflowerblue. Solid-solution names take their reserved SS_FIXED_COLORS
    (deterministic fallbacks if exhausted), and the reserved colors are excluded from
    the line-phase palette so SS and line phases can never collide.
    """
    ss_names = list(ss_names)
    polymorph_names = set(polymorph_names)
    imputed_names = set(imputed_names)
    reserved = set(SS_FIXED_COLORS.values())
    reserved.add(LIQUID_COLOR)
    base_palette = [c for c in PHASE_PALETTE if c not in reserved]
    if not base_palette:
        base_palette = list(PHASE_PALETTE)

    phase_map: dict[str, str] = {"L": LIQUID_COLOR}
    line_phases = [p for p in phase_names if p != "L" and p not in ss_names]
    ordered_line_phases = (
        [p for p in line_phases if p in polymorph_names]
        + [p for p in line_phases if p not in polymorph_names and p not in imputed_names]
        + [p for p in line_phases if p not in polymorph_names and p in imputed_names]
    )
    for idx, phase in enumerate(ordered_line_phases):
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


def subscript_formula(name: str) -> str:
    """Wrap stoichiometric number runs (those following a letter or ')') in <sub> tags.

    A decimal run is wrapped whole: ``Fe1.7Ge`` -> ``Fe<sub>1.7</sub>Ge``.
    Shared by the binary field labels and the ternary legend and hover text.
    """
    return re.sub(r"(?<=[A-Za-z\)])(\d+(?:\.\d+)?)", r"<sub>\1</sub>", str(name))


# A whitespace-separated formula token; the other tokens of an MPDS label (rt, ht, lt, stab,
# m-orth) are dropped.
_FORMULA_TOKEN_RE = re.compile(r"(?:[A-Z][a-z]?(?:\d+(?:\.\d+)?)?)+")
_DECIMAL_PLACES_RE = re.compile(r"\d+\.(\d+)")
_MAX_SNAP_DENOMINATOR = 12
_MAX_STOICH_INTEGER = 999
_MAX_REREDUCED_INTEGER = 20
"""Ceiling on the integers re-derived from the 3-significant-figure atomic fractions when the
exact reduction exceeds ``_MAX_STOICH_INTEGER``. Above it the label stays in fraction form:
the re-reduction is only worth making when it is shorter to read than what it replaces."""


def _formula_text(label) -> str:
    """Formula part of a phase label: everything before the first '/', descriptors dropped."""
    head = str(label).split("/")[0]
    return "".join(tok for tok in head.split() if _FORMULA_TOKEN_RE.fullmatch(tok))


def _three_sig_figs(value: float) -> str:
    """``value`` to 3 significant figures in fixed-point notation (never exponent form)."""
    decimals = 2 - math.floor(math.log10(value)) if value > 0 else 2
    return f"{value:.{max(0, decimals)}f}"


def _rereduced_counts(printed: dict[str, str]) -> dict[str, int] | None:
    """Whole-number counts from already-printed atomic fractions, or ``None`` to keep them.

    ``None`` means the fractions do not reduce to anything worth reading: an integer above
    ``_MAX_REREDUCED_INTEGER``, or an amount that prints as zero.
    """
    amounts = {el: Fraction(text) for el, text in printed.items()}
    if any(amount <= 0 for amount in amounts.values()):
        return None
    scale = reduce(math.lcm, (amount.denominator for amount in amounts.values()))
    counts = {el: int(amount * scale) for el, amount in amounts.items()}
    divisor = reduce(math.gcd, counts.values())
    counts = {el: count // divisor for el, count in counts.items()}
    if max(counts.values()) > _MAX_REREDUCED_INTEGER:
        return None
    return counts


def _source_amounts(source) -> tuple[dict[str, Fraction], Fraction]:
    """Element -> exact amount, plus the default snap tolerance for that source."""
    if isinstance(source, Composition):
        composition, decimals = source, 0
    else:
        text = _formula_text(source)
        if not text:
            raise ValueError(f"{source!r} carries no formula")
        composition = Composition(text)
        decimals = max((len(d) for d in _DECIMAL_PLACES_RE.findall(text)), default=0)
    amounts = {el: Fraction(str(amount)) for el, amount in composition.as_dict().items()}
    tol = Fraction(1, 2 * 10**decimals) if decimals else Fraction(1, 10**6)
    return amounts, tol


def standard_formula(source, components, *, tol: float | None = None) -> str:
    """Write a compound in the system's own component order with whole-number stoichiometry.

    The one compound-label rule for the binary figures. Amounts are snapped to the nearest
    rational with denominator <= 12 within ``tol``, scaled to integers, and reduced. A formula
    needing an integer above 999 is re-derived from its atomic fractions at 3 significant
    figures, and written as those fractions unless the re-derived integers themselves stay at
    or below ``_MAX_REREDUCED_INTEGER``. The result carries no character outside
    ``[A-Za-z0-9.]``; wrap it in :func:`subscript_formula` for a figure that subscripts.

    Args:
        source: a pymatgen ``Composition``, or a phase label to parse. A label is read up to
            the first '/', with non-formula tokens (``rt``, ``ht``, ``lt``, ``stab``,
            ``m-orth``) dropped.
        components: the system's ordered component list; fixes the element order of the output.
        tol (float): snap tolerance. Defaults to half the last decimal place printed in a label
            (``YSi1.67`` -> 0.005) and to 1e-6 for an all-integer label or a ``Composition``.

    Returns:
        str: the label, e.g. ``Al17Cu33``, ``Si5Y3``, ``Au0.690Ga0.310``.

    Raises:
        ValueError: ``source`` carries no parseable formula, or names an element that is not
            one of ``components``.
    """
    symbols = [str(Composition(str(c)).elements[0]) for c in components]
    amounts, default_tol = _source_amounts(source)
    snap_tol = Fraction(str(tol)) if tol is not None else default_tol

    foreign = [el for el in amounts if el not in symbols]
    if foreign:
        raise ValueError(f"{source!r} names {foreign}, which is not among {list(components)}")

    snapped = {}
    for el, amount in amounts.items():
        candidate = amount.limit_denominator(_MAX_SNAP_DENOMINATOR)
        snapped[el] = candidate if abs(candidate - amount) < snap_tol else amount
    snapped = {el: amount for el, amount in snapped.items() if amount > 0}
    if not snapped:
        raise ValueError(f"{source!r} has no non-zero amount")

    scale = reduce(math.lcm, (amount.denominator for amount in snapped.values()))
    counts = {el: int(amount * scale) for el, amount in snapped.items()}
    divisor = reduce(math.gcd, counts.values())
    counts = {el: count // divisor for el, count in counts.items()}

    if max(counts.values()) > _MAX_STOICH_INTEGER:
        total = sum(snapped.values())
        printed = {
            el: _three_sig_figs(float(snapped[el] / total)) for el in symbols if el in snapped
        }
        rereduced = _rereduced_counts(printed)
        if rereduced is not None:
            counts = rereduced
        else:
            return "".join(f"{el}{printed[el]}" for el in symbols if el in printed)
    return "".join(f"{el}{counts[el] if counts[el] != 1 else ''}" for el in symbols if el in counts)


# Greek polymorph prefixes -> unicode letters.
_GREEK_MAP = {
    "alpha": "α",
    "beta": "β",
    "gamma": "γ",
    "delta": "δ",
    "epsilon": "ε",
    "zeta": "ζ",
    "eta": "η",
    "theta": "θ",
    "iota": "ι",
    "kappa": "κ",
    "lambda": "λ",
    "mu": "μ",
    "nu": "ν",
    "xi": "ξ",
    "omicron": "ο",
    "pi": "π",
    "rho": "ρ",
    "sigma": "σ",
    "tau": "τ",
    "upsilon": "υ",
    "phi": "φ",
    "chi": "χ",
    "psi": "ψ",
    "omega": "ω",
}

# Long crystal-structure names -> short forms.
_STRUCT_ABBREV = {
    "orthorhombic": "ortho",
    "rhombohedral": "rhomb",
    "monoclinic": "mono",
    "tetragonal": "tetra",
    "hexagonal": "hex",
    "trigonal": "trig",
    "triclinic": "tric",
    "body centered cubic": "bcc",
    "face centered cubic": "fcc",
    "hexagonal close packed": "hcp",
    "complex cubic a12": "A12",
    "complex cubic a13": "A13",
    "diamond cubic": "dc",
    "graphite": "gra",
}

_TAG_PHASE_TAG_RE = r"(?:rt|lt|ht\d*|r\d*)"  # room/low/high-temperature ordinal tags


def _parse_elemental_phase(name: str):
    """Parse an elemental solid-solution label into ``(greek_word, element, struct, tag)``.

    Accepts both space- and hyphen-separated greek prefixes (e.g. ``"alpha Mn (bcc)"`` and
    ``"alpha-Ga (orthorhombic)"``). Returns ``None`` for compounds (e.g. ``"ZrMn2"``) so the
    caller falls through to formula subscripting.
    """
    s = str(name).strip()
    greek_word = None
    struct_prefix = None
    # "<Structure words> <Element>" ("Diamond cubic Si"): fires only on a capitalised, non-greek
    # leading token.
    toks = s.split()
    if (
        len(toks) >= 2
        and re.fullmatch(r"[A-Z][a-z]?", toks[-1])
        and toks[0][:1].isupper()
        and toks[0].lower() not in _GREEK_MAP
        and not re.fullmatch(r"[A-Z][a-z]?", " ".join(toks[:-1]))
    ):
        return None, toks[-1], " ".join(toks[:-1]), None
    # A greek polymorph prefix ("alpha-Ga (orthorhombic)") or a lowercase structure tag ("fcc-Al")
    # followed by an element symbol.
    m = re.match(r"^([A-Za-z0-9]+)[ \-](.+)$", s)
    if m and re.match(r"^[A-Z][a-z]?(?:$|[\s(])", m.group(2).strip()):
        prefix = m.group(1)
        if prefix.lower() in _GREEK_MAP:
            greek_word = prefix.lower()
            s = m.group(2).strip()
        elif prefix[0].islower():
            struct_prefix = prefix
            s = m.group(2).strip()

    m2 = re.match(r"^([A-Z][a-z]?)\s*(?:\(([^)]*)\))?\s*(.*)$", s)
    if not m2:
        return None
    element = m2.group(1)
    struct = (m2.group(2) or "").strip() or struct_prefix
    tag = (m2.group(3) or "").strip() or None
    # A trailing remainder that is not a temperature tag marks a compound ("ZrMn2", "Ce(FeSi)2").
    if tag and not re.fullmatch(_TAG_PHASE_TAG_RE, tag, re.IGNORECASE):
        return None
    return greek_word, element, struct, tag


def _abbrev_structure(struct: str) -> str:
    """Abbreviate a crystal-structure name; pass short names through, drop unrecognised long ones.

    Returns ``''`` for long, unmapped descriptors (e.g. space-group symbols like ``P6_3/mmc``)
    so the label falls back to just ``(El)``/``(αEl)`` rather than showing truncated noise.
    """
    s = struct.strip()
    key = s.lower()
    if key in _STRUCT_ABBREV:
        return _STRUCT_ABBREV[key]
    m = re.search(r"\b([A-D]\d{1,2})\b", s)  # Strukturbericht token, e.g. A13, B2
    if m:
        return m.group(1)
    if re.search(r"\d", s) and ("-" in s or "/" in s):
        return ""  # space-group symbol (Fm-3m, R-3m, P6_3/mmc) -> omit
    if len(s) <= 6:
        return s
    return ""  # unrecognised long descriptor -> omit


def abbreviate_phase_name(name: str, all_names: list[str], components=None) -> str:
    """Format a phase label: greek polymorph prefixes, subscripted stoichiometries,
    abbreviated crystal structures.

    Elemental phases render as ``αMn`` with the structure parenthesised after it —
    ``α-Mn (bcc)`` — matching ``phase_transitions.json``'s ``<name>-<El> (<structure>)``
    convention, so a corner label in the ternary figure and a field label here read the
    same. The greek prefix is dropped when the element has only one phase present in
    ``all_names`` (``Fe (bcc)``), since there is then nothing to tell apart.

    Compounds render with subscripts (``ZrMn2`` -> ``ZrMn<sub>2</sub>``). Pass ``components``
    (the system's ordered component list) to put a compound through ``standard_formula``
    first, so it reads in component order with whole-number stoichiometry and no polymorph
    suffix; omit it, as the ternary edge sub-plots do, to subscript the raw name.
    ``"Liquid"``/``"L"`` -> ``"L"``.
    """
    name = str(name).strip()
    if name in ("L", "Liquid"):
        return "L"

    parsed = _parse_elemental_phase(name)
    if parsed is not None:
        greek_word, element, struct, tag = parsed
        same_element = sum(
            1
            for other in all_names
            if (_parse_elemental_phase(str(other).strip()) or (None, None, None, None))[1]
            == element
        )
        prefix = _GREEK_MAP.get(greek_word, "") if (greek_word and same_element > 1) else ""
        label = f"{prefix}-{element}" if prefix else element
        abbr = _abbrev_structure(struct) if struct else ""
        if abbr:
            label += f" ({abbr})"
        if tag:
            label += f" {tag}"
        return label

    if components:
        try:
            return subscript_formula(standard_formula(name, components))
        except ValueError:
            pass
    return subscript_formula(name)
