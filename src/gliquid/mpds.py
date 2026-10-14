"""
Author: Joshua Willwerth
Description: MPDS digitized-phase-diagram parsing for gliquid: downloading/caching the
publicly-available Materials Platform for Data Science (MPDS) binary phase-diagram JSONs,
extracting the digitized liquidus, classifying the digitized solid phases, and identifying
the invariant points the fitting constraints are built from.

System input parsing lives in ``gliquid.phase.validate_and_format_system``; the DFT entry
cache/load layer lives in ``gliquid.api``; the DFT-to-MPDS phase assignment and the
solid-coverage gate live in ``gliquid.solid_coverage``.
GitHub: https://github.com/willwerj
ORCID: https://orcid.org/0009-0004-6334-9426
"""

from __future__ import annotations

import logging
import re

import numpy as np
from pymatgen.analysis.phase_diagram import PhaseDiagram
from pymatgen.core import Composition

import gliquid.api as api
import gliquid.cache as cache
import gliquid.config as config
from gliquid.phase import UNARY, validate_and_format_system
from gliquid.plotting.style import standard_formula

logger = logging.getLogger(__name__)

# MPDS shape['phase'] identifiers look like 'Zr/229/cI2' -- prototype/spacegroup/Pearson.
_SHAPE_PHASE_RE = re.compile(r"^(?P<prototype>[^/]+)/(?P<spacegroup>\d+)/(?P<pearson>[^/]+)$")

# MPDS polymorph suffixes and their temperature ordering (see polymorph_rank); 'hp', a
# high-pressure form, sorts last.
_POLYMORPH_RE = re.compile(r"^(lt|rt|ht|hp)(\d*)$", re.I)
_POLYMORPH_BASE_RANK = {"lt": 0, "rt": 10, "": 10, "ht": 20, "hp": 90}
_HTML_TAG_RE = re.compile(r"<[^>]+>")

# The leading parenthesised group of an MPDS phase label: '(Ti)', '(Ti) rt', '(Ti, Cr) ht'.
_COMPONENT_LABEL_RE = re.compile(r"^\(\s*([^()]*?)\s*\)")

# extract_digitized_liquidus fills any consecutive-point gap wider than _FILL_GAP_X inside a
# covered region with points _FILL_STEP_X apart; a wider gap left in its output is an
# undigitized hole between disjoint liquid regions.
_FILL_GAP_X = 0.06
_FILL_STEP_X = 0.03

# How far off a pure end, in mole fraction, a point still counts as on it; _collapse_columns
# keeps one point per composition column there.
_AXIS_X_TOL = 1e-3

# Points at different compositions inside the axis window collapse only when their column
# spans at least this temperature; points at one composition always collapse.
_AXIS_MIN_SPREAD_K = 40.0

# -- Lean records: the pre-fill stitched liquidus stored under the reserved key, plus only
# ``reference``/``chemical_elements``/``temp``/``comp_range`` (see :func:`record_mode`) --
_GLIQUID_KEY = "_gliquid"

#: Bumped when the shape of the ``_gliquid`` block changes.
LEAN_SCHEMA = 1


def record_mode(mpds_json) -> str:
    """Classify an MPDS record: ``'full'``, ``'lean'`` or ``'empty'``.

    * ``'empty'`` -- the ``{"reference": None}`` placeholder ``load_mpds_data`` caches for
      a system MPDS has no digitized diagram for. Checked FIRST, so a placeholder is
      classified the same way in a lean store as in a full one.
    * ``'lean'`` -- the reduction above: no ``shapes``, the stitched liquidus instead.
    * ``'full'`` -- an ordinary MPDS json.
    """
    if not isinstance(mpds_json, dict) or mpds_json.get("reference") is None:
        return "empty"
    block = mpds_json.get(_GLIQUID_KEY)
    if isinstance(block, dict) and block.get("mode") == "lean":
        return "lean"
    return "full"


def require_full_record(mpds_json, caller: str, *, escape: str = "") -> None:
    """Raise unless ``mpds_json`` still carries its digitized ``shapes``.

    On a lean record :func:`identify_mpds_phases` would return no phases, and
    :func:`gliquid.solid_coverage.assess_solid_coverage` would then report nothing unsupported
    and pass the gate. There is no auto-degrade path: a fit without invariant constraints and
    without the coverage gate is a different fit.
    """
    if record_mode(mpds_json) != "lean":
        return
    raise config.CacheModeError(
        f"{caller} needs the digitized 'shapes' block, and this MPDS record is LEAN "
        f"(liquidus-only; see gliquid.mpds.record_mode). Returning an empty phase list "
        f"here would read downstream as 'no unsupported compounds' and pass the solid-"
        f"coverage gate, so this raises instead. Fix: point gliquid at a store migrated "
        f"with `python -m gliquid.cache migrate --mpds-mode full`"
        + (f", or {escape}." if escape else ".")
    )


def lean_record(mpds_json: dict) -> dict:
    """Reduce a full MPDS json to a lean one. Inverse of nothing — ``shapes`` is gone.

    Computes :func:`_stitched_liquidus` ONCE and stores its pre-fill ``(x, T)`` points plus
    the ``covered`` region structure, so that ``extract_digitized_liquidus`` and
    ``liquidus_coverage`` reproduce their full-record answers exactly. The only other fields
    kept are ``chemical_elements``, ``temp``, ``comp_range`` and ``reference``; ``labels`` and
    the top-level ``entry``/``jcode``/``year`` are dropped.

    ``is_partial`` is **derived** at read time, never stored. The two empty cases are
    distinguished by ``covered`` itself: ``None`` means "no liquid field was digitized at all"
    and ``[]`` means "'L' shapes existed but none yielded a usable branch", which
    ``_stitched_liquidus`` reports as partial.

    ``chemical_elements`` is written explicitly, ``None`` when unknown:
    :func:`mpds_frame_matches` treats an ABSENT frame block as matching the caller.
    """
    import gliquid.cache as _cache  # local: cache imports config only, mpds imports cache

    stitched, is_partial, covered = _stitched_liquidus(mpds_json, quiet=True)
    if stitched is None and not covered:
        covered = [] if is_partial else None
    header = _cache.mpds_header(mpds_json, lean=True)
    header.setdefault("chemical_elements", None)
    return {
        **header,
        _GLIQUID_KEY: {
            "mode": "lean",
            "schema": LEAN_SCHEMA,
            "stitched": stitched,
            "covered": covered,
        },
    }


def _lean_stitched(block: dict, quiet: bool) -> tuple[list[list] | None, bool, list[list[float]]]:
    """``_stitched_liquidus``'s answer, read back off a lean record's ``_gliquid`` block.

    Emits the same warnings, except that the disjoint-shape message names the covered regions
    instead of the branch count.
    """
    stitched, covered = block.get("stitched"), block.get("covered")
    if covered is None:
        if not quiet:
            logger.warning("No liquidus data found.")
        return None, False, []
    if not covered:
        if not quiet:
            logger.warning("Insufficient liquidus data.")
        return None, True, []

    is_partial = False
    if covered[0][0] > 0.03 or covered[-1][1] < 0.97:
        if not quiet:
            logger.warning(
                f"MPDS liquidus does not span the entire composition range! "
                f"({100 * covered[0][0]}-{100 * covered[-1][1]})"
            )
        is_partial = True
    if len(covered) > 1:
        if not quiet:
            logger.warning(
                f"MPDS liquid field is drawn as disjoint 'L' shapes covering "
                f"{[[round(100 * lo, 2), round(100 * hi, 2)] for lo, hi in covered]} "
                f"at.%; the liquidus between them is not digitized."
            )
        is_partial = True
    return stitched, is_partial, covered


def shape_to_list(svgpath: str) -> list[list]:
    """Parse SVG path to a list of [X, T] pairs"""
    pairs = [s for s in svgpath.split(" ") if s not in ["L", "M"]]
    return [[float(p.split(",")[0]) / 100, float(p.split(",")[1]) + 273.15] for p in pairs]


def parse_shape_structure(shape: dict) -> dict:
    """MPDS ``shape['phase']`` -> ``{'prototype', 'spacegroup', 'pearson'}``.

    ``'Th0.805Tm0.195/225/cF4'`` -> prototype ``'Th0.805Tm0.195'``, spacegroup ``225``,
    Pearson ``'cF4'``. The spacegroup number is the join key onto
    :data:`gliquid.phase.SS_SPACEGROUPS`, which is how a digitized solid-solution field is
    matched to a loaded solid-solution model.

    Every field is ``None`` when MPDS did not resolve the shape to a structure. That is the
    normal case for the full-composition ``'(A, B)'`` labels and also happens for some
    ordinary compounds.
    Malformed identifiers degrade to all-``None`` rather than raising: this metadata is
    advisory, never load-bearing for parsing.
    """
    raw = shape.get("phase")
    match = _SHAPE_PHASE_RE.match(raw) if isinstance(raw, str) else None
    if match is None:
        return {"prototype": None, "spacegroup": None, "pearson": None}
    return {
        "prototype": match["prototype"],
        "spacegroup": int(match["spacegroup"]),
        "pearson": match["pearson"],
    }


def liquid_shape_paths(mpds_json: dict) -> list[str]:
    """Every liquid-field svgpath in the json — the shapes labelled exactly 'L'.

    A miscibility gap, a monotectic, or a pair of terminal wedges is drawn as SEVERAL
    disjoint one-phase 'L' shapes, and the liquidus is the lower envelope of all of them.

    The label is matched exactly, so the two-phase dome between the branches ('L1 + L2',
    'L + G', ...) is never mistaken for liquid field. A shape whose ``kind`` is something
    other than 'phase' is a drawing overlay or a mislabelled solid compound and is likewise
    not liquid; ``kind``-less shapes are kept.

    Raises:
        config.CacheModeError: on a LEAN record, which has no ``shapes``. Reachable only by
            a direct caller — ``_stitched_liquidus`` short-circuits before it.
    """
    require_full_record(mpds_json, "liquid_shape_paths")
    return [
        shape["svgpath"]
        for shape in mpds_json.get("shapes", [])
        if shape.get("label") == "L"
        and shape.get("kind") in (None, "phase")
        and shape.get("svgpath")
    ]


def _shape_liquidus(svgpath: str, mpds_json: dict) -> list[list] | None:
    """One 'L' shape -> its liquidus branch, sorted by composition, or None.

    Strip the frame-boundary points (the flat lid an open wedge is closed with is frame,
    not liquidus), split what is left into monotonic sections, stitch the sections that
    line up with the pure-element endpoints onto the longest one, and collapse what
    survives to one temperature per composition (:func:`_collapse_columns`), which is what
    removes the ring-closure edges the frame strip cannot see. ``None`` when the shape has
    too few interior points to be a curve.
    """
    liquidus = shape_to_list(svgpath)

    def t_at_boundary(t, boundary):
        return t <= boundary[0] + 4 or t >= boundary[1] - 4

    liquidus = [pt for pt in liquidus if not t_at_boundary(pt[1] - 273.15, mpds_json["temp"])]

    if len(liquidus) < 3:
        return None

    def section_liquidus(points):
        """Splits liquidus into continuous sections."""
        sections, current_section, direction = [], [], None
        for i in range(len(points) - 1):
            x1, x2 = points[i][0], points[i + 1][0]
            new_direction = "increasing" if x2 > x1 else "decreasing" if x2 < x1 else None
            current_section.append(points[i])

            if new_direction != direction:
                if current_section:
                    sections.append(current_section)
                current_section = []
            direction = new_direction

        current_section.append(points[-1])
        sections.append(current_section)
        return sections

    sections = section_liquidus(liquidus)
    sections.sort(key=len, reverse=True)
    main_section = sorted(sections.pop(0))

    lhs = [0, UNARY[mpds_json["chemical_elements"][0]].t_fusion]
    rhs = [1, UNARY[mpds_json["chemical_elements"][1]].t_fusion]

    def within_tol_from_line(p1, p2, p3, tol):
        """Checks if a point is within tolerance from a line."""
        try:
            m = (p2[1] - p1[1]) / (p2[0] - p1[0])
            y_h = m * (p3[0] - p1[0]) + p1[1]
            return abs(p3[1] - y_h) <= tol
        except ZeroDivisionError:
            return abs(p2[1] - p1[1]) <= tol

    for sec in sections:
        sec.sort()
        if sec[-1][0] <= main_section[0][0] and within_tol_from_line(
            main_section[0], lhs, sec[-1], 250
        ):
            main_section = sec + main_section
        elif sec[0][0] >= main_section[-1][0] and within_tol_from_line(
            main_section[-1], rhs, sec[0], 250
        ):
            main_section += sec
        elif len(sec) == 2:
            if sec[0][0] < main_section[0][0] and within_tol_from_line(
                main_section[0], lhs, sec[0], 170
            ):
                main_section = sec + main_section

            elif sec[-1][0] > main_section[-1][0] and within_tol_from_line(
                main_section[-1], rhs, sec[-1], 170
            ):
                main_section += sec

    return _collapse_columns(main_section, mpds_json)


def _collapse_columns(branch: list[list], mpds_json: dict) -> list[list]:
    """One temperature per composition: the ring-closure edges MPDS draws up the axes, gone.

    An 'L' svgpath is a closed RING, not a curve. Having traced the liquidus it closes the
    liquid field by running up a pure-element axis and back across a top lid, and vertices
    of those closure edges reach the stitched section: the frame strip only catches a lid
    sitting ON the frame, and ``within_tol_from_line``'s zero-width fallback judges a vertical
    stitch candidate on the gap between its two ANCHORS, never on the candidate's own
    temperature. Drawn, the survivor is a vertical spike at the edge of the plot.

    The liquidus is single-valued in composition, so two points at the same composition
    cannot both be it, and the liquid field lies ABOVE the liquidus, so the colder one is
    the boundary. At a pure end the liquidus is a single point -- that element's melting
    temperature -- so the whole axis column collapses to one, picked by distance from
    ``t_fusion`` rather than by temperature: a terminal solid solution merged into the same
    ring also reaches the axis, and taking the coldest point there swaps the melting point
    for a solidus.

    Like :func:`_lower_envelope` this only ever DROPS a point, never moves or invents one,
    so the shape's composition span, its ``covered`` interval and ``is_partial`` are
    untouched.
    """
    t_fusion = [UNARY[element].t_fusion for element in mpds_json["chemical_elements"]]

    def axis_end(x):
        """0 or 1 when this composition is ON a pure-element axis, else None."""
        if x <= _AXIS_X_TOL:
            return 0
        if x >= 1 - _AXIS_X_TOL:
            return 1
        return None

    # One point per exact composition first: the coldest off the axes; on an axis the one nearest
    # t_fusion, ties broken cold.
    columns = {}
    for point in branch:
        columns.setdefault(point[0], []).append(point)

    kept = []
    for x, column in columns.items():
        end = axis_end(x)
        if len(column) == 1:
            kept.append(column[0])
        elif end is None:
            kept.append(min(column, key=lambda p: p[1]))
        else:
            kept.append(min(column, key=lambda p: (abs(p[1] - t_fusion[end]), p[1])))

    # Then the closure column at an axis: collapsed only when it spans _AXIS_MIN_SPREAD_K.
    for end in (0, 1):
        window = [p for p in kept if axis_end(p[0]) == end]
        if len(window) < 2:
            continue
        if max(p[1] for p in window) - min(p[1] for p in window) < _AXIS_MIN_SPREAD_K:
            continue
        survivor = min(window, key=lambda p: (abs(p[1] - t_fusion[end]), p[1]))
        kept = [p for p in kept if axis_end(p[0]) != end or p is survivor]

    return sorted(kept)


def _lower_envelope(branches: list[list[list]]) -> list[list]:
    """Union of the liquidus branches, keeping the coolest one wherever they overlap.

    The liquid field lies ABOVE the liquidus, so where two 'L' shapes cover the same
    composition the liquidus is the lower of the two curves. Disjoint wedges — the usual
    reason a diagram has several 'L' shapes — simply concatenate, and the hole between
    them stays a hole: a point is only ever dropped, never moved or invented.

    Branch provenance IS flattened here, which is why ``_stitched_liquidus`` returns its
    ``covered`` intervals separately: the fill step in ``extract_digitized_liquidus`` needs
    them to tell a sparse patch inside one wedge from the hole between two wedges, and it
    cannot recover that from the flat point list alone.
    """
    if len(branches) == 1:
        return branches[0]

    def t_at(branch, x):
        """Branch temperature at composition x, or None where the branch does not reach."""
        if x < branch[0][0] or x > branch[-1][0]:
            return None
        for p1, p2 in zip(branch, branch[1:]):
            if p1[0] <= x <= p2[0]:
                if p2[0] == p1[0]:
                    return min(p1[1], p2[1])
                return p1[1] + (x - p1[0]) / (p2[0] - p1[0]) * (p2[1] - p1[1])
        return branch[-1][1]

    envelope, seen = [], set()
    for i, branch in enumerate(branches):
        for pt in branch:
            if any(
                t is not None and t < pt[1]
                for t in (t_at(other, pt[0]) for j, other in enumerate(branches) if j != i)
            ):
                continue
            if tuple(pt) in seen:  # a vertex shared by two shapes is kept once
                continue
            seen.add(tuple(pt))
            envelope.append(pt)
    return sorted(envelope)


def _stitched_liquidus(
    mpds_json: dict, quiet: bool = False
) -> tuple[list[list] | None, bool, list[list[float]]]:
    """Sectioned-and-stitched digitized liquidus, BEFORE gap filling.

    The front half of ``extract_digitized_liquidus``: stitch a liquidus branch out of
    EVERY 'L' shape (:func:`liquid_shape_paths`, :func:`_shape_liquidus`) and take the
    lower envelope of their union. Every returned point is digitized data — no synthetic
    fill — so consecutive-point composition gaps are only measurable here
    (``liquidus_coverage``); the fill step erases the ones inside a region.

    Returns ``(curve, is_partial, covered)``.

    ``covered`` is the merged set of ``[x_lo, x_hi]`` composition intervals the branches
    actually span — the region structure ``_lower_envelope`` is about to flatten away.
    ``extract_digitized_liquidus`` fills only gaps whose two endpoints fall inside the SAME
    interval, which is what keeps densification from fabricating a liquidus across the hole
    between two disjoint liquid fields. One interval means the union is contiguous and
    every gap is a sampling gap.

    ``is_partial`` describes the UNION's coverage, not one shape's: it is set when the
    branch spans stop short of the pure ends, and also when they do not join up
    (``len(covered) > 1``), since a diagram whose liquid field is drawn in disjoint pieces
    covers a composition interval no digitized shape reaches. A single-'L' diagram has one
    branch, so only the endpoint condition can fire.

    ``quiet`` suppresses the warnings so a metrics pass over a json that was already
    extracted does not double-count them in scanned logs.

    Both ``extract_digitized_liquidus`` and ``liquidus_coverage`` route through here; on a
    lean record the early return below serves the stored pre-fill curve.
    """
    if mpds_json.get("reference") is None:
        if not quiet:
            logger.warning("No data in MPDS JSON.")
        return None, False, []
    block = mpds_json.get(_GLIQUID_KEY)
    if isinstance(block, dict) and block.get("mode") == "lean":
        return _lean_stitched(block, quiet)
    svgpaths = liquid_shape_paths(mpds_json)
    if not svgpaths:
        if not quiet:
            logger.warning("No liquidus data found.")
        return None, False, []

    branches = [
        branch for branch in (_shape_liquidus(path, mpds_json) for path in svgpaths) if branch
    ]
    if not branches:
        if not quiet:
            logger.warning("Insufficient liquidus data.")
        return None, True, []

    is_partial = False
    covered = merge_ranges([[branch[0][0], branch[-1][0]] for branch in branches])

    # If the liquidus does not have endpoints near the ends of the composition range, melting temps
    # won't be good
    if covered[0][0] > 0.03 or covered[-1][1] < 0.97:
        if not quiet:
            logger.warning(
                f"MPDS liquidus does not span the entire composition range! "
                f"({100 * covered[0][0]}-{100 * covered[-1][1]})"
            )
        is_partial = True

    if len(covered) > 1:
        if not quiet:
            logger.warning(
                f"MPDS liquid field is drawn as {len(branches)} disjoint 'L' shapes "
                f"covering {[[round(100 * lo, 2), round(100 * hi, 2)] for lo, hi in covered]} "
                f"at.%; the liquidus between them is not digitized."
            )
        is_partial = True

    return _lower_envelope(branches), is_partial, covered


def extract_digitized_liquidus(mpds_json: dict) -> tuple[list[list] | None, bool]:
    """Extracts digitized liquidus data from MPDS JSON.

    The curve is the lower envelope of the union of EVERY 'L' shape in the diagram
    (:func:`_stitched_liquidus`), then linearly densified across composition gaps wider
    than ``_FILL_GAP_X`` — but **only within a single covered region**. Densification is
    interpolation between two digitized points on the same liquid field; across the hole
    between two disjoint 'L' shapes there is no curve to interpolate, and filling it would
    fabricate a liquidus. Such a hole is left open: ``is_partial`` is True, the two branch
    endpoints stay adjacent in the returned list, and :func:`liquidus_coverage` measures it.

    Consumers that draw the curve must break the line at any surviving gap wider than
    ``_FILL_GAP_X`` — within a region nothing that wide can survive the fill, so such a gap
    is always an undigitized hole.

    Args:
        mpds_json (dict): MPDS digitized phase equilibrium data for the system.

    Returns:
        tuple[list[list] | None, bool]: Digitized liquidus curve that is properly formatted for fitting purposes.
    """
    stitched, is_partial, covered = _stitched_liquidus(mpds_json)
    if stitched is None:
        return None, is_partial
    mpds_liquidus = list(stitched)

    def fill_liquidus(p1, p2, max_interval):
        """Fills in points between two liquidus points (p1 and p2) based on a maximum interval."""
        num_points = int(np.ceil((p2[0] - p1[0]) / max_interval)) + 1  # Include endpoints
        filled_X = np.linspace(p1[0], p2[0], num_points)
        filled_T = np.linspace(p1[1], p2[1], num_points)
        return [[x, t] for x, t in zip(filled_X, filled_T)][1:-1]

    def within_one_region(x_lo, x_hi):
        """True when both gap endpoints lie inside the SAME digitized liquid region."""
        return any(lo - 1e-9 <= x_lo and x_hi <= hi + 1e-9 for lo, hi in covered)

    # Fill in composition ranges with missing liquidus points, INSIDE a region only
    for i in reversed(range(len(mpds_liquidus) - 1)):
        x_lo, x_hi = mpds_liquidus[i][0], mpds_liquidus[i + 1][0]
        if x_hi - x_lo > _FILL_GAP_X and within_one_region(x_lo, x_hi):
            filler = fill_liquidus(mpds_liquidus[i], mpds_liquidus[i + 1], _FILL_STEP_X)
            for point in reversed(filler):
                mpds_liquidus.insert(i + 1, point)

    # Filter out duplicate values in the liquidus curve; greatly improves runtime efficiency
    for i in reversed(range(len(mpds_liquidus) - 1)):
        if mpds_liquidus[i][0] == 0 or mpds_liquidus[i][1] == 0:
            continue
        if (
            abs(1 - mpds_liquidus[i + 1][0] / mpds_liquidus[i][0]) < 0.0005
            and abs(1 - mpds_liquidus[i + 1][1] / mpds_liquidus[i][1]) < 0.0005
        ):
            del mpds_liquidus[i + 1]

    return mpds_liquidus, is_partial


def liquidus_coverage(mpds_json: dict, gap_tol: float | None = None) -> dict | None:
    """Interior sampling coverage of the digitized liquidus, measured BEFORE gap filling.

    ``extract_digitized_liquidus`` linearly interpolates across in-region composition gaps
    wider than ``_FILL_GAP_X``, so its output cannot distinguish digitized points from
    synthetic fill, and a diagram drawing its liquid as separate terminal wedges still comes
    back spanning the full axis. An endpoint-span check therefore over-reports such data as
    complete; this measures the stitched pre-fill curve instead:

    - ``max_gap``: widest composition interval between consecutive digitized points;
    - ``covered_fraction``: fraction of the stitched span made of inter-point gaps no wider
      than ``gap_tol`` (default ``config.liquidus_gap_tol``) — stretches sampled densely
      enough that the extractor's linear fill is faithful interpolation rather than
      fabrication;
    - ``holes``: the undigitized ``[x_lo, x_hi]`` intervals BETWEEN disjoint liquid regions
      that are wider than ``_FILL_GAP_X``, i.e. the gaps ``extract_digitized_liquidus``
      refuses to bridge and that therefore survive into the extracted curve. No measured
      liquidus exists anywhere inside one, so anything that grades a model against the
      digitized curve must exclude them (``BinaryLiquid.calculate_deviation_metrics``).
      Empty for a contiguous liquid field.

    The two scalar metrics are invariant under component-frame mirroring, so the caller
    never needs to mirror before measuring them — but ``x_min``/``x_max``/``holes`` are
    POSITIONS in the json's own frame and are not. Use :func:`mirror_liquidus_coverage`
    when carrying this dict onto a reversed frame. Returns ``None`` when no liquidus can be
    extracted (the extraction failure itself is already reported by
    ``extract_digitized_liquidus``).
    """
    if gap_tol is None:
        gap_tol = config.liquidus_gap_tol
    stitched, _, regions = _stitched_liquidus(mpds_json, quiet=True)
    if not stitched or len(stitched) < 2:
        return None
    xs = [pt[0] for pt in stitched]
    span = xs[-1] - xs[0]
    gaps = [xs[i + 1] - xs[i] for i in range(len(xs) - 1)]
    max_gap = max(gaps)
    covered = sum(g for g in gaps if g <= gap_tol)
    holes = [
        [lo, hi]
        for lo, hi in ((a[1], b[0]) for a, b in zip(regions, regions[1:]))
        if hi - lo > _FILL_GAP_X
    ]
    return {
        "x_min": xs[0],
        "x_max": xs[-1],
        "span": span,
        "n_points": len(xs),
        "max_gap": max_gap,
        "covered_fraction": covered / span if span > 0 else 0.0,
        "holes": holes,
    }


def mirror_liquidus_coverage(coverage: dict) -> dict:
    """A :func:`liquidus_coverage` dict reflected onto the reversed component frame.

    ``span``, ``n_points``, ``max_gap`` and ``covered_fraction`` are mirror-invariant, but
    ``x_min``/``x_max`` and above all ``holes`` are POSITIONS, and ``x -> 1-x`` moves them. A
    hole carried across a frame flip unmirrored masks the opposite end of the diagram.
    An involution, like :func:`mirror_liquidus`.
    """
    mirrored = dict(coverage)
    if "x_min" in coverage and "x_max" in coverage:
        mirrored["x_min"] = 1 - coverage["x_max"]
        mirrored["x_max"] = 1 - coverage["x_min"]
    mirrored["holes"] = [[1 - hi, 1 - lo] for lo, hi in reversed(coverage.get("holes") or [])]
    return mirrored


@config._removed_keywords({"input": "system"})
def load_mpds_data(system, pd_ind=None) -> tuple[dict, tuple[list[list] | None, bool]]:
    """Retrieves MPDS data for a binary system.

    Args:
        system (str or list): System specification (e.g., 'A-B' or ['A', 'B']).
        pd_ind (int | None): Index of the cached MPDS phase diagram to load.

    Returns:
        tuple[dict, tuple[list[list] | None, bool]]: The system MPDS JSON and the digitized
        liquidus curve formatted for fitting. Unary reference data is not returned; take it from
        ``gliquid.phase.UNARY.component_data``. Note that the
        MPDS json in the specified cache directory must follow the alphabetized, hyphenated
        naming convention (e.g. 'A-B.json')

    Raises:
        CacheModeError: the configured store is read-only and holds no record for the
            system; raised before any MPDS request.
        mpds_client.APIError: an MPDS request failed with any code other than 204 (no
            hits); nothing is cached.
    """
    components, _, _ = validate_and_format_system(system)
    if len(components) != 2:
        raise ValueError(
            f"MPDS binary phase-diagram retrieval needs exactly 2 components, got {components}."
        )
    # On-disk keys use the alphabetical system name, so the json is in the alphabetical frame.
    cache_name = api._canonical_sys_name(components)
    # The backend resolves pd_ind: ``variant == ""`` is the indexless ``<sys>.json`` naming;
    # ``"0"``, ``"1"``, ... are the indexed diagrams.
    backend = cache.resolve_backend(None)
    available = set(backend.variants(cache_name, cache.KIND_MPDS))
    has_indexless = "" in available
    has_pd0 = "0" in available
    sys_dir = cache.store_label(backend, cache_name)

    if pd_ind is None:
        variant = "" if has_indexless else "0"
        if has_indexless and has_pd0:
            # Both namings present: the indexless file shadows PD_0; warns.
            logger.warning(
                f"{cache_name}.json shadows {cache_name}_MPDS_PD_0.json in {sys_dir}; "
                f"pd_ind=None is resolving the indexless file. Pass pd_ind explicitly to "
                f"pin which diagram is loaded."
            )
    elif isinstance(pd_ind, int):
        variant = str(pd_ind)
        if variant not in available:
            if has_pd0:
                raise ValueError(f"No matching json with pd_ind={pd_ind} found in cache!")
            if has_indexless:
                # An indexless-only store holds one diagram, which pd_ind=0 names.
                if pd_ind != 0:
                    raise ValueError(
                        f"No matching json with pd_ind={pd_ind} found in cache! "
                        f"{sys_dir} holds only the indexless {cache_name}.json, which is "
                        f"reachable as pd_ind=0 or pd_ind=None."
                    )
                logger.info(
                    f"No {cache_name}_MPDS_PD_0.json in {sys_dir}; pd_ind=0 resolving the "
                    f"indexless {cache_name}.json, the store's only diagram."
                )
                variant = ""
    else:
        raise ValueError("Input for pd_ind must be an integer or 'None'!")

    sys_key = cache.CacheKey(cache_name, cache.KIND_MPDS, variant)
    if backend.exists(sys_key):  # Load from cache
        mpds_json = backend.read_json(sys_key)
        if mpds_json.get("reference", None) is not None:
            logger.info(
                "Reading MPDS json from entry at " + mpds_json["reference"]["entry"] + "..."
            )
    else:  # Try API call
        logger.info("No cached binary phase data found!")
        mpds_json = {"reference": None}
        # Without a key nothing is fetched and the system proceeds with no MPDS reference.
        if not api.get_api_key(api.MPDS_KEY_VAR):
            logger.warning(
                "MPDS_API_KEY not found in environment variables. Proceeding without binary phase data."
            )
            return mpds_json, (None, False)
        config.require_online(f"Fetching the MPDS phase diagram for '{cache_name}'")
        api._require_writable_store(
            backend,
            sys_key if pd_ind is not None else cache.CacheKey(cache_name, cache.KIND_MPDS, ""),
        )
        client = api.get_mpds_client()
        fields = {
            "C": [
                "chemical_elements",
                "entry",
                "comp_range",
                "temp",
                "labels",
                "shapes",
                "reference",
            ]
        }
        valid_jsons = []
        try:
            diagrams = [
                d
                for d in client.get_data(
                    search={"elements": cache_name, "classes": "binary"}, fields=fields
                )
                if d
            ]
            for d in diagrams:
                dia_json = dict(zip(fields["C"], d))
                if dia_json["comp_range"][1] - dia_json["comp_range"][0] > 10:
                    if mpds_json["reference"] is None:
                        mpds_json = dia_json
                    elif (
                        dia_json["comp_range"][1] - dia_json["comp_range"][0]
                        > mpds_json["comp_range"][1] - mpds_json["comp_range"][0]
                    ):
                        mpds_json = dia_json
                if dia_json["comp_range"] != [0, 100]:
                    continue
                if extract_digitized_liquidus(dia_json)[0]:
                    valid_jsons.append(dia_json)
        except api.mpds_api_error() as exc:
            code = getattr(exc, "code", None)
            if code != 204:
                logger.error(
                    f"MPDS request for '{cache_name}' failed with code {code}: "
                    f"{getattr(exc, 'msg', str(exc))}. Nothing was cached."
                )
                raise
            logger.info("Got 0 hits")

        if not valid_jsons:
            valid_jsons = [mpds_json]

        if pd_ind is None:
            mpds_json = valid_jsons[0]
            write_key = cache.CacheKey(cache_name, cache.KIND_MPDS, "")
            backend.write_json(write_key, mpds_json)
            sys_file = backend.locate(write_key)
            if mpds_json.get("reference", None) is None:
                logger.info("No valid phase diagrams found, caching default json")
            else:
                logger.info(
                    f"Caching binary phase data from entry at {mpds_json['reference']['entry']} as {sys_file}..."
                )
        else:
            for ind, dia_json in enumerate(valid_jsons):
                write_key = cache.CacheKey(cache_name, cache.KIND_MPDS, str(ind))
                backend.write_json(write_key, dia_json)
                sys_file = backend.locate(write_key)
                if dia_json.get("reference", None) is None:
                    logger.info("No valid phase diagrams found, caching default json")
                    break
                logger.info(
                    f"Caching binary phase data from entry at {dia_json['reference']['entry']} as {sys_file}..."
                )
            if isinstance(pd_ind, int):
                if pd_ind < len(valid_jsons):
                    mpds_json = valid_jsons[pd_ind]
                else:
                    logger.warning(
                        f"pd_ind={pd_ind} exceeds the number of valid jsons downloaded from API! Returning the first json"
                    )
                    mpds_json = valid_jsons[0]
            else:
                raise ValueError("Input for pd_ind must be an integer or 'None'!")

    return mpds_json, extract_digitized_liquidus(mpds_json)


def mpds_frame_matches(mpds_json: dict, components) -> bool:
    """True when the json's own component frame equals ``components`` (order-sensitive).

    MPDS jsons carry ``chemical_elements`` in the frame their svgpaths are digitized in
    (alphabetical under the canonical cache convention). A json with no frame info is
    trusted to match the caller.
    """
    els = (mpds_json or {}).get("chemical_elements")
    return not els or list(els) == list(components)


def mirror_liquidus(liquidus: list[list]) -> list[list]:
    """A digitized liquidus reflected onto the reversed component frame (x -> 1-x)."""
    return [[1 - x, t] for x, t in reversed(liquidus)]


def mirror_mpds_phases(phases: list[dict]) -> list[dict]:
    """``identify_mpds_phases`` output reflected onto the reversed component frame.

    Mirrors every composition coordinate (comp, tbounds x, cbounds x), keeps temperature
    ordering, restores comp-ascending cbounds/list order. An involution.

    Any other key rides along untouched via the ``dict(p)`` copy -- notably the optional
    ``'structure'`` block, whose spacegroup/Pearson content is frame-independent.
    """
    mirrored = []
    for p in phases:
        q = dict(p)
        q["comp"] = 1 - p["comp"]
        q["tbounds"] = [[1 - x, t] for x, t in p["tbounds"]]
        if "cbounds" in p:
            q["cbounds"] = [[1 - x, t] for x, t in reversed(p["cbounds"])]
        mirrored.append(q)
    return sorted(mirrored, key=lambda d: d["comp"])


def is_component_phase_label(name: str, elements) -> bool:
    """True when ``name`` is MPDS's parenthesised spelling of a COMPONENT solution field.

    Two spellings match: a terminal field on one component, ``'(Ti)'`` / ``'(Ti) rt'``, and
    a field spanning both, ``'(Ti, Cr)'`` / ``'(Ti, Cr) ht'``. Every symbol inside the
    parentheses must be one of ``elements``; with no ``elements`` nothing matches.

    Only the LEADING parenthesised group is read, so a polymorph suffix is irrelevant. A
    caller that passes ``shape['label'].split()[0]`` supplies ``'(Ti)'`` for a terminal
    label and ``'(Ti,'`` for a two-component one; the latter has no closing parenthesis and
    does not match.

    Name-based, not composition-based: line compounds at extreme stoichiometry lie as near
    the pure ends as component phases do, and a component phase can be digitized well away
    from its end.

    :func:`gliquid.solid_coverage.assess_solid_coverage` reads it the other way round: a
    digitized FIELD whose label this rejects is an ordered intermetallic with a homogeneity
    range, and is scored as a compound rather than as a solid solution.
    """
    symbols = set(elements or ())
    if not symbols:
        return False
    match = _COMPONENT_LABEL_RE.match(_HTML_TAG_RE.sub("", name or "").strip())
    if match is None:
        return False
    named = [token.strip() for token in match.group(1).split(",")]
    return all(token in symbols for token in named)


def is_multiphase_field_label(label: str) -> bool:
    """True when an MPDS label names MORE than one phase -- a two-phase FIELD annotation.

    MPDS spells these as the constituent phases joined by ``'+'``, with a component
    phase's parentheses dropped: ``'HgZr3 + G'``, ``'Mn2B Mn + rt'`` (the figure prints
    ``'Mn2B + (Mn) rt'``), ``'CuSe2 rt + L'``, ``'Au3Zn Zn + rt'``. Such a shape is the
    field BETWEEN two phases, not a phase -- even though MPDS routinely tags it
    ``nphases: 1``, which is what lets it reach :func:`identify_mpds_phases` at all.

    A boundary line is named from ``shape['label'].split()[0]``, which would promote the
    first constituent into a phantom compound at the FIELD's composition.

    A ``'+'`` anywhere in the label is the signal: no compound name or polymorph suffix
    contains one.
    """
    return "+" in (label or "")


def is_fluid_phase_label(label: str) -> bool:
    """True when an MPDS label names the liquid or gas field itself -- ``'L'`` or ``'G'``.

    MPDS sometimes emits a liquid or gas field as a ``kind: 'compound'`` STROKE flagged
    ``is_solid: true``, rather than as an area: a terminal liquid sliver too narrow to
    fill gets drawn as a 2- or 3-point line, and a frame edge can capture the nearby
    printed ``'L'`` text. Without this check :func:`identify_mpds_phases` would name the shape
    from ``split_phase_label(...)[0]`` and emit a line compound literally called ``'L'``.

    :func:`liquid_shape_paths` already refuses these shapes on the LIQUIDUS side, via its
    ``kind in (None, 'phase')`` filter. That filter cannot be reused here: every genuine
    line compound is ``kind: 'compound'``, so applying it at the phase-list gate would
    delete all of them. The label is the discriminator, not the kind.

    Matched on the BASE name only, so a polymorph suffix cannot slip a fluid field
    through. No element symbol is ``'L'`` or ``'G'``, so the check cannot collide with an
    element's phase. Multi-phase field labels such as ``'L + G'`` are caught earlier by
    :func:`is_multiphase_field_label`.
    """
    return split_phase_label(label)[0] in ("L", "G")


def split_phase_label(label: str) -> tuple[str, str]:
    """MPDS phase label -> ``(base compound name, polymorph suffix)``.

    ``'LaC<sub>2</sub> rt'`` -> ``('LaC2', 'rt')``; ``'Ce2C3'`` -> ``('Ce2C3', '')``.
    Subscript markup appears in the json's ``labels`` block but not in ``shape['label']``,
    so it is stripped here to make the two blocks joinable on the base name.
    """
    text = _HTML_TAG_RE.sub("", label or "").strip()
    parts = text.split()
    return (parts[0] if parts else ""), " ".join(parts[1:]).lower()


def polymorph_rank(suffix: str) -> int:
    """Order a polymorph suffix by the temperature of the form it names, lowest first.

    ``lt < rt == unsuffixed < ht < ht1 < ht2 ... < hp``. A suffix with no recognizable
    polymorph token ranks WITH the unsuffixed form, so an ordinary compound is never
    mistaken for a high-temperature polymorph ('HfFe2 hex1' is a structure note, not a
    thermal form); a compound token embedded in a longer suffix still counts
    ('Pt3Ga Ga + rt' -> rt).

    Ranking is by label, not by digitized temperature: a diagram can draw two thermal forms
    over overlapping temperature ranges, whose digitized bounds then order them wrongly.
    """
    for token in (suffix or "").split():
        match = _POLYMORPH_RE.match(token)
        if match:
            kind, index = match.group(1).lower(), match.group(2)
            return _POLYMORPH_BASE_RANK[kind] + (int(index) if index else 0)
    return _POLYMORPH_BASE_RANK[""]


ROOM_TEMPERATURE = 298.15
"""K. A phase whose field reaches this is stable at room temperature by inspection."""


def low_temp_threshold(mpds_json: dict, fraction: float | None = None) -> float:
    """Temperature (K) below which a digitized phase counts as stable at low temperature.

    The bottom of the diagram's declared temperature axis plus ``fraction`` of its span.
    Being a fraction of the DECLARED axis, this moves with a wrong axis;
    :func:`is_low_temperature_phase` pairs it with :func:`reaches_room_temperature`, which
    does not.

    Args:
        mpds_json: the source json; only ``temp`` (axis bottom and top, in C) is read.
        fraction: fraction of the axis span. Defaults to
            ``config.coverage_low_temp_axis_fraction``.

    Returns:
        float: the cutoff in K.
    """
    fraction = config.coverage_default("low_temp_axis_fraction", fraction)
    lo, hi = mpds_json["temp"][0], mpds_json["temp"][1]
    return (lo + 273.15) + (hi - lo) * float(fraction)


def reaches_room_temperature(phase: dict) -> bool:
    """Does this phase's digitized field extend down to room temperature or below?

    Independent of the diagram's declared axis, and so independent of whether that axis is
    right. A compound drawn as stable at room temperature IS a low-temperature phase however
    the frame was recorded, and a hole in the digitization below its field cannot change that.

    Args:
        phase: a digitized phase record carrying ``tbounds``.

    Returns:
        bool: True when the field's lower bound is at or below
        :data:`ROOM_TEMPERATURE`; False when it is higher or ``tbounds`` is unusable.
    """
    bounds = phase.get("tbounds") or []
    if not bounds or not bounds[0]:
        return False
    try:
        return float(bounds[0][1]) <= ROOM_TEMPERATURE
    except (TypeError, ValueError, IndexError):
        return False


# Composition slack for matching a phase to an unpaired-form key after a frame mirror.
_UNPAIRED_KEY_TOL = 1e-9


def is_unpaired_ht_form(phase: dict, unpaired_ht) -> bool:
    """Whether ``phase`` is one of the forms :func:`mark_unpaired_ht_forms` reported.

    Matched on the suffix-stripped ``name`` and on ``comp``, so a raw shape and a collapsed
    cluster that inherited the form's identity both test true.

    Args:
        phase: a phase dict carrying ``name`` and ``comp``.
        unpaired_ht: ``(base label, composition)`` keys, in ``phase``'s component frame.

    Returns:
        bool: False for an empty set or a phase with no composition.
    """
    if not unpaired_ht:
        return False
    comp = phase.get("comp")
    if comp is None:
        return False
    base = split_phase_label(phase.get("name") or "")[0]
    return any(
        key == base and abs(float(x) - float(comp)) <= _UNPAIRED_KEY_TOL for key, x in unpaired_ht
    )


def is_low_temperature_phase(
    phase: dict, threshold: float, *, unpaired_ht: frozenset = frozenset()
) -> bool:
    """Whether a digitized phase is a low-temperature phase.

    True iff ``phase['type']`` is ``'lc'``, ``'ss'`` or ``'comp'``, the phase is not an
    unpaired high-temperature form, and its field bottom is below ``threshold`` or reaches
    room temperature (:func:`reaches_room_temperature`). The one rule every consumer of
    "low-temperature phase" applies. Run :func:`collapse_polymorphs` first, so a compound's
    thermal forms are judged as one phase by the bottom of its coolest form.

    Args:
        phase: a phase dict from :func:`identify_mpds_phases` or :func:`collapse_polymorphs`.
        threshold: low-temperature cutoff in K, from :func:`low_temp_threshold`.
        unpaired_ht: ``(base label, composition)`` keys from :func:`mark_unpaired_ht_forms`,
            in the same component frame as ``phase``. A listed phase is never
            low-temperature, whatever its field bottom.

    Returns:
        bool
    """
    if phase.get("type") not in ("lc", "ss", "comp"):
        return False
    if is_unpaired_ht_form(phase, unpaired_ht):
        return False
    try:
        below = phase["tbounds"][0][1] < threshold
    except (KeyError, TypeError, IndexError):
        below = False
    return below or reaches_room_temperature(phase)


def compound_key(label: str, elements) -> str:
    """The compound a phase label names, spelled one way for every thermal form.

    ``plotting.style.standard_formula`` of the suffix-stripped label in ``elements`` order;
    the raw base label when that will not parse, names an element outside ``elements``, or
    is a component label.
    """
    base = split_phase_label(label)[0]
    if not base or base.startswith("("):
        return base
    try:
        return standard_formula(base, list(elements or []))
    except Exception:  # noqa: BLE001 - any unparseable label keys as its own text
        return base


def _label_forms(
    mpds_json: dict | None, elements
) -> dict[str, list[tuple[int, float, float | None]]]:
    """Thermal forms the json's ``labels`` block names, keyed by :func:`compound_key`.

    Each entry is ``(polymorph rank, label temperature in K, label composition)``; the
    composition is None when the entry carries no usable position. Field annotations
    (:func:`is_multiphase_field_label`) and malformed entries are skipped.
    """
    forms: dict[str, list[tuple[int, float, float | None]]] = {}
    for entry in (mpds_json or {}).get("labels") or []:
        try:
            if is_multiphase_field_label(entry[0]):
                continue
            suffix = split_phase_label(entry[0])[1]
            temp = float(entry[1][1]) + 273.15
        except (TypeError, ValueError, IndexError, KeyError):
            continue
        try:
            comp = float(entry[1][0]) / 100
        except (TypeError, ValueError, IndexError):
            comp = None
        forms.setdefault(compound_key(entry[0], elements), []).append(
            (polymorph_rank(suffix), temp, comp)
        )
    return forms


def mark_unpaired_ht_forms(
    phases: list[dict],
    elements,
    *,
    pair_tol: float | None = None,
    mpds_json: dict | None = None,
) -> frozenset[tuple[str, float]]:
    """Keys of the high-temperature forms that have no cooler sibling to pair with.

    A shape whose polymorph suffix ranks above ``rt`` (:func:`polymorph_rank`) is unpaired
    when no shape and no ``labels``-block entry of the same compound with a rank at or
    below ``rt`` lies within ``pair_tol`` of its composition. Compound identity is the
    normalised formula :func:`collapse_polymorphs` pairs by. Component phases (``'('``
    labels) are never reported. Call it on the raw shapes, before collapse, so every form
    is visible to the sibling test.

    Args:
        phases: ``identify_mpds_phases(..., with_polymorph=True)`` output. Without the
            ``'polymorph'`` key a line compound's suffix is unknown and it ranks as
            unsuffixed.
        elements: the component symbols, in the frame ``phases`` are in.
        pair_tol: composition window. Defaults to ``config.coverage_polymorph_pair_tol``.
        mpds_json: the source json, read for its ``labels`` block. None means no labels.

    Returns:
        frozenset: ``(base label, composition)`` of every unpaired form -- the key
        :func:`is_low_temperature_phase` tests against, which a collapsed cluster of such
        forms inherits from its lowest-ranked member.
    """
    tol = config.coverage_polymorph_pair_tol if pair_tol is None else float(pair_tol)
    forms = _label_forms(mpds_json, elements)
    rt_rank = _POLYMORPH_BASE_RANK["rt"]
    keyed = []
    for phase in phases:
        name = phase.get("name") or ""
        base, name_suffix = split_phase_label(name)
        if base.startswith("(") or phase.get("comp") is None:
            continue
        suffix = phase["polymorph"] if "polymorph" in phase else name_suffix
        keyed.append(
            (compound_key(name, elements), polymorph_rank(suffix), float(phase["comp"]), base)
        )
    unpaired = set()
    for key, rank, comp, base in keyed:
        if rank <= rt_rank:
            continue
        siblings = [x for k, r, x, _b in keyed if k == key and r <= rt_rank]
        siblings += [x for r, _t, x in forms.get(key, []) if r <= rt_rank and x is not None]
        if not any(abs(x - comp) <= tol for x in siblings):
            unpaired.add((base, comp))
    return frozenset(unpaired)


def _merge_polymorph_cluster(
    cluster: list[dict],
    label_forms: list[tuple[int, float, float | None]],
    temp_threshold: float | None,
) -> dict:
    """Merge one compound's polymorph shapes into a single phase dict. See
    :func:`collapse_polymorphs` for the policy this implements."""
    ranks = [polymorph_rank(phase.get("polymorph", "")) for phase in cluster]
    lowest_rank = min(ranks)
    low = cluster[ranks.index(lowest_rank)]
    base = split_phase_label(low["name"])[0]
    hottest = max(cluster, key=lambda phase: phase["tbounds"][1][1])

    merged = dict(low)
    merged["tbounds"] = [low["tbounds"][0], hottest["tbounds"][1]]
    distinct = len(set(ranks)) > 1

    # Recover a low-temperature form named in 'labels' but not digitized as its own shape.
    if temp_threshold is not None and not is_low_temperature_phase(low, temp_threshold):
        x_lo = low["tbounds"][0][0]
        cooler = sorted(
            (rank, temp)
            for rank, temp, _comp in label_forms
            if rank <= lowest_rank
            and is_low_temperature_phase(
                {"type": low["type"], "tbounds": [[x_lo, temp]]}, temp_threshold
            )
        )
        if cooler:
            rank_lo, temp_lo = cooler[0]
            if rank_lo < lowest_rank:
                distinct = True
            merged["tbounds"] = [[low["tbounds"][0][0], temp_lo], merged["tbounds"][1]]

    if len(cluster) > 1 or distinct:
        merged["name"] = base
    return merged


def _collapse_polymorph_clusters(
    phases: list[dict],
    mpds_json: dict | None = None,
    *,
    temp_threshold: float | None = None,
    comp_tol: float | None = None,
    elements=None,
) -> list[tuple[dict, list[dict]]]:
    """One ``(merged phase, member shapes)`` pair per compound-composition cluster.

    The pairing and merge policy is :func:`collapse_polymorphs`'s, which returns the merged
    phases alone. Members are copies of the input dicts and keep their ``'polymorph'`` key;
    the merged phase does not carry one. A component phase is its own single-member cluster.
    Pairs are sorted by the merged phase's composition.
    """
    if not phases:
        return []
    if temp_threshold is None and mpds_json:
        temp_threshold = low_temp_threshold(mpds_json)
    comp_tol = config.coverage_default("polymorph_pair_tol", comp_tol)
    if elements is None:
        elements = (mpds_json or {}).get("chemical_elements") or []

    # Component phases ('(Ce) ht1' / '(Ce) ht2') pass through untouched.
    passthrough, candidates = [], []
    for phase in phases:
        target = passthrough if split_phase_label(phase["name"])[0].startswith("(") else candidates
        target.append(dict(phase))

    label_forms = _label_forms(mpds_json, elements)

    groups: dict[str, list[dict]] = {}
    for phase in candidates:
        groups.setdefault(compound_key(phase["name"], elements), []).append(phase)

    collapsed: list[tuple[dict, list[dict]]] = []
    for key, members in groups.items():
        members = sorted(members, key=lambda phase: phase["comp"])
        clusters = [[members[0]]]
        for phase in members[1:]:
            if abs(phase["comp"] - clusters[-1][0]["comp"]) <= comp_tol:
                clusters[-1].append(phase)
            else:
                clusters.append([phase])
        for cluster in clusters:
            merged = _merge_polymorph_cluster(cluster, label_forms.get(key, []), temp_threshold)
            collapsed.append((merged, cluster))

    # drop 'polymorph': the characterization pins compare phase dicts by exact key set
    pairs = collapsed + [(phase, [dict(phase)]) for phase in passthrough]
    pairs.sort(key=lambda pair: pair[0]["comp"])
    for merged, _members in pairs:
        merged.pop("polymorph", None)
    return pairs


def collapse_polymorphs(
    phases: list[dict],
    mpds_json: dict | None = None,
    *,
    temp_threshold: float | None = None,
    comp_tol: float | None = None,
    elements=None,
) -> list[dict]:
    """Collapse a compound's digitized polymorphs into one phase per compound.

    One entry survives per (compound, composition) cluster. It takes its identity and lower
    temperature bound from the lowest-temperature polymorph and its upper bound from the
    compound's highest melting/decomposition temperature, and is named after its lowest-ranked
    form. Two shapes are forms of one compound when their suffix-stripped labels normalise to the
    same formula (``plotting.style.standard_formula`` in ``elements`` order; a label that will
    not parse keys as its own text) and their compositions lie within ``comp_tol``. Entries
    sharing a name at different compositions stay separate.

    Args:
        phases (list): Phase dicts from ``identify_mpds_phases(..., with_polymorph=True)``.
            Without the ``'polymorph'`` key every entry ranks as unsuffixed.
        mpds_json (dict): Source json; only the ``labels`` block is read, to recover a
            low-temperature form named there but never digitized.
        temp_threshold (float): Low-temperature cutoff in K, required for that recovery.
            Defaults to :func:`low_temp_threshold` of ``mpds_json``.
        comp_tol (float): Composition tolerance for treating two shapes of one compound as
            the same cluster. Defaults to ``config.coverage_polymorph_pair_tol``.
        elements: Component symbols fixing the normalised formula's element order. Defaults
            to the json's own ``chemical_elements``.

    Returns:
        list: One phase dict per compound-composition cluster, sorted by composition.
    """
    return [
        merged
        for merged, _members in _collapse_polymorph_clusters(
            phases,
            mpds_json,
            temp_threshold=temp_threshold,
            comp_tol=comp_tol,
            elements=elements,
        )
    ]


def identify_mpds_phases(
    mpds_json: dict,
    verbose=False,
    *,
    with_structure: bool = False,
    with_polymorph: bool = False,
    elements=None,
) -> list[dict]:
    """Identifies MPDS phases from JSON data.

    Emits three phase ``'type'`` values:

    * ``'ss'`` -- a digitized single-phase field (an area shape), carrying ``cbounds``.
    * ``'comp'`` -- a pure component's own phase, labelled ``'(X)'`` and drawn as a boundary
      line (see :func:`is_component_phase_label`); its free energy is the element reference.
    * ``'lc'`` -- a line compound.

    Args:
        mpds_json (dict): MPDS digitized phase equilibrium data for the system.
        verbose (bool): If True, outputs additional debugging information.
        with_structure (bool): If True, add a ``'structure'`` key to every phase dict holding
            the parsed ``shape['phase']`` identifier (see :func:`parse_shape_structure`), as
            :func:`gliquid.solid_coverage.assess_solid_coverage` needs.
        with_polymorph (bool): If True, add a ``'polymorph'`` key holding the label's
            polymorph suffix (``'rt'``, ``'ht1'``, ``''`` when unsuffixed), as
            :func:`collapse_polymorphs` needs. A line compound's ``'name'`` has the suffix
            stripped.
        elements: Component symbols used to recognize ``'(X)'`` component labels, in any order.
            Defaults to the json's own ``chemical_elements``; without either, component labels
            read as ``'lc'``.

    Returns:
        list: A list of dictionaries containing information on equilibrium phase composition and temperature boundaries

    Raises:
        config.CacheModeError: on a LEAN record; see :func:`require_full_record`.
    """
    require_full_record(mpds_json, "identify_mpds_phases")
    if mpds_json.get("reference") is None:
        if verbose:
            logger.warning("System JSON does not contain any data!")
        return []

    if elements is None:
        elements = mpds_json.get("chemical_elements") or []

    phases = []
    for shape in mpds_json.get("shapes", []):
        if shape.get("nphases") == 1 and shape.get("is_solid") and "label" in shape:
            # A two-phase field tagged nphases=1 (see is_multiphase_field_label).
            if is_multiphase_field_label(shape["label"]):
                if verbose:
                    logger.info(
                        f"Skipping {shape['label']!r}: label names more than one "
                        f"phase, so the shape is a field, not a phase"
                    )
                continue

            # A liquid or gas field tagged is_solid=true (see is_fluid_phase_label).
            if is_fluid_phase_label(shape["label"]):
                if verbose:
                    logger.info(
                        f"Skipping {shape['label']!r}: label names the liquid or gas "
                        f"field, so the shape is not a solid phase"
                    )
                continue

            data = shape_to_list(shape.get("svgpath", ""))
            if not data:
                if verbose:
                    logger.info(f"No point data found for phase {shape['label']} in JSON!")
                continue

            data.sort(key=lambda x: x[1])  # Sort by temperature
            tbounds = [data[0], data[-1]]
            extra = {"structure": parse_shape_structure(shape)} if with_structure else {}
            if with_polymorph:
                extra["polymorph"] = split_phase_label(shape["label"])[1]

            if shape.get("kind") == "phase":
                data.sort(key=lambda x: x[0])  # Sort by composition
                cbounds = [data[0], data[-1]]
                if cbounds[-1][0] < 0.03 or cbounds[0][0] > 0.97:
                    continue
                phases.append(
                    {
                        "type": "ss",
                        "name": shape["label"],
                        "comp": tbounds[1][0],
                        "cbounds": cbounds,
                        "tbounds": tbounds,
                        **extra,
                    }
                )
            else:  # A boundary line: a line compound, or a component's own phase
                name = shape["label"].split()[0]
                p_type = "comp" if is_component_phase_label(name, elements) else "lc"
                phases.append(
                    {
                        "type": p_type,
                        "name": name,
                        "comp": tbounds[1][0],
                        "tbounds": tbounds,
                        **extra,
                    }
                )

    if not phases and verbose:
        logger.info("No phase data found in JSON!")

    return sorted(phases, key=lambda x: x["comp"])


def merge_ranges(ranges) -> list[list[float]]:
    """Merge overlapping/adjacent ``[lo, hi]`` composition ranges into a minimal set."""
    ordered = sorted([sorted(r) for r in ranges])
    if not ordered:
        return []
    merged = [list(ordered[0])]
    for lo, hi in ordered[1:]:
        if lo <= merged[-1][1] + 1e-9:
            merged[-1][1] = max(merged[-1][1], hi)
        else:
            merged.append([lo, hi])
    return merged


def flank_bound(comp, side, tiers, *, tol, default):
    """The nearest candidate composition lying strictly more than ``tol`` to one side of ``comp``.

    Args:
        comp: the composition to bound.
        side: ``"left"`` for candidates ``c < comp - tol``, ``"right"`` for ``c > comp + tol``.
        tiers: ``(candidates, rule)`` pairs, tried in order. The first tier holding any
            candidate on ``side`` decides.
        tol: composition slack, keyword-only and required.
        default: the ``(bound, rule)`` pair returned when no tier holds a candidate on ``side``.

    Returns:
        tuple: ``(bound, rule)``. ``bound`` is the largest (left) or smallest (right) qualifying
        candidate of the deciding tier, as the element itself with no conversion, the first of
        equal extremes in iteration order; ``rule`` is that tier's rule. Otherwise ``default``,
        the same object.

    Raises:
        ValueError: if ``side`` is neither ``"left"`` nor ``"right"``.
    """
    if side == "left":
        for candidates, rule in tiers:
            below = [c for c in candidates if c < comp - tol]
            if below:
                return max(below), rule
    elif side == "right":
        for candidates, rule in tiers:
            above = [c for c in candidates if c > comp + tol]
            if above:
                return min(above), rule
    else:
        raise ValueError(f"side must be 'left' or 'right', not {side!r}")
    return default


def find_local_maxima(points):
    """Local maxima of a sequence of ``(x, y)`` points.

    A rise followed by a fall records the middle point of the flat top it reached. The
    first and last points never qualify, so a monotone sequence yields nothing. Returns the
    point objects themselves, not copies.
    """

    def is_gt_prev(index):
        return index > 0 and points[index][1] > points[index - 1][1]

    local_maxima = []
    current_section = []

    for i in range(len(points)):
        if is_gt_prev(i):
            current_section = [points[i]]
        elif current_section and current_section[-1][1] == points[i][1]:
            current_section.append(points[i])
        elif current_section:
            local_maxima.append(current_section[len(current_section) // 2])
            current_section = []

    return local_maxima


def _endpoint_maxima(points):
    """One-sided maxima at the ends of a digitized liquidus.

    :func:`find_local_maxima` needs a rise AND a fall, so it never returns the first or last
    point. Where the digitized curve simply BEGINS (or ends) at a compound's melting point,
    that maximum is real and is the compound's congruent melting point -- the data starts
    there, the diagram does not.

    An endpoint within ``_AXIS_X_TOL`` of a pure end is excluded: the maximum there is that
    element's melting point, not a compound's.

    Args:
        points: ``(x, T)`` pairs in composition order.

    Returns:
        list: the qualifying endpoint objects themselves, not copies.
    """
    if len(points) < 2:
        return []
    out = []
    for index, neighbour in ((0, 1), (-1, -2)):
        point = points[index]
        x = float(point[0])
        if x <= _AXIS_X_TOL or x >= 1.0 - _AXIS_X_TOL:
            continue
        if point[1] > points[neighbour][1]:
            out.append(point)
    return out


def _is_terminal_phase(phase) -> bool:
    """Whether a phase is a pure component or a terminal solid solution on one.

    Three independent markers, any of which is sufficient: MPDS brackets a terminal
    solid-solution label (``(Th)``, ``(Th) rt``), ``'comp'`` types a component's own phase, and
    a stoichiometry within ``_AXIS_X_TOL`` of either pure end is a component whatever it is
    called.
    """
    name = phase.get("name") or ""
    if "(" in name or phase.get("type") == "comp":
        return True
    comp = phase.get("comp")
    if comp is None:
        return False
    return float(comp) <= _AXIS_X_TOL or float(comp) >= 1.0 - _AXIS_X_TOL


def find_local_minima(points):
    """Local minima of a sequence of ``(x, y)`` points.

    :func:`find_local_maxima` run over negated ordinates, mapped back to the original
    point objects through the index carried alongside.
    """
    negated = [(i, -point[1]) for i, point in enumerate(points)]
    return [points[i] for i, _ in find_local_maxima(negated)]


# Composition slack below which a supported stretch is a clipping artefact, not a gap.
SEGMENT_TOL = 1e-9

# Invariant types that record the melting of one named phase: 'cmp' (congruent melting) and
# 'per' (peritectic). 'eut' and 'mig' name the adjacent phases.
_MELTING_INVARIANT_TYPES = ("cmp", "per")

# Composition slack for joining an invariant's recorded phase composition to a phase dict's own.
_INVARIANT_MATCH_TOL = 0.02

# Distance within which an interior liquidus maximum is a phase's own congruent dome.
_CONGRUENT_MATCH_TOL = 0.02


def controlling_invariant(
    phase: dict, invariants, *, allow_eutectic: bool = False, comp_tol: float | None = None
) -> dict | None:
    """The invariant that anchors ``phase``, or ``None``.

    Matches on the phase name carried in an invariant's ``'phases'`` and on the stoichiometry
    carried in its ``'phase_comps'``, so a compound MPDS labels twice at different
    compositions binds to its own record. Where several
    records name the compound, the one whose recorded composition sits nearest wins.

    Names are compared with any polymorph suffix stripped on BOTH sides: a shape labelled
    with its thermal form binds to a record naming the bare compound, and vice versa.

    The composition check separates a phase's own melting record from a distant one that
    merely names it.

    Args:
        phase (dict): One ``identify_mpds_phases`` entry; ``'name'`` and ``'comp'`` are read.
        invariants: ``identify_invariant_points`` output, in the same composition frame as
            ``phase``. May be empty or None.
        allow_eutectic: Fall back to a eutectic the phase flanks when no melting record
            matches. Imputation needs an anchor wherever the phase borders the liquid;
            field bounding needs the record that melts it, and takes the default.
        comp_tol: How near a recorded stoichiometry must sit to ``phase['comp']``. Defaults
            to the module tolerance.

    Returns:
        dict | None: The matched ``'cmp'`` / ``'per'`` invariant, or a flanking ``'eut'``
        when ``allow_eutectic`` and nothing melts it.
    """
    name, comp = phase.get("name"), phase.get("comp")
    if name is None or comp is None:
        return None
    tol = _INVARIANT_MATCH_TOL if comp_tol is None else float(comp_tol)
    comp = float(comp)
    base = split_phase_label(name)[0]

    def _named(inv):
        return base in {split_phase_label(n)[0] for n in (inv.get("phases") or []) if n}

    def _recorded_delta(inv):
        """Nearest recorded stoichiometry to ``comp``, or None when none is within tol."""
        recorded = [c for c in (inv.get("phase_comps") or []) if c is not None]
        if not recorded:
            return None
        delta = min(abs(float(c) - comp) for c in recorded)
        return delta if delta <= tol else None

    matches = []
    for inv in invariants or []:
        if not isinstance(inv, dict) or inv.get("type") not in _MELTING_INVARIANT_TYPES:
            continue
        if inv.get("comp") is None or not _named(inv):
            continue
        delta = _recorded_delta(inv)
        if delta is not None:
            matches.append((delta, inv))
    if matches:
        return min(matches, key=lambda match: match[0])[1]
    if not allow_eutectic:
        return None
    # A eutectic is matched on stoichiometry alone and resolved on its liquid composition.
    flanking = [
        inv
        for inv in invariants or []
        if isinstance(inv, dict)
        and inv.get("type") == "eut"
        and inv.get("comp") is not None
        and _recorded_delta(inv) is not None
    ]
    if not flanking:
        return None
    return min(flanking, key=lambda inv: abs(float(inv["comp"]) - comp))


def is_ordered_field_label(name, elements):
    """True when a digitized field's label names a compound rather than a component."""
    if elements:
        return not is_component_phase_label(name, elements)
    return not (name or "").strip().startswith("(")


# Composition slack for matching a formula-derived composition to a hull composition.
FORMULA_TOL = 1e-6


def formula_comp(label, elements):
    """The composition a phase label's FORMULA implies, or None when it will not parse.

    Args:
        label: an MPDS phase label, polymorph suffix and all.
        elements: the two component symbols, low composition first.

    Returns:
        float | None: fraction of ``elements[1]``, or None for a label that is not an
        INTEGER-subscript formula over exactly these two elements.
    """
    base = split_phase_label(label or "")[0]
    if not base or base.startswith("("):
        return None
    try:
        amounts = Composition(base).get_el_amt_dict()
    except Exception:  # noqa: BLE001 - any unparseable label falls back to the drawn position
        return None
    if not amounts or not set(amounts) <= set(elements):
        return None
    # A fractional subscript is a measured composition, matched with the drawn-position tolerance.
    if not all(float(count).is_integer() for count in amounts.values()):
        return None
    left = amounts.get(elements[0], 0.0)
    right = amounts.get(elements[1], 0.0)
    total = left + right
    return right / total if total > 0 else None


def ignored_comp_ranges(span, liq_coverage=None) -> list[list[float]]:
    """Composition intervals over ``[0, 1]`` that carry no digitized liquidus.

    Everything outside ``span``, unioned with the holes ``liquidus_coverage`` reports --
    the gaps :func:`extract_digitized_liquidus` refuses to bridge, which survive into the
    extracted curve as undigitized stretches. Both are positions, so ``span`` and
    ``liq_coverage`` must already be in the caller's component frame.

    Args:
        span: ``[x_lo, x_hi]`` endpoints of the digitized liquidus. Falsy means nothing was
            digitized, which makes the whole axis ignored.
        liq_coverage: ``liquidus_coverage`` output, or None.

    Returns:
        Merged ``[lo, hi]`` intervals, clipped to ``[0, 1]``.
    """
    if not span:
        return [[0.0, 1.0]]
    lo, hi = float(span[0]), float(span[1])
    holes = [[float(a), float(b)] for a, b in ((liq_coverage or {}).get("holes") or [])]
    merged = merge_ranges([[max(0.0, a), min(1.0, b)] for a, b in ([0.0, lo], [hi, 1.0], *holes)])
    return [r for r in merged if r[1] - r[0] > SEGMENT_TOL]


def _thermal_form_records(members: list[dict], t_max: float, decomp: str) -> list[dict]:
    """Per-form segment records of one compound cluster, ordered by temperature.

    ``members`` are the cluster's raw shapes from :func:`_collapse_polymorph_clusters`,
    ordered here by :func:`polymorph_rank` and then by digitized top. The lowest form starts
    at 0 K and each later form starts where the previous one ends; a form ends at its own
    digitized top and the hottest one ends at ``t_max``. A form whose digitized top does not
    clear the form below it spans no temperature range and gets NO record -- the form below
    it already reaches that top. ``decomp`` therefore lands on the hottest form that does
    span a range, and every form beneath it decomposes by ``'transition'``. Every kept
    record's ``comp_range`` is then widened to the same shared envelope -- the widest ``lb``
    and the widest ``ub`` across every kept record's own ``cbounds`` (a field) or
    ``(comp, comp)`` (a line) -- so every segment of one compound draws at one width.

    Returns:
        list[dict]: ``{'form', 'comp_range', 't_lo', 't_hi', 'decomp'}``, one per member that
        spans a temperature range, lowest first, all sharing one ``comp_range``; never empty.
    """
    ordered = sorted(
        members,
        key=lambda member: (polymorph_rank(member.get("polymorph", "")), member["tbounds"][1][1]),
    )
    records = []
    t_lo = 0.0
    for index, member in enumerate(ordered):
        last = index == len(ordered) - 1
        t_hi = float(t_max) if last else max(float(member["tbounds"][1][1]), t_lo)
        if member["type"] == "ss":
            comp_range = (member["cbounds"][0][0], member["cbounds"][1][0])
        else:
            comp_range = (member["comp"], member["comp"])
        records.append(
            {
                "form": member.get("polymorph", ""),
                "comp_range": comp_range,
                "t_lo": t_lo,
                "t_hi": t_hi,
                "decomp": decomp if last else "transition",
            }
        )
        t_lo = t_hi
    kept = [record for record in records if record["t_hi"] > record["t_lo"]] or records[:1]
    kept[-1]["t_hi"] = max(float(t_max), kept[-1]["t_lo"])
    kept[-1]["decomp"] = decomp
    lb = min(record["comp_range"][0] for record in kept)
    ub = max(record["comp_range"][1] for record in kept)
    for record in kept:
        record["comp_range"] = (lb, ub)
    return kept


def get_low_temp_phase_data(
    mpds_json: dict, dft_ch: PhaseDiagram
) -> tuple[tuple[dict, dict, int | float, dict], tuple[dict, dict, int | float]]:
    """Low-temperature phase tables from an MPDS diagram and from the DFT convex hull.

    Args:
        mpds_json (dict): MPDS digitized phase equilibrium data for the system.
        dft_ch (PhaseDiagram): DFT convex hull of the system. Its component frame is the
            frame every MPDS-side composition is reported in.

    Returns:
        ``((congruent, incongruent, max_phase_temp, forms), (mp_phases, mp_phases_ebelow,
        min_form_e))``.

        MPDS side. ``congruent`` and ``incongruent`` map each low-temperature compound
        (:func:`is_low_temperature_phase` after :func:`collapse_polymorphs`; pure-component
        ``'(X)'`` phases excluded, wide ``'(X)'`` solid-solution fields kept) to
        ``((c_lo, c_hi), T_max)``: its composition range and its melting or decomposition
        temperature in K, tabled congruent when that point lies on the digitized liquidus.
        ``max_phase_temp`` is the highest ``T_max``, or the liquidus minimum when nothing is
        tabled. ``forms`` maps each tabled compound name to its thermal forms as a list of
        ``{'form': suffix or '', 'comp_range': (c_lo, c_hi), 't_lo': K, 't_hi': K,
        'decomp': 'congruent' | 'incongruent' | 'transition'}`` records ordered by
        temperature: one per digitized shape that spans a temperature range, the lowest
        starting at 0 K, each later one starting where the previous ends, the last ending at
        ``T_max`` with the compound's own ``decomp`` and every earlier one ``'transition'``.
        Every record of one compound shares the same ``comp_range`` -- the widest ``c_lo`` and
        the widest ``c_hi`` across its own forms. A compound drawn as a single form, or whose
        hotter forms clear no range above it, has one record spanning 0..``T_max``.

        DFT side. ``mp_phases`` and ``mp_phases_ebelow`` map each stable compound entry name
        to ``((x, x), formation energy per atom)`` and ``((x, x), energy below the hull
        without that entry)``; ``min_form_e`` is the lowest formation energy.

    Raises:
        config.CacheModeError: on a LEAN record.
    """
    require_full_record(mpds_json, "get_low_temp_phase_data")

    dft_phases, dft_phases_ebelow = {}, {}
    min_form_e = 0

    for entry in dft_ch.stable_entries:
        comp_dict = entry.composition.fractional_composition.as_dict()
        if len(comp_dict) == 1:
            continue

        comp = comp_dict.get(api.pd_components(dft_ch)[1], 0)
        form_e = dft_ch.get_form_energy_per_atom(entry)
        dft_phases[entry.name] = ((comp, comp), form_e)
        min_form_e = min(form_e, min_form_e)

        ch_copy = PhaseDiagram([e for e in dft_ch.stable_entries if e != entry])
        e_below_hull = -abs(
            dft_ch.get_hull_energy_per_atom(entry.composition)
            - ch_copy.get_hull_energy_per_atom(entry.composition)
        )
        dft_phases_ebelow[entry.name] = ((comp, comp), e_below_hull)

    mpds_congruent_phases, mpds_incongruent_phases, mpds_forms = {}, {}, {}
    max_phase_temp = 0

    hull_components = api.pd_components(dft_ch)
    raw_phases = identify_mpds_phases(mpds_json, elements=hull_components, with_polymorph=True)
    unpaired_ht = mark_unpaired_ht_forms(raw_phases, hull_components, mpds_json=mpds_json)
    clusters = _collapse_polymorph_clusters(raw_phases, mpds_json, elements=hull_components)
    mpds_liquidus, _ = extract_digitized_liquidus(mpds_json)

    if not clusters:
        return (
            (mpds_congruent_phases, mpds_incongruent_phases, max_phase_temp, mpds_forms),
            (dft_phases, dft_phases_ebelow, min_form_e),
        )

    def phase_decomp_on_liq(phase, liq):
        """Determines if a solid phase decomposes on or near the liquidus."""
        if liq is None:
            return False
        for i in range(len(liq) - 1):
            if liq[i][0] == phase["tbounds"][1][0]:
                return abs(liq[i][1] - phase["tbounds"][1][1]) < 10
            # composition falls between two points:
            elif liq[i][0] < phase["tbounds"][1][0] < liq[i + 1][0]:
                return abs((liq[i][1] + liq[i + 1][1]) / 2 - phase["tbounds"][1][1]) < 10

    temp_threshold = low_temp_threshold(mpds_json)

    # Pure-component phases are not tabled; wide '(X)' solid-solution fields are.
    for phase, members in clusters:
        if phase["type"] == "comp" or not is_low_temperature_phase(
            phase, temp_threshold, unpaired_ht=unpaired_ht
        ):
            continue
        if phase["type"] == "ss":
            comp_range = (phase["cbounds"][0][0], phase["cbounds"][1][0])
        else:
            comp_range = (phase["comp"], phase["comp"])
        t_max = phase["tbounds"][1][1]
        if phase_decomp_on_liq(phase, mpds_liquidus):
            decomp, table = "congruent", mpds_congruent_phases
        else:
            decomp, table = "incongruent", mpds_incongruent_phases
        table[phase["name"]] = (comp_range, t_max)
        mpds_forms[phase["name"]] = _thermal_form_records(members, t_max, decomp)
        max_phase_temp = max(t_max, max_phase_temp)

    if max_phase_temp == 0 and mpds_liquidus:
        max_phase_temp = min(mpds_liquidus, key=lambda x: x[1])[1]

    # Mirrors the MPDS side into the hull's component frame when the json is reversed.
    if not mpds_frame_matches(mpds_json, hull_components):

        def _mirror_ranges(d):
            return {name: ((1 - c2, 1 - c1), t) for name, ((c1, c2), t) in d.items()}

        def _mirror_forms(forms):
            return {
                name: [
                    {
                        **record,
                        "comp_range": (
                            1 - record["comp_range"][1],
                            1 - record["comp_range"][0],
                        ),
                    }
                    for record in records
                ]
                for name, records in forms.items()
            }

        mpds_congruent_phases = _mirror_ranges(mpds_congruent_phases)
        mpds_incongruent_phases = _mirror_ranges(mpds_incongruent_phases)
        mpds_forms = _mirror_forms(mpds_forms)

    return (
        (mpds_congruent_phases, mpds_incongruent_phases, max_phase_temp, mpds_forms),
        (dft_phases, dft_phases_ebelow, min_form_e),
    )


def print_phase_mismatch_chart(low_t_exp_phases: list[dict], dft_phase_comps) -> None:
    """Prints the low-temperature MPDS-vs-DFT phase-mismatch terminal chart.

    Called from ``BinaryLiquid.fit_parameters``: the MP row needs DFT phase compositions,
    which :func:`identify_invariant_points` does not consult. The scale bar is drawn on the
    fitting grid's 0.01 composition step (101 bins), and the COMP legend below is hardcoded
    to the same width.

    Args:
        low_t_exp_phases (list): Low-temperature phase dicts from
            ``identify_invariant_points`` (reads 'cbounds' / 'comp').
        dft_phase_comps (list | None): DFT phase compositions (x of components[1]);
            None/empty leaves the MP row blank.
    """
    n_bins = 101
    mpds_phases_strs = [" "] * n_bins
    mp_phases_strs = [" "] * n_bins
    for phase in low_t_exp_phases:
        if "cbounds" in phase:
            min_c_ind = int(phase["cbounds"][0][0] * 100)
            max_c_ind = min(int(phase["cbounds"][1][0] * 100), n_bins)
            mpds_phases_strs[min_c_ind:max_c_ind] = ["|"] * (max_c_ind - min_c_ind)
        else:
            mpds_phases_strs[min(int(phase["comp"] * 100), n_bins - 1)] = "|"
    for comp in dft_phase_comps or []:
        mp_phases_strs[min(int(comp * 100), n_bins - 1)] = "|"
    # print(), not logging: the aligned monospace rows are the product.
    print("\n--- Low temperature phase mismatch ---")
    print("MPDS:", "[" + "".join(mpds_phases_strs) + "]")
    print("MP:  ", "[" + "".join(mp_phases_strs) + "]")
    print(
        "COMP:",
        " 0"
        + " " * 9
        + "10"
        + " " * 8
        + "20"
        + " " * 8
        + "30"
        + " " * 8
        + "40"
        + " " * 8
        + "50"
        + " " * 8
        + "60"
        + " " * 8
        + "70"
        + " " * 8
        + "80"
        + " " * 8
        + "90"
        + " " * 8
        + "100",
    )


def _find_adj_phases(
    point: list | tuple, low_t_exp_phases, components, component_data, t_tol
) -> tuple[dict, dict]:
    """The nearest low-temperature phase on each side of ``point``.

    Args:
        point: a ``(composition, temperature)`` point.
        low_t_exp_phases: candidate phases; the pure components are added to them.
        components: the two component names.
        component_data: per-component data, read for ``t_fusion``.
        t_tol: how far below ``point``'s temperature a phase may top out and still
            count as present there.

    Returns:
        tuple: ``(left, right)``, each a phase dict, or None where that side is empty.
    """
    # order matters: min() below breaks distance ties by list position
    all_lowt_phases = low_t_exp_phases + [
        {
            "name": components[0],
            "comp": 0,
            "type": "lc",
            "tbounds": [[], [0, component_data[components[0]].t_fusion]],
        },
        {
            "name": components[1],
            "comp": 1,
            "type": "lc",
            "tbounds": [[], [1, component_data[components[1]].t_fusion]],
        },
    ]
    all_lowt_phases = [p for p in all_lowt_phases if p["tbounds"][1][1] + t_tol >= point[1]]
    lhs_phases = [phase for phase in all_lowt_phases if phase["comp"] < point[0]]
    adj_lhs_phase = (
        None if not lhs_phases else min(lhs_phases, key=lambda x: abs(x["comp"] - point[0]))
    )
    rhs_phases = [phase for phase in all_lowt_phases if phase["comp"] > point[0]]
    adj_rhs_phase = (
        None if not rhs_phases else min(rhs_phases, key=lambda x: abs(x["comp"] - point[0]))
    )
    return adj_lhs_phase, adj_rhs_phase


def liquidus_crossings(liquidus, temp):
    """Compositions where the digitized liquidus crosses ``temp``, in curve order.

    The digitization is a polyline, so each crossing is exact under linear interpolation.
    A segment lying flat at ``temp`` contributes nothing -- it has no single crossing.
    """
    out = []
    points = [(float(p[0]), float(p[1])) for p in liquidus]
    for (x0, t0), (x1, t1) in zip(points, points[1:]):
        if t0 == t1:
            continue
        if (t0 - temp) * (t1 - temp) <= 0:
            out.append(x0 + (temp - t0) * (x1 - x0) / (t1 - t0))
    return out


def _unclaimed_maximum_between(maxima, phase_comp, endpoint_comp, tol=_CONGRUENT_MATCH_TOL):
    """Whether an unclaimed liquidus maximum separates a phase from a liquidus endpoint.

    A maximum lying strictly between the two is another compound's congruent dome, and the
    liquid beyond it is conjugate to that compound rather than to this phase. A maximum
    within ``tol`` of ``phase_comp`` is the phase's own dome and never separates it.

    Args:
        maxima: interior liquidus maxima no congruent record has claimed, as ``[comp, temp]``.
        phase_comp: the phase composition.
        endpoint_comp: the candidate peritectic liquid composition.
        tol: composition window within which a maximum counts as the phase's own.

    Returns:
        bool
    """
    lo, hi = sorted((float(phase_comp), float(endpoint_comp)))
    return any(
        lo < float(coords[0]) < hi and abs(float(coords[0]) - float(phase_comp)) > tol
        for coords in maxima
    )


def _place_on_liquidus(liquidus, comp, temp):
    """Refine ``comp`` to the liquidus crossing at ``temp``, and give the curve that point.

    Args:
        liquidus: digitized liquidus points, MUTATED IN PLACE to carry the crossing.
        comp: the composition to refine, from the caller's own endpoint selection.
        temp: the temperature the crossing is taken at.

    Returns:
        float: the refined composition, or ``comp`` unchanged when the curve never
        reaches ``temp``.
    """
    crossings = liquidus_crossings(liquidus, temp)
    if not crossings:
        return comp
    # nearest to the endpoint the caller already chose, so the selection is preserved
    placed = min(crossings, key=lambda x: abs(x - comp))
    if not any(
        abs(float(p[0]) - placed) <= SEGMENT_TOL and abs(float(p[1]) - temp) <= SEGMENT_TOL
        for p in liquidus
    ):
        index = next((i for i, p in enumerate(liquidus) if float(p[0]) > placed), len(liquidus))
        liquidus.insert(index, [placed, float(temp)])
    return placed


_MIG_LABEL_RE = re.compile(r"^L\S*\s*\+\s*L\S*$")
# kelvin between a dome polygon's top and the frame top that reads as clipping
_FRAME_EDGE_TOL_K = 4.0


def is_two_liquid_label(text) -> bool:
    """True for a printed two-liquid field label such as ``L1 + L2``, in any spelling.

    Args:
        text: the label text as printed, HTML markup allowed.
    """
    return bool(_MIG_LABEL_RE.match(_HTML_TAG_RE.sub("", str(text or "")).strip()))


def two_liquid_label_comps(mpds_json: dict, elements=None) -> list[float]:
    """Compositions at which the diagram PRINTS a two-liquid field label.

    Direct evidence of a liquid-liquid gap, independent of whether the digitized geometry
    yields a dome: a stretch of liquidus carrying this label has a second liquid as its
    conjugate, not a solid.

    Args:
        mpds_json: a full MPDS record; its ``labels`` block is read.
        elements: component symbols to report the compositions in. Omitted, the record's
            own frame is used.

    Returns:
        list[float]: one composition per printed label, in composition order.
    """
    mirrored = elements is not None and not mpds_frame_matches(mpds_json, elements)
    comps = []
    for entry in (mpds_json or {}).get("labels") or []:
        try:
            if not is_two_liquid_label(entry[0]):
                continue
            x = float(entry[1][0])
        except (TypeError, ValueError, IndexError, KeyError):
            continue
        if x > 1.0:
            x /= 100.0
        comps.append(1.0 - x if mirrored else x)
    return sorted(comps)


def _two_liquid_lens_pool(mpds_json, components):
    """Printed two-liquid label anchors and two-phase shape outlines, in the evaluation frame.

    Args:
        mpds_json: the MPDS record, in its own component frame.
        components: component names in the evaluation frame.

    Returns:
        tuple: ``(labels, shapes)`` -- one ``[x, T]`` anchor per printed two-liquid label, and
        one ``[[x, T], ...]`` vertex list per two-phase shape whose svgpath parses.
    """
    mirrored = not mpds_frame_matches(mpds_json, components)

    def frame_x(x):
        return 1 - x if mirrored else x

    labels = [
        [frame_x(label[1][0] / 100.0), label[1][1] + 273.15]
        for label in mpds_json["labels"]
        if is_two_liquid_label(label[0])
    ]
    shapes = []
    for shape in mpds_json["shapes"]:
        if shape.get("nphases") != 2:
            continue
        data = shape_to_list(shape["svgpath"])
        if data:
            shapes.append([[frame_x(x), t] for x, t in data])
    return labels, shapes


def _two_liquid_apex_maxima(mpds_json, components, maxima, *, t_tol):
    """``(x, T)`` of every liquidus maximum a printed two-liquid dome tops out at.

    A solid crystallizing at a dome's crest melts by a syntectic reaction, ``L1 + L2 -> S``,
    not congruently. Each printed label claims its nearest maximum, and the claim stands only
    where a two-phase shape tops out there and brackets it in composition -- the pairing
    ``_miscibility_gap_records`` makes.

    Args:
        mpds_json: the MPDS record, in its own component frame.
        components: component names in the evaluation frame.
        maxima: interior liquidus maxima; not modified.
        t_tol: kelvin tolerance between a shape's top and a maximum.

    Returns:
        set: ``(x, T)`` tuples; empty when the diagram prints no two-liquid label.
    """
    if not maxima:
        return set()
    labels, shapes = _two_liquid_lens_pool(mpds_json, components)
    claimed = set()
    for anchor in labels:
        nearest = min(maxima, key=lambda m: abs(m[0] - anchor[0]))
        for data in shapes:
            by_t = sorted(data, key=lambda p: p[1])
            if abs(by_t[-1][1] - nearest[1]) >= t_tol or abs(by_t[-1][0] - nearest[0]) >= 0.05:
                continue
            by_x = sorted(data, key=lambda p: p[0])
            if by_x[0][0] < nearest[0] < by_x[-1][0]:
                claimed.add((nearest[0], nearest[1]))
                break
    return claimed


def _walk_to_isotherm(liquidus, start, step, skip, temp):
    """Composition where the liquidus first falls through ``temp``, walking away from ``start``.

    Examines the segment ``(i, i + step)`` for ``i = start + step * skip``, then
    ``i + step``, and so on while ``i + step`` is an index of ``liquidus``, and stops at the
    first segment with ``liquidus[i + step][1] < temp <= liquidus[i][1]``.

    Args:
        liquidus: ``[x, T]`` points; not modified.
        start: the index the walk starts from.
        step: ``1`` to walk toward the end of the list, ``-1`` toward its start.
        skip: how many steps from ``start`` the first examined segment begins.
        temp: the isotherm temperature.

    Returns:
        The composition on the first crossing segment, linearly interpolated at ``temp``, or
        None when no examined segment crosses it.

    Raises:
        ZeroDivisionError: the crossing segment is vertical and its coordinates are Python
            floats.
    """
    i = start + step * skip
    while 0 <= i + step <= len(liquidus) - 1:
        j = i + step
        if liquidus[j][1] < temp <= liquidus[i][1]:
            m = (liquidus[j][1] - liquidus[i][1]) / (liquidus[j][0] - liquidus[i][0])
            return (temp - liquidus[i][1]) / m + liquidus[i][0]
        i += step
    return None


def _miscibility_gap_records(
    mpds_json,
    components,
    digitized_liq,
    maxima,
    minima,
    low_t_exp_phases,
    component_data,
    temp_range,
    *,
    t_tol,
    holes,
):
    """``mig`` invariant records seeded by the diagram's printed two-liquid labels.

    Each label is matched to the nearest interior liquidus maximum, which becomes the
    apex; a two-phase polygon topping out there supplies the feet, otherwise the walk from
    the adjacent monotectic minimum does. A record is ``partial`` (``apex`` None) when a
    liquidus hole lies between its feet or the matched polygon reaches the frame top. Every
    label is processed; one falling inside a recorded dome is skipped. ``maxima`` and
    ``minima`` lose the points a record consumes.

    Args:
        mpds_json: the MPDS record, in its own component frame.
        components: component names in the evaluation frame.
        digitized_liq: ``[x, T]`` liquidus points in the evaluation frame.
        maxima: interior liquidus maxima, consumed in place.
        minima: liquidus minima, consumed in place.
        low_t_exp_phases: phase records used to name the adjacent solid.
        component_data: per-component reference data.
        temp_range: ``[T_min, T_max]`` of the frame in K.
        t_tol: kelvin tolerance for matching a polygon top to the apex.
        holes: ``[lo, hi]`` composition intervals with no digitized liquidus.

    Returns:
        list[dict]: records with keys ``type, comp, cbounds, tbounds, phases,
        phase_comps, apex, partial, source``.
    """
    misc_gap_labels, two_phase_shapes = _two_liquid_lens_pool(mpds_json, components)
    frame_top = temp_range[1] if temp_range else None

    records = []
    for mgl in misc_gap_labels:
        if len(maxima) < 1:
            break
        if any(
            rec["cbounds"][0][0] - 1e-6 <= mgl[0] <= rec["cbounds"][1][0] + 1e-6 for rec in records
        ):
            continue
        nearest_maxima = min(maxima, key=lambda x: abs(x[0] - mgl[0]))

        tbounds = [None, nearest_maxima]
        cbounds = None
        matched = None
        phases = None
        phase_comps = None

        for data in two_phase_shapes:
            data = sorted(data, key=lambda x: x[1])
            if not (
                abs(data[-1][1] - nearest_maxima[1]) < t_tol
                and abs(data[-1][0] - nearest_maxima[0]) < 0.05
            ):
                continue
            tbounds = [data[0], data[-1]]
            data = sorted(data, key=lambda x: x[0])
            if not data[0][0] < nearest_maxima[0] < data[-1][0]:
                continue
            cbounds = [list(data[0]), list(data[-1])]
            matched = data
            break

        if len(minima) >= 1:
            adj_mono = min(
                minima,
                key=lambda x: (
                    abs(tbounds[1][0] - x[0]) + 2 * (abs(tbounds[1][1] - x[1]) / temp_range[1])
                ),
            )
            tbounds[0] = [tbounds[1][0], adj_mono[1]]
            adj_phases = _find_adj_phases(
                adj_mono, low_t_exp_phases, components, component_data, t_tol
            )

            if adj_mono[0] < tbounds[1][0]:
                if adj_phases[0] is not None:
                    phase_comps = [adj_phases[0]["comp"]]
                    phases = [adj_phases[0]["name"]]
                if not cbounds:
                    lhs_ind = digitized_liq.index(adj_mono)
                    rhs_comp = _walk_to_isotherm(
                        digitized_liq, lhs_ind, step=1, skip=1, temp=adj_mono[1]
                    )
                    if rhs_comp is not None:
                        cbounds = [adj_mono, [rhs_comp, adj_mono[1]]]
            elif adj_mono[0] > tbounds[1][0]:
                if adj_phases[1] is not None:
                    phase_comps = [adj_phases[1]["comp"]]
                    phases = [adj_phases[1]["name"]]
                if not cbounds:
                    rhs_ind = digitized_liq.index(adj_mono)
                    # skip=2 here and skip=1 on the left is a preserved asymmetry; do not unify.
                    lhs_comp = _walk_to_isotherm(
                        digitized_liq, rhs_ind, step=-1, skip=2, temp=adj_mono[1]
                    )
                    if lhs_comp is not None:
                        cbounds = [[lhs_comp, adj_mono[1]], adj_mono]
            if cbounds and cbounds[0][1] != cbounds[1][1]:
                cbounds[0][1] = adj_mono[1]
                cbounds[1][1] = adj_mono[1]
            minima.remove(adj_mono)

        if not cbounds:
            continue

        x_lo, x_hi = cbounds[0][0], cbounds[1][0]
        hole_inside = any(x_lo - 1e-6 <= a and b <= x_hi + 1e-6 for a, b in holes)
        at_frame = (
            matched is not None
            and frame_top is not None
            and abs(max(p[1] for p in matched) - frame_top) <= _FRAME_EDGE_TOL_K
        )
        partial = bool(hole_inside or at_frame)
        records.append(
            {
                "type": "mig",
                "comp": tbounds[1][0],
                "cbounds": cbounds,
                "tbounds": tbounds,
                "phases": phases,
                "phase_comps": phase_comps,
                "apex": None if partial else [float(tbounds[1][0]), float(tbounds[1][1])],
                "partial": partial,
                "source": "label",
            }
        )
        maxima.remove(nearest_maxima)
    return records


def _congruent_invariants(low_t_exp_phases, maxima, endpoint_candidates, syntectic_maxima, t_tol):
    """``cmp`` invariant records: liquidus maxima matched to a congruently melting phase.

    The interior maxima and the endpoint candidates are visited in composition order, and
    each is tested against the nearest phase. A phase matches when its type is ``lc``,
    ``ss`` or ``comp``, its composition is within ``_CONGRUENT_MATCH_TOL`` of the maximum,
    and its top temperature plus ``t_tol`` reaches the maximum's temperature. An endpoint
    candidate never matches a terminal phase; an interior maximum in ``syntectic_maxima``
    never matches.

    Args:
        low_t_exp_phases: low-temperature phase records. MUTATED: re-sorted in place by
            distance to each candidate in turn, and each matched phase's ``type`` is set to
            ``"cmp"``.
        maxima: interior liquidus maxima as ``[comp, temp]``. MUTATED: each matched maximum
            is removed.
        endpoint_candidates: liquidus endpoint maxima as ``[comp, temp]``; not modified.
        syntectic_maxima: ``(comp, temp)`` tuples of the maxima a two-liquid dome tops out
            at; not modified.
        t_tol: kelvin tolerance between a phase's top temperature and a maximum.

    Returns:
        list[dict]: ``cmp`` records with keys ``type, comp, temp, phases, phase_comps``, in
        match order.
    """
    records = []
    candidates = [(coords, False) for coords in maxima] + [
        (coords, True) for coords in endpoint_candidates
    ]
    candidates.sort(key=lambda item: float(item[0][0]))
    for coords, from_endpoint in candidates:
        low_t_exp_phases.sort(key=lambda x: abs(x["comp"] - coords[0]))
        phase = low_t_exp_phases[0]
        # A liquidus endpoint near a terminal solid solution is not taken as a congruent maximum.
        if from_endpoint and _is_terminal_phase(phase):
            continue
        if not from_endpoint and (coords[0], coords[1]) in syntectic_maxima:
            continue
        if (
            phase["type"] in ["lc", "ss", "comp"]
            and abs(phase["comp"] - coords[0]) <= _CONGRUENT_MATCH_TOL
            and phase["tbounds"][1][1] + t_tol >= coords[1]
        ):
            phase["type"] = "cmp"
            records.append(
                {
                    "type": phase["type"],
                    "comp": phase["comp"],
                    "temp": phase["tbounds"][1][1],
                    "phases": [phase["name"]],
                    "phase_comps": [phase["comp"]],
                }
            )
            if not from_endpoint:
                maxima.remove(coords)

    return records


def _peritectic_invariants(low_t_exp_phases, digitized_liq, maxima):
    """``per`` invariant records: the liquidus point each non-congruent phase melts onto.

    Phases are visited in list order. A phase whose name contains ``(`` is skipped, and a
    ``cmp`` phase records nothing but bounds the search for the phases after it. Each other
    phase records the end of a liquidus section at its top temperature that lies nearest
    it, placed on the curve at that temperature. The curve's own two ends, ends with an
    earlier-visited phase in between and ends with an unclaimed maximum in between are
    excluded; a phase left with no end records nothing.

    Args:
        low_t_exp_phases: low-temperature phase records, in visiting order; not modified.
        digitized_liq: ``[x, T]`` liquidus points. MUTATED: each record's liquidus point is
            inserted where the curve does not already carry it (see ``_place_on_liquidus``).
        maxima: interior liquidus maxima no ``cmp`` record claimed; not modified.

    Returns:
        list[dict]: ``per`` records with keys ``type, comp, temp, phases, phase_comps``, in
        phase order.
    """
    records = []
    stable_phase_comps = []

    # Main loop for peritectic phase identification
    for phase in low_t_exp_phases:
        if "(" in phase["name"]:  # Ignore component SS phases
            continue

        # cmp phases record nothing but bound the composition search
        if phase["type"] == "cmp":
            stable_phase_comps.append(phase["comp"])
            continue

        sections, current_section = [], []
        phase_temp = phase["tbounds"][1][1]

        for i in range(len(digitized_liq) - 1):
            liq_point, next_liq_point = digitized_liq[i], digitized_liq[i + 1]
            liq_temp, next_liq_temp = liq_point[1], next_liq_point[1]

            # Liquidus point is above or equal to phase temp
            if liq_temp >= phase_temp:
                current_section.append(liq_point)
                if next_liq_temp >= phase_temp and i + 1 == len(digitized_liq) - 1:
                    current_section.append(next_liq_point)
                    sections.append(current_section)

            # Liquidus point is first point below phase temp
            elif current_section:
                if abs(phase_temp - current_section[-1][1]) > abs(phase_temp - liq_temp):
                    current_section.append(liq_point)  # Add to section if closer to phase temp
                sections.append(current_section)  # End section
                current_section = []

            # Next liquidus point is above phase temp
            elif next_liq_temp >= phase_temp > liq_temp:
                if abs(phase_temp - next_liq_temp) > abs(phase_temp - liq_temp):
                    current_section.append(
                        liq_point
                    )  # Add if below phase temp and closer than next point

        # Find endpoints of liquidus segments excluding the component ends
        endpoints = [
            section[i]
            for section in sections
            for i in [0, -1]
            if section[i] not in [digitized_liq[0], digitized_liq[-1]]
        ]

        # Filter endpoints if there exists a stable phase between the current phase and the liquidus
        for comp in stable_phase_comps:
            endpoints = [
                ep
                for ep in endpoints
                if abs(comp - ep[0]) > abs(phase["comp"] - ep[0])
                or abs(comp - phase["comp"]) > abs(phase["comp"] - ep[0])
            ]

        # A congruent dome the cmp pass left unclaimed also separates this phase from the liquid.
        endpoints = [
            ep for ep in endpoints if not _unclaimed_maximum_between(maxima, phase["comp"], ep[0])
        ]

        # Sort by increasing distance to liquidus to find the shortest distance
        endpoints.sort(key=lambda x: abs(x[0] - phase["comp"]))

        # The peritectic point: the liquidus point closest to the phase, refined onto the curve.
        if endpoints:
            records.append(
                {
                    "type": "per",
                    "comp": _place_on_liquidus(digitized_liq, endpoints[0][0], phase_temp),
                    "temp": phase_temp,
                    "phases": [phase["name"]],
                    "phase_comps": [phase["comp"]],
                }
            )

        stable_phase_comps.append(phase["comp"])

    return records


def _eutectic_invariants(minima, low_t_exp_phases, components, component_data, t_tol):
    """``eut`` invariant records, one per liquidus minimum, with the nearest phase on each side.

    Args:
        minima: liquidus minima as ``[comp, temp]``; not modified.
        low_t_exp_phases: candidate phases for ``_find_adj_phases``; not modified.
        components: the two component names.
        component_data: per-component data, read for ``t_fusion``.
        t_tol: passed to ``_find_adj_phases``.

    Returns:
        list[dict]: ``eut`` records with keys ``type, comp, temp, phases, phase_comps``, in
        ``minima`` order; a side with no phase is None in ``phases`` and ``phase_comps``.
    """
    records = []
    for coords in minima:
        adj_phases = _find_adj_phases(coords, low_t_exp_phases, components, component_data, t_tol)
        phases, phase_comps = zip(
            *[
                (None, None) if phase is None else (phase["name"], phase["comp"])
                for phase in adj_phases
            ]
        )

        records.append(
            {
                "type": "eut",
                "comp": coords[0],
                "temp": coords[1],
                "phases": list(phases),
                "phase_comps": list(phase_comps),
            }
        )

    return records


def identify_invariant_points(
    mpds_json: dict,
    components,
    digitized_liq: list[list],
    component_data: dict,
    temp_range,
    *,
    verbose=False,
    check_full_ss=True,
    t_tol=15,
    liquidus_holes=None,
) -> tuple[list[dict], list[dict], bool]:
    """Identifies invariant points in the MPDS data from the digitized liquidus and JSON.

    DFT phases are not considered. Requires both the liquidus and the JSON of a binary system.

    Args:
        mpds_json (dict): MPDS digitized phase equilibrium data for the system.
        components (list): Component names in the evaluation frame ([A, B]).
        digitized_liq (list): Digitized liquidus points as [x, T] pairs.
        component_data (dict): Per-component reference data; only ``.t_fusion`` is read.
        temp_range (list): Evaluation temperature range [min, max] in K; the max scales
            the composition-vs-temperature distance metric used to pair a miscibility-gap
            dome with its adjacent monotectic.
        verbose (bool): If True, outputs additional debugging information.
        check_full_ss (bool): If True, stop early when the MPDS labels report a solid
            solution over the full composition range.
        t_tol (int): Temperature tolerance for invariant point identification.
        liquidus_holes (list | None): ``[lo, hi]`` composition intervals with no digitized
            liquidus, in the evaluation frame; None means no hole is known.

    Returns:
        tuple: (invariant points, low-temperature phases from the MPDS JSON, full_comp_ss).
        full_comp_ss is True when identification stopped at a full-composition solid solution;
        the caller decides whether fitting proceeds.

    Raises:
        config.CacheModeError: on a LEAN record.
    """
    require_full_record(
        mpds_json,
        "identify_invariant_points",
        escape="fit without invariant constraints by passing disable_inv_constrs=True",
    )
    if mpds_json["reference"] is None:
        logger.warning("System JSON does not contain any data!")
        return [], [], False

    # Phases from the MPDS JSON, polymorphs collapsed to one entry per compound.
    raw_phases = identify_mpds_phases(mpds_json, elements=components, with_polymorph=True)
    unpaired_ht = mark_unpaired_ht_forms(raw_phases, components, mpds_json=mpds_json)
    phases = collapse_polymorphs(raw_phases, mpds_json, elements=components)
    if not mpds_frame_matches(mpds_json, components):
        phases = mirror_mpds_phases(phases)
        unpaired_ht = frozenset((base, 1 - comp) for base, comp in unpaired_ht)
    invariants = [
        phase for phase in phases if phase["type"] == "mig"
    ]  # Miscibility gaps are not phases.
    # ...nor true invariant points, but the invariant list carries them.

    # Filter low-temperature phases
    threshold = low_temp_threshold(mpds_json)
    low_t_exp_phases = [
        phase
        for phase in phases
        if is_low_temperature_phase(phase, threshold, unpaired_ht=unpaired_ht)
        or "(" in phase["name"]
    ]

    if verbose:
        logger.info("--- Low temperature phases including component solid solutions ---")
        for phase in low_t_exp_phases:
            logger.info(phase)

    # Identify full composition solid solutions
    phase_labels = [label[0] for label in mpds_json["labels"]]
    ss_label = f"({components[0]}, {components[1]})"
    ss_label_inv = f"({components[1]}, {components[0]})"
    ss_labels = [
        ss_label,
        f"{ss_label} ht",
        f"{ss_label} rt",
        ss_label_inv,
        f"{ss_label_inv} ht",
        f"{ss_label_inv} rt",
    ]
    full_comp_ss = bool([label for label in phase_labels if label in ss_labels])
    if full_comp_ss and check_full_ss:
        # Logs and stops identification; the caller decides whether fitting proceeds.
        logger.warning(
            f"Full-composition solid solution detected in MPDS labels for "
            f"{components[0]}-{components[1]}; invariant identification stopped early."
        )
        return invariants, low_t_exp_phases, True

    # Locate maxima and minima in liquidus
    maxima = find_local_maxima(digitized_liq)
    minima = find_local_minima(digitized_liq)

    # The ends of the digitized curve are candidates for congruent melting only; ``maxima`` holds
    # interior maxima.
    endpoint_candidates = _endpoint_maxima(digitized_liq)

    # Dome crests are reserved from the full maxima list before _congruent_invariants runs.
    syntectic_maxima = _two_liquid_apex_maxima(mpds_json, components, maxima, t_tol=t_tol)

    # Assign congruent melting points
    if low_t_exp_phases:
        invariants.extend(
            _congruent_invariants(
                low_t_exp_phases, maxima, endpoint_candidates, syntectic_maxima, t_tol
            )
        )

    # Sort by descending temperature for peritectic identification
    low_t_exp_phases.sort(key=lambda x: x["tbounds"][1][1], reverse=True)

    invariants.extend(
        _miscibility_gap_records(
            mpds_json,
            components,
            digitized_liq,
            maxima,
            minima,
            low_t_exp_phases,
            component_data,
            temp_range,
            t_tol=t_tol,
            holes=liquidus_holes or [],
        )
    )

    invariants.extend(_peritectic_invariants(low_t_exp_phases, digitized_liq, maxima))

    # Identify eutectic points
    invariants.extend(
        _eutectic_invariants(minima, low_t_exp_phases, components, component_data, t_tol)
    )

    invariants.sort(key=lambda x: x["comp"])
    invariants = [inv for inv in invariants if inv["type"] not in ["lc", "ss", "comp"]]
    if verbose:
        logger.info("--- Identified invariant points ---")
        for inv in invariants:
            logger.info(inv)
    return invariants, low_t_exp_phases, False


def mig_apex(iv: dict) -> list | None:
    """The ``[x, T]`` apex of a ``mig`` invariant, or None when the record does not know it.

    Reads ``apex`` when the key is present, reports None for a record marked ``partial``,
    and falls back to the upper ``tbounds`` point of a record without an ``apex`` key.

    Args:
        iv: one invariant record.

    Returns:
        list | None: ``[x, T]`` in the record's own units, or None.
    """
    if "apex" in iv:
        apex = iv["apex"]
        return None if apex is None else [float(apex[0]), float(apex[1])]
    if iv.get("partial"):
        return None
    tbounds = iv.get("tbounds") or []
    if len(tbounds) < 2 or not tbounds[1]:
        return None
    return [float(tbounds[1][0]), float(tbounds[1][1])]


def mig_feet(invariants):
    """``(lo, hi)`` composition feet of every miscibility gap, in composition order.

    The feet come from ``cbounds``. A ``mig`` record's ``comp`` is the dome APEX, which is
    not a field boundary.
    """
    feet = []
    for iv in invariants:
        if iv.get("type") != "mig":
            continue
        cbounds = iv.get("cbounds") or []
        if len(cbounds) < 2 or not cbounds[0] or not cbounds[1]:
            continue
        a, b = float(cbounds[0][0]), float(cbounds[1][0])
        feet.append((min(a, b), max(a, b)))
    return sorted(feet)


def bounding_invariant_comps(invariants) -> list[float]:
    """Compositions at which an invariant may bound a neighbouring phase's field.

    Every invariant contributes its ``comp`` except a ``mig``, which contributes the two
    feet ``mig_feet`` reads from its ``cbounds``; a ``mig`` carrying no usable ``cbounds``
    contributes nothing.

    Args:
        invariants: invariant records.

    Returns:
        list[float]: sorted, de-duplicated compositions.
    """
    comps = {
        round(float(iv["comp"]), 9)
        for iv in (invariants or [])
        if isinstance(iv, dict) and iv.get("comp") is not None and iv.get("type") != "mig"
    }
    comps |= {round(float(c), 9) for pair in mig_feet(invariants or []) for c in pair}
    return sorted(comps)


def subtract_ranges(ranges, cuts):
    """``ranges`` with every part of ``cuts`` removed, merged and in composition order."""
    if not cuts:
        return merge_ranges([list(r) for r in ranges])
    out = []
    for lo, hi in merge_ranges([list(r) for r in ranges]):
        cursor = lo
        for a, b in merge_ranges([list(c) for c in cuts]):
            if b <= cursor or a >= hi:
                continue
            if a > cursor:
                out.append([cursor, min(a, hi)])
            cursor = max(cursor, b)
            if cursor >= hi:
                break
        if hi - cursor > SEGMENT_TOL:
            out.append([cursor, hi])
    return merge_ranges(out)


def mig_feet_on_liquidus(iv, liquidus, foot_tol):
    """True when both feet of a ``mig`` record sit on the digitized liquidus at its isotherm.

    A two-liquid dome meets the liquidus at both feet, on the monotectic isotherm. The
    matcher accepts any two-phase shape whose top is near a liquidus maximum, so it also
    catches ordinary ``L + solid`` fields -- and trimming one of those from the mask would
    declare a real solid-conjugate liquidus supported.

    Args:
        iv: one invariant record.
        liquidus: digitized ``[x, T]`` pairs, in the report's component frame.
        foot_tol: kelvin a foot may sit off the liquidus before the record is rejected.
    """
    cbounds = iv.get("cbounds") or []
    if iv.get("type") != "mig" or len(cbounds) < 2 or not liquidus:
        return False
    xs = [float(pt[0]) for pt in liquidus]
    ts = [float(pt[1]) for pt in liquidus]
    for foot in cbounds[:2]:
        if foot is None or len(foot) < 2:
            return False
        x, temp = float(foot[0]), float(foot[1])
        if not xs[0] - SEGMENT_TOL <= x <= xs[-1] + SEGMENT_TOL:
            return False
        if abs(float(np.interp(x, xs, ts)) - temp) > foot_tol:
            return False
    return True


def validated_mig_feet(invariants, liquidus, foot_tol):
    """``mig_feet`` restricted to the records whose feet the liquidus actually confirms."""
    return mig_feet(
        [iv for iv in (invariants or []) if mig_feet_on_liquidus(iv, liquidus, foot_tol)]
    )
