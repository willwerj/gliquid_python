"""
Author: Joshua Willwerth
Description: MPDS digitized-phase-diagram parsing for gliquid: downloading/caching the
publicly-available Materials Platform for Data Science (MPDS) binary phase-diagram JSONs,
extracting the digitized liquidus, classifying the digitized solid phases, and
cross-referencing them against the DFT convex hull for the fitting constraints.

System input parsing lives in ``gliquid.phase.validate_and_format_system``; the DFT entry
cache/load layer lives in ``gliquid.api``.
GitHub: https://github.com/willwerj
ORCID: https://orcid.org/0009-0004-6334-9426
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass

import numpy as np
from pymatgen.analysis.phase_diagram import PhaseDiagram

import gliquid.api as api
import gliquid.cache as cache
import gliquid.config as config
from gliquid.phase import SS_SPACEGROUPS, UNARY, validate_and_format_system

logger = logging.getLogger(__name__)

# MPDS shape['phase'] identifiers look like 'Zr/229/cI2' -- prototype/spacegroup/Pearson.
_SHAPE_PHASE_RE = re.compile(r"^(?P<prototype>[^/]+)/(?P<spacegroup>\d+)/(?P<pearson>[^/]+)$")

# MPDS polymorph suffixes and their temperature ordering (see polymorph_rank). 'hp' is a
# high-PRESSURE form, not a thermal one, so it sorts last and never wins the
# low-temperature slot. Subscript markup only ever appears in the 'labels' block.
_POLYMORPH_RE = re.compile(r"^(lt|rt|ht|hp)(\d*)$", re.I)
_POLYMORPH_BASE_RANK = {"lt": 0, "rt": 10, "": 10, "ht": 20, "hp": 90}
_HTML_TAG_RE = re.compile(r"<[^>]+>")

# extract_digitized_liquidus linearly interpolates across any consecutive-point composition
# gap wider than _FILL_GAP_X, inserting synthetic points at _FILL_STEP_X spacing, but only
# WITHIN a covered region. The liquidus_coverage hole tolerance (config.liquidus_gap_tol)
# is deliberately looser than _FILL_GAP_X.
#
# So any consecutive gap wider than _FILL_GAP_X surviving in the returned curve is an
# undigitized hole between disjoint liquid regions, which is what plotters break on.
_FILL_GAP_X = 0.06
_FILL_STEP_X = 0.03

# ---------------------------------------------------------------------------------------
# Lean records: an MPDS diagram reduced to what a fit and a liquidus plot need.
#
# A LEAN record drops ``shapes`` and carries the already-stitched liquidus under one
# reserved key. Kept: ``reference``, ``chemical_elements``, ``temp``, ``comp_range``,
# ``labels`` and the ``entry``/``jcode``/``year`` citation fields.
#
# A reserved KEY rather than a new class, so ``load_mpds_data``'s ``(dict, (curve,
# is_partial))`` shape is unchanged and every ``shapes`` consumer can detect the
# reduction (:func:`record_mode`).
#
# Stores the PRE-fill curve; callers densify at read time. Storing the filled curve would
# make ``liquidus_coverage`` report max_gap <= 0.06 for every system and silently disable
# the interior-sparsity gate in ``BinaryLiquid.from_cache``.
_GLIQUID_KEY = "_gliquid"

#: Bumped when the shape of the ``_gliquid`` block changes.
LEAN_SCHEMA = 1


def record_mode(mpds_json) -> str:
    """Classify an MPDS record: ``'full'``, ``'lean'`` or ``'empty'``.

    * ``'empty'`` -- the ``{"reference": None}`` placeholder ``load_mpds_data`` caches for
      a system MPDS has no digitized diagram for. Checked FIRST, so a placeholder is
      classified the same way in a lean store as in a full one and every consumer keeps its
      existing "no data" behaviour rather than starting to raise.
    * ``'lean'`` -- the reduction above: no ``shapes``, the stitched liquidus instead.
    * ``'full'`` -- an ordinary MPDS json.
    """
    if not isinstance(mpds_json, dict) or mpds_json.get("reference") is None:
        return "empty"
    block = mpds_json.get(_GLIQUID_KEY)
    if isinstance(block, dict) and block.get("mode") == "lean":
        return "lean"
    return "full"


def _require_full_record(mpds_json, caller: str, *, escape: str = "") -> None:
    """Raise unless ``mpds_json`` still carries its digitized ``shapes``.

    **The failure this prevents is not a crash.** ``identify_mpds_phases`` reads
    ``mpds_json.get("shapes", [])`` and would return ``[]`` on a lean record; an empty phase
    list makes :func:`assess_solid_coverage` report ZERO reported compounds, which reads as
    "nothing unsupported", and the solid-coverage gate PASSES. That is a silent wrong answer
    of exactly the class the gate exists to prevent, so a lean record must stop the caller
    rather than degrade it.

    There is deliberately **no auto-degrade path**. A fit run without invariant constraints
    and without the coverage gate is a *different fit*; returning one silently under the
    same call is the failure mode, not the mitigation.
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
    ``liquidus_coverage`` reproduce their full-record answers exactly.

    ``is_partial`` is **derived** at read time, never stored — storing it would let the two
    drift. The one thing derivation cannot recover from an empty ``covered`` is *why* it is
    empty, so the two empty cases are distinguished by ``covered`` itself: ``None`` means
    "no liquid field was digitized at all" (nothing was ever measured) and ``[]`` means
    "'L' shapes existed but none yielded a usable branch" — which is exactly the case
    ``_stitched_liquidus`` reports as partial. Both are faithful values of ``covered``, not
    a smuggled flag.

    ``chemical_elements`` is written **explicitly, ``None`` when genuinely unknown**:
    :func:`mpds_frame_matches` treats an ABSENT frame block as *matching* the caller, so a
    lean record that merely omitted it would silently mis-mirror every reversed-frame
    construction.
    """
    import gliquid.cache as _cache  # local: cache imports config only, mpds imports cache

    stitched, is_partial, covered = _stitched_liquidus(mpds_json, quiet=True)
    if stitched is None and not covered:
        covered = [] if is_partial else None
    header = _cache.mpds_header(mpds_json)
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

    The warnings are re-emitted so a lean store does not silently lose the "this system's
    liquidus is incomplete" signal. Only the disjoint-shape message differs: it can no
    longer name the branch COUNT (the branches are gone), so it names the covered regions,
    which is the part a reader acts on.
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
    normal case for the full-composition ``'(A, B)'`` labels (Os-Re, Sc-Zr, Hf-Zr) and also
    happens for a minority of ordinary compounds (Al-Cu ``'Cu11Al9 rt'``, Rb-Sn ``'RbSn2'``).
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
    Reading only the first reports a sliver: Ag-Fe spans 0.006 of the composition range
    instead of 1.0, Bi-Si 0.075, Er-Ta 0.169.

    The label is matched exactly, so the two-phase dome between the branches ('L1 + L2',
    'L + G', ...) is never mistaken for liquid field. A shape whose ``kind`` is something
    other than 'phase' is a drawing overlay or a mislabelled solid compound (five in the
    cached corpus, e.g. Hg-Pd, Li-Mo) and is likewise not liquid; ``kind``-less shapes in
    older cached jsons are kept.

    Raises:
        config.CacheModeError: on a LEAN record, which has no ``shapes``. Reachable only by
            a direct caller — ``_stitched_liquidus`` short-circuits before it.
    """
    _require_full_record(mpds_json, "liquid_shape_paths")
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
    not liquidus), split what is left into monotonic sections, and stitch the sections
    that line up with the pure-element endpoints onto the longest one. ``None`` when the
    shape has too few interior points to be a curve.
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

    return sorted(main_section)


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
            if tuple(pt) in seen:  # shapes that share a vertex must not double it up
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
    covers a composition interval no digitized shape reaches (Bi-Si: full span, ~0.86
    hole). A single-'L' diagram has one branch, so only the endpoint condition can fire.

    ``quiet`` suppresses the warnings so a metrics pass over a json that was already
    extracted does not double-count them in scanned logs.

    THIS IS THE ONE SEAM lean mode needs. Both ``extract_digitized_liquidus`` and
    ``liquidus_coverage`` route through here, so the early return below is what makes a
    lean record work end to end — and it is why the reduction stores the pre-fill curve:
    the two callers must keep seeing the same thing they always saw at this point,
    ``liquidus_coverage`` most of all.
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

    # If the liquidus does not have endpoints near the ends of the composition range, melting temps won't be good
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
    synthetic fill: a diagram drawing its liquid as separate wedges at 0-7 and 92-100 at.%
    still comes back spanning the full axis (Bi-Si class — and it is precisely the
    multi-shape envelope in ``_stitched_liquidus`` that makes such a diagram reach full
    span at all), even though the interior between the wedges is now left empty rather than
    fabricated. An endpoint-span check therefore over-reports such data as complete — this
    measures the stitched pre-fill curve instead:

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

    ``span``, ``n_points``, ``max_gap`` and ``covered_fraction`` are mirror-invariant —
    which is why a caller re-framing a system could historically carry the whole dict
    across untouched — but ``x_min``/``x_max`` and above all ``holes`` are POSITIONS, and
    ``x -> 1-x`` moves them. A hole carried across a frame flip unmirrored masks the
    opposite end of the diagram and does so silently: the metrics still look plausible.
    An involution, like :func:`mirror_liquidus`.
    """
    mirrored = dict(coverage)
    if "x_min" in coverage and "x_max" in coverage:
        mirrored["x_min"] = 1 - coverage["x_max"]
        mirrored["x_max"] = 1 - coverage["x_min"]
    mirrored["holes"] = [[1 - hi, 1 - lo] for lo, hi in reversed(coverage.get("holes") or [])]
    return mirrored


def load_mpds_data(input, pd_ind=None) -> tuple[dict, tuple[list[list] | None, bool]]:
    """Retrieves MPDS data for a binary system.

    Args:
        input (str or list): System specification (e.g., 'A-B' or ['A', 'B']).
        pd_ind (int | None): Index of the cached MPDS phase diagram to load.

    Returns:
        tuple[dict, tuple[list[list] | None, bool]]: The system MPDS JSON and the digitized
        liquidus curve formatted for fitting. Unary reference data no longer rides along —
        consumers take it from ``gliquid.phase.UNARY.component_data`` directly. Note that the
        MPDS json in the specified cache directory must follow the alphabetized, hyphenated
        naming convention (e.g. 'A-B.json')
    """
    components, _, _ = validate_and_format_system(input)
    if len(components) != 2:
        raise ValueError(
            f"MPDS binary phase-diagram retrieval needs exactly 2 components, got {components}."
        )
    # On-disk keys (and the MPDS query) canonicalize to the alphabetical name; the raw
    # json therefore stays in the alphabetical frame — consumers mirror at use.
    cache_name = api._canonical_sys_name(components)
    # Which diagram a pd_ind resolves to is a question about the RECORDS a store holds, not
    # about files in a directory, so it is asked of the backend. ``variant == ""`` is the
    # indexless ``<sys>.json`` naming; ``"0"``, ``"1"``, ... are the indexed diagrams.
    backend = cache.resolve_backend(None)
    available = set(backend.variants(cache_name, cache.KIND_MPDS))
    has_indexless = "" in available
    has_pd0 = "0" in available
    sys_dir = cache.store_label(backend, cache_name)

    if pd_ind is None:
        variant = "" if has_indexless else "0"
        if has_indexless and has_pd0:
            # Both namings present: the indexless file SHADOWS PD_0 and the caller almost
            # certainly meant the indexed one. Silent in a single-diagram store (no PD_0
            # sibling), which is where indexless is the legitimate convention.
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
                # An indexless-only store holds exactly one diagram, so pd_ind=0 names it.
                # Without this the requested-index path falls through to the API branch and
                # hands back {"reference": None} with no error — the mirror image of the
                # shadowing trap above, and it fires on the pin (pd_ind=0) recommended to
                # avoid that trap.
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
        # THE remote MPDS path. Guarded ABOVE the key check below, which is the whole
        # point: without a key this function logs a warning and returns
        # ``{"reference": None}``, a record shaped exactly like the real answer "MPDS holds
        # no digitized diagram for this system". Under offline mode that silence would be
        # indistinguishable from that fact, and a fit would proceed against a diagram
        # nobody ever looked for. So it raises instead (config.OfflineError).
        config.require_online(f"Fetching the MPDS phase diagram for '{cache_name}'")
        logger.info("No cached binary phase data found!")
        mpds_json = {"reference": None}
        if not api.get_api_key(api.MPDS_KEY_VAR):
            logger.warning(
                "MPDS_API_KEY not found in environment variables. Proceeding without binary phase data."
            )
            return mpds_json, (None, False)
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
        except api.mpds_api_error():
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
                    f"Caching binary phase data from entry at {dia_json['reference']['entry']} as {sys_file}..."
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
    """True when ``name`` is MPDS's ``'(X)'`` spelling of a pure component's own phase.

    Matches the bare label only (``'(Ti)'``), never the binary solid solution
    ``'(Ti, Cr)'`` -- callers pass ``shape['label'].split()[0]``, so an ``'(A, B)'`` label
    arrives as ``'(A,'`` and cannot collide.

    Deliberately name-based rather than composition-based. A terminal composition is NOT a
    usable discriminator: across the cache, the 0.005-0.03 / 0.97-0.995 bands are populated
    almost entirely by genuine line compounds at extreme stoichiometry (DyB66, LiC12,
    Al1.67B22), while real component phases are digitized anywhere from the frame edge out
    to x ~ 0.19 when the terminal solid solution has appreciable solubility ('(Mn)' at 0.186
    in Mn-Ti). Only the label separates the two.
    """
    return any(name == f"({el})" for el in (elements or ()))


def is_multiphase_field_label(label: str) -> bool:
    """True when an MPDS label names MORE than one phase -- a two-phase FIELD annotation.

    MPDS spells these as the constituent phases joined by ``'+'``, with a component
    phase's parentheses dropped: ``'HgZr3 + G'``, ``'Mn2B Mn + rt'`` (the figure prints
    ``'Mn2B + (Mn) rt'``), ``'CuSe2 rt + L'``, ``'Au3Zn Zn + rt'``. Such a shape is the
    field BETWEEN two phases, not a phase -- even though MPDS routinely tags it
    ``nphases: 1``, which is what lets it reach :func:`identify_mpds_phases` at all.

    The check is needed because a boundary line is named from
    ``shape['label'].split()[0]``, which silently promotes the first constituent into a
    phantom compound sitting at the FIELD's composition. Hg-Zr's Hg3Zr line at 25 at.%
    Zr came back as a second 'HgZr3' (the real one is the 75 at.% line), and B-Mn grew
    an 'Mn2B' at 80 at.% Mn beside the real one at 66.7 at.%. Both were confirmed
    against the source figures. Across the cache the pattern covers 23 shapes in 18
    systems; only the handful that collide with a correctly named shape are visible
    without this check -- the rest add a phantom compound no consumer can spot.

    A ``'+'`` anywhere in the label is the signal: no compound name or polymorph suffix
    in the cached corpus contains one.
    """
    return "+" in (label or "")


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

    Ranking is by label rather than by digitized temperature because the temperature is
    not trustworthy here: C-Ce draws BOTH CeC2 lines over the full temperature axis ~10 K
    apart, so the ht form's own bounds start LOWER than the rt form's (472.5 K vs 491.4 K)
    and any temperature-based ranking picks the wrong form as the low-temperature one.
    """
    for token in (suffix or "").split():
        match = _POLYMORPH_RE.match(token)
        if match:
            kind, index = match.group(1).lower(), match.group(2)
            return _POLYMORPH_BASE_RANK[kind] + (int(index) if index else 0)
    return _POLYMORPH_BASE_RANK[""]


def low_temp_threshold(mpds_json: dict) -> float:
    """Temperature (K) below which a digitized phase counts as stable at low temperature.

    The bottom of the diagram's own temperature axis plus 10% of its span. Relative rather
    than absolute because diagrams are digitized over wildly different ranges: a fixed
    cutoff would admit nearly everything in C-La's 0-4500 C frame and nearly nothing in
    C-Ce's 200-2600 C one.
    """
    lo, hi = mpds_json["temp"][0], mpds_json["temp"][1]
    return (lo + 273.15) + (hi - lo) * 0.10


def _merge_polymorph_cluster(
    base: str,
    cluster: list[dict],
    label_forms: list[tuple[int, float]],
    temp_threshold: float | None,
) -> dict:
    """Merge one compound's polymorph shapes into a single phase dict. See
    :func:`collapse_polymorphs` for the policy this implements."""
    ranks = [polymorph_rank(phase.get("polymorph", "")) for phase in cluster]
    lowest_rank = min(ranks)
    low = cluster[ranks.index(lowest_rank)]
    hottest = max(cluster, key=lambda phase: phase["tbounds"][1][1])

    merged = dict(low)
    merged["tbounds"] = [low["tbounds"][0], hottest["tbounds"][1]]
    distinct = len(set(ranks)) > 1
    # The low-temperature form's own ceiling: above it the liquidus is conjugate to a
    # hotter form. Degenerate (== the melting point) when the digitizer drew both lines
    # full height, which correctly leaves the whole flanking liquidus interval suspect.
    transition = low["tbounds"][1][1] if distinct else None

    # Recover a low-temperature form MPDS named in 'labels' but never digitized as its own
    # shape: C-La labels 'LaC2 rt' at 102 C while digitizing only the ht dome (bottom
    # 1355 K), so without this neither form reaches the low-temperature tables at all.
    if temp_threshold is not None and low["tbounds"][0][1] >= temp_threshold:
        cooler = sorted(
            (rank, temp)
            for rank, temp in label_forms
            if temp < temp_threshold and rank <= lowest_rank
        )
        if cooler:
            rank_lo, temp_lo = cooler[0]
            if rank_lo < lowest_rank:
                distinct = True
                transition = low["tbounds"][0][1]
            merged["tbounds"] = [[low["tbounds"][0][0], temp_lo], merged["tbounds"][1]]

    if len(cluster) > 1 or distinct:
        merged["name"] = base
    if distinct:
        merged["distinct_melting_polymorph"] = True
        merged["polymorph_transition_temp"] = transition
    return merged


def collapse_polymorphs(
    phases: list[dict],
    mpds_json: dict | None = None,
    *,
    temp_threshold: float | None = None,
    comp_tol: float = 0.02,
) -> list[dict]:
    """Collapse a compound's digitized polymorphs into one phase per compound.

    MPDS digitizes every polymorph of a compound as its own shape ('CeC2 rt' AND
    'CeC2 ht'), which double-counts the compound wherever a consumer treats one shape as
    one phase. C-Ce drew two CeC2 bars in the low-temperature phase comparison -- one
    classified congruent, one incongruent, overlapping into a composite color -- because
    :func:`identify_mpds_phases` strips the suffix from a line compound's name and the two
    landed in different tables.

    One entry survives per (compound, composition) cluster. It takes its identity and its
    lower temperature bound from the LOWEST-temperature polymorph -- the form stable at the
    bottom of the diagram, and the only one a 0 K DFT hull can hold -- and its upper bound
    from the compound's highest melting/decomposition temperature, so congruent-vs-
    incongruent classification still sees the real melting point. Entries sharing a name at
    DIFFERENT compositions are a separate digitization defect (Al-Dy labels DyAl2 at both
    0.333 and 0.666, Hg-Zr HgZr3 at 0.25 and 0.75) and are deliberately left alone rather
    than merged into a phantom phase.

    A cluster holding more than one thermal form gets ``'distinct_melting_polymorph': True``
    and ``'polymorph_transition_temp'``. Above that temperature the liquidus is conjugate to
    a form that is NOT the hull's ground state, so no solid free-energy reference exists for
    it -- ``BinaryLiquid.fit_parameters`` masks the flanking composition range on that basis.

    Args:
        phases (list): Phase dicts from ``identify_mpds_phases(..., with_polymorph=True)``.
            Without the ``'polymorph'`` key every entry ranks as unsuffixed, so same-named
            shapes still collapse but none is flagged as a distinct melting form.
        mpds_json (dict): Source json; only the ``labels`` block is read, to recover a
            low-temperature form named there but never digitized (the C-La case above).
        temp_threshold (float): Low-temperature cutoff in K, required for that recovery.
            Defaults to :func:`low_temp_threshold` of ``mpds_json``.
        comp_tol (float): Composition tolerance for treating two same-named shapes as the
            same compound.

    Returns:
        list: One phase dict per compound-composition cluster, sorted by composition.
    """
    if not phases:
        return list(phases)
    if temp_threshold is None and mpds_json:
        temp_threshold = low_temp_threshold(mpds_json)

    # Component phases ('(Ce) ht1' / '(Ce) ht2') are terminal solid solutions, not
    # compounds: downstream code keys them by their own labels and they never
    # double-count a compound. Pass them through untouched.
    passthrough, candidates = [], []
    for phase in phases:
        target = passthrough if split_phase_label(phase["name"])[0].startswith("(") else candidates
        target.append(dict(phase))

    label_forms: dict[str, list[tuple[int, float]]] = {}
    for entry in (mpds_json or {}).get("labels") or []:
        try:
            # Same rule the shapes went through, so the two blocks stay joinable: a
            # field annotation must not seed a thermal form for its first constituent.
            if is_multiphase_field_label(entry[0]):
                continue
            label_base, suffix = split_phase_label(entry[0])
            temp = float(entry[1][1]) + 273.15
        except (TypeError, ValueError, IndexError, KeyError):
            continue
        label_forms.setdefault(label_base, []).append((polymorph_rank(suffix), temp))

    groups: dict[str, list[dict]] = {}
    for phase in candidates:
        groups.setdefault(split_phase_label(phase["name"])[0], []).append(phase)

    collapsed = []
    for base, members in groups.items():
        members = sorted(members, key=lambda phase: phase["comp"])
        clusters = [[members[0]]]
        for phase in members[1:]:
            if abs(phase["comp"] - clusters[-1][0]["comp"]) <= comp_tol:
                clusters[-1].append(phase)
            else:
                clusters.append([phase])
        for cluster in clusters:
            collapsed.append(
                _merge_polymorph_cluster(base, cluster, label_forms.get(base, []), temp_threshold)
            )

    # 'polymorph' has served its purpose here. Dropping it keeps every phase dict on its
    # historical key set, so a compound with nothing to collapse stays byte-identical --
    # the characterization pins compare phase dicts by exact key set.
    result = sorted(collapsed + passthrough, key=lambda phase: phase["comp"])
    for phase in result:
        phase.pop("polymorph", None)
    return result


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

    * ``'ss'`` -- a digitized single-phase FIELD (an area shape), carrying ``cbounds``.
    * ``'comp'`` -- a pure COMPONENT's own phase, labelled ``'(X)'``, drawn as a boundary
      line rather than an area. Not a compound: its free energy is the element reference,
      which is always available. See :func:`is_component_phase_label`.
    * ``'lc'`` -- a line compound.

    ``'comp'`` exists because MPDS routinely digitizes a terminal phase as a
    ``kind='compound'`` boundary line -- Cr-Ti's ``'(Ti) rt'`` is literally the right-hand
    frame edge, ``M 100,600 L 100,1669.69 L 100,2000``. Typing that as a line compound made
    :func:`assess_solid_coverage` report it as a reported compound with no DFT counterpart
    and mask the liquidus from the nearest anchor out to the frame, skipping a system that
    fits fine. Consumers that only need "a low-temperature phase" treat ``'comp'``
    exactly like ``'lc'``/``'ss'``; only the coverage gate distinguishes them.

    Args:
        mpds_json (dict): MPDS digitized phase equilibrium data for the system.
        verbose (bool): If True, outputs additional debugging information.
        with_structure (bool): If True, add a ``'structure'`` key to every phase dict holding
            the parsed ``shape['phase']`` identifier (see :func:`parse_shape_structure`).
            Defaults to False, which keeps the returned dicts byte-identical to the historical
            key set -- the characterization pins compare phase dicts by exact key set, so this
            must stay opt-in. :func:`assess_solid_coverage` is the intended consumer.
        with_polymorph (bool): If True, add a ``'polymorph'`` key holding the label's
            polymorph suffix (``'rt'``, ``'ht1'``, ``''`` when unsuffixed). Opt-in for the
            same key-set reason as ``with_structure``. A line compound's ``'name'`` has the
            suffix stripped, so this is the only way to tell 'CeC2 rt' from 'CeC2 ht' after
            identification; :func:`collapse_polymorphs` is the intended consumer.
        elements: Component symbols used to recognize ``'(X)'`` component labels. Defaults to
            the json's own ``chemical_elements``; pass explicitly for the ~13% of cached jsons
            that carry no frame block, which would otherwise fall back to ``'lc'``. Only the
            SET is read, so frame order does not matter.

    Returns:
        list: A list of dictionaries containing information on equilibrium phase composition and temperature boundaries

    Raises:
        config.CacheModeError: on a LEAN record. This is the single most important guard in
            the lean-mode contract — see :func:`_require_full_record`.
    """
    _require_full_record(mpds_json, "identify_mpds_phases")
    if mpds_json.get("reference") is None:
        if verbose:
            logger.warning("System JSON does not contain any data!")
        return []

    if elements is None:
        elements = mpds_json.get("chemical_elements") or []

    phases = []
    for shape in mpds_json.get("shapes", []):
        if shape.get("nphases") == 1 and shape.get("is_solid") and "label" in shape:
            # A two-phase field MPDS tagged nphases=1. Naming it from split()[0] would
            # invent a compound at the field's composition. See is_multiphase_field_label.
            if is_multiphase_field_label(shape["label"]):
                if verbose:
                    logger.info(
                        f"Skipping {shape['label']!r}: label names more than one "
                        f"phase, so the shape is a field, not a phase"
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


_SG_TO_SS_NAME = {sg: name for name, sg in SS_SPACEGROUPS.items()}


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


@dataclass(frozen=True)
class PhaseCoverage:
    """Whether one digitized solid phase has a free energy behind it, and why."""

    name: str
    kind: str  # 'ss' | 'comp' | 'lc'
    comp: float
    interval: tuple[float, float]  # cbounds for 'ss'; the masked gap for a missing 'lc'
    supported: bool
    reason: str
    spacegroup: int | None = None
    pearson: str | None = None


@dataclass(frozen=True)
class SolidCoverageReport:
    """How much of a liquidus has no solid free-energy reference behind it.

    ``unsupported_fraction`` is the load-bearing number: the union of composition intervals
    whose conjugate solid we cannot evaluate, clipped to the digitized liquidus span and
    divided by it.

    The verdict is **not** a property of the system alone -- it depends on which energies were
    actually loaded, i.e. on ``config.solid_solutions`` / ``ss_ref_mode``. ``ss_models`` is kept
    on the report so a recorded verdict stays interpretable.
    """

    span: tuple[float, float]
    unsupported_fraction: float
    unsupported_ranges: list[list[float]]
    n_compounds: int
    n_missing_compounds: int
    phases: list[PhaseCoverage]
    ss_models: tuple[str, ...]
    thresholds: dict

    @property
    def missing_compound_fraction(self) -> float:
        return self.n_missing_compounds / self.n_compounds if self.n_compounds else 0.0

    def is_insufficient(self) -> tuple[bool, str]:
        """``(verdict, reason)`` against the thresholds captured on this report.

        Two commensurate conditions, either of which is disqualifying:

        1. Too much of the liquidus has no solid reference at all. Catches wide solid
           solutions with no solid-solution model (the Lu-Nd / Ho-Zr class).
        2. Too many of the reported compounds are absent from the DFT hull. Catches systems
           whose missing compounds each occupy a modest composition range but which
           collectively leave the hull unable to represent the solid side (Rb-Sn).
        """
        frac = self.unsupported_fraction
        if frac > self.thresholds["skip_frac"]:
            return True, (
                f"{frac:.0%} of the liquidus has no solid reference "
                f"(> {self.thresholds['skip_frac']:.0%} cap)"
            )
        missing_frac = self.missing_compound_fraction
        if (
            self.n_missing_compounds >= self.thresholds["min_missing"]
            and missing_frac >= self.thresholds["missing_frac"]
        ):
            return True, (
                f"{self.n_missing_compounds}/{self.n_compounds} reported compounds "
                f"have no DFT counterpart (>= {self.thresholds['missing_frac']:.0%} "
                f"of at least {self.thresholds['min_missing']})"
            )
        return False, (
            f"{frac:.0%} unsupported, "
            f"{self.n_missing_compounds}/{self.n_compounds} compounds missing"
        )

    @property
    def summary_line(self) -> str:
        unsupported = [p for p in self.phases if not p.supported]
        detail = (
            "; ".join(
                f"{p.name} [{p.interval[0]:.2f},{p.interval[1]:.2f}] {p.reason}"
                for p in unsupported
            )
            or "none"
        )
        return (
            f"unsupported {self.unsupported_fraction:.0%} of "
            f"[{self.span[0]:.2f},{self.span[1]:.2f}]; compounds missing "
            f"{self.n_missing_compounds}/{self.n_compounds}; "
            f"ss_models={list(self.ss_models)}; {detail}"
        )

    def as_dict(self) -> dict:
        """JSON-safe view for campaign run summaries."""
        return {
            "span": list(self.span),
            "unsupported_fraction": self.unsupported_fraction,
            "unsupported_ranges": [list(r) for r in self.unsupported_ranges],
            "n_compounds": self.n_compounds,
            "n_missing_compounds": self.n_missing_compounds,
            "ss_models": list(self.ss_models),
            "thresholds": dict(self.thresholds),
            "phases": [
                {
                    "name": p.name,
                    "kind": p.kind,
                    "comp": p.comp,
                    "interval": list(p.interval),
                    "supported": p.supported,
                    "reason": p.reason,
                    "spacegroup": p.spacegroup,
                    "pearson": p.pearson,
                }
                for p in self.phases
            ],
        }


def assess_solid_coverage(
    mpds_phases: list[dict],
    invariants,
    span,
    dft_comps,
    ss_models=(),
    *,
    ss_narrow_tol=None,
    dft_cover_tol=None,
    ss_rescue_max_width=None,
    thresholds=None,
) -> SolidCoverageReport:
    """Measure the fraction of a liquidus with no solid free-energy reference behind it.

    Every point on a liquidus is in equilibrium with some solid. That point constrains the
    liquid free energy only if the solid's free energy is known, so the honest measure of
    whether a system is fittable is *how much of the liquidus is conjugate to a solid we
    cannot evaluate*. This function computes exactly that, from data already extracted:
    the digitized phase fields, the identified invariants, the DFT hull compositions and the
    loaded solid-solution models.

    Support rules, per digitized solid phase:

    * A pure component's own phase (``type == 'comp'``, MPDS's ``'(X)'`` label) is always
      supported and is never counted as a reported compound -- the element reference always
      exists. Without this rule a terminal phase drawn as a boundary line reads as a line
      compound at x = 0 or 1, which ``covered_by_dft`` can never match because ``dft_comps``
      is interior-only by construction.
    * A solid-solution field no wider than ``ss_narrow_tol`` is not scored -- its endpoint
      line compound is an adequate stand-in.
    * A wider field whose spacegroup matches a loaded solid-solution model is supported.
    * A wider field whose spacegroup is one of the solid-solution structures but has **no**
      loaded model is unsupported over its whole composition range. A label-string check
      cannot see this: such a field may be labelled ``(A)`` yet span the entire axis.
    * A field whose structure MPDS did not resolve falls back to "supported if any model is
      loaded" -- this is what keeps genuine ``(A, B)`` solid solutions (Os-Re) fittable.
    * A field that is really an ordered compound with a homogeneity range (its spacegroup is
      not a solid-solution structure) is supported when a DFT compound sits inside it, but
      **only if it is narrower than** ``ss_rescue_max_width``. Without that cap a single
      interior DFT compound would rescue an arbitrarily wide field, and complete solid
      solutions with no models (Ag-Au, Ta-W, Se-Te, Bi-Sb, As-Sb) would score as fully
      supported -- the precise failure this gate exists to catch.
    * A line compound with no DFT phase within ``dft_cover_tol`` masks only its **primary
      crystallization field** -- the interval between the flanking invariant compositions --
      not everything out to the next supported anchor. Past a eutectic the conjugate solid is
      a different phase whose energy may be perfectly well known, and that stretch of
      liquidus is still informative. (This is what puts Mn-Y at 0.53 rather than 0.92.)

    Args:
        mpds_phases: ``identify_mpds_phases(..., with_structure=True)`` output.
        invariants: Identified invariant points; only ``'comp'`` is read. May be empty.
        span: ``[x_lo, x_hi]`` of the digitized liquidus.
        dft_comps: Interior (0 < x < 1) DFT hull phase compositions.
        ss_models: Names of loaded solid-solution models, e.g. ``('BCC', 'HCP')``.
        ss_narrow_tol: Override ``config.coverage_ss_narrow_tol`` — the width below which a
            solid-solution field is not scored.
        dft_cover_tol: Override ``config.coverage_dft_cover_tol`` — how near a DFT phase must
            sit to a line compound to count as covering it.
        ss_rescue_max_width: Override ``config.coverage_ss_rescue_max_width`` — the widest
            field an interior DFT compound may rescue.
        thresholds: Override the ``config`` decision thresholds; keys ``skip_frac``,
            ``min_missing``, ``missing_frac``.

    Returns:
        SolidCoverageReport

    Note:
        This function never sees an MPDS record — it takes ``mpds_phases``, already
        identified. It is therefore guarded one level up, at the two places that DO hold a
        json: :func:`identify_mpds_phases`, its only source of phases, and
        ``BinaryLiquid.assess_solid_coverage``, which names itself in the error. Handing a
        lean record's (impossible) empty phase list in here would report zero reported
        compounds, i.e. "nothing unsupported", and the gate would pass — see
        :func:`_require_full_record`.
    """
    ss_narrow_tol = config.coverage_ss_narrow_tol if ss_narrow_tol is None else ss_narrow_tol
    dft_cover_tol = config.coverage_dft_cover_tol if dft_cover_tol is None else dft_cover_tol
    if ss_rescue_max_width is None:
        ss_rescue_max_width = config.coverage_ss_rescue_max_width
    thresholds = (
        dict(thresholds)
        if thresholds
        else {
            "skip_frac": config.coverage_skip_frac,
            "min_missing": config.coverage_min_missing,
            "missing_frac": config.coverage_missing_frac,
        }
    )
    thresholds.update(
        {
            "ss_narrow_tol": ss_narrow_tol,
            "dft_cover_tol": dft_cover_tol,
            "ss_rescue_max_width": ss_rescue_max_width,
        }
    )

    ss_models = tuple(ss_models or ())
    dft_comps = sorted(float(c) for c in (dft_comps or []))
    x_lo, x_hi = (float(span[0]), float(span[1])) if span else (0.0, 1.0)
    width = max(x_hi - x_lo, 1e-9)

    inv_comps = sorted(
        iv["comp"]
        for iv in (invariants or [])
        if isinstance(iv, dict) and iv.get("comp") is not None
    )
    # Supported anchors bound a missing compound's field when no invariant flanks it: the pure
    # elements are always available, and any DFT compound is a valid solid reference.
    anchors = sorted({0.0, 1.0, *dft_comps})

    def covered_by_dft(comp):
        return next((c for c in dft_comps if abs(c - comp) <= dft_cover_tol), None)

    def flank_left(comp):
        return max(
            [c for c in inv_comps if c < comp - 1e-9],
            default=max([a for a in anchors if a < comp - 1e-9], default=0.0),
        )

    def flank_right(comp):
        return min(
            [c for c in inv_comps if c > comp + 1e-9],
            default=min([a for a in anchors if a > comp + 1e-9], default=1.0),
        )

    records, unsupported = [], []
    n_compounds = n_missing = 0

    for phase in mpds_phases:
        structure = phase.get("structure") or {}
        spacegroup, pearson = structure.get("spacegroup"), structure.get("pearson")

        if phase["type"] == "ss":
            lo, hi = phase["cbounds"][0][0], phase["cbounds"][1][0]
            field_width = hi - lo
            ss_name = _SG_TO_SS_NAME.get(spacegroup)

            if field_width <= ss_narrow_tol:
                supported, reason = True, "narrow_field"
            elif ss_name is not None:
                supported = ss_name in ss_models
                reason = f"{'' if supported else 'no_'}ss_model:{ss_name}"
                # Spacegroup alone is not a structure (e.g. NbC is 225 but rock salt, not FCC).
                # Surface the mismatch for audit; deliberately not used to decide.
                if supported and pearson and pearson not in ("cI2", "cF4", "hP2"):
                    reason += f" (pearson={pearson})"
            elif spacegroup is None:
                supported = bool(ss_models)
                reason = f"unknown_structure_{'ss_loaded' if supported else 'no_ss'}"
            elif field_width > ss_rescue_max_width:
                supported, reason = False, f"too_wide_to_rescue:sg{spacegroup}"
            else:
                near = next(
                    (c for c in dft_comps if lo - dft_cover_tol <= c <= hi + dft_cover_tol), None
                )
                supported = near is not None
                reason = f"dft_phase@{near:.3f}" if supported else f"no_ss_model:sg{spacegroup}"

            interval = (lo, hi)
            if not supported:
                unsupported.append([lo, hi])
        elif phase["type"] == "comp":
            # A pure component's own phase. Its solid reference is the element itself, which
            # is always available -- the same fact the `anchors` set above relies on. It is
            # not a reported compound, so it must not enter the n_missing_compounds count
            # either. Treating Cr-Ti's '(Ti) rt' frame edge as a missing line compound is
            # what masked [0.333, 1.0] and skipped a system that fits at MAE 11.9 K.
            supported, reason = True, "component_phase"
            interval = (phase["comp"], phase["comp"])
        else:
            n_compounds += 1
            comp = phase["comp"]
            near = covered_by_dft(comp)
            supported = near is not None
            if supported:
                reason, interval = f"dft_phase@{near:.3f}", (comp, comp)
            else:
                n_missing += 1
                reason = "no_dft_phase"
                interval = (flank_left(comp), flank_right(comp))
                unsupported.append(list(interval))

        records.append(
            PhaseCoverage(
                name=phase["name"],
                kind=phase["type"],
                comp=phase["comp"],
                interval=interval,
                supported=supported,
                reason=reason,
                spacegroup=spacegroup,
                pearson=pearson,
            )
        )

    merged = merge_ranges(unsupported)
    masked = sum(
        min(hi, x_hi) - max(lo, x_lo) for lo, hi in merged if min(hi, x_hi) > max(lo, x_lo)
    )

    return SolidCoverageReport(
        span=(x_lo, x_hi),
        unsupported_fraction=masked / width,
        unsupported_ranges=merged,
        n_compounds=n_compounds,
        n_missing_compounds=n_missing,
        phases=records,
        ss_models=ss_models,
        thresholds=thresholds,
    )


def get_low_temp_phase_data(
    mpds_json: dict, dft_ch: PhaseDiagram
) -> tuple[tuple[dict, dict, int | float], tuple[dict, dict, int | float]]:
    """Extracts low-temperature phase data from MPDS and MP convex hull.

    Pure-component phases (``type == 'comp'``, MPDS's ``'(X)'`` label) are NOT reported: the
    MPDS side of this comparison is about compounds that melt congruently or incongruently,
    and an element is neither. They are excluded from both tables and from
    ``max_phase_temp``. Wide ``'(X)'`` solid-solution fields are still reported -- those
    carry a real composition range, and :func:`identify_mpds_phases` has already dropped the
    terminal ones.

    Args:
        mpds_json (dict): MPDS digitized phase equilibrium data for the system.
        dft_ch (PhaseDiagram): DFT convex hull data formatted with pymatgen.

    Returns:
        Tuples with low temperature phase data for both digitzed and computed phases. The returned data is in the
        following format: (mpds congruently melting phases, mpds incongruently melting phases, max phase decomp temp),
        (dft phase formation energies, dft phase energies below convex hull, minimum phase formation energy)
    Raises:
        config.CacheModeError: on a LEAN record. Guarded here as well as inside
            :func:`identify_mpds_phases` so the message names the function the caller
            actually called — ``plotting/binary_figs.py``'s phase-comparison figure is the
            consumer, and an empty MPDS column there reads as a real finding.
    """
    _require_full_record(mpds_json, "get_low_temp_phase_data")

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

    mpds_congruent_phases, mpds_incongruent_phases = {}, {}
    max_phase_temp = 0

    # Polymorphs are collapsed to one entry per compound: without it C-Ce reports CeC2
    # twice (the rt line lands on the liquidus maximum and is tabled congruent, the ht line
    # misses it by 10 K and is tabled incongruent), and C-La reports LaC2 not at all.
    identified_phases = collapse_polymorphs(
        identify_mpds_phases(mpds_json, elements=api.pd_components(dft_ch), with_polymorph=True),
        mpds_json,
    )
    mpds_liquidus, _ = extract_digitized_liquidus(mpds_json)

    if not identified_phases:
        return (
            (mpds_congruent_phases, mpds_incongruent_phases, max_phase_temp),
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

    # Component phases are excluded: these tables are the MPDS side of a congruent- vs
    # incongruent-MELTING comparison against the DFT hull, and a pure element is neither.
    # Every call site already dropped '('-prefixed names by hand (binary_figs's
    # render_phase_comparison, both matrix plotters' mpds_lowt_phase_count); doing it here
    # also keeps max_phase_temp off them, which those hand filters could not -- a component
    # digitized as the frame edge spans the full temperature axis, so it pinned
    # max_phase_temp to the top of the diagram and squashed the real compound bars.
    # Wide '(X)' solid-solution FIELDS stay: identify_mpds_phases already drops the terminal
    # ones, so a surviving 'ss' has measurable solubility and a real composition range.
    for phase in identified_phases:
        if phase["type"] in {"lc", "ss"} and phase["tbounds"][0][1] < temp_threshold:
            if phase_decomp_on_liq(phase, mpds_liquidus):
                if phase["type"] == "ss":
                    mpds_congruent_phases[phase["name"]] = (
                        (phase["cbounds"][0][0], phase["cbounds"][1][0]),
                        phase["tbounds"][1][1],
                    )
                else:
                    mpds_congruent_phases[phase["name"]] = (
                        (phase["comp"], phase["comp"]),
                        phase["tbounds"][1][1],
                    )
            else:
                if phase["type"] == "ss":
                    mpds_incongruent_phases[phase["name"]] = (
                        (phase["cbounds"][0][0], phase["cbounds"][1][0]),
                        phase["tbounds"][1][1],
                    )
                else:
                    mpds_incongruent_phases[phase["name"]] = (
                        (phase["comp"], phase["comp"]),
                        phase["tbounds"][1][1],
                    )
            max_phase_temp = max(phase["tbounds"][1][1], max_phase_temp)

    if max_phase_temp == 0 and mpds_liquidus:
        max_phase_temp = min(mpds_liquidus, key=lambda x: x[1])[1]

    # The DFT side is already in the hull's own component frame; mirror the MPDS side
    # into it when the json was digitized in the reversed frame.
    hull_components = api.pd_components(dft_ch)
    if not mpds_frame_matches(mpds_json, hull_components):

        def _mirror_ranges(d):
            return {name: ((1 - c2, 1 - c1), t) for name, ((c1, c2), t) in d.items()}

        mpds_congruent_phases = _mirror_ranges(mpds_congruent_phases)
        mpds_incongruent_phases = _mirror_ranges(mpds_incongruent_phases)

    return (
        (mpds_congruent_phases, mpds_incongruent_phases, max_phase_temp),
        (dft_phases, dft_phases_ebelow, min_form_e),
    )


def print_phase_mismatch_chart(low_t_exp_phases: list[dict], dft_phase_comps) -> None:
    """Prints the low-temperature MPDS-vs-DFT phase-mismatch terminal chart.

    Lives at module level (called from ``BinaryLiquid.fit_parameters``, not from
    ``identify_invariant_points``) because the MP row needs DFT phase compositions,
    which the MPDS-only invariant-point identification deliberately does not consult.
    The scale bar is drawn on the fitting grid's 0.01 composition step (101 bins); the
    COMP legend below is hardcoded to the same width, so both would need re-drawing if
    the fitting grid ever changed.

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
    # print() by design: this function IS a console chart renderer (documented logging
    # exemption — the aligned monospace rows are the product, not diagnostics).
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
) -> tuple[list[dict], list[dict], bool]:
    """Identifies invariant points in the MPDS data from the digitized liquidus and JSON.

    This function does not consider DFT phases, which may differ in composition from the
    MPDS data. It requires both complete liquidus and JSON data for a binary system.
    (The DFT-vs-MPDS mismatch chart moved to ``print_phase_mismatch_chart``, called from
    ``BinaryLiquid.fit_parameters`` where DFT phases are in scope.)

    Args:
        mpds_json (dict): MPDS digitized phase equilibrium data for the system.
        components (list): Component names in the evaluation frame ([A, B]).
        digitized_liq (list): Digitized liquidus points as [x, T] pairs.
        component_data (dict): Per-component reference data; only ``.t_fusion`` is read.
        temp_range (list): Evaluation temperature range [min, max] in K; the max scales
            the composition-vs-temperature distance metric used to pair a miscibility-gap
            dome with its adjacent monotectic.
        verbose (bool): If True, outputs additional debugging information.
        check_full_ss (bool): If True, checks for full composition range solid solutions.
        t_tol (int): Temperature tolerance for invariant point identification.

    Returns:
        tuple: (invariant points, low-temperature phases from the MPDS JSON, full_comp_ss).
        full_comp_ss is True when the MPDS labels report a solid solution spanning the
        full composition range; identification stops early (solidus processing is not
        implemented) and the CALLER decides whether fitting can proceed — it can when
        solid-solution phase energies (ss_models) are available.

    Raises:
        config.CacheModeError: on a LEAN record. Invariant points ARE the digitized solid
            fields; without ``shapes`` there is nothing to identify, and an empty invariant
            list would silently remove every invariant constraint from a fit.
    """
    _require_full_record(
        mpds_json,
        "identify_invariant_points",
        escape="fit without invariant constraints by passing disable_inv_constrs=True",
    )
    if mpds_json["reference"] is None:
        logger.warning("System JSON does not contain any data!")
        return [], [], False

    # Identify phases from MPDS JSON. Polymorphs collapse to one entry per compound so a
    # compound is not counted twice, and so a compound whose low-temperature form MPDS
    # named but never digitized still reaches the tables (C-La's 'LaC2 rt').
    phases = collapse_polymorphs(
        identify_mpds_phases(mpds_json, elements=components, with_polymorph=True), mpds_json
    )
    if not mpds_frame_matches(mpds_json, components):
        phases = mirror_mpds_phases(phases)
    invariants = [
        phase for phase in phases if phase["type"] == "mig"
    ]  # Miscibility gaps are not phases.
    # They are also not really 'invariant points' either but we classify them as such for algorithm purposes.

    # Filter low-temperature phases
    low_t_exp_phases = [
        phase
        for phase in phases
        if (
            phase["type"] in ["lc", "ss", "comp"]
            and phase["tbounds"][0][1] < low_temp_threshold(mpds_json)
        )
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
        # Notify-only: identification stops here (solidus processing is not implemented),
        # but the caller decides whether fitting proceeds (it can with ss_models loaded).
        logger.warning(
            f"Full-composition solid solution detected in MPDS labels for "
            f"{components[0]}-{components[1]}; invariant identification stopped early."
        )
        return invariants, low_t_exp_phases, True

    def find_local_minima(points):
        """
        Args:
            points (list of tuples): List of (x, y) points.

        Returns:
            list: Local minima points.
        """

        def is_lt_prev(index):
            return index > 0 and points[index][1] < points[index - 1][1]

        local_minima = []
        current_section = []

        for i in range(len(points)):
            if is_lt_prev(i):
                current_section = [points[i]]
            elif current_section and current_section[-1][1] == points[i][1]:
                current_section.append(points[i])
            elif current_section:
                local_minima.append(current_section[len(current_section) // 2])
                current_section = []

        return local_minima

    def find_local_maxima(points):
        """
        Args:
            points (list of tuples): List of (x, y) points.

        Returns:
            list: Local maxima points.
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

    # Locate maxima and minima in liquidus
    maxima = find_local_maxima(digitized_liq)
    minima = find_local_minima(digitized_liq)

    # Assign congruent melting points
    if low_t_exp_phases:
        for coords in maxima[:]:
            low_t_exp_phases.sort(key=lambda x: abs(x["comp"] - coords[0]))
            phase = low_t_exp_phases[0]
            if (
                phase["type"] in ["lc", "ss", "comp"]
                and abs(phase["comp"] - coords[0]) <= 0.02
                and phase["tbounds"][1][1] + t_tol >= coords[1]
            ):
                phase["type"] = "cmp"
                invariants.append(
                    {
                        "type": phase["type"],
                        "comp": phase["comp"],
                        "temp": phase["tbounds"][1][1],
                        "phases": [phase["name"]],
                        "phase_comps": [phase["comp"]],
                    }
                )
                maxima.remove(coords)

    # Sort by descending temperature for peritectic identification
    low_t_exp_phases.sort(key=lambda x: x["tbounds"][1][1], reverse=True)

    def find_adj_phases(point: list | tuple) -> tuple[dict, dict]:
        """
        Finds adjacent phases near a given point.

        Args:
            point (list | tuple): A point in composition-temperature space.

        Returns:
            tuple: Two nearest adjacent phases.
        """
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

    # Identify liquid-liquid miscibility gap labels
    misc_gap_labels = []
    for label in mpds_json["labels"]:
        delim_label = label[0].split(" ")
        if len(delim_label) == 3 and delim_label[0][0] == "L" and delim_label[2][0] == "L":
            misc_gap_labels.append([label[1][0] / 100.0, label[1][1] + 273.15])

    # Process miscibility gap labels and find the nearest two-phase region which each corresponds to
    for mgl in misc_gap_labels:
        if len(maxima) < 1:
            break
        nearest_maxima = min(maxima, key=lambda x: abs(x[0] - mgl[0]))

        tbounds = [None, nearest_maxima]
        cbounds = None
        phases = None
        phase_comps = None

        for shape in mpds_json["shapes"]:
            # 'drawing' shapes (tie/construction lines) carry no nphases key in
            # native MPDS data (e.g. Mn-Pb), so .get() — this loop only became
            # reachable for such systems once multi-L extraction started
            # returning a liquidus for disjoint-L diagrams.
            if shape.get("nphases") != 2:
                continue
            data = shape_to_list(shape["svgpath"])
            if not data:
                continue
            data.sort(key=lambda x: x[1])
            if not (
                abs(data[-1][1] - nearest_maxima[1]) < t_tol
                and abs(data[-1][0] - nearest_maxima[0]) < 0.05
            ):
                continue
            tbounds = [data[0], data[-1]]
            data.sort(key=lambda x: x[0])
            if not data[0][0] < nearest_maxima[0] < data[-1][0]:
                continue
            cbounds = [data[0], data[-1]]
            break

        if len(minima) >= 1:
            # Adjacent monotectic should be a minima point at minimum distance in x-t space from the misc gap dome
            adj_mono = min(
                minima,
                key=lambda x: (
                    abs(tbounds[1][0] - x[0]) + 2 * (abs(tbounds[1][1] - x[1]) / temp_range[1])
                ),
            )
            tbounds[0] = [tbounds[1][0], adj_mono[1]]
            adj_phases = find_adj_phases(adj_mono)

            if adj_mono[0] < tbounds[1][0]:
                if adj_phases[0] is not None:
                    phase_comps = [adj_phases[0]["comp"]]
                    phases = [adj_phases[0]["name"]]
                if not cbounds:
                    lhs_ind = digitized_liq.index(adj_mono)
                    for i in range(lhs_ind + 1, len(digitized_liq) - 1):
                        if digitized_liq[i + 1][1] < adj_mono[1] <= digitized_liq[i][1]:
                            m = (digitized_liq[i + 1][1] - digitized_liq[i][1]) / (
                                digitized_liq[i + 1][0] - digitized_liq[i][0]
                            )
                            rhs_comp = (adj_mono[1] - digitized_liq[i][1]) / m + digitized_liq[i][0]
                            cbounds = [adj_mono, [rhs_comp, adj_mono[1]]]
                            break
            elif adj_mono[0] > tbounds[1][0]:
                if adj_phases[1] is not None:
                    phase_comps = [adj_phases[1]["comp"]]
                    phases = [adj_phases[1]["name"]]
                if not cbounds:
                    rhs_ind = digitized_liq.index(adj_mono)
                    for i in reversed(range(1, rhs_ind - 1)):
                        if digitized_liq[i - 1][1] < adj_mono[1] <= digitized_liq[i][1]:
                            m = (digitized_liq[i - 1][1] - digitized_liq[i][1]) / (
                                digitized_liq[i - 1][0] - digitized_liq[i][0]
                            )
                            lhs_comp = (adj_mono[1] - digitized_liq[i][1]) / m + digitized_liq[i][0]
                            cbounds = [[lhs_comp, adj_mono[1]], adj_mono]
                            break
            if cbounds and cbounds[0][1] != cbounds[1][1]:
                cbounds[0][1] = adj_mono[1]
                cbounds[1][1] = adj_mono[1]
            minima.remove(adj_mono)

        if cbounds:
            invariants.append(
                {
                    "type": "mig",
                    "comp": tbounds[1][0],
                    "cbounds": cbounds,
                    "tbounds": tbounds,
                    "phases": phases,
                    "phase_comps": phase_comps,
                }
            )
            maxima.remove(nearest_maxima)
        break

    stable_phase_comps = []

    # Main loop for peritectic phase identification
    for phase in low_t_exp_phases:
        if "(" in phase["name"]:  # Ignore component SS phases
            continue

        # Congruent melting points will not be considered for peritectic formation but will limit comp search range
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

        # Sort by increasing distance to liquidus to find the shortest distance
        endpoints.sort(key=lambda x: abs(x[0] - phase["comp"]))

        # Take the closest liquidus point to the phase as the peritectic point
        if endpoints:
            invariants.append(
                {
                    "type": "per",
                    "comp": endpoints[0][0],
                    "temp": phase_temp,
                    "phases": [phase["name"]],
                    "phase_comps": [phase["comp"]],
                }
            )

        stable_phase_comps.append(phase["comp"])

    # Identify eutectic points
    for coords in minima:
        adj_phases = find_adj_phases(coords)
        phases, phase_comps = zip(
            *[
                (None, None) if phase is None else (phase["name"], phase["comp"])
                for phase in adj_phases
            ]
        )

        invariants.append(
            {
                "type": "eut",
                "comp": coords[0],
                "temp": coords[1],
                "phases": list(phases),
                "phase_comps": list(phase_comps),
            }
        )

    invariants.sort(key=lambda x: x["comp"])
    invariants = [inv for inv in invariants if inv["type"] not in ["lc", "ss", "comp"]]
    if verbose:
        logger.info("--- Identified invariant points ---")
        for inv in invariants:
            logger.info(inv)
    return invariants, low_t_exp_phases, False
