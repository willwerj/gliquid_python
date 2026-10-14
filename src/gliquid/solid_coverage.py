"""The solid-energy coverage gate and the DFT-to-MPDS phase assignment, computed from the
diagram facts in :mod:`gliquid.mpds`.

:func:`assess_solid_coverage` measures how much of a binary system's digitized liquidus has a
solid free-energy reference behind it and returns a :class:`SolidCoverageReport` with one
:class:`PhaseCoverage` per digitized solid phase. :func:`assign_dft_phases` matches DFT
convex-hull phases to MPDS phases (:class:`DftMatch`); :func:`matched_dft_comps` and
:func:`rescuable_ss_indices` read that assignment.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, replace

import numpy as np

import gliquid.config as config
from gliquid.mpds import (
    FORMULA_TOL,
    SEGMENT_TOL,
    bounding_invariant_comps,
    compound_key,
    controlling_invariant,
    find_local_minima,
    flank_bound,
    formula_comp,
    is_ordered_field_label,
    is_unpaired_ht_form,
    merge_ranges,
    mig_feet,
    subtract_ranges,
    validated_mig_feet,
)
from gliquid.phase import SS_SPACEGROUPS

_SG_TO_SS_NAME = {sg: name for name, sg in SS_SPACEGROUPS.items()}


@dataclass(frozen=True)
class PhaseCoverage:
    """Whether one digitized solid phase has a free energy behind it, and why.

    ``supported`` and ``borders_liquid`` are separate facts, and BOTH are measured for every
    phase. ``supported`` says whether a free energy exists for this phase; ``borders_liquid``
    says whether the phase is in equilibrium with the liquid over the field it would own --
    a property of the digitized diagram, true or false regardless of what the hull holds, so
    a compound drawn far below the liquidus at its own composition is False even with a hull
    entry behind it. A phase it excludes carries a ``subliquidus:`` prefix on its ``reason``
    whether or not it is supported. Only a phase that is BOTH unsupported and liquidus-bordering
    contributes its ``interval`` to ``SolidCoverageReport.unsupported_ranges``; ``interval`` is
    recorded either way.
    """

    name: str
    kind: str  # 'ss' | 'comp' | 'lc'
    comp: float
    # cbounds for a supported 'ss' field; otherwise the masked field, the primary crystallization
    # field (for a solution field, cbounds joined with the liquidus its flanking invariants
    # control).
    interval: tuple[float, float]
    supported: bool
    reason: str
    spacegroup: int | None = None
    pearson: str | None = None
    borders_liquid: bool = True
    # The DFT hull phase(s) assign_dft_phases matched to this phase, how, and whether the space
    # groups agree; empty and None when none matched.
    dft_comps: tuple[float, ...] = ()
    dft_via: str | None = None
    dft_structure: str | None = None


@dataclass(frozen=True)
class SolidCoverageReport:
    """How much of a liquidus has no solid free-energy reference behind it.

    The composition axis is partitioned three ways. ``ignored_ranges`` carry no digitized
    liquidus at all -- outside the digitized endpoints, or inside a hole the extractor
    refused to bridge -- and are neither support nor a gap in support, because nothing can
    be scored against them. ``unsupported_ranges`` carry drawn liquidus whose conjugate
    solid has no evaluable free energy. What is left is supported, and
    ``supported_total`` -- its total width as a fraction of the WHOLE axis -- is what
    :meth:`is_insufficient` decides on.

    ``unsupported_fraction`` is the masked width divided by the digitized span, kept for
    provenance; it is a fraction of a different denominator than ``supported_total``.

    The verdict is **not** a property of the system alone -- it depends on which energies were
    actually loaded, i.e. on ``config.solid_solutions`` / ``ss_ref_mode``; ``ss_models`` on the
    report records which were loaded.
    """

    span: tuple[float, float]
    unsupported_fraction: float
    unsupported_ranges: list[list[float]]
    n_compounds: int
    n_missing_compounds: int
    phases: list[PhaseCoverage]
    ss_models: tuple[str, ...]
    thresholds: dict
    # None means only the span complement is ignored.
    ignored_ranges: list[list[float]] | None = None
    # Confirmed two-liquid domes with a foot on an unknown solid: neither masked nor supported.
    gap_ranges: list[list[float]] = field(default_factory=list)

    @property
    def missing_compound_fraction(self) -> float:
        return self.n_missing_compounds / self.n_compounds if self.n_compounds else 0.0

    @property
    def supported_fraction(self) -> float:
        """Fraction of the digitized span that keeps a solid free-energy reference."""
        return 1.0 - self.unsupported_fraction

    @property
    def ignored_intervals(self) -> list[tuple[float, float]]:
        """Composition intervals carrying no digitized liquidus, over the whole axis.

        Falls back to the complement of ``span`` when no explicit ranges were supplied, so
        a report built without hole information still partitions the axis correctly.
        """
        lo, hi = float(self.span[0]), float(self.span[1])
        if self.ignored_ranges is None:
            supplied = [[0.0, lo], [hi, 1.0]]
        else:
            supplied = [[float(a), float(b)] for a, b in self.ignored_ranges]
        return [
            (a, b)
            for a, b in merge_ranges([[max(0.0, a), min(1.0, b)] for a, b in supplied])
            if b - a > SEGMENT_TOL
        ]

    @property
    def ignored_total(self) -> float:
        """Total composition width with no digitized liquidus behind it."""
        return sum(b - a for a, b in self.ignored_intervals)

    @property
    def assessed_fraction(self) -> float:
        """Fraction of the composition axis that carries digitized liquidus."""
        return 1.0 - self.ignored_total

    @property
    def supported_segments(self) -> list[tuple[float, float]]:
        """Contiguous stretches of drawn liquidus left over once the mask is removed.

        ``unsupported_ranges`` arrives merged but unclipped -- a missing compound's masked
        field can run past the digitized endpoints -- so the ignored intervals are cut out
        alongside it, which clips to the span as a side effect.
        """
        cuts = sorted(
            (a, b)
            for a, b in (
                (max(float(r[0]), 0.0), min(float(r[1]), 1.0))
                for r in [
                    *(self.unsupported_ranges or []),
                    *self.ignored_intervals,
                    *(self.gap_ranges or []),
                ]
            )
            if b > a
        )
        segments, cursor = [], 0.0
        for a, b in merge_ranges([list(c) for c in cuts]):
            if a - cursor > SEGMENT_TOL:
                segments.append((cursor, a))
            cursor = max(cursor, b)
        if 1.0 - cursor > SEGMENT_TOL:
            segments.append((cursor, 1.0))
        return segments

    @property
    def largest_supported_segment(self) -> float:
        """Composition width of the widest surviving stretch; 0.0 when nothing survives."""
        return max((b - a for a, b in self.supported_segments), default=0.0)

    @property
    def gap_total(self) -> float:
        """Composition width held by domes that are neither masked nor supported."""
        return sum(b - a for a, b in merge_ranges([list(r) for r in (self.gap_ranges or [])]))

    @property
    def supported_total(self) -> float:
        """Total supported width, as a fraction of the whole composition axis."""
        return sum(b - a for a, b in self.supported_segments)

    @property
    def constrained_total(self) -> float:
        """Total width that constrains the liquid: supported liquidus plus confirmed domes.

        A dome is not solid-conjugate liquidus, so it is not in ``supported_segments``, but
        it is not absence of information either: ``BinaryLiquid._mig_constraint_equations``
        turns one into a curvature equation at the apex and a slope equation across the feet.

        Taken as a union rather than a sum: a dome may overlap a masked field, since the
        trim that removes a dome from the mask does not fire on every record.
        """
        spans = merge_ranges(
            [list(seg) for seg in self.supported_segments]
            + [list(r) for r in (self.gap_ranges or [])]
        )
        spans = subtract_ranges(spans, [list(i) for i in self.ignored_intervals])
        return sum(b - a for a, b in spans)

    @property
    def n_supported_segments(self) -> int:
        return len(self.supported_segments)

    def is_insufficient(self) -> tuple[bool, str]:
        """``(verdict, reason)`` against the thresholds captured on this report.

        One criterion, on the liquidus that SURVIVES the mask rather than on the mask: a
        system is insufficient when its TOTAL constraining width, as a fraction of the whole
        composition axis, is below the floor ``thresholds['min_supported_total'] +
        thresholds['ignored_floor_slope'] * ignored_total``.

        A confirmed two-liquid dome counts toward that total (:attr:`constrained_total`).
        It is not solid-conjugate liquidus, so it is not in ``supported_segments``, but it
        does constrain the liquid free energy and the fitter builds equations from it.

        ``min_supported_total = 0`` with ``ignored_floor_slope = 0`` admits everything, which
        is how a caller reaches an assessed-but-ungated report without turning assessment off.

        ``largest_supported_segment``, ``n_compounds`` and ``n_missing_compounds`` are on the
        report as provenance and do not gate.
        """
        total = self.constrained_total
        floor = (
            self.thresholds["min_supported_total"]
            + self.thresholds["ignored_floor_slope"] * self.ignored_total
        )
        dome = f", dome {self.gap_total:.3f}" if self.gap_total else ""
        if total < floor:
            return True, (
                f"constraining liquidus totals {total:.3f} of the composition axis "
                f"(< {floor:.3f} floor); supported {self.supported_total:.3f}{dome} across "
                f"{self.n_supported_segments} segment(s), widest "
                f"{self.largest_supported_segment:.3f}, ignored {self.ignored_total:.3f}"
            )
        return False, (
            f"constraining liquidus totals {total:.3f}; supported {self.supported_total:.3f}"
            f"{dome} across {self.n_supported_segments} segment(s), widest "
            f"{self.largest_supported_segment:.3f}, ignored {self.ignored_total:.3f}"
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
            f"supported {self.supported_total:.3f} / ignored {self.ignored_total:.3f} of "
            f"the axis; unsupported {self.unsupported_fraction:.0%} of "
            f"[{self.span[0]:.2f},{self.span[1]:.2f}]; widest supported segment "
            f"{self.largest_supported_segment:.3f} of {self.n_supported_segments}; "
            f"compounds missing {self.n_missing_compounds}/{self.n_compounds}; "
            f"ss_models={list(self.ss_models)}; {detail}"
        )

    def as_dict(self) -> dict:
        """JSON-safe dict view of the report."""
        return {
            "span": list(self.span),
            "unsupported_fraction": self.unsupported_fraction,
            "unsupported_ranges": [list(r) for r in self.unsupported_ranges],
            "supported_segments": [list(s) for s in self.supported_segments],
            "ignored_ranges": [list(r) for r in self.ignored_intervals],
            "gap_ranges": [list(r) for r in (self.gap_ranges or [])],
            "gap_total": self.gap_total,
            "supported_total": self.supported_total,
            "ignored_total": self.ignored_total,
            "largest_supported_segment": self.largest_supported_segment,
            "n_supported_segments": self.n_supported_segments,
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
                    "borders_liquid": p.borders_liquid,
                    "dft_comps": list(p.dft_comps),
                    "dft_via": p.dft_via,
                    "dft_structure": p.dft_structure,
                }
                for p in self.phases
            ],
        }


def _missing_compound_field(phase, comp, invariants, x_lo, x_hi, inv_comps, liq_minima, anchors):
    """``(interval, lo_rule, hi_rule)`` for a phase with no free energy behind it.

    Used for line compounds, ordered fields with a homogeneity range, and unsupported
    COMPONENT solution fields. The rules are returned unformatted; each caller writes the
    rest of its reason.
    """
    tiers = ((inv_comps, "invariant"), (liq_minima, "minimum"), (anchors, "endpoint"))
    melting_inv = controlling_invariant(phase, invariants)
    x_liq = None if melting_inv is None else float(melting_inv["comp"])
    # A stoichiometry outside the digitized span is not relocated.
    on_span = x_lo <= comp <= x_hi
    if (
        melting_inv is not None
        and melting_inv["type"] == "per"
        and on_span
        and abs(x_liq - comp) > 1e-9
    ):
        # The field runs from the peritectic liquid composition away from the stoichiometry.
        if x_liq < comp:
            lo_bound, lo_rule = flank_bound(
                x_liq, "left", tiers, tol=1e-9, default=(0.0, "endpoint")
            )
            hi_bound, hi_rule = x_liq, "peritectic"
        else:
            lo_bound, lo_rule = x_liq, "peritectic"
            hi_bound, hi_rule = flank_bound(
                x_liq, "right", tiers, tol=1e-9, default=(1.0, "endpoint")
            )
    else:
        lo_bound, lo_rule = flank_bound(comp, "left", tiers, tol=1e-9, default=(0.0, "endpoint"))
        hi_bound, hi_rule = flank_bound(comp, "right", tiers, tol=1e-9, default=(1.0, "endpoint"))
    return (lo_bound, hi_bound), lo_rule, hi_rule


def _spans_printed_gap(interval, gap_comps, gap_spans=()) -> bool:
    """Does a masked field run across a region where the diagram prints ``L1 + L2``?

    A printed label is what raises the question, so an empty ``gap_comps`` is always False.
    ``gap_spans`` then say WHERE the printed region is: an extent is what a field has to
    overlap, and a label's printed x decides only where no extent is available, because a
    leader arrow can print a label outside the lens it names.

    Args:
        interval: ``(lo, hi)`` the phase would mask.
        gap_comps: printed two-liquid label compositions.
        gap_spans: ``(lo, hi)`` extents of the printed two-liquid regions.

    Returns:
        bool: True when the field meets a printed two-liquid region.
    """
    if not gap_comps:
        return False
    lo, hi = float(interval[0]), float(interval[1])
    spans = [(float(a), float(b)) for a, b in (gap_spans or ())]
    if spans:
        return any(min(b, hi) - max(a, lo) > SEGMENT_TOL for a, b in spans)
    return any(lo - 1e-9 <= float(x) <= hi + 1e-9 for x in gap_comps)


def _phase_borders_liquid(phase, interval, liquidus, liq_t_min, subliquidus_margin, point_tol=0.0):
    """Whether the phase reaches the liquidus anywhere over ``interval``.

    The reference temperature is the lowest digitized liquidus temperature across
    ``interval`` -- the stretch this phase would mask. Where the digitization places no
    point inside ``interval``, the liquidus is INTERPOLATED at the phase's own
    composition, so a zero-width line compound is compared against the curve directly
    above it; that reference is lowered by ``point_tol``, since a curve value and a field
    top read at the SAME composition are two digitizations of one point. Only with no
    liquidus at all does ``liq_t_min`` stand in. The phase's ceiling is the highest of its
    ``tbounds`` temperatures.

    Args:
        phase: an :func:`gliquid.mpds.identify_mpds_phases` record, read for ``tbounds`` and
            ``comp``.
        interval: ``(lo, hi)`` composition bounds of the field.
        liquidus: digitized liquidus points.
        liq_t_min: lowest liquidus temperature across the whole diagram.
        subliquidus_margin: how far below the reference the phase may still count.
        point_tol: kelvin of digitization slack on an INTERPOLATED reference only.

    Returns:
        bool: True when the phase is within ``subliquidus_margin`` of the reference,
        and True whenever the data cannot answer -- no liquidus, or no ``tbounds``.
    """
    if liq_t_min is None:
        return True
    temps = [float(pt[1]) for pt in (phase.get("tbounds") or []) if pt]
    if not temps:
        return True
    lo, hi = float(interval[0]), float(interval[1])
    points = sorted((float(pt[0]), float(pt[1])) for pt in (liquidus or []) if pt)
    local = [t for x, t in points if lo - SEGMENT_TOL <= x <= hi + SEGMENT_TOL]
    if local:
        t_ref = min(local)
    elif points:
        comp = phase.get("comp")
        at = lo if comp is None else min(max(float(comp), lo), hi)
        xs = [x for x, _t in points]
        t_ref = float(np.interp(at, xs, [t for _x, t in points])) - float(point_tol)
    else:
        t_ref = liq_t_min
    return t_ref - max(temps) <= subliquidus_margin


def _field_below_ceiling(phase, interval, liquidus, field_ceiling_tol):
    """``interval`` narrowed to where the phase can actually be the primary solid.

    A phase is the first solid to freeze out of the liquid only where the liquidus is no
    hotter than the phase itself can exist. Past that the conjugate solid is something else,
    and the stretch beyond is not this phase's to mask.

    ``_phase_borders_liquid`` is the all-or-nothing form of this test; an empty result here
    is the same verdict it returns False for.

    The crossing is INTERPOLATED, not snapped to the nearest digitized point.

    Args:
        phase: an ``identify_mpds_phases`` record, read for ``tbounds``.
        interval: ``(lo, hi)`` the phase would otherwise mask.
        liquidus: digitized ``[x, T]`` pairs, in the phase's component frame.
        field_ceiling_tol: kelvin of slack, absorbing digitization error in the comparison.

    Returns:
        ``(lo, hi)`` narrowed, or None when the phase is nowhere below the liquidus.
    """
    lo, hi = float(interval[0]), float(interval[1])
    temps = [float(pt[1]) for pt in (phase.get("tbounds") or []) if pt]
    points = sorted((float(pt[0]), float(pt[1])) for pt in (liquidus or []) if pt is not None)
    if not temps or len(points) < 2 or hi <= lo:
        return (lo, hi)
    ceiling = max(temps) + float(field_ceiling_tol)

    xs = [x for x, _t in points]
    ts = [t for _x, t in points]
    if all(t <= ceiling for x, t in points if lo - SEGMENT_TOL <= x <= hi + SEGMENT_TOL):
        # Nothing inside is too hot; the ends still need checking against the curve.
        if float(np.interp(lo, xs, ts)) <= ceiling and float(np.interp(hi, xs, ts)) <= ceiling:
            return (lo, hi)

    def _qualifies(x):
        return float(np.interp(x, xs, ts)) <= ceiling

    # Walk the curve at the digitized resolution, refining each crossing by bisection.
    edges = sorted({lo, hi, *(x for x in xs if lo < x < hi)})
    keep = []
    for left, right in zip(edges, edges[1:]):
        for a, b in ((left, right),):
            qa, qb = _qualifies(a), _qualifies(b)
            if qa and qb:
                keep.append((a, b))
            elif qa != qb:
                low, high = (a, b) if qa else (b, a)
                for _ in range(40):
                    mid = 0.5 * (low + high)
                    if _qualifies(mid):
                        low = mid
                    else:
                        high = mid
                keep.append((min(a, low), max(a, low)) if qa else (min(low, b), max(low, b)))
    if not keep:
        return None
    return (max(lo, min(a for a, _b in keep)), min(hi, max(b for _a, b in keep)))


# Space groups that hold any structure; sharing one does not match two phases.
_UNINFORMATIVE_SPACEGROUPS = frozenset({1, 2})
_ASSIGN_MATCHED = 1e3
_ASSIGN_FORBIDDEN = 1e6
_ASSIGN_TIE = 1e-6
_ASSIGN_RANK = 1.0


@dataclass(frozen=True)
class DftMatch:
    """The DFT hull phase(s) :func:`assign_dft_phases` gives one MPDS phase.

    Attributes:
        hull: composition of the hull phase matched to the MPDS phase.
        extra: compositions of further hull phases on the same drawn range, which represent
            the MPDS phase together with ``hull``.
        via: ``'exact'`` (at the formula stoichiometry), ``'on'`` (inside the drawn range, or
            within ``dft_edge_tol`` of it) or ``'structure'`` (off both, same space group).
        structure: ``'match'``, ``'mismatch'`` or ``'unknown'``: the MPDS phase's space group
            against ``hull``'s, recorded whatever ``via`` is. Space groups 1 and 2 read
            ``'unknown'``.
        residual: composition distance from ``hull`` to the stoichiometry or drawn range.
    """

    hull: float
    extra: tuple[float, ...] = ()
    via: str = "on"
    structure: str = "unknown"
    residual: float = 0.0

    @property
    def comps(self) -> tuple[float, ...]:
        """Every hull composition behind the MPDS phase, ``hull`` first."""
        return (self.hull, *self.extra)


@dataclass(frozen=True)
class _Claim:
    formula: float | None
    lo: float
    hi: float
    comp: float
    spacegroup: int | None


def _spacegroup_of(phase) -> int | None:
    sg = (phase.get("structure") or {}).get("spacegroup")
    return None if sg is None else int(sg)


def _same_structure(sg_mpds, sg_dft) -> bool:
    """True when two space groups are known, equal, and neither is 1 or 2."""
    if sg_mpds is None or sg_dft is None:
        return False
    return int(sg_mpds) == int(sg_dft) and int(sg_mpds) not in _UNINFORMATIVE_SPACEGROUPS


def _structure_note(sg_mpds, sg_dft) -> str:
    if _same_structure(sg_mpds, sg_dft):
        return "match"
    if sg_mpds is None or sg_dft is None or int(sg_mpds) == int(sg_dft):
        return "unknown"
    return "mismatch"


def rescuable_ss_indices(mpds_phases, elements, ss_narrow_tol=None, ss_rescue_max_width=None):
    """Indices of the component solution fields only a DFT hull phase can support.

    A component solution field wider than ``ss_narrow_tol`` whose space group is known but is
    not a solid-solution structure, and which is no wider than ``ss_rescue_max_width``, is
    supported only by a hull phase on it (see :func:`assess_solid_coverage`). These fields
    take part in :func:`assign_dft_phases` alongside the compounds.

    Args:
        mpds_phases: ``identify_mpds_phases(..., with_structure=True)`` records.
        elements: the two component symbols.
        ss_narrow_tol: None uses ``config.coverage_ss_narrow_tol``.
        ss_rescue_max_width: None uses ``config.coverage_ss_rescue_max_width``.

    Returns:
        frozenset[int]: indices into ``mpds_phases``.
    """
    ss_narrow_tol = config.coverage_default("ss_narrow_tol", ss_narrow_tol)
    ss_rescue_max_width = config.coverage_default("ss_rescue_max_width", ss_rescue_max_width)
    out = set()
    for index, phase in enumerate(mpds_phases):
        if phase.get("type") != "ss" or is_ordered_field_label(phase.get("name"), elements):
            continue
        bounds = phase.get("cbounds") or []
        if len(bounds) < 2:
            continue
        width = float(bounds[1][0]) - float(bounds[0][0])
        spacegroup = _spacegroup_of(phase)
        if (
            ss_narrow_tol < width <= ss_rescue_max_width
            and spacegroup is not None
            and spacegroup not in _SG_TO_SS_NAME
        ):
            out.add(index)
    return frozenset(out)


def _dft_claims(mpds_phases, elements, unpaired_ht=frozenset(), rescuable=frozenset()):
    """``{index: _Claim}`` for the phases whose support IS a hull match.

    A component's own phase never consults the hull, and a component solution field only
    when it is in ``rescuable``. An unpaired high-temperature form has no ground state a
    0 K hull could hold, so it never claims a hull phase either.
    """
    claims = {}
    for index, phase in enumerate(mpds_phases):
        if phase.get("comp") is None:
            continue
        kind = phase.get("type")
        if kind == "comp" or is_unpaired_ht_form(phase, unpaired_ht):
            continue
        if (
            kind == "ss"
            and not is_ordered_field_label(phase.get("name"), elements)
            and index not in rescuable
        ):
            continue
        comp = float(phase["comp"])
        bounds = phase.get("cbounds") or []
        if kind == "ss" and len(bounds) >= 2:
            lo, hi = sorted((float(bounds[0][0]), float(bounds[1][0])))
        else:
            lo = hi = comp
        claims[index] = _Claim(
            formula=formula_comp(phase.get("name"), elements or []),
            lo=lo,
            hi=hi,
            comp=comp,
            spacegroup=_spacegroup_of(phase),
        )
    return claims


def _dft_eligibility(claim, x, spacegroup, edge_tol, structure_tol):
    """``(via, residual)`` when hull phase ``x`` may stand behind ``claim``, else None.

    A formula-labelled phase drawn at one composition is matched on its stoichiometry
    alone; any other phase also on its drawn range.
    """
    exact = abs(x - claim.formula) if claim.formula is not None else math.inf
    if exact <= FORMULA_TOL:
        return "exact", exact
    ranged = claim.formula is None or claim.hi > claim.lo
    drawn = max(claim.lo - x, x - claim.hi, 0.0) if ranged else math.inf
    if drawn <= edge_tol:
        return "on", drawn
    off = min(exact, drawn)
    if off <= structure_tol and _same_structure(claim.spacegroup, spacegroup):
        return "structure", off
    return None


def assign_dft_phases(
    mpds_phases,
    elements,
    dft_comps,
    dft_spacegroups=None,
    *,
    unpaired_ht=frozenset(),
    rescuable=frozenset(),
    edge_tol=None,
    structure_tol=None,
) -> dict[int, DftMatch]:
    """Which DFT hull phase stands behind which MPDS phase.

    Each hull phase serves at most one compound, and every polymorph of a compound
    (:func:`gliquid.mpds.compound_key`) shares its compound's hull phase. A hull phase may
    serve a compound when it sits at the compound's formula stoichiometry or on its drawn range
    (within ``edge_tol``), whatever either structure; or, off both by at most
    ``structure_tol``, when the two share a space group other than 1 and 2. A
    formula-labelled phase drawn at a single composition has no drawn range, only its
    stoichiometry.

    The assignment covers as many compounds as possible. Among those assignments it prefers,
    per match, a hull phase at the formula stoichiometry, then one on the drawn range sharing
    the MPDS phase's space group, then any other on the range, then one off it; then it
    minimises the summed composition residual, then prefers hull phases nearer each compound's
    formula (or drawn composition). A hull phase left over that sits on a covered compound's
    drawn range joins that compound as an ``extra``.

    Args:
        mpds_phases: ``identify_mpds_phases(..., with_structure=True)`` records, in the hull's
            component frame.
        elements: the two component symbols.
        dft_comps: interior hull compositions.
        dft_spacegroups: hull composition to space-group number, matched to ``dft_comps``
            within 1e-6; a composition absent or mapped to None has no known structure.
        unpaired_ht: :func:`gliquid.mpds.mark_unpaired_ht_forms` keys, in the records' frame.
        rescuable: :func:`rescuable_ss_indices` -- component solution fields that take part.
        edge_tol: None uses ``config.coverage_dft_edge_tol``.
        structure_tol: None uses ``config.coverage_dft_structure_tol``.

    Returns:
        dict: MPDS phase index to its :class:`DftMatch`; absent when unmatched.
    """
    from scipy.optimize import linear_sum_assignment

    edge_tol = float(config.coverage_default("dft_edge_tol", edge_tol))
    structure_tol = config.coverage_default("dft_structure_tol", structure_tol)
    structure_tol = float(structure_tol)
    known = [(float(k), sg) for k, sg in (dft_spacegroups or {}).items()]
    hull = [
        (float(x), next((sg for k, sg in known if abs(k - float(x)) <= 1e-6), None))
        for x in dft_comps or []
    ]
    claims = _dft_claims(mpds_phases, elements, unpaired_ht, rescuable)
    if not claims or not hull:
        return {}

    groups: dict[str, list[int]] = {}
    for index in claims:
        key = compound_key(mpds_phases[index].get("name") or "", elements or [])
        groups.setdefault(key or f"#{index}", []).append(index)
    keys = list(groups)

    best = {}  # (group, hull index) -> (rank, residual, tie, via)
    for g, key in enumerate(keys):
        for j, (x, spacegroup) in enumerate(hull):
            options = []
            for index in groups[key]:
                claim = claims[index]
                found = _dft_eligibility(claim, x, spacegroup, edge_tol, structure_tol)
                if found is not None:
                    ref = claim.formula if claim.formula is not None else claim.comp
                    rank = {"exact": 0, "on": 2, "structure": 3}[found[0]]
                    if found[0] == "on" and _same_structure(claim.spacegroup, spacegroup):
                        rank = 1
                    options.append((rank, found[1], abs(x - ref), found[0]))
            if options:
                best[g, j] = min(options)
    if not best:
        return {}

    n_groups, n_hull = len(keys), len(hull)
    cost = np.zeros((n_groups, n_hull + n_groups))
    cost[:, :n_hull] = _ASSIGN_FORBIDDEN
    for (g, j), (rank, residual, tie, _via) in best.items():
        cost[g, j] = -_ASSIGN_MATCHED + _ASSIGN_RANK * rank + residual + _ASSIGN_TIE * tie
    rows, cols = linear_sum_assignment(cost)
    primary = {int(g): int(j) for g, j in zip(rows, cols) if j < n_hull and (g, j) in best}

    extras: dict[int, list[int]] = {g: [] for g in primary}
    taken = set(primary.values())
    for j in range(n_hull):
        if j in taken:
            continue
        joins = [
            (*best[g, j][:3], g) for g in primary if (g, j) in best and best[g, j][3] != "structure"
        ]
        if joins:
            extras[min(joins)[-1]].append(j)

    out = {}
    for g, j in primary.items():
        _rank, residual, _tie, via = best[g, j]
        x, spacegroup = hull[j]
        extra = tuple(sorted(hull[k][0] for k in extras[g]))
        for index in groups[keys[g]]:
            out[index] = DftMatch(
                hull=x,
                extra=extra,
                via=via,
                structure=_structure_note(claims[index].spacegroup, spacegroup),
                residual=float(residual),
            )
    return out


def matched_dft_comps(
    mpds_phases,
    elements,
    dft_comps,
    *,
    dft_spacegroups=None,
    unpaired_ht=frozenset(),
):
    """The hull compositions a low-temperature MPDS phase stands behind.

    The complement of the returned set within ``dft_comps`` is the hull's unmatched
    vertices: compositions the DFT hull holds that no reported compound stands behind.
    The claims are :func:`assign_dft_phases`', including every ``extra``.

    Args:
        mpds_phases: digitized phase records, in the hull's component frame.
        elements: the two component symbols.
        dft_comps: interior hull compositions.
        dft_spacegroups: hull composition to space-group number.
        unpaired_ht: :func:`gliquid.mpds.mark_unpaired_ht_forms` keys, in the records' frame.

    Returns:
        set[float]: the claimed hull compositions, as given in ``dft_comps``.
    """
    rescuable = rescuable_ss_indices(mpds_phases, elements)
    assigned = assign_dft_phases(
        mpds_phases,
        elements,
        dft_comps,
        dft_spacegroups,
        unpaired_ht=unpaired_ht,
        rescuable=rescuable,
    )
    return {x for match in assigned.values() for x in match.comps}


@config._removed_keywords(
    {
        name: f"thresholds={{'{name}': ...}}"
        for name in ("ss_narrow_tol", "dft_cover_tol", "ss_rescue_max_width")
    }
)
def assess_solid_coverage(
    mpds_phases: list[dict],
    invariants,
    span,
    dft_comps,
    ss_models=(),
    *,
    liquidus=None,
    dft_spacegroups=None,
    thresholds=None,
    elements=None,
    ignored_ranges=None,
    printed_gap_comps=(),
    unpaired_ht=frozenset(),
) -> SolidCoverageReport:
    """Measure how much of a liquidus has no solid free-energy reference behind it.

    Every threshold named below is documented in :func:`gliquid.config.set_coverage_thresholds`.
    A pure component's own phase (``type == 'comp'``) is supported and is not a reported compound.

    * A field whose label is not a component solution field
      (:func:`gliquid.mpds.is_component_phase_label`) is an ordered intermetallic, scored on the
      compound path by formula and drawn range; its ``reason`` has an ``ordered_field:`` prefix.
    * A solid-solution field no wider than ``ss_narrow_tol`` is not scored. A wider one is
      supported when its spacegroup matches a loaded model, unsupported when it is a
      solid-solution structure with no model, and supported when MPDS did not resolve its
      structure and any model is loaded. A non-solid-solution structure is supported by a DFT
      phase only when narrower than ``ss_rescue_max_width``.
    * An unsupported component solution field masks its cbounds joined with the primary
      crystallization field its flanking invariants give it (``reason`` ``field:<left>/<right>``),
      extended at most ``ss_extension_cap`` on each side unless the drawn component fields cover
      ``ss_width_guard`` of the axis (``reason`` ``cap:<width>``).
    * A line compound or ordered field with no DFT phase from :func:`assign_dft_phases`
      (``dft_edge_tol``, ``dft_structure_tol``) masks its primary crystallization field.
    * A shape in ``unpaired_ht`` is unsupported, masks its primary crystallization field and is
      not a reported compound (``reason`` ``unpaired_ht_polymorph:<left>/<right>``).
    * A phase topping out more than ``subliquidus_margin`` K below the liquidus over its own
      field (``field_ceiling_tol`` K of slack) is ``borders_liquid=False`` and masks nothing.

    A primary crystallization field spans both ways from the stoichiometry for a congruent
    (``'cmp'``) or unmatched compound or one outside ``span``, and from a peritectic (``'per'``)
    liquid composition away from the stoichiometry. Each open side is bounded by the first of:
    ``invariant`` (nearest invariant composition), ``minimum`` (nearest local liquidus minimum),
    ``endpoint`` (nearest supported anchor, ``{0.0, 1.0}`` and ``dft_comps``); a peritectic side
    records ``peritectic``. The rules that fired appear as ``no_dft_phase:<left>/<right>``.

    Args:
        mpds_phases: ``identify_mpds_phases(..., with_structure=True)`` output, from a full record.
        invariants: Identified invariant points; may be empty.
        span: ``[x_lo, x_hi]`` of the digitized liquidus.
        dft_comps: Interior (0 < x < 1) DFT hull phase compositions.
        ss_models: Names of loaded solid-solution models, e.g. ``('BCC', 'HCP')``.
        liquidus: Digitized liquidus ``[x, T]`` pairs in the frame of ``span``; needed for the
            ``minimum`` bound and the sub-liquidus rule.
        dft_spacegroups: DFT hull composition to space-group number.
        thresholds: Per-call coverage knob overrides (:func:`gliquid.config.coverage_settings`).
            The report's ``thresholds`` records every knob in force plus ``printed_gap_comps``,
            ``ss_drawn_width`` and ``ss_guard_fired``.
        unpaired_ht: ``(base label, composition)`` keys from
            :func:`gliquid.mpds.mark_unpaired_ht_forms`, in the frame of ``mpds_phases``.
        printed_gap_comps: Compositions of printed two-liquid labels
            (:func:`gliquid.mpds.two_liquid_label_comps`), frame of ``span``. An unsupported
            field meeting a printed gap (its ``mig`` extent where known) is refused whole.
        ignored_ranges: ``[lo, hi]`` intervals with no digitized liquidus (default: outside ``span``).
        elements: Component symbols, to tell a component solution field from an ordered
            intermetallic. Omitted, a leading ``'('`` in the label decides.

    Returns:
        SolidCoverageReport

    Raises:
        ValueError: for an unknown or config-only knob in ``thresholds`` or a bad value.
        TypeError: for ``ss_narrow_tol``, ``dft_cover_tol`` or ``ss_rescue_max_width`` as keywords.
    """
    settings = config.coverage_settings(thresholds)
    thresholds = {**settings, "printed_gap_comps": [float(x) for x in printed_gap_comps or ()]}

    knobs = dict(
        mpds_phases=mpds_phases,
        invariants=invariants,
        span=span,
        dft_comps=dft_comps,
        ss_models=ss_models,
        liquidus=liquidus,
        elements=elements,
        ignored_ranges=ignored_ranges,
        thresholds=thresholds,
        ss_narrow_tol=settings["ss_narrow_tol"],
        dft_edge_tol=settings["dft_edge_tol"],
        dft_structure_tol=settings["dft_structure_tol"],
        dft_spacegroups=dft_spacegroups,
        ss_rescue_max_width=settings["ss_rescue_max_width"],
        subliquidus_margin=float(settings["subliquidus_margin"] or 0.0),
        mask_miscibility_gaps=settings["mask_miscibility_gaps"],
        mig_foot_tol=settings["mig_foot_tol"],
        printed_gap_comps=tuple(float(x) for x in printed_gap_comps or ()),
        bound_by_supported=settings["bound_by_supported"],
        field_ceiling_tol=settings["field_ceiling_tol"],
        unpaired_ht=frozenset(unpaired_ht or ()),
    )
    # Union pass first: the guard reads how much of the axis the masked drawn fields cover.
    report, drawn = _assess_solid_coverage_pass(extension_cap=None, **knobs)
    cap, guard = settings["ss_extension_cap"], settings["ss_width_guard"]
    cap_active = cap is not None and math.isfinite(cap)
    fired = bool(cap_active and guard is not None and drawn >= guard)
    thresholds["ss_drawn_width"] = float(drawn)
    thresholds["ss_guard_fired"] = fired
    if cap_active and not fired:
        report, _ = _assess_solid_coverage_pass(extension_cap=cap, **knobs)
    return report


def _assess_solid_coverage_pass(
    *,
    mpds_phases,
    invariants,
    span,
    dft_comps,
    ss_models,
    liquidus,
    elements,
    ignored_ranges,
    thresholds,
    ss_narrow_tol,
    dft_edge_tol,
    dft_structure_tol,
    dft_spacegroups,
    ss_rescue_max_width,
    subliquidus_margin,
    mask_miscibility_gaps,
    mig_foot_tol,
    printed_gap_comps,
    bound_by_supported,
    field_ceiling_tol,
    unpaired_ht,
    extension_cap,
) -> tuple[SolidCoverageReport, float]:
    """One assessment under a fixed extension cap; see :func:`assess_solid_coverage`.

    Args:
        extension_cap: The cap applied to every unsupported component solution field, or
            None for the full invariant-bounded extension.

    Returns:
        tuple[SolidCoverageReport, float]: the report, and the summed drawn width of the
        component solution fields it masked, which the width guard reads.
    """
    ss_models = tuple(ss_models or ())
    dft_comps = sorted(float(c) for c in (dft_comps or []))
    x_lo, x_hi = (float(span[0]), float(span[1])) if span else (0.0, 1.0)
    width = max(x_hi - x_lo, 1e-9)

    inv_comps = bounding_invariant_comps(invariants)
    # Supported anchors bound a missing compound's field where no invariant or liquidus minimum
    # flanks it.
    anchors = sorted({0.0, 1.0, *dft_comps})
    # Liquidus minima the invariant finder did not report as eutectics.
    liq_minima = sorted({round(float(p[0]), 9) for p in find_local_minima(list(liquidus or []))})

    liq_t_min = min((float(p[1]) for p in (liquidus or [])), default=None)
    # An infinite ceiling tolerance switches field narrowing off and the comparison to strict.
    point_tol = field_ceiling_tol if math.isfinite(float(field_ceiling_tol)) else 0.0

    # Confirmed two-liquid domes, trimmed from a missing compound's field.
    confirmed_gaps = (
        []
        if mask_miscibility_gaps
        else validated_mig_feet(invariants or [], liquidus, float(mig_foot_tol))
    )
    # Extents of the printed two-liquid regions not already owned by the dome trim, from the matched
    # lens; a foot on a pure element supplies no extent.
    _confirmed = {(round(lo, 9), round(hi, 9)) for lo, hi in confirmed_gaps}
    printed_gap_spans = [
        (lo, hi)
        for lo, hi in mig_feet(invariants or [])
        if 0.0 < lo < hi < 1.0 and (round(lo, 9), round(hi, 9)) not in _confirmed
    ]

    dft_assigned = assign_dft_phases(
        mpds_phases,
        elements,
        dft_comps,
        dft_spacegroups,
        unpaired_ht=unpaired_ht,
        rescuable=rescuable_ss_indices(mpds_phases, elements, ss_narrow_tol, ss_rescue_max_width),
        edge_tol=dft_edge_tol,
        structure_tol=dft_structure_tol,
    )

    records, unsupported = [], []
    drawn_width = 0.0
    masking = []  # (index into records, interval) for every phase that contributes a mask
    n_compounds = n_missing = 0

    for index, phase in enumerate(mpds_phases):
        structure = phase.get("structure") or {}
        spacegroup, pearson = structure.get("spacegroup"), structure.get("pearson")

        if is_unpaired_ht_form(phase, unpaired_ht):
            # No cooler form: unsupported over its primary field, and not a reported compound.
            comp = phase["comp"]
            supported = False
            interval, lo_rule, hi_rule = _missing_compound_field(
                phase, comp, invariants, x_lo, x_hi, inv_comps, liq_minima, anchors
            )
            reason = f"unpaired_ht_polymorph:{lo_rule}/{hi_rule}"
        elif phase["type"] == "ss" and is_ordered_field_label(phase["name"], elements):
            # An ordered intermetallic drawn as an area: scored on the compound path.
            n_compounds += 1
            lo, hi = phase["cbounds"][0][0], phase["cbounds"][1][0]
            comp = phase["comp"]
            near = dft_assigned.get(index)
            supported = near is not None
            if supported:
                reason, interval = f"ordered_field:dft_phase@{near.hull:.3f}", (lo, hi)
            else:
                n_missing += 1
                field, lo_rule, hi_rule = _missing_compound_field(
                    phase, comp, invariants, x_lo, x_hi, inv_comps, liq_minima, anchors
                )
                # Union of the drawn field with the invariant-bounded one where the two meet.
                interval = (
                    (min(lo, field[0]), max(hi, field[1]))
                    if min(hi, field[1]) >= max(lo, field[0]) - SEGMENT_TOL
                    else field
                )
                reason = f"ordered_field:no_dft_phase:{lo_rule}/{hi_rule}"
        elif phase["type"] == "ss":
            lo, hi = phase["cbounds"][0][0], phase["cbounds"][1][0]
            comp = phase["comp"]
            field_width = hi - lo
            ss_name = _SG_TO_SS_NAME.get(spacegroup)

            if field_width <= ss_narrow_tol:
                supported, reason = True, "narrow_field"
            elif ss_name is not None:
                supported = ss_name in ss_models
                reason = f"{'' if supported else 'no_'}ss_model:{ss_name}"
                # Spacegroup mismatch, recorded for audit only.
                if supported and pearson and pearson not in ("cI2", "cF4", "hP2"):
                    reason += f" (pearson={pearson})"
            elif spacegroup is None:
                supported = bool(ss_models)
                reason = f"unknown_structure_{'ss_loaded' if supported else 'no_ss'}"
            elif field_width > ss_rescue_max_width:
                supported, reason = False, f"too_wide_to_rescue:sg{spacegroup}"
            else:
                near = dft_assigned.get(index)
                supported = near is not None
                reason = (
                    f"dft_phase@{near.hull:.3f}" if supported else f"no_ss_model:sg{spacegroup}"
                )

            interval = (lo, hi)
            if not supported:
                # The masked field is the drawn field joined with the liquidus it controls up to its
                # flanking invariants.
                field, lo_rule, hi_rule = _missing_compound_field(
                    phase, comp, invariants, x_lo, x_hi, inv_comps, liq_minima, anchors
                )
                capped = False
                if extension_cap is not None:
                    clipped = (
                        max(field[0], lo - extension_cap),
                        min(field[1], hi + extension_cap),
                    )
                    capped = clipped != tuple(field)
                    field = clipped
                extended = (min(lo, field[0]), max(hi, field[1]))
                if extended != interval:
                    interval = extended
                    reason = f"{reason} field:{lo_rule}/{hi_rule}"
                if capped:
                    reason = f"{reason} cap:{extension_cap:g}"
        elif phase["type"] == "comp":
            # A pure component's own phase: supported by the element, not a reported compound.
            supported, reason = True, "component_phase"
            interval = (phase["comp"], phase["comp"])
        else:
            n_compounds += 1
            comp = phase["comp"]
            near = dft_assigned.get(index)
            supported = near is not None
            if supported:
                reason, interval = f"dft_phase@{near.hull:.3f}", (comp, comp)
            else:
                n_missing += 1
                interval, lo_rule, hi_rule = _missing_compound_field(
                    phase, comp, invariants, x_lo, x_hi, inv_comps, liq_minima, anchors
                )
                reason = f"no_dft_phase:{lo_rule}/{hi_rule}"

        # Whether the phase ever borders the liquid, measured for every phase; only the masking
        # below depends on ``supported``.
        touches_liquid = _phase_borders_liquid(
            phase, interval, liquidus, liq_t_min, subliquidus_margin, point_tol
        )
        if not supported:
            # An infinite subliquidus_margin switches the liquid-contact criterion off.
            reachable = (
                (float(interval[0]), float(interval[1]))
                if touches_liquid and subliquidus_margin == float("inf")
                else _field_below_ceiling(phase, interval, liquidus, field_ceiling_tol)
                if touches_liquid
                else None
            )
            if reachable is None:
                touches_liquid = False
            elif _spans_printed_gap(reachable, printed_gap_comps, printed_gap_spans):
                # Inside a printed liquid-liquid gap: all or nothing, as the walk does.
                reason = f"{reason} printed_gap"
            else:
                if reachable != (float(interval[0]), float(interval[1])):
                    reason = f"{reason} ceiling"
                unsupported.append([reachable[0], reachable[1]])
                masking.append((len(records), reachable))
                if phase["type"] == "ss" and not is_ordered_field_label(phase["name"], elements):
                    drawn_width += float(phase["cbounds"][1][0]) - float(phase["cbounds"][0][0])

        if not touches_liquid:
            reason = f"subliquidus:{reason}"

        match = dft_assigned.get(index)
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
                borders_liquid=touches_liquid,
                dft_comps=() if match is None else match.comps,
                dft_via=None if match is None else match.via,
                dft_structure=None if match is None else match.structure,
            )
        )

    # An interior hull entry with no digitized phase is unevidenced; an empty hull stays supported.
    if not records and any(0.0 < float(comp) < 1.0 for comp in (dft_comps or [])):
        unsupported = [[x_lo, x_hi]]

    # The stretch past a bounding phase inside the field is not masked by this phase.
    if bound_by_supported:
        anchors = [rec for rec in records if _may_bound(rec)]
        if anchors:
            trimmed, cut = [], set()
            for idx, before in masking:
                lo, hi = before
                rec = records[idx]
                home = _home_end(before, rec.comp)
                for anchor in anchors:
                    if lo < anchor.comp < hi:
                        if anchor.comp < home:
                            lo = max(lo, anchor.comp)
                        else:
                            hi = min(hi, anchor.comp)
                if (lo, hi) != before:
                    cut.add(idx)
                if hi - lo > SEGMENT_TOL:
                    trimmed.append([lo, hi])
            unsupported = trimmed
            for idx in cut:
                records[idx] = replace(records[idx], reason=f"{records[idx].reason} bounded")

    merged = merge_ranges(unsupported)
    # A dome leaves the mask only when both its feet rest on supported liquidus.
    anchored, floating = [], []
    for lo, hi in confirmed_gaps:
        target = anchored if not any(_in_ranges(x, merged) for x in (lo, hi)) else floating
        target.append([lo, hi])
    if anchored:
        merged = subtract_ranges(merged, anchored)
    if confirmed_gaps:
        records = [
            replace(rec, reason=f"{rec.reason} mig_trimmed")
            if not rec.supported
            and rec.borders_liquid
            and _overlaps_any(rec.interval, confirmed_gaps)
            else rec
            for rec in records
        ]
    masked = sum(
        min(hi, x_hi) - max(lo, x_lo) for lo, hi in merged if min(hi, x_hi) > max(lo, x_lo)
    )

    report = SolidCoverageReport(
        span=(x_lo, x_hi),
        unsupported_fraction=masked / width,
        unsupported_ranges=merged,
        n_compounds=n_compounds,
        n_missing_compounds=n_missing,
        phases=records,
        ss_models=ss_models,
        thresholds=thresholds,
        ignored_ranges=(
            None if ignored_ranges is None else [[float(a), float(b)] for a, b in ignored_ranges]
        ),
        gap_ranges=merge_ranges(floating),
    )
    return report, drawn_width


# Support reasons that rest on an actual free energy.
_ENERGY_BACKED = ("dft_phase@", "ss_model:", "ordered_field:dft_phase@")


def _rests_on_free_energy(reason) -> bool:
    """True when a phase's SUPPORT comes from an evaluable free energy.

    ``narrow_field``, ``component_phase`` and ``unknown_structure_ss_loaded`` are support
    verdicts -- they say the phase need not be scored, not that its energy is known -- so a
    phase supported by one of them cannot decide how much liquidus its neighbour owns.
    """
    return (reason or "").split(" ")[0].startswith(_ENERGY_BACKED)


def _may_bound(record) -> bool:
    """Whether a phase is entitled to bound a neighbour's primary crystallization field.

    Two conditions: the phase must rest on a real free energy,
    and it must actually meet the liquid: a solid whose whole field lies below the liquidus
    is never the conjugate of any point on it and owns none of it.
    """
    return (
        record.supported
        and record.borders_liquid
        and 0.0 < record.comp < 1.0
        and _rests_on_free_energy(record.reason)
    )


def _home_end(interval, comp):
    """The end of ``interval`` the phase itself sits at.

    A congruent field spans both ways from the stoichiometry, so ``comp`` is inside it. A
    peritectic field runs from the invariant AWAY from the stoichiometry, leaving ``comp``
    outside -- there the phase's own end is the nearer one.
    """
    lo, hi = float(interval[0]), float(interval[1])
    if lo <= comp <= hi:
        return comp
    return lo if abs(comp - lo) <= abs(comp - hi) else hi


def _in_ranges(x, ranges):
    """True when ``x`` lies inside any ``[lo, hi]`` of ``ranges``."""
    return any(lo - SEGMENT_TOL <= x <= hi + SEGMENT_TOL for lo, hi in ranges)


def _overlaps_any(interval, ranges):
    """True when ``interval`` shares composition width with any of ``ranges``."""
    lo, hi = float(interval[0]), float(interval[1])
    return any(min(hi, b) - max(lo, a) > SEGMENT_TOL for a, b in ranges)
