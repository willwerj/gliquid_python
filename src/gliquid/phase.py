"""
Phase representation and unary reference-state data for the gliquid package.

``Phase`` is the generalized phase representation for ANY number of components:
fixed-stoichiometry phases (elemental polymorphs, line compounds) carry a pymatgen
``Composition`` plus scalar energetics; solution phases (liquid, gas, solid
solutions) have ``composition=None`` and evaluate through an attached solution
model. The chemistry lives in the Composition — evaluation axes (e.g. the binary
x_B) are derived via ``fraction_in``.

The module also exposes the single ``UNARY`` registry, loaded from
``phase_transitions.json``: for each element the ordered phase sequence (solid
polymorphs -> liquid -> gas) is stored as ``Phase`` objects whose cumulative
enthalpy/entropy are relative to the DFT ground state (H = 0, S = 0).
``ComponentRef`` gives both the scalar liquid reference used throughout
binary/ternary fitting (``h_liq``, ``s_liq``, ``t_fusion``, ``t_vaporization``,
``polymorphs``) and the per-phase references needed for solid-solution modelling
(``solid_phase``, ``liquid_ref_from_solids``).
"""

from __future__ import annotations

import json
import os
import warnings
from dataclasses import dataclass, field, replace

from pymatgen.core import Composition, DummySpecies, Element

import gliquid.config as config

# Spacegroup numbers / symbols for the solid-solution phases (single source of truth).
SS_SPACEGROUPS = {"BCC": 229, "FCC": 225, "HCP": 194}
SS_SYMBOLS = {"BCC": "Im-3m", "FCC": "Fm-3m", "HCP": "P6_3/mmc"}

# eV/atom -> J/mol conversion used wherever DFT energies enter molar thermodynamics.
EV_ATOM_TO_J_MOL = 96485.0


@dataclass
class Phase:
    """One thermodynamic phase of an n-component system (n >= 1).

    Fixed-stoichiometry phases (elemental polymorphs, line compounds) carry a
    ``composition`` and scalar energetics; solution phases (liquid, gas, solid
    solutions) have ``composition=None`` — their energetics live in ``model``
    (a ``gliquid.solution.SolutionModel``; typed loosely here so the reference
    layer never imports the solution layer). Elemental ladder phases additionally
    carry the step/transition fields, relative to the element's DFT ground state
    (H = 0, S = 0).
    """

    phase_type: str  # 'solid' | 'liquid' | 'gas'
    name: str | None = None
    composition: Composition | None = None  # None => variable-composition solution phase
    t_transition: float | None = None  # transition_temperature_K
    enthalpy: float | None = None  # cumulative enthalpy_J_per_mol
    entropy: float | None = None  # cumulative entropy_J_per_mol_K
    delta_h: float | None = None  # step delta_H_J_per_mol
    delta_s: float | None = None  # step delta_S_J_per_mol_K
    spacegroup_number: int | None = None
    spacegroup_symbol: str | None = None
    material_id: str | None = None
    imputed: bool = False  # phase imputed rather than DFT-computed
    source: str | None = None  # provenance label (e.g. lattice-stability origin)
    energy_per_atom_ev: float | None = None  # raw DFT energy (lattice-stability entries)
    points: list = field(default_factory=list)  # computed boundary points [x, T]
    model: object | None = None  # SolutionModel for solution phases

    @property
    def is_solution(self) -> bool:
        """True for variable-composition phases (liquid, gas, solid solutions)."""
        return self.composition is None

    def gibbs(self, temp_k: float) -> float:
        """G = H - T*S for this phase (J/mol), relative to the reference frame."""
        return (self.enthalpy or 0.0) - temp_k * (self.entropy or 0.0)

    def fraction_in(self, components) -> tuple[float, ...]:
        """This phase's fractional composition along a system's evaluation axes.

        Returns the n-1 independent atomic fractions (components[1:] in order;
        the first component's fraction is the derived remainder), e.g. ``(x_B,)``
        for a binary and ``(x_B, x_C)`` for a ternary. Solution phases have no
        fixed composition and raise.
        """
        if self.composition is None:
            raise ValueError(
                f"Phase '{self.name}' is a solution phase with no fixed "
                f"composition; evaluate its model over the grid instead."
            )
        comps = list(components)
        for c in comps:
            try:
                Element(c)
            except Exception as exc:
                # components[0] is the implicit remainder, so an atomic-fraction result
                # over elemental components[1:] would be silently WRONG for a compound
                # end-member axis — guard every component, not just the queried ones.
                raise NotImplementedError(
                    f"Phase.fraction_in supports elemental components only (got {comps}); "
                    "compound end-member axes will be supported in a future release."
                ) from exc
        return tuple(self.composition.get_atomic_fraction(c) for c in comps[1:])

    @classmethod
    def from_json(cls, phase: dict, composition: Composition | None = None) -> Phase:
        return cls(
            phase_type=phase.get("phase_type"),
            name=phase.get("common_name"),
            composition=composition,
            t_transition=phase.get("transition_temperature_K"),
            enthalpy=phase.get("enthalpy_J_per_mol"),
            entropy=phase.get("entropy_J_per_mol_K"),
            delta_h=phase.get("delta_H_J_per_mol"),
            delta_s=phase.get("delta_S_J_per_mol_K"),
            spacegroup_number=phase.get("spacegroup_number"),
            spacegroup_symbol=phase.get("spacegroup_symbol"),
            material_id=phase.get("materials_project_id"),
        )


@dataclass
class ComponentRef:
    """Reference states of a single system component.

    Today a component is a pure element; a parseable multi-element formula raises
    NotImplementedError until compound (pseudo-binary) end-members land. Unparseable or
    dummy symbols are tolerated so unknown-symbol lookups keep the silent-empty-registry
    contract (``UNARY['Xx']`` returns an all-zero reference).
    """

    symbol: str
    phases: list[Phase] = field(default_factory=list)
    # Tier B solid-solution end-member references (BCC/FCC/HCP structures NOT on
    # the phase ladder): cumulative enthalpy above the element anchor, no transition
    # temperature, and S = 0 except for the builder's scoped metastable-entropy
    # exception (a few HCP entries carry a negative SGTE entropy; see the database's
    # 'metastable_entropy_exception' convention). Deliberately separate from
    # ``phases`` so they never enter ``polymorphs`` (hence never the hull line
    # compounds), ``solid_phase`` lookups, or liquid reconstruction.
    lattice_stabilities: list[Phase] = field(default_factory=list)

    def __post_init__(self):
        try:
            comp = Composition(self.symbol)
        except Exception:
            return  # unparseable symbols keep the silent-empty contract
        if any(isinstance(el, DummySpecies) for el in comp.elements):
            return  # dummy species keep the silent-empty contract
        if len(comp.elements) > 1:
            raise NotImplementedError(
                f"Compound component reference states are not yet supported (got "
                f"'{self.symbol}'); they will be supported in a future release."
            )

    def _last(self, phase_type: str) -> Phase | None:
        return next((p for p in reversed(self.phases) if p.phase_type == phase_type), None)

    @property
    def liquid(self) -> Phase | None:
        return self._last("liquid")

    @property
    def gas(self) -> Phase | None:
        return self._last("gas")

    @property
    def h_liq(self) -> float:
        """Cumulative enthalpy to the liquid (J/mol); 0 if no liquid reference."""
        p = self.liquid
        return p.enthalpy if p and p.enthalpy is not None else 0.0

    @property
    def s_liq(self) -> float:
        """Cumulative entropy to the liquid (J/(mol.K)); 0 if no liquid reference."""
        p = self.liquid
        return p.entropy if p and p.entropy is not None else 0.0

    @property
    def t_fusion(self) -> float:
        """Melting temperature (K); 0 if unknown."""
        p = self.liquid
        return p.t_transition if p and p.t_transition is not None else 0.0

    @property
    def t_vaporization(self) -> float:
        """Boiling/vaporization temperature (K); 0 if unknown."""
        p = self.gas
        return p.t_transition if p and p.t_transition is not None else 0.0

    @property
    def polymorphs(self) -> list[Phase]:
        """Solid phases with a known (>= 0 K) transition temperature.

        Mirrors the legacy ``element_polymorphs`` list, which includes the T = 0 K ground state.
        """
        return [
            p
            for p in self.phases
            if p.phase_type == "solid" and p.t_transition is not None and p.t_transition >= 0
        ]

    def gibbs_ref_expr(self, t_sym):
        """Referenced liquid Gibbs energy of the pure element: ``h_liq - t_sym * s_liq``.

        ``t_sym`` is the caller's temperature symbol (e.g. a sympy Symbol); returned unevaluated.
        """
        return self.h_liq - t_sym * self.s_liq

    def solid_phase(self, spacegroup_number: int) -> Phase | None:
        """Per-phase reference lookup for solid solutions (BCC=229, FCC=225, HCP=194)."""
        return next(
            (
                p
                for p in self.phases
                if p.phase_type == "solid" and p.spacegroup_number == spacegroup_number
            ),
            None,
        )

    def liquid_ref_from_solids(self, solids: list[Phase] | None = None) -> tuple[float, float]:
        """Reconstruct ``(h_liq, s_liq)`` from an ordered solid ladder plus fusion.

        ``S = sum(delta_h_i / T_i)`` over the solid-solid transitions (T > 0) ``+ H_fus / T_melt``;
        ``H`` is the cumulative enthalpy to the liquid.  Passing the full solid ladder reproduces
        the stored ``h_liq``/``s_liq`` -- the forward-compat contract for solid-solution reference
        states, where ``solids`` is instead the subset of phases the model carries.
        """
        liquid = self.liquid
        if liquid is None:
            return 0.0, 0.0
        if solids is None:
            solids = [p for p in self.polymorphs if p.t_transition and p.t_transition > 0]
        s = 0.0
        for p in solids:
            if p.delta_h is not None and p.t_transition:
                s += p.delta_h / p.t_transition
        if liquid.delta_h is not None and liquid.t_transition:
            s += liquid.delta_h / liquid.t_transition
        h = liquid.enthalpy if liquid.enthalpy is not None else 0.0
        return h, s

    def with_liquid_ref(self, h: float, s: float) -> ComponentRef:
        """Copy with the liquid phase's cumulative enthalpy/entropy replaced.

        Used by solid-solution reference reconciliation to install a liquid reference derived
        from a solid-phase subset (see ``liquid_ref_from_solids``) without mutating any shared
        ``Phase`` — the registry stays pristine.
        """
        new = self.copy()
        for i in range(len(new.phases) - 1, -1, -1):
            if new.phases[i].phase_type == "liquid":
                new.phases[i] = replace(new.phases[i], enthalpy=h, entropy=s)
                break
        return new

    def copy(self) -> ComponentRef:
        """Copy with fresh ``Phase`` instances so per-system mutation cannot corrupt the registry."""
        return ComponentRef(
            self.symbol,
            [replace(p, points=list(p.points)) for p in self.phases],
            [replace(p, points=list(p.points)) for p in self.lattice_stabilities],
        )


class UnaryData:
    """Registry of ``ComponentRef`` keyed by element symbol, loaded from ``phase_transitions.json``."""

    def __init__(self, require: bool = False):
        self.elements: dict[str, ComponentRef] = {}
        self.reload(require=require)

    def reload(self, require: bool = True) -> None:
        """(Re)load element reference data from ``config.phase_transitions_file``.

        Rebuilds ``self.elements`` in place.  ``require=False`` tolerates a missing file (e.g.
        during a data-directory switch).
        """
        if os.path.exists(config.phase_transitions_file):
            with open(config.phase_transitions_file) as f:
                raw = json.load(f)
        else:
            raw = {}
        elements = raw.get("elements", {})
        self.elements = {
            symbol: ComponentRef(
                symbol,
                [
                    Phase.from_json(p, composition=Composition(symbol))
                    for p in data.get("phases", [])
                ],
                [
                    Phase(
                        phase_type="solid",
                        name=f"{symbol} {ss_name} (lattice stability)",
                        composition=Composition(symbol),
                        t_transition=None,
                        enthalpy=entry.get("delta_H_J_per_mol"),
                        # 0.0 by the recalculation convention (no transition temperature
                        # to divide by), EXCEPT for the builder's scoped metastable-entropy
                        # exception, which emits a negative SGTE value on a handful of HCP
                        # entries. Read what the builder wrote; default to 0.0 so an older
                        # database file (or one predating the field) is unchanged.
                        entropy=entry.get("delta_S_J_per_mol_K")
                        if entry.get("delta_S_J_per_mol_K") is not None
                        else 0.0,
                        spacegroup_number=entry.get("spacegroup_number"),
                        spacegroup_symbol=entry.get("spacegroup_symbol"),
                        material_id=entry.get("materials_project_id"),
                        imputed=entry.get("materials_project_id") == "omegas_hcp",
                        source=(entry.get("metadata") or {}).get("source"),
                        energy_per_atom_ev=(entry.get("metadata") or {}).get("energy_per_atom_eV"),
                    )
                    for ss_name, entry in data.get("lattice_stabilities", {}).items()
                ],
            )
            for symbol, data in elements.items()
        }
        if not self.elements:
            # phase_transitions.json ships inside the package, so reaching here means the
            # file was overridden (set_data_dir at a directory carrying its own unreadable
            # copy) or the install is damaged -- report whichever location was consulted.
            consulted = config.data_dir if config.data_dir is not None else config._BUNDLED_DATA_DIR
            data_dir_parts = os.path.normpath(consulted).split(os.sep)
            last_two = (
                os.sep.join(data_dir_parts[-2:]) if len(data_dir_parts) >= 2 else str(consulted)
            )
            if require:
                raise FileNotFoundError(
                    f"The following data files were not loaded correctly: {config.phase_transitions_file}. "
                    f"Please ensure the files exist in the data directory '...{os.sep}{last_two}'."
                )
            warnings.warn(
                f"gliquid unary registry loaded empty ({config.phase_transitions_file} not readable); "
                f"all element references will evaluate to zero. Call config.set_data_dir(...) then "
                f"phase.reload().",
                UserWarning,
                stacklevel=2,
            )

    def __getitem__(self, symbol: str) -> ComponentRef:
        """Live ``ComponentRef`` for ``symbol``; an empty ref (all-zero scalars) if unknown."""
        return self.elements.get(symbol) or ComponentRef(symbol, [])

    def get(self, symbol: str, default=None):
        ref = self.elements.get(symbol)
        return (
            ref
            if ref is not None
            else (default if default is not None else ComponentRef(symbol, []))
        )

    def __contains__(self, symbol: str) -> bool:
        return symbol in self.elements

    def component_data(self, components) -> dict[str, ComponentRef]:
        """Per-system reference map ``{symbol: ComponentRef}`` (copies, safe to mutate)."""
        return {comp: self[comp].copy() for comp in components}


# Single registry loaded at import (mirrors config.phase_transitions_file).
UNARY = UnaryData(require=False)


def reload(require: bool = True) -> None:
    """Reload the module-level ``UNARY`` registry (call after ``config.set_data_dir``)."""
    UNARY.reload(require=require)


# --------------------------------------------------------------------------------------
# System/component input parsing — the single funnel every entry point routes through.
# --------------------------------------------------------------------------------------


def validate_and_format_system(
    input, *, allow_compounds: bool = False
) -> tuple[list[str], str, bool]:
    """Parse an n-component system spec, preserving the caller's component order.

    Args:
        input (str or list): Hyphenated string (``'A-B'``, ``'A-B-C'``, ...) or a list of
            two or more component formula strings.
        allow_compounds: Compound (multi-element) components raise NotImplementedError by
            default — this is the single guard every user entry point routes through.
            Internal plumbing that already speaks CompoundPhaseDiagram (the api cache
            layer) passes True.

    Returns:
        tuple[list[str], str, bool]: The components in the ORDER GIVEN, the hyphenated
        system name in that same order, and ``order_changed`` — whether that order
        differs from alphabetical (the canonical convention for fitted parameters and
        on-disk cache keys).
    """
    if isinstance(input, str) and input.count("-") >= 1:
        components = input.split("-")
    elif isinstance(input, list) and all(isinstance(c, str) for c in input) and len(input) >= 2:
        components = list(input)
    else:
        raise ValueError(
            "Input must be a hyphenated string or a list of two or more component formulas."
        )

    # Validate each component as a formula of real periodic-table elements.
    # Compound formulas (e.g. 'CuMg') parse; DummySpecies ('A', 'Xx', etc.) are rejected.
    compounds = []
    for c in components:
        try:
            comp = Composition(c)
        except Exception as exc:
            raise ValueError(f"'{c}' is not a valid composition formula.") from exc
        dummy = [str(el) for el in comp.elements if isinstance(el, DummySpecies)]
        if dummy:
            raise ValueError(
                f"Component '{c}' contains non-element species {dummy}. "
                "Each component must be a real element or a compound of real elements."
            )
        if len(comp.elements) > 1:
            compounds.append(c)
    if compounds and not allow_compounds:
        raise NotImplementedError(
            f"Compound components are not yet supported (got {compounds}); pseudo-binary "
            "systems with compound end-members will be supported in a future release."
        )
    return components, "-".join(components), sorted(components) != components


def resolve_component_order(order, components) -> list[str]:
    """Resolve a presentation/evaluation order against a system's components.

    Args:
        components: The system's components, in their original spellings; the returned
            order is a permutation of these.
        order: ``None`` or ``'alphabetical'`` — sorted order; ``'given'`` — the
            components exactly as passed; otherwise a hyphenated string
            (``'Zr-Hf'``), a list of formula strings, or ``Composition`` objects
            naming every component once. Entries are matched via ``Composition``
            equality, so any formula spelling identifies its component (e.g.
            ``'MgCu'`` matches a ``'CuMg'`` end-member).

    Returns:
        list[str]: A permutation of ``components`` (the ORIGINAL spellings).
    """
    comps = list(components)
    if order is None or (isinstance(order, str) and order.lower() == "alphabetical"):
        return sorted(comps)
    if isinstance(order, str) and order.lower() == "given":
        return comps
    requested = order.split("-") if isinstance(order, str) else list(order)
    if len(requested) != len(comps):
        raise ValueError(
            f"order must name all {len(comps)} components of {comps} exactly once, got {requested}."
        )
    resolved = []
    remaining = list(comps)
    for item in requested:
        try:
            target = item if isinstance(item, Composition) else Composition(item)
        except Exception as exc:
            raise ValueError(f"'{item}' is not a valid composition formula.") from exc
        match = next((c for c in remaining if Composition(c) == target), None)
        if match is None:
            raise ValueError(
                f"'{item}' does not identify a distinct component of "
                f"{comps} (unmatched or duplicated)."
            )
        remaining.remove(match)
        resolved.append(match)
    return resolved
