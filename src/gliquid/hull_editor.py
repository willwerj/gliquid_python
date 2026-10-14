"""Interactive convex hull editor for binary phase diagrams.

A Jupyter widget (ipywidgets + Plotly) that edits a pymatgen ``PhaseDiagram``, or a
``BinaryLiquid``'s ``bl.dft_ch``: modify formation energies and entropies, add entries by
reduced formula and formation energy, remove entries, slide a temperature to view the
Gibbs-energy hull ``G(T)/atom = H_f - T*S``, undo or reset, and write the result back.

An added or edited entry becomes a user-defined entry (``gliquid.api.user_entry``) with no
structure, recording the ids it stands in for under ``data['replaces']``; every other entry
keeps its original object. The total energy of an entry with a desired formation energy is

    E_total = N · (E_f_desired / atom + Σ(frac_i · E_ref_i))

Energies and entropies are in eV/atom and eV/atom/K; ``add_entry`` and ``modify_entry`` take
``units='J/mol'`` for J/mol and J/mol-atom/K, and the widget has a units toggle.

``apply()`` writes back to the source: a ``PhaseDiagram`` source returns the modified
PhaseDiagram; a ``BinaryLiquid`` source gets ``bl.dft_ch`` updated and ``bl.phases`` rebuilt
(enthalpy in J/mol, entropy in J/mol-atom/K) and is returned.

Usage::

    from gliquid.hull_editor import ConvexHullEditor

    editor = ConvexHullEditor(phase_diagram)   # or ConvexHullEditor(binary_liquid)
    editor.display()
    result = editor.apply()

Requires ipywidgets and IPython (``pip install gliquid[notebook]``); constructing an editor
without them raises, naming the install.
"""

from __future__ import annotations

import copy
import logging
import sys
from typing import TYPE_CHECKING

import plotly.graph_objects as go
from pymatgen.analysis.phase_diagram import CompoundPhaseDiagram, PDEntry, PhaseDiagram
from pymatgen.core import Composition

import gliquid.api as api
from gliquid.binary import _endpoint_polymorphs, build_phases_from_chull, hull_phase_fields
from gliquid.cache import (
    USER_ID_PREFIX,
    canonical_entry_id,
    non_dft_entry_kind,
    non_dft_entry_label,
)
from gliquid.imputation.core import REPLACES_KEY

# Per-atom electronvolts <-> per-mole-of-atoms joules. Energies are stored in eV/atom and
# entropies in eV/atom/K; J/mol is an optional input and display unit.
from gliquid.phase import EV_ATOM_TO_J_MOL as J_PER_MOL_PER_EV
from gliquid.phase import SS_SPACEGROUPS, Phase

# ipywidgets/IPython come from the optional ``notebook`` extra; without it both names are
# placeholders that raise, naming the install, at first use.
_EDITOR_EXTRA_HINT = (
    "ConvexHullEditor needs the notebook widget stack (ipywidgets and IPython), which "
    "gliquid ships as an optional extra. Install it with `pip install gliquid[notebook]` "
    "or `pip install ipywidgets ipython`."
)


class _MissingEditorDep:
    """Stand-in for an absent editor dependency; any attribute access or call raises."""

    def __init__(self, module_name: str):
        self._module_name = module_name

    def _fail(self):
        raise ImportError(f"{self._module_name} is not installed. {_EDITOR_EXTRA_HINT}")

    def __getattr__(self, attr):
        self._fail()

    def __call__(self, *args, **kwargs):
        self._fail()


try:
    import ipywidgets as widgets
    from IPython.display import display
except ImportError:
    widgets = _MissingEditorDep("ipywidgets")
    display = _MissingEditorDep("IPython")


def _require_editor_deps():
    """Raise an ImportError naming the widget install when the widget stack is absent."""
    if isinstance(widgets, _MissingEditorDep) or isinstance(display, _MissingEditorDep):
        raise ImportError(_EDITOR_EXTRA_HINT)


# Carries the change-log pane's failures into the logging stream.
logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    pass


# -- Helpers --


def _copy_entries(entries) -> list[PDEntry]:
    """Deep copies of ``entries``, each keeping its type, id, data and structure."""
    return [copy.deepcopy(e) for e in entries]


# -- ConvexHullEditor --


class ConvexHullEditor:
    """Interactive editor for binary PhaseDiagram convex hulls.

    Args:
        source (PhaseDiagram | BinaryLiquid): Either a **binary** pymatgen
            ``PhaseDiagram`` or a ``BinaryLiquid`` object exposing a valid ``dft_ch``
            PhaseDiagram.  The editor operates on a deep copy of the entries so the
            original object is never mutated until :meth:`apply` is called.

    Raises:
        ImportError: If ipywidgets/IPython (the optional ``notebook`` extra) are not
            installed.
        ValueError: If the supplied phase diagram is not binary.
        TypeError: If ``source`` is neither a PhaseDiagram nor a BinaryLiquid-like object.
        NotImplementedError: If the phase diagram is a ``CompoundPhaseDiagram``.
    """

    # ------------------------------------------------------------------ init
    def __init__(self, source):
        # Raises first when the widget extra is missing.
        _require_editor_deps()

        # Resolve the source into a PhaseDiagram + optional BinaryLiquid handle.
        if isinstance(source, PhaseDiagram):
            self._bl = None
            phase_diagram = source
        elif (
            hasattr(source, "dft_ch")
            and hasattr(source, "phases")
            and hasattr(source, "components")
        ):
            self._bl = source
            phase_diagram = source.dft_ch
            if phase_diagram is None:
                raise ValueError("BinaryLiquid has no 'dft_ch' PhaseDiagram to edit.")
        else:
            raise TypeError(
                "ConvexHullEditor expects a pymatgen PhaseDiagram or a "
                f"BinaryLiquid with a 'dft_ch' attribute, got {type(source)!r}."
            )

        if isinstance(phase_diagram, CompoundPhaseDiagram):
            raise NotImplementedError(
                "ConvexHullEditor does not yet support CompoundPhaseDiagram (compound "
                "end-members); pseudo-binary hull editing will be supported in a future release."
            )
        if len(phase_diagram.elements) != 2:
            raise ValueError(
                f"ConvexHullEditor only supports binary phase diagrams.  "
                f"Got {len(phase_diagram.elements)} elements: "
                f"{[str(e) for e in phase_diagram.elements]}"
            )

        # Element A (alphabetically first) sits at x = 0, element B at x = 1.
        self._elements = sorted(phase_diagram.elements, key=lambda el: el.symbol)
        self._el_a, self._el_b = self._elements

        # Working state: entries + a parallel entropy list (eV/atom/K).
        self._entries: list[PDEntry] = _copy_entries(phase_diagram.all_entries)
        self._entropies: list[float] = [0.0] * len(self._entries)

        # Enthalpy (T=0) hull used for back-calculation and enthalpy readout.
        self._pd: PhaseDiagram = self._rebuild_pd()

        # Each compound phase of a BinaryLiquid seeds its entropy, converted from J/mol-atom/K to
        # eV/atom/K, onto one entry at its composition: the same-named one if any, then the one
        # closest in formation enthalpy. Every other entry and elemental reference keeps S = 0.
        if self._bl is not None:
            for phase in self._bl.phases:
                if phase.composition is None or phase.composition.is_element:
                    continue
                fraction = phase.composition.fractional_composition
                candidates = [
                    i
                    for i, e in enumerate(self._entries)
                    if e.composition.fractional_composition == fraction
                ]
                if candidates:
                    i = min(
                        candidates,
                        key=lambda k: (
                            hull_phase_fields(self._entries[k])["name"] != phase.name,
                            abs(
                                J_PER_MOL_PER_EV
                                * self._pd.get_form_energy_per_atom(self._entries[k])
                                - (phase.enthalpy or 0.0)
                            ),
                        ),
                    )
                    self._entropies[i] = float(phase.entropy or 0) / J_PER_MOL_PER_EV

        # Temperature (K) at which the displayed Gibbs hull is evaluated.
        self._temperature: float = 0.0

        # Display and input unit, 'eV' or 'J/mol'; storage is always eV.
        self._display_units: str = "eV"

        # Immutable snapshots for *Reset*
        self._original_entries = _copy_entries(self._entries)
        self._original_entropies = list(self._entropies)

        # Undo stack: list of (description, entries_snapshot, entropies_snapshot)
        self._history: list[tuple[str, list[PDEntry], list[float]]] = []

        # Display state (recomputed on every refresh).
        self._display_pd: PhaseDiagram = self._pd
        self._display_entries: list[PDEntry] = self._entries
        self._stable_idx: set[int] = set()
        self._stable_at_0k: set[int] = set()

        # Build and wire the UI
        self._build_ui()
        self._refresh()

    # ------------------------------------------------------ core bookkeeping
    def _rebuild_pd(self) -> PhaseDiagram:
        """Reconstruct the enthalpy (T=0) PhaseDiagram from working entries."""
        self._pd = PhaseDiagram(self._entries, self._elements)
        return self._pd

    def _make_gibbs(self, temperature: float):
        """Build a Gibbs hull at ``temperature``: G_total = E_total - T*S_total.

        Returns ``(phase_diagram, entries)`` where each entry carries
        ``attribute = i`` (its index into ``self._entries``) so stability can be
        mapped back to the working entries.  Elemental references have S = 0 and
        are therefore unchanged, keeping the formation referencing consistent.
        """
        entries = []
        for i, (entry, s) in enumerate(zip(self._entries, self._entropies)):
            s_total_ev = s * entry.composition.num_atoms  # s is eV/atom/K
            entries.append(
                PDEntry(
                    entry.composition,
                    entry.energy - temperature * s_total_ev,
                    name=getattr(entry, "name", None),
                    attribute=i,
                )
            )
        return PhaseDiagram(entries, self._elements), entries

    def _build_display(self):
        """Recompute the Gibbs hull at the current T and the stable-entry sets at T and at 0 K."""
        self._display_pd, self._display_entries = self._make_gibbs(self._temperature)
        self._stable_idx = {e.attribute for e in self._display_pd.stable_entries}
        stable_at_0k = {id(e) for e in self._pd.stable_entries}
        self._stable_at_0k = {i for i, e in enumerate(self._entries) if id(e) in stable_at_0k}

    def _save_state(self, description: str):
        """Push the current entries + entropies onto the undo stack."""
        self._history.append((description, _copy_entries(self._entries), list(self._entropies)))

    # --------------------------------------------------- energy calculations
    def _calc_required_energy(self, composition: Composition, target_ef_per_atom: float) -> float:
        """Back-calculate the total DFT energy for a desired formation energy.

        Derivation — PhaseDiagram computes:

            E_f = E_total - Σ n_i · E_ref_i        (per formula unit)
            E_f / atom = E_total / N - Σ frac_i · E_ref_i

        Rearranging:

            E_total = N · (E_f/atom + Σ frac_i · E_ref_i)

        Args:
            composition (Composition): Target composition (must only contain elements in
                the PD).
            target_ef_per_atom (float): Desired formation energy in eV / atom.

        Returns:
            float: The total energy the PDEntry must carry so that
            ``PhaseDiagram.get_form_energy_per_atom`` returns *target_ef_per_atom*.
        """
        ref_e_per_atom = sum(
            composition.get_atomic_fraction(el) * self._pd.el_refs[el].energy_per_atom
            for el in composition.elements
        )
        return composition.num_atoms * (target_ef_per_atom + ref_e_per_atom)

    def _free_user_id(self, label: str, keep: int | None = None) -> str:
        """``'user:<label>'``, suffixed ``#2``, ``#3``, ... until no other working entry holds it.

        Args:
            label: the phase name.
            keep: index of a working entry whose id does not count as taken.
        """
        taken = {
            str(getattr(e, "entry_id", None)) for i, e in enumerate(self._entries) if i != keep
        }
        entry_id, n = f"{USER_ID_PREFIX}{label}", 2
        while entry_id in taken:
            entry_id, n = f"{USER_ID_PREFIX}{label}#{n}", n + 1
        return entry_id

    def _edited_entry(self, index: int, energy: float):
        """The user-defined entry that replaces working entry ``index`` at total ``energy``.

        It keeps the entry's phase name, and its id when that is already a user id. An edited
        DFT entry lists its own id under ``data['replaces']``; an edited imputed or user entry
        keeps the ``replaces`` it carried.
        """
        old = self._entries[index]
        kind = non_dft_entry_kind(old)
        label = non_dft_entry_label(old) if kind else old.name
        old_id = getattr(old, "entry_id", None)
        carried = (getattr(old, "data", None) or {}).get(REPLACES_KEY)
        extra = {}
        if kind is None and old_id is not None:
            extra[REPLACES_KEY] = [str(old_id)]
        elif kind is not None and carried:
            extra[REPLACES_KEY] = list(carried)
        if kind == "user" and str(old_id).startswith(USER_ID_PREFIX):
            entry_id = str(old_id)
        else:
            entry_id = self._free_user_id(label, keep=index)
        return api.user_entry(old.composition, energy, label, entry_id=entry_id, **extra)

    # ------------------------------------------------- unit conversion / format
    @staticmethod
    def _ev_from(value: float, units: str) -> float:
        """Convert an input *value* expressed in *units* to the eV basis.

        ``units='eV'``   : value is eV/atom (energy) or eV/atom/K (entropy).
        ``units='J/mol'``: value is J/mol (energy) or J/mol-atom/K (entropy);
        divided by 96485 to reach the eV basis the editor stores internally.
        """
        if units == "eV":
            return float(value)
        if units == "J/mol":
            return float(value) / J_PER_MOL_PER_EV
        raise ValueError(f"units must be 'eV' or 'J/mol', got {units!r}.")

    @staticmethod
    def _convert(value: float, from_units: str, to_units: str) -> float:
        """Convert *value* between the 'eV' and 'J/mol' bases."""
        ev = value / J_PER_MOL_PER_EV if from_units == "J/mol" else value
        return ev * J_PER_MOL_PER_EV if to_units == "J/mol" else ev

    def _disp_energy(self, ev: float) -> float:
        """eV/atom -> current display units."""
        return ev * J_PER_MOL_PER_EV if self._display_units == "J/mol" else ev

    def _disp_entropy(self, ev: float) -> float:
        """eV/atom/K -> current display units."""
        return ev * J_PER_MOL_PER_EV if self._display_units == "J/mol" else ev

    def _fmt_energy(self, ev: float) -> str:
        if self._display_units == "J/mol":
            return f"{self._disp_energy(ev):+.1f}"
        return f"{self._disp_energy(ev):+.5f}"

    def _fmt_entropy(self, ev: float) -> str:
        if self._display_units == "J/mol":
            return f"{self._disp_entropy(ev):+.3f}"
        return f"{self._disp_entropy(ev):+.2e}"

    def _energy_input_value(self, ev: float) -> float:
        if self._display_units == "J/mol":
            return round(self._disp_energy(ev), 3)
        return round(self._disp_energy(ev), 6)

    def _entropy_input_value(self, ev: float) -> float:
        if self._display_units == "J/mol":
            return round(self._disp_entropy(ev), 4)
        return round(self._disp_entropy(ev), 9)

    @property
    def _energy_unit(self) -> str:
        return "J/mol" if self._display_units == "J/mol" else "eV/atom"

    @property
    def _entropy_unit(self) -> str:
        return "J/mol/K" if self._display_units == "J/mol" else "eV/atom/K"

    def _refresh_if_ready(self):
        """Redraw the widget if it has been built.

        Programmatic edits (``add_entry``/``modify_entry``/...) call this so the
        displayed hull, table, and dropdowns stay in sync even when no
        temperature has been set and ``display()`` is called afterwards.
        """
        if getattr(self, "_widget", None) is not None:
            self._refresh()

    # ------------------------------------------------------------ public API
    def modify_entry(
        self,
        entry_index: int,
        new_ef_per_atom: float | None = None,
        new_entropy: float | None = None,
        units: str = "eV",
    ):
        """Change an existing entry's formation energy and/or entropy.

        Args:
            entry_index (int): Index into the working entries.
            new_ef_per_atom (float, optional): Desired formation energy (see *units*).
                Left unchanged if None.
            new_entropy (float, optional): Entropy (see *units*).  Left unchanged if None.
            units ({'eV', 'J/mol'}): Units of the supplied values, default ``'eV'``.
                ``'eV'`` is eV/atom for energy and eV/atom/K for entropy; ``'J/mol'`` is
                J/mol and J/mol-atom/K.  Both are converted to the editor's internal
                eV basis.
        """
        if new_ef_per_atom is not None:
            new_ef_per_atom = self._ev_from(new_ef_per_atom, units)
        if new_entropy is not None:
            new_entropy = self._ev_from(new_entropy, units)

        old = self._entries[entry_index]
        old_ef = self._pd.get_form_energy_per_atom(old)
        old_s = self._entropies[entry_index]

        changes = []
        if new_ef_per_atom is not None:
            changes.append(
                f"Ef {self._fmt_energy(old_ef)}→{self._fmt_energy(new_ef_per_atom)} {self._energy_unit}"
            )
        if new_entropy is not None:
            changes.append(
                f"S {self._fmt_entropy(old_s)}→{self._fmt_entropy(new_entropy)} {self._entropy_unit}"
            )
        self._save_state(f"Modify {old.name}: " + ", ".join(changes or ["(no change)"]))

        if new_ef_per_atom is not None or new_entropy is not None:
            new_energy = (
                old.energy
                if new_ef_per_atom is None
                else self._calc_required_energy(old.composition, new_ef_per_atom)
            )
            self._entries[entry_index] = self._edited_entry(entry_index, new_energy)
            self._rebuild_pd()
        if new_entropy is not None:
            self._entropies[entry_index] = float(new_entropy)
        self._refresh_if_ready()

    def add_entry(
        self,
        formula: str,
        ef_per_atom: float,
        name: str | None = None,
        entropy: float = 0.0,
        units: str = "eV",
    ):
        """Add a brand-new entry to the phase diagram.

        Args:
            formula (str): Reduced formula (e.g. ``'GaRu'``, ``'MgCu2'``).
            ef_per_atom (float): Desired formation energy (see *units*).
            name (str, optional): Display name.  Defaults to the reduced formula.
            entropy (float, optional): Entropy (see *units*).  Defaults to 0.
            units ({'eV', 'J/mol'}): Units of *ef_per_atom* and *entropy*, default
                ``'eV'``.  ``'eV'`` is eV/atom and eV/atom/K; ``'J/mol'`` is J/mol and
                J/mol-atom/K.  Both are converted to the editor's internal eV basis.

        Raises:
            ValueError: an element of ``formula`` is not in the system, or the name is
                ``'L'``, a solid-solution or endpoint polymorph phase of the source, or the
                name of an entry at another composition.
        """
        ef_per_atom = self._ev_from(ef_per_atom, units)
        entropy = self._ev_from(entropy, units)

        comp = Composition(formula)
        for el in comp.elements:
            if el not in self._elements:
                raise ValueError(
                    f"Element {el} in '{formula}' is not part of the "
                    f"{self._el_a}-{self._el_b} phase diagram."
                )
        label = name or comp.reduced_formula
        if label in self._taken_names(comp):
            raise ValueError(f"Phase name {label!r} is already used in this system.")

        self._save_state(
            f"Add {formula} with Ef={self._fmt_energy(ef_per_atom)} {self._energy_unit}, "
            f"S={self._fmt_entropy(entropy)} {self._entropy_unit}"
        )

        total_energy = self._calc_required_energy(comp, ef_per_atom)
        self._entries.append(
            api.user_entry(comp, total_energy, label, entry_id=self._free_user_id(label))
        )
        self._entropies.append(float(entropy))
        self._rebuild_pd()
        self._refresh_if_ready()

    def remove_entry(self, entry_index: int):
        """Remove an entry from the working set.

        Elemental reference entries cannot be removed if they are the last
        remaining entry for that element.
        """
        entry = self._entries[entry_index]

        # Guard: each element keeps an entry
        if entry.composition.is_element:
            el = entry.composition.elements[0]
            n_remaining = sum(
                1
                for e in self._entries
                if e.composition.is_element and el in e.composition.elements
            )
            if n_remaining <= 1:
                self._log(f"⚠ Cannot remove the only {el} reference entry.")
                return

        self._save_state(f"Remove {entry.name}")
        del self._entries[entry_index]
        del self._entropies[entry_index]
        self._rebuild_pd()
        self._refresh_if_ready()

    def set_temperature(self, temperature: float):
        """Set the temperature (K) of the displayed Gibbs hull; :meth:`apply` keeps the entries
        stable there as well as those stable at 0 K."""
        self._temperature = float(temperature)
        if getattr(self, "_temp_slider", None) is not None:
            self._temp_slider.value = self._temperature
        self._refresh()

    @property
    def entries(self) -> list[dict]:
        """Read-only summary of the working entries (index, name, x_B, Ef, S, ``stable`` at the
        current temperature, ``stable_at_0k``, and ``non_dft``: ``'user'``, ``'imputed'`` or
        None)."""
        self._build_display()
        return [
            {
                "index": i,
                "name": e.name,
                "x": e.composition.get_atomic_fraction(self._el_b),
                "ef_per_atom": self._pd.get_form_energy_per_atom(e),
                "entropy": self._entropies[i],
                "stable": i in self._stable_idx,
                "stable_at_0k": i in self._stable_at_0k,
                "non_dft": non_dft_entry_kind(e),
            }
            for i, e in enumerate(self._entries)
        ]

    def find_entry(self, name: str | None = None, x: float | None = None, tol: float = 1e-3) -> int:
        """Return the index of the working entry matching ``name`` and/or the
        composition fraction ``x`` (of element B).

        Raises
        ------
        ValueError
            If no entry matches the criteria.
        """
        matches = []
        for i, e in enumerate(self._entries):
            if name is not None and e.name != name:
                continue
            if x is not None and abs(e.composition.get_atomic_fraction(self._el_b) - x) > tol:
                continue
            matches.append(i)
        if not matches:
            raise ValueError(f"No entry matches name={name!r}, x={x}.")
        return matches[0]

    def undo(self):
        """Revert the most recent change."""
        if not self._history:
            self._log("Nothing to undo.")
            return
        desc, prev_entries, prev_entropies = self._history.pop()
        self._entries = prev_entries
        self._entropies = prev_entropies
        self._rebuild_pd()
        self._refresh_if_ready()
        self._log(f"↩ Undone: {desc}")

    def reset(self):
        """Restore all entries to the state they were in at construction time."""
        self._history.clear()
        self._entries = _copy_entries(self._original_entries)
        self._entropies = list(self._original_entropies)
        self._rebuild_pd()
        self._refresh_if_ready()
        self._log("🔄 Reset to original phase diagram.")

    def get_phase_diagram(self) -> PhaseDiagram:
        """Return the current (enthalpy, T=0) PhaseDiagram. Kept for back-compat."""
        return self._enthalpy_pd()

    def _taken_names(self, composition: Composition) -> set[str]:
        """Names an entry added at ``composition`` may not take: ``'L'``, the source's
        solid-solution and endpoint polymorph phases, and every entry at another composition."""
        taken = {"L"}
        if self._bl is not None:
            taken.update(getattr(self._bl, "ss_models", None) or {})
            component_data = getattr(self._bl, "component_data", None)
            if component_data:
                taken.update(
                    p.name for p in _endpoint_polymorphs(self._bl.components, component_data)
                )
        fraction = composition.fractional_composition
        taken.update(
            hull_phase_fields(e)["name"]
            for e in self._entries
            if e.composition.fractional_composition != fraction
        )
        return taken

    def _ss_spacegroups(self) -> set[int] | None:
        """Spacegroups the source's solid-solution models cover, or None without any."""
        ss_models = getattr(self._bl, "ss_models", None) or {}
        return {SS_SPACEGROUPS[name] for name in ss_models} or None

    def _enthalpy_pd(self) -> PhaseDiagram:
        """A clean PhaseDiagram built from the edited enthalpy entries."""
        return PhaseDiagram(_copy_entries(self._entries), self._elements)

    def _build_phases(self):
        """Build a ``BinaryLiquid``-style phase list from the current edits.

        Solid phases are the endpoint polymorphs, less those a solid-solution model of the
        source covers, plus the entries stable at 0 K or on the Gibbs hull at the current
        temperature (so entropy-stabilized additions are retained while metastable DFT entries
        are dropped); two such entries at one composition are both kept unless they share a
        name, when the one stable at 0 K is kept.  Each entry carries enthalpy (J/mol) from the T=0 hull and entropy
        converted from the editor's eV/atom/K to the J/mol-atom/K the model uses.  The order
        and the trailing liquid placeholder match :func:`gliquid.binary.build_phases_from_chull`.
        """
        components = self._bl.components
        self._build_display()

        phases = _endpoint_polymorphs(components, self._bl.component_data, self._ss_spacegroups())
        for i in sorted(self._stable_at_0k) + sorted(self._stable_idx - self._stable_at_0k):
            entry = self._entries[i]
            composition = entry.composition.get_atomic_fraction(components[1])
            fields = hull_phase_fields(entry)
            if composition in (0, 1) and composition in [
                p.fraction_in(components)[0] for p in phases
            ]:
                continue  # element ground state already covered by a polymorph
            if fields["name"] in {p.name for p in phases}:
                self._log(
                    f"⚠ {fields['name']} stable at {self._temperature:.0f} K shares its name with "
                    "the phase stable at 0 K, which is kept. Add it as an entry with its own name "
                    "to keep both."
                )
                continue
            phases.append(
                Phase(
                    phase_type="solid",
                    composition=entry.composition,
                    enthalpy=J_PER_MOL_PER_EV * self._pd.get_form_energy_per_atom(entry),
                    entropy=J_PER_MOL_PER_EV * self._entropies[i],  # eV/atom/K -> J/mol-atom/K
                    **fields,
                )
            )

        phases.sort(key=lambda p: p.fraction_in(components)[0])
        phases.append(Phase(phase_type="liquid", name="L"))
        return phases

    def apply(self):
        """Write the edits back to the source object (no arguments).

        Returns
        -------
        PhaseDiagram | BinaryLiquid
            For a PhaseDiagram source, the modified enthalpy PhaseDiagram.
            For a BinaryLiquid source, the same BinaryLiquid with ``dft_ch`` and
            ``phases`` updated in place (its solid-solution phases kept), and the ids of the
            DFT entries the edits removed added to ``removed_entry_ids`` (a ternary built with
            ``TernaryLiquidInterpolation.from_binaries`` leaves those out of its hull).
            ``eqs['h_hull_interp']`` is rebuilt from the edited hull at 0 K, and an already
            computed ``hsx`` and the phase points from the new phases.
        """
        if self._bl is None:
            if any(s != 0 for s in self._entropies):
                self._log(
                    "⚠ Entropy edits are not representable in a bare PhaseDiagram "
                    "and were dropped. Initialize the editor from a BinaryLiquid "
                    "to retain entropy."
                )
            return self._enthalpy_pd()

        bl = self._bl
        dft_ch = self._enthalpy_pd()
        phases = self._build_phases()
        removed = set(getattr(bl, "removed_entry_ids", None) or ()) | self._removed_dft_ids()
        bl.dft_ch, bl.phases, bl.removed_entry_ids = dft_ch, phases, sorted(removed)
        if hasattr(bl, "rebuild_hull_interp"):
            bl._ensure_solid_solution_phases()
            bl.rebuild_hull_interp(
                build_phases_from_chull(
                    dft_ch, bl.components, bl.component_data, self._ss_spacegroups()
                )
            )
        if getattr(bl, "hsx", None) is not None and hasattr(bl, "update_phase_points"):
            bl.hsx = None
            bl.update_phase_points()
        return bl

    def _removed_dft_ids(self) -> set[str]:
        """Canonical ids of the original DFT entries that no working entry keeps or replaces."""

        def dft_ids(entries):
            return {
                canonical_entry_id(e.entry_id)
                for e in entries
                if non_dft_entry_kind(e) is None and getattr(e, "entry_id", None) is not None
            }

        replaced = {
            canonical_entry_id(entry_id)
            for e in self._entries
            for entry_id in (getattr(e, "data", None) or {}).get(REPLACES_KEY, ())
        }
        return dft_ids(self._original_entries) - dft_ids(self._entries) - replaced

    # -- UI construction --
    def _build_ui(self):
        """Assemble all ipywidget components."""

        style = {"description_width": "80px"}

        # ---- Header ----
        self._header = widgets.HTML(
            f"<h3 style='margin-bottom:2px;'>Convex Hull Editor - {self._el_a}-{self._el_b}</h3>"
        )

        # ---- Temperature slider ----
        self._temp_slider = widgets.FloatSlider(
            value=self._temperature,
            min=0.0,
            max=2500.0,
            step=10.0,
            description="T (K):",
            continuous_update=False,
            readout_format=".0f",
            layout=widgets.Layout(width="100%"),
            style=style,
        )
        self._temp_slider.observe(self._on_temperature, names="value")

        # ---- Units toggle (switches every display + input field at once) ----
        self._units_toggle = widgets.ToggleButtons(
            options=["eV", "J/mol"],
            value=self._display_units,
            description="Units:",
            tooltips=[
                "eV/atom for energy, eV/atom/K for entropy",
                "J/mol for energy, J/mol-atom/K for entropy",
            ],
            style=style,
        )
        self._units_toggle.observe(self._on_units_change, names="value")

        # ---- Plotly figure (FigureWidget for live embedding) ----
        self._fig = go.FigureWidget()
        self._fig.update_layout(
            template="plotly_white",
            xaxis=dict(
                title=f"x<sub>{self._el_b}</sub>  (atomic fraction)",
                range=[-0.02, 1.02],
            ),
            yaxis=dict(title="Formation energy  (eV / atom)"),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            margin=dict(l=60, r=20, t=40, b=50),
            height=420,
        )

        # ---- Entry summary table (HTML) ----
        self._table_html = widgets.HTML()

        # ---- Modify section ----
        self._mod_dropdown = widgets.Dropdown(
            description="Entry:",
            layout=widgets.Layout(width="100%"),
            style=style,
        )
        self._mod_dropdown.observe(self._on_mod_selection, names="value")

        self._mod_ef = widgets.FloatText(
            description="New E_f:",
            step=0.001,
            layout=widgets.Layout(width="220px"),
            style=style,
        )
        self._mod_s = widgets.FloatText(
            description="New S:",
            step=1e-5,
            layout=widgets.Layout(width="220px"),
            style=style,
        )
        self._mod_btn = widgets.Button(
            description="  Apply",
            button_style="primary",
            icon="check",
            layout=widgets.Layout(width="110px"),
        )
        self._mod_btn.on_click(self._on_modify)

        self._mod_hint = widgets.HTML()
        modify_box = widgets.VBox(
            [
                self._mod_hint,
                self._mod_dropdown,
                widgets.HBox([self._mod_ef, self._mod_s, self._mod_btn]),
            ]
        )

        # ---- Add section ----
        self._add_formula = widgets.Text(
            description="Formula:",
            placeholder="e.g. GaRu, MgCu2",
            layout=widgets.Layout(width="200px"),
            style=style,
        )
        self._add_ef = widgets.FloatText(
            description="E_f:",
            step=0.001,
            layout=widgets.Layout(width="170px"),
            style=style,
        )
        self._add_s = widgets.FloatText(
            description="S:",
            value=0.0,
            step=1e-5,
            layout=widgets.Layout(width="170px"),
            style=style,
        )
        self._add_btn = widgets.Button(
            description="  Add Entry",
            button_style="success",
            icon="plus",
            layout=widgets.Layout(width="120px"),
        )
        self._add_btn.on_click(self._on_add)

        self._add_hint = widgets.HTML()
        add_box = widgets.VBox(
            [
                self._add_hint,
                widgets.HBox([self._add_formula, self._add_ef, self._add_s, self._add_btn]),
            ]
        )

        # ---- Remove section ----
        self._rm_dropdown = widgets.Dropdown(
            description="Entry:",
            layout=widgets.Layout(width="100%"),
            style=style,
        )
        self._rm_btn = widgets.Button(
            description="  Remove",
            button_style="danger",
            icon="trash",
            layout=widgets.Layout(width="110px"),
        )
        self._rm_btn.on_click(self._on_remove)

        remove_box = widgets.VBox(
            [
                widgets.HTML("<b>Remove Entry</b>"),
                self._rm_dropdown,
                self._rm_btn,
            ]
        )

        # ---- Action bar ----
        self._undo_btn = widgets.Button(
            description="  Undo", icon="undo", layout=widgets.Layout(width="100px")
        )
        self._undo_btn.on_click(lambda _: self._do_undo())

        self._reset_btn = widgets.Button(
            description="  Reset All",
            icon="refresh",
            button_style="warning",
            layout=widgets.Layout(width="120px"),
        )
        self._reset_btn.on_click(lambda _: self._do_reset())

        action_bar = widgets.HBox([self._undo_btn, self._reset_btn])

        # ---- Change log ----
        self._log_output = widgets.Output(
            layout=widgets.Layout(
                max_height="160px",
                overflow_y="auto",
                border="1px solid #ddd",
                padding="4px",
            )
        )

        # ---- Assemble ----
        self._widget = widgets.VBox(
            [
                self._header,
                self._units_toggle,
                self._temp_slider,
                self._fig,
                self._table_html,
                widgets.HTML("<hr style='margin:6px 0'>"),
                modify_box,
                widgets.HTML("<hr style='margin:6px 0'>"),
                add_box,
                widgets.HTML("<hr style='margin:6px 0'>"),
                remove_box,
                widgets.HTML("<hr style='margin:6px 0'>"),
                action_bar,
                widgets.HTML("<b>Change Log</b>"),
                self._log_output,
            ],
            layout=widgets.Layout(width="780px"),
        )

    # -- UI refresh helpers --
    def _refresh(self):
        """Redraw everything after a state change."""
        self._sync_unit_labels()
        self._build_display()
        self._update_plot()
        self._update_table()
        self._update_dropdowns()

    def _sync_unit_labels(self):
        """Keep the input-field hints and step sizes aligned with the units."""
        self._mod_hint.value = (
            f"<b>Modify Existing Entry</b>  (E_f in {self._energy_unit}, S in {self._entropy_unit})"
        )
        self._add_hint.value = (
            f"<b>Add New Entry</b>  (E_f in {self._energy_unit}, S in {self._entropy_unit})"
        )
        if self._display_units == "J/mol":
            self._mod_ef.step = self._add_ef.step = 100.0
            self._mod_s.step = self._add_s.step = 0.1
        else:
            self._mod_ef.step = self._add_ef.step = 0.001
            self._mod_s.step = self._add_s.step = 1e-5

    def _entry_sort_key(self, entry):
        """Sort entries by composition fraction of element B."""
        return entry.composition.get_atomic_fraction(self._el_b)

    def _gibbs_ef(self, index: int) -> float:
        """Per-atom Gibbs formation energy of entry ``index`` at current T."""
        ef = self._pd.get_form_energy_per_atom(self._entries[index])
        return ef - self._temperature * self._entropies[index]  # entropy is eV/atom/K

    def _make_label(self, idx: int, entry: PDEntry) -> str:
        """Descriptive one-liner for dropdown options."""
        xb = entry.composition.get_atomic_fraction(self._el_b)
        ef = self._pd.get_form_energy_per_atom(entry)
        if idx in self._stable_idx:
            status = "● on hull"
        else:
            ehull = self._display_pd.get_e_above_hull(self._display_entries[idx])
            status = f"○ +{self._fmt_energy(ehull).lstrip('+')} above hull"
        return f"[{idx}] {entry.name}  (x={xb:.3f})  Ef={self._fmt_energy(ef)}  {status}"

    def _update_dropdowns(self):
        """Refresh both dropdowns with current entry labels."""
        options = [(self._make_label(i, e), i) for i, e in enumerate(self._entries)]
        # Sort by composition for easier browsing
        options.sort(key=lambda opt: self._entry_sort_key(self._entries[opt[1]]))

        self._mod_dropdown.options = options
        self._rm_dropdown.options = options

        # Pre-fill the Ef / S inputs when a selection exists
        if self._mod_dropdown.value is not None:
            idx = self._mod_dropdown.value
            self._mod_ef.value = self._energy_input_value(
                self._pd.get_form_energy_per_atom(self._entries[idx])
            )
            self._mod_s.value = self._entropy_input_value(self._entropies[idx])

    def _update_table(self):
        """Render an HTML summary table of all entries."""
        gibbs = self._temperature > 0
        rows = []
        sorted_entries = sorted(enumerate(self._entries), key=lambda t: self._entry_sort_key(t[1]))
        for idx, entry in sorted_entries:
            xb = entry.composition.get_atomic_fraction(self._el_b)
            ef = self._pd.get_form_energy_per_atom(entry)
            s = self._entropies[idx]
            g = self._gibbs_ef(idx)
            is_stable = idx in self._stable_idx
            ehull = (
                0.0 if is_stable else self._display_pd.get_e_above_hull(self._display_entries[idx])
            )

            dot = "🟢" if is_stable else "🟡" if idx in self._stable_at_0k else "⚪"
            rows.append(
                f"<tr>"
                f"<td>{dot}</td>"
                f"<td><b>{entry.name}</b></td>"
                f"<td>{xb:.3f}</td>"
                f"<td>{self._fmt_energy(ef)}</td>"
                f"<td>{self._fmt_entropy(s)}</td>"
                f"<td>{self._fmt_energy(g)}</td>"
                f"<td>{self._fmt_energy(ehull)}</td>"
                f"</tr>"
            )

        g_header = f"G@{self._temperature:.0f}K" if gibbs else f"G ({self._energy_unit})"
        table = (
            "<div style='max-height:220px; overflow-y:auto; margin:4px 0;'>"
            "<table style='border-collapse:collapse; font-size:13px; width:100%;'>"
            "<thead><tr style='background:#f0f0f0;'>"
            "<th></th>"
            f"<th style='text-align:left; padding:2px 6px;'>Name</th>"
            f"<th style='padding:2px 6px;'>x<sub>{self._el_b}</sub></th>"
            f"<th style='padding:2px 6px;'>E<sub>f</sub> ({self._energy_unit})</th>"
            f"<th style='padding:2px 6px;'>S ({self._entropy_unit})</th>"
            f"<th style='padding:2px 6px;'>{g_header}</th>"
            f"<th style='padding:2px 6px;'>E<sub>hull</sub> ({self._energy_unit})</th>"
            "</tr></thead><tbody>" + "\n".join(rows) + "</tbody></table></div>"
        )
        n_stable = len(self._stable_idx)
        n_kept_at_0k = len(self._stable_at_0k - self._stable_idx)
        n_total = len(self._entries)
        n_changes = len(self._history)
        summary = (
            f"<span style='font-size:12px; color:#555;'>"
            f"{n_stable} stable / {n_total} total entries  ·  "
            + (f"🟡 {n_kept_at_0k} stable at 0 K, kept by apply()  ·  " if n_kept_at_0k else "")
            + f"hull shown at T = {self._temperature:.0f} K  ·  "
            f"{n_changes} change{'s' if n_changes != 1 else ''} made</span>"
        )
        self._table_html.value = summary + table

    def _update_plot(self):
        """Redraw the convex-hull Plotly figure at the current temperature."""
        # Gather data ---------------------------------------------------------
        rows = []
        for i, entry in enumerate(self._entries):
            xb = entry.composition.get_atomic_fraction(self._el_b)
            rows.append(
                (i, xb, self._disp_energy(self._gibbs_ef(i)), entry.name, i in self._stable_idx)
            )

        stable = sorted([r for r in rows if r[4]], key=lambda r: r[1])
        hull_x = [r[1] for r in stable]
        hull_y = [r[2] for r in stable]
        hull_names = [r[3] for r in stable]

        unstable = [r for r in rows if not r[4]]
        unstable_x = [r[1] for r in unstable]
        unstable_y = [r[2] for r in unstable]
        unstable_names = [r[3] for r in unstable]

        gibbs = self._temperature > 0
        energy_label = "G" if gibbs else "E<sub>f</sub>"
        unit = self._energy_unit
        yfmt = ".1f" if self._display_units == "J/mol" else ".4f"

        # Build traces --------------------------------------------------------
        traces: list[go.Scatter] = []

        # Reference line at y = 0
        traces.append(
            go.Scatter(
                x=[0, 1],
                y=[0, 0],
                mode="lines",
                line=dict(color="black", dash="dash", width=1),
                showlegend=False,
                hoverinfo="skip",
            )
        )

        # Hull line
        traces.append(
            go.Scatter(
                x=hull_x,
                y=hull_y,
                mode="lines",
                line=dict(color="#1f77b4", width=2),
                name="Convex Hull",
            )
        )

        # Stable markers + labels
        traces.append(
            go.Scatter(
                x=hull_x,
                y=hull_y,
                mode="markers+text",
                marker=dict(color="#2ca02c", size=11, line=dict(width=1, color="black")),
                text=hull_names,
                textposition="top center",
                textfont=dict(size=11),
                name="Stable",
                hovertemplate=(
                    "%{text}<br>"
                    "x = %{x:.3f}<br>"
                    f"{energy_label} = %{{y:{yfmt}}} {unit}<extra></extra>"
                ),
            )
        )

        # Unstable markers
        if unstable:
            traces.append(
                go.Scatter(
                    x=unstable_x,
                    y=unstable_y,
                    mode="markers",
                    marker=dict(
                        color="#d62728",
                        size=7,
                        symbol="x",
                        line=dict(width=1, color="#d62728"),
                    ),
                    text=unstable_names,
                    name="Unstable",
                    hovertemplate=(
                        "%{text}<br>"
                        "x = %{x:.3f}<br>"
                        f"{energy_label} = %{{y:{yfmt}}} {unit}<extra></extra>"
                    ),
                )
            )

        # Apply to FigureWidget -----------------------------------------------
        y_title = (
            f"Gibbs energy G({self._temperature:.0f} K)  ({unit})"
            if gibbs
            else f"Formation energy  ({unit})"
        )
        with self._fig.batch_update():
            self._fig.data = []
            self._fig.update_layout(yaxis=dict(title=y_title))
        self._fig.add_traces(traces)

    def _log(self, msg: str):
        """Append a message to the change-log pane; characters stdout cannot encode print
        as ``?``."""
        with self._log_output:
            encoding = getattr(sys.stdout, "encoding", None) or "utf-8"
            # Writes into the ipywidgets Output pane (registered logging exemption).
            print(msg.encode(encoding, "replace").decode(encoding))

    # -- Callbacks --
    def _on_temperature(self, change):
        """When the temperature slider moves, rebuild the displayed Gibbs hull."""
        self._temperature = float(change.get("new", 0.0))
        self._refresh()

    def _on_mod_selection(self, change):
        """When the modify-dropdown selection changes, pre-fill the inputs."""
        idx = change.get("new")
        if idx is not None and 0 <= idx < len(self._entries):
            self._mod_ef.value = self._energy_input_value(
                self._pd.get_form_energy_per_atom(self._entries[idx])
            )
            self._mod_s.value = self._entropy_input_value(self._entropies[idx])

    def _on_modify(self, _):
        idx = self._mod_dropdown.value
        if idx is None:
            self._log("⚠ Select an entry first.")
            return
        entry = self._entries[idx]
        new_ef = self._mod_ef.value  # in current display units
        new_s = self._mod_s.value  # in current display units

        # Warn about elemental references
        if entry.composition.is_element:
            self._log(
                f"⚠ Warning: modifying elemental reference {entry.name} "
                "will shift ALL formation energies!"
            )

        old_ef = self._pd.get_form_energy_per_atom(entry)  # eV
        old_s = self._entropies[idx]  # eV
        self.modify_entry(idx, new_ef_per_atom=new_ef, new_entropy=new_s, units=self._display_units)
        new_ef_ev = self._ev_from(new_ef, self._display_units)
        new_s_ev = self._ev_from(new_s, self._display_units)
        self._log(
            f"✓ Modified {entry.name}: "
            f"Ef {self._fmt_energy(old_ef)}→{self._fmt_energy(new_ef_ev)} {self._energy_unit}, "
            f"S {self._fmt_entropy(old_s)}→{self._fmt_entropy(new_s_ev)} {self._entropy_unit}"
        )

    def _on_add(self, _):
        formula = self._add_formula.value.strip()
        ef = self._add_ef.value  # in current display units
        s = self._add_s.value  # in current display units
        if not formula:
            self._log("⚠ Enter a formula.")
            return
        try:
            self.add_entry(formula, ef, entropy=s, units=self._display_units)
            ef_ev = self._ev_from(ef, self._display_units)
            s_ev = self._ev_from(s, self._display_units)
            self._log(
                f"✓ Added {formula} with Ef={self._fmt_energy(ef_ev)} {self._energy_unit}, "
                f"S={self._fmt_entropy(s_ev)} {self._entropy_unit}"
            )
        except Exception as exc:
            # The pane shows the message; the log keeps the traceback.
            logger.warning("Hull editor could not add entry '%s': %s", formula, exc, exc_info=True)
            self._log(f"⚠ {exc}")

    def _on_remove(self, _):
        idx = self._rm_dropdown.value
        if idx is None:
            self._log("⚠ Select an entry first.")
            return
        name = self._entries[idx].name
        self.remove_entry(idx)  # refreshes the display
        # remove_entry logs its own warning on failure, success otherwise:
        if name not in [e.name for e in self._entries]:
            self._log(f"✓ Removed {name}")

    def _on_units_change(self, change):
        """Switch every display and input field between eV and J/mol at once."""
        old_units = change.get("old") or "eV"
        new_units = change.get("new") or "eV"
        self._display_units = new_units
        # Partly entered Add-box values are converted to the new unit.
        self._add_ef.value = self._convert(self._add_ef.value, old_units, new_units)
        self._add_s.value = self._convert(self._add_s.value, old_units, new_units)
        self._refresh()
        self._log(f"Units set to {new_units}.")

    def _do_undo(self):
        self.undo()  # refreshes the display

    def _do_reset(self):
        self.reset()  # refreshes the display

    # -- Display --
    def display(self):
        """Render the full editor widget inside a Jupyter notebook."""
        display(self._widget)

    def _repr_mimebundle_(self, **kwargs):
        """Allow rich display when the object is the last expression in a cell."""
        return self._widget._repr_mimebundle_(**kwargs)
