"""
Interactive Convex Hull Editor for binary phase diagrams.

Provides a Jupyter widget (ipywidgets + Plotly) for modifying a pymatgen-style
PhaseDiagram without writing code:

  - Modify formation energies and entropies of entries
  - Add new entries by reduced formula + desired formation energy (+ entropy)
  - Remove entries
  - Slide a temperature to view the Gibbs-energy hull  G(T)/atom = H_f - T*S
  - Undo / reset changes
  - Write the result back to the source object

Energies and entropies default to the eV basis (eV/atom and eV/atom/K), but the
programmatic ``add_entry`` / ``modify_entry`` methods accept ``units='J/mol'`` to
supply J/mol and J/mol-atom/K instead, and the interactive widget has a units
toggle that switches every display and input field between the two bases at once.

The editor can be initialized from either a pymatgen ``PhaseDiagram`` or a
``BinaryLiquid`` (it then operates on ``bl.dft_ch``).  Calling ``apply()`` with
no arguments writes the edits back to whichever object was supplied:

  - PhaseDiagram source : returns the modified (enthalpy) PhaseDiagram.
  - BinaryLiquid source : updates ``bl.dft_ch`` and rebuilds ``bl.phases`` with
    enthalpy (J/mol) and entropy (converted from the editor's eV/atom/K to the
    J/mol-atom/K the model expects), then returns ``bl``.

The key challenge this tool solves is the **back-calculation** of DFT total
energies.  PhaseDiagram computes formation energies via an internal referencing
scheme:

    E_f / atom = E_total / N - Σ(frac_i · E_ref_i)

When the user specifies a *desired* formation energy, the editor solves for the
total energy that reproduces it:

    E_total = N · (E_f_desired / atom + Σ(frac_i · E_ref_i))

Solid-phase entropy is carried alongside each entry (it cannot live inside a
pymatgen PDEntry).  The temperature slider rebuilds the displayed hull from the
per-atom Gibbs energy ``E_f - T*S`` so entropy-bearing phases descend or ascend
with temperature and can join or leave the hull.

Usage
-----
    from gliquid.hull_editor import ConvexHullEditor

    editor = ConvexHullEditor(phase_diagram)   # or ConvexHullEditor(binary_liquid)
    editor.display()          # show the interactive widget
    # ... make changes via the GUI ...
    result = editor.apply()   # write back to the source object

Author: Joshua Willwerth (generated with assistance from Claude Opus 4.6)
"""

from __future__ import annotations

import copy
from typing import TYPE_CHECKING

import numpy as np
import ipywidgets as widgets
from IPython.display import display
import plotly.graph_objects as go

from pymatgen.core import Composition, Element
from pymatgen.analysis.phase_diagram import PhaseDiagram, PDEntry

if TYPE_CHECKING:
    pass


# Conversion between per-atom electronvolts and per-mole-of-atoms joules
# (1 eV/atom = 96485 J/mol).  The editor stores everything internally in eV:
# energies in eV/atom and entropies in eV/atom/K.  This factor converts to/from
# the J/mol basis used by BinaryLiquid.phases and offered as an optional input /
# display unit (1 eV/atom/K = 96485 J/mol-atom/K).
J_PER_MOL_PER_EV = 96485


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _copy_entry(entry: PDEntry) -> PDEntry:
    """Create a fresh PDEntry copy (avoids shared mutable state)."""
    return PDEntry(
        entry.composition,
        entry.energy,
        name=getattr(entry, "name", None),
        attribute=getattr(entry, "attribute", None),
    )


def _copy_entries(entries) -> list[PDEntry]:
    return [_copy_entry(e) for e in entries]


# ---------------------------------------------------------------------------
# ConvexHullEditor
# ---------------------------------------------------------------------------

class ConvexHullEditor:
    """Interactive editor for binary PhaseDiagram convex hulls.

    Parameters
    ----------
    source : PhaseDiagram | BinaryLiquid
        Either a **binary** pymatgen ``PhaseDiagram`` or a ``BinaryLiquid``
        object exposing a valid ``dft_ch`` PhaseDiagram.  The editor operates on
        a deep copy of the entries so the original object is never mutated until
        :meth:`apply` is called.

    Raises
    ------
    ValueError
        If the supplied phase diagram is not binary.
    TypeError
        If ``source`` is neither a PhaseDiagram nor a BinaryLiquid-like object.
    """

    # ------------------------------------------------------------------ init
    def __init__(self, source):
        # Resolve the source into a PhaseDiagram + optional BinaryLiquid handle.
        if isinstance(source, PhaseDiagram):
            self._bl = None
            phase_diagram = source
        elif hasattr(source, "dft_ch") and hasattr(source, "phases") and hasattr(source, "components"):
            self._bl = source
            phase_diagram = source.dft_ch
            if phase_diagram is None:
                raise ValueError(
                    "BinaryLiquid has no 'dft_ch' PhaseDiagram to edit."
                )
        else:
            raise TypeError(
                "ConvexHullEditor expects a pymatgen PhaseDiagram or a "
                f"BinaryLiquid with a 'dft_ch' attribute, got {type(source)!r}."
            )

        if len(phase_diagram.elements) != 2:
            raise ValueError(
                f"ConvexHullEditor only supports binary phase diagrams.  "
                f"Got {len(phase_diagram.elements)} elements: "
                f"{[str(e) for e in phase_diagram.elements]}"
            )

        # Order the two elements alphabetically by symbol so the diagram is laid
        # out consistently: element A (alphabetically first) sits on the left at
        # x = 0, element B on the right at x = 1.
        self._elements = sorted(phase_diagram.elements, key=lambda el: el.symbol)
        self._el_a, self._el_b = self._elements

        # Working state: entries + a parallel entropy list (eV/atom/K).
        self._entries: list[PDEntry] = _copy_entries(phase_diagram.all_entries)
        self._entropies: list[float] = [0.0] * len(self._entries)
        # Seed entropies from an existing BinaryLiquid phase list (matched by
        # composition) so the editor reflects the current model state.  The
        # model stores entropy in J/mol-atom/K, so convert to eV/atom/K.
        if self._bl is not None:
            comp_to_s = {
                round(p["comp"], 4): float(p.get("entropy", 0) or 0)
                for p in self._bl.phases
                if "comp" in p
            }
            for i, e in enumerate(self._entries):
                xb = round(e.composition.get_atomic_fraction(self._el_b), 4)
                self._entropies[i] = comp_to_s.get(xb, 0.0) / J_PER_MOL_PER_EV

        # Temperature (K) at which the displayed Gibbs hull is evaluated.
        self._temperature: float = 0.0

        # Units used for the interactive display and input fields ('eV' or
        # 'J/mol').  Internal storage is always eV; this only affects rendering.
        self._display_units: str = "eV"

        # Immutable snapshots for *Reset*
        self._original_entries = _copy_entries(self._entries)
        self._original_entropies = list(self._entropies)

        # Enthalpy (T=0) hull used for back-calculation and enthalpy readout.
        self._pd: PhaseDiagram = self._rebuild_pd()

        # Undo stack: list of (description, entries_snapshot, entropies_snapshot)
        self._history: list[tuple[str, list[PDEntry], list[float]]] = []

        # Display state (recomputed on every refresh).
        self._display_pd: PhaseDiagram = self._pd
        self._display_entries: list[PDEntry] = self._entries
        self._stable_idx: set[int] = set()

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
        """Recompute the Gibbs hull and stable-entry set at the current T."""
        self._display_pd, self._display_entries = self._make_gibbs(self._temperature)
        self._stable_idx = {e.attribute for e in self._display_pd.stable_entries}

    def _save_state(self, description: str):
        """Push the current entries + entropies onto the undo stack."""
        self._history.append(
            (description, _copy_entries(self._entries), list(self._entropies))
        )

    # --------------------------------------------------- energy calculations
    def _calc_required_energy(
        self, composition: Composition, target_ef_per_atom: float
    ) -> float:
        """Back-calculate the total DFT energy for a desired formation energy.

        Derivation
        ----------
        PhaseDiagram computes:

            E_f = E_total - Σ n_i · E_ref_i        (per formula unit)
            E_f / atom = E_total / N - Σ frac_i · E_ref_i

        Rearranging:

            E_total = N · (E_f/atom + Σ frac_i · E_ref_i)

        Parameters
        ----------
        composition : Composition
            Target composition (must only contain elements in the PD).
        target_ef_per_atom : float
            Desired formation energy in eV / atom.

        Returns
        -------
        float
            The total energy the PDEntry must carry so that
            ``PhaseDiagram.get_form_energy_per_atom`` returns *target_ef_per_atom*.
        """
        ref_e_per_atom = sum(
            composition.get_atomic_fraction(el) * self._pd.el_refs[el].energy_per_atom
            for el in composition.elements
        )
        return composition.num_atoms * (target_ef_per_atom + ref_e_per_atom)

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

        Parameters
        ----------
        entry_index : int
            Index into the working entries.
        new_ef_per_atom : float, optional
            Desired formation energy (see *units*).  Left unchanged if None.
        new_entropy : float, optional
            Entropy (see *units*).  Left unchanged if None.
        units : {'eV', 'J/mol'}, default 'eV'
            Units of the supplied values.  ``'eV'`` is eV/atom for energy and
            eV/atom/K for entropy; ``'J/mol'`` is J/mol and J/mol-atom/K.  Both
            are converted to the editor's internal eV basis.
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

        if new_ef_per_atom is not None:
            new_energy = self._calc_required_energy(old.composition, new_ef_per_atom)
            self._entries[entry_index] = PDEntry(
                old.composition,
                new_energy,
                name=old.name,
                attribute=old.attribute,
            )
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

        Parameters
        ----------
        formula : str
            Reduced formula (e.g. ``'GaRu'``, ``'MgCu2'``).
        ef_per_atom : float
            Desired formation energy (see *units*).
        name : str, optional
            Display name.  Defaults to the reduced formula.
        entropy : float, optional
            Entropy (see *units*).  Defaults to 0.
        units : {'eV', 'J/mol'}, default 'eV'
            Units of *ef_per_atom* and *entropy*.  ``'eV'`` is eV/atom and
            eV/atom/K; ``'J/mol'`` is J/mol and J/mol-atom/K.  Both are
            converted to the editor's internal eV basis.
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

        self._save_state(
            f"Add {formula} with Ef={self._fmt_energy(ef_per_atom)} {self._energy_unit}, "
            f"S={self._fmt_entropy(entropy)} {self._entropy_unit}"
        )

        total_energy = self._calc_required_energy(comp, ef_per_atom)
        self._entries.append(
            PDEntry(comp, total_energy, name=name or comp.reduced_formula)
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

        # Guard: don't orphan an element
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
        """Set the temperature (K) at which the Gibbs hull is displayed/applied."""
        self._temperature = float(temperature)
        if getattr(self, "_temp_slider", None) is not None:
            self._temp_slider.value = self._temperature
        self._refresh()

    @property
    def entries(self) -> list[dict]:
        """Read-only summary of the working entries (index, name, x_B, Ef, S)."""
        self._build_display()
        return [
            {
                "index": i,
                "name": e.name,
                "x": e.composition.get_atomic_fraction(self._el_b),
                "ef_per_atom": self._pd.get_form_energy_per_atom(e),
                "entropy": self._entropies[i],
                "stable": i in self._stable_idx,
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

    def _enthalpy_pd(self) -> PhaseDiagram:
        """A clean PhaseDiagram built from the edited enthalpy entries."""
        return PhaseDiagram(_copy_entries(self._entries), self._elements)

    def _build_phases(self):
        """Build a ``BinaryLiquid``-style phase list from the current edits.

        Solid phases are those stable on the Gibbs hull at the current
        temperature (so entropy-stabilized additions are retained while
        metastable DFT entries are dropped).  Each carries enthalpy (J/mol) from
        the T=0 hull and entropy converted from the editor's eV/atom/K to the
        J/mol-atom/K the model uses.  The trailing liquid placeholder mirrors
        :func:`gliquid.binary.build_phases_from_chull`.
        """
        components = self._bl.components
        component_data = self._bl.component_data
        self._build_display()

        phases = []
        # Elemental polymorphs (mirrors build_phases_from_chull)
        for i, comp in enumerate(components):
            for polymorph in component_data[comp].get("polymorphs", []):
                phases.append({
                    "name": polymorph["common_name"],
                    "comp": float(i),
                    "enthalpy": polymorph["enthalpy_J_per_mol"],
                    "entropy": polymorph["entropy_J_per_mol_K"],
                    "is_solution": False,
                    "points": [],
                })

        for i in sorted(self._stable_idx):
            entry = self._entries[i]
            composition = entry.composition.fractional_composition.as_dict().get(components[1], 0)
            if composition in [p["comp"] for p in phases]:
                continue  # element ground state already covered by a polymorph
            phases.append({
                "name": entry.name,
                "comp": composition,
                "enthalpy": J_PER_MOL_PER_EV * self._pd.get_form_energy_per_atom(entry),
                "entropy": J_PER_MOL_PER_EV * self._entropies[i],  # eV/atom/K -> J/mol-atom/K
                "is_solution": False,
                "points": [],
            })

        phases.sort(key=lambda x: x["comp"])
        phases.append({"name": "L", "is_solution": True, "points": []})
        return phases

    def apply(self):
        """Write the edits back to the source object (no arguments).

        Returns
        -------
        PhaseDiagram | BinaryLiquid
            For a PhaseDiagram source, the modified enthalpy PhaseDiagram.
            For a BinaryLiquid source, the same BinaryLiquid with ``dft_ch`` and
            ``phases`` updated in place.
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
        bl.dft_ch = self._enthalpy_pd()
        bl.phases = self._build_phases()
        return bl

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------
    def _build_ui(self):
        """Assemble all ipywidget components."""

        style = {"description_width": "80px"}

        # ---- Header ----
        self._header = widgets.HTML(
            f"<h3 style='margin-bottom:2px;'>Convex Hull Editor - "
            f"{self._el_a}-{self._el_b}</h3>"
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
            legend=dict(
                orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1
            ),
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

    # ------------------------------------------------------------------
    # UI refresh helpers
    # ------------------------------------------------------------------
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
            f"<b>Modify Existing Entry</b>  (E_f in {self._energy_unit}, "
            f"S in {self._entropy_unit})"
        )
        self._add_hint.value = (
            f"<b>Add New Entry</b>  (E_f in {self._energy_unit}, "
            f"S in {self._entropy_unit})"
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
        options = [
            (self._make_label(i, e), i)
            for i, e in enumerate(self._entries)
        ]
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
        sorted_entries = sorted(
            enumerate(self._entries), key=lambda t: self._entry_sort_key(t[1])
        )
        for idx, entry in sorted_entries:
            xb = entry.composition.get_atomic_fraction(self._el_b)
            ef = self._pd.get_form_energy_per_atom(entry)
            s = self._entropies[idx]
            g = self._gibbs_ef(idx)
            is_stable = idx in self._stable_idx
            ehull = 0.0 if is_stable else self._display_pd.get_e_above_hull(self._display_entries[idx])

            dot = "🟢" if is_stable else "⚪"
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

        g_header = (
            f"G@{self._temperature:.0f}K" if gibbs else f"G ({self._energy_unit})"
        )
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
            "</tr></thead><tbody>"
            + "\n".join(rows)
            + "</tbody></table></div>"
        )
        n_stable = len(self._stable_idx)
        n_total = len(self._entries)
        n_changes = len(self._history)
        summary = (
            f"<span style='font-size:12px; color:#555;'>"
            f"{n_stable} stable / {n_total} total entries  ·  "
            f"hull shown at T = {self._temperature:.0f} K  ·  "
            f"{n_changes} change{'s' if n_changes != 1 else ''} made</span>"
        )
        self._table_html.value = summary + table

    def _update_plot(self):
        """Redraw the convex-hull Plotly figure at the current temperature."""
        # Gather data ---------------------------------------------------------
        rows = []
        for i, entry in enumerate(self._entries):
            xb = entry.composition.get_atomic_fraction(self._el_b)
            rows.append((i, xb, self._disp_energy(self._gibbs_ef(i)), entry.name, i in self._stable_idx))

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
                marker=dict(
                    color="#2ca02c", size=11, line=dict(width=1, color="black")
                ),
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
        """Append a message to the change-log pane."""
        with self._log_output:
            print(msg)

    # ------------------------------------------------------------------
    # Callbacks
    # ------------------------------------------------------------------
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
        new_s = self._mod_s.value    # in current display units

        # Warn about elemental references
        if entry.composition.is_element:
            self._log(
                f"⚠ Warning: modifying elemental reference {entry.name} "
                "will shift ALL formation energies!"
            )

        old_ef = self._pd.get_form_energy_per_atom(entry)  # eV
        old_s = self._entropies[idx]                       # eV
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
        s = self._add_s.value    # in current display units
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
        # Carry any partly-entered Add-box values across the unit switch so the
        # number the user typed keeps the same physical meaning.
        self._add_ef.value = self._convert(self._add_ef.value, old_units, new_units)
        self._add_s.value = self._convert(self._add_s.value, old_units, new_units)
        self._refresh()
        self._log(f"Units set to {new_units}.")

    def _do_undo(self):
        self.undo()  # refreshes the display

    def _do_reset(self):
        self.reset()  # refreshes the display

    # ------------------------------------------------------------------
    # Display
    # ------------------------------------------------------------------
    def display(self):
        """Render the full editor widget inside a Jupyter notebook."""
        display(self._widget)

    def _repr_mimebundle_(self, **kwargs):
        """Allow rich display when the object is the last expression in a cell."""
        return self._widget._repr_mimebundle_(**kwargs)
