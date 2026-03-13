"""
Interactive Convex Hull Editor for binary phase diagrams.

Provides a Jupyter widget (ipywidgets + Plotly) for modifying a pymatgen-style
PhaseDiagram without writing code:

  - Modify formation energies of existing entries (eV/atom)
  - Add new entries by reduced formula + desired formation energy
  - Remove entries
  - Undo / reset changes
  - Export the modified PhaseDiagram

The key challenge this tool solves is the **back-calculation** of DFT total
energies.  PhaseDiagram computes formation energies via an internal referencing
scheme:

    E_f / atom = E_total / N - Σ(frac_i · E_ref_i)

When the user specifies a *desired* formation energy, the editor solves for the
total energy that reproduces it:

    E_total = N · (E_f_desired / atom + Σ(frac_i · E_ref_i))

Usage
-----
    from gliquid.hull_editor import ConvexHullEditor

    editor = ConvexHullEditor(phase_diagram)
    editor.display()          # show the interactive widget
    # ... make changes via the GUI ...
    new_pd = editor.get_phase_diagram()

Author: Joshua Willwerth (generated with assistance)
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
    phase_diagram : PhaseDiagram
        A **binary** PhaseDiagram object.  The editor will operate on a deep
        copy of its entries so the original object is never mutated.

    Raises
    ------
    ValueError
        If the supplied phase diagram is not binary.
    """

    # ------------------------------------------------------------------ init
    def __init__(self, phase_diagram: PhaseDiagram):
        if len(phase_diagram.elements) != 2:
            raise ValueError(
                f"ConvexHullEditor only supports binary phase diagrams.  "
                f"Got {len(phase_diagram.elements)} elements: "
                f"{[str(e) for e in phase_diagram.elements]}"
            )

        self._elements = list(phase_diagram.elements)
        self._el_a, self._el_b = self._elements

        # Keep an immutable snapshot for *Reset*
        self._original_entries = _copy_entries(phase_diagram.all_entries)

        # Working state
        self._entries: list[PDEntry] = _copy_entries(phase_diagram.all_entries)
        self._pd: PhaseDiagram = self._rebuild_pd()

        # Undo stack: list of (description, entries_snapshot)
        self._history: list[tuple[str, list[PDEntry]]] = []

        # Build and wire the UI
        self._build_ui()
        self._refresh()

    # ------------------------------------------------------ core bookkeeping
    def _rebuild_pd(self) -> PhaseDiagram:
        """Reconstruct the PhaseDiagram from the current working entries."""
        self._pd = PhaseDiagram(self._entries, self._elements)
        return self._pd

    def _save_state(self, description: str):
        """Push the current entries onto the undo stack."""
        self._history.append((description, _copy_entries(self._entries)))

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

    # ------------------------------------------------------------ public API
    def modify_entry(self, entry_index: int, new_ef_per_atom: float):
        """Change an existing entry's energy to achieve a target formation energy.

        Parameters
        ----------
        entry_index : int
            Index into ``self._entries``.
        new_ef_per_atom : float
            Desired formation energy in eV / atom.
        """
        old = self._entries[entry_index]
        old_ef = self._pd.get_form_energy_per_atom(old)

        self._save_state(
            f"Modify {old.name}: Ef {old_ef:+.4f} → {new_ef_per_atom:+.4f} eV/at"
        )

        new_energy = self._calc_required_energy(old.composition, new_ef_per_atom)
        self._entries[entry_index] = PDEntry(
            old.composition,
            new_energy,
            name=old.name,
            attribute=old.attribute,
        )
        self._rebuild_pd()

    def add_entry(self, formula: str, ef_per_atom: float, name: str | None = None):
        """Add a brand-new entry to the phase diagram.

        Parameters
        ----------
        formula : str
            Reduced formula (e.g. ``'GaRu'``, ``'Fe2Si'``).
        ef_per_atom : float
            Desired formation energy in eV / atom.
        name : str, optional
            Display name.  Defaults to the reduced formula.
        """
        comp = Composition(formula)
        for el in comp.elements:
            if el not in self._elements:
                raise ValueError(
                    f"Element {el} in '{formula}' is not part of the "
                    f"{self._el_a}–{self._el_b} phase diagram."
                )

        self._save_state(f"Add {formula} with Ef={ef_per_atom:+.4f} eV/at")

        total_energy = self._calc_required_energy(comp, ef_per_atom)
        self._entries.append(
            PDEntry(comp, total_energy, name=name or comp.reduced_formula)
        )
        self._rebuild_pd()

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
        self._rebuild_pd()

    def undo(self):
        """Revert the most recent change."""
        if not self._history:
            self._log("Nothing to undo.")
            return
        desc, prev = self._history.pop()
        self._entries = prev
        self._rebuild_pd()
        self._log(f"↩ Undone: {desc}")

    def reset(self):
        """Restore all entries to the state they were in at construction time."""
        self._history.clear()
        self._entries = _copy_entries(self._original_entries)
        self._rebuild_pd()
        self._log("🔄 Reset to original phase diagram.")

    def get_phase_diagram(self) -> PhaseDiagram:
        """Return the current (possibly modified) PhaseDiagram."""
        return self._pd

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------
    def _build_ui(self):
        """Assemble all ipywidget components."""

        style = {"description_width": "80px"}

        # ---- Header ----
        self._header = widgets.HTML(
            f"<h3 style='margin-bottom:2px;'>Convex Hull Editor — "
            f"{self._el_a}–{self._el_b}</h3>"
        )

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
        self._mod_btn = widgets.Button(
            description="  Apply",
            button_style="primary",
            icon="check",
            layout=widgets.Layout(width="110px"),
        )
        self._mod_btn.on_click(self._on_modify)

        modify_box = widgets.VBox(
            [
                widgets.HTML("<b>Modify Existing Entry</b>"),
                self._mod_dropdown,
                widgets.HBox([self._mod_ef, self._mod_btn]),
            ]
        )

        # ---- Add section ----
        self._add_formula = widgets.Text(
            description="Formula:",
            placeholder="e.g. GaRu, Fe2Si",
            layout=widgets.Layout(width="220px"),
            style=style,
        )
        self._add_ef = widgets.FloatText(
            description="E_f:",
            step=0.001,
            layout=widgets.Layout(width="200px"),
            style=style,
        )
        self._add_btn = widgets.Button(
            description="  Add Entry",
            button_style="success",
            icon="plus",
            layout=widgets.Layout(width="120px"),
        )
        self._add_btn.on_click(self._on_add)

        add_box = widgets.VBox(
            [
                widgets.HTML("<b>Add New Entry</b>  (eV / atom)"),
                widgets.HBox([self._add_formula, self._add_ef, self._add_btn]),
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
            layout=widgets.Layout(width="750px"),
        )

    # ------------------------------------------------------------------
    # UI refresh helpers
    # ------------------------------------------------------------------
    def _refresh(self):
        """Redraw everything after a state change."""
        self._update_plot()
        self._update_table()
        self._update_dropdowns()

    def _entry_sort_key(self, entry):
        """Sort entries by composition fraction of element B."""
        return entry.composition.get_atomic_fraction(self._el_b)

    def _make_label(self, idx: int, entry: PDEntry) -> str:
        """Descriptive one-liner for dropdown options."""
        xb = entry.composition.get_atomic_fraction(self._el_b)
        ef = self._pd.get_form_energy_per_atom(entry)
        is_stable = entry in self._pd.stable_entries
        if is_stable:
            status = "● on hull"
        else:
            ehull = self._pd.get_e_above_hull(entry)
            status = f"○ +{ehull:.4f} above hull"
        return f"[{idx}] {entry.name}  (x={xb:.3f})  Ef={ef:+.5f}  {status}"

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

        # Pre-fill the Ef input when a selection exists
        if self._mod_dropdown.value is not None:
            ef = self._pd.get_form_energy_per_atom(
                self._entries[self._mod_dropdown.value]
            )
            self._mod_ef.value = round(ef, 6)

    def _update_table(self):
        """Render an HTML summary table of all entries."""
        rows = []
        sorted_entries = sorted(
            enumerate(self._entries), key=lambda t: self._entry_sort_key(t[1])
        )
        for idx, entry in sorted_entries:
            xb = entry.composition.get_atomic_fraction(self._el_b)
            epa = entry.energy_per_atom
            ef = self._pd.get_form_energy_per_atom(entry)
            is_stable = entry in self._pd.stable_entries
            ehull = 0.0 if is_stable else self._pd.get_e_above_hull(entry)

            dot = "🟢" if is_stable else "⚪"
            rows.append(
                f"<tr>"
                f"<td>{dot}</td>"
                f"<td><b>{entry.name}</b></td>"
                f"<td>{xb:.3f}</td>"
                f"<td>{epa:.5f}</td>"
                f"<td>{ef:+.5f}</td>"
                f"<td>{ehull:+.5f}</td>"
                f"</tr>"
            )

        table = (
            "<div style='max-height:220px; overflow-y:auto; margin:4px 0;'>"
            "<table style='border-collapse:collapse; font-size:13px; width:100%;'>"
            "<thead><tr style='background:#f0f0f0;'>"
            "<th></th>"
            f"<th style='text-align:left; padding:2px 6px;'>Name</th>"
            f"<th style='padding:2px 6px;'>x<sub>{self._el_b}</sub></th>"
            "<th style='padding:2px 6px;'>E (eV/at)</th>"
            "<th style='padding:2px 6px;'>E<sub>f</sub> (eV/at)</th>"
            "<th style='padding:2px 6px;'>E<sub>hull</sub> (eV/at)</th>"
            "</tr></thead><tbody>"
            + "\n".join(rows)
            + "</tbody></table></div>"
        )
        n_stable = len(self._pd.stable_entries)
        n_total = len(self._entries)
        n_changes = len(self._history)
        summary = (
            f"<span style='font-size:12px; color:#555;'>"
            f"{n_stable} stable / {n_total} total entries  ·  "
            f"{n_changes} change{'s' if n_changes != 1 else ''} made</span>"
        )
        self._table_html.value = summary + table

    def _update_plot(self):
        """Redraw the convex-hull Plotly figure."""
        # Gather data ---------------------------------------------------------
        stable = sorted(
            self._pd.stable_entries,
            key=lambda e: e.composition.get_atomic_fraction(self._el_b),
        )
        hull_x = [e.composition.get_atomic_fraction(self._el_b) for e in stable]
        hull_y = [self._pd.get_form_energy_per_atom(e) for e in stable]
        hull_names = [e.name for e in stable]

        unstable = [
            e
            for e in self._pd.all_entries
            if e not in self._pd.stable_entries
        ]
        unstable_x = [
            e.composition.get_atomic_fraction(self._el_b) for e in unstable
        ]
        unstable_y = [self._pd.get_form_energy_per_atom(e) for e in unstable]
        unstable_names = [e.name for e in unstable]

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
                    "E<sub>f</sub> = %{y:.4f} eV/at<extra></extra>"
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
                        "E<sub>f</sub> = %{y:.4f} eV/at<extra></extra>"
                    ),
                )
            )

        # Apply to FigureWidget -----------------------------------------------
        with self._fig.batch_update():
            self._fig.data = []
        self._fig.add_traces(traces)

    def _log(self, msg: str):
        """Append a message to the change-log pane."""
        with self._log_output:
            print(msg)

    # ------------------------------------------------------------------
    # Callbacks
    # ------------------------------------------------------------------
    def _on_mod_selection(self, change):
        """When the modify-dropdown selection changes, pre-fill the Ef box."""
        idx = change.get("new")
        if idx is not None and 0 <= idx < len(self._entries):
            ef = self._pd.get_form_energy_per_atom(self._entries[idx])
            self._mod_ef.value = round(ef, 6)

    def _on_modify(self, _):
        idx = self._mod_dropdown.value
        if idx is None:
            self._log("⚠ Select an entry first.")
            return
        entry = self._entries[idx]
        new_ef = self._mod_ef.value

        # Warn about elemental references
        if entry.composition.is_element:
            self._log(
                f"⚠ Warning: modifying elemental reference {entry.name} "
                "will shift ALL formation energies!"
            )

        old_ef = self._pd.get_form_energy_per_atom(entry)
        self.modify_entry(idx, new_ef)
        self._refresh()
        self._log(
            f"✓ Modified {entry.name}: "
            f"Ef {old_ef:+.5f} → {new_ef:+.5f} eV/at"
        )

    def _on_add(self, _):
        formula = self._add_formula.value.strip()
        ef = self._add_ef.value
        if not formula:
            self._log("⚠ Enter a formula.")
            return
        try:
            self.add_entry(formula, ef)
            self._refresh()
            self._log(f"✓ Added {formula} with Ef={ef:+.5f} eV/at")
        except Exception as exc:
            self._log(f"⚠ {exc}")

    def _on_remove(self, _):
        idx = self._rm_dropdown.value
        if idx is None:
            self._log("⚠ Select an entry first.")
            return
        name = self._entries[idx].name
        self.remove_entry(idx)
        self._refresh()
        # remove_entry logs its own warning on failure, success otherwise:
        if name not in [e.name for e in self._entries]:
            self._log(f"✓ Removed {name}")

    def _do_undo(self):
        self.undo()
        self._refresh()

    def _do_reset(self):
        self.reset()
        self._refresh()

    # ------------------------------------------------------------------
    # Display
    # ------------------------------------------------------------------
    def display(self):
        """Render the full editor widget inside a Jupyter notebook."""
        display(self._widget)

    def _repr_mimebundle_(self, **kwargs):
        """Allow rich display when the object is the last expression in a cell."""
        return self._widget._repr_mimebundle_(**kwargs)
