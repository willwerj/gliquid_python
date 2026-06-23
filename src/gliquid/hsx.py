"""
Authors: Abrar Rauf, Joshua Willwerth
Last Modified: June 23 2026
Description: This script takes the phase energy data in the form of enthalpy (H), entropy (S) and composition (X)
and performs transformations to composition-temperature (TX) phase diagrams with well-defined coexistence boundaries
GitHub: https://github.com/AbrarRauf
ORCID: https://orcid.org/0000-0001-5205-0075
"""
from __future__ import annotations

import re
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from scipy.spatial import ConvexHull
from collections import defaultdict

# ---------------------------------------------------------------------------
# Plotting geometry constants (must match the layout used in HSX.plot_tx).
# These are the single source of truth for the pixel<->data conversions used
# by label-collision detection; the tests import them from this module.
# ---------------------------------------------------------------------------
_FIG_W_PX = 750
_FIG_H_PX = 600
_MARGIN_L_PX = 80   # plotly default left margin
_MARGIN_R_PX = 55
_MARGIN_T_PX = 72
_MARGIN_B_PX = 72
_PLOT_W_PX = _FIG_W_PX - _MARGIN_L_PX - _MARGIN_R_PX  # 615
_PLOT_H_PX = _FIG_H_PX - _MARGIN_T_PX - _MARGIN_B_PX  # 456

# Text-metric estimates (fractions of the font size, in px).
_CHAR_W_FACTOR = 0.6   # mean glyph advance width
_LINE_H_FACTOR = 1.2   # line height
_SUB_W_FACTOR = 0.7    # subscript glyph relative width

# Consistent gap between a bottom-anchored label's box and the lower plot boundary,
# as a fraction of the temperature span. Keeps compound labels off the axis line.
_BOTTOM_PAD_FRAC = 0.015

# Greek polymorph prefixes -> unicode letters.
_GREEK_MAP = {
    'alpha': 'α', 'beta': 'β', 'gamma': 'γ', 'delta': 'δ', 'epsilon': 'ε',
    'zeta': 'ζ', 'eta': 'η', 'theta': 'θ', 'iota': 'ι', 'kappa': 'κ',
    'lambda': 'λ', 'mu': 'μ', 'nu': 'ν', 'xi': 'ξ', 'omicron': 'ο',
    'pi': 'π', 'rho': 'ρ', 'sigma': 'σ', 'tau': 'τ', 'upsilon': 'υ',
    'phi': 'φ', 'chi': 'χ', 'psi': 'ψ', 'omega': 'ω',
}

# Long crystal-structure names -> short forms.
_STRUCT_ABBREV = {
    'orthorhombic': 'ortho',
    'rhombohedral': 'rhomb',
    'monoclinic': 'mono',
    'tetragonal': 'tetra',
    'hexagonal': 'hex',
    'trigonal': 'trig',
    'triclinic': 'tric',
    'body centered cubic': 'bcc',
    'face centered cubic': 'fcc',
    'hexagonal close packed': 'hcp',
    'complex cubic a12': 'A12',
    'complex cubic a13': 'A13',
}

_SUB_TAG_RE = re.compile(r'<sub>(.*?)</sub>')
_TAG_PHASE_TAG_RE = r'(?:rt|lt|ht\d*|r\d*)'  # room/low/high-temperature ordinal tags


def _text_glyph_width(text: str) -> float:
    """Estimate visible text width in glyph units (chars), discounting subscripts and HTML tags."""
    # Mark subscript bodies, then strip remaining tags, then weight characters.
    marked = _SUB_TAG_RE.sub(lambda m: '\x00' * len(m.group(1)), text)
    marked = re.sub(r'<[^>]+>', '', marked)
    return sum(_SUB_W_FACTOR if ch == '\x00' else 1.0 for ch in marked)


def _estimate_label_box(label: dict, xlim: tuple[float, float],
                        ylim: tuple[float, float]) -> tuple[float, float, float, float]:
    """Estimate a label's bounding box in data coordinates.

    Returns ``(cx, cy, half_w, half_h)`` where (cx, cy) is the box centre. This is the
    single source of truth for the fragile font/pixel constants; tests import it directly.
    """
    font_size = label.get('font_size', 12)
    glyphs = _text_glyph_width(str(label.get('text', '')))
    w_px = max(glyphs * _CHAR_W_FACTOR * font_size, 0.5 * font_size)
    h_px = _LINE_H_FACTOR * font_size
    if abs(label.get('textangle', 0)) == 90:
        w_px, h_px = h_px, w_px

    span_x = (xlim[1] - xlim[0]) or 1.0
    span_y = (ylim[1] - ylim[0]) or 1.0
    half_w = (w_px / (_PLOT_W_PX / span_x)) / 2.0
    half_h = (h_px / (_PLOT_H_PX / span_y)) / 2.0

    cx, cy = float(label['x']), float(label['y'])
    xanchor = label.get('xanchor', 'center')
    if xanchor == 'left':
        cx += half_w
    elif xanchor == 'right':
        cx -= half_w
    yanchor = label.get('yanchor', 'middle')
    if yanchor == 'bottom':
        cy += half_h
    elif yanchor == 'top':
        cy -= half_h
    return cx, cy, half_w, half_h


def _box_overlap(b1: tuple, b2: tuple) -> tuple[float, float]:
    """Return positive (x, y) penetration depths if AABBs ``b1``/``b2`` overlap, else <=0."""
    return (b1[2] + b2[2] - abs(b1[0] - b2[0]),
            b1[3] + b2[3] - abs(b1[1] - b2[1]))


def _parse_elemental_phase(name: str):
    """Parse an elemental solid-solution label into ``(greek_word, element, struct, tag)``.

    Accepts both space- and hyphen-separated greek prefixes (e.g. ``"alpha Mn (bcc)"`` and
    ``"alpha-Ga (orthorhombic)"``). Returns ``None`` for compounds (e.g. ``"ZrMn2"``) so the
    caller falls through to formula subscripting.
    """
    s = str(name).strip()
    greek_word = None
    struct_prefix = None
    # "<Structure words> <Element>" form (e.g. "Diamond cubic Si", "Face centered cubic Al"):
    # a capitalised structure name preceding a trailing element symbol. The element is the last
    # token; everything before it is the structure. Only fires when the leading token is a
    # capitalised, non-greek word (so "alpha Mn" stays a greek polymorph and a hypothetical
    # "Na Cl" — an element-symbol prefix — is not misread as element Cl).
    toks = s.split()
    if (len(toks) >= 2 and re.fullmatch(r'[A-Z][a-z]?', toks[-1])
            and toks[0][:1].isupper() and toks[0].lower() not in _GREEK_MAP
            and not re.fullmatch(r'[A-Z][a-z]?', ' '.join(toks[:-1]))):
        return None, toks[-1], ' '.join(toks[:-1]), None
    # A leading token before a space/hyphen is either a greek polymorph prefix
    # ("alpha-Ga (orthorhombic)") or a lowercase structure tag ("fcc-Al", "bcc-Cr"),
    # provided what follows starts with an element symbol.
    m = re.match(r'^([A-Za-z0-9]+)[ \-](.+)$', s)
    if m and re.match(r'^[A-Z][a-z]?(?:$|[\s(])', m.group(2).strip()):
        prefix = m.group(1)
        if prefix.lower() in _GREEK_MAP:
            greek_word = prefix.lower()
            s = m.group(2).strip()
        elif prefix[0].islower():
            struct_prefix = prefix
            s = m.group(2).strip()

    m2 = re.match(r'^([A-Z][a-z]?)\s*(?:\(([^)]*)\))?\s*(.*)$', s)
    if not m2:
        return None
    element = m2.group(1)
    struct = (m2.group(2) or '').strip() or struct_prefix
    tag = (m2.group(3) or '').strip() or None
    # Reject compounds: a trailing remainder containing letters that is not a temperature tag
    # means another element/formula followed (e.g. "ZrMn2" -> element "Zr", remainder "Mn2").
    if tag and re.search(r'[A-Za-z]', tag) and not re.fullmatch(_TAG_PHASE_TAG_RE, tag, re.IGNORECASE):
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
    m = re.search(r'\b([A-D]\d{1,2})\b', s)  # Strukturbericht token, e.g. A13, B2
    if m:
        return m.group(1)
    if re.search(r'\d', s) and ('-' in s or '/' in s):
        return ''  # space-group symbol (Fm-3m, R-3m, P6_3/mmc) -> omit
    if len(s) <= 6:
        return s
    return ''  # unrecognised long descriptor -> omit


def _subscript_formula(name: str) -> str:
    """Wrap stoichiometric digit-runs (those following a letter or ')') in <sub> tags."""
    return re.sub(r'(?<=[A-Za-z\)])(\d+)', r'<sub>\1</sub>', str(name))


class HSX:
    """Handles enthalpy (H), entropy (S), and composition (X) transformations for TX phase diagrams."""

    def __init__(self, data_dict: dict, conds: list[float], use_filter_2=False):
        """Initializes the HSX instance with provided phase data and conditions."""
        self.phases = data_dict['phases']
        self.comps = data_dict['comps']
        self.conds = conds
        self.df = pd.DataFrame(data_dict['data'])
        self.phase_color_remap = {}
        self.simplices = []
        self.final_phases = []
        self.df_tx = pd.DataFrame()
        self.use_filter_2 = use_filter_2
    
        # Data scaling
        s_scaler = 100
        h_scaler = 10000
        self.df.columns = ['X [Fraction]', 'S [J/mol/K]', 'H [J/mol]', 'Phase']
        self.df['S [J/mol/K]'] /= s_scaler
        self.df['H [J/mol]'] /= h_scaler

        # Color Mapping
        color_array = px.colors.qualitative.Pastel
        inter_phases = [p for p in self.phases if p != 'L']
        self.color_map = {phase: color_array[i % len(color_array)] for i, phase in enumerate(inter_phases)}
        self.color_map['L'] = 'cornflowerblue'
        self.df['Colors'] = self.df['Phase'].map(self.color_map)

        # Data extraction for convex hull calculation
        df_inter = self.df[self.df['Phase'] != 'L']
        df_liq = self.df[self.df['Phase'] == 'L']
        self.liq_points = df_liq[['X [Fraction]', 'S [J/mol/K]', 'H [J/mol]']].to_numpy()
        self.inter_points = df_inter[['X [Fraction]', 'S [J/mol/K]', 'H [J/mol]']].to_numpy()
        self.points = self.df[['X [Fraction]', 'S [J/mol/K]', 'H [J/mol]']].to_numpy()
        self.scaler = h_scaler / s_scaler

    def hull(self) -> np.ndarray:
        """Computes the lower convex hull of an N-dimensional Xi-S-H space."""
        dim = self.points.shape[1]

        # Initialize bounds for Xi
        x_list = [[1 if j == i - 1 else 0 for j in range(dim - 2)] for i in range(dim - 1)]
        x_list[0] = [0] * (dim - 2)

        # Compute S and H bounds
        s_min, s_extr = np.min(self.points[:, -2]), np.max([self.liq_points[0, -2], self.liq_points[-1, -2]])
        h_max = np.max(self.points[:, -1])
        upper_bound = 20 * h_max

        # Generate fictitious points
        liq_fict_coords = np.column_stack((self.liq_points[:, 0], self.liq_points[:, 1],
                                            np.full(len(self.liq_points), upper_bound)))
        fict_coords = np.vstack([
                                np.append(x_list[i], [s_min, upper_bound]) for i in range(dim - 1)
                                ] + [
                                np.append(x_list[i], [s_extr, upper_bound]) for i in range(dim - 1)
                                ])

        fict_points = np.vstack((fict_coords, liq_fict_coords))
        new_points = np.vstack((self.points, fict_points))
        # Compute convex hull
        new_hull = ConvexHull(new_points, qhull_options="Qt i")

        n_real = len(self.points)
        all_simplices = new_hull.simplices

        # Filter 1: discard simplices with any fictitious vertex (index >= n_real)
        mask_no_fict = np.all(all_simplices < n_real, axis=1)
        self.simplices = all_simplices[mask_no_fict]

        if self.use_filter_2:
            # Filter 2: discard simplices where all 3 vertices are intermetallic (non-liquid) points
            is_inter = (self.df['Phase'] != 'L').values
            inter_counts = np.sum(is_inter[self.simplices], axis=1)
            self.simplices = self.simplices[inter_counts < 3]

        return self.simplices

    def compute_tx(self) -> tuple[pd.DataFrame, list, np.ndarray, np.ndarray]:
        """Computes the TX phase diagram transformation."""
        self.hull()
        temps, valid_simplices, new_phases = [], [], []
        for simplex in self.simplices:
            A, B, C = self.points[simplex]
            n = np.cross(B - A, C - A).astype(float)
            # Degenerate / near-vertical facets yield n[2] ~ 0 and non-physical infinite temperatures.
            if np.isclose(n[2], 0.0, atol=1e-12):
                continue
            T = (-n[1] / n[2]) * self.scaler
            if np.isfinite(T):
                temps.append(T)
                valid_simplices.append(simplex)
                new_phases.append([self.df.loc[simplex[i], 'Phase'] for i in range(3)])
        
        temps = np.array(temps)
        self.final_phases = np.array(new_phases)

        data = [
            [self.points[vertex][0], temps[i], labels[j], self.color_map.get(labels[j])]
            for i, simplex in enumerate(valid_simplices)
            for j, vertex in enumerate(simplex)
            for labels in [self.final_phases[i]]  # Extract labels once per simplex
        ]
        
        self.df_tx = pd.DataFrame(data, columns=['x', 't', 'label', 'color'])
        
        phase_remap = defaultdict(list)
        for entry in data:
            phase_remap[entry[2]].append([entry[0], entry[1]])
        self.phase_color_remap = dict(zip(self.df_tx['label'], self.df_tx['color']))
        return self.df_tx, self.final_phases, np.array(valid_simplices), temps

    
    def plot_tx_scatter(self) -> go.Figure:
        # self.df_tx = self.compute_tx()[0]   
        # convert all temps to Celsius
        self.df_tx['t'] = self.df_tx['t'] - 273.15

        # Create a scatter plot using Plotly Express
        fig = px.scatter(self.df_tx, x='x', y='t', color='label',
                         color_discrete_map=self.phase_color_remap,
                         title=f'{self.comps[0]}-{self.comps[1]} Binary Phase Diagram',
                         width=960, height=700)

        fig.update_traces(marker=dict(size=12))
        # Update the layout to include a legend
        fig.update_layout(showlegend=True)

        # Define axis limits for the 't' axis
        fig.update_layout(
            yaxis=dict(range=[self.conds[0], self.conds[1] + 100], ticksuffix="  "),
            xaxis=dict(range=[0, 1]),
            xaxis_title=f'X_{self.comps[1]}',
            yaxis_title='T [C]',
            plot_bgcolor='white',
            showlegend=True,
            font_size=22
        )

        fig.update_xaxes(
            mirror=True,
            ticks="inside",
            showline=True,
            linecolor='gray',
            linewidth=2,
            tickcolor='gray'
        )

        fig.update_yaxes(
            mirror=True,
            ticks="inside",
            showline=True,
            linecolor='gray',
            linewidth=2,
            tickcolor='gray'
        )

        # Show the plot
        return fig

    def plot_hsx(self) -> go.Figure:
        # Create a figure
        self.simplices = self.hull()
        fig = go.Figure()

        # Scatter plot
        scatter = go.Scatter3d(
            x=self.df['X [Fraction]'], y=self.df['S [J/mol/K]'], z=self.df['H [J/mol]'],
            mode='markers',
            marker=dict(size=6, opacity=1, color=self.df['Colors']),
            name='Scatter',
            showlegend=False
        )

        fig.add_trace(scatter)

        for simplex in self.simplices:
            x_coords = self.points[simplex, 0]
            y_coords = self.points[simplex, 1]
            z_coords = self.points[simplex, 2]

            i = np.array([0])
            j = np.array([1])
            k = np.array([2])

            trace = go.Mesh3d(x=x_coords, y=y_coords, z=z_coords, alphahull=5, opacity=0.3, color='cyan', i=i, j=j, k=k,
                              name='Simplex')

            # Add both the triangle and vertex traces to the figure
            fig.add_trace(trace)

        # Create legend entries
        legend_elements = []

        for name, color in self.color_map.items():
            legend_elements.append(dict(x=0, y=0, xref='paper', yref='paper', text=name, marker=dict(color=color)))

        # Add legend entries
        for entry in legend_elements:
            fig.add_trace(go.Scatter(x=[None], y=[None], mode='lines+text', marker=dict(color=entry['marker']['color']),
                                     name=entry['text']))

        # Update layout and labels
        fig.update_layout(
            scene=dict(xaxis_title='X', yaxis_title='S [J/mol/K]', zaxis_title='H [J/mol]'),
            legend=dict(itemsizing='constant'),
            font_size=15
        )

        # Show the 3D plot
        return fig
    

    def liquidus_invariants(self) -> tuple[dict, list, dict]:
        """Extracts eutectic, peritectic, and congruent melting points from the computed TX phase diagram."""
        self.df_tx, self.final_phases, final_simplices, final_temps = self.compute_tx()
        self.df_tx['t'] -= 273.15
        final_temps -= 273.15

        compositions = np.array([[vertex[0] for vertex in self.points[simplex]] for simplex in final_simplices])

        combined_list = []
        for i in range(len(compositions)):
            row_dict = {}
            for j in range(len(compositions[i])):
                row_dict[compositions[i][j]] = self.final_phases[i][j]
            if len(row_dict) == 2:
                for key in row_dict.keys():
                    if key == 0.0:
                        row_dict[key] = self.comps[0]
                    elif key == 1.0:
                        row_dict[key] = self.comps[1]
            combined_list.append([final_temps[i], row_dict])

        int_phases = [p for p in self.phases if p not in [self.comps[0], self.comps[1], 'L']]

        inv_points = {'Eutectics': [], 'Peritectics': [], 'Congruent Melting': [],
                      'Misc Gaps': [], 'Solid Ties': []}
        peritectic_phases, non_triples = [], []

        for temp, comb_dict in combined_list:

            sorted_dict = dict(sorted(comb_dict.items()))
            comp, phase = list(sorted_dict.keys()), list(sorted_dict.values())

            if len(comp) == 3:
                if len(set(phase)) == 3:
                    if phase[1] == 'L':
                        inv_points['Eutectics'].append([temp, comp[1], comp, phase])
                    else:
                        inv_points['Peritectics'].append([temp, comp[1], comp, phase])
                        peritectic_phases.append(phase[1])
                else:
                    non_triples.append([temp, comp, phase])

        congruents_init = []
        for temp, comp, phase in non_triples:
            if phase[0] == 'L' and phase[2] != 'L':
                comp_diff = abs(comp[0] - comp[1])
                if comp_diff > 0.012:
                    inv_points['Misc Gaps'].append([temp, comp[1], comp, phase])
            elif phase[0] != 'L':
                comp_diff = abs(comp[1] - comp[2])
                if comp_diff > 0.012:
                    inv_points['Misc Gaps'].append([temp, comp[1], comp, phase])
            phase = [p for p in phase if p != 'L']
            if phase and phase[0] in int_phases and phase[0] not in peritectic_phases:
                congruents_init.append([temp, comp[0], comp, phase])

        grouped_data = defaultdict(list)
        for entry in congruents_init:
            grouped_data[entry[3][0]].append(entry)

        inv_points['Congruent Melting'] = [max(entries, key=lambda x: x[0]) for entries in grouped_data.values()]

        # --- Solid-solid tie lines from 2-vertex (collapsed-triangle) simplices. A eutectic or
        # peritectic between two adjacent solids collapses to a solid-solid hull edge whenever the
        # liquid vertex is not a distinct third phase -- e.g. two close-set line compounds, or a
        # eutectic pinned just below a low-melting element. Those facets are 2-vertex simplices the
        # 3-vertex classification above never sees, so the horizontal tie at the top of that
        # two-phase field would go unmarked. Recover the hottest tie per solid pair, dropping any
        # that merely restates an already-detected invariant span or that spuriously bridges the two
        # pure elements (a liquid miscibility-gap / monotectic artifact). ---
        solid_pair_ties = {}  # frozenset(phaseA, phaseB) -> [temp, comp_mid, comps, phases]
        for temp, comb_dict in combined_list:
            if len(comb_dict) != 2:
                continue
            (cA, pA), (cB, pB) = sorted(comb_dict.items())
            if pA == pB or pA == 'L' or pB == 'L':
                continue
            if abs(cB - cA) < 0.012:            # negligible composition gap -> not a real tie
                continue
            # An element-to-element span (cA~0, cB~1) is the top of the terminal (A)+(B) two-phase
            # field: in a (near-)immiscible system the hottest such solid-solid edge sits exactly at
            # the lower element's melting point, a genuine degenerate-eutectic horizontal (Hg-V,
            # Ag-Mo, Cr-La). It is kept, NOT skipped -- monotectic / miscibility-gap artifacts carry
            # an 'L' vertex and were already filtered out above.
            key = frozenset((pA, pB))
            if key not in solid_pair_ties or temp > solid_pair_ties[key][0]:
                solid_pair_ties[key] = [temp, (cA + cB) / 2, [cA, cB], [pA, pB]]

        existing_spans = [(min(c), max(c)) for k in ('Eutectics', 'Peritectics', 'Misc Gaps')
                          for _, _, c, _ in inv_points[k]]
        for entry in solid_pair_ties.values():
            lo, hi = min(entry[2]), max(entry[2])
            if any(abs(elo - lo) < 0.02 and abs(ehi - hi) < 0.02 for elo, ehi in existing_spans):
                continue                         # already drawn as a eutectic/peritectic
            inv_points['Solid Ties'].append(entry)

        # Normalize invariant-point numeric fields to built-in Python floats.
        for inv_type, entries in inv_points.items():
            inv_points[inv_type] = [
                [float(temp), float(comp_mid), [float(c) for c in comp], [str(p) for p in phase]]
                for temp, comp_mid, comp, phase in entries
            ]

        count_dict = {key: len(value) for key, value in inv_points.items()}

        return inv_points, combined_list, count_dict
    
    # ------------------------------------------------------------------
    # Plotting helpers (label formatting, tie-line detection, collisions)
    # ------------------------------------------------------------------
    def _abbreviate_phase_name(self, name: str, all_names: list[str]) -> str:
        """Format a phase label: greek polymorph prefixes, subscripted stoichiometries,
        abbreviated crystal structures.

        Elemental solid solutions render as ``(αMn) bcc``; the greek prefix is dropped when
        the element has only one phase present in ``all_names`` (e.g. ``(Fe) bcc``). Compounds
        render with subscripts (``ZrMn2`` -> ``ZrMn<sub>2</sub>``). ``"Liquid"``/``"L"`` -> ``"L"``.
        """
        name = str(name).strip()
        if name in ('L', 'Liquid'):
            return 'L'

        parsed = _parse_elemental_phase(name)
        if parsed is not None:
            greek_word, element, struct, tag = parsed
            same_element = sum(
                1 for other in all_names
                if (_parse_elemental_phase(str(other).strip()) or (None, None, None, None))[1] == element
            )
            prefix = _GREEK_MAP.get(greek_word, '') if (greek_word and same_element > 1) else ''
            label = f'({prefix}{element})'
            suffix = []
            if struct:
                abbr = _abbrev_structure(struct)
                if abbr:
                    suffix.append(abbr)
            if tag:
                suffix.append(tag)
            if suffix:
                label += ' ' + ' '.join(suffix)
            return label

        return _subscript_formula(name)

    def _merge_close_values(self, values: list[float], tol: float) -> list[float]:
        """Cluster values whose sorted consecutive differences are <= ``tol``; return cluster means."""
        vals = sorted(float(v) for v in values)
        if not vals:
            return []
        groups = [[vals[0]]]
        for v in vals[1:]:
            if v - groups[-1][-1] <= tol:
                groups[-1].append(v)
            else:
                groups.append([v])
        return [sum(g) / len(g) for g in groups]

    @staticmethod
    def _curve_crossings_at_temp(pts: list[tuple[float, float]], temp: float,
                                 temp_tol: float) -> list[float]:
        """Return interpolated x-values where a polyline crosses ``temp`` (within ``temp_tol``)."""
        crossings = []
        for (x0, t0), (x1, t1) in zip(pts, pts[1:]):
            t_lo, t_hi = (t0, t1) if t0 <= t1 else (t1, t0)
            if t_lo - temp_tol <= temp <= t_hi + temp_tol:
                if abs(t1 - t0) < 1e-9:
                    crossings.append((x0 + x1) / 2.0)
                else:
                    frac = max(0.0, min(1.0, (temp - t0) / (t1 - t0)))
                    crossings.append(x0 + frac * (x1 - x0))
        if len(pts) == 1 and abs(pts[0][1] - temp) <= temp_tol:
            crossings.append(pts[0][0])
        return crossings

    def _detect_tie_lines(self, invariant_temps: list[float], boundary_curves: list[list[tuple]],
                          plot_xlim: tuple[float, float], temp_tol: float = 1.0,
                          x_tol: float = 0.5) -> list[dict]:
        """Detect horizontal tie lines at each invariant temperature.

        A boundary curve is treated as a *solid* phase boundary when its x-extent is within
        ``x_tol`` (a near-vertical line). For each invariant temperature the tie line spans the
        outermost solid-boundary crossings; if only one solid boundary crosses, it is paired
        with the nearest liquidus (non-vertical) crossing. Returns ``{temp, x_start, x_end}`` dicts.
        """
        classified = []
        for curve in boundary_curves:
            pts = [(float(x), float(t)) for x, t in curve]
            if not pts:
                continue
            xs = [p[0] for p in pts]
            classified.append((pts, (max(xs) - min(xs)) <= x_tol))

        tie_lines = []
        for temp in invariant_temps:
            solid_x, liq_x = [], []
            for pts, is_solid in classified:
                hits = self._curve_crossings_at_temp(pts, temp, temp_tol)
                (solid_x if is_solid else liq_x).extend(hits)

            solid_merged = self._merge_close_values(solid_x, x_tol) if solid_x else []
            if len(solid_merged) >= 2:
                tie_lines.append({'temp': float(temp),
                                  'x_start': min(solid_merged), 'x_end': max(solid_merged)})
            elif len(solid_merged) == 1 and liq_x:
                sx = solid_merged[0]
                nearest = min(liq_x, key=lambda lx: abs(lx - sx))
                lo, hi = sorted((sx, nearest))
                if hi - lo > x_tol:
                    tie_lines.append({'temp': float(temp), 'x_start': lo, 'x_end': hi})
        return tie_lines

    def _resolve_label_collisions(self, labels: list[dict], xlim: tuple[float, float],
                                  ylim: tuple[float, float], max_iterations: int = 50,
                                  ceiling=None, tie_segments=None) -> list[dict]:
        """Iteratively nudge overlapping labels apart (primarily in y for vertical text).

        ``ceiling`` is an optional ``top(x) -> T`` callable (the liquidus envelope); when given,
        each label is kept below it so labels never cross into the liquid field. ``tie_segments``
        (list of ``(x0, x1, T)``) lets compound labels shift up out of any tie line they straddle.
        Returns copies with adjusted positions; labels displaced from their (post-ceiling) home gain
        ``showarrow=True`` and retain ``home_x``/``home_y`` so the caller can draw a leader arrow.
        """
        out = [dict(lbl) for lbl in labels]
        for lbl in out:
            lbl['x'] = float(lbl['x'])
            lbl['y'] = float(lbl['y'])

        span_y = (ylim[1] - ylim[0]) or 1.0
        one_px_y = span_y / _PLOT_H_PX
        ceil_margin = 0.6 * one_px_y * _LINE_H_FACTOR * 12  # ~ small gap below the liquidus
        bottom_pad = _BOTTOM_PAD_FRAC * span_y  # consistent gap above the lower boundary
        tie_segments = tie_segments or []

        def apply_tie_clearance(l):
            # Shift an in-band (non-pinned, non-float) label up out of any tie line crossing its box.
            if l.get('pin') or l.get('above_liquidus') or not tie_segments:
                return
            cx, cy, half_w, half_h = _estimate_label_box(l, xlim, ylim)
            # Only a tie that genuinely bisects the box counts: a tie sitting *above* the box top
            # does not cross the label, and lifting toward it (then having ``apply_ceiling`` pull the
            # box back down) lands the label straddling that tie -- the deep-eutectic failure on
            # Au-Sm/SmAu6 and Er-Rh/Er3Rh, where the liquidus ceiling sits just above the eutectic
            # tie. The strict ``< cy + half_h`` upper bound keeps such a placement intact.
            blocking = [T for (x0, x1, T) in tie_segments
                        if x1 >= cx - half_w and x0 <= cx + half_w
                        and cy - half_h < T < cy + half_h]
            if blocking:
                shift = (max(blocking) + ceil_margin) - (cy - half_h)
                if shift > 0:
                    l['y'] += shift

        def apply_ceiling(l):
            if ceiling is None or l.get('above_liquidus'):
                return  # labels deliberately floated above the liquidus are exempt
            cx, cy, half_w, half_h = _estimate_label_box(l, xlim, ylim)
            # The box must clear the liquidus across its full width, so use the lowest liquidus
            # temperature spanned by the box (the curve descends across a vertical label).
            samples = [ceiling(x) for x in (cx - half_w, cx, cx + half_w)]
            finite = [c for c in samples if np.isfinite(c)]
            if not finite:
                return
            c = min(finite)
            if cy + half_h > c - ceil_margin:
                l['y'] -= (cy + half_h) - (c - ceil_margin)

        def clamp_bounds(l):
            _, cy, _, half_h = _estimate_label_box(l, xlim, ylim)
            # Bottom-anchored labels (compounds) keep a consistent pad above the lower boundary;
            # labels deliberately floated above the liquidus are exempt from the pad.
            floor = ylim[0] + (0.0 if l.get('above_liquidus') else bottom_pad)
            if cy - half_h < floor:
                l['y'] += floor - (cy - half_h)
            elif cy + half_h > ylim[1] and not l.get('above_liquidus'):
                # Floated labels are meant to sit above the liquidus; never pull them back down.
                l['y'] += ylim[1] - (cy + half_h)

        # Pull labels below the liquidus *before* recording home, so the ceiling adjustment alone
        # does not trigger leader arrows (only collision displacement should). Pinned labels were
        # already placed in a valid slot by the packer and are never moved.
        for l in out:
            if not l.get('pin'):
                apply_tie_clearance(l)
                apply_ceiling(l)
                clamp_bounds(l)
            l.setdefault('home_x', l['x'])
            l.setdefault('home_y', l['y'])

        if len(out) >= 2:
            for _ in range(max_iterations):
                boxes = [_estimate_label_box(l, xlim, ylim) for l in out]
                moved = False
                for i in range(len(out)):
                    for j in range(i + 1, len(out)):
                        pen_x, pen_y = _box_overlap(boxes[i], boxes[j])
                        if pen_x <= 0 or pen_y <= 0:
                            continue
                        pin_i, pin_j = out[i].get('pin'), out[j].get('pin')
                        if pin_i and pin_j:
                            continue  # both fixed (packer guarantees these do not overlap)
                        shift = pen_y / 2.0 + one_px_y
                        if pin_i:        # only j may move
                            out[j]['y'] += (pen_y + one_px_y) * (1 if out[j]['y'] >= out[i]['y'] else -1)
                        elif pin_j:      # only i may move
                            out[i]['y'] += (pen_y + one_px_y) * (1 if out[i]['y'] >= out[j]['y'] else -1)
                        elif out[i]['y'] <= out[j]['y']:
                            out[i]['y'] -= shift
                            out[j]['y'] += shift
                        else:
                            out[i]['y'] += shift
                            out[j]['y'] -= shift
                        moved = True
                for l in out:
                    if not l.get('pin'):
                        apply_tie_clearance(l)
                        apply_ceiling(l)
                        clamp_bounds(l)
                if not moved:
                    break

        for l in out:
            if abs(l['y'] - l['home_y']) > one_px_y or abs(l['x'] - l['home_x']) > 1e-6:
                l['showarrow'] = True
        return out

    @staticmethod
    def _liquidus_top_fn(liq_df: pd.DataFrame, assessed_pts: list | None, combine: str = 'max'):
        """Return ``top(x)`` combining the generated (and assessed) liquidus curves.

        ``combine='max'`` gives the upper envelope (for placing things *above* the liquidus, e.g.
        the 'L' label and legend); ``combine='min'`` gives the lower envelope (for keeping solid
        phase labels *below* both curves). Returns ``-inf`` where neither curve is defined.
        """
        lx = liq_df['x'].to_numpy(dtype=float)
        lt = liq_df['t'].to_numpy(dtype=float)
        order = np.argsort(lx)
        lx, lt = lx[order], lt[order]

        ax = at = None
        if assessed_pts:
            ax = np.array([p[0] for p in assessed_pts], dtype=float)
            at = np.array([p[1] for p in assessed_pts], dtype=float)
            aorder = np.argsort(ax)
            ax, at = ax[aorder], at[aorder]

        reduce_fn = max if combine == 'max' else min

        def top(x: float) -> float:
            vals = []
            if lx.size and lx.min() <= x <= lx.max():
                vals.append(float(np.interp(x, lx, lt)))
            if ax is not None and ax.size and ax.min() <= x <= ax.max():
                vals.append(float(np.interp(x, ax, at)))
            return reduce_fn(vals) if vals else -np.inf

        return top

    def _place_liquid_label(self, liq_df: pd.DataFrame, assessed_pts: list | None,
                            xlim: tuple[float, float], ylim: tuple[float, float],
                            font_size: float = 14) -> tuple[float, float]:
        """Place the ``L`` label in the widest sufficiently-tall empty band above the liquidus."""
        top = self._liquidus_top_fn(liq_df, assessed_pts)
        xs = np.linspace(xlim[0] + 2, xlim[1] - 2, 49)
        tops = np.array([(top(x) if np.isfinite(top(x)) else ylim[0]) for x in xs])
        gaps = ylim[1] - tops
        min_gap = _LINE_H_FACTOR * font_size * (ylim[1] - ylim[0]) / _PLOT_H_PX

        # Find the widest contiguous band tall enough for the label.
        ok = gaps >= min_gap
        best = None  # (run_length, mean_gap, i, j)
        i, n = 0, len(xs)
        while i < n:
            if ok[i]:
                j = i
                while j + 1 < n and ok[j + 1]:
                    j += 1
                cand = (j - i, float(gaps[i:j + 1].mean()), i, j)
                if best is None or cand[0] > best[0] or (cand[0] == best[0] and cand[1] > best[1]):
                    best = cand
                i = j + 1
            else:
                i += 1

        if best is None:  # nothing tall enough: fall back to the single tallest band
            k = int(np.argmax(gaps))
            return float(xs[k]), float(tops[k] + gaps[k] / 2.0)

        # Within the widest band, place at the gap-weighted centroid so the label tracks the
        # open liquid field (centre for symmetric liquidi; shifted toward the deeper side).
        i, j = best[2], best[3]
        seg_x, seg_gap = xs[i:j + 1], gaps[i:j + 1]
        cx = float(np.sum(seg_x * seg_gap) / np.sum(seg_gap))
        return cx, float(top(cx) + (ylim[1] - top(cx)) / 2.0)

    @staticmethod
    def _pack_labels(regions: list[tuple], half_hs: list[float], ceiling: float,
                     tie_temps: list[float], floor: float, margin: float,
                     gap: float = 0.0) -> list:
        """Place each vertical label inside its OWN stability band (``regions[i] = (t_bottom,
        t_top)``), in a tie-free sub-interval of ``[floor, ceiling]``.

        A label is only placed in an interval that overlaps its own region, so it is never shifted
        onto a neighbouring polymorph; labels that do not fit in their band return ``None`` (the
        caller floats those). Returns centre y values aligned with ``regions``.
        """
        n = len(regions)
        results = [None] * n
        if not np.isfinite(ceiling) or ceiling <= floor:
            return results

        cuts = sorted(t for t in tie_temps if floor < t < ceiling)
        intervals, lo = [], floor
        for t in cuts:
            if t - margin > lo:
                intervals.append((lo, t - margin))
            lo = max(lo, t + margin)
        if ceiling > lo:
            intervals.append((lo, ceiling))

        next_bottom = floor
        for idx in sorted(range(n), key=lambda i: regions[i][0]):
            rb, rt = regions[idx]
            rt = min(rt, ceiling)
            hh = half_hs[idx]
            mid = 0.5 * (rb + rt)
            for ilo, ihi in intervals:
                if ihi <= rb or ilo >= rt:
                    continue  # interval does not overlap this label's own band
                bottom = max(ilo, rb, next_bottom)
                top = min(ihi, rt)
                if bottom + 2 * hh <= top:
                    y = min(max(mid, bottom + hh), top - hh)
                    results[idx] = y
                    next_bottom = y + hh + gap
                    break
        return results

    @staticmethod
    def _assign_compound_sides(comps: list[float], half_ws: list[float],
                               xlim: tuple[float, float] = (0.0, 100.0),
                               standoff: float = 1.5, edge_standoff: float = 2.5) -> list[float]:
        """Choose an x for each compound label so adjacent labels fall in DISTINCT two-phase gaps.

        Each compound at composition ``c`` may place its (vertical) label in the gap to its left
        ``(c_prev, c)`` or right ``(c, c_next)``. Processing left-to-right, a label takes its left
        gap by default but switches to the right gap when the left gap was already claimed by the
        previous label or is too narrow for the label box; compounds within 10 at.% of an element
        edge are forced inward to clear the element solid-solution label. Returns one x per input
        compound (in input order). Where no conflict-free side exists the label keeps the roomier
        side and the y-resolver separates the resulting overlap by height.
        """
        n = len(comps)
        order = sorted(range(n), key=lambda i: comps[i])
        result = [0.0] * n
        tiny = 1e-6
        prev_used_right = False      # did the previous label claim the gap shared with this one's left?
        prev_box_right = xlim[0]
        for pos, i in enumerate(order):
            c, hw = float(comps[i]), float(half_ws[i])
            left_nb = comps[order[pos - 1]] if pos > 0 else xlim[0]
            right_nb = comps[order[pos + 1]] if pos < n - 1 else xlim[1]
            near_left_edge, near_right_edge = c - xlim[0] < 10.0, xlim[1] - c < 10.0
            so = edge_standoff if (near_left_edge or near_right_edge) else standoff
            left_x, right_x = c - so, c + so
            left_ok = (left_x - hw > left_nb + tiny and not prev_used_right
                       and left_x - hw > prev_box_right + tiny)
            right_ok = right_x + hw < right_nb - tiny
            if near_left_edge:
                chosen, used_right = right_x, True
            elif near_right_edge:
                chosen, used_right = left_x, False
            elif left_ok:
                chosen, used_right = left_x, False
            elif right_ok:
                chosen, used_right = right_x, True
            else:  # forced overlap -> keep the roomier side, let the y-resolver stack
                if (c - left_nb) >= (right_nb - c):
                    chosen, used_right = left_x, False
                else:
                    chosen, used_right = right_x, True
            result[i] = chosen
            prev_used_right = used_right
            prev_box_right = chosen + hw
        return result

    @staticmethod
    def _place_compound_y(c0: float, ceiling: float, tie_temps: list[float], half_h: float,
                          pad: float, margin: float) -> float | None:
        """Bottom-align a compound label in the LOWEST tie-free sub-band of ``[c0, ceiling]`` that
        is tall enough. Returns the label centre y, or ``None`` if no sub-band fits.

        The bottom-most band keeps ``pad`` above ``c0``; bands above a tie keep ``margin`` above it.
        This realises "drop the label to the bottom, but skip up to the next two-phase region when a
        tie line would cross it" (bugs 1 & 2).
        """
        if not np.isfinite(ceiling) or ceiling <= c0:
            return None
        cuts = sorted(t for t in tie_temps if c0 < t < ceiling)
        intervals, lo = [], c0
        for t in cuts:
            if t - margin > lo:
                intervals.append((lo, t - margin))
            lo = max(lo, t + margin)
        if ceiling > lo:
            intervals.append((lo, ceiling))
        for ilo, ihi in intervals:
            # The floor region keeps ``pad`` above c0; regions above a tie already start at
            # ``tie + margin``, so no extra offset is needed there.
            bottom = ilo + (pad if abs(ilo - c0) < 1e-9 else 0.0)
            if bottom + 2 * half_h <= ihi:
                return bottom + half_h
        return None

    def _place_legend(self, liq_df: pd.DataFrame, assessed_pts: list | None,
                      xlim: tuple[float, float], n_entries: int, max_label_chars: int,
                      float_top_by_side: dict | None = None) -> dict:
        """Pick the top corner with the most clearance and expand ``self.conds[1]`` (the upper
        temperature limit) so the legend sits inside the plot area without overlapping either
        liquidus or the floated polymorph labels on that side. Never placed above the plot.
        """
        float_top_by_side = float_top_by_side or {0: -np.inf, 100: -np.inf}
        span_x = (xlim[1] - xlim[0]) or 1.0
        w_px = max_label_chars * _CHAR_W_FACTOR * 15 + 40   # text + colour swatch
        h_px = n_entries * 21 + 12
        w_data = w_px / (_PLOT_W_PX / span_x)
        inset_x = 0.01 * span_x
        top = self._liquidus_top_fn(liq_df, assessed_pts, combine='max')

        def obstacle(x0, x1, side):
            vals = [top(x) for x in np.linspace(x0, x1, 15)]
            vals = [v for v in vals if np.isfinite(v)]
            liq = max(vals) if vals else -np.inf
            return max(liq, float_top_by_side.get(side, -np.inf))

        # (xanchor, x_paper, x0, x1, side)
        corners = [
            ('right', 0.99, xlim[1] - inset_x - w_data, xlim[1] - inset_x, 100),
            ('left', 0.01, xlim[0] + inset_x, xlim[0] + inset_x + w_data, 0),
        ]
        # Prefer the corner with the lowest obstacle (least y-range expansion, avoids floats).
        corners.sort(key=lambda c: obstacle(c[2], c[3], c[4]))
        xanchor, xp, x0, x1, side = corners[0]
        obs = obstacle(x0, x1, side)

        # Expand conds[1] so the legend band (h_px tall, at the top) clears that obstacle.
        f = (h_px + 6) / _PLOT_H_PX
        if np.isfinite(obs) and f < 1.0:
            needed_c1 = (obs - f * self.conds[0]) / (1.0 - f)
            if needed_c1 > self.conds[1]:
                self.conds[1] = needed_c1

        return {'xanchor': xanchor, 'yanchor': 'top', 'x': xp, 'y': 0.99, 'font': dict(size=15)}

    def plot_tx(self, pred: bool = False, digitized_liquidus: list = None,
                polymorph_transitions: list[dict] | None = None,
                imputed_phases: set | None = None) -> go.Figure:
        """Plots the binary phase diagram from computed phase boundaries and invariant points.

        Args:
            pred (bool): If True, use prediction color scheme for the liquidus.
            digitized_liquidus (list): Digitized experimental liquidus data points.
            polymorph_transitions (list[dict]): List of elemental polymorph transitions, each dict with keys:
                'name' (str), 'comp_x_pct' (float, 0 or 100), 'transition_temp_C' (float),
                'ground_state_name' (str) for the phase below the transition.
            imputed_phases (set): Names of phases imputed by phase-energy imputation; their
                solid boundary lines are drawn dashed and given a single legend entry.
        """
        imputed_phases = imputed_phases or set()
        liq_inv = self.liquidus_invariants()
        inv_points, combined_list = liq_inv[:2]
        
        new_tx = []
        for comb in combined_list:
            temp = comb[0]
            comb_dict = comb[1]
            sorted_dict = {k: v for k, v in sorted(comb_dict.items())}
            comp = list(sorted_dict.keys())
            phase = list(sorted_dict.values())
            if len(comp) == 2:
                new_tx.append([temp, comp, phase])
            else:
                if phase[0] == 'L' and phase[1] == 'L':  # Liquid-Liquid-Solid or Liquid-Liquid-Liquid
                    comp.pop(0)
                    phase.pop(0)
                    new_tx.append([temp, comp, phase])

                elif phase[1] == 'L' and phase[2] == 'L':  # Solid-Liquid-Liquid
                    comp.pop(2)
                    phase.pop(2)
                    new_tx.append([temp, comp, phase])
                else:
                    new_tx.append([temp, comp, phase])
        
        temp_df_tx = [[x, t, phase[j], self.color_map.get(phase[j])] 
                  for t, comp, phase in new_tx for j, x in enumerate(comp)]
        new_df_tx = pd.DataFrame(temp_df_tx, columns=['x', 't', 'label', 'color'])
        new_df_tx['x'] *= 100

        liq_df = self.df_tx[self.df_tx['label'] == 'L'].copy()
        liq_df['x'] *= 100
        liq_df.sort_values(by=['x', 't'], inplace=True)
        liq_df.drop_duplicates(subset='x', keep='first', inplace=True)

        # Use raw compute_tx scatter points for solids so polymorphs at identical
        # composition are preserved (combined_list/new_df_tx can collapse them).
        solid_df = self.df_tx[self.df_tx['label'] != 'L'].copy()
        solid_df['x'] *= 100
        solid_df.drop_duplicates(subset=['x', 't', 'label'], keep='first', inplace=True)

        lhs_tm, rhs_tm = liq_df.iloc[0]['t'], liq_df.iloc[-1]['t']
        max_liq, min_liq = liq_df['t'].max(), liq_df['t'].min()

        fig = go.Figure()

        # Assessed (digitized) liquidus, in plot units (at.% and °C); None when absent.
        assessed_pts = ([[p[0] * 100, p[1] - 273.15] for p in digitized_liquidus]
                        if digitized_liquidus else None)
        if assessed_pts:  # update liquidus temperature range based on digitized liquidus
            max_liq = max(max_liq, max(p[1] for p in assessed_pts))
            min_liq = min(min_liq, min(p[1] for p in assessed_pts))
            fig.add_trace(
                go.Scatter(x=[p[0] for p in assessed_pts], y=[p[1] for p in assessed_pts],
                            mode='lines', line=dict(color='#B82E2E', dash='dash')))

        # expand temperature range based on liquidus extremes
        if max_liq > self.conds[1]:
            self.conds[1] = max_liq + 0.1 * (self.conds[1] - self.conds[0])
        if min_liq < self.conds[0]:
            self.conds[0] = max(-273.15, min_liq - 0.1 * (self.conds[1] - self.conds[0]))

        # Cap excessive headroom above the liquidus. The 'L' label and legend are auto-placed,
        # so no artificial y-extension is needed to accommodate them.
        headroom_cap = max_liq + 0.18 * (max_liq - self.conds[0])
        if self.conds[1] > headroom_cap:
            self.conds[1] = headroom_cap
        
        solid_phases = [p for p in self.phases if p not in [self.comps[0], self.comps[1], 'L']]
        for phase in solid_phases:
            phase_df = solid_df[solid_df['label'] == phase]
            if phase_df.empty:
                continue
            
            # expand temperature range based on minimum decomposition temperatures of solid phases
            phase_decomp_temp = phase_df['t'].max()
            if phase_decomp_temp - 0.1 * (self.conds[1] - self.conds[0]) < self.conds[0]:
                self.conds[0] = max(-273.15, phase_decomp_temp - 0.1 * (self.conds[1] - self.conds[0]))

        # Build per-phase lower-extension limits. For polymorphs at the same composition,
        # only the lowest-temperature phase is extended to plot bottom; upper polymorphs
        # are extended down only to the top of the lower polymorph to avoid full overlap.
        phase_rows = []
        comp_groups = defaultdict(list)
        for phase in solid_phases:
            phase_df = solid_df[solid_df['label'] == phase]
            if phase_df.empty:
                continue
            solid_comp = float(phase_df['x'].iloc[0])
            t_min = float(phase_df['t'].min())
            t_max = float(phase_df['t'].max())
            phase_rows.append((phase, phase_df, solid_comp))
            comp_groups[round(solid_comp, 6)].append({
                'phase': phase,
                't_min': t_min,
                't_max': t_max,
            })

        phase_low_ext = {}
        for group in comp_groups.values():
            ordered = sorted(group, key=lambda d: (d['t_min'], d['t_max']))
            prev_top = None
            for idx, entry in enumerate(ordered):
                if idx == 0:
                    low_ext = -273.15
                else:
                    low_ext = prev_top if prev_top is not None else -273.15
                    # Never extend above where this phase already starts.
                    low_ext = min(low_ext, entry['t_min'])
                phase_low_ext[entry['phase']] = low_ext
                prev_top = entry['t_max']

        # Collect polymorph phase names for separate labeling
        polymorph_names = set()
        if polymorph_transitions:
            polymorph_names = {pt['name'] for pt in polymorph_transitions}
            polymorph_names |= {pt.get('ground_state_name', '') for pt in polymorph_transitions}

        # All phase names (used by _abbreviate_phase_name to decide whether to keep greek prefixes).
        all_phase_names = set(solid_df['label'].unique()) | polymorph_names
        # Labels are collected here, then de-collided in one pass before being drawn.
        label_dicts: list[dict] = []
        # Tie lines already drawn (keyed) for dedup, plus their (x0, x1, T) for label avoidance.
        drawn_tie_lines: set = set()
        tie_segments: list[tuple] = []

        def _tie_key(temp, x0, x1):
            return (round(float(temp), 1), round(min(x0, x1), 1), round(max(x0, x1), 1))

        def _add_tie(x0, x1, temp, dedup=True):
            key = _tie_key(temp, x0, x1)
            if dedup and key in drawn_tie_lines:
                return
            drawn_tie_lines.add(key)
            tie_segments.append((min(float(x0), float(x1)), max(float(x0), float(x1)), float(temp)))
            tie = px.line(x=[x0, x1], y=[temp, temp])
            tie.update_traces(line=dict(color='Silver'))
            fig.add_trace(tie.data[0])

        # Compound (non-polymorph) labels are placed after the conds-convergence loop so they can
        # use the final temperature range and the drawn tie lines (region-fit + escalation below).
        compound_phase_list = []
        for phase, phase_df, solid_comp in phase_rows:
            low_ext_temp = phase_low_ext.get(phase, -273.15)
            new_row_df = pd.DataFrame(
                [{'x': solid_comp, 't': low_ext_temp, 'label': phase, 'color': self.color_map.get(phase)}],
                  columns=phase_df.columns)
            phase_df = pd.concat([phase_df, new_row_df], ignore_index=True)
            line = px.line(phase_df, x='x', y='t', color='label', color_discrete_map=self.phase_color_remap)

            trace = line.data[0]
            if phase in imputed_phases:
                existing_line = trace.line.to_plotly_json() if trace.line is not None else {}
                existing_line['dash'] = 'dash'
                trace.line = existing_line
            fig.add_trace(trace)

            # Skip label here for polymorphs — they are labeled separately below
            if phase not in polymorph_names:
                comp_key = round(float(solid_comp), 6)
                compound_phase_list.append({
                    'phase': phase,
                    'comp': float(solid_comp),
                    't_min': float(low_ext_temp),
                    't_max': float(phase_df['t'].max()),
                    'is_stacked': len(comp_groups.get(comp_key, [])) > 1,
                    'text': self._abbreviate_phase_name(phase, all_phase_names),
                })

        for key in inv_points.keys():
            if key in ['Eutectics', 'Peritectics', 'Misc Gaps', 'Solid Ties']:
                for temp, _, comps, _ in inv_points[key]:
                    comps = [x * 100 for x in comps]
                    _add_tie(min(comps), max(comps), temp)

        # --- Polymorph (element solid-solution) tie lines and label regions, derived from the
        # ACTUAL DFT traces in solid_df (their on-hull stability ranges), not the experimental
        # transition temperatures (which do not coincide with the plotted trace boundaries). ---
        liq_pts = list(zip(liq_df['x'].tolist(), liq_df['t'].tolist())) if not liq_df.empty else []
        poly_traces = {0: [], 100: []}
        compound_bounds = []  # (comp, t_min, t_max) for non-polymorph solids
        for phase, _pdf, solid_comp in phase_rows:
            sub = solid_df[solid_df['label'] == phase]
            if sub.empty:
                continue
            if phase in polymorph_names:
                side = 0 if solid_comp < 50 else 100
                poly_traces[side].append({
                    'name': phase,
                    # RAW lower extent of the DFT trace (e.g. ~-273 for a ground state) so a later
                    # lowering of conds[0] can extend this region downward in-band. Clamped to the
                    # *current* conds[0] only at layout time, never frozen here.
                    't_bottom': phase_low_ext.get(phase, self.conds[0]),
                    't_max': float(sub['t'].max()),
                })
            else:
                compound_bounds.append((solid_comp, float(sub['t'].min()), float(sub['t'].max())))

        def _adjacent_boundary_x(temp, side):
            """Nearest phase boundary inward from the element edge at temp: a liquidus crossing, a
            compound, or — when nothing lies between — the opposite element's solid (so a polymorph
            tie spans to the opposite polymorph)."""
            xs = list(self._curve_crossings_at_temp(
                liq_pts, temp, temp_tol=0.01 * (self.conds[1] - self.conds[0]))) if liq_pts else []
            xs += [c for c, t0, t1 in compound_bounds if t0 - 1 <= temp <= t1 + 1]
            opp = 100 if side == 0 else 0
            if temp < (rhs_tm if opp == 100 else lhs_tm):  # opposite element is solid at temp
                xs.append(float(opp))
            inward = [x for x in xs if (x > side if side == 0 else x < side)]
            return min(inward, key=lambda x: abs(x - side)) if inward else None

        polymorph_regions = []  # (region_dict, comp_pct), placed after the safety-net ties
        for side, traces in poly_traces.items():
            elt_melt = lhs_tm if side == 0 else rhs_tm
            for tr in sorted(traces, key=lambda d: d['t_max']):
                # Tie line at the trace top (polymorph transition or melting), drawn all the way to
                # the adjacent phase boundary so it fully intersects it. Skip the top polymorph's
                # congruent melt at the pure element (a point on the liquidus, not a tie line).
                T = tr['t_max']
                congruent_melt = abs(T - elt_melt) < 0.02 * (self.conds[1] - self.conds[0])
                if self.conds[0] < T < self.conds[1] and not congruent_melt:
                    bx = _adjacent_boundary_x(T, side)
                    if bx is not None and abs(bx - side) >= 0.6:
                        _add_tie(side, bx, T)
                # Label region carries the RAW trace extent; visibility is decided at layout time
                # against the (possibly lowered) conds[0]. A region whose entire trace sits below
                # the current floor is dropped.
                if tr['t_max'] > self.conds[0]:
                    polymorph_regions.append(({'name': tr['name'],
                                               't_bottom': tr['t_bottom'],
                                               't_max': tr['t_max']}, side))

        # --- Safety-net tie lines: connect each incongruently-melting solid phase to the
        # liquidus at the top of its boundary. The eutectic/peritectic invariants above already
        # cover most cases; this recovers any solid<->liquidus tie that invariant detection
        # missed, while staying local (no full-width spans). ---
        if not liq_df.empty and phase_rows:
            liq_top = self._liquidus_top_fn(liq_df, None)
            span = self.conds[1] - self.conds[0]
            congruent_tol = 0.02 * span     # phase top within this of the liquidus -> congruent
            max_tie_span = 40.0             # at.%; tie lines are local, never near-full-width
            for phase, _pdf, comp_s in phase_rows:
                if phase in polymorph_names:
                    continue  # polymorph ties handled above from their traces
                sub = solid_df[solid_df['label'] == phase]
                if sub.empty:
                    continue
                t_max = float(sub['t'].max())
                liq_here = liq_top(comp_s)
                if np.isfinite(liq_here) and abs(liq_here - t_max) <= congruent_tol:
                    continue  # congruent melter -> meets liquidus at a point, no horizontal tie
                crossings = self._curve_crossings_at_temp(liq_pts, t_max, temp_tol=0.02 * span)
                if not crossings:
                    continue
                nearest = min(crossings, key=lambda xc: abs(xc - comp_s))
                if not (0.6 < abs(nearest - comp_s) <= max_tie_span):
                    continue
                _add_tie(comp_s, nearest, t_max)

        # --- Place deferred polymorph/element labels. Preference per label: (1) in-band in its own
        # narrow field; (2) for the lowest-T polymorph, lower conds[0] so it fits (>= absolute zero);
        # (3) in the wider adjacent two-phase region just inside the element, below the liquidus;
        # (4) floated above the liquidus with a leader arrow. ---
        xlim = (0.0, 100.0)
        liq_floor = self._liquidus_top_fn(liq_df, assessed_pts, combine='min')  # below BOTH curves
        liq_high = self._liquidus_top_fn(liq_df, assessed_pts, combine='max')   # above BOTH curves
        _ABS_ZERO = -273.15
        _CLEAR_PX = 10.0          # clearance (px) between a floated label and the liquidus below it
        clear_frac = _CLEAR_PX / _PLOT_H_PX
        gap_frac = 0.004          # stack gap as a fraction of the y-span
        entry_labels = (['Assessed Liquidus'] if digitized_liquidus else [])
        entry_labels.append('Predicted Liquidus' if pred else 'Fitted Liquidus')

        def _env_min(fn, cx, half_w):
            vals = [v for v in (fn(x) for x in np.linspace(cx - half_w, cx + half_w, 11)) if np.isfinite(v)]
            return min(vals) if vals else None

        def _env_max(fn, cx, half_w):
            vals = [v for v in (fn(x) for x in np.linspace(cx - half_w, cx + half_w, 11)) if np.isfinite(v)]
            return max(vals) if vals else None

        # Precompute static per-side data (half-heights as span-invariant fractions, ceilings, ties).
        sides_data = {}
        for side_comp in (0, 100):
            side = [r for (r, c) in polymorph_regions if c == side_comp]
            if not side:
                continue
            label_x = 1.5 if side_comp == 0 else 98.5   # small standoff, like the compound labels
            fx = 8.0 if side_comp == 0 else 92.0        # inward x for two-phase / floated labels
            texts = [self._abbreviate_phase_name(r['name'], all_phase_names) for r in side]
            half_hs_frac, half_w = [], 1.2
            for text in texts:
                probe = {'x': label_x, 'y': 0.0, 'text': text, 'textangle': -90,
                         'font_size': 12, 'xanchor': 'center', 'yanchor': 'middle'}
                _, _, half_w, hh = _estimate_label_box(probe, xlim, (self.conds[0], self.conds[1]))
                half_hs_frac.append(hh / (self.conds[1] - self.conds[0]))
            # Dodge compound columns: a floated/relocated polymorph label sits at ``fx`` with a leader
            # arrow back to the element edge. If a compound shares that composition (e.g. Be12V at
            # 8 at.% vs the side-0 default fx=8) the float lands on top of the compound label. Shift
            # fx to the nearest compound-free x (keeping clear of the in-band element label too) so
            # neither the box nor the arrow collides; an empty same-side gap leaves fx at default.
            obstacles = [label_x] + [cp['comp'] for cp in compound_phase_list
                                     if (cp['comp'] < 50) == (side_comp == 0)]
            clearance = 2 * half_w + 0.8
            if any(abs(o - fx) < clearance for o in obstacles):
                lo, hi = (2.0, 48.0) if side_comp == 0 else (52.0, 98.0)
                cands = [x for x in np.arange(lo, hi + 1e-6, 0.5)
                         if all(abs(o - x) >= clearance for o in obstacles)]
                if cands:
                    fx = min(cands, key=lambda x: abs(x - fx))
            sides_data[side_comp] = dict(
                side=side, label_x=label_x, fx=fx, texts=texts, half_hs_frac=half_hs_frac,
                ceil_raw=_env_min(liq_floor, label_x, half_w),
                fx_ceil_raw=_env_min(liq_floor, fx, half_w),
                float_base=_env_max(liq_high, fx, half_w),
                tie_temps=[T for x0, x1, T in tie_segments if x1 >= label_x - half_w and x0 <= label_x + half_w],
                fx_tie_temps=[T for x0, x1, T in tie_segments if x1 >= fx - half_w and x0 <= fx + half_w],
                bottom_idx=min(range(len(side)), key=lambda i: side[i]['t_bottom']))

        def _layout_side(side_comp, c0, c1):
            """Classify each label: ('inband', y) | ('lower', needed_c0) | ('twophase', y) |
            ('float', half_h_frac), evaluated at the given conds.

            Relocated (``twophase``) labels are packed JOINTLY in the inward two-phase column so two
            never share a slot, and ``_pack_labels`` fills them bottom-to-top in their own
            t_bottom order (monotonic => non-crossing leader arrows / shortest longest arrow)."""
            d = sides_data[side_comp]
            span = c1 - c0
            margin, gap = 0.012 * span, 0.004 * span
            half_hs = [hf * span for hf in d['half_hs_frac']]
            ceil = (d['ceil_raw'] - margin) if d['ceil_raw'] is not None else c1
            fx_ceil = (d['fx_ceil_raw'] - margin) if d['fx_ceil_raw'] is not None else c1
            regions_ty = [(max(r['t_bottom'], c0), min(r['t_max'], c1)) for r in d['side']]
            ys = self._pack_labels(regions_ty, half_hs, ceil, d['tie_temps'], c0, margin, gap)
            out = [None] * len(d['side'])
            for i in range(len(d['side'])):
                if ys[i] is not None:
                    out[i] = ('inband', ys[i])
            # Lowest-T polymorph: lower conds[0] so it fits in its bottom gap.
            bi = d['bottom_idx']
            if out[bi] is None:
                r = d['side'][bi]
                r_top = min(r['t_max'], c1)
                obstacles = sorted(T for T in d['tie_temps'] if c0 < T < min(r_top, ceil))
                gap_top = obstacles[0] if obstacles else min(r_top, ceil)
                needed_c0 = gap_top - 2 * half_hs[bi] - 1.5 * margin
                if needed_c0 >= _ABS_ZERO:
                    out[bi] = ('lower', needed_c0)
            # Remaining labels relocate to the inward two-phase region, packed jointly (shared
            # cursor) so they cannot coincide; any that still do not fit float above the liquidus.
            reloc = [i for i in range(len(d['side'])) if out[i] is None]
            if reloc:
                regions_fx = [(max(d['side'][i]['t_bottom'], c0), fx_ceil) for i in reloc]
                fx_ys = self._pack_labels(regions_fx, [half_hs[i] for i in reloc], fx_ceil,
                                          d['fx_tie_temps'], c0, margin, gap)
                for k, i in enumerate(reloc):
                    out[i] = ('twophase', fx_ys[k]) if fx_ys[k] is not None else ('float', d['half_hs_frac'][i])
            return out

        # Precompute compound-label columns: each compound's x is assigned to a distinct two-phase
        # gap (left/right of the compound); per-column ceiling/ties drive region-fit + escalation.
        compound_cols = []
        if compound_phase_list:
            comps_x = [cp['comp'] for cp in compound_phase_list]
            init_ylim = (self.conds[0], self.conds[1])
            half_ws = []
            for cp in compound_phase_list:
                probe = {'x': cp['comp'], 'y': 0.0, 'text': cp['text'], 'textangle': -90,
                         'font_size': 12, 'xanchor': 'center', 'yanchor': 'middle'}
                half_ws.append(_estimate_label_box(probe, xlim, init_ylim)[2])
            assigned_x = self._assign_compound_sides(comps_x, half_ws, xlim)
            for cp, ax in zip(compound_phase_list, assigned_x):
                # Stacked compounds (>1 phase at the same composition) sit just inside the diagram on
                # one side, separated by their own stability bands rather than left/right gaps.
                label_x = (cp['comp'] + (1.5 if cp['comp'] <= 50 else -1.5)) if cp['is_stacked'] else ax
                probe = {'x': label_x, 'y': 0.0, 'text': cp['text'], 'textangle': -90,
                         'font_size': 12, 'xanchor': 'center', 'yanchor': 'middle'}
                _, _, hw, hh = _estimate_label_box(probe, xlim, init_ylim)
                compound_cols.append(dict(
                    comp=cp['comp'], label_x=label_x, text=cp['text'], is_stacked=cp['is_stacked'],
                    t_min=cp['t_min'], t_max=cp['t_max'], half_w=hw,
                    half_h_frac=hh / (init_ylim[1] - init_ylim[0]),
                    ceil_raw=_env_min(liq_floor, label_x, hw),
                    float_base=_env_max(liq_high, label_x, hw),
                    tie_temps=[T for x0, x1, T in tie_segments
                               if x1 >= label_x - hw and x0 <= label_x + hw]))

        def _layout_compounds(c0, c1):
            """Per compound column: ('inband', y) | ('lower', needed_c0) | ('float', half_h_frac).
            ``inband`` puts the label at the bottom of the lowest two-phase region that fits (with a
            consistent bottom pad), skipping up past any tie line that would cross it (bugs 1 & 2)."""
            span = c1 - c0
            margin, gap, pad = 0.012 * span, 0.004 * span, _BOTTOM_PAD_FRAC * span
            out = []
            for col in compound_cols:
                hh = col['half_h_frac'] * span
                ceil = (col['ceil_raw'] - margin) if col['ceil_raw'] is not None else c1
                if col['is_stacked']:
                    region = (max(col['t_min'], c0), min(col['t_max'], c1))
                    y = self._pack_labels([region], [hh], ceil, col['tie_temps'], c0, margin, gap)[0]
                else:
                    y = self._place_compound_y(c0, ceil, col['tie_temps'], hh, pad, margin)
                if y is not None:
                    out.append(('inband', y))
                    continue
                # Escalate: lower conds[0] so the bottom region grows enough; else float.
                cuts = sorted(T for T in col['tie_temps'] if c0 < T < ceil)
                gap_top = cuts[0] if cuts else ceil
                needed_c0 = gap_top - 2 * hh - pad - margin
                if np.isfinite(gap_top) and needed_c0 >= _ABS_ZERO:
                    out.append(('lower', needed_c0))
                else:
                    out.append(('float', col['half_h_frac']))
            return out

        # Fixed point: 'lower' lowers conds[0]; floats + the legend raise conds[1]. Both change the
        # y-span (hence label heights), so iterate until conds (and placement) are stable.
        legend_params, float_top_by_side = {}, {0: -np.inf, 100: -np.inf}
        for _ in range(8):
            c0_before, c1_before = self.conds[0], self.conds[1]
            float_top_by_side = {0: -np.inf, 100: -np.inf}
            for side_comp, d in sides_data.items():
                placements = _layout_side(side_comp, self.conds[0], self.conds[1])
                for p in placements:
                    if p[0] == 'lower':
                        self.conds[0] = max(min(self.conds[0], p[1]), _ABS_ZERO)
                float_fracs = [p[1] for p in placements if p[0] == 'float']
                if float_fracs and d['float_base'] is not None:
                    m = clear_frac + sum(2 * hf for hf in float_fracs) + gap_frac * len(float_fracs)
                    if m < 0.95:
                        side_c1 = (d['float_base'] - m * self.conds[0]) / (1.0 - m)
                        self.conds[1] = max(self.conds[1], side_c1)
                        float_top_by_side[side_comp] = side_c1
            # Compound columns can also lower conds[0] (region-fit) or raise conds[1] (float).
            for col, p in zip(compound_cols, _layout_compounds(self.conds[0], self.conds[1])):
                if p[0] == 'lower':
                    self.conds[0] = max(min(self.conds[0], p[1]), _ABS_ZERO)
                elif p[0] == 'float' and col['float_base'] is not None:
                    m = clear_frac + 2 * p[1] + gap_frac
                    if m < 0.95:
                        self.conds[1] = max(self.conds[1],
                                            (col['float_base'] - m * self.conds[0]) / (1.0 - m))
            legend_params = self._place_legend(
                liq_df, assessed_pts, xlim, n_entries=len(entry_labels),
                max_label_chars=max(len(s) for s in entry_labels),
                float_top_by_side=float_top_by_side)
            if abs(self.conds[0] - c0_before) < 1e-6 and abs(self.conds[1] - c1_before) < 1e-6:
                break

        # Final emission with the converged conds.
        span = self.conds[1] - self.conds[0]
        clear_d, gap_d = clear_frac * span, gap_frac * span
        for side_comp, d in sides_data.items():
            placements = _layout_side(side_comp, self.conds[0], self.conds[1])
            floated = []
            for r, text, hf, p in zip(d['side'], d['texts'], d['half_hs_frac'], placements):
                kind = p[0]
                label = {'text': text, 'xanchor': 'center', 'yanchor': 'middle',
                         'textangle': -90, 'font_size': 12, 'font_color': 'black', 'pin': True}
                # Visible extent of this region under the converged conds (raw t_bottom may sit
                # below conds[0] for a ground-state trace); the home/midpoint use the visible band.
                vis_bottom = max(r['t_bottom'], self.conds[0])
                vis_top = min(r['t_max'], self.conds[1])
                mid = 0.5 * (vis_bottom + vis_top)
                home_y = min(max(mid, self.conds[0]), self.conds[1])
                if kind == 'inband':
                    label['x'], label['y'] = d['label_x'], p[1]
                elif kind == 'twophase':
                    # In the two-phase region just inside the element, with an arrow to the phase.
                    label['x'], label['y'] = d['fx'], p[1]
                    label['home_x'] = 99.5 if side_comp == 100 else 0.5
                    label['home_y'] = home_y
                else:  # 'float' (or a 'lower' that could not reach absolute zero)
                    label['x'] = d['fx']
                    label['home_x'] = 99.5 if side_comp == 100 else 0.5
                    label['home_y'] = home_y
                    label['above_liquidus'] = True
                    base = d['float_base'] if d['float_base'] is not None else vis_top
                    floated.append((label, base, hf, mid))
                label_dicts.append(label)
            # Stack this side's floated labels just above the liquidus (bottom-to-top by temperature).
            cursor = (max(b for _, b, _, _ in floated) + clear_d) if floated else 0.0
            for label, _base, hf, _mid in sorted(floated, key=lambda t: t[3]):
                hh_d = hf * span
                label['y'] = cursor + hh_d
                cursor = label['y'] + hh_d + gap_d

        # --- Emit compound labels with the converged conds (region-fit + bottom pad + escalation). ---
        margin_d, pad_d = 0.012 * span, _BOTTOM_PAD_FRAC * span
        for col, p in zip(compound_cols, _layout_compounds(self.conds[0], self.conds[1])):
            kind = p[0]
            label = {'text': col['text'], 'xanchor': 'center', 'yanchor': 'middle',
                     'textangle': -90, 'font_size': 12, 'font_color': 'black'}
            vis_bottom, vis_top = max(col['t_min'], self.conds[0]), min(col['t_max'], self.conds[1])
            mid = 0.5 * (vis_bottom + vis_top)
            if kind == 'float':
                # Float above the local liquidus with a leader arrow pointing to the compound.
                label['x'] = col['label_x']
                label['home_x'], label['home_y'] = col['comp'], min(max(mid, self.conds[0]), self.conds[1])
                label['above_liquidus'] = True
                base = col['float_base'] if col['float_base'] is not None else vis_top
                label['y'] = base + clear_d + col['half_h_frac'] * span
            else:  # 'inband' / 'lower' -> a bottom-region slot under the final conds
                ceil = (col['ceil_raw'] - margin_d) if col['ceil_raw'] is not None else self.conds[1]
                y = (p[1] if kind == 'inband' else
                     self._place_compound_y(self.conds[0], ceil, col['tie_temps'],
                                            col['half_h_frac'] * span, pad_d, margin_d))
                if y is None:
                    y = self.conds[0] + pad_d + col['half_h_frac'] * span
                label['x'], label['y'] = col['label_x'], y
                # Home at the placed position: a pure vertical de-collision nudge (the bug-3 forced
                # stacking fallback) must NOT spawn a leader arrow back to an arbitrary point.
                label['home_x'], label['home_y'] = col['label_x'], y
                label['no_arrow'] = True
            label_dicts.append(label)

        # --- Resolve label collisions and draw all collected labels ---
        ylim = (self.conds[0], self.conds[1])
        # Keep in-band labels below the lower of the two liquidus curves.
        liquidus_ceiling = self._liquidus_top_fn(liq_df, assessed_pts, combine='min') if not liq_df.empty else None
        resolved_labels = self._resolve_label_collisions(label_dicts, xlim, ylim, max_iterations=80,
                                                         ceiling=liquidus_ceiling,
                                                         tie_segments=tie_segments)
        px_per_x = _PLOT_W_PX / (xlim[1] - xlim[0])
        px_per_y = _PLOT_H_PX / (ylim[1] - ylim[0])
        for lbl in resolved_labels:
            common = dict(text=lbl['text'], textangle=lbl.get('textangle', -90),
                          font=dict(size=lbl.get('font_size', 12), color=lbl.get('font_color', 'black')))
            if lbl.get('showarrow') and not lbl.get('no_arrow'):
                ax_px = (lbl['x'] - lbl['home_x']) * px_per_x
                ay_px = -(lbl['y'] - lbl['home_y']) * px_per_y
                fig.add_annotation(x=lbl['home_x'], y=lbl['home_y'], ax=ax_px, ay=ay_px,
                                   showarrow=True, arrowhead=2, arrowwidth=1, arrowcolor='gray',
                                   xanchor=lbl.get('xanchor', 'center'),
                                   yanchor=lbl.get('yanchor', 'middle'), **common)
            else:
                fig.add_annotation(x=lbl['x'], y=lbl['y'], showarrow=False,
                                   xanchor=lbl.get('xanchor', 'center'),
                                   yanchor=lbl.get('yanchor', 'middle'), **common)

        if pred:
            self.phase_color_remap['L'] = '#117733'
            fig.add_trace(px.line(liq_df, x='x', y='t', color='label',
                                  color_discrete_map=self.phase_color_remap).data[0])
        else:
            fig.add_trace(px.line(liq_df, x='x', y='t', color='label',
                                  color_discrete_map=self.phase_color_remap).data[0])
        fig.update_traces(line=dict(width=4), showlegend=False)
        if digitized_liquidus:
            fig.add_trace(go.Scatter(x=[None], y=[None], mode='lines', line=dict(color='#B82E2E', dash='dash'),
                                     name='Assessed Liquidus', showlegend=True))
        if imputed_phases:
            fig.add_trace(go.Scatter(x=[None], y=[None], mode='lines', line=dict(color='gray', dash='dash'),
                                     name='Imputed Phase', showlegend=True))
        if pred:
            fig.add_trace(go.Scatter(x=[None], y=[None], mode='lines', marker=dict(color='#117733'),
                                     name='Predicted Liquidus', showlegend=True))
        else:
            fig.add_trace(go.Scatter(x=[None], y=[None], mode='lines', marker=dict(color='cornflowerblue'),
                                     name='Fitted Liquidus', showlegend=True))
        # Legend corner/expansion was already determined above (conds[1] expanded as needed).
        fig.update_layout(
            title=dict(text=f'<b>{self.comps[0]}-{self.comps[1]} DFT-Referenced Phase Diagram</b>',
                   x=0.5, xanchor='center', font=dict(size=18, color='black'), yanchor='bottom'),
            xaxis=dict(range=[0, 100], title='Composition (at. %)'),
            yaxis=dict(range=[self.conds[0], self.conds[1]], title='Temperature (°C)', ticksuffix=" "),
            width=750, # 960 for show()
            height=600, # 700 for show()
            plot_bgcolor='white',
            font=dict(size=13, color='black'),
            showlegend=True,
            legend=legend_params,
            margin=dict(t=72, b=72, r=55)
        )
        axes_params_dict = dict(
            title_font=dict(size=16),
            title_standoff=8,  # Space between title and axis line
            mirror=True,        # Draws lines on all four sides
            showline=True,      # Shows the primary axis lines (bottom, left)
            linecolor='black',
            linewidth=1.5,
            ticks="outside",    # Places ticks outside the plot area, starting at the axis line
            tickcolor='black',
            ticklen=5,
            tickwidth=1,
            minor_ticks="outside", # Places minor ticks outside
            minor=dict(tickcolor='black', ticklen=2, tickwidth=1, nticks=5)
        )
        fig.update_xaxes(
            tickformat=".0f",
            **axes_params_dict
        )
        fig.update_yaxes(
            **axes_params_dict
        )
        l_x, l_y = self._place_liquid_label(liq_df, assessed_pts, (0.0, 100.0),
                                            (self.conds[0], self.conds[1]), font_size=14)
        fig.add_annotation(
            x=l_x,
            y=l_y,
            text='L',
            showarrow=False,
            font=dict(
                size=14,
                color='black'
            )
        )
        fig.add_annotation(
            x=-0.05,
            y=-0.086,  # Position below the x-axis in paper coordinates
            xref="paper",
            yref="paper",
            text=self.comps[0], # Use the component name from the data
            showarrow=False,
            font=dict(color="black", size=13.5),
            xanchor='left',
            yanchor='middle'
        )
        fig.add_annotation(
            x=1.05,
            y=-0.086, # Position below the x-axis in paper coordinates
            xref="paper",
            yref="paper",
            text=self.comps[1], # Use the component name from the data
            showarrow=False,
            font=dict(color="black", size=13.5),
            xanchor='right',
            yanchor='middle'
        )

        return fig

    def get_phase_points(self) -> dict:
        """Extracts phase boundary points from the HSX object and converts to a list of dictionaries for BinaryLiquid"""

        df_tx = self.compute_tx()[0]
        phase_points = {phase: df_tx[df_tx['label'] == phase][['x', 't']].values.tolist() for phase in self.phases}
        liq_df = (df_tx[df_tx['label'] == 'L']
                  .sort_values(['x', 't'])
                  .drop_duplicates(subset='x', keep='first'))
        phase_points['L'] = liq_df[liq_df['t'] >= -273.15][['x', 't']].values.tolist()
        return phase_points
