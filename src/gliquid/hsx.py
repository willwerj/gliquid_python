"""
Authors: Abrar Rauf, Joshua Willwerth
Last Modified: June 30, 2025
Description: This script takes the phase energy data in the form of enthalpy (H), entropy (S) and composition (X)
and performs transformations to composition-temperature (TX) phase diagrams with well-defined coexistence boundaries
GitHub: https://github.com/AbrarRauf
ORCID: https://orcid.org/0000-0001-5205-0075
"""
from __future__ import annotations

import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from scipy.spatial import ConvexHull
from collections import defaultdict

class HSX:
    """Handles enthalpy (H), entropy (S), and composition (X) transformations for TX phase diagrams."""

    def __init__(self, data_dict: dict, conds: list[float]):
        """Initializes the HSX instance with provided phase data and conditions."""
        self.phases = data_dict['phases']
        self.comps = data_dict['comps']
        self.conds = conds
        self.df = pd.DataFrame(data_dict['data'])
        self.phase_color_remap = {}
        self.simplices = []
        self.final_phases = []
        self.df_tx = pd.DataFrame()
    
        # Data scaling
        s_scaler = 100
        h_scaler = 10000
        self.df.columns = ['X [Fraction]', 'S [J/mol/K]', 'H [J/mol]', 'Phase']
        self.df['S [J/mol/K]'] /= s_scaler
        self.df['H [J/mol]'] /= h_scaler

        # Color Mapping
        color_array = px.colors.qualitative.Pastel
        inter_phases = [p for p in self.phases if p != 'L']
        self.color_map = {phase: color for phase, color in zip(inter_phases, color_array)}
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
        # new_hull = ConvexHull(new_points)
        
        def check_common_rows(arr1: np.ndarray, arr2: np.ndarray) -> bool:
            """Checks if any row in arr1 exists in arr2."""
            return any((arr1 == row).all(axis=1).any() for row in arr2)
        
        # Filter hull simplices
        lower_hull_filter1 = [s for s in new_hull.simplices if not check_common_rows(new_points[s], fict_points)]
        lower_hull_filter2 = [s for s in lower_hull_filter1 
                              if sum((v == im).all() for v in self.points[s] for im in self.inter_points) < 3]
        self.simplices = np.array(lower_hull_filter2)
        return self.simplices

    def compute_tx(self) -> tuple[pd.DataFrame, list, np.ndarray, np.ndarray]:
        """Computes the TX phase diagram transformation."""
        self.hull()
        temps, valid_simplices, new_phases = [], [], []
        for simplex in self.simplices:
            A, B, C = self.points[simplex]
            n = np.cross(B - A, C - A).astype(float)
            T = (-n[1] / n[2]) * self.scaler
            if not np.isnan(T):
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
    
    def plot_tx_scatter(self):
        # self.df_tx = self.compute_tx()[0]   
        # convert all temps to Celsius
        self.df_tx['t'] = self.df_tx['t'] - 273.15

        # Option 2: Save the DataFrame to an Excel file
        self.df_tx.to_excel("tx_data.xlsx", index=False) # saves to an excel file in the same directory
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
        fig.show()

    def plot_hsx(self):
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

        inv_points = {'Eutectics': [], 'Peritectics': [], 'Congruent Melting': [], 'Misc Gaps': []}
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
        count_dict = {key: len(value) for key, value in inv_points.items()}

        return inv_points, combined_list, count_dict
    
    def plot_tx(self, pred: bool = False, digitized_liquidus: list = None, gas_temp: int | float = None) -> go.Figure:
        """Plots the binary phase diagram from computed phase boundaries and invariant points."""
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
        solid_df = new_df_tx[new_df_tx['label'] != 'L']

        lhs_tm, rhs_tm = liq_df.iloc[0]['t'], liq_df.iloc[-1]['t']
        max_liq, min_liq = liq_df['t'].max(), liq_df['t'].min()

        fig = go.Figure()

        if digitized_liquidus: # update liquidus temperature range based on digitized liquidus
            max_liq = max(max_liq, max([p[1] - 273.15 for p in digitized_liquidus]))
            min_liq = min(min_liq, min([p[1] - 273.15 for p in digitized_liquidus]))
            fig.add_trace(
                go.Scatter(x=[p[0] * 100 for p in digitized_liquidus], y=[p[1] - 273.15 for p in digitized_liquidus],
                            mode='lines', line=dict(color='#B82E2E', dash='dash')))
        else: # expand temperature range to better accomodate the legend if there is no reference phase diagram
            # highest temp at least 30% of range above lower tm
            if max(lhs_tm, rhs_tm) + 0.3 * (self.conds[1] - self.conds[0]) < self.conds[1]:  
                self.conds[1] += 0.1 * (self.conds[1] - self.conds[0])
            else:
                self.conds[1] = max(lhs_tm, rhs_tm) + 0.36 * (self.conds[1] - self.conds[0])

        if max_liq > self.conds[1]:
            self.conds[1] = max_liq + 0.1 * (self.conds[1] - self.conds[0])
        if min_liq < self.conds[0]:
            self.conds[0] = max(min_liq - 0.1 * (self.conds[1] - self.conds[0]), -273.15)
        
        solid_phases = [p for p in self.phases if p not in [self.comps[0], self.comps[1], 'L']]
        for phase in solid_phases:
            phase_df = solid_df[solid_df['label'] == phase]
            if phase_df.empty:
                continue

            phase_decomp_temp = phase_df['t'].max()
            if phase_decomp_temp - 0.1 * (self.conds[1] - self.conds[0]) < self.conds[0]:
                if phase_decomp_temp - 0.1 * (self.conds[1] - self.conds[0]) < -273.15:
                    continue
                self.conds[0] = phase_decomp_temp - 0.1 * (self.conds[1] - self.conds[0])

        solid_comp_list = []
        idx_tracker = 0
        for phase in solid_phases:
            phase_df = solid_df[solid_df['label'] == phase]
            if phase_df.empty:
                continue

            solid_comp = phase_df['x'].values
            solid_comp = solid_comp[0] 
            solid_comp_list.append(solid_comp)

            new_row_df = pd.DataFrame(
                [{'x': solid_comp, 't': -273.15, 'label': phase, 'color': self.color_map.get(phase)}],
                  columns=phase_df.columns)
            phase_df = pd.concat([phase_df, new_row_df], ignore_index=True)
            line = px.line(phase_df, x='x', y='t', color='label', color_discrete_map=self.phase_color_remap)

            fig.add_trace(line.data[0])
            x_offset = 1.5 if solid_comp < 5 or (idx_tracker > 0 and solid_comp_list[idx_tracker] - solid_comp_list[idx_tracker - 1] < 5) else -2.5
            fig.add_annotation(
                x=solid_comp + x_offset,
                y=self.conds[0],
                yanchor='bottom',
                text=phase,
                showarrow=False,
                textangle=-90,
                borderpad=5,
                font=dict(size=12, color='black')
            )
            idx_tracker += 1

        for key in inv_points.keys():
            if key in ['Eutectics', 'Peritectics', 'Misc Gaps']:
                for temp, _, comps, _ in inv_points[key]:
                    comps = [x * 100 for x in comps]
                    temps = [temp] * 3
                    line = px.line(x=comps, y=temps)
                    line.update_traces(line=dict(color='Silver'))
                    fig.add_trace(line.data[0])
        
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
        if pred:
            fig.add_trace(go.Scatter(x=[None], y=[None], mode='lines', marker=dict(color='#117733'),
                                     name='Predicted Liquidus', showlegend=True))
        else:
            fig.add_trace(go.Scatter(x=[None], y=[None], mode='lines', marker=dict(color='cornflowerblue'),
                                     name='Fitted Liquidus', showlegend=True))
        # if gas_temp and gas_temp - 273.15 < min(liq_df['t'].max(), self.conds[1]) and not digitized_liquidus:
        #     fig.add_trace(go.Scatter(x=[0, 100], y=[gas_temp - 273.15, gas_temp - 273.15],
        #                              mode='lines', line=dict(color='#FFAE43', dash='dash'),
        #                              name='Gas Phase Forms', showlegend=True))
        legend_params = {'yanchor': 'top', 'y': 0.99, 'xanchor': 'right', 'x': 0.99, 'font': dict(size=15)}
        legend_params.update({'xanchor': 'left', 'x': 0.01} if lhs_tm < rhs_tm else {'xanchor': 'right', 'x': 0.99})

        fig.update_layout(
            title=dict(text=f'<b>{self.comps[0]}-{self.comps[1]} DFT-Referenced Phase Diagram</b>',
                   x=0.5, xanchor='center', font=dict(size=18, color='black'), yanchor='bottom'),
            xaxis=dict(range=[0, 100], title='Composition (at. %)'),
            yaxis=dict(range=[max(self.conds[0], -273), self.conds[1]], title='Temperature (°C)', ticksuffix=" "),
            width=750, # 960 for show()
            height=600, # 700 for show()
            plot_bgcolor='white',
            font=dict(size=13, color='black'),
            showlegend=True,
            legend=legend_params,
            margin=dict(t=72, b=72, r=55)  # Also reduce top margin for tighter spacing
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
        fig.add_annotation(
            x=50,
            y=self.conds[1] - 0.08 * (self.conds[1] - self.conds[0]),
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

        def remove_duplicates(coordinates):
            """Removes duplicate x-coordinates while preserving order."""
            coordinates.sort(key=lambda x: (x[0], x[1]))
            filtered_coordinates = []
            current_x = None
            for coord in coordinates:
                if coord[0] != current_x:
                    if coord[1] >= -273.15:
                        filtered_coordinates.append(coord)
                    current_x = coord[0]
            
            return filtered_coordinates
        

        df_tx = self.compute_tx()[0]
        phase_points = {phase: df_tx[df_tx['label'] == phase][['x', 't']].values.tolist() for phase in self.phases}
        phase_points['L'] = remove_duplicates(phase_points['L'])
        return phase_points
