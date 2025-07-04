'''
Author: Abrar Rauf
Edited by Josh 06/26/25
This module contains the classes for ternary interpolation and ternary phase diagram plotting.
'''
import os
import time
import sys
import json
import numpy as np
import pandas as pd
import sympy as sp
import plotly.express as px
import plotly.graph_objects as go

from typing import List
from emmet.core.utils import jsanitize
from mp_api.client import MPRester
from pymatgen.analysis.phase_diagram import PhaseDiagram
from pymatgen.core.composition import Element, Composition
from pymatgen.entries.computed_entries import ComputedStructureEntry
from scipy.spatial import Delaunay
# from auth import key as MAPI_KEY

from gliquid.config import fusion_enthalpies_file, fusion_temps_file
from gliquid.binary import (
    BinaryLiquid,
    linear_expr, exponential_expr, combined_expr)

sys.path.append(os.path.dirname(os.path.abspath(__file__))) # If importing this file into a script from a different dir
from extensive_hull_main import gliq_lowerhull3, gen_hyperplane_eqns2

# mpr = MPRester(MAPI_KEY)
mpr = MPRester("YOUR API KEY HERE")  # Use environment variable for MP_API_KEY

# Define all required symbols
R = 8.314  # J/(mol*K), universal gas constant
x1_sym, x2_sym, t_sym, w12_sym, w23_sym, w31_sym, a_sym, b_sym = sp.symbols('x1 x2 t w12 w23 w31 a12 b12')

_L_LINEAR_EXPR = linear_expr(a_sym, b_sym)
_L_EXP_EXPR = exponential_expr(a_sym, b_sym)
_L_LIN_EXP_EXPR = combined_expr(a_sym, b_sym, sp.Integer(10000))

def build_ternary_thermodynamic_expressions(
    x1=x1_sym, x2=x2_sym, t=t_sym,
    w12=w12_sym, w23=w23_sym, w31=w31_sym,
    g_ref_a_expr: sp.Expr = sp.Symbol('G_ref_A_placeholder'),
    g_ref_b_expr: sp.Expr = sp.Symbol('G_ref_B_placeholder'),
    g_ref_c_expr: sp.Expr = sp.Symbol('G_ref_C_placeholder'),
    l0_ab_expr: sp.Expr = _L_LINEAR_EXPR,
    l1_ab_expr: sp.Expr = _L_LINEAR_EXPR,
    l0_bc_expr: sp.Expr = _L_LINEAR_EXPR,
    l1_bc_expr: sp.Expr = _L_LINEAR_EXPR,
    l0_ca_expr: sp.Expr = _L_LINEAR_EXPR,
    l1_ca_expr: sp.Expr = _L_LINEAR_EXPR,
    l0_abc_expr: sp.Expr = 0
) -> dict[str, sp.Expr]:
    """
    Builds a dictionary of thermodynamic Sympy expressions for a ternary system.

    Args:
        x1, x2, t: Symbols for mole fractions of B, C and temperature.
        g_ref_a_expr: Sympy expression for the reference Gibbs energy of component A.
        g_ref_b_expr: Sympy expression for the reference Gibbs energy of component B.
        g_ref_c_expr: Sympy expression for the reference Gibbs energy of component C.
        w12, w23, w31: Weighting factors for each binary.
        l0_ab_expr, l1_ab_expr: Sympy expressions for AB binary Redlich-Kister parameters.
        l0_bc_expr, l1_bc_expr: Sympy expressions for BC binary Redlich-Kister parameters.
        l0_ca_expr, l1_ca_expr: Sympy expressions for CA binary Redlich-Kister parameters.
        l0_abc_expr: Optional ternary interaction parameter (default 0).

    Returns:
        dict[str, sp.Expr]: A dictionary mapping equation names to their Sympy expressions.
    """
    x_a = 1 - x1 - x2
    x_b = x1
    x_c = x2

    # Reference Gibbs energy
    g_ref = g_ref_a_expr * x_a + g_ref_b_expr * x_b + g_ref_c_expr * x_c

    # Ideal mixing Gibbs energy with piecewise logic to handle zero values
    g_ideal = sp.Piecewise(
        (0, sp.Or(sp.Eq(x_a, 1), sp.Eq(x_b, 1), sp.Eq(x_c, 1))),  # Only 1 component present
        (R * t * (x_a * sp.log(x_a) + x_b * sp.log(x_b)), sp.Eq(x_c, 0)),  # A and B present
        (R * t * (x_a * sp.log(x_a) + x_c * sp.log(x_c)), sp.Eq(x_b, 0)),  # A and C present
        (R * t * (x_b * sp.log(x_b) + x_c * sp.log(x_c)), sp.Eq(x_a, 0)),  # B and C present
        (R * t * (x_a * sp.log(x_a) + x_b * sp.log(x_b) + x_c * sp.log(x_c)), True),  # All present
    )

    # Excess Gibbs energy (Redlich-Kister for each binary, plus optional ternary term)
    g_xs_ab = x_a * x_b * w12 * (l0_ab_expr + l1_ab_expr * (x_a - x_b))
    g_xs_bc = x_b * x_c * w23 * (l0_bc_expr + l1_bc_expr * (x_b - x_c))
    g_xs_ca = x_c * x_a * w31 * (l0_ca_expr + l1_ca_expr * (x_c - x_a))
    g_xs_tern = l0_abc_expr * x_a * x_b * x_c

    g_xs = g_xs_ab + g_xs_bc + g_xs_ca + g_xs_tern
    
    # Total Gibbs energy of liquid phase with piecewise logic for ideal mixing
    g_liquid = g_ref + g_ideal + g_xs

    # Entropy of liquid phase: S = - (dG/dT)_P,x
    s_liquid = -sp.diff(g_liquid, t)

    # Enthalpy of liquid phase: H = G + TS = G - T*(dG/dT)_P,x
    h_liquid = g_liquid + t * s_liquid

    return {
        'g_ref': g_ref,
        'g_ideal': g_ideal,
        'g_xs': g_xs,
        'g_liquid': g_liquid,
        's_liquid': s_liquid,
        'h_liquid': h_liquid
    }


def ordered_binary_systems(elements):
    # given a ternary system, returns the ordered binary systems
    binary_pairs = []
    for i in range(len(elements)):
        next_element = elements[(i + 1) % len(elements)] 
        binary_pairs.append(f"{elements[i]}-{next_element}")

    return binary_pairs

def invert_substrings(input_string):
    substring1, substring2 = input_string.split('-')
    inverted_string = f"{substring2}-{substring1}"
    return inverted_string

def cartesian_to_ternary(df):
    xs = df.iloc[:, 0].values
    ys = df.iloc[:, 1].values
    new_xs = []
    new_ys = []
    for x, y in zip(xs, ys):
        unitvec = np.array([[1, 0], [0.5, np.sqrt(3) / 2]])
        trans_coord = np.dot(np.array([x, y]), unitvec)
        new_xs.append(trans_coord[0])
        new_ys.append(trans_coord[1])

    df.iloc[:, 0] = new_xs
    df.iloc[:, 1] = new_ys

    return df

def ternary_to_cartesian(x_A, x_B):
    x = x_A + 0.5 * x_B
    y = np.sqrt(3) / 2 * x_B
    return x, y

def point_to_surface_height(new_point, liquid_points, triangulation, triangles):
    new_point_cartesian = ternary_to_cartesian(new_point[0], new_point[1])

    simplex = triangulation.find_simplex(new_point_cartesian[:2])
    
    if simplex == -1:
        raise ValueError("The new point is outside the triangulated surface.")

    vertices = triangles[simplex]
    
    v0 = liquid_points[vertices[0]]
    v1 = liquid_points[vertices[1]]
    v2 = liquid_points[vertices[2]]
    
    def find_z_on_triangle(x, y, vertex1, vertex2, vertex3):    
        x1, y1, z1 = vertex1
        x2, y2, z2 = vertex2
        x3, y3, z3 = vertex3

        v1 = np.array([x2 - x1, y2 - y1, z2 - z1])
        v2 = np.array([x3 - x1, y3 - y1, z3 - z1])

        normal = np.cross(v1, v2)
        A, B, C = normal
        D = -A * x1 - B * y1 - C * z1

        if np.isclose(C, 0):
            raise ValueError("The triangle is degenerate or vertical in the xy-plane.")
        
        z = (-D - A * x - B * y) / C
        
        return z
    
    interpolated_z = find_z_on_triangle(new_point_cartesian[0], new_point_cartesian[1], v0, v1, v2)

    int_point = new_point.copy()
    int_point[2] = interpolated_z

    vertical_height = new_point[2] - interpolated_z

    
    return vertical_height, int_point


def generate_comp_grid(delta=0.025, atol=1e-6):
    # generate composition grid for ternary system
    incr = np.arange(0, 1 + delta, delta)
    A, B, C = np.meshgrid(incr, incr, incr)
    x_A = A.flatten()
    x_B = B.flatten()
    x_C = C.flatten()
    valid_indices = np.where(np.isclose(x_A + x_B + x_C, 1, atol=atol))
    x_A = x_A[valid_indices]
    x_B = x_B[valid_indices]
    x_C = x_C[valid_indices]
    decimal_places = max(2, -int(np.log10(atol)))
    x_A = np.round(x_A, decimal_places)
    x_B = np.round(x_B, decimal_places)
    x_C = np.round(x_C, decimal_places)
    return {'A': x_A, 'B': x_B, 'C': x_C}



class ternary_interpolation:
    def __init__(self, tern_sys: List[str], dir: str, **kwargs):
        self.tern_sys = sorted(tern_sys)
        self.binary_sys = ordered_binary_systems(self.tern_sys)
        self.dir = dir # moving forward, I will store all relevant paths in the gliquid/config.py file. Also shouldn't use a builtin name for a variable
        
        self.delta = kwargs.get('delta', 0.025)  # default to 0.025
        self.tern_comp = generate_comp_grid(self.delta)
        self.interp_type = kwargs.get('interp_type', 'linear')  # default to linear interpolation
        self.param_format = kwargs.get('param_format', 'linear')
        self.L_dict = kwargs.get('L_dict', {}) # adding functionality to pass in a dict of L parameters on construction
        self.L_tern = kwargs.get('L_tern', [0, 0])  # ternary interaction parameters (H, S)
    
    def init_ref_data(self):
        # initialize reference data for fusion enthalpies and entropies
        fusion_enthalpy = pd.read_json(os.path.join(fusion_enthalpies_file), typ='series')
        fusion_temp = pd.read_json(os.path.join(fusion_temps_file), typ='series')
        tern_enthalpy = fusion_enthalpy[self.tern_sys].values
        tern_temp = fusion_temp[self.tern_sys]
        tern_entropy = tern_enthalpy/tern_temp
        self.ref_data = {'H': tern_enthalpy, 'S': tern_entropy, 'T': tern_temp}

    def retrieve_system_parameters(self, fitted, df_fitted, df_predicted):
        # retrieve the L parameters for the binary systems
        def get_parameters(df, system, invert=False):
            if invert:
                system = invert_substrings(system)
            try:
                params = df.loc[df['system'] == system, ['L0_a', 'L0_b', 'L1_a', 'L1_b']].values[0]
                if invert:
                    params[2:] *= -1 
                return params
            except IndexError:
                print(f"Predicted parameters for {system} not found. Terminating!")
                # return NaN params
                return [np.nan, np.nan, np.nan, np.nan]

        for sys in self.binary_sys:
            target_df = None
            param_key = 'errors' if fitted == 1 else 'continuous'
            if sys in df_fitted['system'].values or invert_substrings(sys) in df_fitted['system'].values:
                invert = sys not in df_fitted['system'].values
                system = invert_substrings(sys) if invert else sys
                result = df_fitted.loc[df_fitted['system'] == system, param_key].values[0]
                target_df = df_fitted if (result == 'none' if fitted == 1 else result) else df_predicted

                print(f"Using {'fitted' if target_df is df_fitted else 'predicted'} params for {sys}")
                self.L_dict[sys] = get_parameters(target_df, sys, invert=invert)
            else:
                print(f"System {sys} not found")

        # Interaction terms
        # print("Binary interaction terms:")
        # for key, val in self.L_dict.items():
        #     print(f"{key}: {val}")

        return target_df

    def ternary_interpolation(self): # maybe there's a better name for this than the same as the class name?
        # interpolate the ternary system using the binary interaction parameters
        x_A, x_B, x_C = self.tern_comp['A'], self.tern_comp['B'], self.tern_comp['C']

        self.init_ref_data()
        H_A, H_B, H_C = self.ref_data['H']
        S_A, S_B, S_C = self.ref_data['S']

        if not all(sys in self.L_dict.keys() for sys in self.binary_sys): # only do this if L_dict is not already populated
            # df_fitted = pd.read_excel(os.path.join(self.dir, 'fitted_system_data_new.xlsx'))
            df_fitted = pd.read_excel(os.path.join(self.dir, 'composite_fit_results-trimmed+carbides.xlsx'))
            # df_fitted = pd.read_excel(os.path.join(self.dir, 'composite_fit_results.xlsx'))
            df_predicted = pd.read_excel(os.path.join(self.dir, 'predicted_params_final.xlsx'))
            self.bin_df = self.retrieve_system_parameters(1, df_fitted, df_predicted) # '1' used as an enum?

        if self.interp_type == 'linear':
            wAB, wBC, wCA = 1, 1, 1
        elif self.interp_type == 'muggianu':
            wAB = 4*x_A*x_B/(1-(x_A - x_B)**2)
            wBC = 4*x_B*x_C/(1-(x_B - x_C)**2)
            wCA = 4*x_C*x_A/(1-(x_C - x_A)**2) 
        elif self.interp_type == 'kohler':
            wAB = (x_A + x_B)**2
            wBC = (x_B + x_C)**2
            wCA = (x_C + x_A)**2
        elif self.interp_type == 'luck_chou':
            wAB = x_B/(1-x_A)
            wBC = x_C/(1-x_B) 
            wCA = x_A/(1-x_C)

        if self.param_format == 'linear':
            l_expr = _L_LINEAR_EXPR
        elif self.param_format == 'exponential':
            l_expr = _L_EXP_EXPR
        elif self.param_format in ['combined', 'whs']:
            l_expr = _L_LIN_EXP_EXPR


        L_array = np.array([self.L_dict[sys] for sys in self.binary_sys]) # 3 x 4 array in order of binary systems
        symbolic_expressions = build_ternary_thermodynamic_expressions(
            w12=wAB, w23=wBC, w31=wCA,
            g_ref_a_expr=(H_A - t_sym * S_A),
            g_ref_b_expr=(H_B - t_sym * S_B),
            g_ref_c_expr=(H_C - t_sym * S_C),
            l0_ab_expr=l_expr.subs({a_sym: L_array[0][0], b_sym: L_array[0][1]}),
            l1_ab_expr=l_expr.subs({a_sym: L_array[0][2], b_sym: L_array[0][3]}),
            l0_bc_expr=l_expr.subs({a_sym: L_array[1][0], b_sym: L_array[1][1]}),
            l1_bc_expr=l_expr.subs({a_sym: L_array[1][2], b_sym: L_array[1][3]}),
            l0_ca_expr=l_expr.subs({a_sym: L_array[2][0], b_sym: L_array[2][1]}),
            l1_ca_expr=l_expr.subs({a_sym: L_array[2][2], b_sym: L_array[2][3]}),
            l0_abc_expr=self.L_tern[0] if self.L_tern[0] != 0 else 0
        )

        tm_mean = np.mean(self.ref_data['T']) # mean melting point in ternary - used for t-dependent H and S forms
        lambda_args_symbols = [x1_sym, x2_sym, t_sym]
        lambda_args_values = [x_B, x_C, tm_mean]

        h_lambdified = sp.lambdify(lambda_args_symbols, symbolic_expressions['h_liquid'], 'numpy')
        s_lambdified = sp.lambdify(lambda_args_symbols, symbolic_expressions['s_liquid'], 'numpy')
        
        # Suppress divide by zero warnings
        with np.errstate(divide='ignore', invalid='ignore'):
            h_vals_mesh = h_lambdified(*lambda_args_values)
            s_vals_mesh = s_lambdified(*lambda_args_values)
        
        # Replace inf and nan values with finite values if needed
        H = np.where(np.isfinite(h_vals_mesh), h_vals_mesh, 0).flatten()
        S = np.where(np.isfinite(s_vals_mesh), s_vals_mesh, 0).flatten()

        self.hsx_df = pd.DataFrame({'x0': x_B, 'x1': x_C, 'S': S, 'H': H})
        self.hsx_df['Phase Name'] = 'L'
  
    def get_phasedia_entries(self, sys):
        # add a subdir to self.dir
        dft_subdir = os.path.join(self.dir, 'ternary_dft_data')
        if not os.path.exists(dft_subdir):
            os.makedirs(dft_subdir)

        json_path = os.path.join(dft_subdir, f"{sys}_entries.json")
        if os.path.exists(json_path):
            print("Loading ternary DFT energies from cache")
            with open(json_path, 'r') as f:
                sanitized_entries = json.load(f)
        else:
            print("Reading ternary DFT energies from MP")
            entries = mpr.get_entries_in_chemsys(sys)
            sanitized_entries = jsanitize(entries)
            with open(json_path, 'w') as f:
                json.dump(sanitized_entries, f)
        
        return sanitized_entries

    def get_ternary_form_en(self, sys):
        # get the formation energies of the stable phases in the ternary system
        tern_mp_dict = {}
        sys_eles = []
        for val in sys:
            el = Element(val)
            sys_eles.append(el)
        entries_init = self.get_phasedia_entries(sys)
        entries = [ComputedStructureEntry.from_dict(e) for e in entries_init]
        
        pdia = PhaseDiagram(entries)
        entries = pdia.stable_entries
        all_atm_fracs = []
        all_form_ens = []
        phases = []
        for entry in entries:
            comp_str = entry.composition.reduced_formula
            comp = Composition(comp_str)
            entry_eles = comp.elements
            form_en = pdia.get_form_energy_per_atom(entry)
            all_form_ens.append(form_en*96485)
            atm_fracs = []
            for ele in sys_eles:
                if ele in entry_eles:    
                    atm_fracs.append(comp.get_atomic_fraction(ele))
                else:
                    atm_fracs.append(0.0)
            atm_fracs = atm_fracs[1:]
            all_atm_fracs.append(atm_fracs)
            phases.append(comp_str)

        all_atm_fracs_arr = np.array(all_atm_fracs)

        for i, arr in enumerate(all_atm_fracs_arr.T):
            tern_mp_dict[f'x{i}'] = arr

        tern_mp_dict['H'] = all_form_ens
        tern_mp_dict['Phase Name'] = phases

        entropy = [0]*len(all_form_ens)
        tern_mp_dict['S'] = entropy

        tern_mp_df = pd.DataFrame(tern_mp_dict)
        tern_mp_df = tern_mp_df[['x0', 'x1', 'S', 'H', 'Phase Name']]
        tern_mp_df = tern_mp_df.loc[tern_mp_df.groupby('Phase Name')['H'].idxmin()]

        return tern_mp_df

    def add_binary_data(self):
        # add binary data to the ternary data and plot the binaries (optional)
        bin_fig_list = []
        def process_system(sys_name, i, invert=False):
            params = self.L_dict[sys_name]
            # if invert:
            #     sys_name = invert_substrings(sys_name)

            # rows = self.bin_df.loc[self.bin_df['system'] == sys_name]
            # row = rows.iloc[1] if len(rows) == 2 else rows.iloc[0]
            # params = [row['L0_a'], row['L0_b'], row['L1_a'], row['L1_b']]
            print(params, sys_name, invert)
            sys = BinaryLiquid.from_cache(sys_name, params=params, param_format=self.param_format)
            data = sys.update_phase_points()
            figr = sys.hsx.plot_tx()
            bin_fig_list.append(figr)
            
            data_df = pd.DataFrame(data, columns=['X', 'S', 'H', 'Phase Name'])
            x_col = data_df['X']
            if invert:
                x_col = 1 - x_col

            if i == 0:
                x0 = x_col
                x1 = np.zeros_like(x0)
            elif i == 1:
                x1 = x_col
                x0 = 1 - x1
            else:
                x1 = 1 - x_col
                x0 = np.zeros_like(x1)
            i += 1

            data_df['x0'], data_df['x1'] = x0, x1

            return data_df.drop(columns=['X']), i


        i = 0 
        for sys_name in self.L_dict.keys():
            # if sys_name in self.bin_df['system'].values:
            data_df, i = process_system(sys_name, i, invert=sorted(sys_name.split('-')) != sys_name.split('-'))
            # elif invert_substrings(sys_name) in self.bin_df['system'].values:
            #     data_df, i = process_system(sys_name, i, invert=True)
            # else:
            #     print(f"System {sys_name} not found")
            #     continue

            self.hsx_df = pd.concat([self.hsx_df, data_df], ignore_index=True)

        return bin_fig_list
        

    def interpolate(self):
        # create the hsx dataframe for the ternary system
        self.ternary_interpolation() # populates self.hsx_df with ternary liquid phase data
        # self.bin_fig_list = self.add_binary_data()
        self.tern_mp_df = self.get_ternary_form_en(self.tern_sys)
        self.hsx_df = pd.concat([self.hsx_df, self.tern_mp_df], ignore_index=True)
        self.hsx_df = self.hsx_df.drop_duplicates()
        self.hsx_df = self.hsx_df.reset_index(drop=True)

   
class ternary_hsx_plotter(ternary_interpolation): 
    def __init__(self, tern_sys: List[str], dir: str, **kwargs):
        delta = kwargs.get('delta', 0.025)
        interp_type = kwargs.get('interp_type', 'linear')
        param_format = kwargs.get('param_format', 'linear')
        L_tern = kwargs.get('L_tern', [0, 0])  # ternary interaction parameters (H, S)
        L_dict = kwargs.get('L_dict', {})  # binary interaction parameters
        super().__init__(tern_sys, dir, interp_type=interp_type, param_format=param_format, delta=delta, L_tern=L_tern, L_dict=L_dict)
        self.temp_slider = kwargs.get('temp_slider', [0, 0])

    def init_sys(self):
        self.tern_sys_name = '-'.join(sorted(self.tern_sys))
        self.phases = self.hsx_df['Phase Name'].unique().tolist()

        solid_phases = self.phases.copy()
        solid_phases.remove('L')
        color_array = px.colors.qualitative.Dark24
        self.color_map = dict(zip(solid_phases, color_array))
        self.color_map['L'] = 'cornflowerblue'

        fusion_temp = pd.read_json(os.path.join(fusion_temps_file), typ='series')
        tern_temp = fusion_temp[self.tern_sys].values 
        max_temp = np.max(tern_temp) + 500
        min_temp = np.min(tern_temp)
        self.conds = [np.min(np.array([0, min_temp - 200])), max_temp + self.temp_slider[1]]
        self.hsx_df['x0'] = self.hsx_df['x0'].round(4)
        self.hsx_df['x1'] = self.hsx_df['x1'].round(4)
        self.hsx_df = self.hsx_df.rename(columns={'Phase Name': 'Phase'})
        self.hsx_df['Colors'] = self.hsx_df['Phase'].map(self.color_map)
        print('Initialization complete')

    def lower_convexhull(self):
        start_time = time.time()
        self.points = np.array(self.hsx_df[['x0', 'x1', 'S', 'H']])
        self.simplices = gliq_lowerhull3(self.points, vertical_simplices=True)
        self.temps = gen_hyperplane_eqns2(points = self.points, lower_hull = self.simplices, partial_indices = [2])[1]
        nan_indices = []
        for i in range(len(self.temps)):
            if str(self.temps[i]) == 'nan' or str(self.temps[i]) == 'inf' or str(self.temps[i]) == '-inf':
                nan_indices.append(i)

        self.temps = np.delete(self.temps, nan_indices)
        self.simplices = np.delete(self.simplices, nan_indices, axis=0)
        end_time = time.time()
        print(f"Convex hull and partial derivative evaluation time:: {end_time - start_time} seconds")

    def process_data(self):
        self.init_sys()
        self.lower_convexhull()
        phase_equil = []
        for simplex in self.simplices:
            phase1 = self.hsx_df.loc[simplex[0], 'Phase']
            phase2 = self.hsx_df.loc[simplex[1], 'Phase']
            phase3 = self.hsx_df.loc[simplex[2], 'Phase']
            phase4 = self.hsx_df.loc[simplex[3], 'Phase']
            phase_equil.append(np.array([phase1, phase2, phase3, phase4]))

        data = []
        for i, simplex in enumerate(self.simplices):
            phase_labels = phase_equil[i]
            if len(set(phase_labels)) == 0:
                continue
            else:
                x0_coords = [self.points[vertex][0] for vertex in simplex] 
                x1_coords = [self.points[vertex][1] for vertex in simplex]
                t_val = self.temps[i]

            j = 0
            for x0, x1 in zip(x0_coords, x1_coords):
                label = phase_labels[j]
                color = self.color_map[label]
                data.append([x0, x1, t_val, label, color])
                j += 1

        self.equil_df = pd.DataFrame(data, columns=['x0', 'x1', 't', 'label', 'color'])
        self.equil_df = self.equil_df.dropna(subset=['t'])
        self.equil_df['t'] = self.equil_df['t'] - 273.15
        self.equil_df = cartesian_to_ternary(self.equil_df)
        self.equil_liq_df = self.equil_df[self.equil_df['label'] == 'L']
        self.equil_solid_df = self.equil_df[self.equil_df['label'] != 'L']
        self.equil_solid_df = self.equil_solid_df.sort_values('t').drop_duplicates(subset=['x0', 'x1'], keep='last')

        for index, row in self.equil_solid_df.iterrows():
            x0 = row['x0']
            x1 = row['x1']
            label = row['label']
            color = row['color']
            new_row = {'x0': x0, 'x1': x1, 't': self.conds[0], 'label': label, 'color': color}
            new_row_df = pd.DataFrame([new_row])
            self.equil_solid_df = pd.concat([self.equil_solid_df, new_row_df])

        self.equil_liq_df = self.equil_liq_df.sort_values('t').drop_duplicates(subset=['x0', 'x1'], keep='first')
        self.equil_df = pd.concat([self.equil_solid_df, self.equil_liq_df])        
        
    def plot_ternary(self):
        fig = go.Figure()

        scatter = go.Scatter3d(
            x = self.equil_liq_df['x0'], y = self.equil_liq_df['x1'], z = self.equil_liq_df['t'],
            mode = 'markers', marker = dict(size = 5, color = self.equil_liq_df['color']),
            showlegend=False, opacity = 1,
        )

        solid_points = np.array(list(zip(self.equil_solid_df['x0'], self.equil_solid_df['x1'], self.equil_solid_df['t'])))
        liq_points = np.array(list(zip(self.equil_liq_df['x0'], self.equil_liq_df['x1'], self.equil_liq_df['t'])))
        cart_liq_points = [ternary_to_cartesian(point[0], point[1]) for point in liq_points]
        triangulation = Delaunay(cart_liq_points)
        triangles = triangulation.simplices

        # identify outlier points
        # try:
        #     is_outlier = find_outliers_by_local_stats(liq_points, triangulation)
        #     liq_points = liq_points[~is_outlier]
        # except Exception as e:
        #     print('Outlier detection error:', e)

        try:
            for point in solid_points:
                height = point_to_surface_height(point, liq_points, triangulation, triangles)[0]
                if height > 10:
                    new_row = {'x0': point[0], 'x1': point[1], 't': point[2] + 3, 'label': 'L', 'color': 'cornflowerblue'}
                    new_row_df = pd.DataFrame([new_row])
                    self.equil_liq_df = pd.concat([self.equil_liq_df, new_row_df])
        except Exception as e:
            print('Solid meshing error:', e)

        liq_points = np.array(list(zip(self.equil_liq_df['x0'], self.equil_liq_df['x1'], self.equil_liq_df['t'])))     
        cart_liq_points = [ternary_to_cartesian(point[0], point[1]) for point in liq_points]
        triangulation = Delaunay(cart_liq_points)
        triangles = triangulation.simplices

        fig.add_trace(go.Mesh3d(
            x = self.equil_liq_df['x0'], y = self.equil_liq_df['x1'], z = self.equil_liq_df['t'],
            i = triangles[:, 0], j = triangles[:, 1], k = triangles[:, 2],
            opacity = 0.7, colorscale = 'Viridis', intensity = self.equil_liq_df['t'],
            showscale = False,
        ))
        

        for label, group in self.equil_solid_df.groupby('label'):
            fig.add_trace(go.Scatter3d(
                x = group['x0'], y = group['x1'], z = group['t'],
                mode = 'lines', line = dict(color = group['color'], width = 10),
                showlegend = False, opacity = 1,
            ))

        fig.add_trace(go.Scatter3d(
            x=[0, 0.5, 1, 0],
            y=[0, np.sqrt(3)/2, 0, 0],
            z=[self.conds[0], self.conds[0], self.conds[0], self.conds[0]],
            mode='lines',
            line=dict(color='black', width=5),
            name = 'axes',
            showlegend=False
        ))

        fig.add_trace(go.Scatter3d(
            x=[-0.02, 0.48, 0.98, -0.02],
            y=[0.02, np.sqrt(3)/2 + 0.02, .02, .02],
            z=[self.conds[0] - 50, self.conds[0] - 50, self.conds[0]- 50, self.conds[0] - 50],
            mode='text',
            text=[f'<b>{self.tern_sys[0]}</b>', f'<b>{self.tern_sys[2]}</b>', f'<b>{self.tern_sys[1]}</b>'],
            textposition='top center',
            showlegend=False,
            textfont=dict(size=12)
        ))

        legend_elements = []
        for name, color in self.color_map.items():
            legend_elements.append(dict(
                x=0, y=0, z=0, xref='paper', yref='paper', zref='paper',
                text = name, marker = dict(color = color),
            ))
        for entry in legend_elements:
            fig.add_trace(go.Scatter3d(
            x=[None], y=[None], z=[None], mode='lines+text',
            marker=dict(color=entry['marker']['color']),
            name=entry['text'],
            textfont=dict(size=8)  # Adjust the font size here
            ))
        
        fig.update_layout(
            legend=dict(
                x=0.95, y=0.95, xanchor='left', yanchor='top'    
            ),
            autosize = True,
            margin = dict(l = 50, r = 50, b = 50, t = 50),
            scene=dict(
                zaxis = dict(range=[self.conds[0] - 50 - self.temp_slider[0], self.conds[1] + self.temp_slider[1]],
                            title='Temperature (C)'), 
                xaxis = dict(title=' ',
                        showticklabels=False,
                        showaxeslabels=False,
                        showgrid=False,
                ),
                yaxis = dict(title=' ',
                        showticklabels=False,
                        showaxeslabels=False,
                        showgrid=False,
                ),
                xaxis_visible=True,
                yaxis_visible=True,
                zaxis_visible=True,
                bgcolor='white',
                camera=dict(
                    projection=dict(type='orthographic'),         
                )
            )
        )

        return fig


class ternary_gtx_plotter(ternary_interpolation):
    def __init__(self, tern_sys: List[str], dir: str, **kwargs):
        delta = kwargs.get('delta', 0.025)
        interp_type = kwargs.get('interp_type', 'linear')
        param_format = kwargs.get('param_format', 'linear')
        L_tern = kwargs.get('L_tern', [0, 0])  # ternary interaction parameters (H, S)
        L_dict = kwargs.get('L_dict', {})  # binary interaction parameters
        super().__init__(tern_sys, dir, interp_type=interp_type, param_format=param_format, delta=delta, L_tern=L_tern, L_dict=L_dict)
        self.temp_slider = kwargs.get('temp_slider', [0, 0])  # temperature slider for the plot
        self.T_incr = kwargs.get('T_incr', 10)  # temperature increment for the grid

    def init_sys(self):
        self.tern_sys_name = '-'.join(sorted(self.tern_sys))
        self.phases = self.hsx_df['Phase Name'].unique().tolist()

        solid_phases = self.phases.copy()
        solid_phases.remove('L')
        color_array = px.colors.qualitative.Dark24
        self.color_map = dict(zip(solid_phases, color_array))
        self.color_map['L'] = 'cornflowerblue'

        fusion_temp = pd.read_json(os.path.join(fusion_temps_file), typ='series')
        tern_temp = fusion_temp[self.tern_sys].values 
        max_temp = round(np.max(tern_temp) + 500)
        min_temp = round(np.min(tern_temp))
        self.conds = [np.min(np.array([0, min_temp - 200])), max_temp + self.temp_slider[1]]
        self.T_grid = np.arange(self.conds[0], self.conds[1] + self.T_incr, self.T_incr)
        self.hsx_df['x0'] = self.hsx_df['x0'].round(4)
        self.hsx_df['x1'] = self.hsx_df['x1'].round(4)
        self.hsx_df = self.hsx_df.rename(columns={'Phase Name': 'Phase'})
        self.hsx_df['Colors'] = self.hsx_df['Phase'].map(self.color_map)

        self.df_Tgroups = {}
        for T in self.T_grid:
            self.hsx_df['G'] = self.hsx_df['H'] - T*self.hsx_df['S']
            self.df_Tgroups[T] = self.hsx_df[['x0', 'x1', 'G', 'Phase', 'Colors']].copy()
        
        print('Initialization complete')

    def process_data(self):
        self.init_sys()
        start_time = time.time()
        self.equil_df_list = []
        for T in self.T_grid:       
            if T < self.conds[0]:
                continue  
            points = np.array(self.df_Tgroups[T][['x0', 'x1', 'G']])
            simplices = gliq_lowerhull3(points, vertical_simplices=True)
            simplex_vertices = []
            for simplex in simplices:
                simplex_vertices.append(points[simplex])

            final_phases = []
            for simplex in simplices:
                phase1 = self.df_Tgroups[T].loc[simplex[0], 'Phase']
                phase2 = self.df_Tgroups[T].loc[simplex[1], 'Phase']
                phase3 = self.df_Tgroups[T].loc[simplex[2], 'Phase']
                
                phase_arr = np.array([phase1, phase2, phase3])
                final_phases.append(phase_arr)

            data = []
            for i, simplex in enumerate(simplices):
                labels = final_phases[i]
                if len(set(labels)) == 0:
                    continue
                else:
                    x0_coords = [points[vertex][0] for vertex in simplex] 
                    x1_coords = [points[vertex][1] for vertex in simplex]
                    t_val = T

                j = 0
                for x0, x1 in zip(x0_coords, x1_coords):
                    label = labels[j]
                    color = self.color_map[label]
                    data.append([x0, x1, t_val, label, color])
                    j += 1

            temp_df = pd.DataFrame(data, columns=['x0', 'x1', 'T', 'Phase', 'Colors'])

            temp_df = cartesian_to_ternary(temp_df)
            temp_df['T'] = temp_df['T'] - 273.15

            self.equil_df_list.append(temp_df)

        end_time = time.time()
        print(f"Lower hull evaluation and post processing time:: {end_time - start_time} seconds for temperature increment of {self.T_incr}")

    def plot_ternary(self):
        fig = go.Figure()

        self.plotting_df = pd.concat(self.equil_df_list)
        self.liq_plotting_df = self.plotting_df[self.plotting_df['Phase'] == 'L']
        self.solid_plotting_df = self.plotting_df[self.plotting_df['Phase'] != 'L']
        self.solid_plotting_df = self.solid_plotting_df.sort_values('T').drop_duplicates(subset=['x0', 'x1'], keep='last')

        for index, row in self.solid_plotting_df.iterrows():
            x0 = row['x0']
            x1 = row['x1']
            label = row['Phase']
            color = row['Colors']
            new_row = {'x0': x0, 'x1': x1, 'T': self.conds[0], 'Phase': label, 'Colors': color}
            new_row_df = pd.DataFrame([new_row])
            self.solid_plotting_df = pd.concat([self.solid_plotting_df, new_row_df])

        self.liq_plotting_df = self.liq_plotting_df.sort_values('T').drop_duplicates(subset=['x0', 'x1'], keep='first')


        solid_points = np.array(list(zip(self.solid_plotting_df['x0'], self.solid_plotting_df['x1'], self.solid_plotting_df['T'])))
        liq_points = np.array(list(zip(self.liq_plotting_df['x0'], self.liq_plotting_df['x1'], self.liq_plotting_df['T'])))
        cart_liq_points = [ternary_to_cartesian(point[0], point[1]) for point in liq_points]
        self.triangulation = Delaunay(cart_liq_points)
        triangles = self.triangulation.simplices

        # try:
        #     for point in solid_points:
        #         height = point_to_surface_height(point, liq_points, self.triangulation, triangles)[0]
        #         if height > 1:
        #             new_row = {'x0': point[0], 'x1': point[1], 'T': point[2] + 3, 'Phase': 'L', 'Colors': 'cornflowerblue'}
        #             new_row_df = pd.DataFrame([new_row])
        #             self.liq_plotting_df = pd.concat([self.liq_plotting_df, new_row_df])
        # except Exception as e:
        #     print('Solid meshing error:', e)

        liq_points = np.array(list(zip(self.liq_plotting_df['x0'], self.liq_plotting_df['x1'], self.liq_plotting_df['T'])))
        cart_liq_points = [ternary_to_cartesian(point[0], point[1]) for point in liq_points]
        self.triangulation = Delaunay(cart_liq_points)
        triangles = self.triangulation.simplices

        self.plotting_df = pd.concat([self.solid_plotting_df, self.liq_plotting_df])

        # trace = go.Scatter3d(
        #     x = self.liq_plotting_df['x0'], y = self.liq_plotting_df['x1'], z = self.liq_plotting_df['T'],
        #     mode = 'markers', marker = dict(size = 5, color = self.liq_plotting_df['Colors']),
        #     showlegend=False,
        # )
        # fig.add_trace(trace)

        for label, group in self.solid_plotting_df.groupby('Phase'):
            fig.add_trace(go.Scatter3d(
                x = group['x0'], y = group['x1'], z = group['T'],
                mode = 'lines', line = dict(color = group['Colors'], width = 10),
                showlegend = False, opacity = 1,
            ))

        fig.add_trace(go.Mesh3d(
            x = self.liq_plotting_df['x0'], y = self.liq_plotting_df['x1'], z = self.liq_plotting_df['T'],
            i = triangles[:, 0], j = triangles[:, 1], k = triangles[:, 2],
            opacity = 0.7, colorscale = 'Viridis', intensity = self.liq_plotting_df['T'],
            showscale = False,
        ))

        for phase, color in self.color_map.items():
            fig.add_trace(go.Scatter3d(
                x=[None], y=[None], z=[None], mode='markers',
                marker=dict(color=color, size=10, opacity=1.0),
                name=phase,
                textfont=dict(size=8),
                showlegend=True
            ))

        fig.add_trace(go.Scatter3d(
            x=[0, 0.5, 1, 0],
            y=[0, np.sqrt(3)/2, 0, 0],
            z=[self.conds[0], self.conds[0], self.conds[0], self.conds[0]],
            mode='lines',
            line=dict(color='black', width=5),
            name = 'axes',
            showlegend=False
        ))

        # FIXTHISLATER
        # fig.add_trace(go.Scatter3d(
        #     x=[-0.02, 0.48, 0.98, -0.02],
        #     y=[0.02, np.sqrt(3)/2 + 0.02, .02, .02],
        #     z=[self.conds[0] - 50, self.conds[0] - 50, self.conds[0]- 50, self.conds[0] - 50],
        #     mode='text',
        #     text=[f'<b>{self.tern_sys[0]}</b>', f'<b>{self.tern_sys[2]}</b>', f'<b>{self.tern_sys[1]}</b>'],
        #     textposition='top center',
        #     showlegend=False,
        #     textfont=dict(size=12)
        # ))
         
        fig.update_layout(
            legend=dict(
                x=0.95, y=0.95, xanchor='left', yanchor='top'    
            ),
            autosize = True,
            margin = dict(l = 50, r = 50, b = 50, t = 50),
            scene=dict(
                zaxis = dict(range=[self.conds[0] - 50 - self.temp_slider[0], self.conds[1] + self.temp_slider[1]],
                            title='Temperature (C)'), 
                xaxis = dict(title=' ',
                        showticklabels=False,
                        showaxeslabels=False,
                        showgrid=False,
                ),
                yaxis = dict(title=' ',
                        showticklabels=False,
                        showaxeslabels=False,
                        showgrid=False,
                ),
                xaxis_visible=True,
                yaxis_visible=True,
                zaxis_visible=True,
                bgcolor='white',
                camera=dict(
                    projection=dict(type='orthographic'),         
                )
            )
        )

        return fig    

