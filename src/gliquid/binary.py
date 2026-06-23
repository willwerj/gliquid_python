"""
Authors: Joshua Willwerth, Shibo Tan, Abrar Rauf
Last Modified: June 16 2026
Description: This script is designed for the thermodynamic modeling of two-component systems.
It provides tools for fitting the non-ideal mixing parameters of the liquid phase from T=0K DFT-calculated phases and
digitized equilibrium phase boundary data. The data stored and produced may be visualized using the BLPlotter class
GitHub: https://github.com/willwerj 
ORCID: https://orcid.org/0009-0004-6334-9426
"""
from __future__ import annotations

import copy
import json
import math
import subprocess
import sys
import tempfile
import time
import numbers
import os
import numpy as np
import pandas as pd
import sympy as sp

from matplotlib import cm
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.ticker import ScalarFormatter
import plotly.graph_objects as go
from itertools import combinations
from concurrent.futures import ThreadPoolExecutor, as_completed
from io import StringIO
from pymatgen.core import Composition
from pymatgen.analysis.phase_diagram import PDPlotter, PhaseDiagram, PDEntry  # The PMG PDPlotter source code is modified here
import gliquid.load_binary_data as lbd
from gliquid.hsx import HSX

_x_step = 0.01  # Sets composition grid precision; has not been tested for values other than 0.01
_x_prec = len(str(_x_step).split('.')[-1])
_x_vals = np.arange(0, 1 + _x_step, _x_step)
tau = 8000

# Define base thermodynamic symbols
# These can be used by the BinaryLiquid class to construct custom expressions
xb_sym, t_sym, a_sym, b_sym, c_sym, d_sym = sp.symbols('x t a b c d')
_DEFAULT_PARAMS = [0, 0, 0, 0]  # Default parameter values

def build_init_triangle(param_format: str = 'linear', dft_ch: PhaseDiagram = None,
                        asym_scale=float((np.sqrt(5) - 1) / 2.0), invert_scale=-1.0321) -> list[list[float]]:
    
    if param_format in ['linear', 'combined']:
        guess_keys = ['L0_b', 'L1_b']
    elif param_format in ['comb-exp']:
        guess_keys = ['L0_b', 'L1_a']
    elif param_format in ['hs_partition']:
        guess_keys = ['L0_a', 'L1_a']
    else:
        raise ValueError(f"'param_format' must be either 'linear', 'combined', or 'comb-exp'.")
    
    if dft_ch is None:
        raise ValueError("DFT convex hull data must be provided to build initial simplex based on hull features.")
    
    stable_compounds = len(dft_ch.facets) != 1
    re_depth = lbd.get_hull_rel_mid_depth(dft_ch) * 2.0
    l0_sign = 1.0 if stable_compounds else invert_scale
    re_skew = lbd.get_hull_rel_enth_skew(dft_ch) * 4.0
    
    param_guesses = {
        'L0_a': [re_depth * l0_sign, re_depth * asym_scale],
        'L1_a': [re_skew, re_skew * asym_scale * invert_scale],
    }
    param_guesses.update({
        'L0_b': [g / tau * invert_scale for g in param_guesses['L0_a']],
        'L1_b': [g / tau * invert_scale for g in param_guesses['L1_a']]})

    pg1 = param_guesses[guess_keys[0]]
    pg2 = param_guesses[guess_keys[1]]

    return [[pg1[0], pg2[0]], [pg1[0], pg2[1]], [pg1[1], pg2[1]]] 

    
R = 8.314  # J/(mol*K), universal gas constant

def linear_expr(a: sp.Expr, b: sp.Expr) -> sp.Expr:
    return a + b * t_sym

def exponential_expr(a: sp.Expr, b: sp.Expr) -> sp.Expr:
    return a * sp.exp(-t_sym / b)

def combined_expr(a: sp.Expr, b: sp.Expr, c: sp.Expr) -> sp.Expr:
    return exponential_expr(linear_expr(a, b), c)

_L0_LINEAR_EXPR = linear_expr(a_sym, b_sym)
_L1_LINEAR_EXPR = linear_expr(c_sym, d_sym)
_L0_LIN_EXP_EXPR = combined_expr(a_sym, b_sym, sp.Integer(tau))
_L1_LIN_EXP_EXPR = combined_expr(c_sym, d_sym, sp.Integer(tau))

def build_thermodynamic_expressions(param_format: str = 'linear',
                                    ga_expr: sp.Symbol = 0*t_sym,
                                    gb_expr: sp.Symbol =  0*t_sym) -> dict[str, sp.Expr]:
    """
    Builds a dictionary of thermodynamic Sympy expressions based on provided base expressions.

    Args:
        param_format (str): Format of the non-ideal mixing parameters.
        ga_expr: Expression for the referenced Gibbs free energy of component A.
        gb_expr: Expression for the referenced Gibbs free energy of component B.

    Returns:
        dict[str, sp.Expr]: A dictionary mapping equation names to their Sympy expressions.
    """

    if param_format == 'linear':
        l0_expr, l1_expr = _L0_LINEAR_EXPR, _L1_LINEAR_EXPR
    elif param_format in ['combined', 'comb-exp']:
        l0_expr, l1_expr = _L0_LIN_EXP_EXPR, _L1_LIN_EXP_EXPR
    else:
        raise ValueError(f"'param_format' must be either 'linear', 'combined', or 'comb-exp'.")

    xa = 1 - xb_sym

    # Ideal mixing Gibbs energy
    g_ideal = R * t_sym * (xa * sp.log(xa) + xb_sym * sp.log(xb_sym))

    # Excess Gibbs energy (2-term Redlich-Kister polynomial)
    # g_xs = L0 * (xa * xb) + L1 * (xa * xb * (xa - xb))
    # Factored form: xa * xb * (L0 + L1 * (xa - xb))
    g_xs = xa * xb_sym * (l0_expr + l1_expr * (xa - xb_sym))
    
    # Total Gibbs energy of liquid phase
    g_liquid = ga_expr * xa + gb_expr * xb_sym + g_ideal + g_xs
    
    # Entropy of liquid phase: S = - (dG/dT)_P,x
    s_liquid = -sp.diff(g_liquid, t_sym)
    s_l0 = -sp.diff(l0_expr, t_sym)
    s_l1 = -sp.diff(l1_expr, t_sym)
    
    # Enthalpy of liquid phase: H = G + TS = G - T*(dG/dT)_P,x
    # Using s_liquid = - (dG/dT), H = g_liquid + t * s_liquid
    h_liquid = g_liquid + t_sym * s_liquid
    h_l0 = l0_expr + t_sym * s_l0
    h_l1 = l1_expr + t_sym * s_l1
    
    # First derivative of G_liquid with respect to xb
    g_prime = sp.diff(g_liquid, xb_sym)
    
    # Second derivative of G_liquid with respect to xb
    g_double_prime = sp.diff(g_prime, xb_sym)

    return {
        'ga': ga_expr, 'gb': gb_expr,
        'l0': l0_expr, 'h_l0': h_l0, 's_l0': s_l0,
        'h_l0_lambdified': sp.lambdify([t_sym, a_sym, b_sym], h_l0, modules='numpy'),
        's_l0_lambdified': sp.lambdify([t_sym, a_sym, b_sym], s_l0, modules='numpy'),
        'l1': l1_expr, 'h_l1': h_l1, 's_l1': s_l1,
        'h_l1_lambdified': sp.lambdify([t_sym, c_sym, d_sym], h_l1, modules='numpy'),
        's_l1_lambdified': sp.lambdify([t_sym, c_sym, d_sym], s_l1, modules='numpy'),
        'g_ideal': g_ideal, 'g_xs': g_xs, 'g_liquid': g_liquid, 'h_liquid': h_liquid, 's_liquid': s_liquid,
        'h_liq_lambdified': sp.lambdify([xb_sym, t_sym, a_sym, b_sym, c_sym, d_sym], h_liquid, modules='numpy'),
        's_liq_lambdified': sp.lambdify([xb_sym, t_sym, a_sym, b_sym, c_sym, d_sym], s_liquid, modules='numpy'),
        'g_prime': g_prime, 'g_double_prime': g_double_prime,
    }


def build_phases_from_chull(ch: PhaseDiagram, components: list[str], component_data: dict[str, dict[str, any]]) -> list[dict]:
    """
    Updates the phase list with new phase data.

    Args:
        ch (PhaseDiagram): A pymatgen PhaseDiagram object containing stable entries.
        components (list): List of component names, used to determine the composition of each phase for the 'comp' key.
        component_data (dict): A dictionary containing thermodynamic data for components, used to add polymorphs as explicit phases.

    Returns:
        list: Updated list of phase dictionaries.
    """
    phases = []
    for i, comp in enumerate(components):
        comp_x = float(i)  # 0 for component A, 1 for component B
        for polymorph in component_data[comp]['polymorphs']:
            phases.insert(-1, {
                'name': polymorph['common_name'],
                'comp': comp_x,
                'enthalpy': polymorph['enthalpy_J_per_mol'],
                'entropy': polymorph['entropy_J_per_mol_K'],
                'is_solution': False,
                'points': [],
            })
    for entry in ch.stable_entries:
        composition = entry.composition.fractional_composition.as_dict().get(components[1], 0)
        if composition in [p['comp'] for p in phases]:  # Skip if a polymorph of the pure element is already added
            continue
        entry_data = getattr(entry, 'data', None) or {}
        is_imputed = bool(entry_data.get('imputed'))
        phases.insert(-1, {
            'name': entry_data.get('label', entry.name) if is_imputed else entry.name,
            'comp': composition,
            'enthalpy': 96485 * ch.get_form_energy_per_atom(entry),
            'entropy': 0,
            'is_solution': False,
            'imputed': is_imputed,
            'points': [],
        })
    phases.sort(key=lambda x: x['comp'])
    phases.append({'name': 'L', 'is_solution': True, 'points': []})
    return phases


def validate_binary_mixing_parameters(input) -> list[int | float]:
    """
    Args:
        input (list[int | float]): A list containing numerical values representing non-ideal mixing parameters.

    Returns:
        list[int | float]: A validated list of four numerical values representing non-ideal mixing parameters.
    """
    if isinstance(input, (list, tuple)):
        if len(input) == 0:
            return list(_DEFAULT_PARAMS)  # Return a copy; _DEFAULT_PARAMS is mutable
        if all(isinstance(item, numbers.Number) and not isinstance(item, bool) for item in input) and len(input) == 4:
            return [float(i) for i in input]  # Creates a copy of the input parameter list
    raise ValueError("Parameters must be input as a list or tuple in the following format: [L0_a, L0_b, L1_a, L1_b]")

class BinaryLiquid:
    """
    Represents a binary liquid system for thermodynamic modeling and phase diagram generation.

    Attributes:
        init_error (bool): Flag indicating if an error occurred during initialization.
        sys_name (str): Binary system name.
        components (list): List of component names.
        component_data (dict): Thermodynamic data for components.
        mean_elt_tm (float): Mean elemental melting temperature.
        pd_ind (int | None): Index of the MPDS phase diagram used.
        mpds_json (dict): MPDS phase equilibrium data for the system.
        digitized_liq (list): Digitized liquidus data points.
        max_liq_temp (float | None): Maximum temperature on the liquidus line.
        min_liq_temp (float | None): Minimum temperature on the liquidus line.
        temp_range (list): Temperature range for calculations.
        comp_range_fit_lim (float): Composition range limit for fitting.
        ignored_comp_ranges (list): Ignored composition ranges.
        dft_type (str): Functional used for DFT calculations.
        dft_ch (PhaseDiagram): DFT convex hull data formatted with pymatgen.
        phases (list): List of phase data.
        _params (list): Liquid non-ideal mixing parameters [L0_a, L0_b, L1_a, L1_b].
        _param_format (str): Formalism used for non-ideal mixing parameters (e.g., 'linear', 'combined', 'comb-exp').
        eqs (dict): Dictionary of thermodynamic Sympy expressions.
        invariants (list): Identified invariant points.
        low_t_exp_phases (list): Low-temperature phases from MPDS JSON.
        guess_symbols (list): Sympy symbols for corresponding to guessed parameters.
        constraints (list): Sympy equations used to store parameter constraints.
        init_triangle (np.ndarray): Initial simplex for Nelder-Mead optimization.
        nmpath (np.ndarray): Nelder-Mead optimization path, stored after running Nelder-Mead for plotting purposes.
        hsx (HSX): HSX object for phase diagram calculations.
    """

    def __init__(self, sys_name: str, components: list, init_error=False, **kwargs):
        self.init_error = init_error
        self.sys_name = sys_name
        self.components = components
        self.component_data = kwargs.get('component_data', {})
        self.mean_elt_tm = np.mean([self.component_data[comp]['T_fusion'] for comp in self.components])
        self.pd_ind = kwargs.get('pd_ind', None)
        self.mpds_json = kwargs.get('mpds_json', {})
        self.digitized_liq = kwargs.get('digitized_liq', [])
        self.max_liq_temp = max(self.digitized_liq, key=lambda x: x[1])[1] if self.digitized_liq else None
        self.min_liq_temp = min(self.digitized_liq, key=lambda x: x[1])[1] if self.digitized_liq else None
        self.temp_range = kwargs.get('temp_range', [])
        self.comp_range_fit_lim = kwargs.get('comp_range_fit_lim', 0.7)
        self.ignored_comp_ranges = kwargs.get('ignored_comp_ranges', [])
        self.dft_type = kwargs.get('dft_type', "GGA")
        self.dft_ch = kwargs.get('dft_ch', None)
        self.phases = kwargs.get('phases', [])
        self._params = kwargs.get('params', [0, 0, 0, 0])
        self._param_format = kwargs.get('param_format', 'linear')
        self.init_triangle = kwargs.get('init_triangle', build_init_triangle(self._param_format, self.dft_ch))
        self.eqs = kwargs.get('eqs', build_thermodynamic_expressions(self._param_format))
        self.invariants = None
        self.low_t_exp_phases = None
        self.guess_symbols = None
        self.constraints = None
        self._ref_params = None
        self.nmpath = None
        self.hsx = None

    def __str__(self):
        return f"BinaryLiquid({self.sys_name})"
    
    def __repr__(self):
        return (f"BinaryLiquid(sys_name='{self.sys_name}', components={self.components}, "
                f"params={self._params}, param_format='{self._param_format}', dft_type='{self.dft_type}')")

    def _rebuild_thermodynamic_expressions(self) -> None:
        """Rebuild thermodynamic expressions and lambdified callables after unpickling."""
        if self.component_data and self.components:
            eqs = build_thermodynamic_expressions(
                param_format=self._param_format,
                ga_expr=self.component_data[self.components[0]]['H_liq'] -
                t_sym * self.component_data[self.components[0]]['S_liq'],
                gb_expr=self.component_data[self.components[1]]['H_liq'] -
                t_sym * self.component_data[self.components[1]]['S_liq'])
        else:
            eqs = build_thermodynamic_expressions(self._param_format)

        hull_points = np.array([[0, 0]] + [[p['comp'], p['enthalpy']] for p in self.phases if 'comp' in p] + [[1, 0]])
        eqs['h_hull_interp'] = np.interp(_x_vals[1:-1], hull_points[:, 0], hull_points[:, 1])
        self.eqs = eqs

    def __getstate__(self):
        """
        Return a pickle-safe state for multiprocessing.

        Lambdified callables and HSX state are intentionally dropped and rebuilt on load.
        """
        state = self.__dict__.copy()
        state['hsx'] = None
        state['eqs'] = {
            key: value for key, value in state.get('eqs', {}).items()
            if 'lambdified' not in key
        }
        return state

    def __setstate__(self, state):
        """Restore pickle state and regenerate thermodynamic callables."""
        self.__dict__.update(state)
        self.hsx = None
        self._rebuild_thermodynamic_expressions()
    
    @classmethod
    def from_cache(cls, input, dft_type='GGA', pd_ind=None, params=[], param_format='linear',
                   comp_range_fit_lim=0.7, include_imputed=False, **kwargs) -> BinaryLiquid:
        """
        Initializes a BinaryLiquid object from cached data.

        Args:
            input (any): Binary system - can be either a list or hyphenated string
            dft_type (str): Type of DFT calculation.
            pd_ind (int | None): Index of MPDS binary phase diagram data in cache or from API call downloads
            params (list): Initial fitting parameters.
            param_format (str): Format of the excess mixing energy params, either 'linear', 'combined', or 'comb-exp'.
            reconstruction (bool): Flag for liquidus reconstruction from prediction - 
                uses cached elemental data for melting points and referenced entropy instead of digitized liquidus.

        Returns:
            BinaryLiquid: Initialized BinaryLiquid object.
        """
        components, sys_name, order_changed = lbd.validate_and_format_binary_system(input)
        params = validate_binary_mixing_parameters(params)
        if order_changed: # Flip L1 parameters if the order of components has been changed and parameters were input
            params[2:] = [-1 * p for p in params[2:]] 

        ch, _ = lbd.get_dft_convexhull(components, dft_type, include_imputed=include_imputed)
        mpds_json, component_data, (digitized_liq, is_partial) = lbd.load_mpds_data(components, pd_ind=pd_ind)

        phases = build_phases_from_chull(ch, components, component_data)

        if 'temp' in mpds_json:
            temp_range = [mpds_json['temp'][0] + 273.15, mpds_json['temp'][1] + 273.15]
        else:
            comp_tms = [component_data[comp]['T_fusion'] for comp in components]
            temp_range = [min(comp_tms) - 50, max(comp_tms) * 1.1 + 50]

        eqs = build_thermodynamic_expressions(
            param_format=param_format,
            ga_expr=component_data[components[0]]['H_liq'] - \
                t_sym * component_data[components[0]]['S_liq'],
            gb_expr=component_data[components[1]]['H_liq'] - \
                t_sym * component_data[components[1]]['S_liq'])

        hull_points = np.array([[p['comp'], p['enthalpy']] for p in phases if 'comp' in p])
        eqs['h_hull_interp'] = np.interp(_x_vals[1:-1], hull_points[:, 0], hull_points[:, 1])

        comp_range = mpds_json.get('comp_range', [0, 100]) # Need to transform liquidus x when comp_range is partial?
        comp_range = [max(min(digitized_liq, key=lambda x: x[0])[0], comp_range[0]/100.0), 
                      min(max(digitized_liq, key=lambda x: x[0])[0], comp_range[1]/100.0)] if digitized_liq else comp_range
        init_error = not bool(digitized_liq) or bool((comp_range[1] - comp_range[0]) < comp_range_fit_lim)
        kwargs.update({
            'mpds_json': mpds_json,
            'component_data': component_data,
            'digitized_liq': digitized_liq,
            'temp_range': temp_range,
            'dft_type': dft_type,
            'dft_ch': ch,
            'phases': phases,
            'params': params,
            'param_format': param_format,
            'eqs': eqs,
            'pd_ind': pd_ind,
            'comp_range_fit_lim': comp_range_fit_lim
        })
        return cls(sys_name, components, init_error, **kwargs)

    def _build_hsx_data(self, h_a: float, h_b: float, s_a: float, s_b: float,
                        liq_h_vals: list, liq_s_vals: list) -> dict:
        """
        Assemble the HSX data dictionary from endpoint and interior liquid H/S values.

        Args:
            h_a (float): Enthalpy of pure component A liquid (J/mol).
            h_b (float): Enthalpy of pure component B liquid (J/mol).
            s_a (float): Entropy of pure component A liquid (J/mol/K), 0 if H_liq is 0 (reference state).
            s_b (float): Entropy of pure component B liquid (J/mol/K), 0 if H_liq is 0 (reference state).
            liq_h_vals (list): Enthalpy values for interior liquid compositions.
            liq_s_vals (list): Entropy values for interior liquid compositions.

        Returns:
            dict: Data dictionary in HSX format with keys 'X', 'S', 'H', 'Phase Name'.
        """
        data = {
            'X': [x for x in _x_vals],
            'S': [s_a] + list(liq_s_vals) + [s_b],
            'H': [h_a] + list(liq_h_vals) + [h_b],
            'Phase Name': ['L'] * len(_x_vals)
        }
        for phase in self.phases:
            if phase.get('is_solution', False):
                continue
            data['H'].append(phase.get('enthalpy', 0))
            data['S'].append(phase.get('entropy', 0))
            data['X'].append(round(phase['comp'], _x_prec))
            data['Phase Name'].append(phase.get('name', 'Unknown'))
        return data

    def to_HSX(self, fmt="dict") -> dict | pd.DataFrame:
        """
        Converts phase data into HSX format for further calculations.

        For temperature-dependent liquid models (comb-exp, combined),
        the excess enthalpy decays toward zero as T → ∞ via the exp(−T/τ) envelope.
        This means:
          • Exothermic mixing (H_xs < 0): H is most negative at low T
          • Endothermic mixing (H_xs > 0): H is least positive at high T
        Since the lower convex hull requires the lowest enthalpy surface, we
        evaluate H at a small set of candidate temperatures and, per composition,
        select the T that minimises H.  S is evaluated at the same T.
        This avoids both a temperature mesh and an expensive iterative procedure
        while correctly capturing the T-dependence of the mixing energy.

        Args:
            fmt (str): Output format ('dict' or 'dataframe').

        Returns:
            dict | pd.DataFrame: Data in HSX format.
        """
        x_inner = _x_vals[1:-1]
        params = [self.get_L0_a(), self.get_L0_b(), self.get_L1_a(), self.get_L1_b()]

        # Endpoint H and S (always T-independent: pure-element fusion enthalpy/entropy)
        comp_a = self.component_data[self.components[0]]
        comp_b = self.component_data[self.components[1]]
        s_a = 0 if comp_a['H_liq'] == 0 else comp_a['S_liq']
        s_b = 0 if comp_b['H_liq'] == 0 else comp_b['S_liq']
        h_a = comp_a['H_liq']
        h_b = comp_b['H_liq']

        # For the linear model, liquid H and S are analytically T-independent,
        liq_h_vals = self.eqs['h_liq_lambdified'](x_inner, self.mean_elt_tm, *params).flatten().tolist()
        liq_s_vals = self.eqs['s_liq_lambdified'](x_inner, self.mean_elt_tm, *params).flatten().tolist()

        # Build output data
        data = self._build_hsx_data(h_a, h_b, s_a, s_b, liq_h_vals, liq_s_vals)

        if fmt == "dict":
            return data
        if fmt == "dataframe":
            return pd.DataFrame(data)
        else:
            raise ValueError("kwarg 'fmt' must be either 'dict' or 'dataframe'!")
        
    def update_phase_points(self) -> dict:
        """
        Calculates the phase points for given parameter values using the HSX class.

        This method converts phase data into the HSX form and uses HSX code to calculate the liquidus
        and low-temperature DFT phase boundaries.

        Returns:
            data (dict): A dictionary containing the phase data in HSX format, including phase names and components.
        """
        data = self.to_HSX()
        hsx_dict = {
            'data': data,
            'phases': [phase['name'] for phase in self.phases],
            'comps': self.components
        }
        self.hsx = HSX(hsx_dict, [self.temp_range[0] - 273.15, self.temp_range[-1] - 273.15])
        phase_points = self.hsx.get_phase_points()
        for phase in self.phases:
            phase['points'] = phase_points[phase['name']]
        return data

    def get_L0_a(self) -> int | float:
        return self._params[0]

    def get_L0_b(self) -> int | float:
        return self._params[1]

    def get_L1_a(self) -> int | float:
        return self._params[2]

    def get_L1_b(self) -> int | float:
        return self._params[3]
    
    def get_params(self) -> list[int | float]:
        """
        Get a copy of the current parameters such that the BinaryLiquid object will not be modified accidentally.
        
        Returns:
            list: A list of non-ideal mixing parameters in the following format: [L0_a, L0_b, L1_a, L1_b]
        """
        return [p for p in self._params]
    
    def update_params(self, input) -> None:
        """
        Update the non-ideal mixing parameters with validity checks, 
        then recalculate phase boundaries for the new parameters.

        Args:
            input (list[int | float]): A list containing numerical values representing non-ideal mixing parameters.

        Returns:
            None
        """
        self._params = validate_binary_mixing_parameters(input)
        self.update_phase_points()

    def find_invariant_points(self, verbose=False, check_full_ss=True, t_tol=15) -> tuple[list[dict], list[dict]]:
        """
        Identifies invariant points in the MPDS data using the provided MPDS JSON and liquidus data.

        This function does not consider DFT phases, which may differ in composition from the MPDS data. It requires both 
        complete liquidus and JSON data for a binary system.

        Args:
            verbose (bool): If True, outputs additional debugging information.
            t_tol (int): Temperature tolerance for invariant point identification.
            check_full_ss (bool): If True, checks for full composition range solid solutions.

        Returns:
            tuple: A tuple containing two lists:
                - List of identified invariant points.
                - List of low-temperature phases from the MPDS JSON.
        """
        if self.mpds_json['reference'] is None:
            print("System JSON does not contain any data!\n")
            return []

        # Identify phases from MPDS JSON
        phases = lbd.identify_mpds_phases(self.mpds_json)
        self.invariants = [phase for phase in phases if phase['type'] == 'mig']  # Miscibility gaps are not phases. 
        # They are also not really 'invariant points' either but we classify them as such for algorithm purposes.

        # Filter low-temperature phases
        self.low_t_exp_phases = [
            phase for phase in phases
            if (
                phase['type'] in ['lc', 'ss'] and
                phase['tbounds'][0][1] < (self.mpds_json['temp'][0] + 273.15) +
                (self.mpds_json['temp'][1] - self.mpds_json['temp'][0]) * 0.10
            ) or '(' in phase['name']
        ]

        # Terminal output to visualize phase mismatch
        if verbose:
            mpds_phases_strs = [" "] * len(_x_vals)
            mp_phases_strs = [" "] * len(_x_vals)
            # Note: this implementation will probably break if _x_step is modified. Happy debugging!
            for phase in self.low_t_exp_phases:
                if 'cbounds' in phase:
                    min_c_ind = int(phase['cbounds'][0][0] * 100)
                    max_c_ind = min(int(phase['cbounds'][1][0] * 100), len(_x_vals))
                    mpds_phases_strs[min_c_ind:max_c_ind] = ["|"] * (max_c_ind - min_c_ind)
                else:
                    mpds_phases_strs[min(int(phase['comp']*100), len(_x_vals) - 1)] = "|"
            for phase in self.phases:
                if not phase['is_solution'] and phase not in self.components:
                    mp_phases_strs[min(int(phase['comp']*100), len(_x_vals) - 1)] = "|"
            print("\n--- Low temperature phase mismatch ---")
            print("MPDS:", "[" + "".join(mpds_phases_strs) + "]")
            print("MP:  ", "[" + "".join(mp_phases_strs) + "]")
            print("COMP:", " 0" + " "*9 + "10" + " "*8 + "20" + " "*8 + "30" + " "*8 + "40" + " "*8 + "50" + " "*8 +
                   "60" + " "*8 + "70" + " "*8 + "80" + " "*8 + "90" + " "*8 + "100")

        if verbose:
            print('--- Low temperature phases including component solid solutions ---')
            for phase in self.low_t_exp_phases:
                print(phase)

        # Identify full composition solid solutions
        phase_labels = [label[0] for label in self.mpds_json['labels']]
        ss_label = f"({self.components[0]}, {self.components[1]})"
        ss_label_inv = f"({self.components[1]}, {self.components[0]})"
        ss_labels = [
            ss_label, f"{ss_label} ht", f"{ss_label} rt",
            ss_label_inv, f"{ss_label_inv} ht", f"{ss_label_inv} rt"
        ]
        full_comp_ss = bool([label for label in phase_labels if label in ss_labels])
        if full_comp_ss and check_full_ss:
            print('Solidus processing not implemented!')
            self.init_error = True
            return self.invariants, self.low_t_exp_phases

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
        maxima = find_local_maxima(self.digitized_liq)
        minima = find_local_minima(self.digitized_liq)

        # Assign congruent melting points
        if self.low_t_exp_phases:
            for coords in maxima[:]:
                self.low_t_exp_phases.sort(key=lambda x: abs(x['comp'] - coords[0]))
                phase = self.low_t_exp_phases[0]
                if (
                    phase['type'] in ['lc', 'ss'] and
                    abs(phase['comp'] - coords[0]) <= 0.02 and
                    phase['tbounds'][1][1] + t_tol >= coords[1]
                ):
                    phase['type'] = 'cmp'
                    self.invariants.append({
                        'type': phase['type'],
                        'comp': phase['comp'],
                        'temp': phase['tbounds'][1][1],
                        'phases': [phase['name']],
                        'phase_comps': [phase['comp']]
                    })
                    maxima.remove(coords)

        # Sort by descending temperature for peritectic identification
        self.low_t_exp_phases.sort(key=lambda x: x['tbounds'][1][1], reverse=True)

        def find_adj_phases(point: list | tuple) -> tuple[dict, dict]:
            """
            Finds adjacent phases near a given point.

            Args:
                point (list | tuple): A point in composition-temperature space.

            Returns:
                tuple: Two nearest adjacent phases.
            """
            all_lowt_phases = (
                self.low_t_exp_phases +
                [
                    {'name': self.components[0], 'comp': 0, 'type': 'lc',
                        'tbounds': [[], [0, self.component_data[self.components[0]]['T_fusion']]]},
                    {'name': self.components[1], 'comp': 1, 'type': 'lc',
                        'tbounds': [[], [1, self.component_data[self.components[1]]['T_fusion']]]},
                ]
            )
            all_lowt_phases = [p for p in all_lowt_phases if p['tbounds'][1][1] + t_tol >= point[1]]
            lhs_phases = [phase for phase in all_lowt_phases if phase['comp'] < point[0]]
            adj_lhs_phase = None if not lhs_phases else min(lhs_phases, key=lambda x: abs(x['comp'] - point[0]))
            rhs_phases = [phase for phase in all_lowt_phases  if phase['comp'] > point[0]]
            adj_rhs_phase = None if not rhs_phases else min(rhs_phases, key=lambda x: abs(x['comp'] - point[0]))
            return adj_lhs_phase, adj_rhs_phase

        # Identify liquid-liquid miscibility gap labels
        misc_gap_labels = []
        for label in self.mpds_json['labels']:
            delim_label = label[0].split(' ')
            if len(delim_label) == 3 and delim_label[0][0] == 'L' and delim_label[2][0] == 'L':
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

            for shape in self.mpds_json['shapes']:
                if shape['nphases'] != 2:
                    continue
                data = lbd.shape_to_list(shape['svgpath'])
                if not data:
                    continue
                data.sort(key=lambda x: x[1])
                if not (abs(data[-1][1] - nearest_maxima[1]) < t_tol and abs(data[-1][0] - nearest_maxima[0]) < 0.05):
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
                    key=lambda x: abs(tbounds[1][0] - x[0]) + 2 * (abs(tbounds[1][1] - x[1]) / self.temp_range[1])
                )
                tbounds[0] = [tbounds[1][0], adj_mono[1]]
                adj_phases = find_adj_phases(adj_mono)

                if adj_mono[0] < tbounds[1][0]:
                    if adj_phases[0] is not None:
                        phase_comps = [adj_phases[0]['comp']]
                        phases = [adj_phases[0]['name']]
                    if not cbounds:
                        lhs_ind = self.digitized_liq.index(adj_mono)
                        for i in range(lhs_ind + 1, len(self.digitized_liq) - 1):
                            if self.digitized_liq[i + 1][1] < adj_mono[1] <= self.digitized_liq[i][1]:
                                m = ((self.digitized_liq[i + 1][1] - self.digitized_liq[i][1]) /
                                        (self.digitized_liq[i + 1][0] - self.digitized_liq[i][0]))
                                rhs_comp = (adj_mono[1] - self.digitized_liq[i][1]) / m + self.digitized_liq[i][0]
                                cbounds = [adj_mono, [rhs_comp, adj_mono[1]]]
                                break
                elif adj_mono[0] > tbounds[1][0]:
                    if adj_phases[1] is not None:
                        phase_comps = [adj_phases[1]['comp']]
                        phases = [adj_phases[1]['name']]
                    if not cbounds:
                        rhs_ind = self.digitized_liq.index(adj_mono)
                        for i in reversed(range(1, rhs_ind - 1)):
                            if self.digitized_liq[i - 1][1] < adj_mono[1] <= self.digitized_liq[i][1]:
                                m = ((self.digitized_liq[i - 1][1] - self.digitized_liq[i][1]) /
                                        (self.digitized_liq[i - 1][0] - self.digitized_liq[i][0]))
                                lhs_comp = (adj_mono[1] - self.digitized_liq[i][1]) / m + self.digitized_liq[i][0]
                                cbounds = [[lhs_comp, adj_mono[1]], adj_mono]
                                break
                if cbounds and cbounds[0][1] != cbounds[1][1]:
                    cbounds[0][1] = adj_mono[1]
                    cbounds[1][1] = adj_mono[1]
                minima.remove(adj_mono)

            if cbounds:
                self.invariants.append({
                    'type': 'mig',
                    'comp': tbounds[1][0],
                    'cbounds': cbounds,
                    'tbounds': tbounds,
                    'phases': phases,
                    'phase_comps': phase_comps
                })
                maxima.remove(nearest_maxima)
            break

        stable_phase_comps = []

        # Main loop for peritectic phase identification
        for phase in self.low_t_exp_phases:
            if '(' in phase['name']:  # Ignore component SS phases
                continue

            # Congruent melting points will not be considered for peritectic formation but will limit comp search range
            if phase['type'] == 'cmp':
                stable_phase_comps.append(phase['comp'])
                continue

            sections, current_section = [], []
            phase_temp = phase['tbounds'][1][1]

            for i in range(len(self.digitized_liq) - 1):
                liq_point, next_liq_point = self.digitized_liq[i], self.digitized_liq[i + 1]
                liq_temp, next_liq_temp = liq_point[1], next_liq_point[1]

                # Liquidus point is above or equal to phase temp
                if liq_temp >= phase_temp:
                    current_section.append(liq_point)
                    if next_liq_temp >= phase_temp and i + 1 == len(self.digitized_liq) - 1:
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
                        current_section.append(liq_point)  # Add if below phase temp and closer than next point

            # Find endpoints of liquidus segments excluding the component ends
            endpoints = [
                section[i]
                for section in sections
                for i in [0, -1]
                if section[i] not in [self.digitized_liq[0], self.digitized_liq[-1]]
            ]

            # Filter endpoints if there exists a stable phase between the current phase and the liquidus
            for comp in stable_phase_comps:
                endpoints = [
                    ep for ep in endpoints
                    if abs(comp - ep[0]) > abs(phase['comp'] - ep[0])
                    or abs(comp - phase['comp']) > abs(phase['comp'] - ep[0])
                ]

            # Sort by increasing distance to liquidus to find the shortest distance
            endpoints.sort(key=lambda x: abs(x[0] - phase['comp']))

            # Take the closest liquidus point to the phase as the peritectic point
            if endpoints:
                self.invariants.append({
                    'type': 'per',
                    'comp': endpoints[0][0],
                    'temp': phase_temp,
                    'phases': [phase['name']],
                    'phase_comps': [phase['comp']]
                })

            stable_phase_comps.append(phase['comp'])

        # Identify eutectic points
        for coords in minima:
            adj_phases = find_adj_phases(coords)
            phases, phase_comps = zip(*[
                (None, None) if phase is None else (phase['name'], phase['comp'])
                for phase in adj_phases
            ])

            self.invariants.append({
                'type': 'eut',
                'comp': coords[0],
                'temp': coords[1],
                'phases': list(phases),
                'phase_comps': list(phase_comps)
            })

        self.invariants.sort(key=lambda x: x['comp'])
        self.invariants = [inv for inv in self.invariants if inv['type'] not in ['lc', 'ss']]
        if verbose:
            print('--- Identified invariant points ---')
            for inv in self.invariants:
                print(inv)
            print()
        return self.invariants, self.low_t_exp_phases

    def solve_params_from_constraints(self, guessed_vals: dict) -> None:
        """
        Updates the parameters of the object based on guessed values and constraints.

        Args:
            guessed_vals (dict): A dictionary containing guessed values for the parameters.
        """
        for ind, symbol in enumerate([a_sym, b_sym, c_sym, d_sym]):
            try:
                if symbol in guessed_vals:
                    self._params[ind] = float(guessed_vals[symbol])
                elif self.constraints:
                    self._params[ind] = float(self.constraints[symbol].subs(guessed_vals))
            except TypeError:
                raise RuntimeError("Error in constraint equations!")
    
    def lupis_elliott_penalty(self, penalty_cfg: dict | None = None) -> float:
        """
        Assigns a penalty which scales with degree of violation for Lupis-Elliott sign constraints.

        Active penalty term for each violating component uses:
            P = 1 + 2*A * sqrt(d) * |x*y| / (|x| + |y|) 
        where d = x^2 + y^2, A is 'strength', x is normalized enthalpy, and y is entropy.

        Args:
            penalty_cfg (dict | None): Optional Lupis-Elliott penalty configuration.
                Supports shared defaults and optional per-term overrides:
                - 'strength' (float): global default strength, applied to L0. Default: 7.5E-3.
                - Per-term keys: 'l0', 'l1' each containing {'strength'}.
                  L1 defaults to 0 (not penalized) unless 'l1' is explicitly set.
                - Backward compatibility: 'scale' is accepted as an alias for 'strength'

        Returns:
            float: A penalty greater than 1.0 if the parameters violate the Lupis-Elliott sign constraints,
                   otherwise returns 1.0.
        """
        penalty_cfg = penalty_cfg or {}
        global_strength = float(penalty_cfg.get('strength', penalty_cfg.get('scale', 7.5E-3)))
        entropy_scale = float(penalty_cfg.get('hs_ratio', tau))

        def resolve_term_cfg(term: str, default: float | None = None) -> float:
            if default is None:
                default = global_strength
            term_cfg = penalty_cfg.get(term, {})
            if not isinstance(term_cfg, dict):
                term_cfg = {}
            return float(term_cfg.get('strength', term_cfg.get('scale', default)))

        def calculate_penalty(x, y, strength=0.005):
            if strength <= 0:
                return 0.0
            # x*y < 0 is a quick way to check for opposite signs
            if x * y < 0:
                abs_sum = abs(x) + abs(y)
                if abs_sum <= 1e-16:
                    return 0.0
                d = x**2 + y**2
                if d <= 0:
                    return 0.0
                alignment_factor = (2.0 * abs(x * y)) / (abs_sum ** 2)
                return float(strength * np.sqrt(d) * alignment_factor * abs_sum)
            return 0.0

        l0_strength = resolve_term_cfg('l0')
        l1_strength = resolve_term_cfg('l1', default=0.0)
    
        if self._param_format == 'comb-exp':
            lambda_args_vals = [0, self.get_L0_a(), self.get_L0_b()] 
            h_l0 = self.eqs['h_l0_lambdified'](*lambda_args_vals)
            s_l0 = self.eqs['s_l0_lambdified'](*lambda_args_vals)
            return float(1.0 + calculate_penalty(h_l0 / entropy_scale, s_l0, l0_strength))
        elif self._param_format in ['combined', 'linear']:
            largv0, largv1 = [0, self.get_L0_a(), self.get_L0_b()], [0, self.get_L1_a(), self.get_L1_b()]
            h_l0, s_l0 = self.eqs['h_l0_lambdified'](*largv0), self.eqs['s_l0_lambdified'](*largv0)
            h_l1, s_l1 = self.eqs['h_l1_lambdified'](*largv1), self.eqs['s_l1_lambdified'](*largv1)
            return float(1.0 +
                         calculate_penalty(h_l0 / entropy_scale, s_l0, l0_strength) +
                         calculate_penalty(h_l1 / entropy_scale, s_l1, l1_strength))
        return 1.0


    def lxb_penalty(self, penalty_cfg: dict | None = None) -> float:
        """
        Computes a multiplicative distribution-aware penalty on L0_b/L1_b.

        Each active term contributes to a shared factor of the form:

            1 + w * [log(1 + |x - median| / MAD)]^exponent

        The contributions are additive inside a single multiplicative factor:

            penalty = 1 + term(L0_b) + term(L1_b)

        By default, only L0_b is penalized. L1_b is never penalized unless
        ``apply_l1`` is True or an explicit ``l1`` config is provided.

        Args:
            penalty_cfg (dict | None): Dictionary with optional keys:
                - 'l0': {'weight', 'median', 'mad', 'exponent'} — L0_b term config.
                  Defaults: weight=0.10, exponent=2.5. 'median' and 'mad' must be provided.
                - 'l1': {'weight', 'median', 'mad', 'exponent'} — L1_b term config.
                  Not applied by default; requires 'apply_l1': True or explicit 'l1' config.
                - 'apply_l1' (bool): Force-enable/disable L1 term. Default: False.

        Returns:
            float: Multiplicative penalty factor >= 1.0.
        """
        if not penalty_cfg:
            return 1.0

        def _term(x_val: float, cfg: dict | None) -> float:
            if not cfg:
                return 0.0
            weight = float(cfg.get('weight', 0.10))
            median = float(cfg.get('median', 0.0))
            mad = float(cfg.get('mad', 0.0))
            exponent = float(cfg.get('exponent', 2.5))
            if weight <= 0 or mad <= 0:
                return 0.0
            log_term = math.log(1.0 + abs(x_val - median) / mad)
            return weight * (log_term ** exponent)

        use_l1_default = False
        use_l1 = bool(penalty_cfg.get('apply_l1', use_l1_default))

        total = _term(self.get_L0_b(), penalty_cfg.get('l0'))
        if use_l1:
            total += _term(self.get_L1_b(), penalty_cfg.get('l1'))
        return float(1.0 + total)

    def fim_tikhonov_penalty(self, penalty_cfg: dict | None = None) -> float:
        """
        Multiplicative Tikhonov / Fisher-information prior on the free liquid parameters.

        Anchors the free parameters at a reference vector using a supplied precision
        matrix -- typically the baseline Fisher Information Matrix ``J^T J / sigma^2`` in
        the free subspace, normalized per measurement point. Parameter directions the
        liquidus identifies well (large FIM eigenvalues) are held close to the reference;
        poorly-identified directions -- exactly those prone to overfitting when a DFT
        phase is missing -- are left comparatively free. This is the genuine FIM/Tikhonov
        prior that the baseline-anchored ``lxb`` approximation could not express.

            delta   = theta_free - ref_free
            maha    = delta^T @ P @ delta            (squared Mahalanobis distance)
            penalty = 1 + weight * maha ** exponent

        With ``P`` normalized per measurement point, ``maha`` is the mean squared
        standardized liquidus shift induced by moving away from the reference (in units
        of sigma^2), so ``weight`` is portable across systems. ``exponent == 1`` is the
        principled Gaussian/Tikhonov shape (penalty linear in the quadratic form).

        Args:
            penalty_cfg (dict | None): Configuration with keys:
                - 'precision': (k, k) array or length-k diagonal -- the precision metric P.
                - 'ref_params': length-4 reference params [L0_a, L0_b, L1_a, L1_b].
                - 'free_indices': param indices P / ref correspond to. Default [0, 2].
                - 'weight': multiplier on the quadratic form. Default 0.5.
                - 'exponent': power on ``maha``. Default 1.0.

        Returns:
            float: A multiplicative penalty factor >= 1.0 (1.0 when the config is missing
            or degenerate).
        """
        if not penalty_cfg:
            return 1.0
        precision = penalty_cfg.get('precision')
        ref = penalty_cfg.get('ref_params')
        if precision is None or ref is None:
            return 1.0
        weight = float(penalty_cfg.get('weight', 0.5))
        if weight <= 0:
            return 1.0
        exponent = float(penalty_cfg.get('exponent', 1.0))
        free_indices = penalty_cfg.get('free_indices', [0, 2])

        P = np.asarray(precision, dtype=float)
        if P.ndim == 1:
            P = np.diag(P)
        params = self.get_params()
        delta = np.array([params[i] - ref[i] for i in free_indices], dtype=float)
        if P.shape != (delta.size, delta.size):
            return 1.0
        maha = float(delta @ P @ delta)
        if not np.isfinite(maha) or maha <= 0:
            return 1.0
        return float(1.0 + weight * maha ** exponent)

    def _stable_solid_gibbs_at_T(self, comp_x: float, temp_K: float) -> float:
        """
        Returns the Gibbs energy of the thermodynamically stable solid polymorph
        at a given temperature for a pure elemental endpoint.

        For pure elements (comp_x == 0 or 1), the ground state has G = H - T*S = 0 at all T.
        Higher-temperature polymorphs stored in component_data['polymorphs'] may become
        stable above their transition temperature.  This method selects the polymorph
        with the lowest Gibbs energy at *temp_K* and returns that value.

        For non-elemental compositions this returns the enthalpy of the nearest
        DFT phase unchanged (polymorphs are only tracked for pure elements).

        Args:
            comp_x (float): Composition fraction of component B (0 or 1 for pure elements).
            temp_K (float): Temperature in Kelvin at which to evaluate stability.

        Returns:
            float: Gibbs energy (J/mol) of the stable solid at (comp_x, temp_K).
        """
        # Only apply polymorph correction for the pure-element endpoints
        if comp_x not in (0.0, 1.0):
            nearest = min(
                (p for p in self.phases if not p['is_solution'] and 'comp' in p),
                key=lambda p: abs(p['comp'] - comp_x),
                default=None
            )
            return nearest['enthalpy'] if nearest else 0.0

        comp_name = self.components[int(comp_x)]
        polymorphs = self.component_data[comp_name].get('polymorphs', [])

        # Ground state: H=0, S=0 → G=0 at all T
        best_g = 0.0

        for poly in polymorphs:
            t_trans = poly['transition_temperature_K']
            if temp_K < t_trans:
                continue  # This polymorph is not yet stable
            g_poly = poly['enthalpy_J_per_mol'] - temp_K * poly['entropy_J_per_mol_K']
            if g_poly < best_g:
                best_g = g_poly

        return best_g

    def h0_below_ch(self, tol=1e-6) -> bool:
        """
        Checks if the liquid enthalpy curve at T=0K falls below the solid convex hull.

        Args:
            tol (float): Minimum distance that liquid enthalpy curve must be above the solid convex hull.

        Returns:
            bool: True if any part of the liquid enthalpy curve falls below the solid
                  convex hull, False otherwise.
        """
        # Calculate the liquid enthalpy at T=0K.
        lambda_args_vals = [_x_vals[1:-1], 0, self.get_L0_a(), self.get_L0_b(), self.get_L1_a(), self.get_L1_b()]
        h_vals_0k = self.eqs['h_liq_lambdified'](*lambda_args_vals)
        return np.any(h_vals_0k < self.eqs['h_hull_interp'] + tol)
    
    def liquidus_is_continuous(self, tol=2 * _x_step) -> bool:
        """
        Checks if the liquidus line is continuous within a given tolerance.

        Args:
            tol (float): Tolerance for liquidus continuity. Default is twice the step size.

        Returns:
            bool: True if the generated liquidus line is compositionally-continuous, False otherwise.
        """
        last_coords = None
        for coords in self.phases[-1]['points']:
            if last_coords and coords[0] - last_coords[0] > tol:
                return False
            last_coords = coords
        return True
    
    def calculate_deviation_metrics(self, num_points=30, **kwargs) -> tuple[float, float, float, float]:
        """
        Calculates the deviation metrics between the digitized (measured) liquidus and the generated liquidus.

        Args:
            num_points (int): Number of points to sample for deviation metrics. Default is 30.
            **kwargs: Additional keyword arguments:
                - ignored_ranges (bool): If True, ignores the composition ranges specified in self.ignored_comp_ranges.
                - allow_sparse_data (bool): If True, allows for small composition ranges in the generated points.

        Returns:
            tuple: MAE, RMSE, MAPE, RMSPE of the liquidus in temperature units (K) and percentage (%).
        """
        # Convert liquidus data to numpy arrays
        if self.init_error or not self.digitized_liq:
            return (float('inf'),) * 4
        digitized = np.array(self.digitized_liq)
        if not self.phases[-1]['points']:
            self.update_phase_points()
        generated = np.array(self.phases[-1]['points'])

        # Filter out endpoint compositions (x=0 and x=1)
        mask = (digitized[:, 0] != 0) & (digitized[:, 0] != 1)
        if not np.any(mask):
            return (float('inf'),) * 4
        digitized = digitized[mask]
  
        digitized_liq_lims = [digitized[0, 0], digitized[-1, 0]] 
        if kwargs.get('ignored_ranges', True) and self.ignored_comp_ranges: 
            dig_mask = np.ones_like(digitized[:, 0], dtype=bool)
            gen_mask = np.ones_like(generated[:, 0], dtype=bool)
            for lower, upper in self.ignored_comp_ranges:
                gen_mask = gen_mask & ((generated[:, 0] < lower) | (generated[:, 0] > upper))
                dig_mask = dig_mask & ((digitized[:, 0] < lower) | (digitized[:, 0] > upper))    
            generated = generated[gen_mask]
            digitized = digitized[dig_mask]
            ignored_comp_lims = [generated[0, 0], generated[-1, 0]]
        else:
            ignored_comp_lims = [0, 1]

        x_min = max(_x_step, digitized_liq_lims[0], ignored_comp_lims[0])
        x_max = min(1 - _x_step, digitized_liq_lims[1], ignored_comp_lims[1])
        if x_max - x_min < self.comp_range_fit_lim:
            print(f"Error: Large composition range not considered (remaining range = {[float(x_min), float(x_max)]})")
            return (float('inf'),) * 4

        x_mesh = np.linspace(x_min, x_max, num_points)
        mesh_mask = np.ones_like(x_mesh, dtype=bool)
        if kwargs.get('ignored_ranges', True) and self.ignored_comp_ranges:
            for lower, upper in self.ignored_comp_ranges:
                mesh_mask = mesh_mask & ((x_mesh < lower) | (x_mesh > upper))
        x_mesh = x_mesh[mesh_mask]
        
        if len(digitized) < len(x_mesh):
            x_mesh = np.array([pt[0] for pt in digitized])

        if not kwargs.get('allow_sparse_data', False) and len(x_mesh) < 10:
                print("Error: Not enough comparison points for accurate deviation metrics calculation.")
                return (float('inf'),) * 4
        
        # Find closest temperature values for each evaluation point
        Y1 = np.array([digitized[np.argmin(np.abs(digitized[:, 0] - x)), 1] for x in x_mesh])
        Y2 = np.array([generated[np.argmin(np.abs(generated[:, 0] - x)), 1] for x in x_mesh])
        diffs = np.abs(Y1 - Y2)
        pdiffs = diffs / Y1 * 100
        return (float(np.mean(diffs)), float(np.sqrt(np.mean(diffs**2))),
                float(np.mean(pdiffs)), float(np.sqrt(np.mean(pdiffs**2))))
    
    def f(self, guess: list | tuple, **kwargs) -> float:
        """
        Objective function for parameter fitting.

        Args:
            guess (list | tuple): Guessed parameter values to evaluate.
            **kwargs (dict): Additional keyworded arguments to dictate which constraints are applied.

        Returns:
            float: Generated liquidus mean absolute error (MAE) for the given parameter values.
        """
        verbose = kwargs.get('verbose', False)
        guess_dict = {symbol: guess for symbol, guess in zip(self.guess_symbols, guess)}          
        self.solve_params_from_constraints(guess_dict) 
        
        if kwargs.get('check_h0_below_ch', True) and self.h0_below_ch():
            if verbose:
                print(f'T=0K enthalpy constraint violated for params {self.get_params()}')
            return float('inf')
        
        # Update HSX object and generate new phase points
        try:
            self.update_phase_points()
        except (ValueError, TypeError) as e:
            print(e)
            return float('inf')
        
        # Check if generated liquidus is continuous
        if kwargs.get('check_liquidus_continuity', True) and not self.liquidus_is_continuous():
            if verbose:
                print(f'Liquidus continuity constraint violated for guess {self.get_params()}')
            return float('inf')
        
        # Evaluate the liquidus temperature deviation metrics
        _, f_val, _, _ = self.calculate_deviation_metrics(**kwargs)
        if kwargs.get('check_lupis_elliott', True):
            f_val = f_val * self.lupis_elliott_penalty(kwargs.get('lupis_elliott_cfg'))

        # Apply lxb penalty using distribution priors on L0_b/L1_b.
        if kwargs.get('use_lxb_penalty', self._param_format == 'comb-exp'):
            f_val *= self.lxb_penalty(kwargs.get('lxb_penalty_cfg'))

        # Apply Tikhonov/Fisher-information prior anchoring free params at a reference.
        if kwargs.get('use_fim_prior', False):
            f_val *= self.fim_tikhonov_penalty(kwargs.get('fim_prior_cfg'))

        return f_val

    def nelder_mead(self, max_iter=64, tol=0.05, verbose=False, 
                    initial_guesses=[], **kwargs) -> tuple[float, float, np.ndarray]:
        """
        Nelder-Mead algorithm for fitting the liquid non-ideal mixing parameters.

        Args:
            max_iter (int): Maximum number of iterations. Default is 64.
            tol (float): Tolerance for algorithm convergence. Default is 0.05.
            verbose (bool): If True, print updates during optimization. Default is False.
            initial_guesses (list): Reasonable initial values for guessed parameters, determined by self.guess_symbols.
            **kwargs (dict): Additional keyworded arguments passed to f().

        Returns:
            tuple: MAE, RMSE, and Nelder-Mead optimization path.
        """
        if not initial_guesses:
            initial_guesses = self.init_triangle

        # Initial guesses for parameters
        x0 = np.array(initial_guesses, dtype=float)
        self.nmpath = np.empty((3, 5, max_iter), dtype=float)
        initial_time = time.time()

        if verbose:
            print("--- Beginning Nelder-Mead optimization ---")

        for i in range(max_iter):
            start_time = time.time()
            if verbose:
                print("Iteration #", i)

            f_vals = np.empty(x0.shape[0])
            param_vals = np.empty((x0.shape[0], 4))
            for idx, x in enumerate(x0):
                f_vals[idx] = self.f(x, **kwargs) # 3 f() calls
                param_vals[idx] = self.get_params()
            self.nmpath[:, :-1, i] = param_vals
            self.nmpath[:, -1, i] = f_vals
            iworst = np.argmax(f_vals)
            ibest = np.argmin(f_vals)

            # Check if all current simplex vertices are invalid
            if iworst == ibest:
                self.nmpath = self.nmpath[:, :, :i]
                if i == 0:
                    raise RuntimeError("Nelder-Mead initialization has produced a simplex with invalid vertices.")
                else:                    
                    raise RuntimeError("Nelder-Mead algorithm has produced a simplex with invalid vertices.")

            centroid = np.mean(x0[f_vals != f_vals[iworst]], axis=0)
            xreflect = centroid + 1.0 * (centroid - x0[iworst, :])
            f_xreflect = self.f(xreflect, **kwargs) # 1 f() call

            # Simplex reflection step
            if f_vals[iworst] <= f_xreflect < f_vals[2]:
                x0[iworst, :] = xreflect
            # Simplex expansion step
            elif f_xreflect < f_vals[ibest]:
                xexp = centroid + 2.0 * (xreflect - centroid)
                if self.f(xexp, **kwargs) < f_xreflect: # 1 f() call
                    x0[iworst, :] = xexp
                else:
                    x0[iworst, :] = xreflect
            # Simplex contraction step
            else:
                if f_xreflect < f_vals[2]:
                    xcontract = centroid + 0.5 * (xreflect - centroid)
                    if self.f(xcontract, **kwargs) < self.f(x0[iworst, :], **kwargs): # 2 f() calls
                        x0[iworst, :] = xcontract
                    else:  # Simplex shrink step
                        x0[iworst, :] = x0[ibest, :] + 0.5 * (x0[iworst, :] - x0[ibest, :])
                        [imid] = [i for i in [0, 1, 2] if i != iworst and i != ibest]
                        x0[iworst, :] = x0[imid, :] + 0.5 * (x0[imid, :] - x0[ibest, :])
                else:
                    xcontract = centroid + 0.5 * (x0[iworst, :] - centroid)
                    if self.f(xcontract, **kwargs) < self.f(x0[iworst, :], **kwargs): # 2 f() calls
                        x0[iworst, :] = xcontract
                    else:  # Simplex shrink step
                        x0[iworst, :] = x0[ibest, :] + 0.5 * (x0[iworst, :] - x0[ibest, :])
                        [imid] = [i for i in [0, 1, 2] if i != iworst and i != ibest]
                        x0[imid, :] = x0[ibest, :] + 0.5 * (x0[imid, :] - x0[ibest, :])

            if verbose:
                guess_dict = {symbol: float(guess) for symbol, guess in zip(self.guess_symbols, x0[ibest, :])}
                print("Best guess:", guess_dict, f'f={f_vals[ibest]}')
                print("Height of triangle =", 2 * np.max(np.abs(x0 - centroid)))
                print("--- %s seconds elapsed ---" % (time.time() - start_time))

            # Convergence check
            if np.max(np.abs(x0 - centroid)) < tol:
                f_val = self.f(x0[ibest, :], **kwargs)
                kwargs['ignored_ranges'] = False # Re-include all points for final metrics
                mae, rmse, mape, rmspe = self.calculate_deviation_metrics(**kwargs)
                if verbose:
                    print("--- Nelder-Mead converged in %s seconds ---" % (time.time() - initial_time))
                    print("Mean temperature deviation per point between liquidus curves =", mae, '\n')
                self.nmpath = self.nmpath[:, :, :i + 1]
                return f_val, (mae, rmse, mape, rmspe), self.nmpath
                
        raise RuntimeError("Nelder-Mead algorithm did not converge within limit.")

    def fit_parameters(self, verbose=False, n_opts=1, t_tol=15, enable_multi_threading=False, **kwargs) -> list[dict]:
        """
        Fit the liquidus non-ideal mixing parameters for a binary system.

        This function utilizes the Nelder-Mead algorithm to minimize the temperature deviation in the liquidus.

        Args:
            verbose (bool): If True, prints detailed progress and results.
            n_opts (int): Number of optimization attempts. Updates the BinaryLiquid object to reflect the lowest MAE fit
            t_tol (float): Temperature tolerance for invariant point identification.
            enable_multi_threading (bool): If True, uses ThreadPoolExecutor for parallel
                multi-attempt optimization. Defaults to False.
            **kwargs: Additional keyword arguments for fitting options or arguments passed to nelder-mead.
                - max_iter (int): Maximum number of iterations for the Nelder-Mead algorithm. Default is 64.
                - disable_inv_constrs (bool): If True, does not use invariant points as constraints.
                - conservative_masking (bool): If True, auto-detected ignored ranges around
                    missing/solid-solution phases are (a) bounded by the nearest flanking invariant
                    -- where the missing phase's liquidus field actually ends -- instead of spanning
                    to the compound stoichiometry or a pure-element endpoint (avoids over-trimming
                    peritectic compounds, e.g. Gd-Re), (b) suppressed entirely where a stable DFT
                    compound covers the region (see dft_cover_tol, e.g. Au-Ca), and (c) flagged
                    (init_error) and skipped if the masked liquidus fraction exceeds
                    mask_fraction_cap. Default True; set False for legacy endpoint-spanning behavior.
                - mask_fraction_cap (float): Maximum fraction of the digitized liquidus span that
                    conservative_masking may mask before flagging and skipping. Default 0.60.
                - dft_cover_tol (float): Under conservative_masking, a missing experimental phase is
                    treated as adequately represented (and its region left unmasked) if a stable
                    interior DFT compound lies within this composition tolerance of it. Default 0.10.
                - ignored_ranges (bool): If True, ignores the composition ranges specified in self.ignored_comp_ranges.
                - allow_sparse_data (bool): If True, allows for small composition ranges in the generated points.
                - check_full_ss (bool): If True, checks if the full solid solution is present in the system.
                - check_phase_mismatch (bool): If True, checks phase mismatch between invariant points and self.phases.
                - check_lupis_elliott (bool): If True, checks Lupis-Elliott sign constraints.
                - lupis_elliott_cfg (dict): Optional Lupis-Elliott penalty config.
                    Supported keys:
                        * 'strength' (float): global default penalty strength. Default: 7.5E-3.
                        * 'l0' (dict): optional per-term override with {'strength'}.
                        * 'l1' (dict): optional per-term override with {'strength'}.
                          L1 is not penalized unless this key is explicitly provided.
                - check_h0_below_ch (bool): If True, checks if the liquid enthalpy at T=0K is below the solid convex hull.
                - check_liquidus_continuity (bool): If True, checks if the generated liquidus is continuous.
                - params_init (list): Initial parameter guesses for the Nelder-Mead algorithm.
                  parameter magnitude. Default is 1.0.
                - use_lxb_penalty (bool): If True, applies the distribution-aware lxb penalty.
                    Defaults to True for 'comb-exp' models, False for 'combined'/'linear'.
                - lxb_penalty_cfg (dict): Distribution prior configuration dictionary.
                    Supported keys:
                        * 'l0': {'weight', 'median', 'mad', 'exponent'}
                          Defaults: weight=0.10, exponent=2.5. 'median'/'mad' must be provided.
                        * 'l1': {'weight', 'median', 'mad', 'exponent'}
                          Not applied by default; use 'apply_l1': True to enable.
                        * 'apply_l1' (bool): force-enable/disable L1 term. Default: False.
                - use_fim_prior (bool): If True, applies a Tikhonov/Fisher-information
                    prior anchoring the free parameters at a reference. Default False.
                - fim_prior_cfg (dict): Config for the FIM prior. Supported keys:
                        * 'precision': (k, k) array or length-k diagonal precision metric.
                        * 'ref_params': length-4 reference [L0_a, L0_b, L1_a, L1_b].
                        * 'free_indices': param indices for precision/ref. Default [0, 2].
                        * 'weight': multiplier on the quadratic form. Default 0.5.
                        * 'exponent': power on the Mahalanobis distance. Default 1.0.

        Returns:
            list[dict]: Parameter fitting data containing results of all optimization attempts.
        """

        if self.digitized_liq is None:
            print("System missing liquidus data! Ensure that 'BinaryLiquid.digitized_liq' is not empty!\n")
            return []
        
        def find_nearest_phase(composition, tol=0.02):
            sorted_phases = sorted([p for p in self.phases if 'comp' in p], key=lambda x: abs(x['comp'] - composition))
            if not sorted_phases:
                return {}, float('inf')
            nearest = sorted_phases[0]
            deviation = abs(nearest['comp'] - composition)
            if deviation > tol:
                return {}, deviation
            return nearest, deviation
        
        def phase_decomp_near_liq(phase, tol=10):
            for i in range(len(self.digitized_liq) - 1):
                if self.digitized_liq[i][0] == phase['tbounds'][1][0]:
                    return abs(self.digitized_liq[i][1] - phase['tbounds'][1][1]) < tol
                elif self.digitized_liq[i][0] < phase['tbounds'][1][0] < self.digitized_liq[i + 1][0]:
                    return abs((self.digitized_liq[i][1] + self.digitized_liq[i + 1][1]) / 2 - phase['tbounds'][1][1]) < tol

        def merge_comp_ranges(ranges):
            """Merge overlapping/adjacent [lo, hi] composition ranges into a minimal set."""
            if not ranges:
                return []
            ordered = sorted([sorted(r) for r in ranges])
            merged = [list(ordered[0])]
            for lo, hi in ordered[1:]:
                if lo <= merged[-1][1] + 1e-9:
                    merged[-1][1] = max(merged[-1][1], hi)
                else:
                    merged.append([lo, hi])
            return merged

        # Find invariant points, can set self.init_error to True if system is identified to be isomorphous
        if self.invariants is None and self.low_t_exp_phases is None and not kwargs.get('disable_inv_constrs', False):
            self.invariants, self.low_t_exp_phases = self.find_invariant_points(
                verbose=verbose, t_tol=t_tol, check_full_ss=kwargs.get('check_full_ss', True))
            # Low T phases that decompose near the liquidus line and aren't component solid solutions are critical
            critical_phases = [p for p in self.low_t_exp_phases if '(' not in p['name'] and 
                               phase_decomp_near_liq(p, tol=(self.temp_range[1] - self.temp_range[0]) * 0.05)]
        
            # If over half of the low-temperature phases are not represented in DFT, the fit will likely not be the best
            dft_matching_phases = [p for p in critical_phases if find_nearest_phase(p['comp'], tol=0.04)[0]]
            if len(dft_matching_phases) < len(critical_phases)/2.0 and kwargs.get('check_phase_mismatch', True):
                self.init_error = True
        else:
            dft_matching_phases = []

        if self.init_error:
            return []
        
        # Compare invariant points to self.phases to assess solving conditions
        eqs = []
        auto_ignored_ranges = not bool(self.ignored_comp_ranges)

        # Conservative masking: bound auto-detected ignored ranges to the local missing-phase
        # field (nearest flanking invariant) instead of spanning to a pure-element endpoint, and
        # flag-and-skip if the un-modellable masked fraction of the liquidus exceeds a cap.
        conservative_masking = kwargs.get('conservative_masking', True)
        mask_fraction_cap = kwargs.get('mask_fraction_cap', 0.60)
        dft_cover_tol = kwargs.get('dft_cover_tol', 0.10)
        _inv_comps = sorted(iv['comp'] for iv in (self.invariants or []) if 'comp' in iv)
        _dft_comps = [p['comp'] for p in self.phases
                      if not p.get('is_solution') and 0.0 < round(p['comp'], 6) < 1.0]

        def nearest_inv_left(comp, floor=0):
            cands = [x for x in _inv_comps if x < comp - 1e-6]
            return max(cands) if cands else floor

        def nearest_inv_right(comp, ceil=1):
            cands = [x for x in _inv_comps if x > comp + 1e-6]
            return min(cands) if cands else ceil

        def dft_covers(comp):
            """True if a stable interior DFT compound sits within dft_cover_tol of `comp`.
            When DFT has a (possibly off-stoichiometry) compound near a missing experimental
            phase, the convex hull still provides a valid solid reference for the liquidus
            there, so masking that region is unwarranted (e.g. Au-Ca: DFT compounds are
            shifted from the measured ones but coverage is adequate)."""
            return any(abs(c - comp) <= dft_cover_tol for c in _dft_comps)

        if not kwargs.get('disable_inv_constrs', False):
            for inv in self.invariants:
                if inv['type'] == 'mig':
                    x1, t1 = inv['cbounds'][0]  # Bottom left of dome
                    x2, t2 = inv['tbounds'][1]  # Top of dome
                    x3, t3 = inv['cbounds'][1]  # Bottom right of dome

                    eqn1 = sp.Eq(self.eqs['g_double_prime'].subs({xb_sym: x2, t_sym: t2}), 0)
                    eqn4 = sp.Eq(self.eqs['g_prime'].subs({xb_sym: x1, t_sym: t1}), self.eqs['g_prime'].subs({xb_sym: x3, t_sym: t3}))

                    eqs.append([f'mig - {round(x2, 2)} 2nd', '2nd order', t2, eqn1])
                    eqs.append([f'mig - {round(x1, 2)}-{round(x3, 2)} 1st', '1st order', t1, eqn4])

                if inv['type'] == 'cmp':
                    if '(' in inv['phases'][0] and auto_ignored_ranges:
                        if not (conservative_masking and dft_covers(inv['comp'])):
                            if inv['comp'] < 0.5:
                                lo = nearest_inv_left(inv['comp']) if conservative_masking else 0
                                self.ignored_comp_ranges.append([lo, inv['comp']])
                            elif inv['comp'] > 0.5:
                                hi = nearest_inv_right(inv['comp']) if conservative_masking else 1
                                self.ignored_comp_ranges.append([inv['comp'], hi])
                        continue

                    nearest_phase, _ = find_nearest_phase(inv['comp'])
                    if not nearest_phase:
                        continue

                    x1, t1 = nearest_phase['comp'], inv['temp']
                    eqn = sp.Eq(self.eqs['g_liquid'].subs({xb_sym: x1, t_sym: t1}), nearest_phase['enthalpy'])
                    eqs.append([f'cmp - {round(x1, 2)} 0th', '0th order', t1, eqn])

                if inv['type'] == 'per':
                    if '(' in inv['phases'][0] and auto_ignored_ranges:
                        if not (conservative_masking and dft_covers(inv['comp'])):
                            if inv['phase_comps'][0] < inv['comp']:
                                lo = nearest_inv_left(inv['comp']) if conservative_masking else 0
                                self.ignored_comp_ranges.append([lo, inv['comp']])
                            elif inv['phase_comps'][0] > inv['comp']:
                                hi = nearest_inv_right(inv['comp']) if conservative_masking else 1
                                self.ignored_comp_ranges.append([inv['comp'], hi])
                        continue

                    per_phase, _ = find_nearest_phase(inv['phase_comps'][0], tol=0.04)
                    if not per_phase:
                        continue

                    x1, t1 = inv['comp'], inv['temp']
                    x2 = per_phase['comp']
                    g2 = self._stable_solid_gibbs_at_T(x2, t1) if x2 in (0.0, 1.0) else per_phase['enthalpy']

                    eqn1 = sp.Eq(self.eqs['g_liquid'].subs({xb_sym: x1, t_sym: t1}) + self.eqs['g_prime'].subs({xb_sym: x1, t_sym: t1}) * (x2 - x1), g2)
                    eqn2 = sp.Eq(self.eqs['g_liquid'].subs({xb_sym: x1, t_sym: t1}), g2)

                    liq_point_at_phase = min(self.digitized_liq, key=lambda x: abs(x[0] - x2))
                    temp_below_liq = liq_point_at_phase[1] - t1

                    if temp_below_liq > t_tol:
                        eqs.append([f'per - {round(x1, 2)} 0th', '0th order', t1, eqn1])
                    else:
                        eqs.append([f'per - {round(x1, 2)} 0th', 'H-S partition 0th order', t1, eqn2])

                if inv['type'] == 'eut':
                    if None in inv['phase_comps']:
                        continue

                    lhs_phase, _ = find_nearest_phase(inv['phase_comps'][0], tol=0.04)
                    rhs_phase, _ = find_nearest_phase(inv['phase_comps'][1], tol=0.04)

                    invalid_eut = False
                    if not lhs_phase or lhs_phase['comp'] > inv['comp']:
                        if auto_ignored_ranges and not (conservative_masking and dft_covers(inv['phase_comps'][0])):
                            # Conservative: clamp the outer edge to the nearest flanking invariant
                            # (where the missing phase's liquidus field actually ends) instead of
                            # the compound stoichiometry, which over-trims for peritectic compounds.
                            lo = nearest_inv_left(inv['comp']) if conservative_masking else inv['phase_comps'][0]
                            self.ignored_comp_ranges.append([lo, inv['comp']])
                        invalid_eut = True
                    elif '(' in inv['phases'][0] and inv['phase_comps'][0] > 0.05:
                        invalid_eut = True
                    if not rhs_phase or rhs_phase['comp'] < inv['comp']:
                        if auto_ignored_ranges and not (conservative_masking and dft_covers(inv['phase_comps'][1])):
                            hi = nearest_inv_right(inv['comp']) if conservative_masking else inv['phase_comps'][1]
                            self.ignored_comp_ranges.append([inv['comp'], hi])
                        invalid_eut = True
                    elif '(' in inv['phases'][1] and inv['phase_comps'][1] < 0.95:
                        invalid_eut = True
                    if invalid_eut:
                        continue

                    x1 = lhs_phase['comp']
                    x2, t2 = inv['comp'], inv['temp']
                    x3 = rhs_phase['comp']
                    g1 = self._stable_solid_gibbs_at_T(x1, t2) if x1 in (0.0, 1.0) else lhs_phase['enthalpy']
                    g3 = self._stable_solid_gibbs_at_T(x3, t2) if x3 in (0.0, 1.0) else rhs_phase['enthalpy']

                    eqn1 = sp.Eq(self.eqs['g_prime'].subs({xb_sym: x2, t_sym: t2}), (g3 - g1) / (x3 - x1))
                    eqn2 = sp.Eq(self.eqs['g_liquid'].subs({xb_sym: x2, t_sym: t2}) + 
                                self.eqs['g_liquid'].subs({xb_sym: x2, t_sym: t2}) * (x1 - x2), g1)
                    eqn3 = sp.Eq(self.eqs['g_liquid'].subs({xb_sym: x2, t_sym: t2}) + 
                                self.eqs['g_liquid'].subs({xb_sym: x2, t_sym: t2}) * (x3 - x2), g3)

                    eqs.append([f'eut - {round(x2, 2)} 1st', '1st order', t2, eqn1])
                    if g1 <= g3:
                        eqs.append([f'eut - {round(x2, 2)} 0th', '0th order lhs', t2, eqn2])
                    else:
                        eqs.append([f'eut - {round(x2, 2)} 0th', '0th order rhs', t2, eqn3])

        # Conservative masking: merge the auto-detected ignored ranges and, if the
        # un-modellable (masked) fraction of the digitized liquidus exceeds the cap, flag the
        # system as unreliable and skip it rather than fitting a data-starved liquidus.
        if conservative_masking and auto_ignored_ranges and self.ignored_comp_ranges:
            self.ignored_comp_ranges = merge_comp_ranges(self.ignored_comp_ranges)
            liq_lo, liq_hi = self.digitized_liq[0][0], self.digitized_liq[-1][0]
            span = max(liq_hi - liq_lo, 1e-9)
            masked = sum(min(hi, liq_hi) - max(lo, liq_lo) for lo, hi in self.ignored_comp_ranges
                         if min(hi, liq_hi) > max(lo, liq_lo))
            if masked / span > mask_fraction_cap:
                if verbose:
                    print(f"Conservative masking: {masked / span:.0%} of the liquidus would be "
                          f"masked (> {mask_fraction_cap:.0%} cap); flagging system as unreliable "
                          f"(DFT too incomplete) and skipping.")
                self.init_error = True
                return []

        # Test invariant-derived equations for validity as constraints
        two_constr_methods = ['combined', 'linear']
        one_constr_methods = ['comb-exp']
        no1S_constr = ['no1S - L1_b = 0', '0th order', float('inf'), sp.Eq(d_sym, 0)] # enforces L1_b = 0
        nelder_mead_ics = []
        eqs = [eq for eq in eqs if not eq[3] == False] # Remove invalid equations

        if eqs: # 1 or more valid invariant-derived constraint equations
            highest_tm_eq = eqs.pop(eqs.index(max(eqs, key=lambda x: x[2])))
            for eq in eqs: # 2 or more valid invariant-derived constraint equations
                try:
                    if self._param_format in two_constr_methods:
                        self.guess_symbols = [b_sym, d_sym]
                        self.constraints = sp.solve([eq[3], highest_tm_eq[3]], (a_sym, c_sym), rational=False, simplify=False)
                        init_f = min(self.f(v, **kwargs) for v in self.init_triangle)
                        if init_f == float('inf'):
                            continue
                        nelder_mead_ics.append({'f': init_f, 'constrs': [eq, highest_tm_eq], 'init_tri': self.init_triangle})
                    elif self._param_format in one_constr_methods:
                        self.guess_symbols = [b_sym, c_sym]
                        self.constraints = sp.solve([eq[3], no1S_constr[3]], (a_sym, d_sym), rational=False, simplify=False)
                        init_f = min(self.f(v, **kwargs) for v in self.init_triangle)
                        if init_f == float('inf'):
                            continue
                        nelder_mead_ics.append({'f': init_f, 'constrs': [eq, no1S_constr], 'init_tri': self.init_triangle})
                except RuntimeError as e:
                    print("Error while evaluting invariant constraints", e)
                    continue

        # Derive H-S partition constraints using Nelder-Mead for the enthalpy of mixing
        mean_liq_temp = np.mean([point[1] for point in self.digitized_liq])
        if verbose:
            print(f"\nMaximum composition range fitted: {[self.digitized_liq[0][0], self.digitized_liq[-1][0]]}")
            print(f"Ignored composition ranges: {self.ignored_comp_ranges}\n")

        try:
            no_L0Sxs_eq = sp.Eq(sp.diff(self.eqs['l0'], t_sym).subs({t_sym: 0}), 0) # enforce S0xs @0K = 0 -> L0_b != 0 (comb)
            no_L1Sxs_eq = sp.Eq(sp.diff(self.eqs['l1'], t_sym).subs({t_sym: 0}), 0) # enforce S1xs @0K = 0 -> L1_b != 0 (comb)
            self.update_params(kwargs.get('params_init', [])) # Restore parameter defaults

            self.guess_symbols = [a_sym, c_sym]
            self.constraints = sp.solve([no_L0Sxs_eq, no_L1Sxs_eq], (b_sym, d_sym), rational=False, simplify=False)
            hs_partition_triangle = build_init_triangle('hs_partition', self.dft_ch)
            if verbose:
                print("Initial triangle for H-S partition constraints:", hs_partition_triangle)
            init_f, _, _ = self.nelder_mead(tol=10, verbose=verbose, initial_guesses=hs_partition_triangle, **kwargs)

            # Store reference parameter values from the orthogonal H-S partition constraint pass
            self._ref_params = {
                'L0_a': self.get_L0_a(), 'L0_b': self.get_L0_b(),
                'L1_a': self.get_L1_a(), 'L1_b': self.get_L1_b()
            }
            eq1 = sp.Eq(self.eqs['l0'].subs({t_sym: mean_liq_temp}),
                        self.eqs['l0'].subs({t_sym: mean_liq_temp, a_sym: self.get_L0_a(), b_sym: self.get_L0_b()}))
            
            if self._param_format in two_constr_methods:
                eq2 = sp.Eq(self.eqs['l1'].subs({t_sym: mean_liq_temp}),
                            self.eqs['l1'].subs({t_sym: mean_liq_temp, c_sym: self.get_L1_a(), d_sym: self.get_L1_b()}))
                init_tri = [[self.get_L0_b(), self.get_L1_b()],
                            [self.get_L0_b()*0.8, self.get_L1_b()],
                            [self.get_L0_b(), self.get_L1_b()*0.8]]
                self.guess_symbols = [b_sym, d_sym]
                self.constraints = sp.solve([eq1, eq2], (a_sym, c_sym), rational=False, simplify=False)
                hs_init_f = min(self.f(v, **kwargs) for v in init_tri)
                nelder_mead_ics.append({'f': hs_init_f,
                                        'constrs': [['hs_partition', '0th order', mean_liq_temp, e] for e in [eq1, eq2]],
                                        'init_tri': init_tri})
            elif self._param_format in one_constr_methods:
                init_tri = [[self.get_L0_b(), self.get_L1_a()],
                            [self.get_L0_b()*0.8, self.get_L1_a()],
                            [self.get_L0_b(), self.get_L1_a()*0.8]]
                self.guess_symbols = [b_sym, c_sym]
                self.constraints = sp.solve([eq1, no1S_constr[3]], (a_sym, d_sym), rational=False, simplify=False)
                hs_init_f = min(self.f(v, **kwargs) for v in init_tri)
                nelder_mead_ics.append({'f': hs_init_f,
                                        'constrs': [['hs_partition', '0th order', mean_liq_temp, eq1], no1S_constr],
                                        'init_tri': init_tri})
        except RuntimeError as e:
            print("Nelder-Mead process encountered a fatal error while deriving H-S partition constraints: ", e)

        # Sort by ascending initial MAE such that the 'best' constraints are used first (if limited on number of attempts)
        nelder_mead_ics.sort(key=lambda x: x['f']) 
        self.guess_symbols = [b_sym, d_sym] if self._param_format not in one_constr_methods else [b_sym, c_sym]
        solve_symbols = [sym for sym in [a_sym, b_sym, c_sym, d_sym] if sym not in self.guess_symbols]
        fitting_data = []
        enable_multi_threading = bool(enable_multi_threading or kwargs.get('enable_multi_threading', False))

        # Prepare optimization tasks (up to n_opts, limited by available ICs)
        optimization_tasks = []
        for i in range(min(n_opts, len(nelder_mead_ics))):
            optimization_tasks.append((i, nelder_mead_ics[i]))

        if len(optimization_tasks) == 1:
            # Single optimization: run directly without executor overhead
            task_idx, task_ics = optimization_tasks[0]
            result = _run_single_optimization_worker(
                task_idx, task_ics, self, kwargs, verbose,
                one_constr_methods, solve_symbols, mean_liq_temp)
            if result is not None:
                fitting_data.append(result)
        elif optimization_tasks:
            if enable_multi_threading:
                max_workers = min(len(optimization_tasks), n_opts)
                with ThreadPoolExecutor(max_workers=max_workers) as executor:
                    futures = {
                        executor.submit(
                            _run_single_optimization_worker,
                            task_idx, task_ics, self, kwargs, verbose,
                            one_constr_methods, solve_symbols, mean_liq_temp): task_idx
                        for task_idx, task_ics in optimization_tasks
                    }
                    for future in as_completed(futures):
                        task_idx = futures[future]
                        try:
                            result = future.result()
                            if result is not None:
                                fitting_data.append(result)
                        except Exception as e:
                            print(f"Optimization attempt #{task_idx + 1} failed with exception: {e}")
            else:
                for task_idx, task_ics in optimization_tasks:
                    try:
                        result = _run_single_optimization_worker(
                            task_idx, task_ics, self, kwargs, verbose,
                            one_constr_methods, solve_symbols, mean_liq_temp)
                        if result is not None:
                            fitting_data.append(result)
                    except Exception as e:
                        print(f"Optimization attempt #{task_idx + 1} failed with exception: {e}")

        if fitting_data:
            best_fit = min(fitting_data, key=lambda x: x['f'])
            self._params = [best_fit['L0_a'], best_fit['L0_b'], best_fit['L1_a'], best_fit['L1_b']]
            self.nmpath = best_fit['nmpath']
            self.update_phase_points()
        return fitting_data


def _run_single_optimization_worker(task_index, selected_ics, bl_template, run_kwargs_base,
                                    verbose, one_constr_methods, solve_symbols, mean_liq_temp):
    """
    Module-level worker for threaded optimization execution.

    Args:
        task_index (int): Index of the optimization attempt.
        selected_ics (dict): Initial conditions for this optimization attempt.
        bl_template (BinaryLiquid): Template BinaryLiquid object to copy.
        run_kwargs_base (dict): Base kwargs forwarded to Nelder-Mead and objective function.
        verbose (bool): Print progress information.
        one_constr_methods (list[str]): Parameter formats with one free constraint symbol.
        solve_symbols (list[sp.Symbol]): Symbols solved from constraints.
        mean_liq_temp (float): Mean liquidus temperature for reporting L0/L1.

    Returns:
        dict | None: Fitting result dictionary, or None if optimization failed.
    """
    bl_copy: BinaryLiquid = copy.deepcopy(bl_template)
    bl_copy.guess_symbols = [b_sym, d_sym] if bl_copy._param_format not in one_constr_methods else [b_sym, c_sym]

    constrs_str = '/'.join([c[0] for c in selected_ics['constrs']])
    constr_algo = 'hs_partition_constr' if constrs_str.startswith('hs_partition') else 'inv_constr'

    run_kwargs = dict(run_kwargs_base)
    run_kwargs['check_lupis_elliott'] = run_kwargs_base.get('check_lupis_elliott', True)
    run_kwargs['use_lxb_penalty'] = run_kwargs_base.get('use_lxb_penalty', bl_template._param_format == 'comb-exp')
    run_kwargs['lxb_penalty_cfg'] = copy.deepcopy(run_kwargs_base.get('lxb_penalty_cfg'))
    run_kwargs['lupis_elliott_cfg'] = copy.deepcopy(run_kwargs_base.get('lupis_elliott_cfg'))
    run_kwargs['use_fim_prior'] = run_kwargs_base.get('use_fim_prior', False)
    run_kwargs['fim_prior_cfg'] = copy.deepcopy(run_kwargs_base.get('fim_prior_cfg'))

    if verbose:
        print(f"--- Nelder-Mead ICs Attempt #{task_index + 1} (initial f = {round(selected_ics['f'], 2)}) ---")
        for (source, order, temp, eq) in selected_ics['constrs']:
            print(f"Source: {source}, Order: {order}, Temperature: {round(temp, 1)}, Equation: {eq}")
        print("Initial triangle:", selected_ics['init_tri'])

    selected_eqs = [eq[3] for eq in selected_ics['constrs']]
    bl_copy.constraints = sp.solve(selected_eqs, solve_symbols, rational=False, simplify=False)
    try:
        f, (mae, rmse, mape, rmspe), path = bl_copy.nelder_mead(
            verbose=verbose, tol=5,
            initial_guesses=selected_ics['init_tri'], **run_kwargs)
    except RuntimeError as e:
        print("Nelder-Mead process encountered a fatal error: ", e)
        return None

    l0 = float(bl_copy.eqs['l0'].subs({t_sym: mean_liq_temp, a_sym: bl_copy.get_L0_a(), b_sym: bl_copy.get_L0_b()}))
    l1 = float(bl_copy.eqs['l1'].subs({t_sym: mean_liq_temp, c_sym: bl_copy.get_L1_a(), d_sym: bl_copy.get_L1_b()}))
    fit_invs = bl_copy.hsx.liquidus_invariants()[0]
    return {'f': f, 'mae': mae, 'rmse': rmse, 'mape': mape, 'rmspe': rmspe,
            'constrs': constrs_str, 'algo': constr_algo, 'n_iters': path.shape[2], 'nmpath': path,
            'L0_a': bl_copy.get_L0_a(), 'L0_b': bl_copy.get_L0_b(), 'L1_a': bl_copy.get_L1_a(),
            'L1_b': bl_copy.get_L1_b(), 'L0': l0, 'L1': l1,
            'euts': fit_invs['Eutectics'], 'pers': fit_invs['Peritectics'],
            'cmps': fit_invs['Congruent Melting'], 'migs': fit_invs['Misc Gaps']}
    
class BLPlotter:
    """
    A plotting class for BinaryLiquid objects.

    This class contains methods to create various subfigures and visualizations for analyzing
    BinaryLiquid system data. It uses both static matplotlib and interactive Plotly plots.
    """

    def __init__(self, binaryliquid: BinaryLiquid, **plotkwargs):
        """
        Args:
            binaryliquid (BinaryLiquid): BinaryLiquid object containing the system data.
            plotkwargs (dict): Optional keyword arguments for plot customization (e.g., axis margins).
        """
        self._bl = binaryliquid
        self.plotkwargs = plotkwargs or {
            'axes': {'xmargin': 0.005, 'ymargin': 0}
        }

    def get_plot(self, plot_type: str, **kwargs) -> go.Figure | plt.Axes:
        """
        Generates the specified plot for the BinaryLiquid object.

        Args:
            plot_type (str): The type of plot to generate. Supported types include:
                - 'pc': Low-temperature phase comparison plot
                - 'ch', 'ch+g', 'vch': T=0K DFT convex hull plots
                - 'fit', 'fit+liq', 'pred', 'pred+liq': Generated phase diagram plots
                - 'nmp': Nelder-Mead path visualization plot
            kwargs: Additional keyword arguments for customization.

        Returns:
            go.Figure | plt.Axes: The generated plot object (Plotly or Matplotlib).
        """
        valid_plot_types = [
            'pc',
            'ch', 'ch+g', 'vch',
            'fit', 'fit+liq', 'pred', 'pred+liq',
            'nmp'
        ]
        if plot_type not in valid_plot_types:
            raise ValueError(f"Invalid plot type '{plot_type}'. Supported types: {valid_plot_types}")

        fig = None

        # Phase comparison plot
        if plot_type == 'pc':
            fig = self._generate_phase_comparison_plot()

        # Convex hull plots
        elif plot_type in ['ch', 'ch+g', 'vch']:
            fig = self._generate_convex_hull_plot(plot_type, **kwargs)

        # Liquidus fitting and prediction plots
        elif plot_type in ['fit', 'fit+liq', 'pred', 'pred+liq']:
            fig = self._generate_liquidus_fit_plot(plot_type, **kwargs)

        # Nelder-Mead path visualization
        elif plot_type == 'nmp':
            fig = self._generate_nelder_mead_path_plot(**kwargs)

        return fig

    def show(self, plot_type: str, **kwargs) -> None:
        """
        Displays the generated plot.

        Args:
            plot_type (str): The type of plot to generate. Supported types include:
                - 'pc': Low-temperature phase comparison plot
                - 'ch', 'ch+g', 'vch': T=0K DFT convex hull plots
                - 'fit', 'fit+liq', 'pred', 'pred+liq': Generated phase diagram plots
                - 'nmp': Nelder-Mead path visualization plot
            kwargs: Additional keyword arguments passed to `get_plot`.
        """
        fig = self.get_plot(plot_type, **kwargs)

        if isinstance(fig, go.Figure):
            fig.show()
        elif isinstance(fig, plt.Figure):
            fig.figure.show()

    @staticmethod
    def _resolve_stream_path(stream: str | StringIO) -> str:
        if isinstance(stream, str):
            return stream
        if hasattr(stream, "name") and stream.name:
            return str(stream.name)
        raise TypeError("Plotly image export requires a file path or a named stream.")

    @staticmethod
    def _write_plotly_image_with_timeout(fig: go.Figure, stream: str | StringIO, timeout_s: float, **write_kwargs) -> None:
        if timeout_s is None or timeout_s <= 0:
            fig.write_image(stream, **write_kwargs)
            return

        stream_path = BLPlotter._resolve_stream_path(stream)
        with tempfile.TemporaryDirectory(prefix="gliquid_plotly_export_") as temp_dir:
            figure_payload_path = os.path.join(temp_dir, "figure_payload.json")
            kwargs_payload_path = os.path.join(temp_dir, "write_kwargs.json")

            with open(figure_payload_path, "w", encoding="utf-8") as payload_file:
                payload_file.write(fig.to_json())
            with open(kwargs_payload_path, "w", encoding="utf-8") as kwargs_file:
                json.dump(write_kwargs, kwargs_file)

            child_code = (
                "import json\n"
                "import pathlib\n"
                "import plotly.io as pio\n"
                "import sys\n"
                "figure_json = pathlib.Path(sys.argv[1]).read_text(encoding='utf-8')\n"
                "write_kwargs = json.loads(pathlib.Path(sys.argv[3]).read_text(encoding='utf-8'))\n"
                "figure = pio.from_json(figure_json)\n"
                "figure.write_image(sys.argv[2], **write_kwargs)\n"
            )

            try:
                completed = subprocess.run(
                    [sys.executable, "-c", child_code, figure_payload_path, stream_path, kwargs_payload_path],
                    timeout=timeout_s,
                    check=False,
                    capture_output=True,
                    text=True,
                )
            except subprocess.TimeoutExpired as exc:
                raise TimeoutError(f"Plotly image export timed out after {timeout_s:.1f}s") from exc

            if completed.returncode != 0:
                stderr_text = (completed.stderr or "").strip()
                stdout_text = (completed.stdout or "").strip()
                details = stderr_text or stdout_text or "unknown subprocess error"
                raise RuntimeError(f"Plotly subprocess export failed: {details}")

    def write_image(self, plot_type: str, stream: str | StringIO, image_format: str = "svg", export_timeout_s: float = 120.0, **kwargs) -> None:
        """
        Saves the generated plot as an image.

        Args:
            plot_type (str): The type of plot to save.
            stream (str | StringIO): The file path or stream to save the image.
            image_format (str): The format of the image (default is 'svg').
            export_timeout_s (float): Maximum time allowed for Plotly image export.
            kwargs: Additional keyword arguments passed to `get_plot`.
        """
        fig = self.get_plot(plot_type, **kwargs)
        image_format = stream.name.split('.')[-1] if isinstance(stream, StringIO) and stream.name else image_format
        
        if isinstance(fig, go.Figure):
            write_kwargs = {"format": image_format}
            if plot_type in ['ch', 'ch+g', 'vch']:
                write_kwargs.update({"width": 480 * 1.8, "height": 300 * 1.7})  # 960, 700?

            try:
                self._write_plotly_image_with_timeout(
                    fig,
                    stream,
                    timeout_s=export_timeout_s,
                    **write_kwargs,
                )
            except Exception as exc:
                raise RuntimeError(
                    f"Failed to export plot '{plot_type}' to '{stream}' with timeout={export_timeout_s:.1f}s: {exc}"
                ) from exc
        elif isinstance(fig, plt.Figure):
            fig.figure.savefig(stream, format=image_format)
            plt.close(fig)

    def _generate_phase_comparison_plot(self) -> plt.Figure:
        """
        Generates a phase comparison plot showing congruent and incongruent phases
        from MPDS and MP data. The plot consists of two subplots displaying phases
        in different temperature and magnitude ranges.

        Returns:
            plt.Figure: The generated phase comparison plot.
        """
        # Extract low-temperature phase data from MPDS and MP
        ([mpds_congruent_phases, mpds_incongruent_phases, max_phase_temp],
        [mp_phases, mp_phases_ebelow, min_form_e]) = lbd.get_low_temp_phase_data(self._bl.mpds_json, self._bl.dft_ch)

        # Filter out phases containing parentheses
        mpds_congruent_phases = {key: value for key, value in mpds_congruent_phases.items() if '(' not in key}
        mpds_incongruent_phases = {key: value for key, value in mpds_incongruent_phases.items() if '(' not in key}

        # Create subplots with specific layout
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 2), gridspec_kw={'hspace': 0, 'left': 0.117, 'bottom': 0.13, 'right': 0.909})

        def plot_phases(ax, source, color, alpha=0.5):
            """
            Args:
                ax (matplotlib.axes.Axes): The axis to plot on.
                source (dict): Phase data with keys as phase names and values as bounds/magnitudes.
                color (str): Color for the phase fill.
                alpha (float): Transparency for the fill.
            """
            for _, ((lb, ub), mag) in source.items():
                # Ensure a minimum width for labeling
                if ub - lb < 0.026:
                    ave = (ub + lb) / 2
                    lb = ave - 0.013
                    ub = ave + 0.013
                ax.fill_betweenx([min(0, mag), max(0, mag)], lb, ub, color=color, alpha=alpha)
                ax.set_xlim(0, 1)
                ax.margins(x=0, y=0)

        # Plot phases for both subplots
        plot_phases(ax1, mpds_congruent_phases, 'blue')
        plot_phases(ax1, mpds_incongruent_phases, 'purple')
        plot_phases(ax2, mp_phases, 'orange')
        plot_phases(ax2, mp_phases_ebelow, 'red')

        # Check if MPDS phases exist
        mpds_phases = bool(mpds_congruent_phases or mpds_incongruent_phases)

        # Configure y-axis for the first subplot
        if mpds_phases:
            tick_range = np.linspace(0, max_phase_temp, 4)[1:]
            ax1.set_yticks(tick_range)
            ax1.set_yticklabels([format(tick, '.1e') for tick in tick_range])
            ax1.set_ylim(0, 1.1 * max_phase_temp)
        else:
            ax1.set_yticks([])

        ax1.set_ylabel('MPDS', fontsize=11, rotation=90, labelpad=5, fontweight='semibold')
        ax1.yaxis.set_label_position('right')
        ax1.set_xticks([])

        # Configure y-axis for the second subplot
        if mp_phases:
            tick_range = np.linspace(0, min_form_e, 4)
            ax2.set_yticks(tick_range)
            ax2.set_yticklabels([format(tick, '.1e') for tick in tick_range])
            ax2.set_ylim(1.1 * min_form_e, 0)
        elif mpds_phases:
            ax2.set_yticks([0])
            ax2.set_yticklabels([format(0, '.1e')])
            ax2.set_ylim(-1, 0)
        else:
            ax2.set_yticks([])

        ax2.set_ylabel('MP', fontsize=11, rotation=90, labelpad=5, fontweight='semibold')
        ax2.yaxis.set_label_position('right')
        ax2.xaxis.set_minor_locator(plt.MultipleLocator(0.05))
        ax2.set_xticklabels([0, 20, 40, 60, 80, 100])

        # Add a title to the figure
        fig.suptitle('Low Temperature Phase Comparison', fontweight='semibold')

        return fig

    def _generate_convex_hull_plot(self, plot_type: str, **kwargs) -> go.Figure:
            """
            Generates a convex hull plot or phase diagram visualization.

            Args:
                plot_type (str): The type of plot to generate ('ch', 'vch', 'ch+g').
                kwargs: Additional arguments for customization, such as 't_vals' or 't_units' for temperature values.

            Returns:
                go.Figure: The generated Plotly figure.
            """
            if any(len(Composition(comp).as_dict()) > 1 for comp in self._bl.components):
                raise NotImplementedError("This feature is not presently supported for compound components")
            
            if not self._bl.component_data:
                print("BinaryLiquid object phase diagram not initialized! Returning plot without liquid energy")
                plot_type = 'ch' # Override to avoid errors

            if plot_type == 'vch':
                # Generate volume-referenced convex hull
                ch, atomic_vols = lbd.get_dft_convexhull(self._bl.sys_name, self._bl.dft_type, inc_structure_data=True)

                new_entries = [PDEntry(
                    composition=e.composition,
                    energy=atomic_vols[e.composition.reduced_formula] * e.composition.num_atoms
                ) for e in ch.stable_entries]

                vch = PhaseDiagram(new_entries)
                pdp = PDPlotter(vch)
                fig = pdp.get_plot()
                fig.update_yaxes(title={'text': 'Referenced Atomic Volume (Å^3/atom)'})
            else:
                pdp = PDPlotter(self._bl.dft_ch)
            
            if plot_type == 'ch':
                # Use the standard convex hull
                fig = pdp.get_plot()

            else: # plot_type == 'ch+g'
                # Generate convex hull plot with liquidus curves overlaid
                t_vals = kwargs.get('t_vals', [])
                if not isinstance(t_vals, list) or not all(isinstance(t, (int, float)) for t in t_vals):
                    raise ValueError("kwarg 't_vals' must be a list of valid temperatures, either as ints or floats!")
                t_units = kwargs.get('t_units', 'C')
                if not t_units or not isinstance(t_units, str) or t_units not in ['C', 'K']:
                    raise ValueError("kwarg 't_units' must be a string, either 'C' for Celsius or 'K' for Kelvin")
                if t_units and not t_vals:
                    print("No arguments specified for 't_vals', setting 't_units' to 'K'")

                if not t_vals: # Determine max phase temp if there are compounds, else maximum or minimum liquidus temp
                    if not self._bl.phases[-1]['points']:
                        self._bl.update_phase_points()
                    solid_phases = [p for p in self._bl.phases if (p['name'] not in self._bl.components + ['L'])
                                     and len(p['points']) > 0]
                    if solid_phases:
                        max_phase_temp = max([max(p['points'], key=lambda x: x[1])[1] for p in solid_phases])
                    elif max(self._bl.phases[-1]['points'], key=lambda x: x[1])[1] > max(self._bl.component_data.values(),
                                                                                         key=lambda x: x['T_fusion'])['T_fusion']:
                        max_phase_temp = max(self._bl.phases[-1]['points'], key=lambda x: x[1])[1] # liquid misc gap
                    else:
                        max_phase_temp = min(self._bl.phases[-1]['points'], key=lambda x: x[1])[1] # eutectic or azeotrope
                    t_units = 'K'
                    t_vals = [0, max_phase_temp]
                if t_units == 'C':
                    t_vals = [t + 273.15 for t in t_vals if t >= 0]
                else:
                    t_vals = [t for t in t_vals if t >= 0]

                def get_g_curve(A=0, B=0, C=0, D=0, T=0, name="") -> go.Scatter:
                    """
                    Args:
                        A, B, C, D (float): Non-ideal mixing parameters.
                        T (float): Temperature in Kelvin.

                    Returns:
                        go.Scatter: Plotly scatter trace for the Gibbs free energy curve.
                    """
                    g = self._bl.eqs['g_liquid'].subs({t_sym: T, a_sym: A, b_sym: B, c_sym: C, d_sym: D})
                    gliq_vals = sp.lambdify(xb_sym, g, 'numpy')(_x_vals[1:-1]) if g.has(xb_sym) else [0] * len(_x_vals[1:-1])
                    ga = np.float64(self._bl.eqs['ga'].subs({t_sym: T}) / 96485)
                    gb = np.float64(self._bl.eqs['gb'].subs({t_sym: T}) / 96485)
                    if name == "":
                        name += "Ideal " if A == 0 and C == 0 else ""
                        name += "Liquid T=" + str(int(T)) + "K" if t_units == 'K' else str(int(T-273.15)) + "C"

                    return go.Scatter(
                        x=_x_vals,
                        y=[ga] + [g / 96485 for g in gliq_vals] + [gb],
                        mode='lines',
                        name=name
                    )
                # Build the G curves first, then construct a new figure whose data list
                # starts with the G curves followed by the hull traces — identical
                # ordering and legend position to passing them via the data= argument.
                params = self._bl.get_params()
                g_curves = [
                    get_g_curve(A=params[0], B=params[1], C=params[2], D=params[3], T=temp)
                    for temp in reversed(t_vals)
                ]
                hull_fig = pdp.get_plot()
                fig = go.Figure(data=g_curves + list(hull_fig.data), layout=hull_fig.layout)

            fig.update_layout(plot_bgcolor='white',
                              paper_bgcolor='white', 
                              xaxis=dict(title=dict(text="Composition (fraction)", font=dict(size=18)),
                                        tickfont=dict(color='black')),
                              yaxis=dict(title=dict(font=dict(size=18)),
                                        tickfont=dict(color='black')),
                              font=dict(color='black', size=15),  # Sets default font color for all text elements
                              width=750, height=600)
            return fig

    def _generate_liquidus_fit_plot(self, plot_type: str) -> go.Figure:
        """
        Generates liquidus fitting and prediction plots.

        Args:
            plot_type (str): The type of liquidus plot to generate ('fit', 'fit+liq', 'pred', 'pred+liq').

        Returns:
            go.Figure: The generated plot object.
        """

        # Check if the plot type includes the MPDS liquidus
        if plot_type in ['fit+liq', 'pred+liq'] and not self._bl.digitized_liq:
            print("Digitized_liquidus is not initialized! Returning plot without digitized liquidus")

        # Determine if prediction is required
        pred_pd = bool(plot_type in ['pred', 'pred+liq'])

        # Ensure phase points are updated if not already done
        if self._bl.hsx is None:
            self._bl.update_phase_points()

        # Build polymorph transition data for tie lines and labels
        polymorph_transitions = []
        for i, comp in enumerate(self._bl.components):
            comp_data = self._bl.component_data.get(comp, {})
            polymorphs = comp_data.get('polymorphs', [])
            if not polymorphs:
                continue
            # Determine the ground state name from the DFT phases list
            ground_state_name = comp  # default: element name
            for phase in self._bl.phases:
                if phase['name'] != 'L' and 'comp' in phase and phase['comp'] == float(i):
                    if phase.get('enthalpy', 1) == 0:  # ground state has enthalpy = 0
                        ground_state_name = phase['name']
                        break
            for poly in polymorphs:
                polymorph_transitions.append({
                    'name': poly['common_name'],
                    'comp_x_pct': float(i) * 100,  # 0 for component A, 100 for component B
                    'transition_temp_C': poly['transition_temperature_K'] - 273.15,
                    'ground_state_name': ground_state_name,
                })

        # Imputed (phase-energy-imputation) phases are rendered with dashed lines.
        imputed_phases = {p['name'] for p in self._bl.phases if p.get('imputed')}

        # Generate the plot using the HSX plot method
        fig = self._bl.hsx.plot_tx(
            digitized_liquidus=self._bl.digitized_liq if plot_type in ['fit+liq', 'pred+liq'] else None,
            pred=pred_pd,  # Determines generated liquidus color and temperature axis scaling
            polymorph_transitions=polymorph_transitions,
            imputed_phases=imputed_phases,
        )

        return fig

    def _generate_nelder_mead_path_plot(self, **kwargs) -> plt.Figure:
        """
        Generates a visualization of the Nelder-Mead optimization path.

        This method plots the progression of the Nelder-Mead optimization algorithm in the parameter
        space, using triangles to represent each iteration and color coding for iterations and errors.
        To use, BinaryLiquid object field 'nmpath' must be initialized.

        Returns:
            plt.Figure: The generated plot figure.
        """
        if self._bl.nmpath is None:
            raise ValueError("Underlying BinaryLiquid object has no Nelder-Mead path! Generate using `fit_parameters`")
        plot_a_params = kwargs.get('plot_a_params', False)
        fig, ax = plt.subplots(figsize=(8, 5))
        num_iters = self._bl.nmpath.shape[2]
        total_iters_for_scale = int(kwargs.get('nmp_total_iters', num_iters))
        if total_iters_for_scale < 1:
            total_iters_for_scale = num_iters if num_iters > 0 else 1
        total_iters_for_scale = max(total_iters_for_scale, num_iters)

        # Determine the range of temperature deviations (tdev_range)
        tdev_range = [None, None]
        for i in range(num_iters):
            if plot_a_params: # L0_a, L1_a parameters
                path_i = self._bl.nmpath[:, [0, 2, -1], i]
            elif self._bl._param_format == 'comb-exp': # L0_b, L1_a parameters
                path_i = self._bl.nmpath[:, [1, 2, -1], i]
            else: # L0_b, L1_b parameters
                path_i = self._bl.nmpath[:, [1, 3, -1], i]

            t_devs = [float(num) for num in path_i[:, -1] if num != float('inf')]
            if t_devs:
                tdev_range[0] = min(t_devs) if tdev_range[0] is None else min(tdev_range[0], min(t_devs))
                tdev_range[1] = max(t_devs) if tdev_range[1] is None else max(tdev_range[1], max(t_devs))

        override_obj_range = kwargs.get('objective_range', None)
        if override_obj_range is not None:
            if not (isinstance(override_obj_range, (list, tuple)) and len(override_obj_range) == 2):
                raise ValueError("kwarg 'objective_range' must be a 2-item tuple/list: (min_obj, max_obj)")
            tdev_range = [float(override_obj_range[0]), float(override_obj_range[1])]

        if tdev_range[0] is None or tdev_range[1] is None:
            tdev_range = [0.0, 1.0]
        elif tdev_range[0] == tdev_range[1]:
            eps = max(1e-9, abs(tdev_range[0]) * 1e-6)
            tdev_range = [tdev_range[0] - eps, tdev_range[1] + eps]

        # Triangle color mapping (iteration-based)
        sm1 = cm.ScalarMappable(cmap='winter', norm=LogNorm(vmin=1, vmax=total_iters_for_scale))
        triangle_colors = sm1.to_rgba(np.arange(1, num_iters + 1, 1))
        max_tick_exp = int(math.floor(np.log2(total_iters_for_scale)))
        ticks = [2 ** exp for exp in range(max_tick_exp + 1)]
        top_tick = total_iters_for_scale
        lower_tick = ticks[-1]
        if top_tick > lower_tick:
            log_gap_fraction = math.log(top_tick / lower_tick, 2)
            if log_gap_fraction >= 0.18:
                ticks.append(top_tick)
        cbar1 = fig.colorbar(sm1, ax=ax, aspect=14)
        cbar1.minorticks_off()
        cbar1.set_ticks(ticks)
        cbar1.set_ticklabels(ticks)
        cbar1.set_label('Nelder-Mead Iteration', style='italic', labelpad=8, fontsize=12)

        # Marker color mapping (temperature deviation-based)
        sm2 = cm.ScalarMappable(cmap='autumn', norm=plt.Normalize(tdev_range[0], tdev_range[1]))
        cbar2 = fig.colorbar(sm2, ax=ax, aspect=14)
        cbar2.set_label(
            f"Objective Function Value",
            style='italic',
            labelpad=10,
            fontsize=12
        )

        plotted_points = []

        for i in range(num_iters):
            if plot_a_params: # L0_a, L1_a parameters
                path_i = self._bl.nmpath[:, [0, 2, -1], i]
            elif self._bl._param_format == 'comb-exp': # L0_b, L1_a parameters
                path_i = self._bl.nmpath[:, [1, 2, -1], i]
            else: # L0_b, L1_b parameters
                path_i = self._bl.nmpath[:, [1, 3, -1], i]

            triangle = path_i[:, :-1]  # Extract triangle vertices
            t_devs = path_i[:, -1]  # Extract objective values

            # Plot triangles connecting vertices
            coordinates = [triangle[j, :] for j in range(triangle.shape[0])]
            pair_combinations = list(combinations(coordinates, 2))
            for combo in pair_combinations:
                line = np.array(combo)
                ax.plot(
                    line[:, 0], line[:, 1],
                    color=triangle_colors[i],
                    linewidth=(2 - 1.7 * (i / num_iters)),
                    zorder=0
                )

            # Plot markers at triangle vertices
            for point, t_dev in zip(triangle, t_devs):
                if list(point) in plotted_points:
                    continue
                if t_dev != float('inf'):
                    marker_color = sm2.to_rgba(float(t_dev))
                    ax.scatter(
                        point[0],
                        point[1],
                        s=(55 - 54.7 * (i / num_iters)),
                        color=marker_color,
                        marker='^',
                        edgecolor='black',
                        linewidth=0.3,
                        zorder=1
                    )
                else:
                    ax.scatter(
                        point[0],
                        point[1],
                        s=(45 - 44.7 * (i / num_iters)),
                        color='black',
                        label='Incalculable MAE',
                        marker='^',
                        zorder=1
                    )
                plotted_points.append(list(point))

        # Add legend and adjust axis labels
        handles, labels = ax.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        if by_label:
            ax.legend(by_label.values(), by_label.keys())

        # Adjust axis limits for better scaling
        x_axis_range = kwargs.get('x_axis_range', None)
        y_axis_range = kwargs.get('y_axis_range', None)
        if x_axis_range is not None:
            if not (isinstance(x_axis_range, (list, tuple)) and len(x_axis_range) == 2):
                raise ValueError("kwarg 'x_axis_range' must be a 2-item tuple/list: (xmin, xmax)")
            ax.set_xlim(float(x_axis_range[0]), float(x_axis_range[1]))
        if y_axis_range is not None:
            if not (isinstance(y_axis_range, (list, tuple)) and len(y_axis_range) == 2):
                raise ValueError("kwarg 'y_axis_range' must be a 2-item tuple/list: (ymin, ymax)")
            ax.set_ylim(float(y_axis_range[0]), float(y_axis_range[1]))
        if x_axis_range is None or y_axis_range is None:
            ax.autoscale()
            if y_axis_range is None:
                ly, uy = ax.get_ylim()
                ax.set_ylim((uy + ly) / 2 - (uy - ly) / 2 * 1.1, (uy + ly) / 2 + (uy - ly) / 2 * 1.1)
            if x_axis_range is None:
                lx, ux = ax.get_xlim()
                ax.set_xlim((ux + lx) / 2 - (ux - lx) / 2 * 1.1, (ux + lx) / 2 + (ux - lx) / 2 * 1.1)
        if plot_a_params: # L0_a, L1_a parameters
            axes_labels = ['L0_a', 'L1_a'] 
        elif self._bl._param_format == 'comb-exp': # L0_b, L1_a parameters
            axes_labels = ['L0_b', 'L1_a'] 
        else: # L0_b, L1_b parameters
            axes_labels = ['L0_b', 'L1_b'] 
        ax.set_xlabel(axes_labels[0], fontweight='semibold', fontsize=12)
        ax.set_ylabel(axes_labels[1], fontweight='semibold', fontsize=12)
        fig.tight_layout(pad=1.5)

        # Use scientific notation for tick labels
        def set_sci_notation(axis, which='both'):
            # ticks = axis.get_majorticklocs()
            formatter = ScalarFormatter(useMathText=True)
            formatter.set_scientific(True)
            formatter.set_powerlimits((-3, 3))
            if which in ('x', 'both'):
                ax.xaxis.set_major_formatter(formatter)
            if which in ('y', 'both'):
                ax.yaxis.set_major_formatter(formatter)
            fig.canvas.draw_idle()

        set_sci_notation(ax.xaxis, which='x')
        set_sci_notation(ax.yaxis, which='y')

        return fig
