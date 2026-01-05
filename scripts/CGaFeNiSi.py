import os
import json
from matplotlib import pyplot as plt
import plotly.graph_objects as go
import plotly.offline as ploff
from gliquid.config import data_dir
from gliquid.binary import BinaryLiquid, BLPlotter, build_thermodynamic_expressions, t_sym, xb_sym, a_sym, b_sym, c_sym, d_sym
from gliquid.load_binary_data import shape_to_list
from ternary_interpolation.ternary_HSX import ternary_gtx_plotter
import sympy as sp
import numpy as np
from itertools import combinations


def predict_c_ga_system(method=1, show=True):
    if method == 1: # Use predicted parameters for 4-param linear model
        pform = 'linear'
        c_ga_params = [-64640, 7.3, 57820, -37.98] # -7.3 or 7.3? Could try 0
    elif method == 2: # Use Al-C parameters and transmute
        pform = 'combined_no_1S' # can be whatever works best
        print("transmuting Al-C parameters to C-Ga")
        
        al_c_system = BinaryLiquid.from_cache("Al-C", param_format=pform)
        print(al_c_system.mpds_json['reference']['entry'])

        # Had to suppress code block that invalidates eutectics from component solid solutions to run constrained fit
        fit_res = al_c_system.fit_parameters(verbose=True, n_opts=1)
        for fit in fit_res:
            fit.pop('nmpath', None)
            print(fit)

        if show:
            blp = BLPlotter(al_c_system)
            blp.show('fit+liq')
            blp.show('nmp', plot_a_params=True)
        c_ga_params = al_c_system.get_params()
        c_ga_params[2] *= -1 # Flip L1_a to match C-Ga ordering convention

    if show:
        c_ga_system = BinaryLiquid.from_cache("C-Ga", param_format=pform, params=c_ga_params)
        BLPlotter(c_ga_system).show('pred')
    return {c_ga_system.sys_name: {'params': c_ga_params, 'manual_phases': []}}

def fit_c_fe_system(show=True):
    c_fe_system = BinaryLiquid.from_cache("C-Fe", param_format='combined_no_1S')
    manual_phases = [{'name': '(Fe) ht', 'comp': 0.92, 'energy': -0.01, 'points': []}]
    c_fe_system.phases.insert(1, manual_phases[0].copy()) # Add max sol ht Fe phase
    print(c_fe_system.phases)

    # Had to suppress code block that invalidates eutectics from component solid solutions to run constrained fit
    fit_res = c_fe_system.fit_parameters(verbose=True, n_opts=3, ignore_euts=False)
    for fit in fit_res:
        fit.pop('nmpath', None)
        print(fit)

    if show:
        blp = BLPlotter(c_fe_system)
        blp.show('fit+liq')
        blp.show('nmp', plot_a_params=True)
        plt.show()

    return {c_fe_system.sys_name: {'params': c_fe_system.get_params(), 'manual_phases': manual_phases}}

def fit_c_ni_system(show=True):
    c_ni_system = BinaryLiquid.from_cache("C-Ni", param_format='combined_no_1S', comp_range_fit_lim=0)
    print(c_ni_system.mpds_json['reference']['entry'])

    # Had to suppress code block that invalidates eutectics from component solid solutions to run constrained fit
    fit_res = c_ni_system.fit_parameters(verbose=True, n_opts=1, ignore_euts=False, small_range=True)
    for fit in fit_res:
        fit.pop('nmpath', None)
        print(fit)

    if show:
        blp = BLPlotter(c_ni_system)
        blp.show('fit+liq')
        blp.show('nmp', plot_a_params=True)
        plt.show()
    return {c_ni_system.sys_name: {'params': c_ni_system.get_params(), 'manual_phases': []}}


def fit_c_si_system(show=True):
    system = BinaryLiquid.from_cache("C-Si", param_format='combined_no_1S', comp_range_fit_lim=0)
    fit_res = system.fit_parameters(verbose=True, n_opts=1, ignore_euts=False, small_range=True)
    for fit in fit_res:
        fit.pop('nmpath', None)
        print(fit)

    if show:
        blp = BLPlotter(system)
        blp.show('fit+liq')
        blp.show('nmp', plot_a_params=True)
        plt.show()
    return {system.sys_name: {'params': system.get_params(), 'manual_phases': []}}

def fit_fe_ga_system(show=True, params=[]):
    fe_ga_system = BinaryLiquid.from_cache("Fe-Ga", param_format='combined_no_1S', comp_range_fit_lim=0, params=params)
    # fe_ga_system.ignored_comp_ranges.append([0, 0.48])
    manual_phases = [{'name': 'GaFe3 Ga+ ht', 'comp': 0.35, 'energy': -30000, 'points': []},
                     {'name': 'Ga4Fe3', 'comp': 0.571, 'energy': -29000, 'points': []},
                     {'name': '(Fe10) rt', 'comp': 0.10, 'energy': -15400, 'points': []},
                     {'name': '(Fe20) rt', 'comp': 0.20, 'energy': -25500, 'points': []}
                     ]
    fe_ga_system.phases.insert(2, manual_phases[1].copy())
    fe_ga_system.phases.insert(2, manual_phases[0].copy()) 
    fe_ga_system.phases.insert(1, manual_phases[2].copy())  # Add Fe 10 phase
    fe_ga_system.phases.insert(2, manual_phases[3].copy())  # Add Fe 20 phase
    # fe_ga_system.phases[1]['energy'] = -19000
    print(fe_ga_system.phases)

    # fit_res = fe_ga_system.fit_parameters(verbose=True, n_opts=3, check_phase_mismatch=False)
    # for fit in fit_res:
    #     fit.pop('nmpath', None)
    #     print(fit)

    if show:
        blp = BLPlotter(fe_ga_system)
        blp.show('fit+liq')
        # blp.show('ch+g')
        # blp.show('nmp', plot_a_params=True)
        # plt.show()
    return {fe_ga_system.sys_name: {'params': fe_ga_system.get_params(), 'manual_phases': manual_phases}}


def fit_ga_ni_system(show=True, params=[]):
    system = BinaryLiquid.from_cache("Ga-Ni", param_format='combined_no_1S', comp_range_fit_lim=0, params=params, reconstruction=True)
   
    manual_phases = [{'name': 'GaNi5', 'comp': 0.2, 'energy': -23100, 'points': []},
                     {'name': '(Ni89)', 'comp': 0.89, 'energy': -15800, 'points': []},
                     {'name': '(Ni95)', 'comp': 0.95, 'energy': -7900, 'points': []}, 
                     ]
    system.phases.insert(1, manual_phases[0].copy())  # Add GaNi5 phase
    system.phases.insert(7, manual_phases[1].copy())  # Add Ni 89 phase
    system.phases.insert(8, manual_phases[2].copy())  # Add Ni 95 phase
    # system.phases.pop(2)
    # print(system.phases)

    # fit_res = system.fit_parameters(verbose=True, n_opts=5, ignore_euts=False, small_range=True)
    # for fit in fit_res:
    #     fit.pop('nmpath', None)
    #     print(fit)

    if show:
        blp = BLPlotter(system)
        blp.show('fit+liq')
        # blp.show('nmp', plot_a_params=True)
        # plt.show()
    return {system.sys_name: {'params': system.get_params(), 'manual_phases': manual_phases}}


def fit_ga_si_system(show=True, params=[]):
    system = BinaryLiquid.from_cache("Ga-Si", param_format='combined_no_1S', comp_range_fit_lim=0, params=params)
    manual_phases = []

    fit_res = system.fit_parameters(verbose=True, n_opts=5, ignore_euts=False, small_range=True)
    for fit in fit_res:
        fit.pop('nmpath', None)
        print(fit)

    if show:
        blp = BLPlotter(system)
        blp.show('fit+liq')
        blp.show('nmp', plot_a_params=True)
        plt.show()
    return {system.sys_name: {'params': system.get_params(), 'manual_phases': manual_phases}}

def fit_fe_ni_system(show=True, params=[]):
    # Ultra-manual fitting of this system due to large solid solution range

    # Step 1: Fit the Solidus line
    liquid_system = BinaryLiquid.from_cache("Fe-Ni", param_format='combined_no_1S', comp_range_fit_lim=0, params=params)

    # for i, shape in enumerate(liquid_system.mpds_json['shapes']):
    #     print(i, shape.get('label', '-'), shape.get('nphases', '-'))

    delta_H_0k = (0.148-0)*96485 # 14.28 kJ/mol from MP
    delta_H_lit1 = 13220 # Literatue source 1, BCC->FCC transition
    delta_H_lit2 = 2410 # Literature source 2, FCC->BCC transition
    delta_H_lit3 = 7153 # Literature source 3, BCC->FCC transition
    solid_component_data = {'Fe': [0, 912+273.15], 'Ni': [0, 0]} # Fe becomes FCC at 912, Ni is always FCC
    print(f'Fe: H_BCC->FCC = {solid_component_data["Fe"][0]} J/mol, T_BCC->FCC = {solid_component_data["Fe"][1]} K')
    g_fcc_fe = solid_component_data['Fe'][0] - solid_component_data['Fe'][0] / solid_component_data['Fe'][1] * t_sym
    solidus = shape_to_list(liquid_system.mpds_json['shapes'][4]['svgpath']) # (Fe, Ni) solution svgpath
    solidus = sorted([p for p in solidus if p[1] < 1200], key=lambda x: x[0])[:-1] # Filter out upper solidus points
    point_to_remove = [0.47924100000000003, 739.3340000000001]
    solidus = [p for p in solidus if p != point_to_remove]
    
    # Plot the solidus line with Matplotlib
    # plt.figure(figsize=(8, 6))
    # plt.scatter([point[0] for point in solidus], [point[1] for point in solidus],
    #              marker='o', color='b', label='Solidus')
    # plt.xlabel('Composition (Fe)')
    # plt.ylabel('Temperature (K)')
    # plt.title('Fe-Ni Solidus Line')
    # plt.grid(True)
    # plt.legend()
    # plt.show()

    solid_system = BinaryLiquid("Fe-Ni", ["Fe", "Ni"], 
                                component_data=solid_component_data,
                                temp_range=liquid_system.temp_range,
                                mpds_json = liquid_system.mpds_json,
                                dft_ch=liquid_system.dft_ch,
                                digitized_liq=solidus,
                                phases=liquid_system.phases,
                                param_format='combined_no_1S',
                                eqs=build_thermodynamic_expressions('combined_no_1S', ga_expr=g_fcc_fe),
                                )
    solid_system.invariants = [{'type': 'cmp', 'comp': 0.725085, 'temp': 787.148, 'phases': ['FeNi3 rt'], 'phase_comps': [0.725085]}]
    fit_res = solid_system.fit_parameters(verbose=True, n_opts=5, ignore_euts=False, check_full_ss=False, check_h0_below_ch=False)

    for fit in fit_res:
        fit.pop('nmpath', None)
        print(fit)

    if show:
        blp = BLPlotter(solid_system)
        blp.show('fit+liq')
        # blp.show('nmp', plot_a_params=True)
        # plt.show()
        blp.show('ch+g', t_vals=[0, 912+273.15, 1432+273.15], t_units='K')

    # Step 2: Evaluate the solidus fit at 1432C and create 'line compound' phases from the energy values
    print()
    gen_phase_mesh = np.arange(0.1, 1.0, 0.1)
    gen_phase_enth = [solid_system.eqs['g_liquid'].subs({t_sym: 1432+273.15, xb_sym: x, a_sym: solid_system.get_L0_a(),
                                                         b_sym: solid_system.get_L0_b(), c_sym: solid_system.get_L1_a(),
                                                         d_sym: solid_system.get_L1_b()}) for x in gen_phase_mesh]
    manual_phases = [{'name': f'(Fe{100 - int(round(100*x))}, Ni{int(round(100*x))})',
                      'comp': round(x, 2), 'energy': float(enth), 'points': []} for x, enth in zip(gen_phase_mesh, gen_phase_enth)]

    # Step 3: Fit the Liquidus line
    liquid_system.phases = [liquid_system.phases[0]] + [p.copy() for p in manual_phases] + liquid_system.phases[1:]
    liquid_system.invariants = []
    liquid_system.fit_parameters(verbose=True, n_opts=5, ignore_euts=False, check_full_ss=False, check_lupis_elliott=False)

    if show:
        blp = BLPlotter(liquid_system)
        blp.show('fit+liq')
        blp.show('nmp', plot_a_params=True)
        plt.show()
        blp.show('ch+g') # , t_vals=[0, 912+273.15, 1432+273.15], t_units='K')

    return {liquid_system.sys_name: {'params': liquid_system.get_params(), 'manual_phases': manual_phases}}


def fit_fe_si_system(show=True, params=[]):
    system = BinaryLiquid.from_cache("Fe-Si", param_format='combined_no_1S', comp_range_fit_lim=0, params=params)
    manual_phases = []

    fit_res = system.fit_parameters(verbose=True, n_opts=5, ignore_euts=False, small_range=True)
    for fit in fit_res:
        fit.pop('nmpath', None)
        print(fit)

    if show:
        blp = BLPlotter(system)
        blp.show('fit+liq')
        blp.show('nmp', plot_a_params=True)
        plt.show()
    return {system.sys_name: {'params': system.get_params(), 'manual_phases': manual_phases}}

def fit_ni_si_system(show=True, params=[]):
    system = BinaryLiquid.from_cache("Ni-Si", param_format='combined_no_1S', comp_range_fit_lim=0, params=params)
    manual_phases = []

    fit_res = system.fit_parameters(verbose=True, n_opts=5, ignore_euts=False, small_range=True)
    for fit in fit_res:
        fit.pop('nmpath', None)
        print(fit)

    if show:
        blp = BLPlotter(system)
        blp.show('fit+liq')
        blp.show('nmp', plot_a_params=True)
        plt.show()
    return {system.sys_name: {'params': system.get_params(), 'manual_phases': manual_phases}}


def plot_ternary_system(tern_sys, binary_data, param_format='combined_no_1S', temp_slider=[0, 300], t_incr=5, delta=0.025):

    if isinstance(tern_sys, str):
        tern_sys = tern_sys.split('-')
    tern_sys = sorted(tern_sys)

    binary_dict = {sys_name: data.copy() for sys_name, data in binary_data.items() if set(sys_name.split('-')).issubset(set(tern_sys))}
    items_to_remove = []
    items_to_add = {}
    for sys_name, data in binary_dict.items():
        components = sys_name.split('-')
        if not abs(tern_sys.index(components[0]) - tern_sys.index(components[1])) == 1:
            items_to_remove.append(sys_name)  # Remove if components are in wrong order
            data['params'][2] = -data['params'][2]  
            new_sys_name = f"{components[1]}-{components[0]}"
            for phase in data['manual_phases']:
                phase['comp'] = 1 - phase['comp']
            items_to_add[new_sys_name] = {'params': data['params'], 'manual_phases': data['manual_phases']}

    # Remove items after iteration
    for item in items_to_remove:
        binary_dict.pop(item)
    binary_dict.update(items_to_add)
    binary_L_dict = {sys_name: data['params'] for sys_name, data in binary_dict.items()}
    
    plotter = ternary_gtx_plotter(tern_sys, data_dir, interp_type="linear", param_format=param_format,
                                  L_dict=binary_L_dict, temp_slider=temp_slider, T_incr=t_incr, delta=delta)
    plotter.interpolate()

    def binary_to_ternary_phase(phase, sys_name):
        components = sys_name.split('-')
        x0, x1 = 0, 0
        if tern_sys[1] in components:
            if tern_sys[1] == components[1]:
                x0 = phase['comp']
            else: # tern_sys[1] == components[0]
                x0 = 1 - phase['comp']
        
        if tern_sys[2] in components:
            if tern_sys[2] == components[1]:
                x1 = phase['comp']
            else: # tern_sys[2] == components[0]
                x1 = 1 - phase['comp']

        return {'x0': x0, 'x1': x1, 'S': 0, 'H': phase['energy'], 'Phase Name': phase['name']}

    for sys_name, data in binary_dict.items():
        for phase in data['manual_phases']:
            plotter.hsx_df = plotter.hsx_df._append(binary_to_ternary_phase(phase, sys_name), ignore_index=True)

    # remove ga7ni3, feni3, feni, gafe3
    # phases_to_remove = ['Ga7Ni3', 'FeNi3', 'FeNi', 'GaFe3']
    # plotter.hsx_df = plotter.hsx_df[~plotter.hsx_df['Phase Name'].isin(phases_to_remove)].reset_index(drop=True)

    plotter.process_data()
    tern_fig = plotter.plot_ternary()
    ploff.plot(tern_fig, filename=f'./figures/{"".join(tern_sys)}_system.html', auto_open=True)

    # Plot a point at the flux chemistry
    # from ternary_interpolation.ternary_HSX import ternary_to_cartesian
    # cartesian_coords = ternary_to_cartesian(x_A=0.78, x_B=0.11)
    # tern_fig.add_trace(
    #     go.Scatter3d(
    #         x=[cartesian_coords[0]],
    #         y=[cartesian_coords[1]],
    #         z=[778],
    #         mode='markers',
    #         marker=dict(size=4, color='red', symbol='diamond'),
    #         showlegend=False,
    #         opacity=1,
    #     )
    # )
    # print(tern_fig)
    # tern_fig.show()
    # ploff.plot(tern_fig, filename=f'./figures/{"".join(tern_sys)}_manual_system.html', auto_open=True)



if __name__ == "__main__":
    # Note: Use the repo data directory for this script. gliquid.config should be changed
    os.environ["NEW_MP_API_KEY"] = "YOUR_API_KEY_HERE"
    # Fit parameters for the following 10 binary systems:
    # C-Ga, C-Fe, C-Ni, C-Si, Fe-Ga, Ga-Ni, Ga-Si, Fe-Ni, Fe-Si, Ni-Si
    fitted_data_cache = os.path.join(data_dir, "CGaFeNiSi_fitted_data.json")
    fitted_data = {} if not os.path.exists(fitted_data_cache) else json.load(open(fitted_data_cache, 'r'))
    print(f"Fitted data loaded from {fitted_data_cache}:")
    for key, value in fitted_data.items():
        print(f"{key}: {value}")
    print()

    # C-Ga -> DNE, try Al-C instead and transmute parameters for the prediction
    # fitted_data.update(predict_c_ga_system(method=2))
    # json.dump(fitted_data, open(fitted_data_cache, 'w'), indent=4)

    # C-Fe has fittable PD
    # fitted_data.update(fit_c_fe_system())
    # json.dump(fitted_data, open(fitted_data_cache, 'w'), indent=4)

    # C-Ni has fittable PD
    # fitted_data.update(fit_c_ni_system())
    # json.dump(fitted_data, open(fitted_data_cache, 'w'), indent=4)

    # C-Si has fittable PD
    # fitted_data.update(fit_c_si_system())
    # json.dump(fitted_data, open(fitted_data_cache, 'w'), indent=4)

    # Fe-Ga has fittable PD but missing significant phases in DFT
    # fitted_data.update(fit_fe_ga_system(params=fitted_data['Fe-Ga']['params']))
    # json.dump(fitted_data, open(fitted_data_cache, 'w'), indent=4)

    # Ga-Ni has fittable PD
    # fitted_data.update(fit_ga_ni_system(params=fitted_data['Ga-Ni']['params']))
    # json.dump(fitted_data, open(fitted_data_cache, 'w'), indent=4)

    # Ga-Si has fittable PD
    # fitted_data.update(fit_ga_si_system())
    # json.dump(fitted_data, open(fitted_data_cache, 'w'), indent=4)
   
    # Fe-Ni has fittable PD - large solid solution range needs to be handled
    # fitted_data.update(fit_fe_ni_system())
    # json.dump(fitted_data, open(fitted_data_cache, 'w'), indent=4)

    # Fe-Si has fittable PD
    # fitted_data.update(fit_fe_si_system())
    # json.dump(fitted_data, open(fitted_data_cache, 'w'), indent=4)

    # Ni-Si has fittable PD
    # fitted_data.update(fit_ni_si_system())
    # json.dump(fitted_data, open(fitted_data_cache, 'w'), indent=4)

    # Plot a ternary system
    # elements = ['C', 'Ga', 'Fe', 'Ni', 'Si']
    # for combo in combinations(elements, 3):
    #     system_name = '-'.join(combo)
    #     print(f"Plotting Hi-Res {system_name} system...")
    #     plot_ternary_system(system_name, fitted_data, t_incr=1)
    plot_ternary_system('Ga-Fe-Ni', fitted_data, t_incr=10, delta=0.01)
    # plot_ternary_system('C-Ga-Si', fitted_data, t_incr=1)