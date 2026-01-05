from scripts.old.ternary_HSX_jw import ternary_hsx_plotter, ternary_gtx_plotter
import plotly.offline as ploff
import numpy as np
import os


dir = "matrix_data_jsons/"
interp_type_list = ["linear", "muggianu"]
sweep_html_dir = "sweep_htmls/Gliq_trial/"

if not os.path.exists(sweep_html_dir):
    os.makedirs(sweep_html_dir)

spec_eles = ["Al", "Cu", "Cd", "Sn", "Bi", "Mn", "Pt", "Ni", "Mo", "Cr", "W"]

# create a list of all unique ternary systems in spec_eles\
tern_sys_list = []
for i in range(len(spec_eles)):
    for j in range(i+1, len(spec_eles)):
        for k in range(j+1, len(spec_eles)):
            sys = [spec_eles[i], spec_eles[j], spec_eles[k]]
            sys = sorted(sys)
            if sys not in tern_sys_list:
                tern_sys_list.append(sys)

print(tern_sys_list)

skipped_systems = []

tern_sys_list = tern_sys_list[:10]

print(tern_sys_list)

for tern_sys in tern_sys_list:
    tern_sys_str = "_".join(tern_sys)
    for interp_type in interp_type_list:
        gtx_plotter = ternary_gtx_plotter(tern_sys, dir, interp_type, T_incr=5, delta=0.01)
        gtx_plotter.interpolate()

        nancheck_count = 0
        for val in gtx_plotter.L_dict.values():
            if np.any(np.isnan(val)):
                nancheck_count += 1
        if nancheck_count > 0:
            print(f"Skipping system {tern_sys} due to missing params")
            skipped_systems.append(tern_sys)
            continue

        gtx_plotter.process_data()

        gtx_tern_fig = gtx_plotter.plot_ternary()

        ploff.plot(gtx_tern_fig, filename=sweep_html_dir + tern_sys_str + '_gtx_' + interp_type, auto_open=False)


with open(sweep_html_dir + 'skipped_systems.txt', 'w') as f:
    for sys in skipped_systems:
        f.write(str(sys) + '\n')