from scripts.old.ternary_HSX_jw import ternary_gtx_plotter
import plotly.offline as ploff
from gliquid.config import data_dir
import os


def plot_BiCdSn_system():
    # Bi-Cd-Sn system
    os.environ["NEW_MP_API_KEY"] = "YOUR API KEY HERE"
    tern_sys = ["Bi", "Cd", "Sn"]
    binary_L_dict = {"Bi-Cd": [-685, 4.41, 837, -2.96],
                     "Cd-Sn": [10649, -10.13, -7219, 15.72],
                     "Sn-Bi": [-1579, 8.06, 2295, -2.1]}
    
    plotter = ternary_gtx_plotter(tern_sys, data_dir, interp_type="linear", param_format="linear",
                                  L_dict=binary_L_dict, temp_slider=[0, -300], T_incr=1, delta=0.01)
    plotter.interpolate()
    plotter.process_data()
    tern_fig = plotter.plot_ternary()
    ploff.plot(tern_fig, filename='BiCdSn_system.html', auto_open=True)


if __name__ == "__main__":
    plot_BiCdSn_system()