import os

import matplotlib.pyplot as plt
import seaborn as sns
from pyhere import here


def set_size(width, aspect_ratio=1.25, fraction=1):
    if width == "generic_paper":
        width_pt = 472.03123
    elif width == "acm_2c":
        width_pt = 541.295
    elif width == "acm_1c":
        width_pt = 241.14749
    elif width == "arkheion":
        width_pt = 453.54
    elif width == "ieee_2c":
        width_pt = 516.0
    elif width == "ieee_1c":
        width_pt = 252.0
    else:
        width_pt = width

    if aspect_ratio == "golden":
        aspect_ratio = 1.61803398875

    # Width of figure (in pts)
    fig_width_pt = width_pt * fraction

    # Convert from pt to inches
    inches_per_pt = 1 / 72.26999

    # Figure width and height in inches
    fig_width_in = fig_width_pt * inches_per_pt
    fig_height_in = fig_width_in / aspect_ratio

    fig_dim = (fig_width_in, fig_height_in)
    return fig_dim

def set_plot_params(reset_sns=True, reset_mpl=True, style_name="pub_ltx"):
    if reset_sns:
        sns.reset_defaults()
    if reset_mpl:
        plt.rcParams.update(plt.rcParamsDefault)
    if style_name is not None:
        mplstyle_path = os.path.join(here(), "src", "bk_buster", "conf", style_name + ".mplstyle")
        plt.style.use(mplstyle_path)