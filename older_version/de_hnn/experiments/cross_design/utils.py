import numpy as np
import math
import random
import pickle
import matplotlib.pyplot as plt

# Metrics
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.metrics import r2_score

def scatter_hist(x, y, ax, ax_histx, ax_histy, title = None):
    # no labels
    ax_histx.tick_params(axis="x", labelbottom=False)
    ax_histy.tick_params(axis="y", labelleft=False)

    # the scatter plot:
    ax.scatter(x, y)

    ax.set_xlabel('Truth')
    ax.set_ylabel('Predict')

    # now determine nice limits by hand:
    binwidth = 4.0
    xymax = max(np.max(np.abs(x)), np.max(np.abs(y)))
    lim = (int(xymax/binwidth) + 1) * binwidth

    bins = np.arange(-lim, lim + binwidth, binwidth)
    ax_histx.hist(x, bins=bins)
    ax_histy.hist(y, bins=bins, orientation='horizontal')

    if title is not None:
        ax_histx.set_title(title)

def plot_figure(truth, predict, method_name, design_name, output_name):
    r2 = r2_score(truth, predict)
    mae = mean_absolute_error(truth, predict)

    title = method_name + ' on ' + design_name + ': MAE = ' + str(round(mae, 2)) + ', R2 = ' + str(round(r2, 2))

    fig = plt.figure(figsize = (6, 6))
    gs = fig.add_gridspec(2, 2, width_ratios = (4, 1), height_ratios = (1, 4),
                    left = 0.1, right = 0.9, bottom = 0.1, top = 0.9,
                    wspace = 0.05, hspace = 0.05)

    ax = fig.add_subplot(gs[1, 0])
    ax_histx = fig.add_subplot(gs[0, 0], sharex = ax)
    ax_histy = fig.add_subplot(gs[1, 1], sharey=ax)

    scatter_hist(truth, predict, ax, ax_histx, ax_histy, title = title)

    plt.savefig(output_name, dpi = 200)
    plt.clf()

