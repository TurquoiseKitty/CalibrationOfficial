from src.plot_utils import plot_xy_specifyBound
import numpy as np
from scipy import stats
from src.DEFAULTS import normalZ, Upper_quant, Lower_quant, DEFAULT_mean_func
import matplotlib.pyplot as plt


def linear_bandPlot():

    x = np.linspace(0,15,1000)

    y_pred = DEFAULT_mean_func(x)

    y1 = DEFAULT_mean_func(x) + 1 * np.random.randn(1000)
    y_UP1 = y_pred + 1 * normalZ.ppf(Upper_quant)
    y_LO1 = y_pred + 1 * normalZ.ppf(Lower_quant)


    y2 = DEFAULT_mean_func(x) + 2 * np.random.randn(1000)   
    y_UP2 = y_pred + 2 * normalZ.ppf(Upper_quant)
    y_LO2 = y_pred + 2 * normalZ.ppf(Lower_quant)





    fig, (ax1, ax2) = plt.subplots(1, 2)

    plot_xy_specifyBound(
        y_pred = y_pred,

        y_UP = y_UP1,
        y_LO = y_LO1,

        y_true = y1,
        x = x,
        n_subset = 300,

        ylims = [-10, 10],
        xlims = [0, 15],

        ax = ax1,
        title = "Confidence Band, sigma = 1"
    )


    plot_xy_specifyBound(
        y_pred = y_pred,

        y_UP = y_UP2,
        y_LO = y_LO2,

        y_true = y2,
        x = x,
        n_subset = 300,

        ylims = [-10, 10],
        xlims = [0, 15],

        ax = ax2,
        title = "Confidence Band, sigma = 2"
    )


    plt.show(block=True)



if __name__ == "__main__":

    linear_bandPlot()

