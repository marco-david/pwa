import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime


def plot_trajectory(t, y_vec, params=None,save=None, show=False):
    labels = ['x', 'y', 'z']
    for i in range(3):
        y_component = np.array(y_vec)[:, i]
        plt.plot(t, y_component, label=labels[i])

    plt.legend()
    plt.xlabel("Time $t$")
    plt.ylabel(r"Expectation values $\langle\sigma_\alpha\rangle$")
    if params:
        plt.title(f"$|B| = 1$, $\langle\eta^2\\rangle = {params[0]}$, $\\tau_c = {params[1]}$")

    if save:
        plt.savefig(f'network-results/model-output-{datetime.now().strftime("%Y-%m-%d-%H:%M")}.pdf')
    if show:
        plt.show()