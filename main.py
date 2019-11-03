import numpy as np

from simulate import generate_data
from train import get_args, train_models
from serialize import load_models, load_simulated_data
from graphics import plot_trajectory, integrate_and_plot, plot_simulated_data


def run_id():
    with open('run-id.txt', 'r') as f:
        id = int(f.readline().strip())
    with open('run-id.txt', 'w') as f:
        f.write(str(id + 1))
    return id


if __name__ == '__main__':
    # Parameters
    N = 500 # Number of Trajectories (different initial states)
    T = 100 # Time steps

    # Constant Parameters
    tf = 20 # Final time
    var = 0.06 # Noise variance
    tauc = 5 # Noise Correlation Time

    # Run ID
    rid = run_id()

    # === SIMULATE ===
    generate_data(T, tf, var, tauc, N, rid)

    plot_simulated_data(data_id=rid, save=True, params=(var, tauc))

    # === TRAIN ===
    print("\nBeginning training.")
    models = train_models(rid, data_id=rid)

    # === LOAD ===
    #data_id = 2
    #models = load_models(data_id, get_args())

    # === PLOT ===
    integrate_and_plot(models, data_id=rid, params=(var, tauc))