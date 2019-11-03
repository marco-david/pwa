import os, sys
import numpy as np
from qutip import *

from joblib import Parallel, delayed
from collections import defaultdict

from library.Solve import Solve
from library.Noise import UhlenbeckNoise
from library.MagneticField import BField, R_y, theta_linear


# === UTILITY FUNCTIONS ===


def cartesian_product(arrays):
    return np.dstack(np.meshgrid(*arrays, indexing='ij')).reshape(-1, len(arrays))


def deriv(t, traj):
    dt = t[1] # t[0] = 0
    output = []
    
    # Do the first and last values manually because they dont follow the equation
    # First value
    x = (traj[1][0] - traj[0][0])/dt
    y = (traj[1][1] - traj[0][1])/dt
    z = (traj[1][2] - traj[0][2])/dt

    output.append([x, y, z])
        
    # Loop
    for i in range(1, len(t) - 1):
        x = (traj[i + 1][0] - traj[i - 1][0])/(2*dt)
        y = (traj[i + 1][1] - traj[i - 1][1])/(2*dt)
        z = (traj[i + 1][2] - traj[i - 1][2])/(2*dt)
        output.append([x, y, z])
        
    # Last Value
    x = (traj[len(traj) - 1][0] - traj[len(traj) - 2][0])/dt
    y = (traj[len(traj) - 1][1] - traj[len(traj) - 2][1])/dt
    z = (traj[len(traj) - 1][2] - traj[len(traj) - 2][2])/dt
    output.append([x, y, z])
    
    return np.array(output)
    

class Run(object):

    def __init__(self, var, tau_c, tf, T_steps, initial_state):
        self.mean = 0
        self.var = var
        self.tau_c = tau_c
        self.tf = tf
        self.T_steps = T_steps
        self.initial_state = initial_state

        self.noise = UhlenbeckNoise(self.mean, self.var, self.tau_c)
        self.B = BField(R_y, theta_linear, np.array([0, 0, 1]))
        self.solve = Solve(self.B, self.noise, self.tf, initial_state=self.initial_state,
                           eigenbasis=False, steps=self.T_steps)


def get_trajectory(params):
    var, tf, tauc, T, state = params
    run = Run(var, tauc, tf, T, initial_state=state).solve
    return run.times, run.expect


def print_progress(i, N):
    sys.stdout.write(f"\rProcessing {i+1} out of {N}.")
    sys.stdout.flush()


def generate_this_data(params, initial_state, i, N):
    T, tf, var, tauc, run_id = params
    print_progress(i, N)

    t, traj = get_trajectory((var, tf, tauc, T, initial_state))
    p = deriv(t, traj)
    statearray = np.concatenate([t[:, np.newaxis], traj, p], axis=1)

    # Save position and momentum values to file
    directory = f'simulation-data/simulation-data-{run_id}'
    os.makedirs(directory) if not os.path.exists(directory) else None
    np.savetxt(directory + f'/data{i}.csv', statearray, delimiter=',')


def generate_data(T, tf, var, tauc, N, run_id):
    # Create coordinates for the initial states
    theta = np.linspace(0, np.pi / 2, N)
    phi = np.linspace(0, 20 * np.pi, N)

    initial_state = lambda i: np.cos(theta[i] / 2) * basis(2, 0) + np.exp(1j * phi[i]) * np.sin(theta[i] / 2) * basis(2, 1)

    params = (T, tf, var, tauc, run_id)
    Parallel(n_jobs=-1)(delayed(generate_this_data)(params, initial_state(i), i, N) for i in range(N))
