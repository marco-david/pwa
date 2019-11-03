import numpy as np
from qutip import *
import matplotlib.pyplot as plt
from joblib import Parallel, delayed

from library.Solve import Solve
from library.Noise import UhlenbeckNoise
from library.MagneticField import BField, R_y, theta_linear


def cartesian_product(arrays):
    return np.dstack(np.meshgrid(*arrays, indexing='ij')).reshape(-1, len(arrays))


def timeDeriv(t, traj):
    timeDifference = t[1]
    output = []   
    
    # Do the first and last values manually because they dont follow the equation
    # First value
    x = (traj[1][0] - traj[0][0])/timeDifference
    y = (traj[1][1] - traj[0][1])/timeDifference
    z = (traj[1][2] - traj[0][2])/timeDifference

    output.append([x, y, z])
        
    # Loop
    for i in range(1, len(t) - 1):
        x = (traj[i + 1][0] - traj[i - 1][0])/(2*timeDifference)
        y = (traj[i + 1][1] - traj[i - 1][1])/(2*timeDifference)
        z = (traj[i + 1][2] - traj[i - 1][2])/(2*timeDifference)
        output.append([x, y, z])
        
    # Last Value
    x = (traj[len(traj) - 1][0] - traj[len(traj) - 2][0])/timeDifference
    y = (traj[len(traj) - 1][1] - traj[len(traj) - 2][1])/timeDifference
    z = (traj[len(traj) - 1][2] - traj[len(traj) - 2][2])/timeDifference
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


def generate_data(T, tf, var, tauc, N):
    # Create coordinates for the initial states
    theta = np.linspace(0, np.pi / 2, N)
    phi = np.linspace(0, 20 * np.pi, N)

    # MAIN CALL
    for i in range(N):
        initial_state = np.cos(theta[i] / 2) * basis(2, 0) + np.exp(1j * phi[i]) * np.sin(theta[i] / 2) * basis(2, 1)
        t, traj = get_trajectory((var, tf, tauc, T, initial_state))
        p = timeDeriv(t, traj)
        statearray = np.concatenate([t[:, np.newaxis], traj, p], axis=1)

        # Save position and momentum values to file
        np.savetxt(f"simulation-results/data{i}.csv", statearray, delimiter=",")


if __name__ == '__main__':
    # Parameters
    T = 100 # Time steps
    tf = 20 # Final time
    var = 0 # Noise variance
    tauc = 5 # Noise Correlation Time
    N = 500 # Number of Trajectories (different initial states)

    # Generate Data
    #generate_data(T, tf, var, tauc, N)

    # Plot Data
    data = np.loadtxt(f'simulation-results/data0.csv', delimiter=",") # theta = phi = 0, i.e. [0, 0, 1]
    t = data[:, 0]
    traj = data[:, 1:4]

    labels = ['x', 'y', 'z']
    for i in range(3):
        plt.plot(t, traj[:, i], label=labels[i])

    plt.legend()
    plt.xlabel("Time $t$")
    plt.ylabel(r"Expectation values $\langle\sigma_\alpha\rangle$")
    plt.title(f"$B = 1$, $\langle\eta^2\\rangle = {var}$, $\\tau_c = {tauc}$")
    plt.savefig(f'simulation-results/traj-plane{N}.pdf')



