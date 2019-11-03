import numpy as np
import matplotlib.pyplot as plt
from joblib import Parallel, delayed

from library.Solve import Solve
from library.Noise import UhlenbeckNoise
from library.MagneticField import BField, R_y, theta_linear


def theta_periodic(t, args):
    tf = args['tf']
    return 4 * np.pi * t / tf


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
                           eigenbasis=True, steps=self.T_steps)


def get_trajectory(params):
    var, tf, tauc, T = params
    run = Run(var, tauc, tf, T).solve
    return run.times, run.expect


#def run_parallel(params, M=1):
#    results = []
#    for _ in range(M):
#        results.append(get_trajectory(params))
#    results = np.array(results)
#    return np.array([results.mean(), results.std()])


if __name__ == '__main__':
    # Run ID
    N = 2

    # Time steps
    T = 1000

    # Sample N values in each dimension
    var = 0 # = 1/100 B
    tf = 20
    tauc = 5

    # MAIN CALL
    t, traj = get_trajectory((var, tf, tauc, T))
    p = timeDeriv(t, traj)
    statearray = np.concatenate([t[:, np.newaxis], traj, p], axis=1)

    # Save position and momentum values to file
    np.savetxt(f"simulation-results/data{N}.csv", statearray, delimiter=",")

    labels = ['x', 'y', 'z']
    for i in range(3):
        plt.plot(t, traj[:, i], label=labels[i])

    plt.legend()
    plt.xlabel("Time $t$")
    plt.ylabel(r"Expectation values $\langle\sigma_\alpha\rangle$")
    plt.title(f"$B = 1$, $\langle\eta^2\\rangle = {var}$, $\\tau_c = {tauc}$")
    plt.savefig(f'simulation-results/traj-plane{N}.pdf')

    #errors = np.array(e_list)
    #means = np.squeeze(errors.mean(axis=0))
    #stds = np.squeeze(errors.std(axis=0))

    #plt.errorbar(variances, means, yerr=stds, fmt='d')
    #plt.axhline(analytic_error(tf), xmin=0.025, xmax=0.975)

    #plt.yscale('log')
    #plt.show()
    #plt.savefig(f'errors-var-tf--{M}x{N}.pdf')
    #plt.savefig(f'errors-var({vstart},{vend})-{M}x{N}-tf{tf}-tauc{tauc}.pdf')


