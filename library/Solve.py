import numpy as np
from qutip import *


class Solve(object):

    sigma = [sigmax(), sigmay(), sigmaz()]
    steps_default = 100

    def __init__(self, B, noise, tf, steps=steps_default, initial_state=basis(2, 0), observables=sigma, eigenbasis=True):
        self.B = B
        self.noise = noise
        self.H = [[1/2 * Solve.sigma[i], B[i]] for i in {0, 1, 2}] + [[1/2 * sigmax(), noise]]
        self.psi = initial_state

        self.tf = tf
        self.steps = steps
        self.observables = observables

        try:
            self.result = sesolve(self.H, self.psi, np.linspace(0, self.tf, self.steps),
                                  self.observables, args={'tf': tf})
        except TypeError:
            # Qutip only allows FunctionType objects, we're passing a general object with __call__ defined
            raise SystemError("Need modified Qutip version with two extra lines in rhs_generate.py."
                              + "See Qutip Pull Request #1107 on GitHub. ")

        self.times = self.result.times
        self.expect = np.squeeze(self.result.expect).T

        if eigenbasis:
            rot_matrices = self.B.R_t(-self.times, {'tf': tf})
            self.expect = np.einsum('ijt,tj->ti', rot_matrices, self.expect)

    def __getitem__(self, t):
        return self.expect[t]

    def __iter__(self):
        return iter(self.expect)

    def errors(self, t=None):
        if t:
            return 1/2 * (1 - self.expect[t, 2])
        else:
            return 1/2 * (1 - self.expect[:, 2])
