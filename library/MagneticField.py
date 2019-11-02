import numpy as np


def zero(theta):
    return np.zeros(theta.shape) if type(theta) is np.ndarray else 0


def one(theta):
    return np.ones(theta.shape) if type(theta) is np.ndarray else 0


def theta_linear(t, args):
    tf = args['tf']
    return np.pi / 2 * t / tf


def R_y(theta):
    return np.array([[np.cos(theta), zero(theta), np.sin(theta)],
                     [zero(theta), one(theta), zero(theta)],
                     [-np.sin(theta), zero(theta), np.cos(theta)]], dtype=float)


class BField(object):
    
    dimension = 3
    
    def __init__(self, R_fct, theta_fct, init):
        self.R_fct = R_fct
        self.theta_fct = theta_fct
        self.init = init
    
    # Allows to call the BField as a usual function
    def __call__(self, t, args):
        return np.dot(self.R_fct(self.theta_fct(t, args)), self.init)
    
    # Allows to index the BField as a usual vector (of functions)
    def __getitem__(self, key):
        return lambda t, args: self(t, args)[key]
    
    # Necessary method for iterable objects, yields all components
    def __iter__(self):
        for i in range(BField.dimension):
            yield self[i]

    def R_t(self, t, args):
        return self.R_fct(self.theta_fct(t, args))
