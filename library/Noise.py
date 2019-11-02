import numpy as np


class UhlenbeckNoise(object):
    
    steps_default = 500
    
    def __init__(self, mean, var, tau_c, steps=steps_default):
        self.mean = mean
        self.var = var
        self.tau_c = tau_c
        self.f = np.exp(- 1 / self.tau_c)
        
        self.steps = steps
        
        # Generate Noise using Desermo paper
        self.eta = [np.random.normal(self.mean, self.var)]
        for i in range(0, self.steps):
            g = np.random.normal(self.mean, self.var)  # Random Gaussian
            next_value = self.f * self.eta[-1] + np.sqrt(1 - self.f**2) * g
            self.eta.append(next_value)
    
    # Allow to call objects as a usual noise function
    def __call__(self, t, args):
        tf = args['tf']
        index = int(self.steps * t / tf)
        return self.eta[index] if index < len(self.eta) else 0
