import numpy as np

class Optimizer:
    def update(self, param, param_grad):
        raise NotImplementedError

class SGDMomentum(Optimizer):
    def __init__(self, args):
        self.args = args
        self.learning_rate = self.args.momentum_lr
        self.momentum = self.args.momentum
        self.velocity = {}

    def update(self, param, param_grad):
        if id(param) not in self.velocity:
            self.velocity[id(param)] = np.zeros_like(param)
        v = self.velocity[id(param)]
        v[:] = self.momentum * v - self.learning_rate * param_grad
        param += v

class AdaGrad(Optimizer):
    def __init__(self, args):
        self.args = args
        self.learning_rate = self.args.momentum_lr
        self.cache = {}

    def update(self, param, param_grad):
        if id(param) not in self.cache:
            self.cache[id(param)] = np.zeros_like(param)
        cache = self.cache[id(param)]
        cache[:] += param_grad ** 2
        param -= self.learning_rate * param_grad / (np.sqrt(cache) + 1e-7)

class Adam(Optimizer):
    def __init__(self, args):
        self.args = args
        self.learning_rate = self.args.momentum_lr
        self.beta1 = self.args.beta1
        self.beta2 = self.args.beta2
        self.m = {}
        self.v = {}
        self.t = 0

    def update(self, param, param_grad):
        if id(param) not in self.m:
            self.m[id(param)] = np.zeros_like(param)
            self.v[id(param)] = np.zeros_like(param)
        self.t += 1
        m = self.m[id(param)]
        v = self.v[id(param)]
        m[:] = self.beta1 * m + (1 - self.beta1) * param_grad
        v[:] = self.beta2 * v + (1 - self.beta2) * (param_grad ** 2)
        m_hat = m / (1 - self.beta1 ** self.t)
        v_hat = v / (1 - self.beta2 ** self.t)
        param -= self.learning_rate * m_hat / (np.sqrt(v_hat) + 1e-8)