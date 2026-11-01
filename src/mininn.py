from miniTorch import miniTensor
import random

class Module:
    def parameters(self):
        return []

    def zero_grad(self):
        for param in self.parameters():
            param.grad = 0.0


class Neuron(Module):
    def __init__(self, nin, non_lin: str | None = None):
        std = (2 / nin) ** 0.5 if non_lin == "relu" else (1 / nin) ** 0.5
        self.w = [miniTensor(random.gauss(0, std)) for _ in range(nin)]
        self.b = miniTensor(0.0)
        self.non_lin = non_lin
    
    def __call__(self, x):
        out = sum((wi*xi for wi, xi in zip(self.w, x)), self.b)
        return getattr(out, self.non_lin)() if self.non_lin else out
    
    def parameters(self):
        return self.w + [self.b]


class Layer(Module):
    def __init__(self, nin, size, activation: str | None):
        self.nin = nin
        self.size = size
        self.activation = activation

        self.layer = []
        
        for _ in range(self.size):
            self.layer.append(Neuron(nin, self.activation))

    def __call__(self, x):
        return [neuron(x) for neuron in self.layer]
        
    def parameters(self):
        return [p for neuron in self.layer for p in neuron.parameters()]


class MLP(Module):
    def __init__(self, nin, nout, layers, hsize, activation: str | None):
        self.nin = nin
        self.nout = nout
        self.layers = layers
        self.hsize = hsize
        self.activation = activation

        self.mlp = []
        
        for _ in range(self.layers):
            self.mlp.append(Layer(self.nin, self.hsize, self.activation))
            self.nin = self.hsize
        
        self.mlp.append(Layer(self.nin, self.nout, None))

    def __call__(self, x):
        for layer in self.mlp:
            x = layer(x)
        return x

    def parameters(self):
        params = []
        for layer in self.mlp:
            params.extend(layer.parameters())
        return params

