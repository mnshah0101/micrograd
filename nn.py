
from __future__ import annotations
import random
from typing import List, Iterable
from engine import Value

class Neuron:

    def __init__(self, nin):
        self.w = [Value(random.uniform(-1,1)) for _ in range(nin)]
        self.b = Value(random.uniform(-1,1))

    def __call__(self, x):
        dot = sum(wi*xi for wi,xi in zip(self.w, x)) + self.b
        out = dot.tanh()

        return out

    def parameters(self):
        return self.w + [self.b]
        

class Layer:
    def __init__(self, nin, nout):
        self.neurons = [Neuron(nin) for _ in range(nout)]

    def __call__(self,x):
        outs = [n(x) for n in self.neurons]
        return outs[0] if len(outs) ==1 else outs


    def parameters(self):
        return [p for neuron in self.neurons for p in neuron.parameters()]
        


class MLP:

    def __init__(self, nin, nouts):
        sz = [nin] + nouts
        self.layers = [Layer(sz[i], sz[i+1]) for i in range(len(nouts))]

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

    def parameters(self):
        return [p for layer in self.layers for p in layer.parameters()]


class Module:
    def parameters(self) -> List[Value]:
        return []

    def zero_grad(self):
        for p in self.parameters():
            p.grad = 0.0

class Neuron(Module):
    def __init__(self, nin: int, nonlin: bool=True):
        self.w = [Value(random.uniform(-1,1)) for _ in range(nin)]
        self.b = Value(0.0)
        self.nonlin = nonlin

    def __call__(self, x: Iterable[Value]) -> Value:
        act = sum((wi*xi for wi, xi in zip(self.w, x)), self.b)
        return act.tanh() if self.nonlin else act

    def parameters(self) -> List[Value]:
        return self.w + [self.b]

class Layer(Module):
    def __init__(self, nin: int, nout: int, **kwargs):
        self.neurons = [Neuron(nin, **kwargs) for _ in range(nout)]

    def __call__(self, x: Iterable[Value]) -> List[Value] | Value:
        outs = [n(x) for n in self.neurons]
        return outs[0] if len(outs)==1 else outs

    def parameters(self) -> List[Value]:
        params = []
        for n in self.neurons:
            params.extend(n.parameters())
        return params

class MLP(Module):
    def __init__(self, nin: int, nouts: List[int]):
        sz = [nin] + list(nouts)
        self.layers = [Layer(sz[i], sz[i+1], nonlin=(i != len(nouts)-1)) for i in range(len(nouts))]

    def __call__(self, x: Iterable[Value]) -> List[Value] | Value:
        for layer in self.layers:
            x = layer(x if isinstance(x, list) else [x] if not isinstance(x, (list, tuple)) else x)
        return x

    def parameters(self) -> List[Value]:
        params = []
        for l in self.layers:
            params.extend(l.parameters())
        return params
