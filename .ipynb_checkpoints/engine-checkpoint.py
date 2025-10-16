from __future__ import annotations
import math
import random
from typing import Iterable, Set, Callable, Optional


from graphviz import Digraph

def trace(root):
    nodes, edges = set(), set()
    def build(v):
        if v not in nodes:
            nodes.add(v)
            for child in v._prev:
                edges.add((child, v))
                build(child)
    build(root)
    return nodes, edges

def draw_dot(root):
    dot = Digraph(format='svg', graph_attr={'rankdir': 'LR'})  
    nodes, edges = trace(root)

    for n in nodes:
        uid = str(id(n))
        dot.node(
            name=uid,
            label="{ %s | data %.4f | grad %.4f }" % (getattr(n, "label", ""), n.data, n.grad),
            shape='record'
        )
        if n._op:
            dot.node(name=uid + n._op, label=n._op)
            dot.edge(uid + n._op, uid)

    for n1, n2 in edges:
        dot.edge(str(id(n1)), str(id(n2)) + n2._op)

    return dot

class Value:
    def __init__(self, data, _children=(), _op='', label=''):
        self.data = float(data)
        self._prev = set(_children)
        self._op = _op
        self.label = label
        self.grad = 0.0
        self._backward = lambda: None

    def __repr__(self):
        return f"Value(data={self.data}, grad={self.grad})"

    @staticmethod
    def _promote(x):
        return x if isinstance(x, Value) else Value(x)

    def __add__(self, other):
        other = Value._promote(other)
        out = Value(self.data + other.data, (self, other), '+')
        def _backward():
            self.grad += 1.0 * out.grad
            other.grad += 1.0 * out.grad
        out._backward = _backward
        return out
    def __radd__(self, other): 
        return self + other

    def __sub__(self, other):
        return self + (-other)
    def __rsub__(self, other):
        return Value._promote(other) - self

    def __neg__(self):
        return self * -1.0

    def __mul__(self, other):
        other = Value._promote(other)
        out = Value(self.data * other.data, (self, other), '*')
        def _backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad
        out._backward = _backward
        return out
    def __rmul__(self, other): 
        return self * other

    def __truediv__(self, other):
        other = Value._promote(other)
        return self * (other ** -1)
    def __rtruediv__(self, other):
        return Value._promote(other) / self

    def __pow__(self, p): 
        assert isinstance(p, (int, float)), "only scalar powers supported"
        out = Value(self.data ** p, (self,), f'**{p}')
        def _backward():
            self.grad += (p * (self.data ** (p - 1))) * out.grad
        out._backward = _backward
        return out

    def tanh(self):
        t = math.tanh(self.data)
        out = Value(t, (self,), 'tanh')
        def _backward():
            self.grad += (1 - t * t) * out.grad
        out._backward = _backward
        return out

    def exp(self):
        e = math.exp(self.data)
        out = Value(e, (self,), 'exp')
        def _backward():
            self.grad += e * out.grad
        out._backward = _backward
        return out

    def backward(self):
        topo, visited = [], set()
        def build(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build(child)
                topo.append(v)
        build(self)
        self.grad = 1.0
        for v in reversed(topo):
            v._backward()


def main():
    
    x1 = Value(2.0, label = 'x1')
    x2 = Value(0.0, label = 'x2')
    
    w1 = Value(-3.0, label = 'w1')
    w2 = Value(1.0, label = 'w1')
    
    x1w1 = x1 * w1; x1w1.label = 'x1w1'
    x2w2 = x2 * w2; x2w2.label = 'x2w2'
    b = Value(6.88, label='b')
    
    
    x1w1x2w2 = x1w1 + x2w2; x1w1x2w2.label ='x1w1x2w2'
    n = x1w1x2w2 + b; n.label='n'
    o = n.tanh()
    o.grad = 1.0
    draw_dot(o)
    
    
    o.backward()
    #lets create a neuron
    
    
    x1 = Value(2.0, label = 'x1')
    x2 = Value(0.0, label = 'x2')
    
    w1 = Value(-3.0, label = 'w1')
    w2 = Value(1.0, label = 'w1')
    
    x1w1 = x1 * w1; x1w1.label = 'x1w1'
    x2w2 = x2 * w2; x2w2.label = 'x2w2'
    b = Value(6.88, label='b')
    
    
    x1w1x2w2 = x1w1 + x2w2; x1w1x2w2.label ='x1w1x2w2'
    n = x1w1x2w2 + b; n.label='n'
    e = (2*n).exp()
    o = (e-1) / (e+1)
    o.grad = 1.0
    draw_dot(o)
    
    o.backward()
    draw_dot(o)
    
    for i in range(100):
        ypred = [n(x) for x in xs]
        loss = sum((yout-ygt)**2 for ygt, yout in zip(ys, ypred))
        for p in n.parameters():
            p.grad = 0
        loss.backward()
        for p in n.parameters():
            p.data -= 0.05 * p.grad
    
__all__ = ['Value']
