import math

class miniTensor:
    def __init__(self, data: float, _parents: tuple["miniTensor", ...] = ()):
        self.data = data
        self.grad = 0.0
        self._parents = _parents
        self._backward = lambda: None

    def __repr__(self):
        return f"miniTensor({self.data})"

    def __add__(self, other: "miniTensor | float"):
        other = other if isinstance(other, miniTensor) else miniTensor(other)
        out = miniTensor(self.data + other.data, (self, other))

        def _backward():
            self.grad += out.grad
            other.grad += out.grad
        out._backward = _backward

        return out
    
    def __mul__(self, other: "miniTensor | float"):
        other = other if isinstance(other, miniTensor) else miniTensor(other)
        out = miniTensor(self.data * other.data, (self, other))
        
        def _backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad
        out._backward = _backward

        return out

    def __pow__(self, other: float):
        assert isinstance(other, (int, float)), "only int/float powers supported"
        out = miniTensor(self.data ** other, (self,))
        
        def _backward():
            self.grad += other * (self.data ** (other - 1)) * out.grad
        out._backward = _backward
        
        return out

    def __neg__(self):
        return self * -1

    def __sub__(self, other: "miniTensor | float"):
        return self + (-other)

    def __truediv__(self, other: "miniTensor | float"):
        return self * other ** -1

    def __radd__(self, other: float):
        return self + other

    def __rmul__(self, other: float):
        return self * other

    def __rsub__(self, other: float):
        return other + (-self)

    def __rtruediv__(self, other: float):
        return other * self ** -1

    def exp(self):
        out = miniTensor(math.exp(self.data), (self,))
        
        def _backward():
            self.grad += out.data * out.grad
        out._backward = _backward
        
        return out

    def log(self):
        out = miniTensor(math.log(self.data), (self,))
        
        def _backward():
            self.grad += (1 / self.data) * out.grad
        out._backward = _backward
        
        return out

    def relu(self):
        out = miniTensor(max(0.0, self.data), (self,))

        def _backward():
            self.grad += (1 if self.data > 0 else 0) * out.grad
        out._backward = _backward

        return out

    def sigmoid(self):
        if self.data >= 0:
            out = miniTensor(1 / (1 + math.exp(-self.data)), (self,))
        else:
            out = miniTensor(math.exp(self.data) / (1 + math.exp(self.data)), (self,))

        def _backward():
            self.grad += out.data * (1-out.data) * out.grad
        out._backward = _backward

        return out

    def tanh(self):
        out = miniTensor(math.tanh(self.data), (self,))

        def _backward():
            self.grad += (1 - out.data**2) * out.grad
        out._backward = _backward

        return out

    def backward(self):
        # topologically sort the graph and then start propagating gradients
        topo = []
        visited = set()
        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._parents:
                    build_topo(child)
                topo.append(v)
        build_topo(self)
        self.grad = 1.0
        for v in reversed(topo):
            v._backward()

if __name__ == "__main__":
    import torch

    a = miniTensor(2)
    b = miniTensor(3)
    c = a + b
    c.backward()

    ta = torch.tensor(2.0, requires_grad=True)
    tb = torch.tensor(3.0, requires_grad=True)
    tc = ta + tb
    tc.backward()

    print(f"gradients torch: {ta.grad}, {tb.grad}")
    print(f"gradients miniTorch: {a.grad}, {b.grad}")

    # more complex example
    a = miniTensor(2)
    b = miniTensor(3)
    c = a * b
    d = c + a
    e = d * b
    e.backward()

    ta = torch.tensor(2.0, requires_grad=True)
    tb = torch.tensor(3.0, requires_grad=True)
    tc = ta * tb
    td = tc + ta
    te = td * tb
    te.backward()

    print(f"gradients torch: {ta.grad}, {tb.grad}")
    print(f"gradients miniTorch: {a.grad}, {b.grad}")

    # unary ops
    for op, value in [("exp", 0.5), ("log", 0.5), ("relu", 0.5), ("relu", -1.5), ("sigmoid", 0.5), ("tanh", 0.5)]:
        a = miniTensor(value)
        getattr(a, op)().backward()

        ta = torch.tensor(value, requires_grad=True)
        getattr(torch, op)(ta).backward()

        print(f"{op}({value}) gradients torch: {ta.grad}, miniTorch: {a.grad}")
