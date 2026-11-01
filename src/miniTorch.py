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


if __name__ == "__main__":
    import torch

    a = miniTensor(2)
    b = miniTensor(3)
    c = a + b
    c.grad = 1.0
    c._backward()

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
    e.grad = 1.0
    e._backward()
    d._backward()
    c._backward()

    ta = torch.tensor(2.0, requires_grad=True)
    tb = torch.tensor(3.0, requires_grad=True)
    tc = ta * tb
    td = tc + ta
    te = td * tb
    te.backward()

    print(f"gradients torch: {ta.grad}, {tb.grad}")
    print(f"gradients miniTorch: {a.grad}, {b.grad}")
