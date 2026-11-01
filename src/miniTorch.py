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
        return miniTensor(self.data + other.data, (self, other))

    def __mul__(self, other: "miniTensor | float"):
        other = other if isinstance(other, miniTensor) else miniTensor(other)
        return miniTensor(self.data * other.data, (self, other))

    def __pow__(self, other: float):
        assert isinstance(other, (int, float)), "only int/float powers supported"
        return miniTensor(self.data ** other, (self,))

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
