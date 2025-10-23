import numpy as np


class Matrix:

    def __init__(self, value, _children=(), _op='', dtype=np.float32):
        self.value = np.array(value, dtype=dtype)
        self.grad = np.zeros_like(self.value, dtype=self.value.dtype)
        self._backward = lambda: None
        self._prev = set(_children)
        self._op = _op
        self.dtype = self.value.dtype
    ...
    @staticmethod
    def _sum_to_shape(g, shape):
        """Broadcast'li grad'ı hedef shape'e indirger."""
        if shape == ():  # skaler hedef
            return np.array(g.sum(), dtype=g.dtype)
        out = g
        # Fazla öndeki eksenleri indir
        while out.ndim > len(shape):
            out = out.sum(axis=0)
        # Hedef ekseni 1 olan yerlerde topla (keepdims=True)
        for i, s in enumerate(shape):
            if s == 1 and out.shape[i] != 1:
                out = out.sum(axis=i, keepdims=True)
        return out

    def __add__(self, other):
        other = other if isinstance(other, Matrix) else Matrix(np.array(other, dtype=self.dtype))
        out = Matrix(self.value + other.value, (self, other), '+')

        def _add_backward():
            self.grad += Matrix._sum_to_shape(out.grad, self.value.shape)
            other.grad += Matrix._sum_to_shape(out.grad, other.value.shape)
        out._backward = _add_backward
        return out

    def __mul__(self, other):
        other = other if isinstance(other, Matrix) else Matrix(np.array(other, dtype=self.dtype))
        out = Matrix(self.value * other.value, (self, other), '*')

        def _mul_backward():
            self.grad += Matrix._sum_to_shape(other.value * out.grad, self.value.shape)
            other.grad += Matrix._sum_to_shape(self.value * out.grad, other.value.shape)
        out._backward = _mul_backward
        return out
        

    def __repr__(self):
        return f"Matrix({self.value})"
    
    # def __add__(self, other):
    #     other = other if  isinstance(other, Matrix) else Matrix(np.array(other))
    #     out = Matrix(self.value + other.value, (self, other), '+')

    #     def _add_backward():
    #         self.grad += out.grad
    #         other.grad += out.grad
    #     out._backward = _add_backward
    #     return out
    
    def __radd__(self, other):
        return self + other
    
    def __neg__(self):
        return self * -1
    
    def __sub__(self, other):
        return self + (-other)
    
    def __rsub__(self, other):
        return other + (-self)
    
    # def __mul__(self, other):
    #     other = other if isinstance(other, Matrix) else Matrix(np.array(other))
    #     out = Matrix(self.value * other.value, (self, other), '*')

    #     def _mul_backward():
    #         self.grad += other.value * out.grad
    #         other.grad += self.value * out.grad
    #     out._backward = _mul_backward
    #     return out
    
    def __rmul__(self, other):
        return self * other
    
    def __truediv__(self, other):
        return self * other**-1
    
    def __rtruediv__(self, other):
        return other * self**-1
    
    def __pow__(self, other):
        assert isinstance(other, (int, float)), "only supporting int/float powers for now"
        out = Matrix(self.value**other, (self,), '**')

        def _pow_backward():
            self.grad += (other * self.value**(other - 1)) * out.grad
            #other.grad += self.value**other * np.log(self.value) * out.grad
        out._backward = _pow_backward
        return out
    
    def __rpow__(self, other):
        return other**self
    

    def __matmul__(self,other):
        other = other if isinstance(other, Matrix) else Matrix(np.array(other))
        out = Matrix(self.value @ other.value, (self, other), "@")

        def _matmul_backward():
            self.grad += out.grad @ other.value.T
            other.grad += self.value.T @ out.grad
        out._backward = _matmul_backward
        return out
    
    def __rmatmul__(self, other):
        return other @ self
        
    @property
    def T(self):
        out = Matrix(self.value.T, (self,), 'T')

        def _T_backward():
            self.grad += out.grad.T
        out._backward = _T_backward
        return out
    
    def backward(self):
        topo = []
        visited = set()

        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)
        build_topo(self)
        self.grad = np.ones_like(self.value)
        for v in reversed(topo):
            v._backward()

        def zero_grad(self):
            self.grad = np.zeros_like(self.value)


def main():
    a = Matrix([[1,2,3],[4,5,6]], dtype=np.float32)
    b = Matrix([[7,8,9],[10,11,12]], dtype=np.float32)
    c = a + b
    d = a * b
    e = a @ b.T
    f = (a + b) * (a - b) / 2
    g = ((a + b) * (a - b) / 2) ** 2
    h = g.T
    i = h + 10
    j = i / 2
    k = j - 5
    k.backward()
    
    print("a:\n", a.value)
    print("b:\n", b.value)
    print("c (a + b):\n", c.value)
    print("d (a * b):\n", d.value)
    print("e (a @ b.T):\n", e.value)
    print("f ((a + b) * (a - b) / 2):\n", f.value)
    print("g (((a + b) * (a - b) / 2) ** 2):\n", g.value)
    print("h (g.T):\n", h.value)
    print("i (h + 10):\n", i.value)
    print("j (i / 2):\n", j.value)
    print("k (j - 5):\n", k.value)
    
    print("\ngradients:")
    print("dk/da:\n", a.grad)
    print("dk/db:\n", b.grad)
        




if __name__ == "__main__":
    main()
