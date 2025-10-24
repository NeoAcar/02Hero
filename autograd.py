import numpy as np

class Tensor:

    def __init__(self, value, _children=(), _op='', dtype=np.float32, requires_grad=True):
        self.value = np.array(value, dtype=dtype)
        self.grad = np.zeros_like(self.value, dtype=self.value.dtype)
        self._backward = lambda: None
        self._prev = set(_children)
        self._op = _op
        self.dtype = self.value.dtype
        self.requires_grad = requires_grad

    def __repr__(self):
        return f"Tensor({self.value})"
    
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
    

    def __add__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(np.array(other), dtype=self.dtype, requires_grad=False)
        out = Tensor(self.value + other.value, (self, other), '+')

        def _add_backward():
            self.grad += out.grad
            other.grad += out.grad
        out._backward = _add_backward
        return out

    def __mul__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(np.array(other, dtype=self.dtype), requires_grad=False)
        out = Tensor(self.value * other.value, (self, other), '*')

        def _mul_backward():
            self.grad += other.value * out.grad
            other.grad += self.value * out.grad
        out._backward = _mul_backward
        return out
        
        
    def __pow__(self, other):
        assert isinstance(other, (int, float)), "only supporting int/float powers for now"
        out = Tensor(self.value**other, (self,), '**')

        def _pow_backward():
            self.grad += (other * self.value**(other - 1)) * out.grad
            #other.grad += self.value**other * np.log(self.value) * out.grad
        out._backward = _pow_backward
        return out
    

    def exp(self):
        out = Tensor(np.exp(self.value), (self,), 'exp')
        
        def _exp_backward():
            self.grad += out.value * out.grad
        out._backward = _exp_backward
        return out
    

    def __matmul__(self,other):
        other = other if isinstance(other, Tensor) else Tensor(np.array(other), dtype=self.dtype, requires_grad=False)
        out = Tensor(self.value @ other.value, (self, other), "@")

        def _matmul_backward():
            self.grad += out.grad @ other.value.T
            other.grad += self.value.T @ out.grad
        out._backward = _matmul_backward
        return out


    @property
    def T(self):
        out = Tensor(self.value.T, (self,), 'T')

        def _T_backward():
            self.grad += out.grad.T
        out._backward = _T_backward
        return out
    

    
    def __radd__(self, other):
        return self + other
    
    def __neg__(self):
        return self * -1
    
    def __sub__(self, other):
        return self + (-other)
    
    def __rsub__(self, other):
        return other + (-self)
    
    def __rmul__(self, other):
        return self * other
    
    def __truediv__(self, other):
        return self * other**-1
    
    def __rtruediv__(self, other):
        return other * self**-1
    
    def __rmatmul__(self, other):
        return other @ self
    
    def relu(self):
        out = Tensor(np.maximum(0, self.value), (self,), 'ReLU')

        def _relu_backward():
            self.grad += (out.value > 0) * out.grad
        out._backward = _relu_backward
        return out
    
    def sigmoid(self):
        sig = 1 / (1 + np.exp(-self.value))
        out = Tensor(sig, (self,), 'sigmoid')

        def _sigmoid_backward():
            self.grad += sig * (1 - sig) * out.grad
        out._backward = _sigmoid_backward
        return out
    
    def tanh(self):
        t = np.tanh(self.value)
        out = Tensor(t, (self,), 'tanh')

        def _tanh_backward():
            self.grad += (1 - t**2) * out.grad
        out._backward = _tanh_backward
        return out
    
    # def __rpow__(self, other):
    #     # assert if base negative
    #     assert np.all(other > 0), "base must be positive for real-valued exponentiation"
    #     base = other if isinstance(other, Tensor) else Tensor(np.array(other, dtype=self.dtype))
    #     out = Tensor(base.value ** self.value, (base, self), '**')

    #     def _rpow_backward():
    #         # d/dself: base**self * log(base); d/dbase: self * base**(self - 1)
    #         self.grad += out.grad * out.value * np.log(base.value)
    #         base.grad += out.grad * self.value * base.value ** (self.value - 1)
    #     out._backward = _rpow_backward
    #     return out
    
    # def __mul__(self, other):
    #     other = other if isinstance(other, Tensor) else Tensor(np.array(other))
    #     out = Tensor(self.value * other.value, (self, other), '*')

    #     def _mul_backward():
    #         self.grad += other.value * out.grad
    #         other.grad += self.value * out.grad
    #     out._backward = _mul_backward
    #     return out


    # def __add__(self, other):
    #     other = other if  isinstance(other, Tensor) else Tensor(np.array(other))
    #     out = Tensor(self.value + other.value, (self, other), '+')

    #     def _add_backward():
    #         self.grad += out.grad
    #         other.grad += out.grad
    #     out._backward = _add_backward
    #     return out





