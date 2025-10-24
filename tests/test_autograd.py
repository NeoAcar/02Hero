import numpy as np
import pytest
import sys
import os
import torch
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from autograd import Tensor


def reduce_to_scalar_autograd(t: Tensor) -> Tensor:
    """Reduce a Tensor (ndim <= 2) to a scalar using existing ops."""
    if t.value.ndim == 0:
        return t
    if t.value.ndim == 1:
        weights = Tensor(np.ones_like(t.value), dtype=t.dtype, requires_grad=False)
        return weights @ t
    if t.value.ndim == 2:
        left = Tensor(
            np.ones((1, t.value.shape[0]), dtype=t.dtype),
            dtype=t.dtype,
            requires_grad=False,
        )
        right = Tensor(
            np.ones((t.value.shape[1], 1), dtype=t.dtype),
            dtype=t.dtype,
            requires_grad=False,
        )
        return (left @ t) @ right
    raise NotImplementedError("Only tensors up to 2 dimensions are supported.")


def reduce_to_scalar_torch(t: torch.Tensor) -> torch.Tensor:
    if t.dim() == 0:
        return t
    if t.dim() == 1:
        weights = torch.ones_like(t)
        return weights @ t
    if t.dim() == 2:
        left = torch.ones((1, t.shape[0]), dtype=t.dtype, device=t.device)
        right = torch.ones((t.shape[1], 1), dtype=t.dtype, device=t.device)
        return (left @ t) @ right
    raise NotImplementedError("Only tensors up to 2 dimensions are supported.")


def compare_with_torch(builder, inputs):
    tensors = [Tensor(arr, dtype=np.float32) for arr in inputs]
    torch_tensors = [
        torch.tensor(arr, dtype=torch.float32, requires_grad=True) for arr in inputs
    ]

    out_custom = builder(*tensors)
    out_torch = builder(*torch_tensors)

    loss_custom = reduce_to_scalar_autograd(out_custom)
    loss_torch = reduce_to_scalar_torch(out_torch)

    loss_custom.backward()
    loss_torch.backward()

    custom_value = np.asarray(loss_custom.value).reshape(-1)
    torch_value = loss_torch.detach().cpu().numpy().reshape(-1)
    np.testing.assert_allclose(custom_value, torch_value, rtol=1e-5, atol=1e-6)

    for custom_tensor, torch_tensor in zip(tensors, torch_tensors):
        torch_grad = torch_tensor.grad.detach().cpu().numpy()
        np.testing.assert_allclose(
            custom_tensor.grad, torch_grad, rtol=1e-4, atol=1e-5
        )


def test_elementwise_chain_matches_torch():
    rng = np.random.default_rng(0)
    data = rng.standard_normal(8).astype(np.float32)

    def builder(x):
        return ((x + 1.5) * (x - 0.75) / 3.2) + 2.5 ** (x / 4.0)

    compare_with_torch(builder, [data])


def test_activation_chain_matches_torch():
    rng = np.random.default_rng(1)
    data = (rng.standard_normal(6) * 2.0).astype(np.float32)

    def builder(x):
        return x.relu() * x.sigmoid() + x.tanh() * x.exp()

    compare_with_torch(builder, [data])


def test_matmul_chain_matches_torch():
    rng = np.random.default_rng(2)
    a_data = rng.uniform(-1.0, 1.0, size=(2, 3)).astype(np.float32)
    b_data = rng.uniform(-1.0, 1.0, size=(2, 3)).astype(np.float32)

    def builder(a, b):
        product = a @ b.T
        return product * product.T + (a @ a.T)

    compare_with_torch(builder, [a_data, b_data])
