import torch

@torch.library.custom_op("custom_op::my_addmm", mutates_args=())
def my_addmm(b: torch.Tensor, x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    return b + (x @ w)


@my_addmm.register_fake
def _(b: torch.Tensor, x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    M, N = x.shape[0], w.shape[1]
    return torch.empty((M, N), dtype=x.dtype, device=x.device)


def test_custom_addmm(dtype=torch.float16, device="cuda"):
    M, N, K = 32, 64, 128
    b = torch.rand(N, dtype=dtype, device=device)
    x = torch.rand((M, K), dtype=dtype, device=device)
    w = torch.rand((K, N), dtype=dtype, device=device)

    output = torch.ops.custom_op.my_addmm(b, x, w)
    expected = torch.ops.aten.addmm.default(b, x, w)
    torch.testing.assert_close(output, expected)


if __name__ == "__main__":
    test_custom_addmm()
