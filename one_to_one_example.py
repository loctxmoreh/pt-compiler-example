import torch
import torch.nn as nn
import torch._dynamo
from torch._functorch.aot_autograd import aot_module_simplified


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


test_custom_addmm()


class MLP(nn.Module):
  def __init__(self):
    super().__init__()
    self.fc1 = nn.Linear(32, 64)

  def forward(self, x):
    x = self.fc1(x)
    x = torch.nn.functional.gelu(x)
    return x


model = MLP()
batch_size = 8
input_ = torch.randn(batch_size, 32)


def op_replace_backend(gm, sample_inputs):
    def _compiler(gm, sample_inputs):
        for node in gm.graph.nodes:
            if node.op == "call_function" and node.target == torch.ops.aten.addmm.default:
                node.target = torch.ops.custom_op.my_addmm

        return gm.forward

    return aot_module_simplified(
        gm, sample_inputs,
        fw_compiler=_compiler,
    )

expected = model(input_)

torch._dynamo.reset()
fn = torch.compile(backend=op_replace_backend)(model)
output = fn(input_)

torch.testing.assert_close(output, expected)
