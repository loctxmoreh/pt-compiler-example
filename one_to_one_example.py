import torch
from torch import nn
import torch._dynamo
from torch._functorch.aot_autograd import aot_module_simplified

from ops import my_addmm


class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(32, 64)

    def forward(self, x):
        x = self.fc1(x)
        x = torch.nn.functional.gelu(x)
        return x


def replace_addmm_compiler(gm, sample_inputs):
    def _compiler(gm, sample_inputs):
        for node in gm.graph.nodes:
            if node.op == "call_function" and node.target == torch.ops.aten.addmm.default:
                node.target = torch.ops.custom_op.my_addmm

        gm.graph.lint()         # just to make sure
        return gm.forward

    return aot_module_simplified(
        gm, sample_inputs,
        fw_compiler=_compiler,
    )


def test_replace_addmm_compiler():
    model = MLP()
    batch_size = 8
    input_ = torch.randn(batch_size, 32)
    expected = model(input_)

    torch._dynamo.reset()
    fn = torch.compile(backend=replace_addmm_compiler)(model)
    output = fn(input_)

    torch.testing.assert_close(output, expected)

if __name__ == "__main__":
    test_replace_addmm_compiler()
