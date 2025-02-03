import logging

import torch
import torch._dynamo
from torch._functorch.aot_autograd import aot_module_simplified
from torch._inductor.compile_fx import compile_fx
from torchvision.models import efficientnet_v2_s as efficientnet_v2

logger = logging.getLogger(__name__)


def _silu_decomposition(x):
    return x * torch.sigmoid(x)


def simple_silu_decomp(gm, example_inputs):
    assert "silu" in gm.code
    # logger.debug(gm.code)

    def _fw_compiler(gm, example_inputs):
        # Silu is decomposed, so it should not be in the generated code
        assert "silu" not in gm.code

        logger.debug(gm.code)
        return gm

    return aot_module_simplified(
        gm, example_inputs, 
        fw_compiler=_fw_compiler,
        decompositions={
            torch.ops.aten.silu.default: _silu_decomposition,
        }
    )


def manual_silu_decomp(gm, example_inputs):
    pass


def main():
    logging.basicConfig(level=logging.DEBUG)

    dtype = torch.float16
    device = "cuda"
    model = efficientnet_v2().to(dtype).to(device)
    model.eval()
    B, C, H, W = 1, 3, 224, 224    
    input_ = torch.rand((B, C, H, W), dtype=dtype, device=device)

    with torch.inference_mode():
        expected = model(input_)

    with torch.inference_mode():
        torch._dynamo.reset()
        fn = torch.compile(backend=simple_silu_decomp)(model)
        out = fn(input_)

    torch.testing.assert_close(out, expected)


if __name__ == "__main__":
    main()
