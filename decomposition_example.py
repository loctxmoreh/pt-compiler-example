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
    assert "silu" in gm.code

    def _silu_decomposer(gm, example_inputs):
        new_graph = torch.fx.Graph()
        env = {}
        tracer = torch.fx.proxy.GraphAppendingTracer(new_graph)

        for node in gm.graph.nodes:
            if node.op == "call_function" and node.target == torch.ops.aten.silu.default:
                proxy_args = [
                    torch.fx.Proxy(env[x.name], tracer) if isinstance(x, torch.fx.Node) else x
                    for x in node.args
                ]
                output_proxy = _silu_decomposition(*proxy_args)

                new_node = output_proxy.node
                env[node.name] = new_node
            else:
                new_node = new_graph.node_copy(node, lambda n: env[n.name])
                env[node.name] = new_node

        new_graph.lint()
        new_gm = torch.fx.GraphModule(gm, new_graph)
        new_gm.recompile()
        assert "silu" not in new_gm.code
        return new_gm

    return aot_module_simplified(
        gm, example_inputs,
        fw_compiler=_silu_decomposer,
    )


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
        backend = simple_silu_decomp
        backend = manual_silu_decomp
        fn = torch.compile(backend=backend)(model)
        out = fn(input_)

    torch.testing.assert_close(out, expected)


if __name__ == "__main__":
    main()
