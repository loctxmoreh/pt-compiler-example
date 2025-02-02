from functools import wraps
import logging
from typing import Callable, List

import torch
from torch import nn
import torch._dynamo
from torch._functorch.aot_autograd import aot_module_simplified
from torch._inductor.compile_fx import compile_fx
from torchvision.models import resnet50

from ops import fused_conv2d_batchnorm

logger = logging.getLogger(__name__)


def conv2d_bn_fuser(gm: torch.fx.GraphModule, sample_inputs: List[torch.Tensor]) -> Callable:

    def _match_conv_batch_norm_pattern(node): 
        if node.op != "call_function" or node.target != torch.ops.aten._native_batch_norm_legit_no_training.default:
            return False

        conv_node = node.args[0]
        if conv_node.op != "call_function":
            return False

        return conv_node.target == torch.ops.aten.convolution.default

    def _iterate_and_fuse(gm, sample_inputs):
        new_graph = torch.fx.Graph()
        env = {}

        for node in gm.graph.nodes:
            if not _match_conv_batch_norm_pattern(node):
                new_node = new_graph.node_copy(node, lambda n: env.get(n, n))
                env[node] = new_node
            else:
                bn_node = node
                conv_node = node.args[0]

                new_fused_conv_bn = new_graph.call_function(
                    torch.ops.custom_op.fused_conv2d_bn, 
                    (
                        env.get(conv_node.args[0], conv_node.args[0]),  # input
                        env.get(conv_node.args[1], conv_node.args[1]),  # conv_weight
                        env.get(conv_node.args[2], conv_node.args[2]),  # conv_bias
                        env.get(bn_node.args[1], bn_node.args[1]),      # bn_weight
                        env.get(bn_node.args[2], bn_node.args[2]),      # bn_bias
                        env.get(bn_node.args[3], bn_node.args[3]),      # bn_running_mean
                        env.get(bn_node.args[4], bn_node.args[4]),      # bn_running_var
                        bn_node.args[6],        # eps
                        conv_node.args[3],      # stride
                        conv_node.args[4],      # padding
                        conv_node.args[5],      # dilation
                        conv_node.args[8],      # group
                    )
                )
                env[bn_node] = new_fused_conv_bn
                env[conv_node] = new_fused_conv_bn

        new_graph.lint()
        new_graph.eliminate_dead_code() # will remove old redundant conv2d nodes
        new_gm = torch.fx.GraphModule(gm, new_graph)

        logger.debug(new_gm.code)
        assert "torch.ops.custom_op.fused_conv2d_bn" in new_gm.code, "No fusion happen!?"

        # finally, pass the new fx graph to inductor
        # return compile_fx(new_gm, sample_inputs)

        return new_gm

    # fw_compiler = compile_fx
    fw_compiler = _iterate_and_fuse
    return aot_module_simplified(
        gm, sample_inputs, fw_compiler=fw_compiler,
    )


def main():
    logging.basicConfig(level=logging.DEBUG)

    dtype = torch.float16
    device = "cuda"
    model = resnet50().to(dtype).to(device)
    model.eval()
    B, C, H, W = 1, 3, 224, 224    
    input_ = torch.rand((B, C, H, W), dtype=dtype, device=device)

    with torch.inference_mode():
        expected = model(input_)

    with torch.inference_mode():
        torch._dynamo.reset()
        fn = torch.compile(backend=conv2d_bn_fuser)(model)
        out = fn(input_)

    torch.testing.assert_close(out, expected, atol=1e-3, rtol=1e-2)


if __name__ == "__main__":
    main()
