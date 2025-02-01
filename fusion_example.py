import logging

import torch
from torch import nn
import torch._dynamo
from torch._functorch.aot_autograd import aot_module_simplified
from torchvision.models import resnet50

from fused_conv2d_bn import fused_conv2d_batchnorm

logger = logging.getLogger(__name__)


# def conv2d_bn_pattern(x, conv_weight, conv_bias, bn_weight, bn_bias, bn_running_mean,
#                       bn_running_var, eps, stride, padding, dilation, groups):
#     conv = torch.conv2d(x, conv_weight, conv_bias, stride, padding, dilation, groups)
#     bn = torch.nn.functional.batch_norm(conv, bn_running_mean, bn_running_var,
#                                         bn_weight, bn_bias, 
#                                         True,       # training
#                                         0.1,        # momentum
#                                         eps)
#     return bn


def conv2d_bn_aten_pattern(x, conv_weight, conv_bias, bn_weight, bn_bias,
                           bn_running_mean, bn_running_var):
    conv = torch.ops.aten.convolution.default(x, conv_weight, None,
                                              [1, 1], [0, 0], [1, 1], False,
                                              [0, 0], 1)
    bn = torch.ops.aten._native_batch_norm_legit_no_training.default(conv,
                                                                     bn_running_mean,
                                                                     bn_running_var,
                                                                     bn_weight,
                                                                     bn_bias,
                                                                     0.1,
                                                                     1e-5,)
    out = bn[0]
    save_mean = bn[1]
    save_invstd = bn[2]
    return bn, save_mean, save_invstd


def fused_conv2d_bn_pattern(x, conv_weight, conv_bias, bn_weight, bn_bias,
                            bn_running_mean, bn_running_var):
    out = fused_conv2d_batchnorm(x, conv_weight, conv_bias, bn_weight, bn_bias,
                                 bn_running_mean, bn_running_var, eps=1e-5,
                                 stride=[1, 1], padding=[0, 0], dilation=[1, 1],
                                 groups=1)
    return out, bn_running_mean, bn_running_var


def fuser(gm, sample_inputs):

    def _match_conv_batch_norm_pattern(node): 
        if node.op != "call_function" or node.target != torch.ops.aten._native_batch_norm_legit_no_training.default:
            return False

        conv_node = node.args[0]
        if conv_node.op != "call_function":
            return False

        return conv_node.target == torch.ops.aten.convolution.default

    def _fuse_by_iterating(gm, sample_inputs):
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
                        env[conv_node.args[0]],
                        env.get(conv_node.args[1], conv_node.args[1]),  # conv_weight
                        env.get(conv_node.args[2], conv_node.args[2]),  # conv_bias
                        env.get(bn_node.args[1], bn_node.args[1]),    # bn_weight
                        env.get(bn_node.args[2], bn_node.args[2]),    # bn_bias
                        env.get(bn_node.args[3], bn_node.args[3]),    # bn_running_mean
                        env.get(bn_node.args[4], bn_node.args[4]),    # bn_running_var
                        bn_node.args[6],    # eps
                        conv_node.args[3],   # stride
                        conv_node.args[4],   # padding
                        conv_node.args[5],   # dilation
                        conv_node.args[8],   # group

                    )
                )
                env[node] = new_fused_conv_bn
                env[conv_node] = new_fused_conv_bn

        new_graph.print_tabular()
        new_graph.lint()
        new_graph.eliminate_dead_code()
        new_gm = torch.fx.GraphModule(gm, new_graph)
        new_gm.recompile()
        new_gm.print_readable()
        return new_gm

    # fw_compiler = _fuse_by_matcher
    fw_compiler = _fuse_by_iterating
    return aot_module_simplified(
        gm, sample_inputs, fw_compiler=fw_compiler,
    )


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
    fn = torch.compile(backend=fuser)(model)
    out = fn(input_)

torch.testing.assert_close(out, expected)
