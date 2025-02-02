import math
from typing import Optional, List

import torch
from torch.nn import functional as F
from torchvision.models import resnet18


def _naive_conv2d_batchnorm(x, conv_weight, conv_bias, bn_weight, bn_bias,
                            bn_running_mean, bn_running_var, eps, stride,
                            padding, dilation, groups):
    return F.batch_norm(
        F.conv2d(x, conv_weight, conv_bias, stride, padding, dilation, groups), 
        bn_running_mean, bn_running_var, bn_weight, bn_bias, eps=eps
    )


@torch.library.custom_op("custom_op::fused_conv2d_bn", mutates_args=())
def fused_conv2d_batchnorm(x: torch.Tensor, 
                           conv_weight: torch.Tensor,
                           conv_bias: Optional[torch.Tensor], 
                           bn_weight: Optional[torch.Tensor], 
                           bn_bias: Optional[torch.Tensor],
                           bn_running_mean: Optional[torch.Tensor], 
                           bn_running_var: Optional[torch.Tensor], 
                           eps: float, 
                           stride: List[int],
                           padding: List[int], 
                           dilation: List[int], 
                           groups: int) -> List[torch.Tensor]:
    if conv_bias is None:
        conv_bias = torch.zeros(conv_weight.size(0), device=x.device, dtype=x.dtype)
    
    bn_scale = bn_weight / torch.sqrt(bn_running_var + eps)
    fused_weight = conv_weight * bn_scale.view(-1, 1, 1, 1)
    fused_bias = (conv_bias - bn_running_mean) * bn_scale + bn_bias

    # see: https://github.com/pytorch/pytorch/blob/2fd1b6b3610eb84cd615360a8fd23756a7f2e743/aten/src/ATen/native/miopen/BatchNorm_miopen.cpp#L136
    save_mean = torch.empty(0, dtype=x.dtype)
    save_var = torch.empty(0, dtype=x.dtype)

    return F.conv2d(x, fused_weight, fused_bias, stride, padding, dilation, groups), save_mean, save_var


@fused_conv2d_batchnorm.register_fake
def _(x, conv_weight, conv_bias, bn_weight, bn_bias, bn_running_mean,
      bn_running_var, eps, stride, padding, dilation, groups):
    B, C, H, W = x.shape
    c_out, c_in, k_h, k_w = conv_weight.shape
    pad_h, pad_w = padding
    stride_h, stride_w = stride
    dilation_h, dilation_w = dilation

    H_out = (H + 2 * pad_h - dilation_h * (k_h - 1) - 1) // stride_h + 1
    W_out = (W + 2 * pad_w - dilation_h * (k_w - 1) - 1) // stride_w + 1
    C_out = c_out

    save_mean = torch.empty(0, dtype=x.dtype, device=x.device)
    save_var = torch.empty(0, dtype=x.dtype, device=x.device)

    return torch.empty((B, C_out, H_out, W_out), dtype=x.dtype, device=x.device), save_mean, save_var


def prepare_inputs(B, C, H, W, dtype=torch.float16, device="cuda"):
    model = resnet18().to(dtype).to(device)
    x = torch.rand((B, C, H, W), dtype=dtype, device="cuda")

    conv_layer = model.layer1[0].conv1
    conv_weight = conv_layer.weight
    conv_bias = conv_layer.bias
    stride = conv_layer.stride
    padding = conv_layer.padding
    dilation = conv_layer.dilation 
    groups = conv_layer.groups

    bn_layer = model.layer1[0].bn1
    bn_weight = bn_layer.weight
    bn_bias = bn_layer.bias
    bn_running_mean = bn_layer.running_mean 
    bn_running_var = bn_layer.running_var 
    eps = bn_layer.eps

    return (x, conv_weight, conv_bias, bn_weight, bn_bias, bn_running_mean,
            bn_running_var, eps, stride, padding, dilation, groups)


def test_correctness():
    args = prepare_inputs(B=1, C=64, H=224, W=224)
    expected = _naive_conv2d_batchnorm(*args)
    result, *_ = fused_conv2d_batchnorm(*args)
    torch.testing.assert_close(expected, result)


def test_fake_impl():
    args = prepare_inputs(B=1, C=64, H=224, W=224)
    meta_args = tuple(x.clone().to("meta") if isinstance(x, torch.Tensor) else x for x in args)
    expected = _naive_conv2d_batchnorm(*args)
    results, *_ = fused_conv2d_batchnorm._abstract_fn(*meta_args)
    assert expected.shape == results.shape, f"{expected.shape=}, {results.shape=}"


if __name__ == "__main__":
    test_correctness()
    test_fake_impl()
