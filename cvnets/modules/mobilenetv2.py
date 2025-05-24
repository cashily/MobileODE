#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2023 Apple Inc. All Rights Reserved.
#

from typing import Optional, Union

from torch import Tensor, nn
import torch
from cvnets.layers import ConvLayer2d
from cvnets.layers.activation import build_activation_layer
from cvnets.modules import BaseModule, SqueezeExcitation
from utils.math_utils import make_divisible
import torch.nn.functional as F

class InvertedResidualSE(BaseModule):
    """
    This class implements the inverted residual block with squeeze-excitation unit, as described in
    `MobileNetv3 <https://arxiv.org/abs/1905.02244>`_ paper

    Args:
        opts: command-line arguments
        in_channels (int): :math:`C_{in}` from an expected input of size :math:`(N, C_{in}, H_{in}, W_{in})`
        out_channels (int): :math:`C_{out}` from an expected output of size :math:`(N, C_{out}, H_{out}, W_{out)`
        expand_ratio (Union[int, float]): Expand the input channels by this factor in depth-wise conv
        dilation (Optional[int]): Use conv with dilation. Default: 1
        stride (Optional[int]): Use convolutions with a stride. Default: 1
        use_se (Optional[bool]): Use squeeze-excitation block. Default: False
        act_fn_name (Optional[str]): Activation function name. Default: relu
        se_scale_fn_name (Optional [str]): Scale activation function inside SE unit. Defaults to hard_sigmoid
        kernel_size (Optional[int]): Kernel size in depth-wise convolution. Defaults to 3.
        squeeze_factor (Optional[bool]): Squeezing factor in SE unit. Defaults to 4.

    Shape:
        - Input: :math:`(N, C_{in}, H_{in}, W_{in})`
        - Output: :math:`(N, C_{out}, H_{out}, W_{out})`
    """

    def __init__(
        self,
        opts,
        in_channels: int,
        out_channels: int,
        expand_ratio: Union[int, float],
        dilation: Optional[int] = 1,
        stride: Optional[int] = 1,
        use_se: Optional[bool] = False,
        act_fn_name: Optional[str] = "relu",
        se_scale_fn_name: Optional[str] = "hard_sigmoid",
        kernel_size: Optional[int] = 3,
        squeeze_factor: Optional[int] = 4,
        *args,
        **kwargs
    ) -> None:
        hidden_dim = make_divisible(int(round(in_channels * expand_ratio)), 8)
        act_fn = build_activation_layer(opts, act_type=act_fn_name, inplace=True)

        super().__init__()

        block = nn.Sequential()
        if expand_ratio != 1:
            block.add_module(
                name="exp_1x1",
                module=ConvLayer2d(
                    opts,
                    in_channels=in_channels,
                    out_channels=hidden_dim,
                    kernel_size=1,
                    use_act=False,
                    use_norm=True,
                ),
            )
            block.add_module(name="act_fn_1", module=act_fn)

        block.add_module(
            name="conv_3x3",
            module=ConvLayer2d(
                opts,
                in_channels=hidden_dim,
                out_channels=hidden_dim,
                stride=stride,
                kernel_size=kernel_size,
                groups=hidden_dim,
                use_act=False,
                use_norm=True,
                dilation=dilation,
            ),
        )
        block.add_module(name="act_fn_2", module=act_fn)

        if use_se:
            se = SqueezeExcitation(
                opts=opts,
                in_channels=hidden_dim,
                squeeze_factor=squeeze_factor,
                scale_fn_name=se_scale_fn_name,
            )
            block.add_module(name="se", module=se)

        block.add_module(
            name="red_1x1",
            module=ConvLayer2d(
                opts,
                in_channels=hidden_dim,
                out_channels=out_channels,
                kernel_size=1,
                use_act=False,
                use_norm=True,
            ),
        )

        self.block = block
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.exp = expand_ratio
        self.dilation = dilation
        self.use_se = use_se
        self.stride = stride
        self.act_fn_name = act_fn_name
        self.kernel_size = kernel_size
        self.use_res_connect = self.stride == 1 and in_channels == out_channels

    def forward(self, x: Tensor, *args, **kwargs) -> Tensor:
        y = self.block(x)
        return x + y if self.use_res_connect else y

    def __repr__(self) -> str:
        return "{}(in_channels={}, out_channels={}, stride={}, exp={}, dilation={}, use_se={}, kernel_size={}, act_fn={})".format(
            self.__class__.__name__,
            self.in_channels,
            self.out_channels,
            self.stride,
            self.exp,
            self.dilation,
            self.use_se,
            self.kernel_size,
            self.act_fn_name,
        )


class InvertedResidual(BaseModule):
    """
    This class implements the inverted residual block, as described in `MobileNetv2 <https://arxiv.org/abs/1801.04381>`_ paper

    Args:
        opts: command-line arguments
        in_channels (int): :math:`C_{in}` from an expected input of size :math:`(N, C_{in}, H_{in}, W_{in})`
        out_channels (int): :math:`C_{out}` from an expected output of size :math:`(N, C_{out}, H_{out}, W_{out)`
        stride (Optional[int]): Use convolutions with a stride. Default: 1
        expand_ratio (Union[int, float]): Expand the input channels by this factor in depth-wise conv
        dilation (Optional[int]): Use conv with dilation. Default: 1
        skip_connection (Optional[bool]): Use skip-connection. Default: True

    Shape:
        - Input: :math:`(N, C_{in}, H_{in}, W_{in})`
        - Output: :math:`(N, C_{out}, H_{out}, W_{out})`

    .. note::
        If `in_channels =! out_channels` and `stride > 1`, we set `skip_connection=False`

    """

    def __init__(
        self,
        opts,
        in_channels: int,
        out_channels: int,
        stride: int,
        expand_ratio: Union[int, float],
        dilation: int = 1,
        skip_connection: Optional[bool] = True,
        *args,
        **kwargs
    ) -> None:
        assert stride in [1, 2]
        hidden_dim = make_divisible(int(round(in_channels * expand_ratio)), 8)
        super().__init__()

        block = nn.Sequential()
        if expand_ratio != 1:
            block.add_module(
                name="exp_1x1",
                module=ConvLayer2d(
                    opts,
                    in_channels=in_channels,
                    out_channels=hidden_dim,
                    kernel_size=1,
                    use_act=True,
                    use_norm=True,
                ),
            )

        block.add_module(
            name="conv_3x3",
            module=ConvLayer2d(
                opts,
                in_channels=hidden_dim,
                out_channels=hidden_dim,
                stride=stride,
                kernel_size=3,
                groups=hidden_dim,
                use_act=True,
                use_norm=True,
                dilation=dilation,
            ),
        )

        block.add_module(
            name="red_1x1",
            module=ConvLayer2d(
                opts,
                in_channels=hidden_dim,
                out_channels=out_channels,
                kernel_size=1,
                use_act=False,
                use_norm=True,
            ),
        )

        self.block = block
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.exp = expand_ratio
        self.dilation = dilation
        self.stride = stride
        self.use_res_connect = (
            self.stride == 1 and in_channels == out_channels and skip_connection
        )

    def forward(self, x: Tensor, *args, **kwargs) -> Tensor:
        # print(x.shape)
        # print(self.block)
        if self.use_res_connect:
            return x + self.block(x)
        else:
            return self.block(x)

    def __repr__(self) -> str:
        return "{}(in_channels={}, out_channels={}, stride={}, exp={}, dilation={}, skip_conn={})".format(
            self.__class__.__name__,
            self.in_channels,
            self.out_channels,
            self.stride,
            self.exp,
            self.dilation,
            self.use_res_connect,
        )
class AdaptiveDiscretizedNeuralODE8_Spatial_pointwise_view_17(nn.Module):
    def __init__(self, i,in_channels, out_channels,num_layers=30):
        super().__init__()
        self.num_layers = num_layers  # 设置层数
        self.inn=int(in_channels ** 0.5)
        self.out=int(out_channels ** 0.5)
        self.oc = i*i
        # 初始化 sigma 和卷积核
        self.sigma = nn.Sequential(
            nn.BatchNorm2d(self.oc),
            nn.ReLU6()
        )
        
        
        # 初始化 delta_t 和卷积核参数
        self.delta_t = nn.Parameter(torch.empty(num_layers, 1).uniform_(1e-4, 1))  # 用于时间步长的参数
        self.matrices = nn.Parameter(torch.empty(num_layers, 1, 1, self.out, self.out))  # 用于卷积核的参数

        # 使用 He 正态分布初始化所有矩阵
        torch.nn.init.normal_(self.matrices, mean=0, std=0.1)  # 初始化卷积核
        torch.nn.init.normal_(self.delta_t, mean=0.01, std=0.005)  # 初始化 delta_t

        self.relu6 = nn.ReLU6()
        self.tanh = nn.Tanh()
        self.sigmoid=nn.Sigmoid()
    
    def feature_reshape(self, x, b, c, W):
        x_reshaped = x.view(b, c, -1)  # (b, c, H*W)
        x_reshaped = x_reshaped.permute(0, 2, 1).contiguous()  # (b, H*W, c)
        # print(self.inn,self.out,x.shape)
        if self.inn != self.out:
            
            x_reshaped = x_reshaped.view(b, -1, self.inn, self.inn)  # 重新 reshape 为 (b, H*W, sqrt(c), sqrt(c))
            x_expanded = F.interpolate(x_reshaped, size=(self.out, self.out), mode='bilinear', align_corners=False)
        else:
            x_expanded = x_reshaped.view(b, -1, self.out, self.out)
        

        return x_expanded
    def forward(self, x):
        B, C, H, W = x.size()
        sqrt_c = int(C ** 0.5)
        
        # 重塑输入
        x1 = self.feature_reshape(x, B, C, W)  # (b, H*W, sqrt(c), sqrt(c))
        y0 = x1

        # 通过 delta_t 规范化
        delta_t_normalized = self.relu6(self.delta_t)
        # print(x1.size(),y0.size(),self.oc)
        for layer in range(self.num_layers):
            new_x = self.matrices[layer]  # 获取当前层的卷积核

            dydt = -y0 + self.sigma(y0 + new_x * x1)

            # 使用当前通道的 delta_t 更新状态
            y0 = y0 + delta_t_normalized[layer]  * dydt


        # 最终输出的形状处理
        y_out = (y0 + x1).view(B, H, W, self.out * self.out)  # 处理维度为 (b, H, W, sqrt_c * sqrt_c)
        y0 = y_out.permute(0, 3, 1, 2)  # 最终转换为 (b, c, H, W)
        # pdb.set_trace()
        return y0
class InvertedResidual_ode(BaseModule):
    """
    This class implements the inverted residual block, as described in `MobileNetv2 <https://arxiv.org/abs/1801.04381>`_ paper

    Args:
        opts: command-line arguments
        in_channels (int): :math:`C_{in}` from an expected input of size :math:`(N, C_{in}, H_{in}, W_{in})`
        out_channels (int): :math:`C_{out}` from an expected output of size :math:`(N, C_{out}, H_{out}, W_{out)`
        stride (Optional[int]): Use convolutions with a stride. Default: 1
        expand_ratio (Union[int, float]): Expand the input channels by this factor in depth-wise conv
        dilation (Optional[int]): Use conv with dilation. Default: 1
        skip_connection (Optional[bool]): Use skip-connection. Default: True

    Shape:
        - Input: :math:`(N, C_{in}, H_{in}, W_{in})`
        - Output: :math:`(N, C_{out}, H_{out}, W_{out})`

    .. note::
        If `in_channels =! out_channels` and `stride > 1`, we set `skip_connection=False`

    """

    def __init__(
        self,
        i: int,
        o: int,
        opts,
        in_channels: int,
        out_channels: int,
        stride: int,
        expand_ratio: Union[int, float],
        dilation: int = 1,
        skip_connection: Optional[bool] = True,
        num_layers: int = 10,
        *args,
        **kwargs
    ) -> None:
        hidden_dim = make_divisible(int(round(in_channels * expand_ratio)), 8)
        super().__init__()
        

        block = nn.Sequential()
        if expand_ratio != 1:
            block.add_module(
                name="exp_1x1",
                module=nn.Sequential(AdaptiveDiscretizedNeuralODE8_Spatial_pointwise_view_17(i, in_channels, hidden_dim,num_layers),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True))                     
            )

        block.add_module(
            name="conv_3x3",
            module=ConvLayer2d(
                opts,
                in_channels=hidden_dim,
                out_channels=hidden_dim,
                stride=stride,
                kernel_size=3,
                groups=hidden_dim,
                use_act=True,
                use_norm=True,
                dilation=dilation,
            ),
        )

        block.add_module(
            name="red_1x1",
            module=AdaptiveDiscretizedNeuralODE8_Spatial_pointwise_view_17(o, hidden_dim, out_channels,num_layers)
        )

        self.block = block
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.exp = expand_ratio
        self.dilation = dilation
        self.stride = stride
        self.use_res_connect = (
            self.stride == 1 and in_channels == out_channels and skip_connection
        )

    def forward(self, x: Tensor, *args, **kwargs) -> Tensor:
        # print(x.shape)
        # print(self.block)
        if self.use_res_connect:
            return x + self.block(x)
        else:
            return self.block(x)

    def __repr__(self) -> str:
        return "{}(in_channels={}, out_channels={}, stride={}, exp={}, dilation={}, skip_conn={})".format(
            self.__class__.__name__,
            self.in_channels,
            self.out_channels,
            self.stride,
            self.exp,
            self.dilation,
            self.use_res_connect,
        )