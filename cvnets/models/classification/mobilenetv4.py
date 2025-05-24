
import argparse
from typing import Dict, List, Optional, Tuple

from torch import nn

from cvnets.layers import (
    ConvLayer2d,
    Dropout,
    GlobalPool,
    Identity,
    LinearLayer,
    SeparableConv2d,
)
from cvnets.models import MODEL_REGISTRY
from cvnets.models.classification.base_image_encoder import BaseImageEncoder
from cvnets.models.classification.config.mobilenetv4 import get_configuration
from utils.math_utils import bound_fn
from cvnets.modules import InvertedResidual_ode

@MODEL_REGISTRY.register(name="mobilenetv4", type="classification")
class MobileNetv1_ode(BaseImageEncoder):
    """
    This class defines the `MobileNet architecture <https://arxiv.org/abs/1704.04861>`_
    """

    def __init__(self, opts, *args, **kwargs) -> None:

        image_channels = 3
        num_classes = getattr(opts, "model.classification.n_classes", 1000)
        classifier_dropout = getattr(
            opts, "model.classification.classifier_dropout", 0.0
        )
        if classifier_dropout == 0.0:
            width_mult = getattr(
                opts, "model.classification.mobilenetv4.width_multiplier", 1.0
            )
            val = round(0.1 * width_mult, 3)
            classifier_dropout = bound_fn(min_val=0.0, max_val=0.1, value=val)

        super().__init__(opts, *args, **kwargs)

        cfg = get_configuration(opts=opts)

        self.model_conf_dict = dict()
        input_channels = 32
        self.conv_1 = ConvBN(
                    in_channels=3,
                    out_channels=32,
                    kernel_size=3,
                    stride=2,
                )
        self.model_conf_dict["conv1"] = {"in": image_channels, "out": input_channels}

        self.layer_1, out_channels = self._make_layer(
            opts=opts, mv4_config=cfg["layer1"], input_channel=input_channels
        )
        self.model_conf_dict["layer1"] = {"in": input_channels, "out": out_channels}
        input_channels = out_channels


        self.layer_2, out_channels = self._make_layer(
            opts=opts, mv4_config=cfg["layer2"], input_channel=input_channels
        )
        self.model_conf_dict["layer2"] = {"in": input_channels, "out": out_channels}
        input_channels = out_channels

        self.layer_3, out_channels = self._make_layer(
            opts=opts, mv4_config=cfg["layer3"], input_channel=input_channels
        )
        self.model_conf_dict["layer3"] = {"in": input_channels, "out": out_channels}
        input_channels = out_channels


        self.layer_4, out_channels = self._make_layer(
            opts=opts,
            mv4_config=cfg["layer4"],
            input_channel=input_channels,
            dilate=1,
        )

        self.model_conf_dict["layer4"] = {"in": input_channels, "out": out_channels}
        input_channels = out_channels

        self.layer_5, out_channels = self._make_layer(
            opts=opts,
            mv4_config=cfg["layer5"],
            input_channel=input_channels,
            dilate=1,
        )
        self.model_conf_dict["layer5"] = {"in": input_channels, "out": out_channels}
        input_channels = out_channels


        self.conv_1x1_exp = Identity()
        self.model_conf_dict["exp_before_cls"] = {
            "in": input_channels,
            "out": input_channels,
        }

        pool_type = getattr(opts, "model.layer.global_pool", "mean")

        last_channels = cfg["last_channels"]
        self.classifier = nn.Sequential()
        self.classifier.add_module(
            name="global_pool", module=GlobalPool(pool_type=pool_type, keep_dim=False)
        )
        self.classifier.add_module(
            name="fc1",
            module=ConvBN(
                    in_channels=input_channels,
                    out_channels=last_channels,
                    kernel_size=1,
                    stride=1,
                ),
        )
        self.classifier.add_module(
            name="classifier_fc",
            module=LinearLayer(
                in_features=last_channels, out_features=num_classes, bias=True
            ),
        )
        self.model_conf_dict["cls"] = {"in": input_channels, "out": num_classes}

        # check model
        self.check_model()

        # weight initialization
        self.reset_parameters(opts=opts)

    @classmethod
    def add_arguments(cls, parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
        """Add model specific arguments"""
        group = parser.add_argument_group(title=cls.__name__)
        group.add_argument(
            "--model.classification.mobilenetv4.width-multiplier",
            type=float,
            default=1.0,
            help="Width multiplier for MobileNetv4. Default: 1.0",
        )

        return parser
    def _make_layer(
        self,
        opts,
        input_channel,
        mv4_config: Dict,
        dilate: Optional[bool] = False,
        *args,
        **kwargs
    ) -> Tuple[nn.Sequential, int]:
        block_type = mv4_config.get("block_type", "conv_bn")
        if block_type.lower() == "conv_bn":
            return self._make_conv_bn_layer(
                opts=opts, input_channel=input_channel, mv4_config=mv4_config, dilate=dilate
            )
        else:
            return self._make_uib_layer(
                opts=opts, input_channel=input_channel, mv4_config=mv4_config
            )
    def _make_conv_bn_layer(
        self,
        opts,
        mv4_config: Dict or List,
        input_channel: int,
        dilate: Optional[bool] = False,
        *args,
        **kwargs
    ) -> Tuple[nn.Module, int]:
        prev_dilation = self.dilation
        mv4_block = []
        # print(mv4_config)
        out_channels1 = mv4_config.get("out_channels1")
        
        k1 = mv4_config.get("kernel_size1")



        stride = mv4_config.get("stride", 1)

        mv4_block.append(
            ConvBN(
                in_channels=input_channel,
                out_channels=out_channels1,
                kernel_size=k1,
                stride=stride,
            ),
        )
        input_channel = out_channels1
        if 2 == mv4_config.get("layer", 0) or 3 == mv4_config.get("layer", 0):
            k2 = mv4_config.get("kernel_size2")
            out_channels2 = mv4_config.get("out_channels2")
            mv4_block.append(
                ConvBN(
                    in_channels=input_channel,
                    out_channels=out_channels2,
                    kernel_size=k2,
                    stride=1,
                ),
            )
            input_channel = out_channels2

        return nn.Sequential(*mv4_block), input_channel
    def _make_uib_layer(
        self,
        opts,
        mv4_config,
        input_channel: int,
        dilate: Optional[bool] = False,
        *args,
        **kwargs
    ) -> Tuple[nn.Module, int]:
        prev_dilation = self.dilation
        mv4_block = nn.Sequential()
        count = 0
        if 4 == mv4_config.get("layer", 0):
            block_dict=[[ 5, 5, 2, 96, 3.0],  # ExtraDW
                    [0, 3, 1, 96, 2.0],  # IB
                    [0, 3, 1, 96, 2.0],  # IB
                    [0, 3, 1, 96, 2.0],  # IB
                    [0, 3, 1, 96, 2.0],  # IB
                    [3, 0, 1, 96, 4.0],  # ConvNext
                    # 7px
                    ]
        elif 5 == mv4_config.get("layer", 0):
            block_dict=[[ 3, 3, 2, 128, 6.0],  # ExtraDW
                        [5, 5, 1, 128, 4.0],  # ExtraDW
                        [0, 5, 1, 128, 4.0],  # IB
                        [0, 5, 1, 128, 3.0],  # IB
                        [0, 3, 1, 128, 4.0],  # IB
                        [0, 3, 1, 128, 4.0],  # IB
                        ]
        for i in range(len(block_dict)):
            for start_k, middle_k, s, f, e in [
                block_dict[i]
            ]:
                block_name = "mv4_s_{}_idx_{}".format(s, count)

                layer = UniversalInvertedBottleneck(
                    input_channel, f, e, start_k, middle_k, s
                )
                mv4_block.add_module(name=block_name, module=layer)
                count += 1
                input_channel = f

        return mv4_block, input_channel


def make_divisible(value, divisor, min_value=None, round_down_protect=True):
    if min_value is None:
        min_value = divisor
    new_value = max(min_value, int(value + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if round_down_protect and new_value < 0.9 * value:
        new_value += divisor
    return new_value


class ConvBN(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1):
        super(ConvBN, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, (kernel_size - 1)//2, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
    
    def forward(self, x):
        return self.block(x)


class UniversalInvertedBottleneck(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 expand_ratio,
                 start_dw_kernel_size,
                 middle_dw_kernel_size,
                 stride,
                 middle_dw_downsample: bool = True,
                 use_layer_scale: bool = False,
                 layer_scale_init_value: float = 1e-5):
        super(UniversalInvertedBottleneck, self).__init__()
        self.start_dw_kernel_size = start_dw_kernel_size
        self.middle_dw_kernel_size = middle_dw_kernel_size

        if start_dw_kernel_size:
           self.start_dw_conv = nn.Conv2d(in_channels, in_channels, start_dw_kernel_size, 
                                          stride if not middle_dw_downsample else 1,
                                          (start_dw_kernel_size - 1) // 2,
                                          groups=in_channels, bias=False)
           self.start_dw_norm = nn.BatchNorm2d(in_channels)
        
        expand_channels = make_divisible(in_channels * expand_ratio, 8)
        self.expand_conv = nn.Conv2d(in_channels, expand_channels, 1, 1, bias=False)
        self.expand_norm = nn.BatchNorm2d(expand_channels)
        self.expand_act = nn.ReLU(inplace=True)

        if middle_dw_kernel_size:
           self.middle_dw_conv = nn.Conv2d(expand_channels, expand_channels, middle_dw_kernel_size,
                                           stride if middle_dw_downsample else 1,
                                           (middle_dw_kernel_size - 1) // 2,
                                           groups=expand_channels, bias=False)
           self.middle_dw_norm = nn.BatchNorm2d(expand_channels)
           self.middle_dw_act = nn.ReLU(inplace=True)
        
        self.proj_conv = nn.Conv2d(expand_channels, out_channels, 1, 1, bias=False)
        self.proj_norm = nn.BatchNorm2d(out_channels)

        if use_layer_scale:
            self.gamma = nn.Parameter(layer_scale_init_value * torch.ones((out_channels)), requires_grad=True)

        self.use_layer_scale = use_layer_scale
        self.identity = stride == 1 and in_channels == out_channels

    def forward(self, x):
        shortcut = x

        if self.start_dw_kernel_size:
            x = self.start_dw_conv(x)
            x = self.start_dw_norm(x)

        x = self.expand_conv(x)
        x = self.expand_norm(x)
        x = self.expand_act(x)

        if self.middle_dw_kernel_size:
            x = self.middle_dw_conv(x)
            x = self.middle_dw_norm(x)
            x = self.middle_dw_act(x)

        x = self.proj_conv(x)
        x = self.proj_norm(x)

        if self.use_layer_scale:
            x = self.gamma * x

        return x + shortcut if self.identity else x
