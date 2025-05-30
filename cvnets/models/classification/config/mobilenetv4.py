
import math
from typing import Dict

from utils.math_utils import make_divisible


def get_configuration(opts) -> Dict:
    width_mult = getattr(opts, "model.classification.mobilenetv4.width_multiplier", 1.0)

    def scale_channels(in_channels):
        return make_divisible(int(math.ceil(in_channels * width_mult)), 16)

    config = {
        "conv1_out": scale_channels(32),
        "layer1": {
            "out_channels1": scale_channels(32),
            "out_channels2": scale_channels(32),
            "kernel_size1":3,
            "kernel_size2":1,
            "stride": 2,
            "block_type":"conv_bn",
            "layer":2
        },
        "layer2": {
            "out_channels1": scale_channels(96),
            "out_channels2": scale_channels(64),
            "kernel_size1":3,
            "kernel_size2":1,
            "stride": 2,
            "block_type":"conv_bn",
            "layer":3
        },
        "layer3": {
            "out_channels": scale_channels(96),
            "stride": 2,
            "block_type":"uib",
            "layer":4
        },
        "layer4": {
            "out_channels": scale_channels(128),
            "stride": 2,
            "block_type":"uib",
            "layer":5,
        },
        "layer5": {
            "out_channels1": scale_channels(960),
            "kernel_size1":1,
            "stride": 1,
            "block_type":"conv_bn",
        }, 
        "last_channels" :1280,       
    }
    return config
