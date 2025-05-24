
import math
from typing import Dict

from utils.math_utils import make_divisible
from utils import logger

def get_configuration(opts) -> Dict:
    mode = getattr(opts, "model.classification.mit.mode", "small")
    if mode is None:
        logger.error("Please specify mode")

    head_dim = getattr(opts, "model.classification.mit.head_dim", None)
    num_heads = getattr(opts, "model.classification.mit.number_heads", 4)
    if head_dim is not None:
        if num_heads is not None:
            logger.error(
                "--model.classification.mit.head-dim and --model.classification.mit.number-heads "
                "are mutually exclusive."
            )
    elif num_heads is not None:
        if head_dim is not None:
            logger.error(
                "--model.classification.mit.head-dim and --model.classification.mit.number-heads "
                "are mutually exclusive."
            )
    width_mult = getattr(opts, "model.classification.mobilenetv1.width_multiplier", 1.0)

    def scale_channels(in_channels):
        return make_divisible(int(math.ceil(in_channels * width_mult)), 16)
    mv2_exp_mult = 2
    config = {
        "conv1_out": scale_channels(32),
        "layer1": {"out_channels": scale_channels(64), "stride": 1, "repeat": 1},
        "layer2": {
            "out_channels": scale_channels(144),
            "stride": 2,
            "repeat": 1,
            "layer":2
        },
        "layer3": {
            "out_channels": scale_channels(256),
            "stride": 2,
            "repeat": 1,
            "layer":3
        },
        "layer4": {
            "out_channels": scale_channels(576),
            "stride": 2,
            "repeat": 5,
            "layer":4
        },
        "layer5": {  # 7x7
            "out_channels": 160,
            "transformer_channels": 240,
            "ffn_dim": 480,
            "transformer_blocks": 3,
            "patch_h": 2,
            "patch_w": 2,
            "stride": 2,
            "mv_expand_ratio": mv2_exp_mult,
            "head_dim": head_dim,
            "num_heads": 4,
            "block_type": "mobilevit",
        },
        "last_layer_exp_factor": 4,
    }
    return config
