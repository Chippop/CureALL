from collections import OrderedDict

import torch


def convert_unimol_ckpt(ckpt: OrderedDict, no_head_layernorm: bool = True) -> OrderedDict:
    """Convert a UniMol checkpoint to a format compatible with the current model.

    :param ckpt: A checkpoint from a UniMol model.
    :return: A checkpoint compatible with the current model.
    """
    for k in list(ckpt.keys()):
        if k.startswith("lm_head.") or k.startswith("pair2coord_proj.") or k.startswith("dist_head"):
            ckpt.pop(k)
        if no_head_layernorm:
            if k.startswith("encoder.final_head_layer_norm"):
                ckpt.pop(k)
    return ckpt
