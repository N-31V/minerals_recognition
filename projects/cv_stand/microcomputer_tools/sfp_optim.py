from collections import OrderedDict
from typing import Dict

import torch
from torchvision.models import ResNet

from core.operatio.optimization.sfp_tools import _check_nonzero_filters, _prune_filters, \
    _prune_batchnorm, _parse_sd, _collect_sd, sizes_from_state_dict

from pruned_resnet_mk import PRUNED_MODELS_FOR_MK

MODELS_FROM_LENGHT = {
    122: 'ResNet18',
    218: 'ResNet34',
    320: 'ResNet50',
    626: 'ResNet101',
    932: 'ResNet152',
}


def _prune_resnet_block_mk(block: Dict) -> None:
    """Prune block of ResNet"""
    keys = list(block.keys())
    if 'downsample' in keys:
        keys.remove('downsample')
    filters = _check_nonzero_filters(block[keys[0]]['weight'])
    block[keys[0]]['weight'] = _prune_filters(
        weight=block[keys[0]]['weight'],
        saving_filters=filters,
    )
    channels = filters
    final_conv = keys[-2]
    keys = keys[1:-2]

    for key in keys:
        if key.startswith('conv'):
            filters = _check_nonzero_filters(block[key]['weight'])
            block[key]['weight'] = _prune_filters(
                weight=block[key]['weight'],
                saving_filters=filters,
                saving_channels=channels
            )
            channels = filters
        elif key.startswith('bn'):
            block[key] = _prune_batchnorm(bn=block[key], saving_channels=channels)
    block[final_conv]['weight'] = _prune_filters(
        weight=block[final_conv]['weight'],
        saving_channels=channels,
    )


def prune_resnet_state_dict_mk(
        state_dict: OrderedDict
) ->OrderedDict:
    """Prune state_dict of ResNet in a microcomputer-compatible way.

    Args:
        state_dict: ``state_dict`` of ResNet model.

    Returns:
        state_dict.
    """
    sd = _parse_sd(state_dict)
    for layer in ['layer1', 'layer2', 'layer3', 'layer4']:
        for block in sd[layer].values():
            _prune_resnet_block_mk(block=block)
    sd = _collect_sd(sd)
    return sd


def prune_resnet(model: ResNet) -> ResNet:
    """Prune ResNet

    Args:
        model: ResNet model.
        mk: If ``true`` prunes the model in a microcomputer-compatible way.

    Returns:
        Pruned ResNet model.

    Raises:
        AssertionError if model is not Resnet.
    """
    assert isinstance(model, ResNet), "Supports only ResNet models"
    model_type = MODELS_FROM_LENGHT[len(model.state_dict())]
    pruned_sd = prune_resnet_state_dict_mk(model.state_dict())
    sizes = sizes_from_state_dict(pruned_sd)
    model = PRUNED_MODELS_FOR_MK[model_type](sizes=sizes)
    model.load_state_dict(pruned_sd)
    return model


def load_sfp_resnet_model(
        model: ResNet,
        state_dict_path: str,
) -> torch.nn.Module:
    """Loads SFP state_dict to model.

    Args:
        model: An instance of the base model.
        state_dict_path: Path to state_dict file.
        mk: If ``true`` loads the model in a microcomputer-compatible way.

    Raises:
        AssertionError if model is not Resnet.
    """
    assert isinstance(model, ResNet), "Supports only ResNet models"
    state_dict = torch.load(state_dict_path, map_location='cpu')
    sizes = sizes_from_state_dict(state_dict)
    model_type = MODELS_FROM_LENGHT[len(model.state_dict())]
    model = PRUNED_MODELS_FOR_MK[model_type](sizes=sizes)
    model.load_state_dict(state_dict)
    return model
