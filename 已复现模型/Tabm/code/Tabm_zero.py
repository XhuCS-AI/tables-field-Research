import torch
from torch import Tensor
import torch.nn as nn
from typing import Literal

import Tabm.code.deep as deep

DEFAULT_SHARE_TRAINING_BATCHES = True

class Tabm(nn.Module):
    """TabM. 没有特征嵌入的版本"""
    def __init__(
        self,
        *,
        n_num_features: int,
        cat_cardinalities: list[int],
        n_classes: None | int,
        backbone: dict,
        arch_type: Literal[
            # TabM
            'tabm',
            #
            # TabM-mini
            'tabm-mini',
            # TabM. The first adapter is initialized from the normal distribution.
            # This variant was not used in the paper, but it may be useful in practice.
            'tabm-normal',
            #
            # TabM-mini. The adapter is initialized from the normal distribution.
            # This variant was not used in the paper.
            'tabm-mini-normal',
        ],
        k: None | int = None,
        share_training_batches: bool = DEFAULT_SHARE_TRAINING_BATCHES,
    ) -> None:
        # >>> Validate arguments.
        assert n_num_features >= 0
        assert n_num_features or cat_cardinalities
        if arch_type == 'plain':
            assert k is None
            assert (
                share_training_batches
            ), 'If `arch_type` is set to "plain", then `simple` must remain True'
        else:
            assert k is not None
            assert k > 0

        super().__init__()

        first_adapter_sections = []

        if n_num_features == 0:
            d_num = 0
        else:
            d_num = n_num_features
            first_adapter_sections.extend(1 for _ in range(n_num_features))

        first_adapter_sections.extend(cat_cardinalities)
        d_cat = len(cat_cardinalities)

        # >>> Backbone
        d_flat = d_num + d_cat
        self.minimal_ensemble_adapter = None
        self.backbone = deep.make_module(d_in=d_flat, **backbone)

        first_adapter_init = (
            'normal'
            if arch_type in ('tabm-mini-normal', 'tabm-normal')
            else 'random-signs'
        )

        if arch_type in ('tabm', 'tabm-normal'):
            assert first_adapter_init is not None
            deep.make_efficient_ensemble(
                self.backbone,
                deep.LinearEfficientEnsemble,
                k=k,
                ensemble_scaling_in=True,
                ensemble_scaling_out=True,
                ensemble_bias=True,
                scaling_init='ones',
            )
            _init_first_adapter(
                _get_first_ensemble_layer(self.backbone).r,
                first_adapter_init,
                first_adapter_sections,
            )

        elif arch_type in ('tabm-mini', 'tabm-mini-normal'):
            assert first_adapter_init is not None
            self.minimal_ensemble_adapter = deep.ScaleEnsemble(
                k,
                d_flat,
                init='random-signs',
            )
            _init_first_adapter(
                self.minimal_ensemble_adapter.weight,
                first_adapter_init,
                first_adapter_sections,
            )
        else:
            raise ValueError(f'Unknown arch_type: {arch_type}')

        # >>> Output
        d_block = backbone['d_block']
        d_out = 1 if n_classes is None else n_classes
        self.output = nn.Linear(d_block, d_out)

        # >>>
        self.arch_type = arch_type
        self.k = k
        self.share_training_batches = share_training_batches

    def forward(
        self, x_num: None | Tensor = None, x_cat: None | Tensor = None
    ) -> Tensor:
        x = []
        if x_num is not None:
            x.append(x_num)
        if x_cat is not None:
            x.append(x_cat.float())
        x = torch.column_stack([x_.flatten(1, -1) for x_ in x])

        if self.share_training_batches or not self.training:
            # (B, D) -> (B, K, D)
            x = x[:, None].expand(-1, self.k, -1)
        else:
            # (B * K, D) -> (B, K, D)
            x = x.reshape(len(x) // self.k, self.k, *x.shape[1:])
        if self.minimal_ensemble_adapter is not None:
            x = self.minimal_ensemble_adapter(x)

        x = self.backbone(x)
        x = self.output(x)
        return x



@torch.inference_mode()
def _init_first_adapter(
    weight: Tensor,
    distribution: Literal['normal', 'random-signs'],
    init_sections: list[int],
) -> None:
    assert weight.ndim == 2
    assert weight.shape[1] == len(init_sections)

    if distribution == 'normal':
        init_fn_ = nn.init.normal_
    elif distribution == 'random-signs':
        init_fn_ = deep.init_random_signs_
    else:
        raise ValueError(f'Unknown distribution: {distribution}')

    section_bounds = [0, *torch.tensor(init_sections).cumsum(0).tolist()]
    for i in range(len(init_sections)):
        w = torch.empty((len(weight), 1), dtype=weight.dtype, device=weight.device)
        init_fn_(w)
        weight[:, section_bounds[i] : section_bounds[i + 1]] = w

def _get_first_ensemble_layer(
    backbone: deep.MLP,
) -> deep.LinearEfficientEnsemble:
    if isinstance(backbone, deep.MLP):
        return backbone.blocks[0][0]
    else:
        raise RuntimeError(f'Unsupported backbone: {backbone}')
