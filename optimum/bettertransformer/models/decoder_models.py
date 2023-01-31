# Copyright 2023 The HuggingFace and Meta Team.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.from typing import TYPE_CHECKING
import warnings
from typing import TYPE_CHECKING

import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import BetterTransformerBaseLayer


if TYPE_CHECKING:
    from transformers import PretrainedConfig


def _canonical_mask(
    mask,
    mask_name,
    other_type,
    other_name,
    target_type,
    check_other,
):
    if mask is not None:
        _mask_dtype = mask.dtype
        _mask_is_float = torch.is_floating_point(mask)
        if _mask_dtype != torch.bool and not _mask_is_float:
            raise AssertionError(f"only bool and floating types of {mask_name} are supported")
        if check_other and other_type is not None:
            if _mask_dtype != other_type:
                warnings.warn(
                    f"Support for mismatched {mask_name} and {other_name} "
                    "is deprecated. Use same type for both instead."
                )
        if not _mask_is_float:
            mask = torch.zeros_like(mask, dtype=target_type).masked_fill_(mask, float("-inf"))
    return mask


def _none_or_dtype(input):
    if input is None:
        return None
    elif isinstance(input, torch.Tensor):
        return input.dtype
    raise RuntimeError("input to _none_or_dtype() must be None or torch.Tensor")


class OPTAttentionLayerBetterTransformer(BetterTransformerBaseLayer):
    def __init__(self, opt_layer, config):
        r"""
        A simple conversion of the OPT Attention layer to its `BetterTransformer` implementation.

        Args:
            opt_layer (`torch.nn.Module`):
                The original OPT Layer where the weights needs to be retrieved.
        """
        super().__init__(config, opt_layer)

        self.embed_dim = config.hidden_size
        self.num_heads = config.num_attention_heads

        self.k_proj = opt_layer.k_proj
        self.v_proj = opt_layer.v_proj
        self.q_proj = opt_layer.q_proj

        self.in_proj_weight = nn.Parameter(
            torch.cat(
                [
                    opt_layer.q_proj.weight,
                    opt_layer.k_proj.weight,
                    opt_layer.v_proj.weight,
                ]
            )
        )
        self.in_proj_bias = nn.Parameter(
            torch.cat(
                [
                    opt_layer.q_proj.bias,
                    opt_layer.k_proj.bias,
                    opt_layer.v_proj.bias,
                ]
            )
        )
        self.out_proj = opt_layer.out_proj

    def merge_masks(self, attn_mask, key_padding_mask, query):
        r"""
        Determine mask type and combine masks if necessary. If only one mask is provided, that mask
        and the corresponding mask type will be returned. If both masks are provided, they will be both
        expanded to shape ``(batch_size, num_heads, seq_len, seq_len)``, combined with logical ``or``
        and mask type 2 will be returned
        Args:
            attn_mask: attention mask of shape ``(seq_len, seq_len)``, mask type 0
            key_padding_mask: padding mask of shape ``(batch_size, seq_len)``, mask type 1
            query: query embeddings of shape ``(batch_size, seq_len, embed_dim)``
        Returns:
            merged_mask: merged mask
            mask_type: merged mask type (0, 1, or 2)
        """

        attn_mask = _canonical_mask(
            mask=attn_mask,
            mask_name="attn_mask",
            other_type=_none_or_dtype(key_padding_mask),
            other_name="key_padding_mask",
            target_type=query.dtype,
            check_other=False,
        )

        if attn_mask is not None:
            mask_type = 0
            merged_mask = attn_mask
        if key_padding_mask is not None:
            mask_type = 1
            merged_mask = key_padding_mask
        if (attn_mask is not None) and (key_padding_mask is not None):
            # In this branch query can't be a nested tensor, so it has a shape
            batch_size, seq_len, _ = query.shape
            mask_type = 2
            key_padding_mask_expanded = key_padding_mask.view(batch_size, 1, 1, seq_len).expand(
                -1, self.num_heads, -1, -1
            )
            attn_mask_expanded = attn_mask.view(1, 1, seq_len, seq_len).expand(batch_size, self.num_heads, -1, -1)
            merged_mask = attn_mask_expanded + key_padding_mask_expanded
        return merged_mask, mask_type

    def forward(self, hidden_states, attention_mask, **__):
        query, key, value = self.q_proj(hidden_states), self.k_proj(hidden_states), self.v_proj(hidden_states)

        # convert `attention_mask` which is a causal mask to a simple mask
        causal_mask = attention_mask.clone()
        if len(attention_mask.shape) == 4:
            attention_mask = attention_mask.squeeze(1)[:, 0]

        # merged_mask, mask_type = self.merge_masks(causal_mask, attention_mask.bool(), query)
        multihead_attn = torch.nn.MultiheadAttention(self.embed_dim, self.num_heads, batch_first=True,  dtype=query.dtype)
        hidden_states, attn_output_weights = multihead_attn(query, key, value)
        
        return (hidden_states[0], None, None)
