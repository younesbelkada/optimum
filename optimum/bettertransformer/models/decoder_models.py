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


def wrapped_scaled_dot_product(query, key, value, attention_mask=None, head_mask=None):
    return torch.nn.functional.scaled_dot_product_attention(query, key, value, None, 0.0, True), None


class GPT2AttentionLayerBetterTransformer(BetterTransformerBaseLayer):
    def __init__(self, gpt_layer, config):
        super().__init__(config, gpt_layer)

        self.gpt_layer = gpt_layer
        self.gpt_layer._attn = wrapped_scaled_dot_product

    def forward(self, *args, **kwargs):
        return self.gpt_layer(*args, **kwargs)


class OPTAttentionLayerBetterTransformer(BetterTransformerBaseLayer):
    def __init__(self, opt_layer, config):
        r"""
        A simple conversion of the OPT Attention layer to its `BetterTransformer` implementation.

        Args:
            opt_layer (`torch.nn.Module`):
                The original OPT Layer where the weights needs to be retrieved.
        """
        super().__init__(config, opt_layer)

        self.embed_dim = opt_layer.embed_dim
        self.num_heads = opt_layer.num_heads
        self.dropout = opt_layer.dropout
        self.head_dim = self.embed_dim // self.num_heads

        if (self.head_dim * self.num_heads) != self.embed_dim:
            raise ValueError(
                f"embed_dim must be divisible by num_heads (got `embed_dim`: {self.embed_dim}"
                f" and `num_heads`: {self.num_heads})."
            )
        self.scaling = self.head_dim**-0.5
        self.is_decoder = opt_layer.is_decoder

        self.k_proj = opt_layer.k_proj
        self.v_proj = opt_layer.v_proj
        self.q_proj = opt_layer.q_proj
        self.out_proj = opt_layer.out_proj

    def _shape(self, tensor: torch.Tensor, seq_len: int, bsz: int):
        return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()

    def forward(
        self,
        hidden_states,
        key_value_states = None,
        past_key_value = None,
        attention_mask = None,
        layer_head_mask = None,
        output_attentions = False,
    ):
        """Input shape: Batch x Time x Channel"""
        # get query proj
        query_states = self.q_proj(hidden_states) * self.scaling
        value_states = self.v_proj(hidden_states)
        key_states = self.k_proj(hidden_states)

        attn_output = torch.nn.functional.scaled_dot_product_attention(query_states, key_states, value_states, None, 0.0, True)

        attn_output = self.out_proj(attn_output)

        return attn_output, None, past_key_value
