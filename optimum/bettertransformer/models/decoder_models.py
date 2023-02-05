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
        super().__init__(config, opt_layer)

        self.opt_layer = opt_layer
        self.opt_layer._attn = wrapped_scaled_dot_product

    def forward(self, *args, **kwargs):
        return self.opt_layer(*args, **kwargs)
