# coding=utf-8
# Copyright 2022 The HuggingFace Team. All rights reserved.
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
# limitations under the License.
import unittest

import numpy as np
import torch
from transformers import AutoFeatureExtractor

from testing_bettertransformer_utils import BetterTransformersTestMixin


ALL_AUDIO_MODELS_TO_TEST = [
    "openai/whisper-tiny",
]


class BetterTransformersAudioTest(BetterTransformersTestMixin, unittest.TestCase):
    r""" """
    all_models_to_test = ALL_AUDIO_MODELS_TO_TEST

    def _generate_random_audio_data(self):
        np.random.seed(10)
        t = np.linspace(0, 5.0, int(5.0 * 22050), endpoint=False)
        # generate pure sine wave at 220 Hz
        audio_data = 0.5 * np.sin(2 * np.pi * 220 * t)
        return audio_data

    def prepare_inputs_for_class(self, model_id):
        input_audio = self._generate_random_audio_data()

        feature_extractor = AutoFeatureExtractor.from_pretrained(model_id)

        input_dict = {
            "input_features": feature_extractor(input_audio, return_tensors="pt").input_features,
            "decoder_input_ids": torch.LongTensor([0]),
        }
        return input_dict
