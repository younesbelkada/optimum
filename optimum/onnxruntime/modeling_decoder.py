#  Copyright 2022 The HuggingFace Team. All rights reserved.
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
"""Classes handling causal-lm related architectures in ONNX Runtime."""

import logging
import shutil
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple, Union

import torch
from huggingface_hub import hf_hub_download
from huggingface_hub.utils import EntryNotFoundError
from transformers import AutoModelForCausalLM, GenerationConfig
from transformers.file_utils import add_start_docstrings_to_model_forward
from transformers.modeling_outputs import CausalLMOutputWithCrossAttentions

import onnxruntime

from ..exporters import TasksManager
from ..exporters.onnx import export_models, get_decoder_models_for_export
from ..onnx.utils import _get_external_data_paths
from ..utils import check_if_transformers_greater
from ..utils.file_utils import validate_file_exists
from ..utils.save_utils import maybe_load_preprocessors, maybe_save_preprocessors
from .base import ORTDecoder
from .modeling_ort import ORTModel
from .utils import ONNX_DECODER_NAME, ONNX_DECODER_WITH_PAST_NAME, get_provider_for_device, parse_device


if TYPE_CHECKING:
    from transformers import PretrainedConfig


if check_if_transformers_greater("4.25.0"):
    from transformers.generation import GenerationMixin
else:
    from transformers.generation_utils import GenerationMixin


logger = logging.getLogger(__name__)

DECODER_INPUTS_DOCSTRING = r"""
    Args:
        input_ids (`torch.LongTensor`):
            Indices of decoder input sequence tokens in the vocabulary of shape `(batch_size, sequence_length)`.
        attention_mask (`torch.LongTensor`, *optional*):
            Mask to avoid performing attention on padding token indices, of shape
            `(batch_size, sequence_length)`. Mask values selected in `[0, 1]`.
        past_key_values (`tuple(tuple(torch.FloatTensor), *optional*, defaults to `None`)`
            Contains the precomputed key and value hidden states of the attention blocks used to speed up decoding.
            The tuple is of length `config.n_layers` with each tuple having 2 tensors of shape
            `(batch_size, num_heads, sequence_length, embed_size_per_head)`.
"""

CAUSALLM_ONNX_MODEL_DOCSTRING = r"""
    Args:
        input_ids (`torch.LongTensor`):
            Indices of decoder input sequence tokens in the vocabulary of shape `(batch_size, sequence_length)`.
        attention_mask (`torch.LongTensor`):
            Mask to avoid performing attention on padding token indices, of shape
            `(batch_size, sequence_length)`. Mask values selected in `[0, 1]`.
        past_key_values (`tuple(tuple(torch.FloatTensor), *optional*, defaults to `None`)`
            Contains the precomputed key and value hidden states of the attention blocks used to speed up decoding.
            The tuple is of length `config.n_layers` with each tuple having 2 tensors of shape
            `(batch_size, num_heads, sequence_length, embed_size_per_head)`.
"""

_TOKENIZER_FOR_DOC = "AutoTokenizer"

TEXT_GENERATION_EXAMPLE = r"""
    Example of text generation:

    ```python
    >>> from transformers import {processor_class}
    >>> from optimum.onnxruntime import {model_class}
    >>> import torch

    >>> tokenizer = {processor_class}.from_pretrained("{checkpoint}")
    >>> model = {model_class}.from_pretrained("{checkpoint}")

    >>> inputs = tokenizer("My name is Arthur and I live in", return_tensors="pt")

    >>> gen_tokens = model.generate(**inputs,do_sample=True,temperature=0.9, min_length=20,max_length=20)
    >>> tokenizer.batch_decode(gen_tokens)  # doctest: +IGNORE_RESULT
    ```

    Example using `transformers.pipelines`:

    ```python
    >>> from transformers import {processor_class}, pipeline
    >>> from optimum.onnxruntime import {model_class}

    >>> tokenizer = {processor_class}.from_pretrained("{checkpoint}")
    >>> model = {model_class}.from_pretrained("{checkpoint}")
    >>> onnx_gen = pipeline("text-generation", model=model, tokenizer=tokenizer)

    >>> text = "My name is Arthur and I live in"
    >>> gen = onnx_gen(text)
    ```
"""

DECODER_ONNX_FILE_PATTERN = r"(.*)?decoder((?!with_past).)*?\.onnx"
DECODER_WITH_PAST_ONNX_FILE_PATTERN = r"(.*)?decoder(.*)?with_past(.*)?\.onnx"


class ORTModelDecoder(ORTModel):
    """
    Base class for implementing models with a causal language modeling head using ONNX Runtime inference.
    """

    def __init__(
        self,
        decoder_session: onnxruntime.InferenceSession,
        config: "PretrainedConfig",
        decoder_with_past_session: Optional[onnxruntime.InferenceSession] = None,
        use_io_binding: Optional[bool] = None,
        model_save_dir: Optional[Union[str, Path, TemporaryDirectory]] = None,
        preprocessors: Optional[List] = None,
        generation_config: Optional[GenerationConfig] = None,
        **kwargs,
    ):
        """
        Args:
            decoder_session (`onnxruntime.InferenceSession`):
                The ONNX Runtime inference session associated to the decoder.
            config ([~`transformers.PretrainedConfig`]):
                An instance of the configuration associated to the model. Initializing with a config file does
                not load the weights associated with the model, only the configuration.
            decoder_with_past_session (`Optional[onnxruntime.InferenceSession]`, *optional*):
                The ONNX Runtime inference session associated to the decoder with past key values.
            use_io_binding (`Optional[bool]`, defaults to `None`):
                Whether use IOBinding during inference to avoid memory copy between the host and devices. Defaults to
                `True` if the device is CUDA, otherwise defaults to `False`.
            model_save_dir (`str`, *optional*, defaults to `""`):
                The directory under which the model exported to ONNX was saved.
            preprocessors (`Optional[List]`, defaults to `None`):
                The list of the preprocessors (tokenizer, processor, feature_extractor) to save alongside the ORTModel.
            generation_config (`Optional[GenerationConfig]`, defaults to `None`):
                The generation configuration used by default when calling `generate()`.
                Refer to https://huggingface.co/docs/transformers/main/en/main_classes/text_generation#transformers.GenerationMixin.generate.
        """
        self.shared_attributes_init(
            decoder_session,
            use_io_binding=use_io_binding,
            model_save_dir=model_save_dir,
        )
        self.config = config

        # TODO: remove at version 2.0
        def show_deprecated_argument(arg_name):
            if kwargs.pop(arg_name, None) is not None:
                logger.warning(
                    f"The {arg_name} argument to create an {self.__class__.__name__} is deprecated, and not used "
                    "anymore."
                )

        show_deprecated_argument("last_decoder_model_name")
        show_deprecated_argument("last_decoder_with_past_model_name")
        if kwargs:
            raise ValueError(
                f"{self.__class__.__name__} received {', '.join(kwargs.keys())}, but do not accept those arguments."
            )

        self.use_cache = decoder_with_past_session is not None
        self.decoder = ORTDecoder(decoder_session, self)
        self.decoder_model_path = Path(decoder_session._model_path)
        self.decoder_model_name = self.decoder_model_path.name

        self.decoder_with_past = None
        self.decoder_with_past_model_path = None
        self.decoder_with_past_model_name = None
        if self.use_cache:
            self.decoder_with_past = ORTDecoder(decoder_with_past_session, self)
            self.decoder_with_past_model_path = Path(decoder_with_past_session._model_path)
            self.decoder_with_past_model_name = self.decoder_with_past_model_path.name

        if generation_config is None:
            generation_config = GenerationConfig.from_model_config(config)
        self.generation_config = generation_config

    @staticmethod
    def load_model(
        decoder_path: Union[str, Path],
        decoder_with_past_path: Optional[Union[str, Path]] = None,
        provider: str = "CPUExecutionProvider",
        session_options: Optional[onnxruntime.SessionOptions] = None,
        provider_options: Optional[Dict] = None,
    ):
        """
        Creates an instance of [`~optimum.onnxruntime.ORTModelDecoder`].
        Three inference sessions will be created for respectively the decoder and decoder with past key values
        models. The default provider is `CPUExecutionProvider` to match the default behaviour in PyTorch/TensorFlow/JAX.

        Args:
            decoder_path (`str` or `Path`):
                The path of the decoder ONNX model.
            decoder_with_past_path (`str` or `Path`, *optional*):
                The path of the decoder with past key values ONNX model.
            provider(`str`, *optional*, defaults to `"CPUExecutionProvider"`):
                The ONNX Runtime provider to use for loading the model.
            session_options (`Optional[onnxruntime.SessionOptions]`, *optional*),:
                ONNX Runtime session options to use for loading the model.
            provider_options (`Optional[Dict]`, *optional*):
                Provider option dictionary corresponding to the provider used. See available options
                for each provider: https://onnxruntime.ai/docs/api/c/group___global.html.
        """
        decoder_session = ORTModel.load_model(decoder_path, provider, session_options, provider_options)

        decoder_with_past_session = None
        # If a decoder_with_past_path is provided, an inference session for the decoder with past key/values as inputs
        # will be enabled
        if decoder_with_past_path is not None:
            decoder_with_past_session = ORTModel.load_model(
                decoder_with_past_path, provider, session_options, provider_options
            )

        return decoder_session, decoder_with_past_session

    def _save_pretrained(
        self,
        save_directory: Union[str, Path],
        decoder_file_name: str = ONNX_DECODER_NAME,
        decoder_with_past_file_name: str = ONNX_DECODER_WITH_PAST_NAME,
        **kwargs,
    ):
        """
        Saves the model decoder and decoder with past key values as well as its configuration file to a
        directory, so that it can be re-loaded using the
        [`~optimum.onnxruntime.modeling_causal.ORTModelDecoder.from_pretrained`] class method.

        Args:
            save_directory (`str` or `Path`):
                The directory where to save the model files.
            decoder_file_name (`str`, *optional*, defaults to `optimum.onnxruntime.utils.ONNX_DECODER_NAME`):
                The decoder model file name. Overwrites the default file name and allows one to save the decoder model
                with a different name.
            decoder_with_past_file_name (`str`, *optional*, defaults to `optimum.onnxruntime.utils.ONNX_DECODER_WITH_PAST_NAME`):
                The decoder with past key values model file name overwriting the default file name, allowing to save
                the decoder model with a different name.
        """
        src_paths = [self.decoder_model_path]
        dst_file_names = [decoder_file_name]

        if self.use_cache:
            src_paths.append(self.decoder_with_past_model_path)
            dst_file_names.append(decoder_with_past_file_name)

        # add external data paths in case of large models
        src_paths, dst_file_names = _get_external_data_paths(src_paths, dst_file_names)

        for src_path, dst_file_name in zip(src_paths, dst_file_names):
            dst_path = Path(save_directory) / dst_file_name
            shutil.copyfile(src_path, dst_path)

    @classmethod
    def _from_pretrained(
        cls,
        model_id: Union[str, Path],
        config: "PretrainedConfig",
        use_auth_token: Optional[Union[bool, str]] = None,
        revision: Optional[str] = None,
        force_download: bool = False,
        cache_dir: Optional[str] = None,
        decoder_file_name: str = ONNX_DECODER_NAME,
        decoder_with_past_file_name: str = ONNX_DECODER_WITH_PAST_NAME,
        subfolder: str = "",
        local_files_only: bool = False,
        use_cache: bool = True,
        provider: str = "CPUExecutionProvider",
        session_options: Optional[onnxruntime.SessionOptions] = None,
        provider_options: Optional[Dict[str, Any]] = None,
        use_io_binding: Optional[bool] = None,
        model_save_dir: Optional[Union[str, Path, TemporaryDirectory]] = None,
        **kwargs,
    ):
        model_path = Path(model_id)

        if not validate_file_exists(model_id, decoder_file_name, subfolder=subfolder, revision=revision):
            decoder_path = ORTModelDecoder.infer_onnx_filename(
                model_id,
                DECODER_ONNX_FILE_PATTERN,
                "decoder_file_name",
                subfolder=subfolder,
                use_auth_token=use_auth_token,
                revision=revision,
            )
        else:
            decoder_path = model_path / subfolder / decoder_file_name
        decoder_regular_onnx_filenames = ORTModelDecoder._generate_regular_names_for_filename(ONNX_DECODER_NAME)
        if decoder_path.name not in decoder_regular_onnx_filenames:
            logger.warning(
                f"The ONNX file {decoder_path.name} is not a regular name used in optimum.onnxruntime that are {decoder_regular_onnx_filenames}, the "
                f"{cls.__name__} might not behave as expected."
            )

        decoder_with_past_path = None
        if use_cache is True:
            if not validate_file_exists(model_id, decoder_with_past_file_name, subfolder=subfolder, revision=revision):
                try:
                    decoder_with_past_path = ORTModelDecoder.infer_onnx_filename(
                        model_id,
                        DECODER_WITH_PAST_ONNX_FILE_PATTERN,
                        "decoder_with_past_file_name",
                        subfolder=subfolder,
                        use_auth_token=use_auth_token,
                        revision=revision,
                    )
                except FileNotFoundError as e:
                    raise FileNotFoundError(
                        "The parameter `use_cache=True` was passed to ORTModelDecoder.from_pretrained()"
                        " but no ONNX file using past key values could be found in"
                        f" {str(Path(model_id, subfolder))}, with the error:\n    {e}"
                    )
            else:
                decoder_with_past_path = model_path / subfolder / decoder_with_past_file_name

            decoder_with_past_regular_onnx_filenames = ORTModelDecoder._generate_regular_names_for_filename(
                ONNX_DECODER_WITH_PAST_NAME
            )

            if (
                decoder_with_past_path is not None
                and decoder_with_past_path.name not in decoder_with_past_regular_onnx_filenames
            ):
                logger.warning(
                    f"The ONNX file {decoder_with_past_path.name} is not a regular name used in optimum.onnxruntime that are {decoder_with_past_regular_onnx_filenames}, "
                    f"the {cls.__name__} might not behave as expected."
                )

            decoder_with_past_path = decoder_with_past_path if use_cache else None

        preprocessors = None
        if model_path.is_dir():
            model = cls.load_model(
                decoder_path=decoder_path,
                decoder_with_past_path=decoder_with_past_path,
                provider=provider,
                session_options=session_options,
                provider_options=provider_options,
            )
            new_model_save_dir = model_path
            preprocessors = maybe_load_preprocessors(model_id)
        else:
            attribute_name_to_filename = {
                "last_decoder_model_name": decoder_path.name,
                "last_decoder_with_past_model_name": decoder_with_past_path.name if use_cache else None,
            }
            paths = {}
            for attr_name, filename in attribute_name_to_filename.items():
                if filename is None:
                    continue
                model_cache_path = hf_hub_download(
                    repo_id=model_id,
                    subfolder=subfolder,
                    filename=filename,
                    use_auth_token=use_auth_token,
                    revision=revision,
                    cache_dir=cache_dir,
                    force_download=force_download,
                    local_files_only=local_files_only,
                )

                # try download external data
                try:
                    hf_hub_download(
                        repo_id=model_id,
                        subfolder=subfolder,
                        filename=filename + "_data",
                        use_auth_token=use_auth_token,
                        revision=revision,
                        cache_dir=cache_dir,
                        force_download=force_download,
                        local_files_only=local_files_only,
                    )
                except EntryNotFoundError:
                    # model doesn't use external data
                    pass

                paths[attr_name] = Path(model_cache_path).name
            new_model_save_dir = Path(model_cache_path).parent
            preprocessors = maybe_load_preprocessors(model_id, subfolder=subfolder)

            last_decoder_with_past_name = paths.get("last_decoder_with_past_model_name", None)
            if last_decoder_with_past_name is not None:
                last_decoder_with_past_name = new_model_save_dir / last_decoder_with_past_name

            model = cls.load_model(
                decoder_path=new_model_save_dir / paths["last_decoder_model_name"],
                decoder_with_past_path=last_decoder_with_past_name,
                provider=provider,
                session_options=session_options,
                provider_options=provider_options,
            )

        if model_save_dir is None:
            model_save_dir = new_model_save_dir

        generation_config = None
        try:
            generation_config = GenerationConfig.from_pretrained(
                model_id,
                cache_dir=cache_dir,
                force_download=force_download,
                local_files_only=local_files_only,
                use_auth_token=use_auth_token,
                revision=revision,
                subfolder=subfolder,
            )
        except OSError:
            logger.info("Generation config file not found, using a generation config created from the model config.")

        return cls(
            model[0],
            config,
            decoder_with_past_session=model[1],
            use_io_binding=use_io_binding,
            model_save_dir=model_save_dir,
            preprocessors=preprocessors,
            generation_config=generation_config,
        )

    @classmethod
    def _from_transformers(
        cls,
        model_id: str,
        config: "PretrainedConfig",
        use_auth_token: Optional[Union[bool, str]] = None,
        revision: str = "main",
        force_download: bool = True,
        cache_dir: Optional[str] = None,
        subfolder: str = "",
        local_files_only: bool = False,
        trust_remote_code: bool = False,
        use_cache: bool = True,
        provider: str = "CPUExecutionProvider",
        session_options: Optional[onnxruntime.SessionOptions] = None,
        provider_options: Optional[Dict[str, Any]] = None,
        use_io_binding: Optional[bool] = None,
        task: Optional[str] = None,
    ) -> "ORTModelDecoder":
        if task is None:
            task = cls._auto_model_to_task(cls.auto_model_class)

        save_dir = TemporaryDirectory()
        save_dir_path = Path(save_dir.name)

        model = TasksManager.get_model_from_task(
            task,
            model_id,
            subfolder=subfolder,
            revision=revision,
            cache_dir=cache_dir,
            config=config,
            use_auth_token=use_auth_token,
            local_files_only=local_files_only,
            force_download=force_download,
            trust_remote_code=trust_remote_code,
        )

        onnx_config_constructor = TasksManager.get_exporter_config_constructor(model=model, exporter="onnx", task=task)
        onnx_config = onnx_config_constructor(model.config, use_past=use_cache)

        output_names = [ONNX_DECODER_NAME]
        if use_cache is True:
            output_names.append(ONNX_DECODER_WITH_PAST_NAME)

        models_and_onnx_configs = get_decoder_models_for_export(model, onnx_config)
        export_models(
            models_and_onnx_configs=models_and_onnx_configs,
            opset=onnx_config.DEFAULT_ONNX_OPSET,
            output_dir=save_dir_path,
            output_names=output_names,
        )

        config.save_pretrained(save_dir_path)
        maybe_save_preprocessors(model_id, save_dir_path, src_subfolder=subfolder)

        return cls._from_pretrained(
            save_dir_path,
            config,
            use_cache=use_cache,
            provider=provider,
            session_options=session_options,
            provider_options=provider_options,
            use_io_binding=use_io_binding,
            model_save_dir=save_dir,
        )

    def to(self, device: Union[torch.device, str, int]):
        """
        Changes the ONNX Runtime provider according to the device.

        Args:
            device (`Union[torch.device, str, int]`):
                Device ordinal for CPU/GPU supports. Setting this to -1 will leverage CPU, a positive will run
                the model on the associated CUDA device id. You can pass native `torch.device` or a `str` too.

        Returns:
            `ORTModel`: the model placed on the requested device.
        """
        device, provider_options = parse_device(device)

        provider = get_provider_for_device(device)
        self.device = device
        self.decoder.session.set_providers([provider], provider_options=[provider_options])
        if self.decoder_with_past is not None:
            self.decoder_with_past.session.set_providers([provider], provider_options=[provider_options])
        self.providers = self.decoder.session.get_providers()

        return self


class ORTModelForCausalLM(ORTModelDecoder, GenerationMixin):
    """
    ONNX model with a causal language modeling head for ONNX Runtime inference.
    """

    auto_model_class = AutoModelForCausalLM
    main_input_name = "input_ids"

    @add_start_docstrings_to_model_forward(
        CAUSALLM_ONNX_MODEL_DOCSTRING.format("batch_size, sequence_length")
        + TEXT_GENERATION_EXAMPLE.format(
            processor_class=_TOKENIZER_FOR_DOC,
            model_class="ORTModelForCausalLM",
            checkpoint="optimum/gpt2",
        )
    )
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,
        labels: Optional[torch.LongTensor] = None,
        **kwargs,
    ) -> CausalLMOutputWithCrossAttentions:
        if past_key_values is None or self.decoder_with_past is None:
            outputs = self.decoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
            )
        else:
            outputs = self.decoder_with_past(
                input_ids=input_ids[:, -1:],
                past_key_values=past_key_values,
                attention_mask=attention_mask,
                labels=labels,
            )

        return CausalLMOutputWithCrossAttentions(
            loss=outputs.get("loss", None), logits=outputs.logits, past_key_values=outputs.past_key_values
        )

    # Adapted from transformers.models.gpt2.modeling_gpt2.GPT2LMHeadModel.prepare_inputs_for_generation
    def prepare_inputs_for_generation(self, input_ids, past_key_values=None, **kwargs):
        # if model is used as a decoder in encoder-decoder model, the decoder attention mask is created on the fly

        attention_mask = kwargs.get("attention_mask", None)  # input_ids.new_ones(input_ids.shape)
        use_cache = kwargs.get("use_cache", None)

        return {
            "input_ids": input_ids,
            "past_key_values": past_key_values,
            "use_cache": use_cache,
            "position_ids": None,
            "attention_mask": attention_mask,
            "token_type_ids": None,
        }

    # Copied from transformers.models.gpt2.modeling_gpt2.GPT2LMHeadModel._reorder_cache
    @staticmethod
    def _reorder_cache(past: Tuple[Tuple[torch.Tensor]], beam_idx: torch.Tensor) -> Tuple[Tuple[torch.Tensor]]:
        return tuple(
            tuple(past_state.index_select(0, beam_idx.to(past_state.device)) for past_state in layer_past)
            for layer_past in past
        )

    def can_generate(self):
        """Returns True to validate the check that the model using `GenerationMixin.generate()` can indeed generate."""
        return True
