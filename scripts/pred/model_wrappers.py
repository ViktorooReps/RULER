# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
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

import json
import logging
import nltk
import requests
import random
import torch

from nltk.tokenize import word_tokenize
from typing import Dict, List, Optional


class TestModel:
    def __init__(self):
        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            nltk.download('punkt')

    def __call__(self, prompt: str, **kwargs) -> Dict[str, List[str]]:
        tokens = word_tokenize(prompt)
        return {'text': [random.choice(tokens)]}

    def process_batch(self, prompts: List[str], **kwargs) -> List[dict]:
        return [self.__call__(prompt, **kwargs) for prompt in prompts]


class HuggingFaceModel:
    def __init__(self, name_or_path: str, revision: str, **generation_kwargs) -> None:
        from transformers import pipeline, AutoModelForCausalLM

        self.setup_tokenizer(name_or_path, revision=revision, padding_side='left', truncation=True)

        self.generation_kwargs = generation_kwargs
        self.stop = self.generation_kwargs.pop('stop') + [self.tokenizer.eos_token]

        if 'Yarn-Llama' in name_or_path:
            model_kwargs = None
        else:
            model_kwargs = {"attn_implementation": "flash_attention_2"}

        print(f'Model init: {name_or_path}')

        try:
            self.pipeline = pipeline(
                "text-generation",
                model=name_or_path,
                tokenizer=self.tokenizer,
                trust_remote_code=True,
                device_map="auto",
                torch_dtype=torch.bfloat16,
                model_kwargs=model_kwargs,
            )
        except Exception as e:
            logging.warning('Failed to create the pipeline!', exc_info=e)
            self.pipeline = None
            self.model = AutoModelForCausalLM.from_pretrained(
                name_or_path, 
                trust_remote_code=True,
                device_map="auto", 
                torch_dtype=torch.bfloat16
            )

    def setup_tokenizer(self, name_or_path: str, **tokenizer_kwargs):
        from transformers import AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(name_or_path, **tokenizer_kwargs)

        if self.tokenizer.pad_token is None:
            # add pad token to allow batching (known issue for llama2)
            self.tokenizer.padding_side = 'left'
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        
    def __call__(self, prompt: str, **kwargs) -> dict:
        return self.process_batch([prompt], **kwargs)[0]

    def process_batch(self, prompts: List[str], **kwargs) -> List[dict]:
        if self.pipeline is None:
            inputs = self.tokenizer(prompts, return_tensors="pt").to(self.model.device)
            output = self.model.generate(
                **inputs,
                **self.generation_kwargs
            )
            generated_texts = []
            output_start = inputs.input_ids.shape[1]  # shape: (B, L)
            for llm_result in output:
                text = self.tokenizer.decode(llm_result[output_start:], skip_special_tokens=True)
                generated_texts.append(text)
        else:
            output = self.pipeline(text_inputs=prompts, **self.generation_kwargs)
            assert len(output) == len(prompts)
            generated_texts = [llm_result[0]["generated_text"] for llm_result in output]

        results = []

        for text, prompt in zip(generated_texts, prompts):
            # remove the input form the generated text
            if text.startswith(prompt):
                text = text[len(prompt):]

            if self.stop is not None:
                for s in self.stop:
                    text = text.split(s)[0]

            results.append({'text': [text]})

        return results
    

class LandmarkAttentionModel(HuggingFaceModel):
    def __init__(self, name_or_path: str, revision: str, **generation_kwargs) -> None:
        from llama_mem import LlamaForCausalLM
        from transformers import pipeline

        self.setup_tokenizer(name_or_path, revision=revision, truncation=False, padding_side="right", padding=True, use_fast=False)

        self.generation_kwargs = generation_kwargs
        self.stop = self.generation_kwargs.pop('stop') + [self.tokenizer.eos_token]

        model = LlamaForCausalLM.from_pretrained(
            name_or_path, 
            revision=revision, 
            torch_dtype=torch.bfloat16, 
            device_map="auto"
        )

        if torch.cuda.device_count() > 1: 
            from accelerate import load_checkpoint_and_dispatch

            model = load_checkpoint_and_dispatch(model, checkpoint=name_or_path, device_map="auto")

        mem_id = self.tokenizer.convert_tokens_to_ids("<landmark>")
        model.set_mem_id(mem_id)

        # using flash for inference is only implemented for when offloading kv to cpu
        self.pipeline = pipeline(
            "text-generation",
            model=model,
            device_map="auto",
            tokenizer=self.tokenizer,
            offload_cache_to_cpu=generation_kwargs.get('offload_cache_to_cpu', False),
            use_flash=generation_kwargs.get('use_flash', True),
            cache_top_k=generation_kwargs.get('cache_top_k', 5)
        )

        print(f'Generation kwargs: {generation_kwargs}')



class MambaModel:
    def __init__(self, name_or_path: str, **generation_kwargs) -> None:
        from transformers import AutoTokenizer
        from mamba_ssm.models.mixer_seq_simple import MambaLMHeadModel

        self.tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neox-20b")
        self.device = "cuda"
        self.model = MambaLMHeadModel.from_pretrained(name_or_path, device=self.device, dtype=torch.bfloat16)
        self.generation_kwargs = generation_kwargs
        self.stop = self.generation_kwargs.pop('stop')
        self.max_genlen = self.generation_kwargs.pop('max_new_tokens')
        self.minp = 0.0

    def __call__(self, prompt: str, **kwargs) -> Dict[str, List[str]]:
        # tokenize
        tokens = self.tokenizer(prompt, return_tensors="pt")
        input_ids = tokens.input_ids.to(self.device)
        max_length = input_ids.shape[1] + self.max_genlen

        # generate
        out = self.model.generate(
            input_ids=input_ids,
            max_length=max_length,
            cg=True,
            return_dict_in_generate=True,
            output_scores=True,
            enable_timing=False,
            **self.generation_kwargs,
        )
        assert len(out.sequences) == 1
        # detok
        return {'text': [self.tokenizer.decode(out.sequences[0][input_ids.shape[1]:])]}

    def process_batch(self, prompts: List[str], **kwargs) -> List[dict]:
        # FIXME: naive implementation
        return [self.__call__(prompt, **kwargs) for prompt in prompts]
