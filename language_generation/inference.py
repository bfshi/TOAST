#    Copyright 2023 Rohan Taori, Ishaan Gulrajani, Tianyi Zhang, Yann Dubois, Xuechen Li
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.

import os
import json
import copy
import logging
from dataclasses import dataclass, field
from typing import Dict, Optional, Sequence

import torch
import transformers
import utils
from torch.utils.data import Dataset
from transformers import Trainer
from transformers import LlamaForCausalLM, LlamaTokenizer
from transformers import GenerationConfig
from models.llama_top_down import LlamaTopDownForCausalLM
from models.llama_lora import LlamaLoRAForCausalLM
from models.llama_top_down_lora import LlamaTopDownLoRAForCausalLM

from fastchat.model.model_adapter import get_conversation_template

IGNORE_INDEX = -100
DEFAULT_PAD_TOKEN = "[PAD]"
DEFAULT_EOS_TOKEN = "</s>"
DEFAULT_BOS_TOKEN = "<s>"
DEFAULT_UNK_TOKEN = "<unk>"


@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="facebook/opt-125m")


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    output_dir: str = field(
        default="none",
        metadata={"help": "output path"},
    )


@dataclass
class InferenceArguments:
    model_max_length: int = field(
        default=1024,
        metadata={"help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."},
    )
    load_in_8bit: bool = field(
        default=False,
        metadata={"help": "Load the model in 8-bit mode."},
    )
    inference_dtype: torch.dtype = field(
        default=torch.float32,
        metadata={"help": "The dtype to use for inference."},
    )
    model: str = field(
        default="llama",  # choices: llama, llama-lora, llama-topdown, llama-topdown-lora
        metadata={"help": "If we are using top-down model"},
    )
    checkpoint: str = field(
        default="none",
        metadata={"help": "checkpoint for finetuned model (whole model / lora weights / top-down weights"},
    )


def smart_tokenizer_and_embedding_resize(
    special_tokens_dict: Dict,
    tokenizer: transformers.PreTrainedTokenizer,
    model: transformers.PreTrainedModel,
):
    """Resize tokenizer and embedding.

    Note: This is the unoptimized version that may make your embedding size not be divisible by 64.
    """
    num_new_tokens = tokenizer.add_special_tokens(special_tokens_dict)
    model.resize_token_embeddings(len(tokenizer))

    if num_new_tokens > 0:
        input_embeddings = model.get_input_embeddings().weight.data
        output_embeddings = model.get_output_embeddings().weight.data

        input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)
        output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)

        input_embeddings[-num_new_tokens:] = input_embeddings_avg
        output_embeddings[-num_new_tokens:] = output_embeddings_avg


def inference():
    parser = transformers.HfArgumentParser((ModelArguments, TrainingArguments, InferenceArguments))
    model_args, training_args, inference_args = parser.parse_args_into_dataclasses()

    if inference_args.model == 'llama':
        model = LlamaForCausalLM.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=training_args.cache_dir,
        )

    elif inference_args.model == 'llama-lora':
        model = LlamaLoRAForCausalLM.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=training_args.cache_dir,
        )
    elif inference_args.model == 'llama-topdown':
        model = LlamaTopDownForCausalLM.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=training_args.cache_dir,
        )
        model.var_loss_coef = 0
    elif inference_args.model == 'llama-topdown-lora':
        model = LlamaTopDownLoRAForCausalLM.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=training_args.cache_dir,
        )
        model.var_loss_coef = 0
    else:
        raise NotImplementedError

    if inference_args.checkpoint != 'none':
        state_dict = torch.load(inference_args.checkpoint)
        model.load_state_dict(state_dict, strict=False)

    tokenizer = LlamaTokenizer.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        model_max_length=inference_args.model_max_length,
        padding_side="right",
        use_fast=False,
        bos_token=DEFAULT_BOS_TOKEN,
        eos_token=DEFAULT_EOS_TOKEN,
        unk_token=DEFAULT_UNK_TOKEN,
    )

    special_tokens_dict = dict()
    if tokenizer.pad_token is None:
        special_tokens_dict["pad_token"] = DEFAULT_PAD_TOKEN
    if tokenizer.eos_token is None:
        special_tokens_dict["eos_token"] = DEFAULT_EOS_TOKEN
    if tokenizer.bos_token is None:
        special_tokens_dict["bos_token"] = DEFAULT_BOS_TOKEN
    if tokenizer.unk_token is None:
        special_tokens_dict["unk_token"] = DEFAULT_UNK_TOKEN

    smart_tokenizer_and_embedding_resize(
        special_tokens_dict=special_tokens_dict,
        tokenizer=tokenizer,
        model=model,
    )

    model.cuda()
    model.eval()

    generation_config = GenerationConfig(
        temperature=0.7,
        do_sample=True,
    )

    log = []
    for instruction in [
        # Generic
        "How can I improve my time management skills?",
        "What are the most effective ways to deal with stress?",
        "What are the main differences between Python and JavaScript programming languages?",
        "How can I increase my productivity while working from home?",
        "Can you explain the basics of quantum computing?",
        "What are the differences between plant-based and animal-based protein sources?",
        "How can I develop my critical thinking skills?",
        "What are the major challenges faced by the education sector today?",
        "What are the primary factors that influence consumer behavior?",
        "What are the most effective strategies for conflict resolution in the workplace?",

        # Knowledge
        "What are some potential implications of using a single-use plastic bottle versus a reusable bottle on both the environment and human health?",
        "What factors would you consider when designing an inclusive and accessible public transportation system?",
        "How can governments utilize fiscal and monetary policies to combat economic recessions?",

        # Roleplay
        "How would you introduce yourself as a medieval knight at a royal banquet?",
        "As a pirate captain, what would you say to your crew to motivate them to search for hidden treasure?",
        "If you were a Shakespearean character, how would you declare your love for someone in a soliloquy?",

        # Common-sense
        "How can you determine if a restaurant is popular among locals or mainly attracts tourists, and why might this information be useful?",
        "What are some subtle clues that suggest someone is pretending to understand a topic or conversation when they are actually confused or uninformed?",
        "Why might someone choose to use a paper map or ask for directions instead of relying on a GPS device or smartphone app?",

        # Fermi
        "How many times does the average human blink in a lifetime? Try to explain your answer. Your explanation should take the reader through your reasoning step-by-step.",
        "How many atoms are in a grain of salt? Try to explain your answer. Your explanation should take the reader through your reasoning step-by-step.",
        "How many lightning strikes occur on Earth each day? Try to explain your answer. Your explanation should take the reader through your reasoning step-by-step.",

        # Counterfactual
        "What if the Internet had been invented during the Renaissance period?",
        "What if the Aztecs had successfully repelled the Spanish conquistadors?",
        "What if the Black Death had not occurred in the 14th century?",

        # Writing
        "Can you help me write a formal email to a potential business partner proposing a joint venture?",
        "Can you help me write a resignation letter to my current employer, while leaving on good terms and expressing gratitude for the opportunities provided?",
        "Use an appropriate format to structure a formal letter of recommendation for a student applying to a prestigious graduate program in computer science.",
    ]:
        print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
        print("Instruction:", instruction)
        conv = get_conversation_template("vicuna")
        conv.append_message(conv.roles[0], instruction)
        conv.append_message(conv.roles[1], None)
        input = conv.get_prompt()
        inputs = tokenizer(input, return_tensors="pt")
        outputs = model.generate(input_ids=inputs["input_ids"].cuda(),
                                 generation_config=generation_config,
                                 max_new_tokens=inference_args.model_max_length,
                                 return_dict_in_generate=True,
                                 output_scores=True,
                                 eos_token_id=tokenizer.eos_token_id)
        # input_length is the length of the input prompt for decoder-only models, like the GPT family, and 1 for
        # encoder-decoder models, like BART or T5.
        input_length = 1 if model.config.is_encoder_decoder else inputs.input_ids.shape[1]
        generated_tokens = outputs.sequences[:, input_length:]
        response = tokenizer.decode(generated_tokens[0])

        print("Response:", response)
        print()

        log.append({"id": id, "instruction": instruction, "output": response})

    with open(os.path.join(training_args.output_dir, "output.json"), "w") as outfile:
        json.dump(log, outfile)

if __name__ == "__main__":
    inference()
