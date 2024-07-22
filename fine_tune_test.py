import datasets, dataclasses
import os, json

from transformers import TrainingArguments
import transformers
from peft import PeftModel

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    GPT2Tokenizer,
    GPTNeoXTokenizerFast,
    LlamaTokenizer,
    LlamaTokenizerFast,
)

from trl import SFTConfig, SFTTrainer, DataCollatorForCompletionOnlyLM
from peft import PromptTuningConfig

from typing import Dict
import math, torch
os.environ["TOKENIZERS_PARALLELISM"] = "True"
os.environ["TORCH_COMPILE_DEBUG"] = "True"

model_name = "Maykeye/TinyLLama-v0"
# model_name = "/granite/granite-13b-base-v2/step_300000_ckpt"
tempdir = "tmp"
use_flash_attn = False

model = AutoModelForCausalLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)


def tokenizer_and_embedding_resize(
    special_tokens_dict: Dict,
    tokenizer: transformers.PreTrainedTokenizer,
    model: transformers.PreTrainedModel,
    multiple_of: int = 8,
):
    """Resize tokenizer and embedding."""
    num_new_tokens = tokenizer.add_special_tokens(special_tokens_dict)
    embedding_size = int(multiple_of * math.ceil(len(tokenizer) / multiple_of))
    num_new_tokens = num_new_tokens + embedding_size - len(tokenizer)
    model.resize_token_embeddings(embedding_size)
    if num_new_tokens > 0:
        input_embeddings = model.get_input_embeddings().weight.data
        output_embeddings = model.get_output_embeddings().weight.data

        input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(
            dim=0, keepdim=True
        )
        output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(
            dim=0, keepdim=True
        )

        input_embeddings[-num_new_tokens:] = input_embeddings_avg
        output_embeddings[-num_new_tokens:] = output_embeddings_avg

if isinstance(tokenizer, (LlamaTokenizer, LlamaTokenizerFast)):
    tokenizer.add_special_tokens(
        {
            "bos_token": "<s>",
            "eos_token": "</s>",
            "unk_token": "<unk>",
            "pad_token": "<pad>",
        }
    )
elif isinstance(tokenizer, (GPT2Tokenizer, GPTNeoXTokenizerFast)):
    tokenizer.add_special_tokens(
        {
            "pad_token": "<pad>",
        }
    )

special_tokens_dict = {}
if tokenizer.pad_token is None:
    special_tokens_dict["pad_token"] = "<PAD>"
if tokenizer.eos_token is None:
    special_tokens_dict["eos_token"] = "</s>"
if tokenizer.bos_token is None:
    special_tokens_dict["bos_token"] = "<s>"
if tokenizer.unk_token is None:
    special_tokens_dict["unk_token"] = "<unk>"

tokenizer_and_embedding_resize(
    special_tokens_dict=special_tokens_dict,
    tokenizer=tokenizer,
    model=model,
    multiple_of=8,
)

#################### DATA PREPROCESSING ###################

response_template = "\n### Label:"
response_template_ids = tokenizer.encode(
    response_template, add_special_tokens=False
)[2:]

data_collator = DataCollatorForCompletionOnlyLM(
    response_template_ids,
    tokenizer=tokenizer,
    ignore_index= -100,
)

format_dataset = lambda example: {  
    "output": example["output"] + tokenizer.eos_token
}

# data_files = {"train": "/Users/abhishekm/Documents/Codes/fms-hf-tuning/tests/data/twitter_complaints_small.json"}
data_files = {"train": "/data/abhishek/datafile/twitter_complaints.json"}

json_dataset = datasets.load_dataset("json", data_files=data_files)

formatted_train_dataset = json_dataset["train"].map(format_dataset)

##########################################################

################# Trainer CONFIG ############################

train_args = TrainingArguments(
    num_train_epochs=3,
    # use_cpu=True,
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    gradient_accumulation_steps=1,
    learning_rate=0.01,
    weight_decay=0,
    warmup_ratio=0.03,
    lr_scheduler_type="cosine",
    logging_steps=1,
    include_tokens_per_second=True,
    save_strategy="epoch",
    output_dir=tempdir,
    torch_compile=True,
    torch_compile_backend="inductor",
    torch_compile_mode="default"
)
transformer_train_arg_fields = [x.name for x in dataclasses.fields(SFTConfig)]
transformer_kwargs = {
    k: v
    for k, v in train_args.to_dict().items()
    if k in transformer_train_arg_fields
}
training_args = SFTConfig(**transformer_kwargs)
print("training_args", training_args)

##########################################################

####################### TUNING #####################

trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=formatted_train_dataset,
    packing=False,
    data_collator=data_collator,
    dataset_text_field="output",
    # args=training_args,
    args=train_args,
    max_seq_length=2048,
)
trainer.train()

print("### MODEL TUNING DONE !!")