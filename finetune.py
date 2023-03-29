"""
    llama-4b trainer with support of Stanford Alpaca-like JSON datasets (short for SAD)
    Intended to use with https://github.com/johnsmith0031/alpaca_lora_4bit

    SAD structure:
    [
        {
            "instruction": "Give null hypothesis",
            "input": "6 subjects were given a drug (treatment group) and an additional 6 subjects a placebo (control group).",
            "output": "Drug is equivalent of placebo"
        },
        {
            "instruction": "What does RNA stand for?",
            "input": "",
            "output": "RNA stands for ribonucleic acid."
        }
    ]
"""

import sys

import peft
import peft.tuners.lora
assert peft.tuners.lora.is_gptq_available()

import torch
import transformers
from datasets import load_dataset
from autograd_4bit import load_llama_model_4bit_low_ram
from peft import LoraConfig, get_peft_model, get_peft_model_state_dict, PeftModel

# ! Config
from arg_parser import get_config
import train_data


ft_config = get_config()




##################
def generate_prompt(data_point):
    # sorry about the formatting disaster gotta move fast
    return f"""
{data_point["func_documentation_string"]}
{data_point["whole_func_string"]}"""

def tokenize(prompt, add_eos_token=True):
    # there's probably a way to do this with the tokenizer settings
    # but again, gotta move fast
    result = tokenizer(
        prompt,
        truncation=True,
        max_length=ft_config.cutoff_len,
        padding=False,
        return_tensors=None,
    )
    if (
        result["input_ids"][-1] != tokenizer.eos_token_id
        and len(result["input_ids"]) < ft_config.cutoff_len
        and add_eos_token
    ):
        result["input_ids"].append(tokenizer.eos_token_id)
        result["attention_mask"].append(1)

    result["labels"] = result["input_ids"].copy()

    return result

def generate_and_tokenize_prompt(data_point):
    full_prompt = generate_prompt(data_point)
    tokenized_full_prompt = tokenize(full_prompt)
    #if not train_on_inputs:
    if True:
        user_prompt = generate_prompt({**data_point, "whole_func_string": ""})
        tokenized_user_prompt = tokenize(user_prompt, add_eos_token=False)
        user_prompt_len = len(tokenized_user_prompt["input_ids"])

        tokenized_full_prompt["labels"] = [
            -100
        ] * user_prompt_len + tokenized_full_prompt["labels"][
            user_prompt_len:
        ]  # could be sped up, probably
    return tokenized_full_prompt

def generate_and_tokenize_prompt_batch(examples):
    return [generate_and_tokenize_prompt(e) for e in examples]


###################

# * Show loaded parameters
if ft_config.local_rank == 0:
    print(f"{ft_config}\n")

if ft_config.gradient_checkpointing:
    print('Disable Dropout.')

# Load Basic Model
model, tokenizer = load_llama_model_4bit_low_ram(ft_config.llama_q4_config_dir,
                                                  ft_config.llama_q4_model,
                                                  device_map=ft_config.device_map,
                                                  groupsize=ft_config.groupsize)

# Config Lora
lora_config = LoraConfig(
    r=ft_config.lora_r,
    lora_alpha=ft_config.lora_alpha,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=ft_config.lora_dropout,
    bias="none",
    task_type="CAUSAL_LM",
)
if ft_config.lora_apply_dir is None:
    model = get_peft_model(model, lora_config)#.to(torch.bfloat16)
else:
    model = PeftModel.from_pretrained(model, ft_config.lora_apply_dir, device_map={'': 0}, torch_dtype=torch.float32)  # ! Direct copy from inference.py
    print(ft_config.lora_apply_dir, 'loaded')


# Scales to half
print('Fitting 4bit scales and zeros to half')
for n, m in model.named_modules():
    if '4bit' in str(type(m)):
        if m.groupsize == -1:
            m.zeros = m.zeros.half()
        m.scales = m.scales.half()

# Set tokenizer
tokenizer.pad_token_id = 0

if not ft_config.skip:
    # Load Data
    train_data = load_dataset("code_search_net", "go", split="train").shuffle().map(generate_and_tokenize_prompt, num_proc=30)
    #val_data = load_dataset("code_search_net", "go", split="validation").shuffle().map(generate_and_tokenize_prompt, num_proc=30)

    # Use gradient checkpointing
    if ft_config.gradient_checkpointing:
        print('Applying gradient checkpointing ...')
        from gradient_checkpointing import apply_gradient_checkpointing
        apply_gradient_checkpointing(model, checkpoint_ratio=ft_config.gradient_checkpointing_ratio)

    # Disable Trainer's DataParallel for multigpu
    if not ft_config.ddp and torch.cuda.device_count() > 1:
        model.is_parallelizable = True
        model.model_parallel = True

    trainer = transformers.Trainer(
        model=model,
        train_dataset=train_data,
        eval_dataset=None,
        args=transformers.TrainingArguments(
            per_device_train_batch_size=ft_config.mbatch_size,
            gradient_accumulation_steps=ft_config.gradient_accumulation_steps,
            warmup_steps=ft_config.warmup_steps,
            num_train_epochs=ft_config.epochs,
            learning_rate=ft_config.lr,
            #bf16=True,
            fp16=True,
            logging_steps=ft_config.logging_steps,
            evaluation_strategy="no",
            save_strategy="steps",
            eval_steps=None,
            save_steps=ft_config.save_steps,
            output_dir=ft_config.lora_out_dir,
            save_total_limit=ft_config.save_total_limit,
            load_best_model_at_end=False,
            ddp_find_unused_parameters=False if ft_config.ddp else None,
        ),
        data_collator=transformers.DataCollatorForSeq2Seq(
            tokenizer, pad_to_multiple_of=8, return_tensors="pt", padding=True
        ),
    )
    model.config.use_cache = False

    # Set Model dict
    old_state_dict = model.state_dict
    model.state_dict = (
        lambda self, *_, **__: get_peft_model_state_dict(self, old_state_dict())
    ).__get__(model, type(model))

    # Run Trainer
    trainer.train()

    print('Train completed.')

# Save Model
model.save_pretrained(ft_config.lora_out_dir)

if ft_config.checkpoint:
    print("Warning: Merge model + LoRA and save the whole checkpoint not implemented yet.")

print('Model Saved.')