#!/bin/bash

sudo nvidia-smi -i 0 -pl 380
sudo nvidia-smi -i 1 -pl 380

CUDA_VISIBLE_DEVICES="0,1" python finetune.py \
	--lora_out_dir lora-golang-30 \
	--llama_q4_model ./models/LLaMA-HF-4bit-128g/llama-30b-4bit-128g/llama-30b-4bit-128g.safetensors \
	--llama_q4_config_dir models/llama-30b \
	--mbatch_size 4 \
	--batch_size 128 \
	--epochs 1 \
	--grad_chckpt \
	--cutoff_len 1024 \
	--groupsize 128 \
	--save_steps 1 \
	--logging_steps 1 \
	--warmup_steps 20
