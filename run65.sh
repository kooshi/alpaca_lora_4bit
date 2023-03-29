#!/bin/bash

sudo nvidia-smi -i 0 -pl 350
sudo nvidia-smi -i 1 -pl 350

CUDA_VISIBLE_DEVICES="0,1" python finetune.py --lora_out_dir lora-golang --llama_q4_model ./models/LLaMA-HF-4bit-128g/llama-65b-4bit-128g/llama-65b-4bit-128g.safetensors --llama_q4_config_dir models/llama-65b \
	--mbatch_size 1 \
	--batch_size 1 \
	--epochs 1 \
	--cutoff_len 1024 \
	--grad_chckpt True \
	--groupsize 128
