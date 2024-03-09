#!/bin/bash

accelerate launch --num_cpu_threads_per_process 1 train_textual_inversion.py \
    --pretrained_model_name_or_path="/data/StableDiffusion/orangechillmix_v70.safetensors" \
    --dataset_config="/app/data/scz.toml" \
    --output_dir="/app/out/" \
    --output_name="ti_scz"\
    --save_model_as=safetensors \
    --prior_loss_weight=1.0 \
    --max_train_steps=3000 \
    --learning_rate=5e-6 \
    --optimizer_type="AdamW8bit" \
    --xformers \
    --mixed_precision="no" \
    --cache_latents \
    --gradient_checkpointing \
    --token_string=scz --init_word=library --num_vectors_per_token=8