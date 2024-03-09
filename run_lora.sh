#!/bin/bash

accelerate launch --num_cpu_threads_per_process 1 train_network.py \
    --pretrained_model_name_or_path="/models/checkpoints/chilloutmix_NiPrunedFp32Fix.safetensors" \
    --dataset_config="data/spo-15.toml" \
    --output_dir="/app/out/" \
    --output_name="spo" \
    --save_model_as=safetensors \
    --prior_loss_weight=1.0 \
    --max_train_steps=1200 \
    --learning_rate=1e-4 \
    --optimizer_type="AdamW8bit" \
    --mixed_precision="no" \
    --cache_latents \
    --gradient_checkpointing \
    --save_every_n_epochs=10 \
    --xformers \
    --network_module=networks.lora \
    --network_dim 5
