#!/bin/bash
# animagineXLV3_v30
# dreamshaperXL_sfwTurboDpmppSDE
# realvisxlV40_v40Bakedvae

accelerate launch --num_cpu_threads_per_process 1 sd-scripts/sdxl_train.py \
    --pretrained_model_name_or_path="/models/checkpoints/dreamshaperXL_v21TurboDPMSDE.safetensors" \
    --dataset_config="/app/data/girl.toml" \
    --output_dir="/app/out/" \
    --output_name="girl" \
    --save_model_as=safetensors \
    --max_train_steps=400 \
    --learning_rate=1e-4 \
    --lr_scheduler=cosine_with_restarts \
    --optimizer_type="AdamW" \
    --cache_latents \
    --cache_text_encoder_outputs \
    --gradient_checkpointing \
    --logging_dir="/app/logs/" \
    --log_with="tensorboard" \
    --mixed_precision="bf16" \
    --sdpa

#    --mixed_precision="fp16" \
