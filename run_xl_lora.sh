#!/bin/bash
# animagineXLV3_v30
# dreamshaperXL_v21TurboDPMSDE
# realvisxlV40_v40Bakedvae

accelerate launch --num_cpu_threads_per_process 1 sdxl_train_network.py \
    --pretrained_model_name_or_path="/models/checkpoints/dreamshaperXL_v21TurboDPMSDE.safetensors" \
    --dataset_config="/app/data/kkrn.toml" \
    --output_dir="/app/out/" \
    --output_name="spo" \
    --save_model_as=safetensors \
    --max_train_steps=1500 \
    --learning_rate=1e-4 \
    --lr_scheduler=cosine_with_restarts \
    --lr_scheduler_num_cycles=2 \
    --mixed_precision="fp16" \
    --cache_latents \
    --cache_text_encoder_outputs \
    --gradient_checkpointing \
    --save_every_n_epochs=10 \
    --network_train_unet_only \
    --bucket_reso_steps 32 \
    --network_dim 24 \
    --optimizer_type="AdamW" \
    --sdpa \
    --logging_dir="/app/logs/" \
    --log_with="tensorboard" \
    --network_module=networks.lora \
    --network_args "conv_dim=4" "conv_alpha=1"

#    --xformers \
#    --optimizer_type="adafactor" \
#    --optimizer_args "scale_parameter=False" "relative_step=False" "warmup_init=False" \
#    --lr_scheduler=cosine_with_restart \
#    --lr_scheduler_num_cycles=1 \
#    --lr_scheduler=polynomial \
#    --lr_scheduler_power=0.5 \
#    --lr_warmup_steps=100 \
#    --unet_lr=0.5 \
