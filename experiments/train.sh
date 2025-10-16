export TORCH_DISTRIBUTED_ENABLE_DTENSOR=0
export TORCH_DISABLE_DYNAMO=1

deepspeed --num_gpus=8 fine-tune.py \
	--model_family llama3.2_3b_instruction \
	--config_path /home/ycyoon/work/aside/experiments/configs/config_llama_3_2_3b_SFTv110.json \
	--train_version SFTv110 \
	--emb_type forward_rot \
	--model_ix 0 \
	--run_number ASIDE \
	--train_type full \
	--num_train_epochs 2 \
	--per_device_train_batch_size 4 \
	--gradient_accumulation_steps 8 \
	--learning_rate 2e-5 \
	--lr_scheduler_type cosine \
	--warmup_ratio 0 \
	--logging_steps 10 \
	--evaluation_strategy epoch \
	--save_strategy epoch \
	--eval_steps 1 \
	--save_steps 1 \
	--save_total_limit 1 \
	--load_best_model_at_end True \
	--prediction_loss_only True \
	--bf16 True \
	--embedding_init rot_isoclinic \
	--rotation_alpha 1.57079633 \
	--learned_rotation False \
	--add_linear_shift False \
	--rotation_direction right \
	--gradual_rotation False

