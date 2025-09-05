#!/usr/bin/env bash

# Create terminal outputs directory
OUT_DIR=./eval_logs/terminal_outputs
mkdir -p "$OUT_DIR"

# run_and_log: run a command and tee stdout/stderr to a timestamped file that includes model_name and embedding_type
# Skip logging if output_dir starts with "test" or "debug"
run_and_log() {
	# Build args array
	args=("$@")
	model_name="unknown_model"
	embedding_type="unknown_embedding"
	output_dir=""
	for ((i=0;i<${#args[@]};i++)); do
		if [ "${args[$i]}" = "--model_name" ] && [ $((i+1)) -lt ${#args[@]} ]; then
			model_name="${args[$((i+1))]}"
		fi
		if [ "${args[$i]}" = "--embedding_type" ] && [ $((i+1)) -lt ${#args[@]} ]; then
			embedding_type="${args[$((i+1))]}"
		fi
		if [ "${args[$i]}" = "--output_dir" ] && [ $((i+1)) -lt ${#args[@]} ]; then
			output_dir="${args[$((i+1))]}"
		fi
	done

	# Check if output_dir contains folder starting with "test" or "debug"
	if [[ "$output_dir" =~ .*/test[^/]* ]] || [[ "$output_dir" =~ .*/debug[^/]* ]]; then
		echo "[run_and_log] Skipping logging for test/debug output dir: $output_dir"
		echo "[run_and_log] Running: ${args[*]}"
		("${args[@]}")
		return
	fi

	safe_model=$(echo "$model_name" | sed 's#[^A-Za-z0-9._-]#_#g' | sed 's#/$##')
	safe_embed=$(echo "$embedding_type" | sed 's#[^A-Za-z0-9._-]#_#g')
	ts=$(date +"%Y%m%d_%H%M%S")
	logfile="$OUT_DIR/${ts}_${safe_model}_${safe_embed}.log"

	echo "[run_and_log] $ts - Running: ${args[*]}"
	echo "[run_and_log] Logging to $logfile"

	("${args[@]}") 2>&1 | tee "$logfile"
}

# CUDA_VISIBLE_DEVICES=0 run_and_log python evaluations/safety_evals/test_safety.py --model_name models/rgtnet_llama-3.1-8b-instruct_20250807_2116/merged_epoch_0/ --embedding_type rgtnet_orthonly --datasets all --batch_size 32 --use_deepspeed 0 --output_dir ./eval_logs/rgtnet_orthonly &
# sleep 1
# CUDA_VISIBLE_DEVICES=1 run_and_log python evaluations/safety_evals/test_safety.py --model_name models/rgtnet_llama-3.1-8b-instruct_20250807_2116/merged_epoch_0/ --embedding_type rgtnet --datasets all --batch_size 32 --use_deepspeed 0 --output_dir ./eval_logs/rgtnet &
# sleep 1

# CUDA_VISIBLE_DEVICES=2 run_and_log python evaluations/safety_evals/test_safety.py --model_name models/rgtnet_llama-3.2-1b-instruct_20250807_1044/merged_epoch_0/ --embedding_type rgtnet_orthonly --datasets all --batch_size 32 --use_deepspeed 0 --output_dir ./eval_logs/rgtnet_orthonly &
# sleep 1
# CUDA_VISIBLE_DEVICES=3 run_and_log python evaluations/safety_evals/test_safety.py --model_name models/rgtnet_llama-3.2-1b-instruct_20250807_1044/merged_epoch_0/ --embedding_type rgtnet --datasets all --batch_size 32 --use_deepspeed 0 --output_dir ./eval_logs/rgtnet &
# sleep 1

# CUDA_VISIBLE_DEVICES=4 run_and_log python evaluations/safety_evals/test_safety.py --model_name models/rgtnet_llama-3.2-3b-instruct_20250803_1735/merged_epoch_0/ --embedding_type rgtnet_orthonly --datasets all --batch_size 32 --use_deepspeed 0 --output_dir ./eval_logs/rgtnet_orthonly &
# sleep 1
# CUDA_VISIBLE_DEVICES=5 run_and_log python evaluations/safety_evals/test_safety.py --model_name models/rgtnet_llama-3.2-3b-instruct_20250803_1735/merged_epoch_0/ --embedding_type rgtnet --datasets all --batch_size 32 --use_deepspeed 0 --output_dir ./eval_logs/rgtnet &
# sleep 1

# CUDA_VISIBLE_DEVICES=6 run_and_log python evaluations/safety_evals/test_safety.py --model_name models/rgtnet_gemma-7b_20250811_1447/merged_epoch_0/ --embedding_type rgtnet_orthonly --datasets all --batch_size 32 --use_deepspeed 0 --output_dir ./eval_logs/rgtnet_orthonly &
# sleep 1
# CUDA_VISIBLE_DEVICES=7 run_and_log python evaluations/safety_evals/test_safety.py --model_name models/rgtnet_gemma-7b_20250811_1447/merged_epoch_0/ --embedding_type rgtnet --datasets all --batch_size 32 --use_deepspeed 0 --output_dir ./eval_logs/rgtnet &
# sleep 1

# Gemma-2-27B: Reduce batch size due to large model size
# CUDA_VISIBLE_DEVICES=2 run_and_log python evaluations/safety_evals/test_safety.py --model_name models/rgtnet_qwen2.5-7b_20250814_0759/merged_epoch_1/ --embedding_type rgtnet_orthonly --datasets all --batch_size 4 --use_deepspeed 1 --output_dir ./eval_logs/rgtnet_orthonly &
# sleep 1
# CUDA_VISIBLE_DEVICES=3 run_and_log python evaluations/safety_evals/test_safety.py --model_name models/rgtnet_qwen2.5-7b_20250814_0759/merged_epoch_1/ --embedding_type rgtnet --datasets all --batch_size 4 --use_deepspeed 1 --output_dir ./eval_logs/rgtnet &
# sleep 1

CUDA_VISIBLE_DEVICES=2 run_and_log python evaluations/safety_evals/test_safety.py --model_name models/rgtnet_llama-3.2-1b-instruct_20250807_1044/merged_epoch_0/ --embedding_type single_emb --datasets all --batch_size 32 --use_deepspeed 0 --output_dir ./eval_logs/single_emb &
#sleep 1
CUDA_VISIBLE_DEVICES=3 run_and_log python evaluations/safety_evals/test_safety.py --model_name models/rgtnet_llama-3.2-1b-instruct_20250807_1044/merged_epoch_0/ --embedding_type ise --datasets all --batch_size 32 --use_deepspeed 0 --output_dir ./eval_logs/ise &
sleep 1

CUDA_VISIBLE_DEVICES=4 run_and_log python evaluations/safety_evals/test_safety.py --model_name models/rgtnet_llama-3.2-3b-instruct_20250803_1735/merged_epoch_0/ --embedding_type single_emb --datasets all --batch_size 32 --use_deepspeed 0 --output_dir ./eval_logs/single_emb &
sleep 1
CUDA_VISIBLE_DEVICES=5 run_and_log python evaluations/safety_evals/test_safety.py --model_name models/rgtnet_llama-3.2-3b-instruct_20250803_1735/merged_epoch_0/ --embedding_type ise --datasets all --batch_size 32 --use_deepspeed 0 --output_dir ./eval_logs/ise &
sleep 1