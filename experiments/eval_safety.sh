#python evaluations/safety_evals/test_safety.py --model_name ./models/Qwen2.5-7B/forward_rot/train_checkpoints/SFTv70/from_inst_run_ASIDE/last/ --embedding_type forward_rot --base_model Qwen/Qwen2.5-7B --datasets all --batch_size 32 --use_deepspeed 0
#python evaluations/safety_evals/test_safety.py --model_name models/rgtnet_qwen2.5-7b_20250814_0759/merged_epoch_1/ --embedding_type rgtnet --datasets all --batch_size 32 --use_deepspeed 0 --output_dir ./eval_logs/rgtnet
# Example orth-only (no explicit role_mask growth) run:
# python evaluations/safety_evals/test_safety.py --model_name models/rgtnet_qwen2.5-7b_20250814_0759/merged_epoch_1/ --embedding_type rgtnet_orthonly --datasets all --batch_size 32 --use_deepspeed 0 --output_dir ./eval_logs/rgtnet_orthonly

CUDA_VISIBLE_DEVICES=0 python evaluations/safety_evals/test_safety.py --model_name models/rgtnet_llama-3.1-8b-instruct_20250807_2116/merged_epoch_0/ --embedding_type rgtnet_orthonly --datasets all --batch_size 32 --use_deepspeed 0 --output_dir ./eval_logs/rgtnet_orthonly &
CUDA_VISIBLE_DEVICES=1 python evaluations/safety_evals/test_safety.py --model_name models/rgtnet_llama-3.1-8b-instruct_20250807_2116/merged_epoch_0/ --embedding_type rgtnet --datasets all --batch_size 32 --use_deepspeed 0 --output_dir ./eval_logs/rgtnet  &

CUDA_VISIBLE_DEVICES=2 python evaluations/safety_evals/test_safety.py --model_name models/rgtnet_llama-3.2-1b-instruct_20250807_1044/merged_epoch_0/ --embedding_type rgtnet_orthonly --datasets all --batch_size 32 --use_deepspeed 0 --output_dir ./eval_logs/rgtnet_orthonly  &
CUDA_VISIBLE_DEVICES=3 python evaluations/safety_evals/test_safety.py --model_name models/rgtnet_llama-3.2-1b-instruct_20250807_1044/merged_epoch_0/ --embedding_type rgtnet --datasets all --batch_size 32 --use_deepspeed 0 --output_dir ./eval_logs/rgtnet  &

CUDA_VISIBLE_DEVICES=4 python evaluations/safety_evals/test_safety.py --model_name models/rgtnet_llama-3.2-3b-instruct_20250803_1735/merged_epoch_0/ --embedding_type rgtnet_orthonly --datasets all --batch_size 32 --use_deepspeed 0 --output_dir ./eval_logs/rgtnet_orthonly  &
CUDA_VISIBLE_DEVICES=5 python evaluations/safety_evals/test_safety.py --model_name models/rgtnet_llama-3.2-3b-instruct_20250803_1735/merged_epoch_0/ --embedding_type rgtnet --datasets all --batch_size 32 --use_deepspeed 0 --output_dir ./eval_logs/rgtnet  &

CUDA_VISIBLE_DEVICES=6 python evaluations/safety_evals/test_safety.py --model_name models/rgtnet_gemma-7b_20250811_1447/merged_epoch_0/ --embedding_type rgtnet_orthonly --datasets all --batch_size 32 --use_deepspeed 0 --output_dir ./eval_logs/rgtnet_orthonly &
CUDA_VISIBLE_DEVICES=7 python evaluations/safety_evals/test_safety.py --model_name models/rgtnet_gemma-7b_20250811_1447/merged_epoch_0/ --embedding_type rgtnet --datasets all --batch_size 32 --use_deepspeed 0 --output_dir ./eval_logs/rgtnet  &

# CUDA_VISIBLE_DEVICES=2 python evaluations/safety_evals/test_safety.py --model_name models/rgtnet_gemma-2-27b_20250810_2328/merged_epoch_0/ --embedding_type rgtnet_orthonly --datasets all --batch_size 32 --use_deepspeed 0 --output_dir ./eval_logs/rgtnet_orthonly &
# CUDA_VISIBLE_DEVICES=3 python evaluations/safety_evals/test_safety.py --model_name models/rgtnet_gemma-2-27b_20250810_2328/merged_epoch_0/ --embedding_type rgtnet --datasets all --batch_size 32 --use_deepspeed 0 --output_dir ./eval_logs/rgtnet  &
