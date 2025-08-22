
# StruQ evaluations for each model
echo "Starting StruQ evaluations..."

# Llama-3.1-8b
CUDA_VISIBLE_DEVICES=0 python evaluations/struq/test_on_struq.py --model models/rgtnet_llama-3.1-8b-instruct_20250807_2116/merged_epoch_0/ --embedding_type rgtnet_orthonly --output_dir ./eval_logs &
CUDA_VISIBLE_DEVICES=1 python evaluations/struq/test_on_struq.py --model models/rgtnet_llama-3.1-8b-instruct_20250807_2116/merged_epoch_0/ --embedding_type rgtnet --output_dir ./eval_logs &

# Llama-3.2-1b
CUDA_VISIBLE_DEVICES=2 python evaluations/struq/test_on_struq.py --model models/rgtnet_llama-3.2-1b-instruct_20250807_1044/merged_epoch_0/ --embedding_type rgtnet_orthonly --output_dir ./eval_logs &
CUDA_VISIBLE_DEVICES=3 python evaluations/struq/test_on_struq.py --model models/rgtnet_llama-3.2-1b-instruct_20250807_1044/merged_epoch_0/ --embedding_type rgtnet --output_dir ./eval_logs &

# Llama-3.2-3b
CUDA_VISIBLE_DEVICES=4 python evaluations/struq/test_on_struq.py --model models/rgtnet_llama-3.2-3b-instruct_20250803_1735/merged_epoch_0/ --embedding_type rgtnet_orthonly --output_dir ./eval_logs &
CUDA_VISIBLE_DEVICES=5 python evaluations/struq/test_on_struq.py --model models/rgtnet_llama-3.2-3b-instruct_20250803_1735/merged_epoch_0/ --embedding_type rgtnet --output_dir ./eval_logs &

# Gemma-7b
CUDA_VISIBLE_DEVICES=6 python evaluations/struq/test_on_struq.py --model models/rgtnet_gemma-7b_20250811_1447/merged_epoch_0/ --embedding_type rgtnet_orthonly --output_dir ./eval_logs &
CUDA_VISIBLE_DEVICES=7 python evaluations/struq/test_on_struq.py --model models/rgtnet_gemma-7b_20250811_1447/merged_epoch_0/ --embedding_type rgtnet --output_dir ./eval_logs &

# Gemma-2-27b (commented out like safety evals)
# CUDA_VISIBLE_DEVICES=2 python evaluations/struq/test_on_struq.py --model models/rgtnet_gemma-2-27b_20250810_2328/merged_epoch_0/ --embedding_type rgtnet_orthonly --output_dir ./eval_logs &
# CUDA_VISIBLE_DEVICES=3 python evaluations/struq/test_on_struq.py --model models/rgtnet_gemma-2-27b_20250810_2328/merged_epoch_0/ --embedding_type rgtnet --output_dir ./eval_logs &

wait
echo "All evaluations completed."