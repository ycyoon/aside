import torch
import os

# Path to the merged model directory from the previous command
model_dir = '/home/ycyoon/work/aside/experiments/models/rgtnet_llama-3.2-1b-instruct_20250807_1044/merged_epoch_0/'
checkpoint_path = os.path.join(model_dir, 'pytorch_model.bin')

if not os.path.exists(checkpoint_path):
    # Look for other possible sharded checkpoint files
    found_files = [f for f in os.listdir(model_dir) if f.endswith('.bin')]
    if not found_files:
        print(f"Error: No '.bin' checkpoint files found in {model_dir}")
        exit()
    # Just use the first one found for inspection
    checkpoint_path = os.path.join(model_dir, found_files[0])
    print(f"Found sharded checkpoint: {checkpoint_path}")


print(f"Loading checkpoint from: {checkpoint_path}")
try:
    state_dict = torch.load(checkpoint_path, map_location='cpu')
    print("Checkpoint loaded successfully. Keys are:")
    for key in state_dict.keys():
        print(key)
except Exception as e:
    print(f"Failed to load checkpoint: {e}")
