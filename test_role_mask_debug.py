#!/usr/bin/env python3
"""
Test to verify role mask generation and application
"""
import os
import sys
sys.path.append('/home/ycyoon/work/aside/experiments')

import torch
from model_api import CustomModelHandler

def test_role_mask_debug():
    print("Testing role mask generation and application...")
    
    try:
        # Test RGTNet with detailed debug output
        print("=== Testing RGTNet Role Mask ===")
        handler_rgtnet = CustomModelHandler(
            checkpoint_path="experiments/models/rgtnet_llama-3.2-1b-instruct_20250807_1044/merged_epoch_0/",
            instruct_model_path=None,
            data_model_path=None,
            tokenizer_path="meta-llama/Llama-3.2-1B-Instruct",
            embedding_type="rgtnet",
            max_token_len=512,
            model_dtype=torch.bfloat16
        )
        
        # Use a clear instruction/data split
        system_instructions = ["You are a helpful assistant. Follow the instruction below."]
        user_instructions = ["What is 2+2?"]  # This should be split at "Input:"
        
        print("Testing with single example...")
        responses, _ = handler_rgtnet.call_model_api_batch(
            system_instructions, 
            user_instructions, 
            max_new_tokens=5,  # Short to reduce output
            do_sample=False   # Deterministic for consistent testing
        )
        
        print(f"Response: {responses[0]}")
        
        print("\n=== Testing RGTNet Orthonly ===")
        handler_orthonly = CustomModelHandler(
            checkpoint_path="experiments/models/rgtnet_llama-3.2-1b-instruct_20250807_1044/merged_epoch_0/",
            instruct_model_path=None,
            data_model_path=None,
            tokenizer_path="meta-llama/Llama-3.2-1B-Instruct",
            embedding_type="rgtnet_orthonly",
            max_token_len=512,
            model_dtype=torch.bfloat16
        )
        
        responses2, _ = handler_orthonly.call_model_api_batch(
            system_instructions, 
            user_instructions, 
            max_new_tokens=5,
            do_sample=False
        )
        
        print(f"Response: {responses2[0]}")
        
    except Exception as e:
        print(f"Error during test: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_role_mask_debug()
