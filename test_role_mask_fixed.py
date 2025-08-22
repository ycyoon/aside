#!/usr/bin/env python3
"""
Test script to verify RGTNet role mask generation
"""
import os
import sys
sys.path.append('/home/ycyoon/work/aside/experiments')

import torch
from model_api import CustomModelHandler

def test_role_mask():
    # Test RGTNet role mask generation
    print("Testing RGTNet role mask generation...")
    
    try:
        # Test RGTNet 
        print("=== Testing RGTNet ===")
        handler_rgtnet = CustomModelHandler(
            checkpoint_path="experiments/models/rgtnet_llama-3.2-1b-instruct_20250807_1044/merged_epoch_0/",
            instruct_model_path=None,
            data_model_path=None,
            tokenizer_path="meta-llama/Llama-3.2-1B-Instruct",
            embedding_type="rgtnet",
            max_token_len=512,
            model_dtype=torch.bfloat16
        )
        
        # Test RGTNet Orthonly
        print("=== Testing RGTNet Orthonly ===")
        handler_orthonly = CustomModelHandler(
            checkpoint_path="experiments/models/rgtnet_llama-3.2-1b-instruct_20250807_1044/merged_epoch_0/",
            instruct_model_path=None,
            data_model_path=None,
            tokenizer_path="meta-llama/Llama-3.2-1B-Instruct",
            embedding_type="rgtnet_orthonly",
            max_token_len=512,
            model_dtype=torch.bfloat16
        )
        
        # Test with simple inputs
        system_instructions = ["You are a helpful assistant. Follow the instruction below."]
        user_instructions = ["What is 2+2?"]
        
        print("=== Comparing responses ===")
        responses_rgtnet, _ = handler_rgtnet.call_model_api_batch(
            system_instructions, 
            user_instructions, 
            max_new_tokens=20,
            do_sample=True,
            temperature=0.7
        )
        
        responses_orthonly, _ = handler_orthonly.call_model_api_batch(
            system_instructions, 
            user_instructions, 
            max_new_tokens=20,
            do_sample=True,
            temperature=0.7
        )
        
        print(f"RGTNet response: {responses_rgtnet[0]}")
        print(f"RGTNet Orthonly response: {responses_orthonly[0]}")
        print(f"Responses are identical: {responses_rgtnet[0] == responses_orthonly[0]}")
        
        print("Role mask test completed successfully!")
        
    except Exception as e:
        print(f"Error during role mask test: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_role_mask()
