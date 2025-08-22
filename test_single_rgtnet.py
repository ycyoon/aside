#!/usr/bin/env python3
"""
Test script to run a single RGTNet model.
"""
import os
import sys
sys.path.append('/home/ycyoon/work/aside/experiments')

import torch
from model_api import CustomModelHandler

def test_single_rgtnet_model():
    """
    Loads and tests a single RGTNet model to verify its functionality.
    """
    model_path = "/home/ycyoon/work/aside/experiments/models/rgtnet_qwen2.5-7b_20250814_0759/merged_epoch_1"
    tokenizer_path = "Qwen/Qwen2.5-7B-Instruct" # Assuming this is the correct tokenizer for the model
    
    print(f"--- Testing RGTNet Model ---")
    print(f"Model Path: {model_path}")
    print(f"Tokenizer Path: {tokenizer_path}")
    
    try:
        handler = CustomModelHandler(
            checkpoint_path=model_path,
            instruct_model_path=None,
            data_model_path=None,
            tokenizer_path=tokenizer_path,
            embedding_type="rgtnet",
            max_token_len=512,
            model_dtype=torch.bfloat16
        )
        
        system_instructions = ["You are a helpful assistant. Follow the instruction below carefully."]
        user_instructions = ["Write a short story about a robot who discovers music."]
        
        print("\n--- Generating Response ---")
        responses = handler.call_model_api_batch(
            system_instructions, user_instructions, max_new_tokens=100
        )
        
        print("\n--- Model Response ---")
        print(responses[0])
        print("\n--- Test Complete ---")
        
    except Exception as e:
        print(f"An error occurred during the test: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_single_rgtnet_model()
