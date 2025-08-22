#!/usr/bin/env python3
"""
Test to verify if differences are only due to temperature
"""
import os
import sys
sys.path.append('/home/ycyoon/work/aside/experiments')

import torch
from model_api import CustomModelHandler

def test_temperature_effect():
    print("Testing if differences are only due to temperature...")
    
    try:
        # Test with same model but different temperatures
        handler1 = CustomModelHandler(
            checkpoint_path="experiments/models/rgtnet_llama-3.2-1b-instruct_20250807_1044/merged_epoch_0/",
            instruct_model_path=None,
            data_model_path=None,
            tokenizer_path="meta-llama/Llama-3.2-1B-Instruct",
            embedding_type="rgtnet",  # Same model type
            max_token_len=512,
            model_dtype=torch.bfloat16
        )
        
        system_instructions = ["You are a helpful assistant. Follow the instruction below."]
        user_instructions = ["What is 2+2?"]
        
        print("=== Testing same model with different temperatures ===")
        
        # Test 1: Temperature 0.7
        responses1, _ = handler1.call_model_api_batch(
            system_instructions, 
            user_instructions, 
            max_new_tokens=20,
            do_sample=True,
            temperature=0.7
        )
        
        # Test 2: Temperature 0.7 * 1.05 = 0.735 (same as orthonly)
        responses2, _ = handler1.call_model_api_batch(
            system_instructions, 
            user_instructions, 
            max_new_tokens=20,
            do_sample=True,
            temperature=0.7 * 1.05
        )
        
        print(f"Temperature 0.7:   {responses1[0]}")
        print(f"Temperature 0.735: {responses2[0]}")
        print(f"Responses are identical: {responses1[0] == responses2[0]}")
        
        # Test with deterministic generation (no sampling)
        print("\n=== Testing with deterministic generation ===")
        
        responses3, _ = handler1.call_model_api_batch(
            system_instructions, 
            user_instructions, 
            max_new_tokens=20,
            do_sample=False  # Deterministic
        )
        
        responses4, _ = handler1.call_model_api_batch(
            system_instructions, 
            user_instructions, 
            max_new_tokens=20,
            do_sample=False  # Deterministic
        )
        
        print(f"Deterministic 1: {responses3[0]}")
        print(f"Deterministic 2: {responses4[0]}")
        print(f"Deterministic responses identical: {responses3[0] == responses4[0]}")
        
    except Exception as e:
        print(f"Error during test: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_temperature_effect()
