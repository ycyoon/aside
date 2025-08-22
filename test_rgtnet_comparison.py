#!/usr/bin/env python3
"""
Test script to compare RGTNet vs RGTNet_orthonly role mask generation
"""
import os
import sys
sys.path.append('/home/ycyoon/work/aside/experiments')

import torch
from model_api import CustomModelHandler

def test_both_rgtnet_variants():
    print("Testing RGTNet vs RGTNet_orthonly role mask generation...")
    
    try:
        # Test regular RGTNet
        print("\n=== Testing RGTNet ===")
        handler_rgtnet = CustomModelHandler(
            checkpoint_path="/home/ycyoon/work/RGTNet/models/rgtnet_llama-3.2-1b-instruct_20250807_1044/merged_epoch_0/",
            instruct_model_path=None,
            data_model_path=None,
            tokenizer_path="meta-llama/Llama-3.2-1B-Instruct",
            embedding_type="rgtnet",
            max_token_len=512,
            model_dtype=torch.bfloat16
        )
        
        # Test RGTNet orthonly
        print("\n=== Testing RGTNet_orthonly ===")
        handler_orthonly = CustomModelHandler(
            checkpoint_path="/home/ycyoon/work/RGTNet/models/rgtnet_llama-3.2-1b-instruct_20250807_1044/merged_epoch_0/",
            instruct_model_path=None,
            data_model_path=None,
            tokenizer_path="meta-llama/Llama-3.2-1B-Instruct",
            embedding_type="rgtnet_orthonly",
            max_token_len=512,
            model_dtype=torch.bfloat16
        )
        
        # Test with the same inputs
        system_instructions = ["You are a helpful assistant. Follow the instruction below carefully."]
        user_instructions = ["Input:\nCalculate 2+2 and explain your reasoning."]
        
        print("\n=== RGTNet Response ===")
        responses_rgtnet = handler_rgtnet.call_model_api_batch(
            system_instructions, user_instructions, max_new_tokens=50
        )
        print("Response:", responses_rgtnet[0])
        
        print("\n=== RGTNet_orthonly Response ===")
        responses_orthonly = handler_orthonly.call_model_api_batch(
            system_instructions, user_instructions, max_new_tokens=50
        )
        print("Response:", responses_orthonly[0])
        
        print("\n=== Comparison ===")
        if responses_rgtnet[0] == responses_orthonly[0]:
            print("⚠️  SAME RESPONSE - Role masking is NOT working properly!")
        else:
            print("✅ DIFFERENT RESPONSES - Role masking appears to be working!")
        
    except Exception as e:
        print(f"Error during role mask comparison test: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_both_rgtnet_variants()
