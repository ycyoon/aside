#!/usr/bin/env python3
"""
Multiple runs test to verify consistent differences
"""
import os
import sys
sys.path.append('/home/ycyoon/work/aside/experiments')

import torch
from model_api import CustomModelHandler

def test_multiple_runs():
    print("Testing multiple runs to verify consistent differences...")
    
    try:
        # Test RGTNet 
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
        handler_orthonly = CustomModelHandler(
            checkpoint_path="experiments/models/rgtnet_llama-3.2-1b-instruct_20250807_1044/merged_epoch_0/",
            instruct_model_path=None,
            data_model_path=None,
            tokenizer_path="meta-llama/Llama-3.2-1B-Instruct",
            embedding_type="rgtnet_orthonly",
            max_token_len=512,
            model_dtype=torch.bfloat16
        )
        
        # Test with multiple examples
        system_instructions = [
            "You are a helpful assistant. Follow the instruction below.",
            "Please answer the following question.",
            "Solve this problem step by step."
        ]
        user_instructions = [
            "What is 5+3?",
            "How do you make coffee?", 
            "What is the capital of France?"
        ]
        
        print("=== Testing multiple examples ===")
        
        differences = 0
        total_tests = len(system_instructions)
        
        for i in range(total_tests):
            sys_inst = [system_instructions[i]]
            usr_inst = [user_instructions[i]]
            
            responses_rgtnet, _ = handler_rgtnet.call_model_api_batch(
                sys_inst, usr_inst, max_new_tokens=30, do_sample=True, temperature=0.8
            )
            
            responses_orthonly, _ = handler_orthonly.call_model_api_batch(
                sys_inst, usr_inst, max_new_tokens=30, do_sample=True, temperature=0.8
            )
            
            identical = responses_rgtnet[0] == responses_orthonly[0]
            if not identical:
                differences += 1
                
            print(f"\nTest {i+1}: {usr_inst[0]}")
            print(f"RGTNet:     {responses_rgtnet[0]}")
            print(f"Orthonly:   {responses_orthonly[0]}")
            print(f"Identical:  {identical}")
        
        print(f"\n=== Summary ===")
        print(f"Total tests: {total_tests}")
        print(f"Different responses: {differences}")
        print(f"Difference rate: {differences/total_tests*100:.1f}%")
        
        if differences > 0:
            print("✅ SUCCESS: Models now produce different outputs!")
        else:
            print("❌ ISSUE: Models still produce identical outputs")
        
    except Exception as e:
        print(f"Error during test: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_multiple_runs()
