#!/usr/bin/env python3
"""
Test the orthogonal vs regular RGTNet behavior with role-aware input modifications.
"""

import os
import sys
import torch
from model_api import CustomModelHandler

def test_orthogonal_difference():
    """Test that rgtnet and rgtnet_orthonly produce different outputs with role-aware modifications."""
    
    print("=== Testing Orthogonal vs Regular RGTNet ===")
    
    # Test models
    model_paths = {
        'rgtnet': '/home/ycyoon/work/aside/experiments/models/rgtnet_qwen2.5-7b_20250814_0759',
        'rgtnet_orthonly': '/home/ycyoon/work/aside/experiments/models/rgtnet_qwen2.5-7b_20250814_0759_orthonly'
    }
    
    # Test prompt with clear instruction/data separation
    prompt = """Please analyze the following code:

def factorial(n):
    if n <= 1:
        return 1
    return n * factorial(n-1)

What is the time complexity?"""
    
    handlers = {}
    
    for model_name, model_path in model_paths.items():
        if not os.path.exists(model_path):
            print(f"[WARNING] Model path not found: {model_path}")
            continue
            
        print(f"\n--- Loading {model_name} ---")
        try:
            handler = CustomModelHandler(
                checkpoint_path=model_path,
                instruct_model_path=None,
                data_model_path=None,
                tokenizer_path=model_path,  # Use same path
                embedding_type="single_emb",
                model_dtype=torch.bfloat16,
            )
            
            handlers[model_name] = handler
            print(f"[SUCCESS] Loaded {model_name}")
            
        except Exception as e:
            print(f"[ERROR] Failed to load {model_name}: {e}")
            continue
    
    if len(handlers) < 2:
        print("[ERROR] Need both models to compare!")
        return
    
    print(f"\n--- Generating Responses ---")
    responses = {}
    
    for model_name, handler in handlers.items():
        print(f"\nGenerating with {model_name}...")
        try:
            response = handler.generate(
                prompt=prompt,
                max_new_tokens=100,
                do_sample=False,  # Deterministic for comparison
            )
            responses[model_name] = response
            print(f"[{model_name}] Response: {response[:100]}...")
            
        except Exception as e:
            print(f"[ERROR] Generation failed for {model_name}: {e}")
            responses[model_name] = None
    
    # Compare responses
    print(f"\n--- Comparison Results ---")
    if 'rgtnet' in responses and 'rgtnet_orthonly' in responses:
        if responses['rgtnet'] and responses['rgtnet_orthonly']:
            are_different = responses['rgtnet'] != responses['rgtnet_orthonly']
            print(f"Responses are different: {are_different}")
            
            if are_different:
                print("\n🎉 SUCCESS: Models produce different outputs!")
                print("This suggests role-aware processing is working.")
                
                print(f"\nrgtnet response length: {len(responses['rgtnet'])}")
                print(f"rgtnet_orthonly response length: {len(responses['rgtnet_orthonly'])}")
                
                # Show first difference
                min_len = min(len(responses['rgtnet']), len(responses['rgtnet_orthonly']))
                for i in range(min_len):
                    if responses['rgtnet'][i] != responses['rgtnet_orthonly'][i]:
                        print(f"First difference at position {i}:")
                        print(f"  rgtnet: '{responses['rgtnet'][max(0,i-10):i+10]}'")
                        print(f"  orthonly: '{responses['rgtnet_orthonly'][max(0,i-10):i+10]}'")
                        break
            else:
                print("\n❌ ISSUE: Models produce identical outputs")
                print("Role-aware processing may not be working correctly.")
        else:
            print("One or both responses are None - generation failed")
    else:
        print("Missing responses for comparison")

if __name__ == "__main__":
    test_orthogonal_difference()
