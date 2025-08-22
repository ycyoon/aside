#!/usr/bin/env python3
"""
Simple test to verify role-aware wrapper functionality.
"""

import os
import sys
import torch
sys.path.append('/home/ycyoon/work/aside/experiments')

from model_api import load_rgtnet_model_and_tokenizer

def test_role_aware_wrapper():
    """Test the role-aware wrapper directly."""
    
    print("=== Testing Role-Aware Wrapper ===")
    
    # Test with a simple model path (modify orthonly path for testing)
    model_path = '/home/ycyoon/work/aside/experiments/models/rgtnet_qwen2.5-7b_20250814_0759'
    
    if not os.path.exists(model_path):
        print(f"[ERROR] Model path not found: {model_path}")
        return
    
    print(f"Loading model from: {model_path}")
    
    try:
        # Load regular model
        model_regular, tokenizer = load_rgtnet_model_and_tokenizer(
            checkpoint_path=model_path,
            tokenizer_path=model_path,  # Use same path for tokenizer
            model_dtype=torch.bfloat16,
            device='cuda:0'
        )
        
        print(f"Regular model loaded: {type(model_regular)}")
        print(f"Has _embedding_type: {hasattr(model_regular, '_embedding_type')}")
        if hasattr(model_regular, '_embedding_type'):
            print(f"Embedding type: {model_regular._embedding_type}")
        
        # Manually change embedding type to test orthonly behavior
        print(f"\nModifying model to test orthonly behavior...")
        if hasattr(model_regular, '_embedding_type'):
            original_type = model_regular._embedding_type
            model_regular._embedding_type = 'rgtnet_orthonly'
            print(f"Changed embedding type from {original_type} to {model_regular._embedding_type}")
        
        model_orthonly = model_regular  # Use same model but with different embedding type
        
        # Test role mask processing
        print(f"\n--- Testing Role Mask Processing ---")
        
        # Create test inputs
        test_text = "Analyze this code: def test(): return 42"
        inputs = tokenizer(test_text, return_tensors='pt').to('cuda:0')
        
        # Create role mask (instruction=0, data=1)
        seq_len = inputs['input_ids'].shape[1]
        role_mask = torch.zeros_like(inputs['input_ids'])
        # Assume last half is "data" for testing
        role_mask[:, seq_len//2:] = 1
        
        print(f"Input shape: {inputs['input_ids'].shape}")
        print(f"Role mask: {role_mask[0].tolist()}")
        
        # Test regular model
        print(f"\n--- Testing Regular Model ---")
        with torch.no_grad():
            output_regular = model_regular.forward(
                input_ids=inputs['input_ids'],
                attention_mask=inputs.get('attention_mask'),
                role_mask=role_mask
            )
        
        # Test orthonly model 
        print(f"\n--- Testing Orthonly Model ---")
        with torch.no_grad():
            output_orthonly = model_orthonly.forward(
                input_ids=inputs['input_ids'],
                attention_mask=inputs.get('attention_mask'),
                role_mask=role_mask
            )
        
        # Compare outputs
        print(f"\n--- Comparing Outputs ---")
        
        # Get logits
        logits_regular = output_regular.logits
        logits_orthonly = output_orthonly.logits
        
        print(f"Regular logits shape: {logits_regular.shape}")
        print(f"Orthonly logits shape: {logits_orthonly.shape}")
        
        # Compare logits
        diff = torch.abs(logits_regular - logits_orthonly).mean().item()
        print(f"Mean absolute difference: {diff}")
        
        if diff > 1e-6:
            print("🎉 SUCCESS: Models produce different outputs!")
            print("Role-aware processing is working.")
        else:
            print("❌ ISSUE: Models produce very similar outputs")
            print("Role-aware processing may not be effective.")
        
    except Exception as e:
        print(f"[ERROR] Test failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_role_aware_wrapper()
