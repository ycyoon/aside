#!/usr/bin/env python3
"""
Direct test of role-aware wrapper functionality.
"""

import os
import sys
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

sys.path.append('/home/ycyoon/work/aside/experiments')
from model_api import _add_role_aware_wrapper

def test_wrapper_direct():
    """Test the role-aware wrapper directly without complex loading."""
    
    print("=== Direct Wrapper Test ===")
    
    # Use a standard HF model for testing
    model_name = "Qwen/Qwen2.5-0.5B"  # Small model for testing
    
    print(f"Loading base model: {model_name}")
    
    try:
        # Load base model and tokenizer
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            device_map='cuda:0'
        )
        
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        print(f"Base model loaded: {type(model)}")
        
        # Test regular wrapper
        print(f"\n--- Creating Regular Wrapper ---")
        model_regular = _add_role_aware_wrapper(model, model_path="test_rgtnet")
        print(f"Regular wrapper created: {type(model_regular)}")
        print(f"Embedding type: {getattr(model_regular, '_embedding_type', 'Not found')}")
        
        # Test orthonly wrapper  
        print(f"\n--- Creating Orthonly Wrapper ---")
        model_orthonly = _add_role_aware_wrapper(model, model_path="test_rgtnet_orthonly")
        print(f"Orthonly wrapper created: {type(model_orthonly)}")
        print(f"Embedding type: {getattr(model_orthonly, '_embedding_type', 'Not found')}")
        
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
        max_diff = torch.abs(logits_regular - logits_orthonly).max().item()
        
        print(f"Mean absolute difference: {diff}")
        print(f"Max absolute difference: {max_diff}")
        
        if diff > 1e-6:
            print("🎉 SUCCESS: Models produce different outputs!")
            print("Role-aware processing is working.")
            
            # Show some logit values
            print(f"\nSample logit comparison (first 3 positions):")
            for i in range(min(3, logits_regular.shape[1])):
                reg_val = logits_regular[0, i, 0].item()
                orth_val = logits_orthonly[0, i, 0].item()
                print(f"  Position {i}: regular={reg_val:.4f}, orthonly={orth_val:.4f}, diff={abs(reg_val-orth_val):.4f}")
                
        else:
            print("❌ ISSUE: Models produce very similar outputs")
            print("Role-aware processing may not be effective.")
        
    except Exception as e:
        print(f"[ERROR] Test failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_wrapper_direct()
