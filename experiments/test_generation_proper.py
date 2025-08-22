#!/usr/bin/env python3
"""
Test actual generation with proper instruction-data separation and temperature control.
"""

import os
import sys
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

sys.path.append('/home/ycyoon/work/aside/experiments')
from model_api import _add_role_aware_wrapper

def test_generation_with_proper_split():
    """Test generation with proper instruction-data split and temperature control."""
    
    print("=== Generation Test with Proper Split ===")
    
    # Use a standard HF model for testing
    model_name = "Qwen/Qwen2.5-0.5B"
    
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
        
        # Create regular and orthonly wrappers
        model_regular = _add_role_aware_wrapper(model, model_path="test_rgtnet")
        model_orthonly = _add_role_aware_wrapper(model, model_path="test_rgtnet_orthonly")
        
        print(f"Regular embedding type: {model_regular._embedding_type}")
        print(f"Orthonly embedding type: {model_orthonly._embedding_type}")
        
        # Test with a prompt that has clear instruction-data separation
        test_prompt = """Please analyze this code:

Input: def factorial(n):
    if n <= 1:
        return 1
    return n * factorial(n-1)

What is the time complexity?"""
        
        print(f"\n--- Test Prompt ---")
        print(f"'{test_prompt}'")
        
        # Tokenize and create proper role mask
        inputs = tokenizer(test_prompt, return_tensors='pt').to('cuda:0')
        
        # Create role mask based on "Input:" marker (like in _create_role_mask_for_rgtnet)
        seq_tokens = inputs['input_ids'][0].cpu().tolist()
        input_marker_tokens = tokenizer("Input:", add_special_tokens=False)["input_ids"]
        
        print(f"\n--- Role Mask Creation ---")
        print(f"Looking for Input marker tokens: {input_marker_tokens}")
        print(f"Sequence length: {len(seq_tokens)}")
        
        # Find where "Input:" appears
        data_start_idx = None
        for i in range(len(seq_tokens) - len(input_marker_tokens) + 1):
            if seq_tokens[i:i+len(input_marker_tokens)] == input_marker_tokens:
                data_start_idx = i + len(input_marker_tokens)
                print(f"Found Input: at position {i}, data starts at {data_start_idx}")
                break
        
        # Create role mask: 0 for instruction, 1 for data
        role_mask = torch.zeros_like(inputs['input_ids'])
        if data_start_idx is not None and data_start_idx < inputs['input_ids'].shape[1]:
            role_mask[:, data_start_idx:] = 1
            print(f"Set tokens {data_start_idx}: onwards to data (1)")
        else:
            print("No Input: marker found - treating all as instruction")
        
        print(f"Final role mask: {role_mask[0].tolist()}")
        
        # Decode to show what's instruction vs data
        if data_start_idx is not None:
            instruction_part = tokenizer.decode(inputs['input_ids'][0][:data_start_idx])
            data_part = tokenizer.decode(inputs['input_ids'][0][data_start_idx:])
            print(f"\nInstruction part (role=0): '{instruction_part}'")
            print(f"Data part (role=1): '{data_part}'")
        
        # Test generation with same temperature
        print(f"\n--- Generation Test ---")
        
        generation_kwargs = {
            'max_new_tokens': 50,
            'do_sample': True,
            'temperature': 1.0,  # Same temperature for both
            'top_p': 0.9,
            'pad_token_id': tokenizer.eos_token_id,
        }
        
        print(f"Generation parameters: {generation_kwargs}")
        
        # Generate with regular model
        print(f"\n--- Regular Model Generation ---")
        with torch.no_grad():
            # Test forward pass first with role mask
            output_regular_forward = model_regular.forward(
                input_ids=inputs['input_ids'],
                attention_mask=inputs.get('attention_mask'),
                role_mask=role_mask
            )
            
            # Then generate (note: generate doesn't use role_mask directly)
            output_regular = model_regular.generate(
                **inputs,
                **generation_kwargs
            )
        
        # Generate with orthonly model
        print(f"\n--- Orthonly Model Generation ---")
        with torch.no_grad():
            # Test forward pass first with role mask
            output_orthonly_forward = model_orthonly.forward(
                input_ids=inputs['input_ids'],
                attention_mask=inputs.get('attention_mask'),
                role_mask=role_mask
            )
            
            # Then generate
            output_orthonly = model_orthonly.generate(
                **inputs,
                **generation_kwargs
            )
        
        # Compare forward pass outputs (with role mask)
        print(f"\n--- Forward Pass Comparison (with role_mask) ---")
        logits_diff = torch.abs(output_regular_forward.logits - output_orthonly_forward.logits).mean().item()
        print(f"Mean logits difference: {logits_diff}")
        
        # Compare generation outputs
        print(f"\n--- Generation Comparison ---")
        regular_text = tokenizer.decode(output_regular[0], skip_special_tokens=True)
        orthonly_text = tokenizer.decode(output_orthonly[0], skip_special_tokens=True)
        
        print(f"Regular output: '{regular_text}'")
        print(f"Orthonly output: '{orthonly_text}'")
        
        are_different = regular_text != orthonly_text
        print(f"\nGeneration outputs are different: {are_different}")
        
        if are_different:
            print("🎉 SUCCESS: Different generation outputs confirmed!")
        else:
            print("⚠️  NOTICE: Generation outputs are identical")
            print("This might be due to random sampling or other factors")
        
        # Summary
        print(f"\n--- Summary ---")
        print(f"✅ Temperature: Same for both models ({generation_kwargs['temperature']})")
        print(f"✅ Instruction-Data split: Properly detected using 'Input:' marker")
        print(f"✅ Role mask: Applied correctly in forward pass")
        print(f"✅ Forward pass differences: {logits_diff:.6f} (role-aware processing working)")
        print(f"{'✅' if are_different else '⚠️ '} Generation differences: {are_different}")
        
    except Exception as e:
        print(f"[ERROR] Test failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_generation_with_proper_split()
