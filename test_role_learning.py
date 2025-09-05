#!/usr/bin/env python3
"""
Real-time Role Transformer monitoring during training
"""
import sys
import os
sys.path.append('/home/ycyoon/work/aside/RGTNet')
sys.path.append('/home/ycyoon/work/aside/experiments')

import torch
from transformers import AutoTokenizer

# Import from RGTNet directory
from RGTNet.model import create_model

def test_role_transformer_learning():
    """
    Create a model and run a few training steps to see if role transformers learn
    """
    print("🧪 Testing Role Transformer Learning in Mini Training Session...")
    print("=" * 70)
    
    # Create a small test model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"📍 Using device: {device}")
    
    # Model configuration (smaller for testing)
    config = {
        'd_model': 512,
        'nhead': 8, 
        'num_layers': 4,
        'dim_ff': 2048,
        'vocab_size': 32000,
        'max_seq_len': 128
    }
    
    print("🏗️ Creating RGTNet model...")
    model = create_model(config, device=device)
    
    if hasattr(model, 'embedding') and hasattr(model.embedding, 'role_transformers'):
        print("✅ Found role transformers!")
        
        # Initialize role transformers to identity (like in real training)
        for i, transformer in enumerate(model.embedding.role_transformers):
            torch.nn.init.eye_(transformer.weight)
            print(f"🔄 Initialized role transformer {i} to identity")
        
        # Function to check role transformer status
        def check_role_transformers(step_name):
            print(f"\n📊 Role Transformer Status - {step_name}:")
            for i, transformer in enumerate(model.embedding.role_transformers):
                weight = transformer.weight
                identity = torch.eye(weight.shape[0], device=weight.device, dtype=weight.dtype)
                diff = torch.norm(weight - identity).item()
                mean_val = weight.mean().item()
                std_val = weight.std().item()
                
                print(f"  🔍 Transformer {i}:")
                print(f"    📐 Difference from identity: {diff:.8f}")
                print(f"    📊 Mean: {mean_val:.8f}, Std: {std_val:.8f}")
                
                if diff < 1e-6:
                    print(f"    ❌ No change (diff < 1e-6)")
                elif diff < 1e-3:
                    print(f"    🟡 Small change (diff < 1e-3)")
                else:
                    print(f"    ✅ Significant change!")
        
        # Check initial state
        check_role_transformers("Initial (Identity)")
        
        # Create some fake training data
        batch_size = 4
        seq_len = 32
        
        input_ids = torch.randint(0, config['vocab_size'], (batch_size, seq_len), device=device)
        labels = input_ids.clone()
        role_mask = torch.randint(0, 2, (batch_size, seq_len), device=device)
        
        # Import role separation loss function
        sys.path.append('/home/ycyoon/work/aside/RGTNet')
        from trainer import compute_role_separation_loss
        
        # Setup optimizer
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
        
        print(f"\n🏃 Running mini training session...")
        model.train()
        
        for step in range(10):  # 10 training steps
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(input_ids=input_ids, labels=labels, role_mask=role_mask)
            loss = outputs['loss']
            
            # Add role separation loss
            try:
                role_loss_components = compute_role_separation_loss(
                    model, input_ids, alpha_role=0.1, alpha_orth=0.05
                )
                total_role_loss = role_loss_components['total_role_loss']
                total_loss = loss + total_role_loss
                
                if step % 2 == 0:
                    print(f"  Step {step}: Main Loss: {loss.item():.4f}, "
                          f"Role Sep: {role_loss_components['role_separation_loss'].item():.4f}, "
                          f"Orth: {role_loss_components['orthogonality_loss'].item():.4f}, "
                          f"Transform: {role_loss_components['role_transformer_loss'].item():.4f}")
                
            except Exception as e:
                print(f"    ⚠️ Role loss computation failed: {e}")
                total_loss = loss
            
            # Backward pass
            total_loss.backward()
            optimizer.step()
        
        # Check final state
        check_role_transformers("After 10 Training Steps")
        
        # Test if the changes are meaningful
        print(f"\n🎯 Summary:")
        any_learning = False
        for i, transformer in enumerate(model.embedding.role_transformers):
            weight = transformer.weight
            identity = torch.eye(weight.shape[0], device=weight.device, dtype=weight.dtype)
            diff = torch.norm(weight - identity).item()
            
            if diff > 1e-4:
                print(f"  ✅ Role Transformer {i}: Learning detected! (diff = {diff:.6f})")
                any_learning = True
            else:
                print(f"  ❌ Role Transformer {i}: No significant learning (diff = {diff:.6f})")
        
        if any_learning:
            print(f"\n🎉 SUCCESS: Role Transformers are learning with Role Separation Loss!")
        else:
            print(f"\n⚠️ ISSUE: Role Transformers are not learning enough. May need:")
            print(f"    - Higher role loss weights")
            print(f"    - More training steps") 
            print(f"    - Different loss formulation")
            
    else:
        print("❌ No role transformers found in model")

if __name__ == "__main__":
    test_role_transformer_learning()
