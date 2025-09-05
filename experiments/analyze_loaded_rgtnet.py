"""
Analyze already loaded RGTNet model from orthogonal analysis
"""

import sys
sys.path.append(".")

from orthogonal_role_analysis import OrthogonalRoleAnalyzer
import torch
import numpy as np

def analyze_loaded_rgtnet():
    """Analyze the role transformers from already loaded model."""
    
    rgtnet_models_config = [
        {
            "name": "RGTNet", 
            "model_path": "/home/ycyoon/work/aside/experiments/models/rgtnet_qwen2.5-7b_20250814_0759/merged_epoch_1",
            "base_model": "Qwen/Qwen2.5-7B",
            "embedding_type": "rgtnet",
            "load_from_checkpoint": True,
            "force_native": True
        }
    ]
    
    analyzer = OrthogonalRoleAnalyzer(rgtnet_models_config)
    handler = analyzer._load_model_handler(rgtnet_models_config[0])
    
    model = handler.model
    tokenizer = handler.tokenizer
    device = next(model.parameters()).device
    
    print(f"✅ Model loaded successfully")
    print(f"Model type: {type(model)}")
    print(f"Is native RGTNet: {getattr(model, 'is_native_rgtnet', False)}")
    
    # Check if model has role-sensitive embedding
    if hasattr(model, 'embedding'):
        print("\n--- Analyzing RoleSensitiveEmbedding ---")
        role_embedding = model.embedding
        print(f"Role embedding type: {type(role_embedding)}")
        
        if hasattr(role_embedding, 'role_transformers'):
            role_transformers = role_embedding.role_transformers
            print(f"Number of role transformers: {len(role_transformers)}")
            
            # Analyze each role transformer
            for i, transformer in enumerate(role_transformers):
                weight = transformer.weight.data
                print(f"\n--- Role Transformer {i} ---")
                print(f"Weight shape: {weight.shape}")
                
                # Check if it's still identity matrix
                identity = torch.eye(weight.shape[0], device=weight.device, dtype=weight.dtype)
                diff_from_identity = torch.norm(weight - identity)
                
                print(f"Difference from identity matrix: {diff_from_identity:.6f}")
                
                # Check some sample values
                print(f"Diagonal sample (should be ~1 for identity): {weight.diag()[:5]}")
                print(f"Off-diagonal sample (should be ~0 for identity): {weight[0, 1:6]}")
                
                # Calculate how much it learned
                if diff_from_identity < 1e-4:
                    print("❌ Still essentially identity matrix - no learning!")
                elif diff_from_identity < 0.1:
                    print("⚠️  Minimal learning from identity")
                else:
                    print("✅ Learned meaningful transformation")
                    
            # Test the role embedding directly
            print("\n--- Testing RoleSensitiveEmbedding directly ---")
            
            with torch.no_grad():
                user_emb = role_embedding(input_ids, user_role_mask)
                agent_emb = role_embedding(input_ids, agent_role_mask)
                
                diff = torch.norm(user_emb - agent_emb)
                cosine_sim = torch.cosine_similarity(
                    user_emb.flatten(), 
                    agent_emb.flatten(), 
                    dim=0
                )
                
                print(f"Direct embedding difference:")
                print(f"  Norm difference: {diff:.6f}")
                print(f"  Cosine similarity: {cosine_sim:.6f}")
                
                if diff < 1e-6:
                    print("❌ CRITICAL: Role embeddings are identical!")
                else:
                    print("✅ Role embeddings show difference")
        else:
            print("❌ No role_transformers found in embedding")
    else:
        print("❌ No embedding found")
        
    # Alternative: Check module structure
    print("\n--- Model Structure Analysis ---")
    for name, module in model.named_modules():
        if 'embed' in name.lower() and 'role' in name.lower():
            print(f"Found role-related embedding: {name}")
        if 'embed' in name.lower():
            print(f"Found embedding module: {name} -> {type(module)}")
            
    # Test direct embedding difference
    print("\n--- Testing direct embedding approach ---")
    
    # Test with simple input
    test_text = "Hello world"
    tokens = tokenizer.encode(test_text, add_special_tokens=False)
    if not tokens:
        tokens = [tokenizer.unk_token_id]
        
    input_ids = torch.tensor([tokens]).to(device)
    print(f"Test tokens: {tokens}")
    print(f"Test text: {test_text}")
    
    # Create role masks
    user_role_mask = torch.zeros_like(input_ids)  # 0 for user
    agent_role_mask = torch.ones_like(input_ids)  # 1 for agent
    
    print(f"User role mask: {user_role_mask}")
    print(f"Agent role mask: {agent_role_mask}")
    
    # Test at the very beginning - check if embedding layer responds to role_mask
    with torch.no_grad():
        try:
            # Get embeddings through model call (without output_hidden_states)
            user_outputs = model(input_ids=input_ids, role_mask=user_role_mask)
            agent_outputs = model(input_ids=input_ids, role_mask=agent_role_mask)
            
            print(f"✅ Model call successful")
            print(f"User output type: {type(user_outputs)}")
            
            # Check if outputs have hidden states or logits
            if hasattr(user_outputs, 'logits'):
                user_logits = user_outputs.logits
                agent_logits = agent_outputs.logits
                
                diff = torch.norm(user_logits - agent_logits)
                cosine_sim = torch.cosine_similarity(
                    user_logits.flatten(), 
                    agent_logits.flatten(), 
                    dim=0
                )
                
                print(f"\nFinal output (logits) difference:")
                print(f"  Norm difference: {diff:.6f}")
                print(f"  Cosine similarity: {cosine_sim:.6f}")
            else:
                print("❌ No logits found in output")
                
        except Exception as e:
            print(f"❌ Error in model call: {e}")
            
    # Check if the model architecture actually implements role-aware embedding
    print("\n--- Architecture Deep Dive ---")
    
    # Look for RoleSensitiveEmbedding specifically
    found_role_embedding = False
    for name, module in model.named_modules():
        if 'RoleSensitiveEmbedding' in str(type(module)):
            print(f"✅ Found RoleSensitiveEmbedding: {name}")
            found_role_embedding = True
        elif 'Embedding' in str(type(module)):
            print(f"Regular embedding found: {name} -> {type(module)}")
            
    if not found_role_embedding:
        print("❌ MAJOR ISSUE: No RoleSensitiveEmbedding found in model!")
        print("This suggests the model was loaded as regular HF model, not native RGTNet")

if __name__ == "__main__":
    analyze_loaded_rgtnet()
