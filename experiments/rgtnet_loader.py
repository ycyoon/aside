"""
RGTNet Model Loading Utility

This module provides a reusable function to load RGTNet models from checkpoints.
Based on the loading pattern from model_test.py.

Usage:
    from rgtnet_loader import load_rgtnet_model
    
    model, tokenizer = load_rgtnet_model(
        model_dir="/path/to/model",
        base_model_name="meta-llama/Llama-3.2-3B-Instruct",
        bias_delta=5.0
    )
"""

import torch
import os
import json
from types import SimpleNamespace
from transformers import AutoTokenizer, AutoConfig

def load_rgtnet_model(model_dir, base_model_name=None, bias_delta=5.0, device=None, dtype=None):
    """
    Load RGTNet model from directory (supports both merged model and DeepSpeed checkpoint formats)
    
    Args:
        model_dir (str): Path to model directory
        base_model_name (str, optional): Base model name for DeepSpeed checkpoints
        bias_delta (float): Bias delta parameter for role-gated attention
        device (torch.device, optional): Device to load model on
        dtype (torch.dtype, optional): Data type for model
        
    Returns:
        tuple: (model, tokenizer) - loaded model and tokenizer
        
    Raises:
        ValueError: If model directory doesn't exist or required parameters are missing
        Exception: If model loading fails
    """
    import sys
    import os
    
    # Add RGTNet directory to path for native model loading
    rgtnet_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "RGTNet")
    if rgtnet_path not in sys.path:
        sys.path.insert(0, rgtnet_path)
    
    from model import create_model as create_rgtnet, load_checkpoint as load_rgtnet_checkpoint
    
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if dtype is None:
        dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
        
    print(f"🚀 Loading RGTNet model from: {model_dir}")
    print(f"💻 Using device: {device}, dtype: {dtype}")
    
    if not os.path.exists(model_dir):
        raise ValueError(f"Model directory not found: {model_dir}")

    cfg_path = os.path.join(model_dir, "rgtnet_config.json")
    
    try:
        if os.path.exists(cfg_path):
            print("💡 Found rgtnet_config.json. Loading in Merged Model mode.")
            # Load native config
            with open(cfg_path, "r") as f:
                rgtnet_cfg = json.load(f)

            tok_name = rgtnet_cfg.get("tokenizer_name") or rgtnet_cfg.get("pretrained_model_name")
            if not tok_name:
                raise ValueError("Tokenizer name not found in rgtnet_config.json (tokenizer_name or pretrained_model_name)")
            
            ns_args = SimpleNamespace(
                vocab_size=rgtnet_cfg.get("vocab_size"),
                d_model=rgtnet_cfg.get("d_model"),
                nhead=rgtnet_cfg.get("nhead"),
                num_layers=rgtnet_cfg.get("num_layers"),
                dim_feedforward=rgtnet_cfg.get("dim_feedforward"),
                dropout=rgtnet_cfg.get("dropout", 0.1),
                bias_delta=rgtnet_cfg.get("bias_delta", bias_delta),
                max_seq_len=rgtnet_cfg.get("max_seq_len", 2048),
                pretrained_model_name=None,
                gradient_checkpointing=False,
            )
            pad_id = rgtnet_cfg.get("pad_token_id")

        else:
            print("⚠️ rgtnet_config.json not found. Assuming DeepSpeed Checkpoint mode.")
            if base_model_name is None:
                raise ValueError("base_model_name is required for DeepSpeed checkpoint mode")
            
            tok_name = base_model_name
            base_config = AutoConfig.from_pretrained(tok_name, trust_remote_code=True)
            
            ns_args = SimpleNamespace(
                vocab_size=base_config.vocab_size,
                d_model=base_config.hidden_size,
                nhead=base_config.num_attention_heads,
                num_layers=base_config.num_hidden_layers,
                dim_feedforward=base_config.intermediate_size,
                dropout=0.1,
                bias_delta=bias_delta,
                max_seq_len=base_config.max_position_embeddings,
                pretrained_model_name=None,
                gradient_checkpointing=False,
            )
            pad_id = base_config.pad_token_id

        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(tok_name, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        if pad_id is None:
            pad_id = tokenizer.pad_token_id or tokenizer.eos_token_id

        # Build model and load checkpoint
        print("🔧 Creating RGTNet model architecture...")
        model = create_rgtnet(ns_args, pad_idx=pad_id)
        
        print("📥 Loading checkpoint weights...")
        model = load_rgtnet_checkpoint(model_dir, model, device=str(device))
        
        print("🔄 Moving model to target device and dtype...")
        model = model.to(device=device, dtype=dtype)
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
        print("✅ RGTNet model and tokenizer loaded successfully!")
        
        # Basic validation
        model.eval()
        test_input = torch.randint(0, min(1000, tokenizer.vocab_size), (1, 10), device=device)
        role_mask = torch.zeros_like(test_input)
        with torch.no_grad():
            outputs = model(input_ids=test_input, role_mask=role_mask)
            logits = outputs["logits"] if isinstance(outputs, dict) else outputs.logits
            print(f"✅ Validation forward pass successful! Output shape: {tuple(logits.shape)}")
        
        return model, tokenizer
        
    except Exception as e:
        print(f"❌ Error loading RGTNet model: {e}")
        import traceback
        traceback.print_exc()
        raise

def test_loading():
    """Test function for RGTNet model loading"""
    print("🧪 Testing RGTNet model loading function...")
    
    # Example usage - update path as needed
    model_dir = "/home/ycyoon/work/aside/experiments/models/rgtnet_llama-3.2-3b-instruct_20250831_1646/merged_epoch_7"
    base_model = "meta-llama/Llama-3.2-3B-Instruct"
    
    try:
        model, tokenizer = load_rgtnet_model(
            model_dir=model_dir,
            base_model_name=base_model,
            bias_delta=5.0
        )
        print("✅ RGTNet model loading test successful!")
        
        # Example forward pass - ensure tensors are on the same device as model
        model_device = next(model.parameters()).device
        text = "Hello, how are you?"
        inputs = tokenizer(text, return_tensors="pt")
        input_ids = inputs["input_ids"].to(model_device)
        role_mask = torch.zeros_like(input_ids)
        
        with torch.no_grad():
            outputs = model(input_ids=input_ids, role_mask=role_mask)
            logits = outputs["logits"] if isinstance(outputs, dict) else outputs.logits
            print(f"✅ Forward pass successful! Logits shape: {logits.shape}")
            print(f"📍 Model device: {model_device}, Input device: {input_ids.device}")
            
        return model, tokenizer
            
    except Exception as e:
        print(f"❌ RGTNet loading test failed: {e}")
        import traceback
        traceback.print_exc()
        return None, None

if __name__ == "__main__":
    test_loading()
