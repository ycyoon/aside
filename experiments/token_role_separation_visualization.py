"""
Token Role Separation Visualization for Real Data

This script creates PCA-based visualizations showing how tokens from different roles
(instruction vs data) are separated in the hidden state space across different layers
and model architectures.

Expected output: A multi-panel plot comparing Standard LM, ASIDE, and RGTNet models
across early, middle, and final layers with individual tokens labeled.
"""

import sys
import os
import argparse
import json
import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import pandas as pd
from tqdm import tqdm
import pickle
from datetime import datetime

# Add parent directory to path
if "../.." not in sys.path:
    sys.path.append("../..")

# 기존 CustomModelHandler 대신 직접 로딩 함수들 import
from model_api import (
    load_single_emb_model_and_tokenizer,
    load_rgtnet_model_and_tokenizer, 
    load_forward_rot_model_and_tokenizer
)
from rgtnet_model import RoleAwareTransformerDecoder  # Native RGTNet 직접 import

class TokenRoleSeparationVisualizer:
    """
    Creates visualizations showing how tokens are separated by role across layers.
    """
    
    def __init__(self, models_config, max_tokens_per_role=50):
        """
        Initialize the visualizer with model configurations.
        
        Args:
            models_config (list): List of model configurations
            max_tokens_per_role (int): Maximum tokens to visualize per role
        """
        self.models_config = models_config
        self.max_tokens_per_role = max_tokens_per_role
        self.token_data = {}
        
    def create_test_examples(self):
        """
        Create diverse test examples with clear role boundaries.
        """
        examples = [
            {
                "instruction": "Solve the following math problem:",
                "data": "What is 2 + 2?"
            },
            {
                "instruction": "Translate the following text to French:",
                "data": "Hello, how are you today?"
            },
            {
                "instruction": "Analyze the sentiment of this review:",
                "data": "This movie was absolutely fantastic! Great acting and plot."
            },
            {
                "instruction": "Extract key information from this document:",
                "data": "The quarterly report shows a 15% increase in revenue compared to last year."
            },
            {
                "instruction": "Classify the following text:",
                "data": "Breaking news: Scientists discover new planet in distant galaxy."
            },
            {
                "instruction": "Summarize the main points:",
                "data": "Climate change is affecting global weather patterns. Rising temperatures are melting ice caps and causing sea levels to rise."
            },
            {
                "instruction": "Answer the following question:",
                "data": "What are the benefits of renewable energy sources?"
            },
            {
                "instruction": "Convert this to JSON format:",
                "data": "Name: John Smith, Age: 30, City: New York"
            },
            {
                "instruction": "Find errors in this code:",
                "data": "def calculate_sum(a, b):\n    return a + b\nprint(calculate_sum(5))"
            },
            {
                "instruction": "Explain the concept:",
                "data": "What is machine learning and how does it work?"
            }
        ]
        return examples
    
    def extract_token_representations(self, handler, examples, layers_to_extract=None):
        """
        Extract hidden representations for tokens from specified layers.
        
        Args:
            handler: Model handler instance
            examples: List of instruction-data examples
            layers_to_extract: List of layer indices to extract (e.g., [0, 14, 27])
        
        Returns:
            dict: Token representations organized by layer and role
        """
        if layers_to_extract is None:
            # Default to first, middle, and last layers
            num_layers = len(handler.model.model.layers)
            layers_to_extract = [0, num_layers // 2, num_layers - 1]
        
        # Load prompt template
        try:
            with open("./data/prompt_templates.json", "r") as f:
                templates = json.load(f)
            template = templates[0]
        except FileNotFoundError:
            template = {
                "system_prompt": "You are a helpful assistant.",
                "infix": "Input:",
                "suffix": ""
            }
        
        token_data = {layer: {"instruction": [], "data": [], "tokens": [], "token_strs": []} 
                     for layer in layers_to_extract}
        
        # Track total tokens collected per role across all examples (global budget)
        total_inst_collected = 0
        total_data_collected = 0
        
        for example in tqdm(examples, desc="Processing examples"):
            # Stop if we already have enough tokens globally
            if total_inst_collected >= self.max_tokens_per_role and total_data_collected >= self.max_tokens_per_role:
                break
            # Format and tokenize
            instruction_text = format_prompt(example["instruction"], template, "system")
            data_text = format_prompt(example["data"], template, "user")

            print('[DEBUG] instruction', instruction_text)
            print('[DEBUG] data', data_text)
            
            # Tokenize the full combined text to get accurate boundaries
            full_text = instruction_text + data_text
            full_tokens = handler.tokenizer(full_text, add_special_tokens=True)["input_ids"]
            
            # Find boundary between instruction and data by tokenizing separately
            inst_tokens_only = handler.tokenizer(instruction_text, add_special_tokens=False)["input_ids"]
            
            # Calculate boundary more carefully
            # Check if BOS token actually exists in the sequence
            has_bos = (handler.tokenizer.bos_token_id is not None and 
                      len(full_tokens) > 0 and 
                      full_tokens[0] == handler.tokenizer.bos_token_id)
            
            bos_offset = 1 if has_bos else 0
            inst_boundary = bos_offset + len(inst_tokens_only)
            
            all_tokens = full_tokens
            seq_len = len(all_tokens)
            
            # Decide which token indices to collect for this example according to remaining global budget
            remain_inst = max(0, self.max_tokens_per_role - total_inst_collected)
            remain_data = max(0, self.max_tokens_per_role - total_data_collected)
            
            selected_inst_indices = list(range(0, min(inst_boundary, remain_inst)))
            selected_data_indices = list(range(inst_boundary, min(seq_len, inst_boundary + remain_data)))
            
            # Convert to tensor
            input_ids = torch.tensor([all_tokens], dtype=torch.long)
            attention_mask = torch.ones_like(input_ids)
            
            # Move to device
            device = next(handler.model.parameters()).device
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            
            # Create segment_ids if needed
            embedding_type = getattr(handler, 'embedding_type', 'single_emb')
            segment_ids = None
            role_mask = None
            if embedding_type in ("forward_rot", "ise", "rgtnet", "rgtnet_orthonly"):
                # Create segment_ids based on token boundary
                seg_ids = [0] * inst_boundary  # Instruction tokens (including BOS)
                seg_ids.extend([1] * (len(all_tokens) - inst_boundary))  # Data tokens (including EOS)
                segment_ids = torch.tensor([seg_ids], dtype=torch.long).to(device)

                print('[DEBUG] segment_ids', segment_ids)

                # For RGTNet, also create role_mask (same as segment_ids)
                if embedding_type == "rgtnet":
                    role_mask = segment_ids.clone()
            
        # Forward pass to get hidden states
            handler.model.eval()
            with torch.no_grad():
                if segment_ids is not None:
                    if role_mask is not None:
                        # RGTNet with role_mask
                        outputs = handler.model(
                            input_ids=input_ids,
                            attention_mask=attention_mask,
                            segment_ids=segment_ids,
                            role_mask=role_mask,
                            output_hidden_states=True,
                            return_dict=True
                        )
                    else:
                        # ASIDE/ISE/orthonly with segment_ids only
                        outputs = handler.model(
                            input_ids=input_ids,
                            attention_mask=attention_mask,
                            segment_ids=segment_ids,
                            output_hidden_states=True,
                            return_dict=True
                        )
                else:
                    # Standard LM
                    outputs = handler.model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        output_hidden_states=True,
                        return_dict=True
                    )
            
            # Extract representations for specified layers
            hidden_states = outputs.hidden_states
            
            for layer_idx in layers_to_extract:
                if layer_idx < len(hidden_states):
                    # Move to CPU and ensure float32 to avoid bfloat16 -> numpy issues later
                    layer_hidden = hidden_states[layer_idx].squeeze(0).cpu().to(torch.float32)  # [seq_len, hidden_size]
                    
                    # Append selected instruction tokens first, then data tokens
                    for i in selected_inst_indices:
                        token_data[layer_idx]["instruction"].append(layer_hidden[i])
                        token_data[layer_idx]["tokens"].append(all_tokens[i])
                        token_str = handler.tokenizer.decode([all_tokens[i]])
                        token_str = token_str.replace("Ġ", " ").strip()
                        if not token_str:
                            token_str = f"<{all_tokens[i]}>"
                        token_data[layer_idx]["token_strs"].append(token_str)
                        print('[DEBUG246] instruction', token_str)
                    
                    for i in selected_data_indices:
                        token_data[layer_idx]["data"].append(layer_hidden[i])
                        token_data[layer_idx]["tokens"].append(all_tokens[i])
                        token_str = handler.tokenizer.decode([all_tokens[i]])
                        token_str = token_str.replace("Ġ", " ").strip()
                        if not token_str:
                            token_str = f"<{all_tokens[i]}>"
                        token_data[layer_idx]["token_strs"].append(token_str)
                        print('[DEBUG256] data', token_str)

            # Update global counters once per example
            total_inst_collected += len(selected_inst_indices)
            total_data_collected += len(selected_data_indices)
        
        return token_data, layers_to_extract
    
    def create_pca_visualization(self, all_model_data, layers_to_extract, output_path):
        """
        Create a comprehensive PCA visualization comparing models and layers.
        
        Args:
            all_model_data (dict): Token data for all models
            layers_to_extract (list): Layer indices that were extracted
            output_path (str): Path to save the visualization
        """
        # Create figure with subplots
        n_models = len(all_model_data)
        n_layers = len(layers_to_extract)
        
        fig, axes = plt.subplots(n_models, n_layers, figsize=(5 * n_layers, 4 * n_models))
        if n_models == 1:
            axes = axes.reshape(1, -1)
        if n_layers == 1:
            axes = axes.reshape(-1, 1)
        
        # Define colors for different roles
        colors = {
            'instruction': '#1f77b4',  # Blue
            'data': '#ff7f0e',  # Orange
            'attack': '#d62728'  # Red (for future use)
        }
        
        for model_idx, (model_name, model_data) in enumerate(all_model_data.items()):
            for layer_idx, layer in enumerate(layers_to_extract):
                ax = axes[model_idx, layer_idx]
                
                # Combine all representations for this layer
                inst_list = model_data[layer]["instruction"]
                data_list = model_data[layer]["data"]

                # If no representations available, skip this subplot
                if len(inst_list) == 0 and len(data_list) == 0:
                    print(f"Warning: no token representations for model {model_name}, layer {layer}; skipping")
                    continue

                # Stack tensors and ensure float32 for sklearn (handles bfloat16 outputs)
                if len(inst_list) > 0:
                    inst_reprs = torch.stack(inst_list).to(torch.float32)
                else:
                    # create empty with correct hidden dim from data_list
                    inst_reprs = torch.empty((0, data_list[0].shape[-1]), dtype=torch.float32)

                if len(data_list) > 0:
                    data_reprs = torch.stack(data_list).to(torch.float32)
                else:
                    # create empty with correct hidden dim from inst_list
                    data_reprs = torch.empty((0, inst_list[0].shape[-1]), dtype=torch.float32)

                # Concatenate and convert to numpy for scikit-learn
                all_reprs = torch.cat([inst_reprs, data_reprs], dim=0).numpy()
                
                # Normalize representations: mean-center for all, scale only for deeper layers
                if all_reprs.shape[0] > 1:
                    # Mean centering preserves relative geometry at layer 0
                    all_reprs = all_reprs - all_reprs.mean(axis=0, keepdims=True)
                    if layer != 0:
                        from sklearn.preprocessing import StandardScaler
                        scaler = StandardScaler(with_mean=False)
                        all_reprs = scaler.fit_transform(all_reprs)
                
                all_labels = (["instruction"] * inst_reprs.shape[0] + 
                             ["data"] * data_reprs.shape[0])
                
                # Apply PCA
                pca = PCA(n_components=2, random_state=42)
                pca_result = pca.fit_transform(all_reprs)
                
                # Align PCA sign so that PC1 increases from instruction -> data (reduce mirror flips)
                try:
                    inst_len = inst_reprs.shape[0]
                    inst_mean = all_reprs[:inst_len].mean(axis=0)
                    data_mean = all_reprs[inst_len:].mean(axis=0)
                    direction = data_mean - inst_mean
                    # Correlate direction with PC1
                    sign = np.sign(np.dot(direction, pca.components_[0]))
                    if sign < 0:
                        pca_result[:, 0] *= -1
                        pca.components_[0] *= -1
                except Exception:
                    pass
                
                # Plot instruction tokens
                inst_mask = np.array(all_labels) == "instruction"
                ax.scatter(pca_result[inst_mask, 0], pca_result[inst_mask, 1], 
                          c=colors['instruction'], alpha=0.7, s=50, 
                          label='Instruction/Meta', zorder=2)
                
                # Plot data tokens  
                data_mask = np.array(all_labels) == "data"
                ax.scatter(pca_result[data_mask, 0], pca_result[data_mask, 1],
                          c=colors['data'], alpha=0.6, s=50,
                          label='User/Data Tokens', zorder=1)

                # Set robust axis limits to avoid overly dispersed visuals
                try:
                    x = pca_result[:, 0]
                    y = pca_result[:, 1]
                    x_min, x_max = np.percentile(x, [5, 95])
                    y_min, y_max = np.percentile(y, [5, 95])
                    # Add 10% padding
                    x_pad = 0.1 * (x_max - x_min + 1e-6)
                    y_pad = 0.1 * (y_max - y_min + 1e-6)
                    ax.set_xlim(x_min - x_pad, x_max + x_pad)
                    ax.set_ylim(y_min - y_pad, y_max + y_pad)
                except Exception:
                    pass
                
                # Add some token labels for interesting tokens
                # Get combined token strings
                all_token_strs = (model_data[layer]["token_strs"][:len(inst_reprs)] + 
                                 model_data[layer]["token_strs"][len(inst_reprs):len(inst_reprs)+len(data_reprs)])
                
                # Label some interesting tokens
                interesting_tokens = ["solve", "2+2", "2", "+", "math", "problem", "translate", "sentiment"]
                for i, token_str in enumerate(all_token_strs):
                    if any(interesting in token_str.lower() for interesting in interesting_tokens):
                        if i < len(pca_result):
                            ax.annotate(token_str, (pca_result[i, 0], pca_result[i, 1]), 
                                      fontsize=8, alpha=0.8,
                                      xytext=(5, 5), textcoords='offset points')
                
                # Set title and labels
                layer_name = f"Layer {layer}"
                if layer == layers_to_extract[0]:
                    layer_name = f"Layer {layer}"
                elif layer == layers_to_extract[-1]:
                    layer_name = f"Final Layer ({layer})"
                else:
                    layer_name = f"Middle Layer ({layer})"
                
                ax.set_title(layer_name, fontsize=12)
                ax.set_xlabel("PC1", fontsize=10)
                ax.set_ylabel("PC2", fontsize=10)
                
                # Add legend to first subplot
                if model_idx == 0 and layer_idx == 0:
                    ax.legend(fontsize=10)
                
                # Add model name as y-label for first column
                if layer_idx == 0:
                    ax.set_ylabel(f"{model_name}\n\nPC2", fontsize=12)
        
        # Set main title
        plt.suptitle("Comparison of Token Role Separation Across Layers (Real Data)", 
                    fontsize=16, y=0.98)
        
        # Add subtitle for PCA
        fig.text(0.5, 0.02, "PCA-Projected 2D Subspace (Real Hidden States)", 
                ha='center', fontsize=12, style='italic')
        
        plt.tight_layout()
        plt.subplots_adjust(top=0.92, bottom=0.08)
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Token role separation visualization saved to {output_path}")
    
    def _load_native_model(self, model_config):
        """
        Load native model directly without wrapper fallback.
        """
        model_path = model_config["model_path"] 
        base_model = model_config["base_model"]
        embedding_type = model_config["embedding_type"]
        
        if embedding_type == "single_emb":
            return load_single_emb_model_and_tokenizer(
                base_model, 
                use_flash_attention=True,
                device_map="auto"
            )
        
        elif embedding_type == "forward_rot":
            return load_forward_rot_model_and_tokenizer(
                model_path,
                base_model, 
                rotation_alpha=np.pi / 2,
                rotation_direction="right"
            )
        
        elif embedding_type in ("rgtnet", "rgtnet_orthonly"):
            # 강제로 Native RGTNet 로드
            model, tokenizer = self._force_load_native_rgtnet(
                model_path, base_model, embedding_type
            )
            return model, tokenizer
        
        else:
            raise ValueError(f"Unknown embedding type: {embedding_type}")
    
    def _force_load_native_rgtnet(self, checkpoint_path, base_model, embedding_type):
        """
        Force load native RGTNet implementation, bypassing wrapper fallback.
        """
        import torch
        from transformers import AutoTokenizer
        
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(base_model)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        # Try to load native RGTNet directly
        try:
            print(f"[FORCE NATIVE] Attempting direct native RGTNet load from {checkpoint_path}")
            
            # Check for native checkpoint files
            import os
            pytorch_model_path = os.path.join(checkpoint_path, "pytorch_model.bin")
            if not os.path.exists(pytorch_model_path):
                raise FileNotFoundError(f"No pytorch_model.bin found in {checkpoint_path}")
            
            # Load state dict and check for native keys
            state_dict = torch.load(pytorch_model_path, map_location="cpu")
            
            native_keys = [
                "embedding.role_transformers.0.weight",
                "embedding.role_transformers.1.weight"
            ]
            
            has_native = any(key in state_dict for key in native_keys)
            
            if has_native:
                print(f"[FORCE NATIVE] Native RGTNet keys found! Loading directly...")
                
                # Load native model architecture
                model = RoleAwareTransformerDecoder.from_pretrained(
                    checkpoint_path,
                    embedding_type=embedding_type
                )
                
                # Mark as native
                setattr(model, "is_native_rgtnet", True)
                print(f"[FORCE NATIVE] Successfully loaded native RGTNet!")
                
                return model, tokenizer
                
            else:
                print(f"[FORCE NATIVE] No native keys found. Available keys sample:")
                print(list(state_dict.keys())[:10])
                raise ValueError("Not a native RGTNet checkpoint")
                
        except Exception as e:
            print(f"[FORCE NATIVE] Failed to load native: {e}")
            
            # Fallback이지만 경고 출력
            print(f"[WARNING] Falling back to wrapper - this may affect accuracy!")
            return load_rgtnet_model_and_tokenizer(
                checkpoint_path,
                base_model, 
                embedding_type=embedding_type,
                device_map="auto"
            )
    
    def run_analysis(self, output_dir):
        """
        Run the complete token role separation analysis with native models.
        
        Args:
            output_dir (str): Directory to save results and visualizations
        """
        os.makedirs(output_dir, exist_ok=True)
        
        # Create test examples
        examples = self.create_test_examples()
        
        all_model_data = {}
        layers_to_extract = None
        
        for model_config in tqdm(self.models_config, desc="Processing models"):
            model_name = model_config["name"]
            print(f"\nProcessing {model_name}...")
            
            # Load native model directly
            model, tokenizer = self._load_native_model(model_config)
            
            # Verify if we got native RGTNet
            if model_config["embedding_type"] in ("rgtnet", "rgtnet_orthonly"):
                is_native = getattr(model, "is_native_rgtnet", False)
                print(f"[MODEL CHECK] {model_name} - Native RGTNet: {is_native}")
                print(f"[MODEL CHECK] Model type: {type(model)}")
                
                if not is_native:
                    print(f"[ERROR] {model_name} is using wrapper instead of native!")
                    print("This will affect the experimental results!")
            
            # Create a simple handler-like object for compatibility
            class SimpleHandler:
                def __init__(self, model, tokenizer, embedding_type):
                    self.model = model
                    self.tokenizer = tokenizer
                    self.embedding_type = embedding_type
            
            handler = SimpleHandler(model, tokenizer, model_config["embedding_type"])
            
            # Extract token representations
            token_data, layers = self.extract_token_representations(handler, examples)
            all_model_data[model_name] = token_data
            
            if layers_to_extract is None:
                layers_to_extract = layers
            
            # Clean up GPU memory
            del handler, model
            torch.cuda.empty_cache()
        
        # Create visualization
        viz_path = os.path.join(output_dir, "token_role_separation_comparison.png")
        self.create_pca_visualization(all_model_data, layers_to_extract, viz_path)
        
        # Save raw data
        data_path = os.path.join(output_dir, "token_hidden_states_data.pkl")
        with open(data_path, "wb") as f:
            pickle.dump({
                "model_data": all_model_data,
                "layers": layers_to_extract,
                "examples": examples
            }, f)
        
        print(f"Analysis complete! Results saved to {output_dir}")
        return all_model_data, layers_to_extract


def main():
    parser = argparse.ArgumentParser(description="Token role separation visualization")
    parser.add_argument("--output_dir", type=str, default="./token_role_separation_results",
                       help="Directory to save results")
    parser.add_argument("--max_tokens", type=int, default=30,
                       help="Maximum tokens per role to visualize")
    
    args = parser.parse_args()
    
    # Define model configurations to compare
    models_config = [
        {
            "name": "Standard LM",
            "model_path": "Qwen/Qwen2.5-7B",
            "base_model": "Qwen/Qwen2.5-7B", 
            "embedding_type": "single_emb",
            "load_from_checkpoint": False
        },
        {
            "name": "ASIDE",
            "model_path": "models/Qwen2.5-7B/forward_rot/train_checkpoints/SFTv70/from_inst_run_ASIDE/last/",
            "base_model": "Qwen/Qwen2.5-7B",
            "embedding_type": "forward_rot",
            "load_from_checkpoint": True
        },
        {
            "name": "Ours (RGTNet-orthonly)",
            "model_path": "models/rgtnet_qwen2.5-7b_20250814_0759/merged_epoch_1/",
            "base_model": "Qwen/Qwen2.5-7B",
            "embedding_type": "rgtnet_orthonly",
            "load_from_checkpoint": True
        },
        {
            "name": "Ours (RGTNet-roleaware)",
            "model_path": "models/rgtnet_qwen2.5-7b_20250814_0759/merged_epoch_1/",
            "base_model": "Qwen/Qwen2.5-7B",
            "embedding_type": "rgtnet",
            "load_from_checkpoint": True
        }
    ]
    
    # Create visualizer and run analysis
    visualizer = TokenRoleSeparationVisualizer(models_config, args.max_tokens)
    visualizer.run_analysis(args.output_dir)


if __name__ == "__main__":
    main()
