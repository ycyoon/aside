"""
Orthogonal Role Analysis for Same Words/Phrases

This script analyzes whether embeddings of the same words maintain orthogonal relationships
when assigned different roles (instruction vs data) across layers in ASIDE and RGTNet models.

Expected behavior:
- ASIDE: Strong separation in early layers, gradual convergence in deeper layers
- RGTNet: Consistent separation maintained across all layers
"""

import sys
import os
import argparse
import json
import numpy as np
import torch
import types  # FIX: for MethodType
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
import pandas as pd
from tqdm import tqdm
import pickle
from datetime import datetime
from transformers import AutoTokenizer, AutoConfig, AutoModelForCausalLM
from types import SimpleNamespace
# Use your native implementation
from rgtnet_model import create_model as create_rgtnet, load_checkpoint as load_rgtnet_checkpoint
import glob
from rgtnet_loader import load_rgtnet_model
# Add parent directory to path
if "../.." not in sys.path:
    sys.path.append("../..")

from model_api import CustomModelHandler
from typing import Optional, Dict, Any


# Add RGTNet ModelHandler
class RGTNetModelHandler:
    def __init__(self, model, tokenizer, embedding_type="rgtnet"):
        self.model = model
        self.tokenizer = tokenizer
        self.supports_role_inputs = True
        self.embedding_type = embedding_type  # Set the correct embedding type
        self._hidden_states = []

    def _extract_logits_and_last_hidden(self, outputs):
        """Normalize various native outputs to (logits, last_hidden_state)."""
        logits = None
        last_hidden = None

        # dict-like
        if isinstance(outputs, dict):
            logits = outputs.get('logits', None)
            last_hidden = outputs.get('last_hidden_state', None)

        # HF-like object with attrs
        if logits is None and hasattr(outputs, 'logits'):
            logits = outputs.logits
        if last_hidden is None and hasattr(outputs, 'last_hidden_state'):
            last_hidden = outputs.last_hidden_state

        # raw tensor => treat as logits
        if logits is None and torch.is_tensor(outputs):
            logits = outputs

        # Fallbacks
        if last_hidden is None:
            last_hidden = logits

        return logits, last_hidden

    def encode_batch(self, texts, role_mask):
        """Encode batch of texts with role information"""
        inputs = self.tokenizer(texts, return_tensors="pt", padding=True, truncation=True)
        input_ids = inputs.input_ids.to(self.model.device)

        # Create proper role_mask if needed
        if role_mask is not None:
            if isinstance(role_mask, int):
                role_mask = torch.full_like(input_ids, role_mask)
            role_mask = role_mask.to(self.model.device)

        with torch.no_grad():
            outputs = self.model(input_ids=input_ids, role_mask=role_mask)
            _, last_hidden = self._extract_logits_and_last_hidden(outputs)
            return last_hidden

    def __call__(self, input_ids, role_mask=None, attention_mask=None, output_hidden_states=False, return_dict=True):
        """Call the model and collect hidden states for analysis"""
        self._hidden_states = []

        # Register hooks to collect intermediate layer outputs
        hooks = []
        if output_hidden_states and hasattr(self.model, 'layers'):
            def hook_fn(module, _in, out):
                # Support tuple outputs
                if isinstance(out, (tuple, list)):
                    out = out[0]
                self._hidden_states.append(out.detach().clone())

            for layer in self.model.layers:
                hooks.append(layer.register_forward_hook(hook_fn))

        try:
            outputs_raw = self.model(input_ids=input_ids, role_mask=role_mask, attention_mask=attention_mask)
            logits_tensor, last_hidden_tensor = self._extract_logits_and_last_hidden(outputs_raw)

            if not output_hidden_states:
                # Minimal HF-like shim without hidden states
                class OutputNoHidden:
                    def __init__(self, logits):
                        self.logits = logits
                        self.hidden_states = None
                return OutputNoHidden(logits_tensor)

            # Build first hidden state from embeddings
            embed_output = self.model.embedding(input_ids, role_mask)
            if getattr(self.model, "pos_encoder", None) is not None:
                pos_ids = torch.arange(0, input_ids.size(-1), device=input_ids.device).unsqueeze(0).expand(input_ids.size(0), -1)
                embed_output = embed_output + self.model.pos_encoder(pos_ids)

            # Compose all hidden states: [embedding] + per-layer hooks + [final]
            all_hidden_states = [embed_output] + self._hidden_states
            if last_hidden_tensor is not None and (len(all_hidden_states) == 0 or all_hidden_states[-1] is not last_hidden_tensor):
                all_hidden_states.append(last_hidden_tensor)

            class OutputWithHiddenStates:
                def __init__(self, logits, hidden_states):
                    self.logits = logits
                    self.hidden_states = hidden_states

            return OutputWithHiddenStates(logits_tensor, all_hidden_states)

        finally:
            for hook in hooks:
                hook.remove()



class OrthogonalRoleAnalyzer:
    """
    Analyzes orthogonal relationships between same words in different roles across layers.
    """
    
    def __init__(self, models_config, rgtnet_cfg: Optional[Dict[str, Any]] = None, *args, **kwargs):
        # 기존 초기화 로직이 있다면 유지하고, 아래 두 줄만 추가해 주세요.
        self.models_config = models_config
        # 기본값(dict) – runner에서 넘겨주지 않은 경우를 대비
        self.rgtnet_cfg = rgtnet_cfg or {
            "RGTNET_FORCE_BASIS": "rand",
            "RGTNET_ALPHA_OVERRIDE": 0.8,
            "RGTNET_ROTATION_MODE": "pure",
            "RGTNET_MIN_DELTA": 0.01,
            "RGTNET_FALLBACK_RANDOM": True,
            "RGTNET_ASSERT_MIN_ROT_DELTA": 0.002,
        }
        self.test_words = self._create_test_words()
        
    def _create_test_words(self):
        """Create a diverse set of 50+ test words and phrases."""
        return [
            # Math expressions
            "2+2", "12*3", "17+26", "(4+5)*6", "100-37", "36/6", "7*8", "3**4", "sqrt(16)", "9-3+2",
            
            # Common verbs
            "solve", "calculate", "compute", "explain", "analyze", "describe", "identify", "classify",
            "translate", "summarize", "compare", "evaluate", "generate", "create", "find", "determine",
            
            # Nouns and entities
            "problem", "question", "result", "answer", "solution", "method", "algorithm", "data",
            "information", "text", "document", "email", "report", "code", "function", "variable",
            
            # Phrases and sentences
            "Hello world", "Good morning", "Thank you", "Please help", "How are you", "What is this",
            "I need help", "Can you explain", "Show me how", "Tell me about", "Let me know",
            
            # Technical terms
            "machine learning", "artificial intelligence", "neural network", "deep learning",
            "gradient descent", "backpropagation", "transformer", "attention", "embedding",
            
            # Common adjectives
            "good", "bad", "fast", "slow", "big", "small", "important", "simple", "complex", "easy"
        ]
    
    def _call_model_safely(self, handler, input_ids, attention_mask=None, position_ids=None, 
                          segment_ids=None, role_mask=None, output_hidden_states=True, return_dict=True):
        """Safely call model handling both native RGTNet and HF models"""
        is_native_rgtnet = getattr(handler.model, 'is_native_rgtnet', False)
        
        if is_native_rgtnet:
            # Use the handler's __call__ method for native RGTNet
            return handler(
                input_ids=input_ids,
                role_mask=role_mask,
                attention_mask=attention_mask,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict
            )
        else:
            # HuggingFace or ASIDE/ForwardRot models
            kwargs = {
                'input_ids': input_ids,
                'attention_mask': attention_mask,
                'output_hidden_states': output_hidden_states,
                'return_dict': return_dict
            }
            
            # Add optional parameters only if provided
            if position_ids is not None:
                kwargs['position_ids'] = position_ids
            if segment_ids is not None:
                kwargs['segment_ids'] = segment_ids
            if role_mask is not None:
                kwargs['role_mask'] = role_mask
                
            return handler.model(**kwargs)

    def extract_same_word_embeddings(self, handler, layers_to_extract=None):
        """Extract embeddings with proper position_ids handling."""
        """
        Extract embeddings for the same words when used in instruction vs data roles.
        
        Args:
            handler: Model handler instance
            layers_to_extract: List of layer indices to extract
            
        Returns:
            dict: Embeddings organized by layer, word, and role
        """
        if layers_to_extract is None:
            # Default to analyzing multiple layers
            if hasattr(handler.model, 'model') and hasattr(handler.model.model, 'layers'):
                num_layers = len(handler.model.model.layers)
            elif hasattr(handler.model, 'layers'):
                num_layers = len(handler.model.layers)
            else:
                num_layers = 32  # Default assumption
        
        embeddings_data = {layer: {"instruction": [], "data": [], "words": []} 
                          for layer in layers_to_extract}
        
        embedding_type = getattr(handler, 'embedding_type', 'single_emb')
        # Choose correct device for inputs; for HF sharded models, use the device of embed tokens
        device = getattr(handler, 'input_device', None)
        if device is None:
            device = next(handler.model.parameters()).device
        
        for word_idx, word in enumerate(tqdm(self.test_words, desc="Processing test words")):
            # Create two versions: one as instruction, one as data
            # For instruction role: same word appears in instruction position
            # For data role: same word appears in data position
            
            # Instruction version: word is instruction, empty data
            instruction_text = word
            data_text = ""
            
            # Tokenize instruction version
            inst_tokens = handler.tokenizer(instruction_text, add_special_tokens=True, return_tensors="pt")
            inst_input_ids = inst_tokens["input_ids"].to(device)
            inst_attention_mask = inst_tokens["attention_mask"].to(device)
            
            # Create segment_ids for instruction version
            inst_segment_ids = None
            inst_role_mask = None
            
            if embedding_type in ("forward_rot", "ise", "rgtnet", "rgtnet_orthonly"):
                # All tokens are instruction (0)
                inst_segment_ids = torch.zeros_like(inst_input_ids).to(device)
                    
                if embedding_type in ("rgtnet", "rgtnet_orthonly"):
                    # role_mask: 0 for instruction tokens
                    inst_role_mask = torch.zeros_like(inst_input_ids).to(device)
            
            # Data version: word is data, empty instruction  
            instruction_text = ""
            data_text = word
            
            # Tokenize data version
            data_tokens = handler.tokenizer(data_text, add_special_tokens=True, return_tensors="pt")
            data_input_ids = data_tokens["input_ids"].to(device)
            data_attention_mask = data_tokens["attention_mask"].to(device)
            
            # Create segment_ids for data version  
            data_segment_ids = None
            data_role_mask = None
            
            if embedding_type in ("forward_rot", "ise", "rgtnet", "rgtnet_orthonly"):
                # All tokens are data (1)
                data_segment_ids = torch.ones_like(data_input_ids).to(device)
                
                if embedding_type in ("rgtnet", "rgtnet_orthonly"):
                    # role_mask: 1 for data tokens
                    data_role_mask = torch.ones_like(data_input_ids).to(device)            # Debug output for first few words
            if word_idx < 3:
                print(f"\n[DEBUG] Word: '{word}'")
                print(f"Instruction version - input_ids: {inst_input_ids}")
                print(f"Instruction version - segment_ids: {inst_segment_ids}")
                print(f"Instruction version - role_mask: {inst_role_mask}")
                print(f"Data version - input_ids: {data_input_ids}")
                print(f"Data version - segment_ids: {data_segment_ids}")
                print(f"Data version - role_mask: {data_role_mask}")
            
            # Extract embeddings for instruction version
            handler.model.eval()
            with torch.no_grad():
                # Auto-generate position_ids if needed
                inst_position_ids = torch.arange(0, inst_input_ids.size(-1), dtype=torch.long, device=device).unsqueeze(0)
                data_position_ids = torch.arange(0, data_input_ids.size(-1), dtype=torch.long, device=device).unsqueeze(0)
                
                supports_role = getattr(handler, 'supports_role_inputs', False)
                # For ForwardRot models, we need segment_ids even if supports_role is False
                needs_segment_ids = embedding_type in ("forward_rot", "ise")
                
                # Instruction version forward pass
                if (supports_role and inst_segment_ids is not None) or (needs_segment_ids and inst_segment_ids is not None):
                    # Pass role-specific tensors only if model supports them
                    if inst_role_mask is not None:
                        inst_outputs = self._call_model_safely(
                            handler, inst_input_ids, inst_attention_mask, inst_position_ids,
                            inst_segment_ids, inst_role_mask
                        )
                    else:
                        inst_outputs = self._call_model_safely(
                            handler, inst_input_ids, inst_attention_mask, inst_position_ids,
                            inst_segment_ids, None
                        )
                else:
                    inst_outputs = self._call_model_safely(
                        handler, inst_input_ids, inst_attention_mask, inst_position_ids,
                        None, inst_role_mask
                    )
                
                # Extract embeddings for data version
                if (supports_role and data_segment_ids is not None) or (needs_segment_ids and data_segment_ids is not None):
                    if data_role_mask is not None:
                        data_outputs = self._call_model_safely(
                            handler, data_input_ids, data_attention_mask, data_position_ids,
                            data_segment_ids, data_role_mask
                        )
                    else:
                        data_outputs = self._call_model_safely(
                            handler, data_input_ids, data_attention_mask, data_position_ids,
                            data_segment_ids, None
                        )
                else:
                    data_outputs = self._call_model_safely(
                        handler, data_input_ids, data_attention_mask, data_position_ids,
                        None, data_role_mask
                    )
            
            # Extract hidden states for specified layers
            inst_hidden_states = inst_outputs.hidden_states
            data_hidden_states = data_outputs.hidden_states
            
            for layer_idx in layers_to_extract:
                if layer_idx < len(inst_hidden_states) and layer_idx < len(data_hidden_states):
                    # Get layer hidden states
                    inst_layer_hidden = inst_hidden_states[layer_idx].squeeze(0).cpu().to(torch.float32)  # [seq_len, hidden_size]
                    data_layer_hidden = data_hidden_states[layer_idx].squeeze(0).cpu().to(torch.float32)  # [seq_len, hidden_size]
                    
                    # For instruction version: extract embeddings from instruction tokens (role_mask == 0)
                    if inst_role_mask is not None:
                        inst_mask = (inst_role_mask[0] == 0).cpu()  # instruction tokens
                        if inst_mask.any():
                            inst_embedding = inst_layer_hidden[inst_mask].mean(dim=0)  # mean pool instruction tokens
                        else:
                            inst_embedding = inst_layer_hidden.mean(dim=0)  # fallback to all tokens
                            print(f"[WARNING] No instruction tokens found for word '{word}' in instruction version")
                    else:
                        # Fallback: use all tokens
                        inst_embedding = inst_layer_hidden.mean(dim=0)
                    
                    # For data version: extract embeddings from data tokens (role_mask == 1)
                    if data_role_mask is not None:
                        data_mask = (data_role_mask[0] == 1).cpu()  # data tokens
                        if data_mask.any():
                            data_embedding = data_layer_hidden[data_mask].mean(dim=0)  # mean pool data tokens
                        else:
                            data_embedding = data_layer_hidden.mean(dim=0)  # fallback to all tokens
                            print(f"[WARNING] No data tokens found for word '{word}' in data version")
                    else:
                        # Fallback: use all tokens
                        data_embedding = data_layer_hidden.mean(dim=0)
                    
                    embeddings_data[layer_idx]["instruction"].append(inst_embedding)
                    embeddings_data[layer_idx]["data"].append(data_embedding)
                    embeddings_data[layer_idx]["words"].append(word)
                    
                    # Debug output for first word, first layer
                    if word_idx < 3 and layer_idx == layers_to_extract[0]:
                        print(f"[DEBUG] Layer {layer_idx}, Word '{word}':")
                        print(f"  Instruction embedding shape: {inst_embedding.shape}")
                        print(f"  Data embedding shape: {data_embedding.shape}")
                        print(f"  Cosine similarity: {torch.cosine_similarity(inst_embedding.unsqueeze(0), data_embedding.unsqueeze(0), dim=1).item():.6f}")
                        print(f"  Euclidean distance: {torch.norm(inst_embedding - data_embedding).item():.6f}")
        
        return embeddings_data
    
    def calculate_orthogonality_metrics(self, embeddings_data, layers_to_extract):
        """
        Calculate various metrics to measure orthogonality and separation.
        
        Args:
            embeddings_data: Dictionary of embeddings by layer
            layers_to_extract: List of layer indices
            
        Returns:
            dict: Metrics for each layer
        """
        metrics = {}
        
        for layer_idx in layers_to_extract:
            if layer_idx not in embeddings_data:
                print(f"⚠️ Warning: Layer {layer_idx} not found in embeddings_data, skipping...")
                continue
                
            layer_data = embeddings_data[layer_idx]
            inst_embeddings = torch.stack(layer_data["instruction"])
            data_embeddings = torch.stack(layer_data["data"])
            
            # Calculate cosine similarities between same words in different roles
            cosine_sims = []
            for i in range(len(inst_embeddings)):
                inst_emb = inst_embeddings[i]
                data_emb = data_embeddings[i]
                
                # Normalize embeddings
                inst_norm = inst_emb / (inst_emb.norm() + 1e-8)
                data_norm = data_emb / (data_emb.norm() + 1e-8)
                
                # Calculate cosine similarity
                cosine_sim = torch.dot(inst_norm, data_norm).item()
                cosine_sims.append(cosine_sim)
            
            # Calculate Euclidean distances
            euclidean_dists = []
            for i in range(len(inst_embeddings)):
                dist = torch.norm(inst_embeddings[i] - data_embeddings[i]).item()
                euclidean_dists.append(dist)
            
            # Combine all embeddings for silhouette analysis
            all_embeddings = torch.cat([inst_embeddings, data_embeddings], dim=0).numpy()
            labels = [0] * len(inst_embeddings) + [1] * len(data_embeddings)  # 0=instruction, 1=data
            
            # Calculate silhouette score (higher = better separation)
            try:
                silhouette = silhouette_score(all_embeddings, labels)
            except:
                silhouette = 0.0
            
            metrics[layer_idx] = {
                "cosine_similarities": cosine_sims,
                "mean_cosine_similarity": np.mean(cosine_sims),
                "std_cosine_similarity": np.std(cosine_sims),
                "euclidean_distances": euclidean_dists,
                "mean_euclidean_distance": np.mean(euclidean_dists),
                "std_euclidean_distance": np.std(euclidean_dists),
                "silhouette_score": silhouette,
                "num_words": len(inst_embeddings)
            }
        
        return metrics
    
    def create_visualization(self, all_model_data, layers_to_extract, output_path):
        """
        Create visualization similar to the provided image but for ASIDE vs RGTNet only.
        
        Args:
            all_model_data: Dictionary containing embeddings for all models
            layers_to_extract: List of layer indices
            output_path: Path to save the visualization
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
        }
        
        # Select subset of words for visualization (focus on multi-token phrases that show separation)
        viz_words = ["2+2", "12*3", "17+26", "Hello world", "What is", "The answer"]
        
        for model_idx, (model_name, model_embeddings) in enumerate(all_model_data.items()):
            for layer_idx, layer in enumerate(layers_to_extract):
                ax = axes[model_idx, layer_idx]
                
                if layer not in model_embeddings:
                    print(f"⚠️ Warning: Layer {layer} not found in {model_name} embeddings, skipping visualization...")
                    continue
                    
                layer_data = model_embeddings[layer]
                inst_embeddings = torch.stack(layer_data["instruction"])
                data_embeddings = torch.stack(layer_data["data"])
                words = layer_data["words"]
                
                # Combine embeddings for PCA
                all_embeddings = torch.cat([inst_embeddings, data_embeddings], dim=0).numpy()
                
                # Apply PCA
                pca = PCA(n_components=2, random_state=42)
                pca_result = pca.fit_transform(all_embeddings)
                
                # Split back into instruction and data
                inst_pca = pca_result[:len(inst_embeddings)]
                data_pca = pca_result[len(inst_embeddings):]
                
                # Plot instruction embeddings with larger size and lower alpha
                ax.scatter(inst_pca[:, 0], inst_pca[:, 1], 
                          c=colors['instruction'], alpha=0.8, s=80, 
                          label='Instruction tokens', zorder=3, marker='o', edgecolors='black', linewidth=0.5)
                
                # Plot data embeddings with different marker and higher alpha
                ax.scatter(data_pca[:, 0], data_pca[:, 1],
                          c=colors['data'], alpha=0.9, s=70,
                          label='User/Data tokens', zorder=2, marker='^', edgecolors='black', linewidth=0.5)
                
                # Add annotations for selected words
                for i, word in enumerate(words):
                    if word in viz_words:
                        # Draw line connecting same word in different roles
                        ax.plot([inst_pca[i, 0], data_pca[i, 0]], 
                               [inst_pca[i, 1], data_pca[i, 1]], 
                               'k--', alpha=0.3, linewidth=1)
                        
                        # Annotate the word
                        mid_x = (inst_pca[i, 0] + data_pca[i, 0]) / 2
                        mid_y = (inst_pca[i, 1] + data_pca[i, 1]) / 2
                        ax.annotate(f'"{word}"', (mid_x, mid_y), 
                                  fontsize=8, alpha=0.8, ha='center',
                                  xytext=(0, 10), textcoords='offset points')
                
                # Set title and labels
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
        #plt.suptitle("Comparison of Token Role Separation Across Layers", 
        #            fontsize=16, y=0.98)
        
        # Add subtitle
        #fig.text(0.5, 0.02, "Illustrative 2D Subspace", 
        #        ha='center', fontsize=12, style='italic')
        
        plt.tight_layout()
        plt.subplots_adjust(top=0.92, bottom=0.08)
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Orthogonal role analysis visualization saved to {output_path}")
    
    def create_metrics_plot(self, all_metrics, layers_to_extract, output_path):
        """
        Create plots showing quantitative metrics across layers.
        """
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Prepare data for plotting
        models = list(all_metrics.keys())
        layers = layers_to_extract
        
        # Plot 1: Mean Cosine Similarity (lower = more orthogonal)
        ax = axes[0, 0]
        for model in models:
            similarities = [all_metrics[model][layer]["mean_cosine_similarity"] for layer in layers]
            ax.plot(layers, similarities, marker='o', label=model, linewidth=2)
        ax.set_xlabel("Layer")
        ax.set_ylabel("Mean Cosine Similarity")
        ax.set_title("Cosine Similarity Between Same Words\n(Lower = More Orthogonal)")
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Plot 2: Mean Euclidean Distance (higher = more separated)
        ax = axes[0, 1]
        for model in models:
            distances = [all_metrics[model][layer]["mean_euclidean_distance"] for layer in layers]
            ax.plot(layers, distances, marker='s', label=model, linewidth=2)
        ax.set_xlabel("Layer")
        ax.set_ylabel("Mean Euclidean Distance")
        ax.set_title("Euclidean Distance Between Same Words\n(Higher = More Separated)")
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Plot 3: Silhouette Score (higher = better separation)
        ax = axes[1, 0]
        for model in models:
            scores = [all_metrics[model][layer]["silhouette_score"] for layer in layers]
            ax.plot(layers, scores, marker='^', label=model, linewidth=2)
        ax.set_xlabel("Layer")
        ax.set_ylabel("Silhouette Score")
        ax.set_title("Silhouette Score for Role Separation\n(Higher = Better Separation)")
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Plot 4: Distribution of cosine similarities
        ax = axes[1, 1]
        for model in models:
            # Get all cosine similarities for the last layer
            last_layer = layers_to_extract[-1]
            sims = all_metrics[model][last_layer]["cosine_similarities"]
            ax.hist(sims, alpha=0.6, label=f"{model} (Layer {last_layer})", bins=20)
        ax.set_xlabel("Cosine Similarity")
        ax.set_ylabel("Frequency")
        ax.set_title("Distribution of Cosine Similarities\n(Final Layer)")
        ax.legend()
        ax.axvline(x=0, color='red', linestyle='--', alpha=0.7, label='Perfect Orthogonal')
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Metrics plots saved to {output_path}")
    
    def run_analysis(self, output_dir):
        """Run analysis with proper native loading."""
        os.makedirs(output_dir, exist_ok=True)

        all_metrics = {}
        all_model_embeddings = {}  # collect per-model embeddings for visualization
        layers_to_extract = None

        for model_config in tqdm(self.models_config, desc="Processing models"):
            model_name = model_config["name"]
            print(f"\nProcessing {model_name}...")

            handler = self._load_model_handler(model_config)

            # Model type info
            is_native = getattr(getattr(handler, "model", None), "is_native_rgtnet", False)
            print(f"[MODEL CHECK] {model_name} - Native RGTNet: {is_native}")

            # Determine number of transformer layers robustly
            num_layers = None
            if hasattr(handler.model, "config") and hasattr(handler.model.config, "num_hidden_layers"):
                num_layers = handler.model.config.num_hidden_layers
            elif hasattr(handler.model, "layers"):
                num_layers = len(handler.model.layers)
            elif hasattr(handler.model, "model") and hasattr(handler.model.model, "layers"):
                num_layers = len(handler.model.model.layers)
            else:
                num_layers = 32  # sensible default

            # Choose 0 (embedding), middle, last layer indices
            layers_to_extract = [0, max(1, num_layers // 2), max(1, num_layers - 1)]

            embeddings_data = self.extract_same_word_embeddings(handler, layers_to_extract)
            all_model_embeddings[model_name] = embeddings_data

            # Metrics
            metrics = self.calculate_orthogonality_metrics(embeddings_data, layers_to_extract)
            all_metrics[model_name] = metrics

            # Cleanup
            del handler
            torch.cuda.empty_cache()

        print(f"📊 Using common layers for visualization: {sorted(layers_to_extract)}")

        # Visualizations
        viz_path = os.path.join(output_dir, "orthogonal_role_separation.png")
        self.create_visualization(all_model_embeddings, layers_to_extract, viz_path)

        metrics_path = os.path.join(output_dir, "orthogonality_metrics.png")
        self.create_metrics_plot(all_metrics, layers_to_extract, metrics_path)

        # Save raw data
        data_path = os.path.join(output_dir, "orthogonal_analysis_data.pkl")
        with open(data_path, "wb") as f:
            pickle.dump({
                "embeddings": all_model_embeddings,
                "metrics": all_metrics,
                "layers": layers_to_extract,
                "test_words": self.test_words
            }, f)

        self._generate_summary_report(all_metrics, layers_to_extract, output_dir)
        print(f"Orthogonal role analysis complete! Results saved to {output_dir}")
        return all_model_embeddings, all_metrics, layers_to_extract
    
    def _generate_summary_report(self, all_metrics, layers_to_extract, output_dir):
        """Generate a text summary of the analysis results."""
        report_path = os.path.join(output_dir, "analysis_summary.txt")
        
        with open(report_path, "w") as f:
            f.write("Orthogonal Role Analysis Summary\n")
            f.write("=" * 50 + "\n\n")
            
            f.write(f"Test conducted on {len(self.test_words)} words/phrases\n")
            f.write(f"Layers analyzed: {layers_to_extract}\n")
            f.write(f"Models compared: {list(all_metrics.keys())}\n\n")
            
            for model_name, metrics in all_metrics.items():
                f.write(f"\n{model_name} Results:\n")
                f.write("-" * 30 + "\n")
                
                for layer in layers_to_extract:
                    layer_metrics = metrics[layer]
                    f.write(f"Layer {layer}:\n")
                    f.write(f"  Mean Cosine Similarity: {layer_metrics['mean_cosine_similarity']:.4f} ± {layer_metrics['std_cosine_similarity']:.4f}\n")
                    f.write(f"  Mean Euclidean Distance: {layer_metrics['mean_euclidean_distance']:.4f} ± {layer_metrics['std_euclidean_distance']:.4f}\n")
                    f.write(f"  Silhouette Score: {layer_metrics['silhouette_score']:.4f}\n")
                    f.write("\n")
                
                # Calculate trends
                first_layer = layers_to_extract[0]
                last_layer = layers_to_extract[-1]
                cosine_change = metrics[last_layer]['mean_cosine_similarity'] - metrics[first_layer]['mean_cosine_similarity']
                distance_change = metrics[last_layer]['mean_euclidean_distance'] - metrics[first_layer]['mean_euclidean_distance']
                
                f.write(f"Trends from Layer {first_layer} to {last_layer}:\n")
                f.write(f"  Cosine Similarity Change: {cosine_change:+.4f} ({'convergence' if cosine_change > 0 else 'divergence'})\n")
                f.write(f"  Distance Change: {distance_change:+.4f} ({'separation' if distance_change > 0 else 'convergence'})\n")
                f.write("\n")
        
        print(f"Summary report saved to {report_path}")
    
    def _load_model_handler(self, model_config):
        """
        Load model with native enforcement if specified.
        """
        force_native = model_config.get("force_native", False)
        
        if model_config["embedding_type"] == "rgtnet" and force_native:
            # Use native RGTNet loader for proper role separation
            print(f"🚀 [NATIVE RGTNET] Loading native RGTNet model for {model_config['name']}")
            return self._load_native_rgtnet(model_config)
        else:
            # Use standard CustomModelHandler for non-RGTNet or when not forcing native
            print(f"📦 [STANDARD] Loading {model_config['name']} with CustomModelHandler")
            return CustomModelHandler(
                checkpoint_path=model_config["model_path"],
                instruct_model_path=None,
                data_model_path=None,
                tokenizer_path=model_config["base_model"],
                embedding_type=model_config["embedding_type"],
                load_from_checkpoint=model_config.get("load_from_checkpoint", True),
            )

    def _is_native_rgtnet_checkpoint(self, model_path: str) -> bool:
        """Robust native RGTNet signature detection."""
        import os, glob, torch

        # 1) Config file is definitive
        if os.path.exists(os.path.join(model_path, "rgtnet_config.json")):
            return True

        # 2) Inspect a shard for native-specific keys
        shard_files = sorted(glob.glob(os.path.join(model_path, "pytorch_model*.bin")))
        if not shard_files:
            return False

        try:
            sd = torch.load(shard_files[0], map_location="cpu")
            keys = list(sd.keys())

            # Accept if any of these are present
            has_U = any(k.endswith(".self_attn.U") for k in keys)
            has_role_lin = any(k.startswith("embedding.role_transformers.") for k in keys)
            has_pos = any(k == "pos_encoder.weight" for k in keys)

            return has_U or has_role_lin or has_pos
        except Exception:
            return False

    def _cfg(self, key: str, default=None, cast: str | None = None):
        """
        Fetch config value from self.rgtnet_cfg only (no environment).
        cast: "int" | "float" | "bool"
        """
        v = self.rgtnet_cfg.get(key, default)
        if cast == "float":
            try:
                return float(v)
            except Exception:
                return float(default) if default is not None else None
        if cast == "int":
            try:
                return int(v)
            except Exception:
                return int(default) if default is not None else None
        if cast == "bool":
            if isinstance(v, str):
                return v.strip().lower() in ("1", "true", "yes", "y", "t")
            return bool(v)
        return v

    def _inject_embedding_rotation(self, model, alpha_override=None):
        """
        Inject/configure embedding rotation for analysis.
        alpha_override: optional float; if None, use self.rgtnet_cfg['RGTNET_ALPHA_OVERRIDE'].
        """
        # Read config (no env)
        force_basis = self._cfg("RGTNET_FORCE_BASIS", "rand")
        if alpha_override is None:
            alpha_override = self._cfg("RGTNET_ALPHA_OVERRIDE", 0.3, "float")
        else:
            alpha_override = float(alpha_override)
        rot_mode = self._cfg("RGTNET_ROTATION_MODE", "pure")
        min_delta = self._cfg("RGTNET_MIN_DELTA", 0.0, "float")
        fallback_rand = self._cfg("RGTNET_FALLBACK_RANDOM", False, "bool")
        assert_min = self._cfg("RGTNET_ASSERT_MIN_ROT_DELTA", 0.0, "float")

        emb = getattr(model, "embedding", None)
        if emb is None:
            print("[RGTNet Native] ❌ No embedding module found; cannot inject rotation")
            return False

        # 1) layer0에서 U 추출
        U = None
        try:
            attn0 = model.layers[0].self_attn if hasattr(model, "layers") else None
            U = getattr(attn0, "U", None)
            if isinstance(U, torch.nn.Parameter):
                U = U.data
        except Exception:
            U = None

        if U is not None:
            print(f"[RGTNet Native] rotation basis found: {tuple(U.shape)}")
        else:
            print("[RGTNet Native] ⚠️ rotation basis U not found")

        # 2) 항상 래핑(네이티브 set_alpha가 있어도 무조건 주입)
        if not hasattr(emb, "_orig_forward"):
            emb._orig_forward = emb.forward  # 보존

        orig_forward = emb._orig_forward

        # 버퍼/상태 등록
        if not hasattr(emb, "_rotation_injected"):
            emb.register_buffer("_rot_alpha", torch.tensor(float(alpha_override)))
            emb._rotation_injected = True
        else:
            emb._rot_alpha.fill_(float(alpha_override))
        emb._rot_basis = U
        emb._force_mode = force_basis
        emb._rot_mode = rot_mode

        def _make_forced_basis(D: int, target_device, target_dtype, H: int = None):
            """
            Create an orthogonal basis without using bf16 CUDA QR (unsupported).
            Compute QR on CPU float32, then cast/move to target.
            """
            cpu = torch.device("cpu")
            if force_basis == "negi":
                if H and D % H == 0:
                    dh = D // H
                    return -torch.eye(dh, device=target_device, dtype=target_dtype).unsqueeze(0).expand(H, dh, dh).contiguous()
                return -torch.eye(D, device=target_device, dtype=target_dtype)

            if force_basis == "rand":
                if H and D % H == 0:
                    dh = D // H
                    Qs = []
                    for _ in range(H):
                        A = torch.randn(dh, dh, device=cpu, dtype=torch.float32)
                        q, _ = torch.linalg.qr(A)  # CPU fp32
                        Qs.append(q.to(device=target_device, dtype=target_dtype))
                    return torch.stack(Qs, dim=0).contiguous()
                A = torch.randn(D, D, device=cpu, dtype=torch.float32)
                q, _ = torch.linalg.qr(A)  # CPU fp32
                return q.to(device=target_device, dtype=target_dtype).contiguous()
            return None

        def rotated_forward(self, input_ids, role_mask=None, *args, **kwargs):
            # 네이티브 role_mask 경로는 무시: base embedding만 얻음
            try:
                x = orig_forward(input_ids, role_mask=None, *args, **kwargs)  # [B,T,D]
            except TypeError:
                x = orig_forward(input_ids, *args, **kwargs)

            if role_mask is None or float(self._rot_alpha.item()) == 0.0:
                return x

            B, T, D = x.shape
            m = role_mask.to(x.device).unsqueeze(-1).to(x.dtype)
            U_local = self._rot_basis

            # 강제 basis 생성/선택
            if self._force_mode:
                H = None
                if isinstance(U_local, torch.Tensor) and U_local.dim() == 3:
                    H = U_local.shape[0]
                elif hasattr(model, "nhead"):
                    H = int(getattr(model, "nhead"))
                # cache forced basis to avoid repeated QR
                cached = getattr(self, "_forced_basis_cache", None)
                if cached is not None and isinstance(cached, torch.Tensor) and cached.device == x.device and cached.dtype == x.dtype:
                    U_dev = cached
                else:
                    U_dev = _make_forced_basis(D, x.device, x.dtype, H)
                    self._forced_basis_cache = U_dev
            else:
                U_dev = U_local.to(x.device, dtype=x.dtype) if isinstance(U_local, torch.Tensor) else None

            if U_dev is None:
                return x

            alpha = self._rot_alpha
            mode = self._rot_mode

            # 진단 로그(한 번만 출력하고 싶으면 필요 시 가드 추가)
            try:
                if U_dev.dim() == 2 and U_dev.shape == (D, D):
                    delta = (U_dev - torch.eye(D, device=x.device, dtype=x.dtype)).abs().mean().item()
                    print(f"[DEBUG U] 2D mean|U-I|={delta:.6e}")
                elif U_dev.dim() == 3:
                    H_, dh, _ = U_dev.shape
                    eye = torch.eye(dh, device=x.device, dtype=x.dtype)
                    delta = (U_dev - eye).abs().mean().item()
                    print(f"[DEBUG U] 3D mean|U-I|={delta:.6e} (H={H_}, dh={dh})")
            except Exception:
                pass

            # 2D
            if U_dev.dim() == 2 and U_dev.shape == (D, D):
                core = x @ U_dev
                x_new = core if mode == "pure" else (1.0 - alpha) * x + alpha * core
                return x * (1.0 - m) + x_new * m

            # 3D per-head
            if U_dev.dim() == 3:
                H_, dh1, dh2 = U_dev.shape
                if dh1 == dh2 and D % H_ == 0 and (D // H_) == dh1:
                    xv = x.view(B, T, H_, dh1)
                    core = torch.einsum('bthd,hde->bthe', xv, U_dev).reshape(B, T, D)
                    x_new = core if mode == "pure" else (1.0 - alpha) * x + alpha * core
                    return x * (1.0 - m) + x_new * m

            return x

        # 바인딩
        emb.forward = types.MethodType(rotated_forward, emb)

        # 알파 setter는 유지(디버그 편의)
        def set_alpha_fn(self, val: float):
            self._rot_alpha.data = torch.tensor(float(val), device=self._rot_alpha.device)
        def get_alpha_fn(self):
            return float(self._rot_alpha.item())
        emb.set_alpha = types.MethodType(set_alpha_fn, emb)
        emb.get_alpha = types.MethodType(get_alpha_fn, emb)

        print(f"[RGTNet Native] ✅ injected embedding rotation wrapper (alpha={alpha_override}, force={force_basis or 'none'}, mode={rot_mode})")
        return True

    def _sanity_check_layer0(self, model, tokenizer, alpha_value: float):
        try:
            text = "Hello world"
            ids = tokenizer(text, return_tensors="pt")["input_ids"].to(model.device)
            rm0 = torch.zeros_like(ids); rm1 = torch.ones_like(ids)
            with torch.no_grad():
                e0 = model.embedding(ids, rm0)
                e1 = model.embedding(ids, rm1)
                cos = torch.nn.functional.cosine_similarity(e0.flatten(1), e1.flatten(1), dim=1).mean().item()
                l2 = (e1 - e0).norm(p=2, dim=-1).mean().item()
            print(f"[SANITY L0] alpha={alpha_value} | cos={cos:.4f} | mean||Δ||={l2:.6f}")
        except Exception as e:
            print(f"[SANITY L0] failed: {e}")

    def _load_native_rgtnet(self, model_config):
        """Load native RGTNet and enforce embedding rotation."""
        import os, json, torch
        from transformers import AutoTokenizer, AutoConfig

        # Robust import of native model creator/loader
        try:
            from rgtnet_model import create_model as create_rgtnet, load_checkpoint as load_rgtnet_checkpoint  # experiments/
        except Exception:
            try:
                from experiments.rgtnet_model import create_model as create_rgtnet, load_checkpoint as load_rgtnet_checkpoint
            except Exception:
                from model import create_model as create_rgtnet, load_checkpoint as load_rgtnet_checkpoint  # fallback

        model_dir = model_config["model_path"]
        base_model = model_config.get("base_model")
        user_alpha = float(model_config.get("alpha_embedding", 0.3))
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32

        cfg_path = os.path.join(model_dir, "rgtnet_config.json")
        if os.path.exists(cfg_path):
            with open(cfg_path) as f:
                cfg = json.load(f)
            chosen_alpha = float(cfg.get("alpha_embed", cfg.get("alpha_embedding", user_alpha)))
            args = SimpleNamespace(
                vocab_size=cfg["vocab_size"],
                d_model=cfg["d_model"],
                nhead=cfg["nhead"],
                num_layers=cfg["num_layers"],
                dim_feedforward=cfg["dim_feedforward"],
                dropout=cfg.get("dropout", 0.1),
                bias_delta=cfg.get("bias_delta", 5.0),
                max_seq_len=cfg.get("max_seq_len", 2048),
                gradient_checkpointing=False,
                num_key_value_heads=cfg.get("num_key_value_heads", None),
                architecture_type=cfg.get("architecture_type", "llama"),
                mlp_type=cfg.get("mlp_type", "gated"),
                activation=cfg.get("activation", "silu"),
                attention_bias=cfg.get("attention_bias", False),
                mlp_bias=cfg.get("mlp_bias", False),
                alpha_embedding=chosen_alpha,
            )
            pad_id = cfg.get("pad_token_id", 0)
        else:
            if not base_model:
                raise RuntimeError("Missing base_model for native RGTNet fallback without rgtnet_config.json")
            hf_cfg = AutoConfig.from_pretrained(base_model, trust_remote_code=True)
            chosen_alpha = user_alpha
            args = SimpleNamespace(
                vocab_size=hf_cfg.vocab_size,
                d_model=hf_cfg.hidden_size,
                nhead=hf_cfg.num_attention_heads,
                num_layers=hf_cfg.num_hidden_layers,
                dim_feedforward=hf_cfg.intermediate_size,
                dropout=0.1,
                bias_delta=5.0,
                max_seq_len=getattr(hf_cfg, "max_position_embeddings", 2048),
                gradient_checkpointing=False,
                num_key_value_heads=getattr(hf_cfg, "num_key_value_heads", None),
                architecture_type=getattr(hf_cfg, "model_type", "llama"),
                mlp_type="gated",
                activation="silu",
                attention_bias=False,
                mlp_bias=False,
                alpha_embedding=chosen_alpha,
            )
            pad_id = getattr(hf_cfg, "pad_token_id", 0) or 0

        # Tokenizer
        tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        # Build and load
        print(f"[RGTNet Native] alpha_embedding (requested) = {args.alpha_embedding}")
        model = create_rgtnet(args, pad_idx=pad_id)
        model = load_rgtnet_checkpoint(model_dir, model, device=str(device))
        model = model.to(device=device, dtype=dtype).eval()
        setattr(model, "is_native_rgtnet", True)
        setattr(model, "device", device)
        model.config = SimpleNamespace(num_hidden_layers=args.num_layers)

        # Enforce rotation on embedding (inject if needed)
        self._inject_embedding_rotation(getattr(model, "module", model))

        # Sanity log
        self._sanity_check_layer0(getattr(model, "module", model), tokenizer, args.alpha_embedding)

        # Wrap handler
        handler = RGTNetModelHandler(model, tokenizer, embedding_type="rgtnet")
        return handler

    def _load_hf_causal_lm(self, model_config):
        """Load a standard HuggingFace Causal LM from a checkpoint directory."""
        checkpoint_path = model_config["model_path"]
        base_model = model_config["base_model"]

        tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        print(f"[HF] Loading model from {checkpoint_path}")
        model = AutoModelForCausalLM.from_pretrained(
            checkpoint_path,
            device_map="auto",
            trust_remote_code=True
        )

        setattr(model, "is_native_rgtnet", False)

        # Determine input device for sharded setups
        input_device = None
        dev_map = getattr(model, 'hf_device_map', None)
        if isinstance(dev_map, dict):
            for key in ('model.embed_tokens', 'transformer.wte', 'model.decoder.embed_tokens'):
                if key in dev_map:
                    input_device = torch.device(dev_map[key])
                    break
        if input_device is None:
            try:
                input_device = next(model.parameters()).device
            except StopIteration:
                input_device = torch.device('cpu')

        class HFModelHandler:
            def __init__(self, model, tokenizer, embedding_type):
                self.model = model
                self.tokenizer = tokenizer
                self.embedding_type = embedding_type
                self.max_token_len = getattr(model.config, 'max_position_embeddings', 4096)
                self.supports_role_inputs = False
                self.input_device = input_device

        return HFModelHandler(model, tokenizer, model_config["embedding_type"])
