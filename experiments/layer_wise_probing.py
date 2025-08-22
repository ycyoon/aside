"""
Layer-wise Probing Experiment for Role Classification

This script evaluates how well different model architectures (RGT, ASIDE, Vanilla)
maintain role information (instruction vs data) across their internal layers.

The experiment measures whether models internally distinguish between instruction
and data tokens, not just in their final outputs.

Key Concepts:
- **Layer-wise Analysis**: Extract hidden states from each transformer layer
- **Role Classification**: Train linear classifiers to predict token roles
- **Architectural Comparison**: Compare RGT, ASIDE, and vanilla models

Expected Results:
- RGT: High accuracy from Layer 0, maintained across all layers
- ASIDE: High initial accuracy, gradual decline in deeper layers  
- Vanilla: Low initial accuracy, gradual improvement but lower peak

Usage:
    python layer_wise_probing.py --model_path models/rgtnet_model \
                                 --embedding_type rgtnet \
                                 --output_dir ./probing_results
"""

import sys
import os
import argparse
import json
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import pickle
from datetime import datetime

# Add parent directory to path
if "../.." not in sys.path:
    sys.path.append("../..")

from model_api import CustomModelHandler, format_prompt

# Fix PyTorch Dynamo logger incompatibility
try:
    torch._dynamo.config.suppress_errors = True
except:
    pass

class RoleProbingDataset:
    """
    Dataset for role probing experiments with challenging examples.
    
    Creates instruction-data pairs where the same words appear in both contexts,
    making role classification more challenging and meaningful.
    """
    
    def __init__(self, tokenizer, max_length=512):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.examples = self._create_challenging_examples()
        
    def _create_challenging_examples(self):
        """
        Create challenging examples where same words appear in instruction and data.
        
        These examples test whether models rely on architectural role separation
        rather than simple word-based heuristics.
        """
        examples = []
        
        # Template-based examples with word overlap
        templates = [
            {
                "instruction": "Summarize the following text about {topic}:",
                "data": "The research on {topic} shows significant progress. {topic} has been studied extensively by researchers worldwide. Recent developments in {topic} indicate promising future applications.",
                "topics": ["AI", "climate change", "medicine", "economics", "technology"]
            },
            {
                "instruction": "Translate the following {language} text to English:",
                "data": "This is a sample text in {language}. The {language} language has many unique features.",
                "topics": ["French", "Spanish", "German", "Italian", "Portuguese"]
            },
            {
                "instruction": "Analyze the sentiment of this {context} review:",
                "data": "This {context} experience was quite remarkable. The {context} exceeded my expectations in many ways.",
                "topics": ["restaurant", "movie", "book", "product", "service"]
            },
            {
                "instruction": "Extract key information from this {domain} document:",
                "data": "The {domain} report indicates several important findings. This {domain} analysis covers multiple aspects of the subject.",
                "topics": ["financial", "medical", "legal", "technical", "academic"]
            },
            {
                "instruction": "Classify the following {category} content:",
                "data": "This {category} material contains various elements. The {category} structure follows standard conventions.",
                "topics": ["news", "academic", "commercial", "personal", "official"]
            }
        ]
        
        # Generate examples from templates
        for template in templates:
            for topic in template["topics"]:
                instruction = template["instruction"].format(topic=topic)
                data = template["data"].format(topic=topic, language=topic, context=topic, domain=topic, category=topic)
                
                examples.append({
                    "instruction": instruction,
                    "data": data,
                    "template_type": "challenging"
                })
        
        # Add simple examples for baseline
        simple_examples = [
            {
                "instruction": "Count the number of words in the following text:",
                "data": "The quick brown fox jumps over the lazy dog."
            },
            {
                "instruction": "Identify the main topic of this passage:",
                "data": "Machine learning algorithms have revolutionized data analysis."
            },
            {
                "instruction": "Find all proper nouns in this sentence:",
                "data": "John visited Paris last summer and met Marie at the Louvre."
            },
            {
                "instruction": "Determine if this statement is true or false:",
                "data": "The Earth revolves around the Sun in approximately 365 days."
            },
            {
                "instruction": "Extract dates from the following text:",
                "data": "The conference will be held on March 15, 2024, and will continue until March 18, 2024."
            }
        ]
        
        for example in simple_examples:
            example["template_type"] = "simple"
            examples.append(example)
            
        return examples
    
    def tokenize_and_label(self, template):
        """
        Tokenize examples and create role labels for each token.
        
        Args:
            template (dict): Prompt template for formatting
            
        Returns:
            list: List of (input_ids, role_labels) tuples
        """
        tokenized_examples = []
        
        for example in self.examples:
            # Format instruction and data using template
            instruction_text = format_prompt(example["instruction"], template, "system")
            data_text = format_prompt(example["data"], template, "user")
            
            # Tokenize separately to track boundaries
            inst_tokens = self.tokenizer(instruction_text, add_special_tokens=False)["input_ids"]
            data_tokens = self.tokenizer(data_text, add_special_tokens=False)["input_ids"]
            
            # Combine tokens
            all_tokens = [self.tokenizer.bos_token_id] + inst_tokens + data_tokens + [self.tokenizer.eos_token_id]
            
            # Create role labels (0: instruction, 1: data)
            role_labels = [0]  # BOS token as instruction
            role_labels.extend([0] * len(inst_tokens))  # Instruction tokens
            role_labels.extend([1] * len(data_tokens))  # Data tokens  
            role_labels.append(1)  # EOS token as data
            
            # Truncate if too long
            if len(all_tokens) > self.max_length:
                all_tokens = all_tokens[:self.max_length]
                role_labels = role_labels[:self.max_length]
            
            # Pad if too short
            while len(all_tokens) < self.max_length:
                all_tokens.append(self.tokenizer.pad_token_id)
                role_labels.append(-100)  # Ignore padding in loss
            
            tokenized_examples.append({
                "input_ids": torch.tensor(all_tokens),
                "role_labels": torch.tensor(role_labels),
                "template_type": example["template_type"],
                "attention_mask": torch.tensor([1 if token != self.tokenizer.pad_token_id else 0 for token in all_tokens])
            })
            
        return tokenized_examples


class HiddenStateExtractor:
    """
    Extracts hidden states from all layers of transformer models.
    """
    
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
        self.device = next(model.parameters()).device
        
    def extract_hidden_states(self, input_ids, attention_mask):
        """
        Extract hidden states from all transformer layers.
        
        Args:
            input_ids (torch.Tensor): Input token IDs
            attention_mask (torch.Tensor): Attention mask
            
        Returns:
            dict: Hidden states for each layer
        """
        self.model.eval()
        hidden_states = {}
        
        with torch.no_grad():
            # Move inputs to device
            input_ids = input_ids.to(self.device)
            attention_mask = attention_mask.to(self.device)
            
            # Forward pass with output_hidden_states=True
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
                return_dict=True
            )
            
            # Extract hidden states from each layer
            all_hidden_states = outputs.hidden_states
            
            for layer_idx, layer_hidden_states in enumerate(all_hidden_states):
                hidden_states[f"layer_{layer_idx}"] = layer_hidden_states.cpu()
                
        return hidden_states


class LayerWiseProbe:
    """
    Linear probe for classifying token roles from hidden states.
    """
    
    def __init__(self, hidden_dim, num_classes=2, random_state=42):
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        self.random_state = random_state
        self.classifier = LogisticRegression(
            random_state=random_state,
            max_iter=1000,
            class_weight='balanced'
        )
        
    def train(self, hidden_states, labels):
        """
        Train the linear probe on hidden states.
        
        Args:
            hidden_states (np.ndarray): Hidden states [num_tokens, hidden_dim]
            labels (np.ndarray): Role labels [num_tokens]
        """
        # Filter out padding tokens (label = -100)
        valid_mask = labels != -100
        valid_hidden_states = hidden_states[valid_mask]
        valid_labels = labels[valid_mask]
        
        if len(valid_labels) == 0:
            return
            
        self.classifier.fit(valid_hidden_states, valid_labels)
        
    def evaluate(self, hidden_states, labels):
        """
        Evaluate the probe on test data.
        
        Args:
            hidden_states (np.ndarray): Hidden states [num_tokens, hidden_dim]
            labels (np.ndarray): Role labels [num_tokens]
            
        Returns:
            dict: Evaluation metrics
        """
        # Filter out padding tokens
        valid_mask = labels != -100
        valid_hidden_states = hidden_states[valid_mask]
        valid_labels = labels[valid_mask]
        
        if len(valid_labels) == 0:
            return {"accuracy": 0.0, "num_tokens": 0}
            
        predictions = self.classifier.predict(valid_hidden_states)
        accuracy = accuracy_score(valid_labels, predictions)
        
        return {
            "accuracy": accuracy,
            "num_tokens": len(valid_labels),
            "predictions": predictions,
            "true_labels": valid_labels
        }


def run_layer_wise_probing_experiment(model_path, embedding_type, base_model, output_dir, batch_size=8):
    """
    Run complete layer-wise probing experiment.
    
    Args:
        model_path (str): Path to model checkpoint
        embedding_type (str): Type of embedding (rgtnet, forward_rot, etc.)
        base_model (str): Base model path
        output_dir (str): Directory to save results
        batch_size (int): Batch size for processing
        
    Returns:
        dict: Experiment results
    """
    print(f"Starting layer-wise probing experiment for {embedding_type} model")
    print(f"Model path: {model_path}")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize model handler
    print("Loading model...")
    handler = CustomModelHandler(
        model_path, base_model, base_model, model_path, None,
        0, embedding_type=embedding_type,
        load_from_checkpoint=True,
    )
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    handler.model.to(device)
    
    # Load tokenizer
    tokenizer = handler.tokenizer
    
    # Load prompt template
    with open("../data/prompt_templates.json", "r") as f:
        templates = json.load(f)
    template = templates[0]
    
    # Create dataset
    print("Creating probing dataset...")
    dataset = RoleProbingDataset(tokenizer)
    tokenized_examples = dataset.tokenize_and_label(template)
    
    # Split into train/test
    train_examples, test_examples = train_test_split(
        tokenized_examples, test_size=0.3, random_state=42
    )
    
    print(f"Training examples: {len(train_examples)}")
    print(f"Test examples: {len(test_examples)}")
    
    # Extract hidden states
    print("Extracting hidden states...")
    extractor = HiddenStateExtractor(handler.model, tokenizer)
    
    # Process training data
    train_hidden_states = {}
    train_labels = []
    
    for example in tqdm(train_examples, desc="Processing training data"):
        hidden_states = extractor.extract_hidden_states(
            example["input_ids"].unsqueeze(0),
            example["attention_mask"].unsqueeze(0)
        )
        
        # Store hidden states for each layer
        for layer_name, layer_states in hidden_states.items():
            if layer_name not in train_hidden_states:
                train_hidden_states[layer_name] = []
            train_hidden_states[layer_name].append(layer_states.squeeze(0))  # Remove batch dimension
        
        train_labels.append(example["role_labels"])
    
    # Process test data
    test_hidden_states = {}
    test_labels = []
    
    for example in tqdm(test_examples, desc="Processing test data"):
        hidden_states = extractor.extract_hidden_states(
            example["input_ids"].unsqueeze(0),
            example["attention_mask"].unsqueeze(0)
        )
        
        for layer_name, layer_states in hidden_states.items():
            if layer_name not in test_hidden_states:
                test_hidden_states[layer_name] = []
            test_hidden_states[layer_name].append(layer_states.squeeze(0))
        
        test_labels.append(example["role_labels"])
    
    # Concatenate all examples
    print("Concatenating hidden states...")
    for layer_name in train_hidden_states:
        train_hidden_states[layer_name] = torch.cat(train_hidden_states[layer_name], dim=0)
        test_hidden_states[layer_name] = torch.cat(test_hidden_states[layer_name], dim=0)
    
    train_labels = torch.cat(train_labels, dim=0)
    test_labels = torch.cat(test_labels, dim=0)
    
    # Train and evaluate probes for each layer
    print("Training layer-wise probes...")
    results = {}
    layer_names = sorted(train_hidden_states.keys(), key=lambda x: int(x.split('_')[1]))
    
    for layer_name in tqdm(layer_names, desc="Training probes"):
        layer_idx = int(layer_name.split('_')[1])
        hidden_dim = train_hidden_states[layer_name].shape[-1]
        
        # Initialize probe
        probe = LayerWiseProbe(hidden_dim)
        
        # Train probe
        train_states_np = train_hidden_states[layer_name].numpy()
        train_labels_np = train_labels.numpy()
        
        # Flatten to [total_tokens, hidden_dim]
        train_states_flat = train_states_np.reshape(-1, hidden_dim)
        train_labels_flat = train_labels_np.reshape(-1)
        
        probe.train(train_states_flat, train_labels_flat)
        
        # Evaluate probe
        test_states_np = test_hidden_states[layer_name].numpy()
        test_labels_np = test_labels.numpy()
        
        test_states_flat = test_states_np.reshape(-1, hidden_dim)
        test_labels_flat = test_labels_np.reshape(-1)
        
        eval_results = probe.evaluate(test_states_flat, test_labels_flat)
        
        results[layer_idx] = {
            "accuracy": eval_results["accuracy"],
            "num_tokens": eval_results["num_tokens"],
            "layer_name": layer_name
        }
        
        print(f"Layer {layer_idx}: Accuracy = {eval_results['accuracy']:.4f} ({eval_results['num_tokens']} tokens)")
    
    # Save results
    results_file = os.path.join(output_dir, f"{embedding_type}_probing_results.json")
    with open(results_file, "w") as f:
        json.dump(results, f, indent=4)
    
    # Create visualization
    create_probing_visualization(results, embedding_type, output_dir)
    
    print(f"Results saved to {results_file}")
    return results


def create_probing_visualization(results, embedding_type, output_dir):
    """
    Create visualization of layer-wise probing results.
    
    Args:
        results (dict): Probing results by layer
        embedding_type (str): Model embedding type
        output_dir (str): Output directory for plots
    """
    # Extract data for plotting
    layers = sorted(results.keys())
    accuracies = [results[layer]["accuracy"] for layer in layers]
    
    # Create plot
    plt.figure(figsize=(12, 8))
    plt.plot(layers, accuracies, marker='o', linewidth=2, markersize=8, label=embedding_type)
    plt.xlabel("Layer Index", fontsize=14)
    plt.ylabel("Role Classification Accuracy", fontsize=14)
    plt.title(f"Layer-wise Role Classification Accuracy - {embedding_type}", fontsize=16)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=12)
    plt.ylim(0.0, 1.0)
    
    # Add horizontal line at 50% (random chance)
    plt.axhline(y=0.5, color='red', linestyle='--', alpha=0.7, label='Random Chance')
    plt.legend()
    
    # Save plot
    plot_file = os.path.join(output_dir, f"{embedding_type}_layer_wise_accuracy.png")
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Visualization saved to {plot_file}")


def compare_models(result_files, output_dir):
    """
    Compare layer-wise probing results across multiple models.
    
    Args:
        result_files (list): List of result file paths
        output_dir (str): Output directory for comparison plots
    """
    plt.figure(figsize=(14, 10))
    
    for result_file in result_files:
        with open(result_file, "r") as f:
            results = json.load(f)
        
        # Extract model name from filename
        model_name = os.path.basename(result_file).replace("_probing_results.json", "")
        
        layers = sorted([int(k) for k in results.keys()])
        accuracies = [results[str(layer)]["accuracy"] for layer in layers]
        
        plt.plot(layers, accuracies, marker='o', linewidth=2, markersize=8, label=model_name)
    
    plt.xlabel("Layer Index", fontsize=14)
    plt.ylabel("Role Classification Accuracy", fontsize=14)
    plt.title("Layer-wise Role Classification Accuracy Comparison", fontsize=16)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=12)
    plt.ylim(0.0, 1.0)
    plt.axhline(y=0.5, color='red', linestyle='--', alpha=0.7, label='Random Chance')
    
    # Save comparison plot
    comparison_file = os.path.join(output_dir, "model_comparison.png")
    plt.savefig(comparison_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Model comparison saved to {comparison_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Layer-wise probing experiment for role classification")
    
    parser.add_argument("--model_path", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--embedding_type", type=str, required=True, 
                       choices=["rgtnet", "rgtnet_orthonly", "forward_rot", "ise", "single_emb"],
                       help="Type of embedding used in the model")
    parser.add_argument("--base_model", type=str, default=None, help="Base model path")
    parser.add_argument("--output_dir", type=str, default="./probing_results", 
                       help="Directory to save results")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size for processing")
    parser.add_argument("--compare", nargs="+", default=None,
                       help="List of result files to compare")
    
    args = parser.parse_args()
    
    if args.compare:
        # Compare existing results
        compare_models(args.compare, args.output_dir)
    else:
        # Run new experiment
        results = run_layer_wise_probing_experiment(
            args.model_path,
            args.embedding_type, 
            args.base_model,
            args.output_dir,
            args.batch_size
        )
        
        print("Experiment completed successfully!")
        print(f"Results saved to {args.output_dir}")
