import os
import sys
if ".." not in sys.path:
    sys.path.append("..")

# os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import argparse
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
from transformers import AutoModel, AutoTokenizer, AutoConfig, AutoModelForCausalLM
from datasets import load_dataset
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Literal, Optional, Dict, List, Tuple
from dataclasses import dataclass
import logging
from tqdm import tqdm
import json


from model import CustomLlamaConfig, CustomLLaMA
from model_api import CustomModelHandler, format_prompt

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class ProbeConfig:
    # These four are set in the main function
    model_name: str 
    model_path: str
    base_model: str 

    template_type: Literal["base", "single", "double", "ise"]
    embedding_type: Literal["single_emb", "double_emb", "ise", "forward_rot"]

    probe_type: Literal["probe", "actdiff"] = "probe"
    data_path: str = "../data/train_data/alpaca_data_cleaned_gpt4.json"

    # # BASE
    # model_path = None
    # model_name: str = "meta-llama/Llama-3.1-8B"  

    # SFT (single)
    # model_path = "/nfs/scistore23/chlgrp/ezverev/projects/side/models/llama_3.1_8b/pretrained_vanilla/train_checkpoints/SFTv110/from_base_run_11_fix/last" 
    # model_name = "Embeddings-Collab/llama_3.1_8b_single_emb_emb_SFTv110_from_base_run_11_fix"

    # ISE 
    # model_path =  "/nfs/scistore23/chlgrp/ezverev/projects/side/models/llama_3.1_8b/ise/train_checkpoints/SFTv110/from_base_run_2_fix/last"  
    # model_name = "Embeddings-Collab/llama_3.1_8b_ise_emb_SFTv110_from_base_run_2_fix"

    # ASIDE 
    # model_path =  "/nfs/scistore23/chlgrp/ezverev/projects/side/models/llama_3.1_8b/forward_rot/train_checkpoints/SFTv110/from_base_run_15_fix/last" 
    # model_name = "Embeddings-Collab/llama_3.1_8b_forward_rot_emb_SFTv110_from_base_run_15_fix" 



    inference_batch_size: int = 16  # Small batch size for model inference
    probe_batch_size: int = 64    # Larger batch size for probe training
    num_epochs: int = 5
    learning_rate: float = 1e-3
    max_length: int = 512
    dataset_downsample: float = 0.1
    train_size: float = 0.7
    val_size: float = 0.15
    test_size: float = 0.15
    random_seed: int = 42
    eval_on_test: bool = True
    torch_dtype: torch.dtype = torch.float16
    pad_token_id: Optional[str] = "eos"
    save_dir: str = "./probe_middle_template"
    cache_dir: str = "./feature_cache"  # New: directory for cached features

    token_pos: Literal["first", "middle", "last", "mean"] = "middle"

    verbose: bool = True
    # layers: Optional[List[int]] = (5,7,10,15,20)
    # layers: Optional[List[int]] = (15,)
    layers: Optional[List[int]] = "all"
    template: bool = True



class PrecomputedFeatureDataset(Dataset):
    def __init__(self, features: torch.Tensor, labels: torch.Tensor):
        self.features = features
        self.labels = labels
        
    def __len__(self):
        return len(self.labels)
        
    def __getitem__(self, idx):
        return {
            "features": self.features[idx],
            "label": self.labels[idx]
        }


class InstructionDataset(Dataset):
    def __init__(self, data_path, tokenizer, max_length: int = 512):
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        # Load the dataset
        with open(data_path, "r") as file:
            dataset = json.load(file)
        
        # Create positive examples (instructions)
        self.instructions = [row["instruction"] for row in dataset]
        
        # Create negative examples (inputs, filtering out empty ones)
        self.inputs = [row["input"] for row in dataset if row["input"].strip()]
        
        # Balance the dataset
        min_len = min(len(self.instructions), len(self.inputs))
        self.instructions = self.instructions[:min_len]
        self.inputs = self.inputs[:min_len]
        
        # Combine with labels
        self.texts = self.instructions + self.inputs
        self.labels = [1] * len(self.instructions) + [0] * len(self.inputs)

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]

        return {
            "text": text,
            "label": label
        }
        
    
    @staticmethod
    def analyze_lengths(tokenizer, dataset_path: str = "yahma/alpaca-cleaned") -> dict:
        """Get basic length statistics for the dataset."""
        dataset = load_dataset(dataset_path)["train"]
        lengths = []
        
        # Get lengths for both instructions and inputs
        for row in dataset:
            lengths.append(len(tokenizer.encode(row["instruction"])))
            if row["input"].strip():  # only non-empty inputs
                lengths.append(len(tokenizer.encode(row["input"])))
        
        lengths = np.array(lengths)
        return {
            "max": int(np.max(lengths)),
            "mean": int(np.mean(lengths)),
            "p95": int(np.percentile(lengths, 95)),
            "p99": int(np.percentile(lengths, 99))
        }

class LinearProbe(nn.Module):
    def __init__(self, hidden_size: int):
        super().__init__()
        self.linear = nn.Linear(hidden_size, 1, bias=False)  # No bias term
    
    def forward(self, x):
        return torch.sigmoid(self.linear(x))

    def get_concept_direction(self):
        # Normalize the weights to get a unit vector
        w = self.linear.weight.squeeze()
        return w / torch.norm(w)
    
    def concept_activation(self, x):
        # Get cosine similarity with the concept direction
        direction = self.get_concept_direction()
        return F.cosine_similarity(x, direction.unsqueeze(0), dim=1)


class CustomHandlerFeatureExtractor:
    def __init__(
        self,
        handler: CustomModelHandler,
        token_pos: Literal["first", "middle", "last"],
        layer: int,
        use_template: bool,
        template_type: Literal["base", "single", "double", "ise"],
    ):
        self.handler = handler
        self.token_pos = token_pos
        self.layer = layer

        self.system_prompt_len, self.template_infix_len, self.template_suffix_len = handler.get_template_parameters(template_type)

        with open("../data/prompt_templates.json", "r") as f:
            templates = json.load(f)
        self.template = templates[0]

        self.use_template = use_template
        
    def extract_features_from_texts(self, texts, labels):
        all_features = []
        for text, label in zip(texts, labels):

            if self.use_template:
                if label == 1:
                    # Instruction
                    inst_text = text
                    data_text = ""
                else:
                    inst_text = ""
                    data_text = text
                    
                instruction_prompt = format_prompt(inst_text, self.template, "system")
                data_prompt = format_prompt(data_text, self.template, "user")


                output, inst_tokens, data_tokens, probe_tokens, inst_hidden, data_hidden, probe_hidden, last_hidden, inp = self.handler.generate_one_token_with_hidden_states(
                    instruction_prompt, data_prompt,
                    system_prompt_len=self.system_prompt_len,
                    template_infix_len=self.template_infix_len,
                    template_suffix_len=self.template_suffix_len,
                )

                if label == 1:
                    hidden = inst_hidden
                else:
                    hidden = data_hidden
            else:

                if label == 1:
                    # Instruction
                    output, tokens, hidden, inp = self.handler.generate_with_hidden_states_instruction_only(
                        text, max_new_tokens=1
                    )
                else:
                    # Data
                    output, tokens, hidden, inp = self.handler.generate_with_hidden_states_data_only(
                        text, max_new_tokens=1
                    )

            layer_hidden = hidden[:, self.layer, :] # (seq_len, hidden_dim)

            if self.token_pos == "first":
                token_pos = 0
            elif self.token_pos == "last":
                token_pos = -1
            elif self.token_pos == "middle":
                seq_length = layer_hidden.size(0)
                token_pos = seq_length // 2
            elif self.token_pos == "mean":
                token_pos = "mean"
            else:
                raise ValueError(f"Invalid token position: {self.token_pos}")

            if token_pos == "mean":
                features = layer_hidden.mean(dim=0).to(torch.float32)
            else:
                features = layer_hidden[token_pos].to(torch.float32)

            all_features.append(features)
        
        return all_features
            


class ProbeTrainer:
    def __init__(
        self,
        config: ProbeConfig,
        device: torch.device
    ):
        self.config = config
        self.device = device
        self.results = {}
        
    def train_epoch(
        self,
        probe: LinearProbe,
        dataloader: DataLoader,
        optimizer: torch.optim.Optimizer,
        verbose: bool = False
    ) -> float:
        probe.train()
        total_loss = 0
        
        for batch in tqdm(dataloader, disable=not verbose):
            features = batch["features"].to(self.device)
            labels = batch["label"].to(self.device)
            
            predictions = probe(features).squeeze()
            loss = F.binary_cross_entropy(predictions, labels)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
        avg_loss = total_loss / len(dataloader)
        if verbose:
            logger.info(f"Training loss: {avg_loss:.4f}")
        return avg_loss
    
    def evaluate(
        self,
        probe: LinearProbe,
        dataloader: DataLoader
    ) -> Tuple[float, float]:
        probe.eval()
        total_correct = 0
        total_samples = 0
        total_loss = 0
        
        with torch.no_grad():
            for batch in dataloader:
                features = batch["features"].to(self.device)
                labels = batch["label"].to(self.device)
                
                predictions = probe(features).squeeze()
                loss = F.binary_cross_entropy(predictions, labels)
                total_loss += loss.item()
                
                predictions = (predictions >= 0.5).float()
                total_correct += (predictions == labels).sum().item()
                total_samples += labels.size(0)
        
        accuracy = total_correct / total_samples
        avg_loss = total_loss / len(dataloader)
        return accuracy, avg_loss

def plot_results(results: List[Dict]):
    df = pd.DataFrame(results)
    plt.figure(figsize=(10, 6))
    
    for token_pos in df["token_pos"].unique():
        token_data = df[df["token_pos"] == token_pos]
        plt.plot(
            token_data["layer"],
            token_data["val_accuracy"],
            marker="o",
            label=f"Token Position: {token_pos}"
        )
        plt.savefig(f"probing_results_{token_pos}.png")
    
    plt.xlabel("Layer")
    plt.ylabel("Validation Accuracy")
    plt.title("Linear Probe Performance Across Layers")
    plt.legend()
    plt.grid(True)
    return plt


def compute_or_load_features(
    dataset: Dataset,
    feature_extractor: CustomHandlerFeatureExtractor,
    cache_path: str,
    device: torch.device,
    inference_batch_size: int = 2
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute features or load from cache if available.
    Returns (features, labels) tensors.
    """
    if os.path.exists(cache_path):
        print(f"Loading cached features from {cache_path}")
        # Load cached features
        cached = torch.load(cache_path)
        return cached['features'].to(device).to(torch.float32),  cached['labels'].to(device)
    
    print(f"Computing features and saving to {cache_path}")
        
    # Create dataloader with small batch size for inference
    dataloader = DataLoader(dataset, batch_size=inference_batch_size)
    
    features_list = []
    labels_list = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Computing features"):
            # input_ids = batch["input_ids"].to(device)
            # attention_mask = batch["attention_mask"].to(device)
            texts = batch["text"]
            labels = batch["label"]

            batch_features = feature_extractor.extract_features_from_texts(texts, labels)

            features_list.extend(batch_features)
            labels_list.extend(labels)
            
            # features = feature_extractor.extract_features(input_ids, attention_mask)
            # features_list.append(features.cpu())
            # labels_list.append(batch["label"].cpu())
    
    # get a tensor of shape (num_examples, hidden_dim)
    features = torch.stack(features_list, dim=0)
    labels = torch.tensor(labels_list, dtype=torch.float32).to(device)
    
    # Cache the results
    torch.save({'features': features, 'labels': labels}, cache_path)
    
    return features, labels



def setup_model_and_tokenizer(config: ProbeConfig) -> Tuple[AutoModel, AutoTokenizer]:
    """Initialize and setup the model and tokenizer."""
    model = AutoModel.from_pretrained(config.model_name, torch_dtype=config.torch_dtype)
    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    
    if config.pad_token_id == "eos":
        tokenizer.pad_token = tokenizer.eos_token
        
    return model, tokenizer


def setup_custom_model_handler(config: ProbeConfig) -> CustomModelHandler:

    AutoConfig.register("custom_llama", CustomLlamaConfig)
    AutoModelForCausalLM.register(CustomLlamaConfig, CustomLLaMA)
    model_to_load = config.model_path if config.model_path else config.model_name
    handler = CustomModelHandler(model_to_load, config.base_model, config.base_model, model_to_load, None,
                            0, embedding_type=config.embedding_type,
                            load_from_checkpoint=True,
                            model_dtype=torch.bfloat16
                            )
    
    return handler
    

def prepare_dataset(config: ProbeConfig, tokenizer: AutoTokenizer) -> InstructionDataset:
    """Create and prepare the dataset, including downsampling if needed."""
    dataset = InstructionDataset(config.data_path, tokenizer, config.max_length)
    
    if config.dataset_downsample < 1.0:
        subset_indices = torch.randint(
            0, len(dataset),
            (int(len(dataset) * config.dataset_downsample),),
            generator=torch.Generator().manual_seed(config.random_seed)
        )
        dataset = torch.utils.data.Subset(dataset, subset_indices)
    
    return dataset


def create_datasets(
    features: torch.Tensor,
    labels: torch.Tensor,
    config: ProbeConfig
) -> Tuple[PrecomputedFeatureDataset, PrecomputedFeatureDataset, PrecomputedFeatureDataset]:
    # Create splits
    dataset_size = len(features)
    indices = torch.randperm(dataset_size, generator=torch.Generator().manual_seed(config.random_seed))
    
    train_size = int(dataset_size * config.train_size)
    val_size = int(dataset_size * config.val_size)
    
    train_indices = indices[:train_size]
    val_indices = indices[train_size:train_size + val_size]
    test_indices = indices[train_size + val_size:]
    
    # Create datasets
    train_dataset = PrecomputedFeatureDataset(features[train_indices], labels[train_indices])
    val_dataset = PrecomputedFeatureDataset(features[val_indices], labels[val_indices])
    test_dataset = PrecomputedFeatureDataset(features[test_indices], labels[test_indices])
    return train_dataset, val_dataset, test_dataset


def create_dataloaders(
    features: torch.Tensor,
    labels: torch.Tensor,
    config: ProbeConfig
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Create train, validation, and test dataloaders from features."""
    train_dataset, val_dataset, test_dataset = create_datasets(features, labels, config)
    
    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=config.probe_batch_size, shuffle=True, generator=torch.Generator().manual_seed(config.random_seed))
    val_loader = DataLoader(val_dataset, batch_size=config.probe_batch_size)
    test_loader = DataLoader(test_dataset, batch_size=config.probe_batch_size)
    
    return train_loader, val_loader, test_loader


def train_layer_probe(
    handler: CustomModelHandler,
    dataset: InstructionDataset,
    layer: int,
    config: ProbeConfig,
    device: torch.device
) -> Dict:
    """Train and evaluate a probe for a specific layer."""
    # Setup cache path
    model_shortname = config.model_name.split('/')[-1]
    data_shortname = config.data_path.split('/')[-1].split('.')[0]
    cache_path = os.path.join(
        config.cache_dir,
        f"features_{model_shortname}_{data_shortname}_{config.dataset_downsample}_{layer}_{config.token_pos}_{config.template}.pt"
    )

    # Extract features
    feature_extractor = CustomHandlerFeatureExtractor(handler, token_pos=config.token_pos, layer=layer, use_template=config.template, template_type=config.template_type)
    features, labels = compute_or_load_features(
        dataset=dataset,
        feature_extractor=feature_extractor,
        cache_path=cache_path,
        device=device,
        inference_batch_size=config.inference_batch_size
    )
    
    # Create dataloaders
    train_loader, val_loader, test_loader = create_dataloaders(features, labels, config)
    
    # Initialize trainer and probe
    trainer = ProbeTrainer(config, device)
    probe = LinearProbe(handler.model.config.hidden_size).to(device)
    optimizer = torch.optim.Adam(probe.parameters(), lr=config.learning_rate)
    
    # Train
    best_val_acc = -1
    best_val_loss = float('inf')
    best_probe = None
    
    for epoch in range(config.num_epochs):
        train_loss = trainer.train_epoch(probe, train_loader, optimizer, verbose=config.verbose)
        val_acc, val_loss = trainer.evaluate(probe, val_loader)
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_val_loss = val_loss
            best_probe = probe.state_dict()
    
    if config.verbose:
        logger.info(f"Best validation accuracy: {best_val_acc:.4f}")
    
    # Test evaluation
    test_acc, test_loss = None, None
    if config.eval_on_test:
        probe.load_state_dict(best_probe)
        test_acc, test_loss = trainer.evaluate(probe, test_loader)
        if config.verbose:
            logger.info(f"Test accuracy: {test_acc:.4f}")
    
    # Save the best probe if directory is provided
        os.makedirs(config.save_dir, exist_ok=True)
        filename = f"probe_{model_shortname}_{data_shortname}_layer{layer}_{config.token_pos}.pt"
        save_path = os.path.join(config.save_dir, filename)
        torch.save(best_probe, save_path)
    
    return {
        'layer': layer,
        'token_pos': config.token_pos,
        'val_accuracy': best_val_acc,
        'val_loss': best_val_loss,
        'test_accuracy': test_acc,
        'test_loss': test_loss
    }


def batch_cosine_similarity_with_direction(
    vectors: torch.Tensor,
    direction: torch.Tensor
) -> torch.Tensor:
    """Compute cosine similarity between vectors and a direction."""
    return F.cosine_similarity(vectors, direction.unsqueeze(0), dim=1)


def batch_predict_concept_activation(
    vectors: torch.Tensor,
    direction: torch.Tensor,
    threshold: float = 0
) -> torch.Tensor:
    """Predict the concept activation based on cosine similarity."""
    similarities = batch_cosine_similarity_with_direction(vectors, direction)
    return (similarities >= threshold).float()


def train_layer_actdiff(
    handler: CustomModelHandler,
    dataset: InstructionDataset,
    layer: int,
    config: ProbeConfig,
    device: torch.device
) -> Dict:
    """Train and evaluate a probe for a specific layer."""
    # Setup cache path
    model_shortname = config.model_name.split('/')[-1]
    data_shortname = config.data_path.split('/')[-1].split('.')[0]
    cache_path = os.path.join(
        config.cache_dir,
        f"features_{model_shortname}_{data_shortname}_{config.dataset_downsample}_{layer}_{config.token_pos}.pt"
    )

    # Extract features
    feature_extractor = CustomHandlerFeatureExtractor(handler, token_pos=config.token_pos, layer=layer, use_template=config.template, template_type=config.template_type)
    features, labels = compute_or_load_features(
        dataset=dataset,
        feature_extractor=feature_extractor,
        cache_path=cache_path,
        device=device,
        inference_batch_size=config.inference_batch_size
    )


    train_dataset, val_dataset, test_dataset = create_datasets(features, labels, config)
    train_features, train_labels = train_dataset.features, train_dataset.labels
    val_features, val_labels = val_dataset.features, val_dataset.labels
    test_features, test_labels = test_dataset.features, test_dataset.labels


    positive_examples = train_features[train_labels == 1]
    negative_examples = train_features[train_labels == 0]

    # Compute activation differences
    actdiff_vector = positive_examples.mean(dim=0) - negative_examples.mean(dim=0)

    # Compute train, val, and test predictions
    train_predictions = batch_predict_concept_activation(train_features, actdiff_vector)
    val_predictions = batch_predict_concept_activation(val_features, actdiff_vector)

    # Compute accuracies
    train_accuracy = (train_predictions == train_labels).float().mean().item()
    val_accuracy = (val_predictions == val_labels).float().mean().item()

    
    if config.verbose:
        logger.info(f"Layer {layer}, Training accuracy: {train_accuracy:.4f}")
        logger.info(f"Best (and only) validation accuracy: {val_accuracy:.4f}")
    
    # Test evaluation
    test_accuracy = None
    if config.eval_on_test:
        test_predictions = batch_predict_concept_activation(test_features, actdiff_vector)
        test_accuracy = (test_predictions == test_labels).float().mean().item()
        if config.verbose:
            logger.info(f"Test accuracy: {test_accuracy:.4f}")
    
    # Save the best probe if directory is provided
        os.makedirs(config.save_dir, exist_ok=True)
        filename = f"actdiff_{model_shortname}_{data_shortname}_layer{layer}_{config.token_pos}.pt"
        save_path = os.path.join(config.save_dir, filename)
        torch.save(actdiff_vector, save_path)
    
    return {
        'layer': layer,
        'token_pos': config.token_pos,
        'val_accuracy': val_accuracy,
        'test_accuracy': test_accuracy,
    }

def save_results(
    results_df: pd.DataFrame,
    config: ProbeConfig,
    model_shortname: str
) -> None:
    """Save results to CSV and create visualization."""
    # Save CSV
    csv_savepath = f"probing_results_{model_shortname}_{config.token_pos}.csv"
    csv_savepath = os.path.join(config.save_dir, csv_savepath)
    results_df.to_csv(
        csv_savepath,
        index=False
    )
    
    # Create visualization
    plt.figure(figsize=(10, 6))
    plt.plot(results_df['layer'], results_df['val_accuracy'],
             marker='o', label='Validation Accuracy')
    
    if config.eval_on_test:
        plt.plot(results_df['layer'], results_df['test_accuracy'],
                marker='o', label='Test Accuracy')
    
    plt.xlabel('Layer')
    plt.ylabel('Accuracy')
    plt.title(f'Probing Accuracy vs Layer (Token Position: {config.token_pos})')
    plt.legend()
    plt.grid(True)
    png_savepath = f'probing_results_{model_shortname}_{config.token_pos}.png'
    png_savepath = os.path.join(config.save_dir, png_savepath)
    plt.savefig(png_savepath)
    plt.close()


def probe_all_layers(config: ProbeConfig) -> pd.DataFrame:
    """Main function to probe all layers of the model."""
    # Setup device
    device = torch.device("mps" if torch.backends.mps.is_available() else 
                         "cuda" if torch.cuda.is_available() else "cpu")

    print(f"Device is {device}") 
    
    # Initialize model and dataset
    # model, tokenizer = setup_model_and_tokenizer(config)
    # model = model.to(device)

    handler = setup_custom_model_handler(config)
    handler.model.to(device)


    dataset = prepare_dataset(config, handler.tokenizer)
    
    # Create directories
    os.makedirs(config.cache_dir, exist_ok=True)
    
    # Get number of layers
    if config.layers is None or config.layers == "all":
        num_layers = handler.model.config.num_hidden_layers + 1 # +1 for embeddings
        layers = list(range(num_layers))
    else:
        layers = config.layers
    
    # Probe each layer
    all_results = []
    model_shortname = config.model_name.split('/')[-1]
    
    for layer in layers:
        logger.info(f"Processing layer {layer}")
        
        # Train and evaluate probe for this layer
        if config.probe_type == "probe":
            layer_results = train_layer_probe(handler, dataset, layer, config, device)
        elif config.probe_type == "actdiff":
            layer_results = train_layer_actdiff(handler, dataset, layer, config, device)
        else:
            raise ValueError(f"Invalid probe type: {config.probe_type}")

        all_results.append(layer_results)
        
        # Save intermediate results
        results_df = pd.DataFrame(all_results)
        save_results(results_df, config, model_shortname)
    
    return pd.DataFrame(all_results)

def main():
    """Entry point for the probing analysis."""
    parser = argparse.ArgumentParser(description="Extracting the instruction feature")

    parser.add_argument("--model_path", type=str, default=None, help="Path to the model")
    parser.add_argument("--model_name", type=str, default=None, help="Name of the model")
    parser.add_argument("--base_model", type=str, default=None, help="Base model name")

    parser.add_argument("--embedding_type", type=str, default=None, help="Embedding type")
    parser.add_argument("--template_type", type=str, default=None, help="Template type")

    parser.add_argument("--layer", type=int, default=None, help="Layers to probe")

    args = parser.parse_args()

    
    # Initialize configuration
    config = ProbeConfig(
        model_name=args.model_name,
        model_path=args.model_path,
        base_model=args.base_model,
        embedding_type=args.embedding_type,
        template_type=args.template_type,
        layers=[args.layer] if args.layer is not None else "all",
    )


    config.layers = [args.layer] if args.layer is not None else config.layers
    

    
    # Run probing analysis
    results_df = probe_all_layers(config)
    
    print("\nFinal Results:")
    print(results_df)
    return results_df

if __name__ == "__main__":
    results_df = main()