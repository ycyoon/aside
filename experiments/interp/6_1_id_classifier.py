import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import argparse
import os
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from tqdm import tqdm
import json

class LinearClassifier(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.linear = nn.Linear(input_dim, 1)
        
    def forward(self, x):
        return torch.sigmoid(self.linear(x))

def load_and_prepare_data(
        data_dir, layer, subsample=1.0, eval_size=0.3, seed=42, verbose=False,
        pre_loaded_inst_hidden=None, pre_loaded_data_hidden=None
):
    if pre_loaded_inst_hidden is None:
        inst_hidden_full = torch.load(os.path.join(data_dir, "inst_hidden_states.pt")).float()
    else:
        inst_hidden_full = pre_loaded_inst_hidden.float()
    
    if pre_loaded_data_hidden is None:
        data_hidden_full = torch.load(os.path.join(data_dir, "data_hidden_states.pt")).float()
    else:
        data_hidden_full = pre_loaded_data_hidden.float()
    
    # Get the minimum length for balanced dataset
    min_len = min(len(inst_hidden_full), len(data_hidden_full))
    balanced_size = int(min_len * subsample)
    
    if verbose:
        print(f"Original sizes - Instruction: {len(inst_hidden_full)}, Data: {len(data_hidden_full)}")
        print(f"Balanced size with subsample {subsample}: {balanced_size}")
    
    # Set random seed
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Select layer
    inst_hidden = inst_hidden_full[:, layer, :].detach()
    data_hidden = data_hidden_full[:, layer, :].detach()

    
    # Randomly sample balanced dataset
    inst_indices = torch.randperm(len(inst_hidden))[:balanced_size]
    data_indices = torch.randperm(len(data_hidden))[:balanced_size]
    
    inst_hidden = inst_hidden[inst_indices]
    data_hidden = data_hidden[data_indices]
    
    # Create labels
    inst_labels = torch.ones(balanced_size)
    data_labels = torch.zeros(balanced_size)
    
    # Combine data
    features = torch.cat([inst_hidden, data_hidden])
    labels = torch.cat([inst_labels, data_labels])
    
    # Shuffle the combined dataset with seed
    shuffle_indices = torch.randperm(len(features))
    features = features[shuffle_indices]
    labels = labels[shuffle_indices]
    
    # Split into train and eval
    X_train, X_eval, y_train, y_eval = train_test_split(
        features, labels, test_size=eval_size, random_state=seed
    )

    if verbose:
        print(f"Train size: {len(X_train)}, Eval size: {len(X_eval)}")
    
    return X_train, X_eval, y_train, y_eval

def train_classifier(X_train, X_eval, y_train, y_eval, hidden_dim=4096, 
                    lr=1e-3, batch_size=32, epochs=5, device="cuda", verbose=False):
    # Create datasets and dataloaders
    train_dataset = TensorDataset(X_train, y_train.unsqueeze(1))
    eval_dataset = TensorDataset(X_eval, y_eval.unsqueeze(1))
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    eval_loader = DataLoader(eval_dataset, batch_size=batch_size)
    
    # Initialize model and optimizer
    model = LinearClassifier(hidden_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.BCELoss()
    
    # Training loop
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        
        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            
            optimizer.zero_grad()
            output = model(batch_X)
            loss = criterion(output, batch_y)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        if verbose:
            print(f"Epoch {epoch+1}/{epochs}, Loss: {train_loss/len(train_loader):.4f}")
    
    # Evaluation
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for batch_X, batch_y in eval_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            outputs = model(batch_X)
            predicted = (outputs > 0.5).float()
            total += batch_y.size(0)
            correct += (predicted == batch_y).sum().item()
    
    accuracy = correct / total
    if verbose:
        print(f"Evaluation Accuracy: {accuracy:.4f}")
    
    return accuracy

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("data_dir", help="Directory containing hidden states .pt files")
    parser.add_argument("--layer", help="Layer to use (number or 'all')", required=True)
    parser.add_argument("--subsample", type=float, default=1.0, help="Fraction of data to use")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--verbose", action="store_true", help="Print detailed information")
    parser.add_argument("--plot-only", action="store_true", help="Only plot accuracies, loaded from the data dir")
    args = parser.parse_args()
    
    device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
    if args.verbose:
        print(f"Using device: {device}")
    
    if args.layer.lower() == "all":
        if not args.plot_only:
            # Compute the accuracies if needed
            # Load full tensors once
            inst_hidden = torch.load(os.path.join(args.data_dir, "inst_hidden_states.pt"))
            data_hidden = torch.load(os.path.join(args.data_dir, "data_hidden_states.pt"))
            num_layers = inst_hidden.shape[1]
            hidden_size = inst_hidden.shape[2]
            accuracies = []
            
            for layer in tqdm(range(num_layers)):
                if args.verbose:
                    print(f"\nTraining classifier for layer {layer}")

                X_train, X_eval, y_train, y_eval = load_and_prepare_data(
                    args.data_dir, layer, args.subsample, seed=args.seed, verbose=args.verbose,
                    pre_loaded_inst_hidden=inst_hidden, pre_loaded_data_hidden=data_hidden
                )

                hidden_dim = X_train.shape
                
                accuracy = train_classifier(
                    X_train, X_eval, y_train, y_eval, 
                    device=device, verbose=args.verbose, hidden_dim=hidden_size,
                )
                accuracies.append(accuracy)
            
            # Save the accuracies to a json file
            with open(os.path.join(args.data_dir, 'layer_accuracies.json'), 'w') as f:
                json.dump(accuracies, f)
        else: 
            # Load the accuracies from the json file
            with open(os.path.join(args.data_dir, 'layer_accuracies.json'), 'r') as f:
                accuracies = json.load(f)

        # Plot accuracies
        plt.figure(figsize=(10, 6))
        plt.plot(range(num_layers), accuracies, marker='o')
        plt.xlabel('Layer')
        plt.ylabel('Evaluation Accuracy')
        plt.ylim(0, 1)
        plt.title(f'Linear Classification Accuracy by Layer, data from {args.data_dir}')
        plt.grid(True)
        plt.savefig(os.path.join(args.data_dir, 'layer_accuracies.png'))
        plt.close()
        
    else:
        layer = int(args.layer)
        X_train, X_eval, y_train, y_eval = load_and_prepare_data(
            args.data_dir, layer, args.subsample, seed=args.seed, verbose=args.verbose
        )
        
        train_classifier(
            X_train, X_eval, y_train, y_eval, 
            device=device, verbose=args.verbose, hidden_dim=hidden_size,
        )

if __name__ == "__main__":
    main()