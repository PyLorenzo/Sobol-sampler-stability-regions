"""
Author: Lorenzo Baldazzi
Date: 2026-06-08
Goal: Train a Neural Network to predict EFTCAMB stability.
"""

import argparse
import pickle
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# ----------------------------------------------------------------------
# 1.NEURAL NETWORK ARCHITECTURE
# ----------------------------------------------------------------------
class StabilityOracle(nn.Module):
    def __init__(self, input_dim):
        super(StabilityOracle, self).__init__()
        # Classic Multi-Layer Perceptron (MLP).
        # 3 hidden layers are generally sufficient for complex non-linear 2D boundaries.
        self.network = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1) # Single output (log-odds of stability probability)
        )

    def forward(self, x):
        return self.network(x)

# ----------------------------------------------------------------------
# 2. MAIN TRAINING FUNCTION
# ----------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="Train NN on EFTCAMB stability data.")
    ap.add_argument("--data", default="/home/lbaldazzi/Documents/Dottorato/Scripts/Stability_regions/Sobol_sampling/TPM/tpm_stability_map.pkl",
                    help="Path to the .pkl file generated from Phase 1 and 2")
    ap.add_argument("--epochs", type=int, default=1000, help="Number of training epochs")
    ap.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    args = ap.parse_args()

    print("=" * 64)
    print(" Phase 3: Neural Network Fast Prior Training")
    print("=" * 64)

    # ---- A. Loading Data ----
    print(f"[1/4] Loading data from {args.data}...")
    with open(args.data, "rb") as f:
        data = pickle.load(f)
    
    X = data["points"]
    y = data["stable"].astype(np.float32) # The network needs float, not booleans
    param_names = data["param_names"]
    
    print(f"      Found {X.shape[0]} samples for parameters: {param_names}")
    print(f"      Stable fraction: {y.mean():.3f}")

    # ---- B. Preprocessing ----
    print("[2/4] Preprocessing and Data Split...")
    # Split into Training (80%) and Validation (20%) sets
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    # CRUCIAL: Neural networks work poorly with parameters on different scales.
    # We must standardize data (mean 0, variance 1).
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)

    # Convertiamo in tensori PyTorch
    X_train_t = torch.tensor(X_train_scaled, dtype=torch.float32)
    y_train_t = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)
    X_val_t = torch.tensor(X_val_scaled, dtype=torch.float32)
    y_val_t = torch.tensor(y_val, dtype=torch.float32).view(-1, 1)

    # ---- C. Network and Optimizer Initialization ----
    print("[3/4] Model Initialization...")
    model = StabilityOracle(input_dim=X.shape[1])
    
    # Binary Cross Entropy with Logits included (more numerically stable than standard Sigmoid + BCE)
    criterion = nn.BCEWithLogitsLoss() 
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    # ---- D. Training Loop ----
    print("[4/4] Starting Training...")
    best_val_loss = float('inf')

    for epoch in range(args.epochs):
        model.train()
        optimizer.zero_grad()
        
        # Forward pass
        outputs = model(X_train_t)
        loss = criterion(outputs, y_train_t)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        # Validation every 100 epochs
        if (epoch + 1) % 100 == 0:
            model.eval()
            with torch.no_grad():
                val_outputs = model(X_val_t)
                val_loss = criterion(val_outputs, y_val_t)
                
                # Calculate accuracy
                predictions = (torch.sigmoid(val_outputs) >= 0.5).float()
                accuracy = (predictions == y_val_t).float().mean().item()
            
            print(f"      Epoca [{epoch+1}/{args.epochs}] | Train Loss: {loss.item():.4f} | Val Loss: {val_loss.item():.4f} | Val Acc: {accuracy*100:.2f}%")
            
            # Save the best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(model.state_dict(), "/home/lbaldazzi/Documents/Dottorato/Scripts/Stability_regions/Sobol_sampling/TPM/tpm_stability_model.pt")

    # ---- E. Saving Artifacts ----
    print("\n[Completed] Saving Scaler...")
    # We must also save the scaler, otherwise the MCMC sampler won't know how to transform new points
    with open("/home/lbaldazzi/Documents/Dottorato/Scripts/Stability_regions/Sobol_sampling/TPM/tpm_scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)
        
    print("Generated artifacts:")
    print(" - tpm_stability_model.pt (Neural network weights)")
    print(" - tpm_scaler.pkl (Object for parameter normalization)")

if __name__ == "__main__":
    main()