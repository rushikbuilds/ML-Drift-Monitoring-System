import sys
import pickle
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.datasets import fetch_openml
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

class Elec2MLP(nn.Module):
    def __init__(self, input_dim: int, num_classes: int = 2) -> None:
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 64)
        self.bn1 = nn.BatchNorm1d(64)
        self.fc2 = nn.Linear(64, 32)
        self.bn2 = nn.BatchNorm1d(32)
        self.fc3 = nn.Linear(32, 16) # penultimate layer
        self.bn3 = nn.BatchNorm1d(16)
        self.output = nn.Linear(16, num_classes)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.dropout(self.relu(self.bn1(self.fc1(x))))
        h = self.dropout(self.relu(self.bn2(self.fc2(h))))
        h = self.dropout(self.relu(self.bn3(self.fc3(h))))
        return self.output(h)

def train_elec2(model, X_train, y_train, device="cpu", epochs=20, batch_size=256, lr=1e-3):
    model.to(device)
    X_train_t = torch.tensor(X_train, dtype=torch.float32, device=device)
    y_train_t = torch.tensor(y_train, dtype=torch.long, device=device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    n_samples = len(X_train_t)
    model.train()
    for epoch in range(1, epochs + 1):
        permutation = torch.randperm(n_samples, device=device)
        for i in range(0, n_samples, batch_size):
            indices = permutation[i : i + batch_size]
            batch_x, batch_y = X_train_t[indices], y_train_t[indices]
            optimizer.zero_grad()
            loss = criterion(model(batch_x), batch_y)
            loss.backward()
            optimizer.step()
    return model

def main():
    parser = argparse.ArgumentParser(description="Elec2 Model Training")
    parser.add_argument("--reference-size", type=int, default=5000)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("⏬ Downloading/Loading Electricity Market Dataset (Elec2) from OpenML...")
    elec = fetch_openml('electricity', version=1, as_frame=True, parser='auto')
    df = elec.frame
    
    target_col = 'class'
    df = df.dropna()
    y = LabelEncoder().fit_transform(df[target_col])
    X = df.drop(columns=[target_col])
    
    numeric_features = X.select_dtypes(include=['float64', 'int64']).columns.tolist()

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', Pipeline([('imputer', SimpleImputer(strategy='median')), ('scaler', StandardScaler())]), numeric_features),
        ], remainder='drop'
    )

    print(f"📊 Dataset shape: {df.shape}. Using {args.reference_size} for reference, rest for streaming.")
    
    X_ref_df = X.iloc[:args.reference_size].copy()
    y_ref = y[:args.reference_size]
    X_stream_df = X.iloc[args.reference_size:].copy()
    y_stream = y[args.reference_size:]

    print("🏋️ Training PyTorch MLP on Reference Data...")
    X_ref_scaled = preprocessor.fit_transform(X_ref_df).astype(np.float32)
    model = Elec2MLP(input_dim=X_ref_scaled.shape[1], num_classes=2)
    train_elec2(model, X_ref_scaled, y_ref, device=device, epochs=15)
    print("✅ Training complete.")

    artifacts_dir = Path("artifacts")
    artifacts_dir.mkdir(exist_ok=True)
    
    # Save Model
    torch.save(model.state_dict(), artifacts_dir / "elec2_model.pth")
    # Save Preprocessor
    with open(artifacts_dir / "elec2_preprocessor.pkl", "wb") as f:
        pickle.dump(preprocessor, f)
        
    # Save Data
    X_ref_df.to_csv(artifacts_dir / "elec2_reference_features.csv", index=False)
    X_stream_df.to_csv(artifacts_dir / "elec2_stream_features.csv", index=False)
    np.save(artifacts_dir / "elec2_reference_labels.npy", y_ref)
    np.save(artifacts_dir / "elec2_stream_labels.npy", y_stream)
    
    print(f"Stored training artifacts in {artifacts_dir}/")

if __name__ == "__main__":
    main()
