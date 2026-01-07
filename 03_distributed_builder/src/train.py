"""
Training Loop Base
==================
Simple training loop per ODIN-100M.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from pathlib import Path
import yaml
import sys

# Add architect's code to path
sys.path.insert(0, str(Path(__file__).parent.parent / "01_architect_mathematician" / "src"))


def load_config(config_path: str) -> dict:
    """Carica configurazione YAML"""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def train_step(model, batch, optimizer, criterion):
    """Singolo step di training"""
    optimizer.zero_grad()
    
    input_ids = batch['input_ids']
    targets = batch['targets']
    
    logits, _ = model(input_ids)
    
    # Shift per autoregressive loss
    logits = logits[:, :-1, :].contiguous()
    targets = targets[:, 1:].contiguous()
    
    loss = criterion(logits.view(-1, logits.size(-1)), targets.view(-1))
    
    loss.backward()
    
    # Gradient clipping
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    
    optimizer.step()
    
    return loss.item()


def train_epoch(model, dataloader, optimizer, criterion, device):
    """Training per una epoch"""
    model.train()
    total_loss = 0
    
    for batch_idx, batch in enumerate(dataloader):
        batch = {k: v.to(device) for k, v in batch.items()}
        loss = train_step(model, batch, optimizer, criterion)
        total_loss += loss
        
        if batch_idx % 100 == 0:
            print(f"  Batch {batch_idx}, Loss: {loss:.4f}")
    
    return total_loss / len(dataloader)


def main():
    """Main training entry point"""
    # Config
    config_path = Path(__file__).parent.parent / "01_architect_mathematician" / "config.yaml"
    config = load_config(config_path)
    
    print("=" * 50)
    print("ODIN-100M Training")
    print("=" * 50)
    print(f"Model: {config['model']['name']}")
    print(f"Architecture: {config['model']['architecture']}")
    
    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    # TODO: Initialize model from architect's code
    # TODO: Initialize dataset from data chef's generators
    # TODO: Initialize optimizer with config parameters
    
    print("\n[Placeholder] Training loop ready.")
    print("Waiting for:")
    print("  - Complete model from Architect")
    print("  - Dataset from Data Chef")


if __name__ == "__main__":
    main()
