import pandas as pd
import torch
import numpy as np

def main():
    import sys
    print("Testing SAE...")
    # Load SAE weights
    sys.path.insert(0, ".")
    from task11_sae import TopKSAE
    device = "cpu"
    sae = TopKSAE(d_model=768, expansion_factor=8, k=32).to(device)
    sae.load_state_dict(torch.load("results/sae_weights.pth", map_location=device))
    sae.eval()

    print("Loaded SAE. Checking W_dec for 1449...")
    print(sae.W_dec[1449][:5])
    
    # We will simulate passing 1s
    X = torch.ones(10, 768)
    _, sparse_acts = sae(X)
    print("Max activation of 1449:", sparse_acts[:, 1449].max().item())
    
main()
