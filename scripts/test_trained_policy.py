"""Test trained policy performance"""
import numpy as np
import torch
import sys
sys.path.append('/media/lht/T7TwoTB/code/fourier_value_functions')

from fvf.env.drone.go_to_target_env import GoToTargetEnv
from pathlib import Path

print("=" * 60)
print("Test Trained Policy")
print("=" * 60)

# Find latest checkpoint - Fixed version
checkpoint_path = Path("data/outputs/2025.11.21/06.09.09_train_so3_implicit_lowdim_policy_drone_go_to/checkpoints/epoch=0010-test_mean_score=0.000.ckpt")

if not checkpoint_path.exists():
    print(f"[ERROR] Checkpoint not found: {checkpoint_path}")
    
    # Try to find any checkpoint
    print("\n[INFO] Searching for any checkpoint...")
    checkpoint_dirs = list(Path("data/outputs").glob("*/checkpoints"))
    if checkpoint_dirs:
        print(f"[INFO] Found {len(checkpoint_dirs)} checkpoint directories:")
        for d in checkpoint_dirs:
            ckpts = list(d.glob("*.ckpt"))
            print(f"  - {d}: {len(ckpts)} checkpoints")
    exit(1)

print(f"[OK] Loading checkpoint: {checkpoint_path}")

# Load checkpoint
ckpt = torch.load(checkpoint_path, map_location='cpu')
print(f"     Epoch: {ckpt.get('epoch', 'unknown')}")

# Check if policy was trained
if 'state_dict' in ckpt:
    print(f"     Number of parameters: {len(ckpt['state_dict'])}")
    
    # Check if parameters are all zeros (not trained)
    param_values = [v.abs().sum().item() for v in ckpt['state_dict'].values() if v.dtype == torch.float32]
    if len(param_values) > 0:
        avg_param = np.mean(param_values)
        print(f"     Average parameter value: {avg_param:.6f}")
        
        if avg_param < 1e-6:
            print("     [WARNING] Parameters near zero, possibly not trained")
        else:
            print("     [OK] Parameters updated")
    else:
        print("     [WARNING] No float parameters found")
else:
    print("     [ERROR] No state_dict in checkpoint")

print("\n[INFO] To test actual policy performance, check WandB evaluation logs")