"""Check what the training actually logged"""
from pathlib import Path

output_dir = Path("data/outputs/2025.11.21/07.43.37_train_so3_implicit_lowdim_policy_drone_go_to")

print("=" * 60)
print("Check Training Logs")
print("=" * 60)

# Method 1: Check stdout/stderr if captured
log_files = list(output_dir.glob("*.log")) + list(output_dir.glob("*.out"))
print(f"\n[LOG FILES] Found: {[f.name for f in log_files]}")

for log_file in log_files:
    print(f"\n[FILE] {log_file.name}")
    with open(log_file) as f:
        lines = f.readlines()
        
    # Look for evaluation results
    for i, line in enumerate(lines):
        if 'Evaluation completed' in line or 'rewards' in line.lower() or 'mean_score' in line:
            # Print context around this line
            start = max(0, i - 2)
            end = min(len(lines), i + 5)
            for j in range(start, end):
                marker = " >>> " if j == i else "     "
                print(f"{marker}{lines[j].rstrip()}")
            print()

# Method 2: Check wandb logs
wandb_dir = output_dir / "wandb"
if wandb_dir.exists():
    print(f"\n[WANDB] Checking wandb logs...")
    
    # Find latest run
    runs = sorted(wandb_dir.glob("run-*/"))
    if runs:
        latest_run = runs[-1]
        print(f"  Latest run: {latest_run.name}")
        
        # Check if wandb has the data
        import wandb
        api = wandb.Api()
        try:
            # You'll need to find the run ID
            run_id = latest_run.name.split('-')[-1]
            run = api.run(f"andyliu7081-northeastern-university/drone_go_to/{run_id}")
            
            history = run.history(keys=["test/mean_score", "train/mean_score"])
            print(f"\n  History shape: {history.shape}")
            print(f"\n  test/mean_score values:")
            print(history[["_step", "test/mean_score"]].dropna())
        except Exception as e:
            print(f"  Could not fetch wandb data: {e}")