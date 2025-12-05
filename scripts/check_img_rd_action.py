# # debug_action_stats.py
# import zarr
# import numpy as np

# # 你的数据集路径
# zarr_path = "task_data/drone_goto_image_rd"

# # 打开数据集
# root = zarr.open(zarr_path, 'r')
# print("Keys:", list(root.keys()))

# # 读取 action
# actions = root['data']['action'][:]
# print(f"\nAction shape: {actions.shape}")
# print(f"Action dtype: {actions.dtype}")

# # 统计
# print(f"\n=== Action Statistics ===")
# print(f"Mean: {actions.mean(axis=0)}")
# print(f"Std:  {actions.std(axis=0)}")
# print(f"Min:  {actions.min(axis=0)}")
# print(f"Max:  {actions.max(axis=0)}")

# # Norm 统计（最重要！）
# norms = np.linalg.norm(actions, axis=-1)
# print(f"\n=== Action Norm Statistics ===")
# print(f"Norm mean: {norms.mean():.4f}")
# print(f"Norm std:  {norms.std():.4f}")
# print(f"Norm min:  {norms.min():.4f}")
# print(f"Norm max:  {norms.max():.4f}")
# print(f"Norm 95th percentile: {np.percentile(norms, 95):.4f}")

# # 建议的 scale_factor
# suggested_scale = 10.0 / np.percentile(norms, 95)
# print(f"\n=== Suggested scale_factor ===")
# print(f"Current: 10.0")
# print(f"Suggested: {suggested_scale:.1f}")

# debug_spherical_stats.py
import zarr
import numpy as np

zarr_path = "task_data/drone_goto_image_rd"
root = zarr.open(zarr_path, 'r')
actions = root['data']['action'][:]

# 转换到 spherical coordinates
x, y, z = actions[:, 0], actions[:, 1], actions[:, 2]
r = np.linalg.norm(actions, axis=-1)
theta = np.arccos(np.clip(z / (r + 1e-8), -1, 1))  # [0, π]
phi = np.arctan2(y, x)  # [-π, π]

print("=== Spherical Coordinates ===")
print(f"r:     mean={r.mean():.4f}, std={r.std():.4f}, min={r.min():.4f}, max={r.max():.4f}")
print(f"theta: mean={theta.mean():.4f}, std={theta.std():.4f}, min={theta.min():.4f}, max={theta.max():.4f}")
print(f"phi:   mean={phi.mean():.4f}, std={phi.std():.4f}, min={phi.min():.4f}, max={phi.max():.4f}")

print("\n=== Rectangular (原始) ===")
print(f"x: mean={x.mean():.4f}, std={x.std():.4f}")
print(f"y: mean={y.mean():.4f}, std={y.std():.4f}")
print(f"z: mean={z.mean():.4f}, std={z.std():.4f}")