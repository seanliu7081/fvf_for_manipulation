# # # 创建一个测试脚本 debug_dataset.py
# # import torch
# # import numpy as np
# # from fvf.dataset.drone_dataset import DroneDataset
# # from fvf.dataset.drone_image_dataset import DroneImageDataset

# # # 加载两个数据集
# # ds_kp = DroneDataset(
# #     path="/media/lht/T7TwoTB/code/fourier_value_functions/task_data/drone_goto",
# #     horizon=2, 
# #     pad_before=1, 
# #     pad_after=0, 
# #     action_coords="spherical"
# # )

# # ds_img = DroneImageDataset(
# #     path="/media/lht/T7TwoTB/code/fourier_value_functions/task_data/drone_goto_image",
# #     horizon=2, 
# #     pad_before=1, 
# #     pad_after=0, 
# #     action_coords="spherical",
# #     crop_image_size=84
# # )

# # print("=== DroneDataset (works, 0.82) ===")
# # sample_kp = ds_kp[0]
# # print(f"keys: {sample_kp.keys()}")
# # print(f"obs keys: {sample_kp['obs'].keys()}")
# # print(f"action shape: {sample_kp['action'].shape}")
# # print(f"action[0]: {sample_kp['action'][0]}")
# # print(f"keypoints shape: {sample_kp['obs']['keypoints'].shape}")
# # print(f"keypoints[0]: {sample_kp['obs']['keypoints'][0]}")

# # print("\n=== DroneImageDataset (doesn't work) ===")
# # sample_img = ds_img[0]
# # print(f"keys: {sample_img.keys()}")
# # print(f"obs keys: {sample_img['obs'].keys()}")
# # print(f"action shape: {sample_img['action'].shape}")
# # print(f"action[0]: {sample_img['action'][0]}")
# # print(f"keypoints shape: {sample_img['obs']['keypoints'].shape}")
# # print(f"keypoints[0]: {sample_img['obs']['keypoints'][0]}")

# # # 检查 normalizer
# # print("\n=== Normalizers ===")
# # norm_kp = ds_kp.get_normalizer()
# # norm_img = ds_img.get_normalizer()

# # print("Keypoint dataset action stats:")
# # print(f"  {norm_kp['action'].get_input_stats()}")

# # print("Image dataset action stats:")
# # print(f"  {norm_img['action'].get_input_stats()}")


# import torch
# import numpy as np
# from fvf.dataset.drone_dataset import DroneDataset
# from fvf.dataset.drone_image_dataset import DroneImageDataset

# ds_kp = DroneDataset(
#     path="/media/lht/T7TwoTB/code/fourier_value_functions/task_data/drone_goto",
#     horizon=2, pad_before=1, pad_after=0, action_coords="spherical"
# )

# ds_img = DroneImageDataset(
#     path="/media/lht/T7TwoTB/code/fourier_value_functions/task_data/drone_goto_image",
#     horizon=2, pad_before=1, pad_after=0, action_coords="spherical", crop_image_size=84
# )

# print("=== Sample 0 ===")
# s_kp = ds_kp[0]
# s_img = ds_img[0]

# print(f"DroneDataset keypoints shape: {s_kp['obs']['keypoints'].shape}")
# print(f"DroneImageDataset keypoints shape: {s_img['obs']['keypoints'].shape}")

# print(f"\nDroneDataset keypoints:\n{s_kp['obs']['keypoints']}")
# print(f"\nDroneImageDataset keypoints:\n{s_img['obs']['keypoints']}")

# print(f"\nDroneDataset action: {s_kp['action']}")
# print(f"DroneImageDataset action: {s_img['action']}")

# # 检查 normalizer
# norm_kp = ds_kp.get_normalizer()
# norm_img = ds_img.get_normalizer()

# print("\n=== Normalizer keys ===")
# print(f"DroneDataset: {list(norm_kp.params_dict.keys())}")
# print(f"DroneImageDataset: {list(norm_img.params_dict.keys())}")

# # 测试 normalize
# print("\n=== After normalization ===")
# nkp = norm_kp['keypoints'].normalize(s_kp['obs']['keypoints'])
# print(f"DroneDataset normalized keypoints shape: {nkp.shape}")
# print(f"DroneDataset normalized keypoints:\n{nkp}")

# nimg_kp = norm_img['keypoints'].normalize(s_img['obs']['keypoints'].view(-1, 9))
# print(f"\nDroneImageDataset normalized keypoints shape: {nimg_kp.shape}")
# print(f"DroneImageDataset normalized keypoints:\n{nimg_kp}")

import torch
from fvf.dataset.drone_dataset import DroneDataset
from fvf.dataset.drone_image_dataset import DroneImageDataset

# 加载两个 dataset
ds1 = DroneDataset(
    path='/media/lht/T7TwoTB/code/fourier_value_functions/task_data/drone_goto',
    horizon=2,
    pad_before=1,
    pad_after=0,
    action_coords='spherical'
)

ds2 = DroneImageDataset(
    path='/media/lht/T7TwoTB/code/fourier_value_functions/task_data/drone_goto_image',
    horizon=2,
    pad_before=1,
    pad_after=0,
    action_coords='spherical',
    crop_image_size=84
)

print('=== DroneDataset ===')
sample1 = ds1[0]
print(f"keypoints shape: {sample1['obs']['keypoints'].shape}")
print(f"keypoints[0]: {sample1['obs']['keypoints'][0]}")
print(f"action shape: {sample1['action'].shape}")
print(f"action[0]: {sample1['action'][0]}")

print()
print('=== DroneImageDataset ===')
sample2 = ds2[0]
print(f"keypoints shape: {sample2['obs']['keypoints'].shape}")
print(f"keypoints[0]: {sample2['obs']['keypoints'][0]}")
print(f"action shape: {sample2['action'].shape}")
print(f"action[0]: {sample2['action'][0]}")