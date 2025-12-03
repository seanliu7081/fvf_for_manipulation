# import zarr
# import numpy as np

# z = zarr.open("/media/lht/T7TwoTB/code/fourier_value_functions/task_data/drone_goto_image")
# print(z.tree())

# if 'keypoints' in z:
#     kp = z['keypoints']
#     print(f"\nkeypoints shape: {kp.shape}")
#     print(f"keypoints[0]: {kp[0]}")
#     print(f"keypoints range: [{kp[:].min()}, {kp[:].max()}]")
# elif 'keypoint' in z:
#     kp = z['keypoint']
#     print(f"\nkeypoint shape: {kp.shape}")
#     print(f"keypoint[0]: {kp[0]}")
#     print(f"keypoint range: [{kp[:].min()}, {kp[:].max()}]")

# # 检查 keypoints
# # kp = z['data/keypoints']
# # print(f"keypoints shape: {kp.shape}")
# # print(f"keypoints[0]: {kp[0]}")
# # print(f"keypoints[100]: {kp[100]}")
# # print(f"keypoints range: [{kp[:].min()}, {kp[:].max()}]")
# # print(f"keypoints all zeros?: {(kp[:] == 0).all()}")
# # print(f"keypoints zero ratio: {(kp[:] == 0).sum() / kp[:].size:.4f}")

# kp = z['data/keypoints'][:]
# episode_ends = z['meta/episode_ends'][:]

# print(f"Episode ends: {episode_ends[:10]}")
# print(f"Total episodes: {len(episode_ends)}")

# # 检查每个 episode 的 keypoints
# start = 0
# for i, end in enumerate(episode_ends[:5]):
#     ep_kp = kp[start:end]
#     zero_ratio = (ep_kp == 0).sum() / ep_kp.size
#     print(f"Episode {i}: [{start}:{end}], zero_ratio={zero_ratio:.2f}, first_kp={ep_kp[0]}")
#     start = end

# # 找一个非零的 keypoints
# nonzero_idx = np.where(np.abs(kp).sum(axis=(1,2)) > 0.01)[0]
# print(f"\nNon-zero keypoints indices (first 10): {nonzero_idx[:10]}")
# if len(nonzero_idx) > 0:
#     print(f"First non-zero keypoint: {kp[nonzero_idx[0]]}")


import zarr
import matplotlib.pyplot as plt

z = zarr.open('task_data/drone_goto_image', 'r')
imgs = z['data/img']

fig, axes = plt.subplots(2, 5, figsize=(15, 6))
for i, ax in enumerate(axes.flat):
    idx = i * 100
    ax.imshow(imgs[idx])
    ax.set_title(f'Frame {idx}')
    ax.axis('off')
plt.tight_layout()
plt.savefig('sample_images_third_person.png')
print('Saved to sample_images_third_person.png')
