#!/usr/bin/env python3
"""
Augmentation Visualization Script for PushT Dataset
Usage: python visualize_augmentation.py [--mode MODE] [--sample SAMPLE] [--rotation ROT] [--tx TX] [--ty TY]
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import numpy as np
import matplotlib.pyplot as plt
import cv2
from matplotlib.widgets import Slider, Button, RadioButtons

from fvf.dataset.pusht_image_dataset import PushTImageDatasetAug
from fvf.utils import data_augmentation


class AugmentationVisualizer:
    def __init__(self, dataset_path='pusht_demo.zarr'):
        """Initialize the visualizer with datasets"""
        print("Loading datasets...")
        
        # Load dataset with augmentation
        self.dataset_aug = PushTImageDatasetAug(
            path=dataset_path,
            horizon=2,
            pad_before=1,
            pad_after=0,
            seed=42,
            use_augmentation=True,
            max_rotation_deg=50.0,
            max_translation_pix=15
        )
        
        # Load dataset without augmentation
        self.dataset_no_aug = PushTImageDatasetAug(
            path=dataset_path,
            horizon=2,
            pad_before=1,
            pad_after=0,
            seed=42,
            use_augmentation=False
        )
        
        print(f"Dataset loaded. Size: {len(self.dataset_aug)} samples")
        
        # Initialize parameters
        self.sample_idx = 0
        self.rotation_deg = 0.0
        self.translation_x = 0.0
        self.translation_y = 0.0
        self.mode = 'manual'
        
    def apply_manual_augmentation(self, img, agent_pos, action, 
                                  rotation_deg=0.0,
                                  translation_x=0.0,
                                  translation_y=0.0):
        """Apply exact (deterministic) rotation and translation"""
        T, C, H, W = img.shape
        
        angle_rad = np.deg2rad(rotation_deg)
        center_x, center_y = W / 2, H / 2
        
        # Create rotation matrix for image
        M_img = cv2.getRotationMatrix2D((center_x, center_y), rotation_deg, 1.0)
        M_img[0, 2] += translation_x
        M_img[1, 2] += translation_y
        
        # Apply to images
        aug_img = np.zeros_like(img)
        for t in range(T):
            for c in range(C):
                aug_img[t, c] = cv2.warpAffine(
                    img[t, c], M_img, (W, H),
                    flags=cv2.INTER_LINEAR,
                    borderMode=cv2.BORDER_REPLICATE
                )
        
        # Create rotation matrix for coordinates
        cos_a = np.cos(angle_rad)
        sin_a = np.sin(angle_rad)
        R = np.array([[cos_a, -sin_a],
                      [sin_a, cos_a]])
        
        # Apply to positions and actions
        aug_agent_pos = np.zeros_like(agent_pos)
        aug_action = np.zeros_like(action)
        
        for t in range(T):
            aug_agent_pos[t] = R @ agent_pos[t] + np.array([translation_x, translation_y])
            aug_action[t] = R @ action[t]
        
        return aug_img, aug_agent_pos, aug_action
    
    def visualize(self, mode='manual', sample_idx=0, rotation_deg=0.0, 
                  translation_x=0.0, translation_y=0.0, save_path=None):
        """Main visualization function"""
        
        fig = plt.figure(figsize=(18, 10))
        
        if mode == 'manual':
            # Manual mode: Apply exact transformations
            fig.suptitle(f'Manual Augmentation - Sample {sample_idx}\n' + 
                        f'Rotation: {rotation_deg:.1f}°, Translation: ({translation_x:.1f}, {translation_y:.1f})', 
                        fontsize=14)
            
            # Get raw sample
            sample = self.dataset_no_aug.sampler.sample_sequence(sample_idx)
            
            # Process data
            T = sample['img'].shape[0]
            x_pos = (sample['state'][:,0] - 255.0)
            y_pos = (sample['state'][:,1] - 255.0) * -1
            agent_pos_orig = np.concatenate((x_pos[..., np.newaxis], y_pos[..., np.newaxis]), axis=-1).reshape(T, 2)
            obs_orig = np.moveaxis(sample['img'],-1,1) / 255
            action_orig = sample['action'].copy()
            
            # Apply manual augmentation
            obs_aug, agent_pos_aug, action_aug = self.apply_manual_augmentation(
                obs_orig.copy(), 
                agent_pos_orig.copy(), 
                action_orig.copy(),
                rotation_deg=rotation_deg,
                translation_x=translation_x,
                translation_y=translation_y
            )
            
        else:  # mode == 'dataset'
            # Dataset mode: Show actual dataset augmentation
            fig.suptitle(f'Dataset Augmentation - Sample {sample_idx}\n' + 
                        f'Config: Max Rotation ±{self.dataset_aug.max_rotation_deg}°, ' +
                        f'Max Translation ±{self.dataset_aug.max_translation_pix}px', 
                        fontsize=14)
            
            # Get samples
            aug_data = self.dataset_aug[sample_idx]
            orig_data = self.dataset_no_aug[sample_idx]
            
            # Convert to numpy
            obs_orig = orig_data['obs']['image'].numpy()
            obs_aug = aug_data['obs']['image'].numpy()
            agent_pos_orig = orig_data['obs']['agent_pos'].numpy()
            agent_pos_aug = aug_data['obs']['agent_pos'].numpy()
            action_orig = orig_data['action'].numpy()
            action_aug = aug_data['action'].numpy()
        
        # Create subplots
        axes = []
        for i in range(6):
            ax = plt.subplot(2, 3, i+1)
            axes.append(ax)
        
        t = 0  # First timestep
        
        # Plot images and analysis
        self._plot_images(axes, obs_orig[t], obs_aug[t], 
                         agent_pos_orig[t], agent_pos_aug[t],
                         action_orig[t], action_aug[t])
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Saved visualization to {save_path}")
        
        plt.show()
        
        # Print statistics
        self._print_statistics(sample_idx, agent_pos_orig[t], agent_pos_aug[t],
                              action_orig[t], action_aug[t])
    
    def _plot_images(self, axes, img_orig, img_aug, pos_orig, pos_aug, act_orig, act_aug):
        """Helper function to plot all subplots"""
        center_x, center_y = 96/2, 96/2
        scale = 15
        
        # 1. Original Image
        ax = axes[0]
        img_orig_vis = np.transpose(img_orig, (1, 2, 0))
        ax.imshow(img_orig_vis)
        ax.set_title('Original Image', fontsize=12)
        ax.axis('off')
        
        # 2. Augmented Image
        ax = axes[1]
        img_aug_vis = np.transpose(img_aug, (1, 2, 0))
        ax.imshow(img_aug_vis)
        ax.set_title('Augmented Image', fontsize=12)
        ax.axis('off')
        
        # 3. Difference Heatmap
        ax = axes[2]
        diff = np.abs(img_aug_vis - img_orig_vis)
        im = ax.imshow(np.mean(diff, axis=2), cmap='hot', vmin=0, vmax=0.2)
        ax.set_title('Difference Heatmap', fontsize=12)
        ax.axis('off')
        plt.colorbar(im, ax=ax, fraction=0.046)
        
        # 4. Original with Vectors
        ax = axes[3]
        ax.imshow(img_orig_vis)
        agent_x = center_x + pos_orig[0]
        agent_y = center_y - pos_orig[1]
        ax.plot(agent_x, agent_y, 'bo', markersize=10, label='Position')
        ax.arrow(agent_x, agent_y, 
                act_orig[0] * scale, 
                -act_orig[1] * scale,
                head_width=3, head_length=2, fc='blue', ec='blue', linewidth=2)
        ax.legend(loc='upper right', fontsize=8)
        ax.set_title('Original + Vectors', fontsize=12)
        ax.axis('off')
        
        # 5. Augmented with Vectors
        ax = axes[4]
        ax.imshow(img_aug_vis)
        agent_x_aug = center_x + pos_aug[0]
        agent_y_aug = center_y - pos_aug[1]
        ax.plot(agent_x_aug, agent_y_aug, 'ro', markersize=10, label='Aug Position')
        ax.arrow(agent_x_aug, agent_y_aug,
                act_aug[0] * scale,
                -act_aug[1] * scale,
                head_width=3, head_length=2, fc='red', ec='red', linewidth=2)
        ax.legend(loc='upper right', fontsize=8)
        ax.set_title('Augmented + Vectors', fontsize=12)
        ax.axis('off')
        
        # 6. Overlay Comparison
        ax = axes[5]
        ax.imshow(img_aug_vis, alpha=0.7)
        # Original in blue
        ax.plot(agent_x, agent_y, 'bo', markersize=8, alpha=0.5, label='Original')
        ax.arrow(agent_x, agent_y, 
                act_orig[0] * scale, 
                -act_orig[1] * scale,
                head_width=2, head_length=1.5, fc='blue', ec='blue', alpha=0.5, linewidth=1.5)
        # Augmented in red
        ax.plot(agent_x_aug, agent_y_aug, 'ro', markersize=10, label='Augmented')
        ax.arrow(agent_x_aug, agent_y_aug,
                act_aug[0] * scale,
                -act_aug[1] * scale,
                head_width=3, head_length=2, fc='red', ec='red', linewidth=2)
        ax.legend(loc='upper right', fontsize=8)
        ax.set_title('Overlay Comparison', fontsize=12)
        ax.axis('off')
    
    def _print_statistics(self, sample_idx, pos_orig, pos_aug, act_orig, act_aug):
        """Print augmentation statistics with detailed rotation analysis"""
        print(f"\n{'='*60}")
        print(f"Statistics for Sample {sample_idx}:")
        print(f"{'='*60}")
        
        # Method 1: Calculate rotation from action vectors
        rotation_from_action = 0.0
        if np.linalg.norm(act_orig) > 0.001:
            angle_orig = np.arctan2(act_orig[1], act_orig[0])
            angle_aug = np.arctan2(act_aug[1], act_aug[0])
            rotation_from_action = np.rad2deg(angle_aug - angle_orig)
            
            # Normalize to [-180, 180]
            rotation_from_action = (rotation_from_action + 180) % 360 - 180
        
        # Method 2: Calculate rotation from position change (if position moved)
        # This assumes the rotation center is at origin (0,0)
        rotation_from_position = 0.0
        if np.linalg.norm(pos_orig) > 0.001:
            # Remove translation component first
            # Estimate translation as the change in position
            translation = pos_aug - pos_orig
            
            # For pure rotation around origin, the magnitude should be preserved
            # We can estimate rotation angle from the position vectors
            angle_pos_orig = np.arctan2(pos_orig[1], pos_orig[0])
            angle_pos_aug = np.arctan2(pos_aug[1] - translation[1], 
                                       pos_aug[0] - translation[0])
            rotation_from_position = np.rad2deg(angle_pos_aug - angle_pos_orig)
            rotation_from_position = (rotation_from_position + 180) % 360 - 180
        
        # Calculate translation
        translation_applied = pos_aug - pos_orig
        
        # For manual mode, we can also compare with expected values
        if hasattr(self, 'last_manual_rotation'):
            print(f"\nExpected Transformation (Manual Mode):")
            print(f"  Rotation: {self.last_manual_rotation:.2f}°")
            print(f"  Translation: ({self.last_manual_tx:.2f}, {self.last_manual_ty:.2f}) pixels")
        
        print(f"\nActual Transformation Applied:")
        print(f"  Rotation (from action vectors): {rotation_from_action:.2f}°")
        if abs(rotation_from_position) > 0.01:
            print(f"  Rotation (from position): {rotation_from_position:.2f}°")
        print(f"  Translation: ({translation_applied[0]:.2f}, {translation_applied[1]:.2f}) pixels")
        
        print(f"\nMagnitude Changes:")
        print(f"  Position change (L2): {np.linalg.norm(pos_orig - pos_aug):.4f}")
        print(f"  Action change (L2): {np.linalg.norm(act_orig - act_aug):.4f}")
        print(f"  Action magnitude change: {np.linalg.norm(act_aug):.4f} - {np.linalg.norm(act_orig):.4f} = {np.linalg.norm(act_aug) - np.linalg.norm(act_orig):.4f}")
        
        print(f"\nOriginal values:")
        print(f"  Position: [{pos_orig[0]:.2f}, {pos_orig[1]:.2f}]")
        print(f"  Action: [{act_orig[0]:.3f}, {act_orig[1]:.3f}] (magnitude: {np.linalg.norm(act_orig):.3f})")
        
        print(f"\nAugmented values:")
        print(f"  Position: [{pos_aug[0]:.2f}, {pos_aug[1]:.2f}]")
        print(f"  Action: [{act_aug[0]:.3f}, {act_aug[1]:.3f}] (magnitude: {np.linalg.norm(act_aug):.3f})")
    
    def interactive_mode(self):
        """Run interactive matplotlib mode with sliders"""
        fig = plt.figure(figsize=(18, 12))
        
        # Create initial plot
        self.visualize('manual', 0, 0, 0, 0)
        
        # Add sliders
        ax_sample = plt.axes([0.15, 0.02, 0.3, 0.03])
        ax_rot = plt.axes([0.15, 0.06, 0.3, 0.03])
        ax_tx = plt.axes([0.55, 0.02, 0.3, 0.03])
        ax_ty = plt.axes([0.55, 0.06, 0.3, 0.03])
        
        slider_sample = Slider(ax_sample, 'Sample', 0, min(100, len(self.dataset_aug)-1), 
                              valinit=0, valstep=1)
        slider_rot = Slider(ax_rot, 'Rotation', -90, 90, valinit=0, valstep=1)
        slider_tx = Slider(ax_tx, 'Trans X', -30, 30, valinit=0, valstep=1)
        slider_ty = Slider(ax_ty, 'Trans Y', -30, 30, valinit=0, valstep=1)
        
        def update(val):
            plt.clf()
            self.visualize('manual', 
                          int(slider_sample.val),
                          slider_rot.val,
                          slider_tx.val,
                          slider_ty.val)
            plt.draw()
        
        slider_sample.on_changed(update)
        slider_rot.on_changed(update)
        slider_tx.on_changed(update)
        slider_ty.on_changed(update)
        
        plt.show()
    
    def compare_augmentation_levels(self, sample_idx=5, save_path=None):
        """Compare different augmentation strengths"""
        fig, axes = plt.subplots(3, 4, figsize=(16, 12))
        
        sample = self.dataset_no_aug.sampler.sample_sequence(sample_idx)
        
        T = sample['img'].shape[0]
        x_pos = (sample['state'][:,0] - 255.0)
        y_pos = (sample['state'][:,1] - 255.0) * -1
        agent_pos_orig = np.concatenate((x_pos[..., np.newaxis], y_pos[..., np.newaxis]), axis=-1).reshape(T, 2)
        obs_orig = np.moveaxis(sample['img'],-1,1) / 255
        action_orig = sample['action'].copy()
        
        test_params = [
            (0, 0, 0),      # No augmentation
            (5, 0, 0),      # Small rotation only
            (15, 0, 0),     # Medium rotation only
            (30, 0, 0),     # Large rotation only
            (0, 5, 0),      # Small X translation
            (0, 0, 5),      # Small Y translation
            (0, 10, 10),    # Medium translation
            (5, 5, 5),      # Small both
            (15, 10, 10),   # Medium both
            (30, 15, 15),   # Large both
            (45, 20, 20),   # Very large both
            (-30, -10, 10), # Negative rotation
        ]
        
        for idx, (rot, tx, ty) in enumerate(test_params):
            row = idx // 4
            col = idx % 4
            
            obs_aug, _, _ = self.apply_manual_augmentation(
                obs_orig.copy(), agent_pos_orig.copy(), action_orig.copy(),
                rotation_deg=rot, translation_x=tx, translation_y=ty
            )
            
            ax = axes[row, col]
            ax.imshow(np.transpose(obs_aug[0], (1, 2, 0)))
            ax.set_title(f'R:{rot}° T:({tx},{ty})', fontsize=10)
            ax.axis('off')
        
        plt.suptitle(f'Augmentation Strength Comparison - Sample {sample_idx}', fontsize=14)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Saved comparison to {save_path}")
        
        plt.show()


def main():
    parser = argparse.ArgumentParser(description='Visualize data augmentation for PushT dataset')
    parser.add_argument('--dataset', type=str, default='pusht_demo.zarr',
                       help='Path to dataset')
    parser.add_argument('--mode', type=str, choices=['manual', 'dataset', 'interactive', 'compare'],
                       default='interactive', help='Visualization mode')
    parser.add_argument('--sample', type=int, default=0, help='Sample index')
    parser.add_argument('--rotation', type=float, default=0, help='Rotation in degrees (manual mode)')
    parser.add_argument('--tx', type=float, default=0, help='Translation X in pixels (manual mode)')
    parser.add_argument('--ty', type=float, default=0, help='Translation Y in pixels (manual mode)')
    parser.add_argument('--save', type=str, default=None, help='Save path for visualization')
    
    args = parser.parse_args()
    
    # Initialize visualizer
    viz = AugmentationVisualizer(args.dataset)
    
    # Run visualization based on mode
    if args.mode == 'interactive':
        print("Running interactive mode. Use sliders to control augmentation.")
        viz.interactive_mode()
    elif args.mode == 'compare':
        print("Comparing different augmentation levels...")
        viz.compare_augmentation_levels(args.sample, args.save)
    else:
        viz.visualize(args.mode, args.sample, args.rotation, args.tx, args.ty, args.save)


if __name__ == "__main__":
    main()