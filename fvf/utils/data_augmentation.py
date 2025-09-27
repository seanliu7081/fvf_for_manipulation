import numpy as np
import numpy.random as npr
import cv2

def random_crop(img, out_size):
    T, C, H, W = img.shape

    x = npr.randint(0, W - out_size)
    y = npr.randint(0, H - out_size)
    img = img[:, :, y:y+out_size, x:x+out_size]

    return img

def random_rotation_translation(img, agent_pos, action, 
                                max_rotation_deg=5.0,  # Small rotation
                                max_translation_pix=5):  # Small translation
    """
    Apply small random rotation and translation to images and corresponding actions.
    
    Args:
        img: Image tensor of shape (T, C, H, W)
        agent_pos: Agent position of shape (T, 2)
        action: Actions of shape (T, 2)
        max_rotation_deg: Maximum rotation in degrees
        max_translation_pix: Maximum translation in pixels
    
    Returns:
        Augmented (img, agent_pos, action)
    """
    T, C, H, W = img.shape
    
    # Random rotation angle (same for all timesteps in sequence)
    angle_deg = npr.uniform(-max_rotation_deg, max_rotation_deg)
    angle_rad = np.deg2rad(angle_deg)
    
    # Random translation (same for all timesteps in sequence)
    tx = npr.uniform(-max_translation_pix, max_translation_pix)
    ty = npr.uniform(-max_translation_pix, max_translation_pix)
    
    # Image center for rotation
    center_x, center_y = W / 2, H / 2
    
    # Create rotation matrix for image
    M_img = cv2.getRotationMatrix2D((center_x, center_y), angle_deg, 1.0)
    M_img[0, 2] += tx  # Add translation
    M_img[1, 2] += ty
    
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
    
    # Apply to agent positions and actions
    # First, shift to image center, rotate, then shift back and translate
    aug_agent_pos = np.zeros_like(agent_pos)
    aug_action = np.zeros_like(action)
    
    for t in range(T):
        # Agent position (already in centered coordinates from dataset)
        aug_agent_pos[t] = R @ agent_pos[t] + np.array([tx, ty])
        
        # Actions (only need rotation)
        aug_action[t] = R @ action[t]
    
    return aug_img, aug_agent_pos, aug_action
