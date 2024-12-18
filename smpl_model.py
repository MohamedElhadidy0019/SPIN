import numpy as np
import torch
from smplx import SMPL
from smplify import SMPLify

# Example: Replace this with your actual 24 COCO keypoints (x, y, confidence)
coco_keypoints = np.array([
    [200, 300, 1.0],  # Nose
    [220, 320, 1.0],  # Left eye
    [180, 320, 1.0],  # Right eye
    [230, 400, 1.0],  # Left ear
    [170, 400, 1.0],  # Right ear
    [300, 500, 1.0],  # Left shoulder
    [100, 500, 1.0],  # Right shoulder
    [320, 700, 1.0],  # Left elbow
    [80, 700, 1.0],   # Right elbow
    [350, 900, 1.0],  # Left wrist
    [50, 900, 1.0],   # Right wrist
    [250, 1100, 1.0], # Left hip
    [150, 1100, 1.0], # Right hip
    [280, 1400, 1.0], # Left knee
    [120, 1400, 1.0], # Right knee
    [300, 1700, 1.0], # Left ankle
    [100, 1700, 1.0], # Right ankle
    [250, 400, 1.0],  # Left big toe
    [200, 400, 1.0],  # Right big toe
    [240, 200, 1.0],  # Left small toe
    [210, 200, 1.0],  # Right small toe
    [260, 300, 1.0],  # Left heel
    [190, 300, 1.0],  # Right heel
    [240, 360, 1.0],  # Neck
])

# Image dimensions (width and height)
image_width, image_height = 640, 480

# Normalize keypoints (x and y values)
coco_keypoints[:, 0] /= image_width  # Normalize x-coordinates
coco_keypoints[:, 1] /= image_height  # Normalize y-coordinates

# Convert keypoints to PyTorch tensors
keypoints_2d = torch.tensor(coco_keypoints[:, :2], dtype=torch.float32).unsqueeze(0)  # Shape: (1, 24, 2)
confidences = torch.tensor(coco_keypoints[:, 2], dtype=torch.float32).unsqueeze(0)  # Shape: (1, 24)

# Device setup (use GPU if available)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
keypoints_2d = keypoints_2d.to(device)
confidences = confidences.to(device)

# Load the SMPL model
smpl_model = SMPL(
    model_path='/home/mohamed/repos/pose_estimation_task/SPIN/basicModel_neutral_lbs_10_207_0_v1.0.0.pkl',  # Update the path
    gender='neutral',
    batch_size=1
).to(device)

# Initialize SMPLify
smplify = SMPLify(
    batch_size=1,
    joints_2d_conf_thresh=1e-3,  # Confidence threshold for 2D keypoints
    use_cuda=torch.cuda.is_available()
)

# Prepare keypoints dictionary
keypoints = {
    'keypoints_2d': keypoints_2d,
    'confidences': confidences
}

# Run SMPLify optimization
output = smplify(
    keypoints=keypoints,
    init_pose=None,  # Optionally provide an initial pose
    init_shape=None
)

# Extract optimized body parameters
pred_pose = output['pred_pose']  # Pose parameters
pred_shape = output['pred_shape']  # Shape parameters

# Generate the body mesh
vertices = smpl_model(
    betas=pred_shape,
    body_pose=pred_pose[:, 1:],
    global_orient=pred_pose[:, :1]
).vertices.detach().cpu().numpy()

# Print or save the vertices for visualization
print("Generated SMPL Body Vertices:", vertices)

# Optionally, you can use a 3D viewer (like matplotlib or MeshLab) to visualize the body mesh.

    # model_path='/home/mohamed/repos/pose_estimation_task/SPIN/basicModel_neutral_lbs_10_207_0_v1.0.0.pkl',  # Change to your SMPL model path