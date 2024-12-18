import numpy as np
import torch
import smplx
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
MODEL='/home/mohamed/repos/pose_estimation_task/SPIN/basicModel_neutral_lbs_10_207_0_v1.0.0.pkl' # Change to your SMPL model path

def estimate_smpl_body(coco_keypoints, image_width, image_height):
    """
    Estimate SMPL body parameters from 2D keypoints
    
    Args:
        coco_keypoints (np.ndarray): 24 COCO keypoints with [x, y, confidence]
        image_width (int): Width of the input image
        image_height (int): Height of the input image
    
    Returns:
        dict: Estimated body parameters and vertices
    """
    # Device setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Normalize keypoints
    normalized_keypoints = coco_keypoints.copy()
    normalized_keypoints[:, 0] /= image_width
    normalized_keypoints[:, 1] /= image_height

    # Convert keypoints to PyTorch tensors
    keypoints_2d = torch.tensor(normalized_keypoints[:, :2], dtype=torch.float32).unsqueeze(0)
    confidences = torch.tensor(normalized_keypoints[:, 2], dtype=torch.float32).unsqueeze(0)

    # Load SMPL model
    smpl_model = smplx.SMPL(
        model_path=MODEL,
        gender='neutral',
        batch_size=1
    ).to(device)

    # Prepare keypoints
    keypoints = {
        'keypoints_2d': keypoints_2d.to(device),
        'confidences': confidences.to(device)
    }

    # Initialize and run optimization
    try:
        from smplify import SMPLify
        smplify = SMPLify(batch_size=1)
        
        output = smplify(
            keypoints=keypoints,
            init_pose=None,
            init_shape=None
        )

        # Extract body parameters
        pred_pose = output['pred_pose']
        pred_shape = output['pred_shape']

        # Generate body mesh
        body_model = smpl_model(
            betas=pred_shape,
            body_pose=pred_pose[:, 1:],
            global_orient=pred_pose[:, :1]
        )
        vertices = body_model.vertices.detach().cpu().numpy()

        return {
            'vertices': vertices,
            'pose': pred_pose,
            'shape': pred_shape
        }

    except Exception as e:
        print(f"SMPLify optimization error: {e}")
        return None

def visualize_smpl_body(vertices):
    """
    Visualize the SMPL body mesh in 3D
    
    Args:
        vertices (np.ndarray): Vertices of the SMPL body mesh
    """
    # Specific SMPL model triangle face connections 
    # (You might need to replace this with actual SMPL mesh faces)
    from smplx import SMPL
    smpl = SMPL(model_path=MODEL)
    faces = smpl.faces

    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot the mesh
    ax.plot_trisurf(
        vertices[0, :, 0], 
        vertices[0, :, 1], 
        vertices[0, :, 2], 
        triangles=faces, 
        color='lightblue', 
        alpha=0.7, 
        edgecolor='gray'
    )
    
    # Set labels and title
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('Reconstructed SMPL Body Mesh')
    
    plt.tight_layout()
    plt.show()

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


image_width, image_height = 640, 480

# Estimate body
body_estimation = estimate_smpl_body(coco_keypoints, image_width, image_height)

if body_estimation:
    # Visualize the body
    visualize_smpl_body(body_estimation['vertices'])
else:
    print("Body estimation failed.")