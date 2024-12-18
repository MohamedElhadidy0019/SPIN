import json
import matplotlib.pyplot as plt
from PIL import Image

# Load the COCO annotations file
with open("/home/mohamed/repos/pose_estimation_task/SPIN/coco_2014/annotations/person_keypoints_train2014.json", "r") as f:
    coco_data = json.load(f)

# Function to count valid keypoints
def count_valid_keypoints(keypoints):
    valid_count = 0
    for i in range(0, len(keypoints), 3):
        x, y, v = keypoints[i], keypoints[i + 1], keypoints[i + 2]
        if x != 0 or y != 0 or v != 0:  # A valid keypoint
            valid_count += 1
    return valid_count

# Find the first annotation with at least 10 valid keypoints
annotations = coco_data["annotations"]
first_valid_annotation = next(
    ann for ann in annotations if count_valid_keypoints(ann["keypoints"]) >= 10
)

# Extract the image information
image_id = first_valid_annotation["image_id"]
image_info = next(img for img in coco_data["images"] if img["id"] == image_id)

# Load the image
image_path = image_info["file_name"]  # Ensure the images are in the correct path
image = Image.open("/home/mohamed/repos/pose_estimation_task/SPIN/coco_2014/train2014/"+image_path)

# Extract and filter keypoints
keypoints = first_valid_annotation["keypoints"]
# loop and print xyz
for i in range(0, len(keypoints), 3):
    print(f'i: {i/3}  x: {keypoints[i]}, y: {keypoints[i + 1]}, z: {keypoints[i + 2]}')
keypoints_xy = [(keypoints[i], keypoints[i + 1]) for i in range(0, len(keypoints), 3)]

# Plot the image
plt.figure(figsize=(10, 10))
plt.imshow(image)
plt.axis("off")

# Plot valid keypoints with numbering
for i, (x, y) in enumerate(keypoints_xy):
    if x > 0 and y > 0:  # Only plot valid keypoints
        plt.plot(x, y, "ro")  # Red dot for keypoint
        plt.text(x + 2, y, str(i), color="yellow", fontsize=12, bbox=dict(facecolor="black", alpha=0.5))

plt.title(f"Image ID: {image_id} with Keypoints")
plt.show()
