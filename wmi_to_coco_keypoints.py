import json
import os
import numpy as np

def convert_company_to_coco(input_file, output_file, image_dir):
    # Load the company JSON file
    with open(input_file, 'r') as f:
        data = json.load(f)

    # Transformation guidelines
    mapping = {
        0: 0,  # Head
        1: None,  # Empty
        2: None,  # Empty
        3: None,  # Empty
        4: None,  # Empty
        5: 6,    # Left shoulder
        6: 5,    # Right shoulder
        7: 4,    # Left elbow
        8: 3,    # Right elbow
        9: 1,    # Left wrist
        10: 2,   # Right wrist
        11: 10,  # Left hip
        12: 9,   # Right hip
        13: [13, 14],  # Left knee      
        14: [11, 12],  # Right knee
        15: [17, 18],  # Left ankle
        16: [15, 16],  # Right ankle
    }

    def process_keypoints(company_keypoints):
        """
        Transform company keypoints to COCO keypoints.
        """
        coco_keypoints = []

        for coco_idx in range(17):
            company_idx = mapping[coco_idx]
            if company_idx is None:
                # Empty point
                coco_keypoints.extend([0, 0, 0])
            elif isinstance(company_idx, list):
                # Combine multiple points
                points = [
                    np.array(company_keypoints[idx * 3:(idx + 1) * 3])
                    for idx in company_idx if company_keypoints[idx * 3:(idx + 1) * 3] != [0, 0, 0]
                ]
                if points:
                    avg_point = np.mean(points, axis=0)
                    coco_keypoints.extend(avg_point.tolist())
                else:
                    coco_keypoints.extend([0, 0, 0])
            else:
                # Direct mapping
                coco_keypoints.extend(company_keypoints[company_idx * 3:(company_idx + 1) * 3])
  
        return coco_keypoints

    # Filter out missing images
    valid_image_ids = set()
    valid_images = []

    for image in data["images"]:
        image_path = os.path.join(image_dir, image["file_name"])
        if os.path.exists(image_path):
            valid_images.append(image)
            valid_image_ids.add(image["id"])

    data["images"] = valid_images

    # Filter annotations based on valid images
    data["annotations"] = [
        annotation for annotation in data["annotations"] if annotation["image_id"] in valid_image_ids
    ]
    # Process annotations

    for annotation in data["annotations"]:
        company_keypoints = annotation["keypoints"]
        coco_keypoints = process_keypoints(company_keypoints)
        annotation["keypoints"] = coco_keypoints
        annotation["num_keypoints"] = sum(1 for i in range(0, len(coco_keypoints), 3) if coco_keypoints[i + 2] > 0)

    # Save the converted JSON file
    with open(output_file, 'w') as f:
        json.dump(data, f, indent=4)

# Example usage
convert_company_to_coco("/home/mohamed/repos/pose_estimation_task/coco/coco/annotations/person_keypoints_train2017_1.json", "coco_dataset.json","/home/mohamed/repos/pose_estimation_task/coco/coco/images/train2017")
