import os
from os.path import join
import sys
import json
import numpy as np
# from .read_openpose import read_openpose

def coco_extract(dataset_path, out_path):

    # convert joints to global order
    joints_idx = [19, 20, 21, 22, 23, 9, 8, 10, 7, 11, 6, 3, 2, 4, 1, 5, 0] # 17 indices, as coco has 17 points

    # bbox expansion factor
    scaleFactor = 1.2

    # structs we need
    imgnames_, scales_, centers_, parts_, openposes_ = [], [], [], [], []

    # json annotation file
    # json_path = os.path.join(dataset_path, 
    #                          'annotations', 
    #                          'person_keypoints_train2014.json')
    json_path = os.path.join(dataset_path)
    json_data = json.load(open(json_path, 'r'))

    imgs = {}
    for img in json_data['images']:
        imgs[img['id']] = img

    counter = 0
    for annot in json_data['annotations']:
        # keypoints processing
        keypoints = annot['keypoints']
        keypoints = np.reshape(keypoints, (17,3))
        keypoints[keypoints[:,2]>0,2] = 1
        # check if all major body joints are annotated
        if sum(keypoints[5:,2]>0) < 4:
            continue
        counter += 1
        # image name
        image_id = annot['image_id']
        img_name = str(imgs[image_id]['file_name'])
        img_name_full = join('train2014', img_name)
        # keypoints
        part = np.zeros([24,3])
        part[joints_idx] = keypoints# keypoints has 17 points, so the rest of the part points are zero, as part has 24 points
        # scale and center
        bbox = annot['bbox']
        center = [bbox[0] + bbox[2]/2, bbox[1] + bbox[3]/2]
        scale = scaleFactor*max(bbox[2], bbox[3])/200
        # read openpose detections
        # json_file = os.path.join(openpose_path, 'coco',
        #     img_name.replace('.jpg', '_keypoints.json'))
        # openpose = read_openpose(json_file, part, 'coco')
        openpose = np.zeros([25,3])
        
        # store data
        imgnames_.append(img_name_full)
        centers_.append(center)
        scales_.append(scale)
        parts_.append(part)
        openposes_.append(openpose)
        if counter %15 ==0:
            break

    # store the data struct
    if not os.path.isdir(out_path):
        os.makedirs(out_path)
    out_file = os.path.join(out_path, 'coco_2014_train.npz')
    np.savez(out_file, imgname=imgnames_,
                       center=centers_,
                       scale=scales_,
                       part=parts_,
                       openpose=openposes_)
    print(f'Extracted {counter} images')
    

def main():
    coco_extract('/home/mohamed/repos/pose_estimation_task/SPIN/coco_2014/coco_dataset_copy.json', '/home/mohamed/repos/pose_estimation_task/SPIN/coco_2014/')

if __name__ == '__main__':
    main()
