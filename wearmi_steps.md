
## Preprocessing wearmi dataset to make it suitable for training
1. run `wmi_to_coco_keypoints.py` to convert the dataset keypoints to COCO keypoints, you should edit in the script the path of the json file, and the path of the images folder
2. run `datasets/preprocess/coco_wearami.py` to generate the .npy file using that will be used for training, you should edit in the script the path of the json file, and the path of the images folder
3. step 2 should result in a .npz file that contains the keypoints and the images paths, this will be the model's input, copy this npz file from the dataset dircetory you specified in step 2 to the `datasets/preprocess` directory to `data/dataset_extras` , you should replace the existing file
4. run this command to start training `python train.py --name train_example --pretrained_checkpoint data/model_checkpoint.pt --run_smplify`
   


