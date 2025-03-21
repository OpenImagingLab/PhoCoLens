# process svd output to dataset that can be used by NullSpaceDiff   
# first, move the output to a single folder
# second, move the original images to a single folder
# third, resize the images to 512x512

#find the directory of the process file
import os
import shutil
import numpy as np
import cv2
from tqdm import tqdm
tools_dir = os.path.dirname(os.path.abspath(__file__))
svd_dir = os.path.join(os.path.dirname(tools_dir), 'NullSpaceDiff')
nullspace_dir = os.path.join(os.path.dirname(tools_dir), 'SVDeconv')
dataset = "phlatcam"

exp_name = "fft-svd-1280-1408-meas-decoded_sim_spatial_weight"
output_name = exp_name
output_val_dir = os.path.join(nullspace_dir, "data/%s/%s/val"%(dataset,output_name))
output_train_dir = os.path.join(nullspace_dir, "data/%s/%s/train"%(dataset,output_name))
source_svd_train_dir = os.path.join(svd_dir, 'output/%s'%dataset, exp_name, "train")
source_svd_val_dir = os.path.join(svd_dir, 'output/%s'%dataset, exp_name, "val")
source_orig_dir = os.path.join(svd_dir, "data/%s/orig"%dataset)

index = 0
for cls in tqdm(os.listdir(source_svd_train_dir)):
    cls_dir = os.path.join(source_svd_train_dir, cls)
    if not os.path.isdir(cls_dir):
        continue
    for file in os.listdir(cls_dir):
        if file.endswith('png') and file.startswith('output_'):
            os.makedirs(os.path.join(output_train_dir, "inputs"), exist_ok=True)
            shutil.copy(os.path.join(cls_dir, file), os.path.join(os.path.join(output_train_dir, "inputs"), file[7:]))
            gt_file = os.path.join(source_orig_dir, cls, file[7:]).replace('png', 'JPEG')
            os.makedirs(os.path.join(output_train_dir, "gts"), exist_ok=True)
            shutil.copy(gt_file, os.path.join(os.path.join(output_train_dir, "gts"), file[7:]))

for cls in tqdm(os.listdir(source_svd_val_dir)):
    cls_dir = os.path.join(source_svd_val_dir, cls)
    if not os.path.isdir(cls_dir):
        continue
    for file in os.listdir(cls_dir):
        if file.endswith('png') and file.startswith('output_'):
            os.makedirs(os.path.join(output_val_dir, "inputs"), exist_ok=True)
            shutil.copy(os.path.join(cls_dir, file), os.path.join(os.path.join(output_val_dir, "inputs"), file[7:]))
            gt_file = os.path.join(source_orig_dir, cls, file[7:]).replace('png', 'JPEG')
            os.makedirs(os.path.join(output_val_dir, "gts"), exist_ok=True)
            shutil.copy(gt_file, os.path.join(os.path.join(output_val_dir, "gts"), file[7:]))
                        

# resize the images to 512x512, and save them to a new folder: inputs_512, gts_512
output_dirs = [output_train_dir, output_val_dir]
for output_dir in output_dirs:
    inputs = os.path.join(output_dir, 'inputs')
    gts = os.path.join(output_dir, 'gts')
    
    # create the new folders for resized images
    inputs_512 = os.path.join(output_dir, 'inputs_512')
    gts_512 = os.path.join(output_dir, 'gts_512')
    for dir in [inputs_512, gts_512]:
        if not os.path.exists(dir):
            os.makedirs(dir)
    for file in tqdm(os.listdir(inputs)):
        img = cv2.imread(os.path.join(inputs, file))
        img = cv2.resize(img, (512, 512))
        cv2.imwrite(os.path.join(inputs_512, file), img)
    for file in tqdm(os.listdir(gts)):
        img = cv2.imread(os.path.join(gts, file))
        img = cv2.resize(img, (512, 512))
        cv2.imwrite(os.path.join(gts_512, file), img)


