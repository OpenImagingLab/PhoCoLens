# copy gt data to data/phlatcam/val/gts and data/phlatcam/train/gts
# the source data is from data/phlatcam/orig, and the train / val split is from data/phlatcam/text_files/train_target.txt and data/phlatcam/text_files/val_target.txt
import os
import random
import cv2
from tqdm import tqdm

def copy_gt_data(source_dir, target_dir, target_list):
    with open(target_list, 'r') as f:
        lines = f.readlines()
    for line in tqdm(lines):
        line = line.strip()
        source_path = os.path.join(source_dir, line)
        img_name = os.path.basename(line)
        target_path = os.path.join(target_dir, img_name.replace('.JPEG', '.png'))
        os.makedirs(os.path.dirname(target_path), exist_ok=True)
        print(f"Copy {source_path} to {target_path}")
        img = cv2.imread(source_path)
        cv2.imwrite(target_path, img)
# copy_gt_data('data/phlatcam', 'data/phlatcam/val/gts', 'data/phlatcam/text_files/val_target.txt')
# copy_gt_data('data/phlatcam', 'data/phlatcam/train/gts', 'data/phlatcam/text_files/train_target.txt')

# generate the train and val split for decoded simulated captures
def generate_train_val_split(target_train_list, target_val_list, train_list, val_list):
    with open(train_list, 'r') as f:
        train_lines = f.readlines()
    with open(val_list, 'r') as f:
        val_lines = f.readlines()
    # create the target train and val list
    with open(target_train_list, 'w') as f:
        for line in train_lines:
            img_name = line.split(os.sep)[-1].replace('.JPEG', '.png')
            target_line = "train/decoded_sim_captures/" + img_name
            f.write(target_line)
    with open(target_val_list, 'w') as f:
        for line in val_lines:
            img_name = line.split(os.sep)[-1].replace('.JPEG', '.png')
            target_line = "val/decoded_sim_captures/" + img_name
            f.write(target_line)

generate_train_val_split('data/phlatcam/text_files/decoded_sim_captures_train.txt', 'data/phlatcam/text_files/decoded_sim_captures_val.txt', 'data/phlatcam/text_files/train_target.txt', 'data/phlatcam/text_files/val_target.txt')