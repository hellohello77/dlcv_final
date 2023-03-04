import glob
import os
from os.path import join
import sys
# data_dir = "/home/stan/1000GB_Dir/DLCV_final/student_data"
data_dir = sys.argv[1]
All_dir = sorted(glob.glob(join(data_dir, "mfcc", "train", "*")))
subset = [d.split("/")[-1] for d in All_dir]
valid_set = subset[-94:]
task = "train"
images_dir = join(data_dir, "video_frames_files", task)
mfcc_dir = join(data_dir, "mfcc", task)
face_dir = join(data_dir, "face", task)
bbox_dir = join(data_dir, task, "bbox")
seg_dir = join(data_dir, task, "seg")
task = "valid"
valid_images_dir = join(data_dir, "video_frames_files", task)
valid_mfcc_dir = join(data_dir, "mfcc", task)
valid_face_dir = join(data_dir, "face", task)
valid_bbox_dir = join(data_dir, task, "bbox")
valid_seg_dir = join(data_dir, task, "seg")
# print(subset)
os.makedirs(join(data_dir, "video_frames_files", "valid"), exist_ok=True)
os.makedirs(join(data_dir, "mfcc", "valid"), exist_ok=True)
os.makedirs(join(data_dir, "face", "valid"), exist_ok=True)
os.makedirs(join(data_dir, "valid", "bbox"), exist_ok=True)
os.makedirs(join(data_dir, "valid", "seg"), exist_ok=True)

for Set in valid_set:
    os.system(f"mv {join(images_dir, Set)} {valid_images_dir}")
    os.system(f"mv {join(mfcc_dir, Set)} {valid_mfcc_dir}")
    os.system(f"mv {join(face_dir, Set)} {valid_face_dir}")
    bbox = Set + "_bbox.csv"
    seg = Set + "_seg.csv"
    os.system(f"mv {join(bbox_dir, bbox)} {valid_bbox_dir}")
    os.system(f"mv {join(seg_dir, seg)} {valid_seg_dir}")
    # exit(0)
