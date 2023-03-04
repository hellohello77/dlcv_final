import os
from os.path import join

data_dir = "/home/stan/1000GB_Dir/DLCV_final/student_data/"

task = "valid"
subset = "de82a6f3-1a55-4996-ae68-35032678ff66"
frame_dir = join(data_dir, "video_frames_files", task, subset)
face_dir = join(data_dir, "face", task, subset)
frame_count = str(4521)

if os.path.exists(join(frame_dir, f"{frame_count}.jpg")):
    os.makedirs(frame_count, exist_ok=True)
    image_path = join(frame_dir, frame_count + ".jpg")
    os.system(f"cp {image_path} {frame_count}")
    import glob
    face_images = glob.glob(join(face_dir, f"*_{frame_count}.jpg"))
    for im in face_images:
        os.system(f"cp {im} {frame_count}")
else:
    print(frame_count, "not exists")