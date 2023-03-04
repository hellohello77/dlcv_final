import cv2
import time
import os
import argparse
from moviepy.editor import VideoFileClip
from os.path import join
import pandas as pd
import uuid
import glob
from tqdm import trange, tqdm
import logging
import sys
import torchaudio
import csv
import torch
import numpy as np

logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=os.environ.get("LOGLEVEL", "INFO").upper(),
    stream=sys.stdout,
)
logger = logging.getLogger("DLCV Final Preprocessing")


def mp42wav(data_dir="student_data/videos", output_dir="student_data/wav_files"):
    # convert mp4 to wav

    files = sorted([os.path.join(data_dir, x) for x in os.listdir(data_dir)])
    os.makedirs(output_dir, exist_ok=True)
    for f in files:
        print(f)
        video = VideoFileClip(f)
        audio = video.audio
        print(str(f.split(".")[-2] + ".wav"))
        audio.write_audiofile(
            os.path.join(output_dir, str(f.split(".")[-2] + ".wav").split("/")[-1])
        )


def crop_face(data_dir="student_data", output_dir="student_data/face"):

    train_image_dir = join(data_dir, "video_frames_files", "train")
    test_image_dir = join(data_dir, "video_frames_files", "test")
    train_csv_dir = join(data_dir, "train", "bbox")
    test_csv_dir = join(data_dir, "test", "bbox")
    train_csv_files = sorted(glob.glob(join(train_csv_dir, "*.csv")))
    test_csv_files = sorted(glob.glob(join(test_csv_dir, "*.csv")))

    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(join(output_dir, "train"), exist_ok=True)
    pbar = tqdm(train_csv_files)
    for csv_files in pbar:

        df = pd.read_csv(csv_files)
        tmp_dir = join(
            train_image_dir, csv_files.split("/")[-1].replace("_bbox.csv", "")
        )
        images = glob.glob(join(tmp_dir, "*.jpg"))
        os.makedirs(
            join(
                output_dir, "train", csv_files.split("/")[-1].replace("_bbox.csv", "")
            ),
            exist_ok=True,
        )
        pbar.set_description(csv_files.split("/")[-1].replace("_bbox.csv", ""))
        pbar.refresh()
        for image in tqdm(images):
            im = cv2.imread(image)
            frame = int(image.split("/")[-1].replace(".jpg", ""))
            for person in np.unique(df["person_id"]):
                person_df = df[df["person_id"] == person]
                location = person_df[person_df["frame_id"] == frame]
                x1, y1, x2, y2 = (
                    location["x1"].values,
                    location["y1"].values,
                    location["x2"].values,
                    location["y2"].values,
                )
                face = im[int(y1) : int(y2), int(x1) : int(x2), :]
                if face.size != 0:
                    cv2.imwrite(
                        join(
                            output_dir,
                            "train",
                            csv_files.split("/")[-1].replace("_bbox.csv", ""),
                            f"{person}_{frame}.jpg",
                        ),
                        face,
                    )
    os.makedirs(join(output_dir, "test"), exist_ok=True)
    for csv_files in tqdm(test_csv_files):

        df = pd.read_csv(csv_files)
        tmp_dir = join(
            test_image_dir, csv_files.split("/")[-1].replace("_bbox.csv", "")
        )
        images = glob.glob(join(tmp_dir, "*.jpg"))
        os.makedirs(
            join(output_dir, "test", csv_files.split("/")[-1].replace("_bbox.csv", "")),
            exist_ok=True,
        )
        for image in tqdm(images):

            im = cv2.imread(image)
            frame = int(image.split("/")[-1].replace(".jpg", ""))
            for person in np.unique(df["person_id"]):
                person_df = df[df["person_id"] == person]
                location = person_df[person_df["frame_id"] == frame]
                x1, y1, x2, y2 = (
                    location["x1"].values,
                    location["y1"].values,
                    location["x2"].values,
                    location["y2"].values,
                )
                face = im[int(y1) : int(y2), int(x1) : int(x2), :]
                if face.size != 0:
                    cv2.imwrite(
                        join(
                            output_dir,
                            "test",
                            csv_files.split("/")[-1].replace("_bbox.csv", ""),
                            f"{person}_{frame}.jpg",
                        ),
                        face,
                    )


def video_to_frames(data_dir, output_dir):
    """Function to extract frames from input video file
    and save them as separate frames in an output directory.
    Args:
        input_loc: Input video file.
        output_loc: Output directory to save the frames.
    Returns:
        None
    """
    train_seg_dir = join(data_dir, "train", "seg")
    test_seg_dir = join(data_dir, "test", "seg")
    train__bbox_dir = join(data_dir, "train", "bbox")
    test_bbox_dir = join(data_dir, "test", "bbox")
    video_dir = join(data_dir, "videos")
    train_data = glob.glob(join(train_seg_dir, "*.csv"))
    test_data = glob.glob(join(test_seg_dir, "*.csv"))
    train_csv = {}
    os.makedirs(output_dir, exist_ok=True)
    for data in train_data:

        tmp = data.split("/")[-1].replace("_seg.csv", "")
        if os.path.isdir(join(output_dir, "train", tmp)):
            path = join(output_dir, "train", tmp)
            logger.info(f"{path} already exists")
            continue
        seg_df = pd.read_csv(data)
        # bbox_df = pd.read_csv(join(train__bbox_dir, tmp + "_bbox.csv"))
        video = cv2.VideoCapture(join(video_dir, tmp + ".mp4"))
        os.makedirs(join(output_dir, "train", tmp), exist_ok=True)
        All_frames = []
        for i in trange(
            int(video.get(cv2.CAP_PROP_FRAME_COUNT)), desc=f"Loading {tmp} ... "
        ):
            if video.isOpened():
                ret, frame = video.read()
                if ret == True:
                    All_frames.append(frame)

        collect_frame_idx = [False for i in range(len(All_frames))]
        for i in range(len(seg_df)):
            idx = list(range(seg_df.loc[i, "start_frame"], seg_df.loc[i, "end_frame"]))
            for j in idx:
                collect_frame_idx[j] = True
        for i in range(len(All_frames)):
            if collect_frame_idx[i]:

                cv2.imwrite(join(output_dir, "train", tmp, f"{i}.jpg"), All_frames[i])
        path = join(output_dir, "train", tmp)
        logger.info(f"{path} Done")
    for data in test_data:

        tmp = data.split("/")[-1].replace("_seg.csv", "")
        seg_df = pd.read_csv(data)
        # bbox_df = pd.read_csv(join(train__bbox_dir, tmp + "_bbox.csv"))
        video = cv2.VideoCapture(join(video_dir, tmp + ".mp4"))
        os.makedirs(join(output_dir, "test", tmp), exist_ok=True)
        All_frames = []
        for i in trange(int(video.get(cv2.CAP_PROP_FRAME_COUNT))):
            if video.isOpened():
                ret, frame = video.read()
                if ret == True:
                    All_frames.append(frame)

        collect_frame_idx = [False for i in range(len(All_frames))]
        for i in range(len(seg_df)):
            idx = list(range(seg_df.loc[i, "start_frame"], seg_df.loc[i, "end_frame"]))
            for j in idx:
                collect_frame_idx[j] = True
        for i in range(len(All_frames)):
            if collect_frame_idx[i]:

                cv2.imwrite(join(output_dir, "test", tmp, f"{i}.jpg"), All_frames[i])
        path = join(output_dir, "test", tmp)
        logger.info(f"{path} Done")


def wav_segmentation_mfcc(data_dir, output_dir="mfcc"):
    wav_dir = join(data_dir, "wav_files")
    train_csv_dir = join(data_dir, "train", "seg")
    test_csv_dir = join(data_dir, "test", "seg")
    # wav_files = sorted([os.path.join(wav_dir, x) for x in os.listdir(wav_dir)])
    train_csv_files = sorted(
        [os.path.join(train_csv_dir, x) for x in os.listdir(train_csv_dir)]
    )
    test_csv_files = sorted(
        [os.path.join(test_csv_dir, x) for x in os.listdir(test_csv_dir)]
    )
    train_out_dir = join(output_dir, "train")
    test_out_dir = join(output_dir, "test")
    os.makedirs(train_out_dir, exist_ok=True)
    sample_rate = 16000
    mfcc_transform = torchaudio.transforms.MFCC(sample_rate=sample_rate)
    for i in trange(len(train_csv_files), desc="Train files"):
        wav_file = os.path.join(
            wav_dir, train_csv_files[i].split("/")[-1].split("_")[-2] + ".wav"
        )
        # logging.info(f"Processing {train_csv_files[i]} ...")
        ori_audio, ori_samople_rate = torchaudio.load(wav_file)
        # print(ori_audio, ori_samople_rate)
        transform = torchaudio.transforms.Resample(ori_samople_rate, sample_rate)
        audio = transform(ori_audio)

        # print(audio.shape)
        id = []
        start_frame = []
        end_frame = []
        ttm = []
        with open(train_csv_files[i], newline="") as csvfile:
            rows = csv.reader(csvfile)
            for row in rows:
                id.append(row[0])
                start_frame.append(row[1])
                end_frame.append(row[2])
                ttm.append(row[3])
        # print(start_frame)
        # for i in range(len(start_frame)):
        for j in range(1, len(start_frame)):
            onset = int(int(start_frame[j]) / 30 * sample_rate)
            offset = int(int(end_frame[j]) / 30 * sample_rate)
            if onset > audio.shape[1] or onset == offset:
                continue
            crop_audio = audio[:, onset:offset]
            crop_audio = crop_audio.view(2, -1)
            mfcc_data = mfcc_transform(crop_audio)
            # print("crop_audio",crop_audio)
            out = os.path.join(train_out_dir, wav_file.split("/")[-1].split(".")[-2])
            os.makedirs(out, exist_ok=True)
            out = os.path.join(
                out, f"{id[j]}_{start_frame[j]}_{end_frame[j]}_{ttm[j]}.pt"
            )
            torch.save(mfcc_data, out)
            # logging.info(f"Saving {out} ...")

    for i in trange(len(test_csv_files), desc="Testing files"):
        wav_file = os.path.join(
            wav_dir, test_csv_files[i].split("/")[-1].split("_")[-2] + ".wav"
        )
        # logging.info(f"Processing {test_csv_files[i]} ...")
        ori_audio, ori_samople_rate = torchaudio.load(wav_file)
        # print(ori_audio, ori_samople_rate)
        transform = torchaudio.transforms.Resample(ori_samople_rate, sample_rate)
        audio = transform(ori_audio)

        # print(audio.shape)
        id = []
        start_frame = []
        end_frame = []
        ttm = []
        with open(test_csv_files[i], newline="") as csvfile:
            rows = csv.reader(csvfile)
            for row in rows:
                id.append(row[0])
                start_frame.append(row[1])
                end_frame.append(row[2])
        # print(start_frame)
        # for i in range(len(start_frame)):
        for j in range(1, len(start_frame)):
            onset = int(int(start_frame[j]) / 30 * sample_rate)
            offset = int(int(end_frame[j]) / 30 * sample_rate)
            if onset > audio.shape[1] or onset == offset:
                continue
            crop_audio = audio[:, onset:offset]
            crop_audio = crop_audio.view(2, -1)
            mfcc_data = mfcc_transform(crop_audio)
            # print("crop_audio",crop_audio)
            out = os.path.join(test_out_dir, wav_file.split("/")[-1].split(".")[-2])
            os.makedirs(out, exist_ok=True)
            out = os.path.join(out, f"{id[j]}_{start_frame[j]}_{end_frame[j]}.pt")
            torch.save(mfcc_data, out)
            # logging.info(f"Saving {out} ...")


if __name__ == "__main__":
    # save all the video as frames
    # video_root = "student_data/videos"
    # # input_loc = "./student_data/videos/0b4cacb1-970f-4ef0-85da-371d81f899e0.mp4"
    # # output_loc = './student_data/frames/0b4cacb1-970f-4ef0-85da-371d81f899e0/'
    # files = sorted([os.path.join(video_root,x) for x in os.listdir(video_root)])
    # print("files",files[0])
    # print("out","./student_data/frames/"+str(files[0].split(".")[-2].split("/")[-1]))

    # for i in range(len(files)):

    #     print("processing ",i)
    #     out = "./student_data/frames/"+str(files[i].split(".")[-2].split("/")[-1])

    #     os.makedirs(out, exist_ok=True)

    #     # print("files",files)
    #     print("out",out)
    #     video_to_frames(files[i], out)
    # print("processing")

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_dir", type=str, default="/home/stan/1000GB_Dir/DLCV_final/student_data"
    )
    # parser.add_argument(
    #     "--output",
    #     type=str,
    #     default="/home/stan/1000GB_Dir/DLCV_final/student_data/frame",
    # )

    args = parser.parse_args()
    mp42wav(
        data_dir=join(args.data_dir, "videos"),
        output_dir=join(args.data_dir, "wav_files"),
    )
    wav_segmentation_mfcc(
        data_dir=args.data_dir,
        output_dir=join(args.data_dir, "mfcc"),
    )
    video_to_frames(
        data_dir=args.data_dir, output_dir=join(args.data_dir, "video_frames_files")
    )
    crop_face(data_dir=args.data_dir, output_dir=join(args.data_dir, "face"))
