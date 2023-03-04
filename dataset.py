import pandas as pd
from os.path import join
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
import json
import os
import logging
import sys
import json
import glob
from torchvision.io import read_image
from PIL import Image
import numpy as np
import torch
from tqdm import tqdm
from torchvision import transforms
from PIL import ImageFile
from os.path import join
import logging
from torch.nn.utils.rnn import pad_sequence

logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=os.environ.get("LOGLEVEL", "INFO").upper(),
    stream=sys.stdout,
)
logger = logging.getLogger("DLCV Final DataSet")


class dataset(Dataset):
    def __init__(
        self,
        data_dir="/home/stan/1000GB_Dir/DLCV_final/student_data",
        task="train",
        transform=None,
        mfcc_max_length=1000,
        hop=3,
        only_face=False,
    ):
        self.data_dir = data_dir
        self.task = task
        self.transform = transform
        self.mfcc_max_length = mfcc_max_length
        self.bbox_files = sorted(glob.glob(join(self.data_dir, task, "bbox", "*.csv")))
        self.seg_files = sorted(glob.glob(join(self.data_dir, task, "seg", "*.csv")))
        self.hop = hop
        self.only_face = only_face
        self.images_dir = join(self.data_dir, "video_frames_files", task)
        self.mfcc_dir = join(self.data_dir, "mfcc", task)
        self.face_dir = join(self.data_dir, "face", task)

        if only_face:
            self.images_files = []
            with open(f"{self.task}_files.txt", "r") as f:
                for line in f:
                    self.images_files.append(join(self.images_dir, line.strip()))
        else:
            self.images_files = sorted(glob.glob(join(self.images_dir, "*", "*.jpg")))[
                :: self.hop
            ]
        self.mfcc_files = sorted(glob.glob(join(self.mfcc_dir, "*", "*.pt")))
        # for f in tqdm(self.subset.keys(), desc="collecting files"):
        #     if (
        #         os.path.isdir(join(self.images_dir, f))
        #         and os.path.isdir(join(self.mfcc_dir, f))
        #         and os.path.isdir(join(self.face_dir, f))
        #     ):
        #         images_files = sorted(glob.glob(join(self.images_dir, f, "*.jpg")))
        #         self.images_files.extend(images_files[:: self.hop])
        #         # self.face_files.extend(glob.glob(join(self.face_dir, f, "*.jpg")))
        #         self.mfcc_files.extend(glob.glob(join(self.mfcc_dir, f, "*.pt")))
        #     else:
        #         logging.info(f"Error : {join(self.images_dir, f)} not exists !!")
        # self.images_files = sorted(self.images_files)
        # self.mfcc_files = sorted(self.mfcc_files)

        self.image2mfcc = {}
        for f in tqdm(self.mfcc_files, desc="Mapping files"):
            subset_key = f.split("/")[-2]
            filename = f.split("/")[-1].replace(".pt", "")
            start_frame, end_frame = int(filename.split("_")[1]), int(
                filename.split("_")[2]
            )
            for frame in range(start_frame, end_frame):
                self.image2mfcc[join(self.images_dir, subset_key, f"{frame}.jpg")] = f

        if self.task != "test":
            self.mfcc2label = {}
            for f in self.mfcc_files:
                self.mfcc2label[f] = int(
                    f.split("/")[-1].replace(".pt", "").split("_")[-1]
                )

    def __len__(self):
        return len(self.images_files)

    def __getitem__(self, index):
        if self.task != "test":
            images_file = self.images_files[index]
            mfcc_file = self.image2mfcc[images_file]
            label = self.mfcc2label[mfcc_file]
            return images_file, mfcc_file, label
        else:
            images_file = self.images_files[index]
            mfcc_file = self.image2mfcc[images_file]
            return images_file, mfcc_file

    def collate_fn(self, data):

        if self.task != "test":

            # Issue
            # Need to padding the mfcc data
            images_file, mfcc_file, label = zip(*data)
            image, mfcc_data, faces, mask_length = [], [], [], []
            for i in range(len(images_file)):
                if self.only_face:
                    image.append(
                        torch.zeros(
                            size=(
                                3,
                                self.transform.transforms[0].size[0],
                                self.transform.transforms[0].size[1],
                            )
                        )
                    )
                else:
                    image.append(self.transform(Image.open(images_file[i])))
                mfcc = torch.load(mfcc_file[i]).mean(dim=0).squeeze(0).transpose(1, 0)
                mfcc_data.append(mfcc[: self.mfcc_max_length, :])
                mask_length.append(min(len(mfcc), self.mfcc_max_length))
                # for face in faces_file[i]:
                #     faces[i].append(self.transform(Image.open(face)))
                subset = images_file[i].split("/")[-2]
                frame = images_file[i].split("/")[-1].replace(".jpg", "")
                person_id = mfcc_file[i].split("/")[-1].split("_")[0]
                if os.path.exists(
                    join(self.face_dir, subset, f"{person_id}_{frame}.jpg")
                ):
                    faces.append(
                        self.transform(
                            Image.open(
                                join(self.face_dir, subset, f"{person_id}_{frame}.jpg")
                            )
                        )
                    )
                else:
                    faces.append(
                        torch.zeros(
                            size=(
                                3,
                                self.transform.transforms[0].size[0],
                                self.transform.transforms[0].size[1],
                            )
                        )
                    )
            return (
                torch.stack(image),
                pad_sequence(mfcc_data, batch_first=True),
                mask_length,
                torch.stack(faces),
                torch.tensor(label),
            )
        else:
            images_file, mfcc_file = zip(*data)
            image, mfcc_data, faces, mask_length = [], [], [], []
            subsets, frames, person_ids = [], [], []
            for i in range(len(images_file)):
                if self.only_face:
                    image.append(
                        torch.zeros(
                            size=(
                                3,
                                self.transform.transforms[0].size[0],
                                self.transform.transforms[0].size[1],
                            )
                        )
                    )
                else:
                    image.append(self.transform(Image.open(images_file[i])))
                mfcc = torch.load(mfcc_file[i]).mean(dim=0).squeeze(0).transpose(1, 0)
                mfcc_data.append(mfcc[: self.mfcc_max_length, :])
                mask_length.append(min(len(mfcc), self.mfcc_max_length))
                subset = images_file[i].split("/")[-2]
                frame = images_file[i].split("/")[-1].replace(".jpg", "")
                person_id = mfcc_file[i].split("/")[-1].split("_")[0]
                subsets.append(subset)
                frames.append(frame)
                person_ids.append(person_id)
                if os.path.exists(
                    join(self.face_dir, subset, f"{person_id}_{frame}.jpg")
                ):
                    faces.append(
                        self.transform(
                            Image.open(
                                join(self.face_dir, subset, f"{person_id}_{frame}.jpg")
                            )
                        )
                    )
                else:
                    faces.append(
                        torch.zeros(
                            size=(
                                3,
                                self.transform.transforms[0].size[0],
                                self.transform.transforms[0].size[1],
                            )
                        )
                    )
            return (
                torch.stack(image),
                pad_sequence(mfcc_data, batch_first=True),
                mask_length,
                torch.stack(faces),
                (subsets, frames, person_ids),
            )


if __name__ == "__main__":
    from multiprocessing import cpu_count

    transform = transforms.Compose(
        [
            transforms.Resize((128, 128)),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ]
    )
    train_set = dataset(transform=transform)

    train_loader = DataLoader(
        train_set,
        batch_size=16,
        shuffle=False,
        num_workers=cpu_count(),
        collate_fn=train_set.collate_fn,
    )

    for image, mfcc, face, label in tqdm(train_loader):
        print(image.shape)
        print(np.shape(mfcc))
        print(mfcc[0].shape)
        print(len(face))
        print(face[0].shape)
        print(label)
        exit(0)
