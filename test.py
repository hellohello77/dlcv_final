import torch
import numpy as np
import pandas as pd
from tqdm import tqdm, trange
from os.path import join
import logging
from torch.utils.data import DataLoader
from torchvision import transforms
import sys
from torch.utils.data import DataLoader
from dataset import dataset
from multiprocessing import cpu_count
import argparse
import timm
import random
import torch.nn as nn
from model import Speech_Encoder, Predictor
import os
import matplotlib.pyplot as plt
from copy import deepcopy
import glob

logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=os.environ.get("LOGLEVEL", "INFO").upper(),
    stream=sys.stdout,
)
test_logger = logging.getLogger("DLCV Final Test")


def main(args):
    transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ]
    )
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # avail_pretrained_models = timm.list_models("*vit*", pretrained=True)
    image_encoder = timm.create_model("vit_base_patch16_224", pretrained=True).to(
        device
    )
    for param in image_encoder.parameters():
        param.requires_grad = False
    speech_encoder = torch.load(args.encoder_checkpoint).to(device)
    predictor = torch.load(args.predictor_checkpoint).to(device)
    test_set = dataset(transform=transform, task="test", hop=1)
    test_loader = DataLoader(
        test_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=cpu_count(),
        collate_fn=test_set.collate_fn,
    )
    task = "test"
    bbox_files = sorted(glob.glob(join(args.data_dir, task, "bbox", "*.csv")))
    seg_files = sorted(glob.glob(join(args.data_dir, task, "seg", "*.csv")))
    table = {f.split("/")[-1].replace("_seg.csv", ""): {} for f in seg_files}
    for key in table.keys():
        df = pd.read_csv(join(args.data_dir, task, "seg", f"{key}_seg.csv"))

        for i in range(len(df)):
            person_id, start_frame, end_frame = (
                df.loc[i, "person_id"],
                df.loc[i, "start_frame"],
                df.loc[i, "end_frame"],
            )
            if person_id not in table[key].keys():
                table[key][person_id] = {}
            table[key][person_id][(start_frame, end_frame)] = []
    speech_encoder.eval()
    predictor.eval()
    with torch.no_grad():
        for image, mfcc, mask_length, face, info in tqdm(test_loader):
            subsets, frames, person_ids = info

            image_feature = image_encoder.forward_features(image.to(device)).mean(
                axis=1
            )
            face_feature = image_encoder.forward_features(face.to(device)).mean(axis=1)
            mask = torch.ones(size=(mfcc.shape[0], mfcc.shape[1]))
            for i in range(len(mask)):
                for j in range(mask_length[i], len(mask[0])):
                    mask[i][j] = 0
            speech_feature = speech_encoder(mfcc.to(device), mask.to(device))
            feature = torch.cat(
                [
                    speech_feature,
                    (1 - args.face_alpha) * image_feature
                    + args.face_alpha * face_feature,
                ],
                dim=-1,
            )
            pred = predictor(feature).squeeze(-1)
            prediction = pred > 0.5

            for s, f, id, p in zip(subsets, frames, person_ids, prediction):
                for key in table[s][int(id)].keys():
                    if key[0] <= int(f) <= key[1]:
                        table[s][int(id)][key].append(p.to(int).item())
                        break
        import csv

        with open(args.output, "w") as file:
            writer = csv.writer(file)
            writer.writerow(["Id", "Predicted"])
            lines = []
            count, total = 0, 0
            for subset in table.keys():
                for person_id in table[subset].keys():
                    for key in table[subset][person_id].keys():
                        start_frame, end_frame = key
                        id = f"{subset}_{person_id}_{start_frame}_{end_frame}"
                        if len(table[subset][person_id][key]) > 0:
                            predicted = int(
                                np.round(np.mean(table[subset][person_id][key]))
                            )
                        else:
                            # Some bugs happen here, part of range did not being predicted
                            predicted = 0
                            count += 1
                        total += 1
                        lines.append([id, predicted])
            print(f"Miss rate {count * 100 / total} %")
            lines = sorted(lines)
            for l in lines:
                writer.writerow(l)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_dir", type=str, default="/home/stan/1000GB_Dir/DLCV_final/student_data"
    )
    parser.add_argument("--output", type=str, default="final.csv")
    # randomness
    parser.add_argument("--seed", type=int, default=9999)
    # save
    parser.add_argument("--encoder_checkpoint", type=str, default="face_encoder.pt")
    parser.add_argument("--predictor_checkpoint", type=str, default="face_predictor.pt")
    parser.add_argument("--batch_size", type=int, default=50)
    parser.add_argument("--face_alpha", type=float, default=0.7)
    args = parser.parse_args()
    main(args)
