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

logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=os.environ.get("LOGLEVEL", "INFO").upper(),
    stream=sys.stdout,
)
train_logger = logging.getLogger("DLCV Final Train")
valid_logger = logging.getLogger("DLCV Final Valid")


def normalize(x):
    return (x - x.min()) / (x.max() - x.min())

def evaluate(args):
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
    speech_encoder = torch.load(args.speech_checkpoint).to(device)
    predictor = torch.load(args.predictor_checkpoint).to(device)

    valid_set = dataset(transform=transform, task="valid")
    valid_loader = DataLoader(
        valid_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=cpu_count(),
        collate_fn=valid_set.collate_fn,
    )
    criterion = nn.BCELoss()
    with torch.no_grad():
        # speech_encoder.eval()
        # predictor.eval()
        valid_loss = []
        valid_prediction, valid_label = [], []
        pbar = tqdm(valid_loader)
        for image, mfcc, mask_length, face, label in pbar:
            # We need a embedding to convert mfcc from 40 to 768
            with torch.no_grad():
                image_feature = image_encoder.forward_features(
                    image.to(device)
                ).mean(axis=1)
                face_feature = image_encoder.forward_features(face.to(device)).mean(
                    axis=1
                )
            mask = torch.ones(size=(mfcc.shape[0], mfcc.shape[1]))
            for i in range(len(mask)):
                for j in range(mask_length[i], len(mask[0])):
                    mask[i][j] = 0
            speech_feature = speech_encoder(mfcc.to(device), mask.to(device))
            feature = torch.cat(
                [
                    args.speech_alpha * normalize(speech_feature),
                    normalize(
                        (1 - args.face_alpha) * image_feature
                        + args.face_alpha * face_feature
                    ),
                ],
                dim=-1,
            )
            pred = predictor(feature).squeeze(-1)

            loss = criterion(pred, label.to(device).to(torch.float))

            prediction = pred > 0.5
            acc = (
                torch.sum(prediction == label.to(device)).item()
                / len(prediction)
                * 100
            )
            pbar.set_description(
                f"Valid: Loss : {loss.item():.4f}, Acc : {acc:.2f}%"
            )
            pbar.refresh()

            valid_loss.append(loss.item())
            valid_label.extend(label.tolist())
            valid_prediction.extend(prediction.tolist())
        Acc = np.mean(np.array(valid_label) == np.array(valid_prediction)) * 100
        valid_logger.info(
            f"Valid Loss: {np.mean(valid_loss):.4f} Acc : {Acc:.2f}%"
        )
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
    encoder_layer = nn.TransformerEncoderLayer(
        d_model=40,
        nhead=4,
        batch_first=True,
    )
    speech_encoder = Speech_Encoder(
        encoder=nn.TransformerEncoder(encoder_layer, num_layers=6)
    ).to(device)
    predictor = Predictor(768 * 2).to(device)

    train_set = dataset(transform=transform)
    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=cpu_count(),
        collate_fn=train_set.collate_fn,
    )
    valid_set = dataset(transform=transform, task="valid")
    valid_loader = DataLoader(
        valid_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=cpu_count(),
        collate_fn=valid_set.collate_fn,
    )
    # test_set = dataset(transform=transform, task='test')
    # test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False, num_workers=cpu_count(), collate_fn=train_set.collate_fn)

    optimizer = getattr(torch.optim, args.optimizer)(
        list(speech_encoder.parameters()) + list(predictor.parameters()),
        **args.optim_param[args.optimizer],
    )
    criterion = nn.BCELoss()
    Train_Loss, Train_ACC, Valid_Loss, Valid_ACC = [], [], [], []
    Best_acc = 0
    Best_encoder = speech_encoder
    Best_predictor = predictor
    for epoch in trange(args.epochs):
        train_loss = []
        train_prediction, train_label = [], []
        pbar = tqdm(train_loader)
        speech_encoder.train()
        predictor.train()
        for image, mfcc, mask_length, face, label in pbar:
            # We need a embedding to convert mfcc from 40 to 768
            with torch.no_grad():
                image_feature = image_encoder.forward_features(image.to(device)).mean(
                    axis=1
                )
                face_feature = image_encoder.forward_features(face.to(device)).mean(
                    axis=1
                )
            mask = torch.ones(size=(mfcc.shape[0], mfcc.shape[1]))
            for i in range(len(mask)):
                for j in range(mask_length[i], len(mask[0])):
                    mask[i][j] = 0
            speech_feature = speech_encoder(mfcc.to(device), mask.to(device))
            feature = torch.cat(
                [
                    args.speech_alpha * normalize(speech_feature),
                    normalize(
                        (1 - args.face_alpha) * image_feature
                        + args.face_alpha * face_feature
                    ),
                ],
                dim=-1,
            )
            pred = predictor(feature).squeeze(-1)

            loss = criterion(pred, label.to(device).to(torch.float))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            prediction = pred > 0.5
            acc = (
                torch.sum(prediction == label.to(device)).item() / len(prediction) * 100
            )
            pbar.set_description(
                f"Epoch[{epoch}/{args.epochs}] Train: Loss : {loss.item():.4f}, Acc : {acc:.2f}%"
            )
            pbar.refresh()

            train_loss.append(loss.item())
            train_label.extend(label.tolist())
            train_prediction.extend(prediction.tolist())
            args.speech_alpha = min(
                args.speech_alpha_upper, args.speech_alpha * args.speech_coef
            )
        Acc = np.mean(np.array(train_label) == np.array(train_prediction)) * 100
        Train_ACC.append(Acc)
        Train_Loss.append(np.mean(train_loss))
        train_logger.info(
            f"Epoch[{epoch}/{args.epochs}], Train Loss: {np.mean(train_loss):.4f} Acc : {Acc:.2f}%"
        )
        with torch.no_grad():
            # speech_encoder.eval()
            # predictor.eval()
            valid_loss = []
            valid_prediction, valid_label = [], []
            pbar = tqdm(valid_loader)
            for image, mfcc, mask_length, face, label in pbar:
                # We need a embedding to convert mfcc from 40 to 768
                with torch.no_grad():
                    image_feature = image_encoder.forward_features(
                        image.to(device)
                    ).mean(axis=1)
                    face_feature = image_encoder.forward_features(face.to(device)).mean(
                        axis=1
                    )
                mask = torch.ones(size=(mfcc.shape[0], mfcc.shape[1]))
                for i in range(len(mask)):
                    for j in range(mask_length[i], len(mask[0])):
                        mask[i][j] = 0
                speech_feature = speech_encoder(mfcc.to(device), mask.to(device))
                feature = torch.cat(
                    [
                        args.speech_alpha * normalize(speech_feature),
                        normalize(
                            (1 - args.face_alpha) * image_feature
                            + args.face_alpha * face_feature
                        ),
                    ],
                    dim=-1,
                )
                pred = predictor(feature).squeeze(-1)

                loss = criterion(pred, label.to(device).to(torch.float))

                prediction = pred > 0.5
                acc = (
                    torch.sum(prediction == label.to(device)).item()
                    / len(prediction)
                    * 100
                )
                pbar.set_description(
                    f"Epoch[{epoch}/{args.epochs}] Valid: Loss : {loss.item():.4f}, Acc : {acc:.2f}%"
                )
                pbar.refresh()

                valid_loss.append(loss.item())
                valid_label.extend(label.tolist())
                valid_prediction.extend(prediction.tolist())
            Acc = np.mean(np.array(valid_label) == np.array(valid_prediction)) * 100
            Valid_ACC.append(Acc)
            Valid_Loss.append(np.mean(valid_loss))
            valid_logger.info(
                f"Epoch[{epoch}/{args.epochs}], Valid Loss: {np.mean(valid_loss):.4f} Acc : {Acc:.2f}%"
            )
        # image_encoder.forward_features

        if Best_acc < Acc:
            Best_acc = Acc
            Best_encoder = deepcopy(speech_encoder)
            Best_predictor = deepcopy(predictor)
            torch.save(Best_predictor, "no_teacher_face_predictor.pt")
            torch.save(Best_encoder, "no_teacher_face_encoder.pt")

        plt.plot(Train_Loss, label="train loss")
        plt.plot(Valid_Loss, label="valid loss")
        plt.xlabel("epoch")
        plt.ylabel("loss")
        plt.legend(loc="best")
        plt.title("Loss")
        plt.savefig("Loss.jpg")
        plt.clf()
        plt.plot(Train_ACC, label="train acc")
        plt.plot(Valid_ACC, label="valid acc")
        plt.legend(loc="best")
        plt.xlabel("epoch")
        plt.ylabel("Acc")
        plt.title("Acc")
        plt.savefig("Acc.jpg")
        plt.clf()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_dir", type=str, default="/home/stan/1000GB_Dir/DLCV_final/student_data"
    )
    parser.add_argument(
        "--speech_checkpoint", type=str, default="/home/stan/1000GB_Dir/DLCV_final/student_data"
    )
    # randomness
    parser.add_argument("--seed", type=int, default=9999)
    # save
    parser.add_argument("--save_path", type=str, default="p2_final_model.bin")
    parser.add_argument("--output", type=str, default="p2_output")
    parser.add_argument("--batch_size", type=int, default=50)
    parser.add_argument("--optimizer", type=str, default="Adam")
    parser.add_argument(
        "--optim_param",
        type=dict,
        default={
            "Adam": {"lr": 2e-6, "weight_decay": 3e-4, "betas": (0.9, 0.99)},
            "SGD": {"lr": 7e-5, "weight_decay": 5e-3, "momentum": 0.0},
        },
    )
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--face_alpha", type=float, default=0.7)
    parser.add_argument("--speech_alpha", type=float, default=1)
    parser.add_argument("--speech_alpha_upper", type=float, default=1)
    parser.add_argument("--speech_coef", type=float, default=1.0007)
    parser.add_argument("--eval", action="store_true", default=False)
    
    args = parser.parse_args()
    if args.eval:
        evaluate(args)
    else:
        main(args)
