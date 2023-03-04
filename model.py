import torch
import torch.nn as nn


class Speech_Encoder(nn.Module):
    def __init__(
        self,
        encoder,
    ):
        super(Speech_Encoder, self).__init__()
        self.encoder = encoder
        self.project = nn.Linear(
            40, 768
        )  # 40 is the dim of mfcc, 768 is the dim of image features

    def forward(self, x, mask):

        x = self.encoder(x, src_key_padding_mask=mask)
        x = self.project(x)

        return x.mean(axis=1)


class Predictor(nn.Module):
    def __init__(
        self,
        input_dim,
    ):
        super(Predictor, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 100),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(100, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):

        return self.model(x)
