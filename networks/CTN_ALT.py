import torch.nn as nn
import torch

import constants


class CTN_ALT(torch.nn.Module):
    """Alternate CNN + Transformer Architectureqq"""
    def __init__(self, model_config):
        super().__init__()
        self.conv = nn.Conv2d(1,
                              model_config["OUT_CHANNELS"],
                              kernel_size=3,
                              stride=1,
                              padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.transformer = nn.Transformer(
            d_model=model_config["D_MODEL"],
            nhead=model_config["N_HEADS"],
            num_encoder_layers=model_config["TRANSFORMER_LAYERS"],
            num_decoder_layers=model_config["TRANSFORMER_LAYERS"],
            batch_first=True)
        self.flatten = nn.Flatten()
        self.dropout = nn.Dropout(p=model_config["FC_DROPOUT"])
        self.fc1 = nn.Linear(model_config["HIDDEN_SIZE_1"],
                             model_config["HIDDEN_SIZE_2"])
        self.fc2 = nn.Linear(model_config["HIDDEN_SIZE_2"],
                             constants.DATA["NUM_CLASSES"])

    def forward(self, inp):
        x = inp.unsqueeze(1)  # Add channel dimension
        x = self.conv(x.float())
        x = self.pool(x)
        src = x[:, 0, :, :].float()
        tgt = x[:, 1, :, :].float()
        x = self.transformer(src, tgt)
        x = self.flatten(x)
        x = self.dropout(x)
        x = self.fc1(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x
