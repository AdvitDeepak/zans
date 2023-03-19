import torch
import torch.nn as nn

import constants


class CTN(torch.nn.Module):
    """
    CNN + Transformer Architecture
    """
    def __init__(self, model_config):
        super().__init__()

        # CNN
        self.conv1 = nn.Conv2d(in_channels=constants.DATA['NUM_ELECTRODES'], out_channels=model_config["OUT_CHANNELS"], kernel_size=(1,10), stride=1, padding='same')
        self.elu1 = nn.ELU(alpha=1.0)
        self.maxpool1 = nn.MaxPool2d(kernel_size=(1,3))
        self.batchnorm1 = nn.BatchNorm2d(num_features=model_config["OUT_CHANNELS"], eps=0.001, momentum=0.99, affine=True)
        self.dropout1 = nn.Dropout(model_config["C_DROPOUT"])

        # TRANSFORMER
        self.transformer = nn.Transformer(
            d_model=model_config["D_MODEL"],
            nhead=model_config["N_HEADS"],
            num_encoder_layers=model_config['TRANSFORMER_LAYERS'],
            num_decoder_layers=model_config['TRANSFORMER_LAYERS'],
            dropout=model_config["T_DROPOUT"],
            batch_first=True)
        self.flatten = nn.Flatten()

        # Final Linear Layer
        self.fc = nn.Linear(model_config["HIDDEN_SIZE"], constants.DATA["NUM_CLASSES"])

    def forward(self, inp, src_mask=None):
        x = self.elu1(self.conv1(inp.float()))
        x = self.maxpool1(x)
        x = self.dropout1(self.batchnorm1(x))
        x = torch.squeeze(x)
        tgt = torch.roll(x, -1, dims=2)
        x = self.transformer(x, tgt, src_mask)
        x = self.flatten(x)
        x = self.fc(x)
        return x
