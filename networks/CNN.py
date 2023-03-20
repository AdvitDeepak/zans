import torch
import torch.nn as nn

import constants


class CNN(torch.nn.Module):
    def __init__(self, model_config):
        super().__init__()
        self.batch_size = model_config['BATCH_SIZE']

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=constants.DATA['NUM_ELECTRODES'],
                      out_channels=model_config['OUT_CHANNELS'][0],
                      kernel_size=(1, 5),
                      stride=1,
                      padding='same'),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(1, 2)),
            nn.BatchNorm2d(num_features=model_config['OUT_CHANNELS'][0],
                           eps=0.001,
                           momentum=0.99,
                           affine=True),
            nn.Dropout(model_config['C_DROPOUT'][0]),
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=model_config['OUT_CHANNELS'][0],
                      out_channels=model_config['OUT_CHANNELS'][1],
                      kernel_size=(1, 10),
                      stride=1,
                      padding='same'),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(1, 2), padding=(0, 1)),
            nn.BatchNorm2d(num_features=model_config['OUT_CHANNELS'][1],
                           eps=0.001,
                           momentum=0.99,
                           affine=True),
            nn.Dropout(model_config['C_DROPOUT'][0]),
        )

        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=model_config['OUT_CHANNELS'][1],
                      out_channels=model_config['OUT_CHANNELS'][2],
                      kernel_size=(1, 15),
                      stride=1,
                      padding='same'),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(1, 2), padding=(0, 1)),
            nn.BatchNorm2d(num_features=model_config['OUT_CHANNELS'][2],
                           eps=0.001,
                           momentum=0.99,
                           affine=True),
            nn.Dropout(model_config['C_DROPOUT'][0]),
        )

        self.flatten = nn.Flatten()

        # Define linear layers
        self.linear_layers = nn.Sequential(
            nn.Linear(model_config["HIDDEN_SIZES"][0],
                      model_config["HIDDEN_SIZES"][1]), nn.ReLU(),
            nn.Dropout(model_config['F_DROPOUT']),
            nn.Linear(model_config["HIDDEN_SIZES"][1],
                      model_config["HIDDEN_SIZES"][2]), nn.ReLU(),
            nn.Dropout(model_config['F_DROPOUT']),
            nn.Linear(model_config["HIDDEN_SIZES"][2],
                      constants.DATA['NUM_CLASSES']))

    def forward(self, inputs):
        batch_size = inputs.size(0)
        if batch_size != self.batch_size:
            self.batch_size = batch_size

        x = inputs.to(torch.float32)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.flatten(x)
        x = self.linear_layers(x)

        return x
