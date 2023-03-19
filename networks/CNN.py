import torch
import torch.nn as nn

import constants


class CNN(torch.nn.Module):
    def __init__(self, model_config):
        super().__init__()

        self.input_size = 125
        self.hidden_size = 60
        self.output_size = 4 
        self.num_layers = 2 
        self.dropout_rate = 0.20 
        self.batch_size = batch_size

        self.conv1 =  nn.Sequential(
            nn.Conv2d(in_channels=22, out_channels=32, kernel_size=(1,5), stride=1, padding='same'), 
            nn.ReLU(), 
            nn.MaxPool2d(kernel_size=(1,2)), 
            nn.BatchNorm2d(num_features=32, eps=0.001, momentum=0.99, affine=True), 
            nn.Dropout(0.55), 
        ) 

        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(1,10), stride=1, padding='same'), 
            nn.ReLU(), 
            nn.MaxPool2d(kernel_size=(1,2), padding=(0,1)), 
            nn.BatchNorm2d(num_features=64, eps=0.001, momentum=0.99, affine=True), 
            nn.Dropout(0.65), 
        )

        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(1,15), stride=1, padding='same'),
            nn.ReLU(), 
            nn.MaxPool2d(kernel_size=(1,2), padding=(0,1)),
            nn.BatchNorm2d(num_features=128, eps=0.001, momentum=0.99, affine=True),
            nn.Dropout(0.75),
        )   

        self.flatten = nn.Flatten() 

        # Define linear layers
        self.linear_layers = nn.Sequential(
            nn.Linear(4*1024, 4*128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(4*128, 4*16),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(4*16, 4)
        )
        
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
