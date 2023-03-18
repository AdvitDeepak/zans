import torch
import torch.nn as nn

import constants

class CNN(torch.nn.Module):
    def __init__(
        self,
        batch_size=0,
        n_layers=1,
    ):
        super().__init__()
        self.num_classes = 4

        # Conv. block 1
        # basic_cnn_model.add(Conv2D(filters=25, kernel_size=(10,1), padding='same', activation='elu', input_shape=(250,1,22)))
        self.conv1 = nn.Conv2d(in_channels=22, out_channels=25, kernel_size=(1,5), stride=1, padding='same')
        self.elu1 = nn.ELU(alpha=1.0)
        # basic_cnn_model.add(MaxPooling2D(pool_size=(3,1), padding='same')) # Read the keras documentation
        self.maxpool1 = nn.MaxPool2d(kernel_size=(1,2))
        # basic_cnn_model.add(BatchNormalization())
        self.batchnorm1 = nn.BatchNorm2d(num_features=25, eps=0.001, momentum=0.99, affine=True)
        # basic_cnn_model.add(Dropout(0.5))
        self.dropout1 = nn.Dropout(0.5)

        # Conv. block 2
        # basic_cnn_model.add(Conv2D(filters=50, kernel_size=(10,1), padding='same', activation='elu'))
        self.conv2 = nn.Conv2d(in_channels=25, out_channels=50, kernel_size=(1,5), stride=1, padding='same')
        self.elu2 = nn.ELU(alpha=1.0)
        # basic_cnn_model.add(MaxPooling2D(pool_size=(3,1), padding='same'))
        self.maxpool2 = nn.MaxPool2d(kernel_size=(1,2), padding=(0,1))
        # basic_cnn_model.add(BatchNormalization())
        self.batchnorm2 = nn.BatchNorm2d(num_features=50, eps=0.001, momentum=0.99, affine=True)
        # basic_cnn_model.add(Dropout(0.5))
        self.dropout2 = nn.Dropout(0.5)

        # Conv. block 3
        # basic_cnn_model.add(Conv2D(filters=100, kernel_size=(10,1), padding='same', activation='elu'))
        self.conv3 = nn.Conv2d(in_channels=50, out_channels=100, kernel_size=(1,5), stride=1, padding='same')
        self.elu3 = nn.ELU(alpha=1.0)
        # basic_cnn_model.add(MaxPooling2D(pool_size=(3,1), padding='same'))
        self.maxpool3 = nn.MaxPool2d(kernel_size=(1,2), padding=(0,1))
        # basic_cnn_model.add(BatchNormalization())
        self.batchnorm3 = nn.BatchNorm2d(num_features=100, eps=0.001, momentum=0.99, affine=True)
        # basic_cnn_model.add(Dropout(0.5))
        self.dropout3 = nn.Dropout(0.5)

        # Conv. block 4
        # basic_cnn_model.add(Conv2D(filters=200, kernel_size=(10,1), padding='same', activation='elu'))
        self.conv4 = nn.Conv2d(in_channels=100, out_channels=200, kernel_size=(1,5), stride=1, padding='same')
        self.elu4 = nn.ELU(alpha=1.0)
        # basic_cnn_model.add(MaxPooling2D(pool_size=(3,1), padding='same'))
        self.maxpool4 = nn.MaxPool2d(kernel_size=(1,2))
        # basic_cnn_model.add(BatchNormalization())
        self.batchnorm4 = nn.BatchNorm2d(num_features=200, eps=0.001, momentum=0.99, affine=True)
        # basic_cnn_model.add(Dropout(0.5))
        self.dropout4 = nn.Dropout(0.5)

        # Conv. block 5
        # basic_cnn_model.add(Conv2D(filters=200, kernel_size=(10,1), padding='same', activation='elu'))
        self.conv5 = nn.Conv2d(in_channels=200, out_channels=400, kernel_size=(1,5), stride=1, padding='same')
        self.elu5 = nn.ELU(alpha=1.0)
        # basic_cnn_model.add(MaxPooling2D(pool_size=(3,1), padding='same'))
        self.maxpool5 = nn.MaxPool2d(kernel_size=(1,2))
        # basic_cnn_model.add(BatchNormalization())
        self.batchnorm5 = nn.BatchNorm2d(num_features=400, eps=0.001, momentum=0.99, affine=True)
        # basic_cnn_model.add(Dropout(0.5))
        self.dropout5 = nn.Dropout(0.5)

        # Output layer with Softmax activation
        # basic_cnn_model.add(Flatten()) # Flattens the input
        self.flatten = nn.Flatten()
        # basic_cnn_model.add(Dense(4, activation='softmax')) # Output FC layer with softmax activation
        self.fc = nn.Linear(3200, self.num_classes)
        self.softmax = nn.Softmax(dim=1)
    
    def forward(self, inputs):
        outputs = inputs.to(torch.float32)
        # print(outputs.shape)
        # Conv Layer 1
        outputs = self.conv1(outputs)
        outputs = self.elu1(outputs)
        outputs = self.maxpool1(outputs)
        outputs = self.batchnorm1(outputs)
        outputs = self.dropout1(outputs)
        # print(outputs.shape)
        # Conv Layer 2
        outputs = self.conv2(outputs)
        outputs = self.elu2(outputs)
        outputs = self.maxpool2(outputs)
        outputs = self.batchnorm2(outputs)
        outputs = self.dropout2(outputs)
        # print(outputs.shape)
        # Conv Layer 3
        outputs = self.conv3(outputs)
        outputs = self.elu3(outputs)
        outputs = self.maxpool3(outputs)
        outputs = self.batchnorm3(outputs)
        outputs = self.dropout3(outputs)
        # print(outputs.shape)
        # Conv Layer 4
        outputs = self.conv4(outputs)
        outputs = self.elu4(outputs)
        outputs = self.maxpool4(outputs)
        outputs = self.batchnorm4(outputs)
        outputs = self.dropout4(outputs)
        # print(outputs.shape)
        # Conv Layer 5
        outputs = self.conv5(outputs)
        outputs = self.elu5(outputs)
        outputs = self.maxpool5(outputs)
        outputs = self.batchnorm5(outputs)
        outputs = self.dropout5(outputs)
        # print(outputs.shape)
        # FC layer
        outputs = self.flatten(outputs)
        outputs = self.fc(outputs)
        # outputs = self.softmax(outputs)
        return outputs