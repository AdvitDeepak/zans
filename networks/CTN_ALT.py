import torch.nn as nn
import torch 

class CTN_ALT(torch.nn.Module):
    """
    Basic logistic regression on 224x224x3 images.
    """
#print the size of theoutput
    def __init__(self, batch_size):
        super().__init__()
        self.num_heads = 5
        self.conv = nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.transformer = nn.Transformer(d_model=125, nhead=self.num_heads, num_encoder_layers=12, num_decoder_layers=12, batch_first=True)
        self.num_classes = 4
        self.flatten = nn.Flatten()
        self.dropout = nn.Dropout(p=0.4)
        self.fc1 = nn.Linear(1375, 64)  
        self.fc2 = nn.Linear(64, self.num_classes)

    def forward(self, inp):
        # print(inp.shape)
        x = inp.unsqueeze(1) # Add channel dimension
        x = self.conv(x.float())
        x = self.pool(x)
        # print(x.shape)
        src = x[:, 0, :, :].float()
        tgt = x[:, 1, :, :].float()
        #src and trt isnt equal to the d model 
        x = self.transformer(src, tgt) 
        x = self.flatten(x)
        x = self.dropout(x) 
        
        x = self.fc1(x) 
        x = self.dropout(x)
        x = self.fc2(x)
        return x
