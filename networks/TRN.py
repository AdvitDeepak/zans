import torch
import torch.nn as nn


class TRN(torch.nn.Module):
    """
    Basic logistic regression on 224x224x3 images.
    """

    def __init__(self, batch_size):
        super().__init__()
        self.transformer = nn.Transformer(d_model=250, nhead=5, num_encoder_layers=12, num_decoder_layers=12, batch_first=True)
        self.num_classes = 4
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(250 * 22, self.num_classes)
        self.sigmoid = nn.Sigmoid()


#update the data loader stuff 
    def forward(self, inp):
        src = inp[:, 0, :, :].float()
        tgt = inp[:, 1, :, :].float()
        x = self.transformer(src, tgt) 
        x = self.flatten(x)
        x = self.fc(x)
        x = self.sigmoid(x)
        return x
    
    # https://github.com/pytorch/tutorials/blob/main/beginner_source/transformer_tutorial.py
    def generate_square_subsequent_mask(sz):
        """Generates an upper-triangular matrix of -inf, with zeros on diag."""
        return torch.triu(torch.ones(sz, sz) * float('-inf'), diagonal=1)
