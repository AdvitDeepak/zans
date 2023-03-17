import torch
import torch.nn as nn


class TRN(torch.nn.Module):
    """
    Basic logistic regression on 224x224x3 images.
    """

    def __init__(self, batch_size):
        super().__init__()
        self.num_heads = 5
        self.transformer = nn.Transformer(d_model=250, nhead=self.num_heads, num_encoder_layers=12, num_decoder_layers=12, batch_first=True)
        self.num_classes = 4
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(250 * 22, self.num_classes)
        # self.sigmoid = nn.Sigmoid()


#update the data loader stuff 
    def forward(self, inp, src_mask=None):
        src = inp[:, 0, :, :].float()
        tgt = inp[:, 1, :, :].float()
        x = self.transformer(src, tgt, src_mask) 
        x = self.flatten(x)
        x = self.fc(x)
        # x = self.sigmoid(x)
        return x
    
    # https://github.com/pytorch/tutorials/blob/main/beginner_source/transformer_tutorial.py
    def generate_square_subsequent_mask(self, sz, batch_size):
        """Generates an upper-triangular matrix of -inf, with zeros on diag."""
        return torch.triu(torch.ones(self.num_heads * batch_size, sz, sz) * float('-inf'), diagonal=1)
