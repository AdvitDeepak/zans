import torch
import torch.nn as nn


class DRN(torch.nn.Module):
    """
    Basic logistic regression on 224x224x3 images.
    """

    def __init__(self, batch_size):
        super().__init__()
        self.num_heads = 5
        self.num_classes = 4
        self.decoder = nn.TransformerDecoderLayer(d_model=250, nhead=self.num_heads)
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(250 * 22, self.num_classes)
        # self.sigmoid = nn.Sigmoid()


#update the data loader stuff 
    def forward(self, tgt):
        memory = self.generate_square_subsequent_mask(tgt) * tgt
        print(tgt.dtype, memory.dtype)
        x = self.decoder(tgt, memory) 
        x = self.flatten(x)
        x = self.fc(x)
        # x = self.sigmoid(x)
        return x
    
    # https://github.com/pytorch/tutorials/blob/main/beginner_source/transformer_tutorial.py
    def generate_square_subsequent_mask(self, tgt):
        """Generates an upper-triangular matrix of -inf, with zeros on diag."""
        return torch.triu(torch.ones_like(tgt) * float('-inf'), diagonal=1)
