import torch
import torch.nn as nn

import constants


class TRN(torch.nn.Module):
    """Transformer Architecture"""
    def __init__(self, model_config):
        super().__init__()
        input_size = constants.DATA["TRIM_END"] // constants.DATA[
            "AUG_SUBSAMPLE_SIZE"]  # 250
        self.num_heads = model_config["N_HEADS"]
        self.transformer = nn.Transformer(
            d_model=input_size,
            nhead=model_config['N_HEADS'],
            num_encoder_layers=model_config['N_LAYERS'],
            num_decoder_layers=model_config['N_LAYERS'],
            batch_first=True)
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(input_size * constants.DATA["NUM_ELECTRODES"],
                            constants.DATA["NUM_CLASSES"])

    def forward(self, inp, src_mask=None):
        src = inp[:, 0, :, :].float()
        tgt = inp[:, 1, :, :].float()
        x = self.transformer(src, tgt, src_mask)
        x = self.flatten(x)
        x = self.fc(x)
        return x

    # https://github.com/pytorch/tutorials/blob/main/beginner_source/transformer_tutorial.py
    def generate_square_subsequent_mask(self, sz, batch_size):
        """Generates an upper-triangular matrix of -inf, with zeros on diag."""
        return torch.triu(torch.ones(self.num_heads * batch_size, sz, sz) *
                          float('-inf'),
                          diagonal=1)
