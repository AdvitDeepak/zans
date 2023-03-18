import torch
import torch.nn as nn

import constants


class RNN(torch.nn.Module):
    def __init__(self, model_config):
        super(RNN, self).__init__()
        input_size = constants.DATA["TRIM_END"] // constants.DATA[
            "AUG_SUBSAMPLE_SIZE"]  # 250
        self.n_layers = model_config['N_LAYERS']
        self.hidden_size = model_config['HIDDEN_SIZE']
        self.batch_size = model_config['BATCH_SIZE']
        self.rnn_dropout = model_config['RNN_DROPOUT']
        self.fc_dropout = model_config['FC_DROPOUT']

        self.rnn = nn.GRU(
            input_size,
            self.hidden_size,
            num_layers=self.n_layers,
            batch_first=True,
            dropout=self.rnn_dropout,
        )
        self.decoder = nn.Linear(
            self.hidden_size * constants.DATA["NUM_ELECTRODES"],
            constants.DATA["NUM_CLASSES"])
        self.dropout = nn.Dropout(self.fc_dropout)

    def init_hidden(self):
        return torch.randn(self.n_layers, self.batch_size, self.hidden_size)

    def forward(self, inputs):
        # Avoid breaking if the last batch has a different size
        batch_size = inputs.size(0)
        if batch_size != self.batch_size:
            self.batch_size = batch_size

        inputs = inputs.to(torch.float32)
        output, hidden = self.rnn(inputs, self.init_hidden())
        # 32, 22, 128 (BATCH_SIZE, NUM_NODES, HIDDEN_SIZE)
        output = self.dropout(output)
        output = self.decoder(output.reshape(self.batch_size, -1))
        return output