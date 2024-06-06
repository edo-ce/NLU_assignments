import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from utils import DEVICE

# module implementing variational dropout
class VariationalDropout(nn.Module):
    def __init__(self, dropout=0.1):
        super().__init__()
        self.dropout = dropout

    def forward(self, x):
        if not self.dropout:
            return x
        # create a (64 x 1 x emb_size/hid_size) mask
        m = torch.empty(x.size(0), 1, x.size(2)).bernoulli_(1 - self.dropout).to(DEVICE)
        mask = m / (1 - self.dropout)
        # replicate the mask for the length of the sequence
        mask = mask.expand_as(x)
        return mask * x

class LM_LSTM_REGULARIZED(nn.Module):
    def __init__(self, emb_size, hidden_size, output_size, pad_index=0, is_dropout=True, dropout=0.1, n_layers=1, name="lstm_regularized"):
        super(LM_LSTM_REGULARIZED, self).__init__()
        # Token ids to vectors
        self.embedding = nn.Embedding(output_size, emb_size, padding_idx=pad_index)
        self.lstm = nn.LSTM(emb_size, hidden_size, n_layers, bidirectional=False, batch_first=True)
        self.pad_token = pad_index
        # Linear layer to project the hidden layer to our output space
        self.output = self.embedding.weight
        self.dropout = VariationalDropout(dropout)
        self.is_dropout = is_dropout
        self.name = name

    def forward(self, input_sequence, training=True):
        emb = self.embedding(input_sequence)
        
        # apply dropout if needed and if we are in training mode
        if self.is_dropout and training:
            emb = self.dropout(emb)
        lstm_out, _  = self.lstm(emb)

        # apply dropout if needed and if we are in training mode
        if self.is_dropout and training:
            lstm_out = self.dropout(lstm_out)

        output = torch.matmul(lstm_out, self.output.t())
        output = output.permute(0, 2, 1)
        return output