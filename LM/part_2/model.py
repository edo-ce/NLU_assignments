import torch
import torch.nn as nn
from torch.autograd import Variable
from utils import DEVICE

# module implementing variational dropout
class VariationalDropout(nn.Module):
    def __init__(self, dropout=0.1):
        super().__init__()
        self.dropout = dropout

    def forward(self, x):
        # TODO: do the right checks
        if not self.training or not self.dropout:
            return x
        # m = x.data.new(1, x.size(1), x.size(2)).bernoulli_(1 - self.dropout)
        # create a (64 x 1 x emb_size/hid_size) mask
        m = torch.empty(x.size(0), 1, x.size(2)).bernoulli_(1 - self.dropout).to(DEVICE)
        mask = Variable(m, requires_grad=False) / (1 - self.dropout)
        # replicate the mask for the length of the sequence
        mask = mask.expand_as(x)
        return mask * x

class LM_LSTM_REGULARIZED(nn.Module):
    def __init__(self, emb_size, hidden_size, output_size, pad_index=0, dropout=0.1, n_layers=1):
        super(LM_LSTM_REGULARIZED, self).__init__()
        # Token ids to vectors, we will better see this in the next lab
        self.embedding = nn.Embedding(output_size, emb_size, padding_idx=pad_index)
        self.lstm = nn.LSTM(emb_size, hidden_size, n_layers, bidirectional=False, batch_first=True)
        self.pad_token = pad_index
        # Linear layer to project the hidden layer to our output space
        # self.output = nn.Linear(hidden_size, output_size)
        self.output = self.embedding.weight
        self.dropout = VariationalDropout(dropout)

    def forward(self, input_sequence):
        emb = self.embedding(input_sequence)
        emb = self.dropout(emb)
        lstm_out, _  = self.lstm(emb)
        lstm_out = self.dropout(lstm_out)
        # output = self.output(lstm_out).permute(0,2,1)
        output = torch.matmul(lstm_out, self.output.t())
        output = output.permute(0, 2, 1)
        return output