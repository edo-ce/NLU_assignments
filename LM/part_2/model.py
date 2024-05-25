import torch
import torch.nn as nn
from torch.autograd import Variable

# module implementing variational dropout
class VariationalDropout(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, dropout=0.1):
        if not self.training or not dropout:
            return x
        # m = x.data.new(1, x.size(1), x.size(2)).bernoulli_(1 - dropout)
        m = x.data.empty(x.size(0), 1, x.size(2)).bernoulli_(1 - dropout)
        mask = Variable(m, requires_grad=False) / (1 - dropout)
        mask = mask.expand_as(x)
        return mask * x

class LM_LSTM_REGULARIZED(nn.Module):
    def __init__(self, emb_size, hidden_size, output_size, pad_index=0, out_dropout=0.1,
                 emb_dropout=0.1, n_layers=1):
        super(LM_LSTM_REGULARIZED, self).__init__()
        # Token ids to vectors, we will better see this in the next lab
        self.embedding = nn.Embedding(output_size, emb_size, padding_idx=pad_index)
        self.lstm = nn.LSTM(emb_size, hidden_size, n_layers, bidirectional=False, batch_first=True)
        self.pad_token = pad_index
        # Linear layer to project the hidden layer to our output space
        # self.output = nn.Linear(hidden_size, output_size)
        self.output = self.embedding.weight
        self.dropout = VariationalDropout()

    def forward(self, input_sequence):
        emb = self.embedding(input_sequence)
        emb = self.dropout(emb)
        lstm_out, _  = self.lstm(emb)
        lstm_out = self.dropout(lstm_out)
        # output = self.output(lstm_out).permute(0,2,1)
        output = torch.matmul(lstm_out, self.output.t())
        output = output.permute(0, 2, 1)
        return output