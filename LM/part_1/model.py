import torch.nn as nn

# original RNN model
class LM_RNN(nn.Module):
    def __init__(self, emb_size, hidden_size, output_size, pad_index=0, out_dropout=0.1,
                 emb_dropout=0.1, n_layers=1, name="rnn_original"):
        super(LM_RNN, self).__init__()
        # Token ids to vectors
        self.embedding = nn.Embedding(output_size, emb_size, padding_idx=pad_index)
        # Pytorch's RNN layer
        self.rnn = nn.RNN(emb_size, hidden_size, n_layers, bidirectional=False, batch_first=True)
        self.pad_token = pad_index
        # Linear layer to project the hidden layer to our output space
        self.output = nn.Linear(hidden_size, output_size)
        self.name = name

    def forward(self, input_sequence):
        emb = self.embedding(input_sequence)
        rnn_out, _  = self.rnn(emb)
        output = self.output(rnn_out).permute(0,2,1)
        return output

# LSTM model with dropout
class LM_LSTM(nn.Module):
    def __init__(self, emb_size, hidden_size, output_size, pad_index=0, is_dropout=True, out_dropout=0.1,
                 emb_dropout=0.1, n_layers=1, name="lstm"):
        super(LM_LSTM, self).__init__()
        # Token ids to vectors
        self.embedding = nn.Embedding(output_size, emb_size, padding_idx=pad_index)
        # after embedding dropout
        self.embedding_dropout = nn.Dropout(emb_dropout)
        self.lstm = nn.LSTM(emb_size, hidden_size, n_layers, bidirectional=False, batch_first=True)
        # before output dropout
        self.output_dropout = nn.Dropout(out_dropout)
        self.pad_token = pad_index
        # Linear layer to project the hidden layer to our output space
        self.output = nn.Linear(hidden_size, output_size)
        self.is_dropout = is_dropout
        self.name = name if not is_dropout else name + "_dropout"

    def forward(self, input_sequence):
        emb = self.embedding(input_sequence)
        # after embedding dropout
        if self.is_dropout:
            emb = self.embedding_dropout(emb)
        lstm_out, _  = self.lstm(emb)
        # before output dropout
        if self.is_dropout:
            lstm_out = self.output_dropout(lstm_out)
        output = self.output(lstm_out).permute(0,2,1)
        return output