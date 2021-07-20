import torch
import torch.nn as nn

# LSTM model definition
class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, seq_size, output_size, batch_size):
        super(LSTM, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.output_size = output_size
        self.batch_size = batch_size
        self.seq_size = seq_size

        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, bidirectional=True)
        self.h2inter = nn.Linear(2*hidden_size, 1)  
        self.h3o = nn.Linear(seq_size, output_size)  
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU()

    def forward(self, x):
        output, (h_n, c_n) = self.lstm(x)
        output = self.h2inter(output)
        output = self.relu(output)
        output = self.h3o(torch.squeeze(output))
        output = self.sigmoid(output)

        return output

    