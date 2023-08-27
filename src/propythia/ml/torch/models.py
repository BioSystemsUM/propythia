import torch
from torch import nn

import sys
sys.path.append("../")
from .utils import calc_maxpool_output

class MLP(nn.Module):
    """
    Implementation of DeepHe's MLP model
    - Paper: https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1008229
    - Code using Keras: https://github.com/xzhang2016/DeepHE/blob/master/DNN.py
    """

    def __init__(self, input_size, hidden_size, output_size, dropout):
        super(MLP, self).__init__()

        self.linear = nn.Linear(input_size, hidden_size * 2)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.fc_last = nn.Linear(hidden_size * 2, output_size)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.linear(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc_last(x)
        x = self.sigmoid(x)
        return x


class CNN(nn.Module):
    """
    Implementation of Primer's CNN model (https://github.com/abidlabs/deep-learning-genomics-primer/blob/master/A_Primer_on_Deep_Learning_in_Genomics_Public.ipynb)
    """

    def __init__(self, input_size, hidden_size, output_size, dropout, sequence_length):
        super(CNN, self).__init__()

        self.conv1 = nn.Conv1d(input_size, hidden_size, kernel_size=12, stride=1, padding=0, dilation=1)
        self.maxpool = nn.MaxPool1d(kernel_size=12, stride=5, padding=0, dilation=1)
        
        max_pool_output = calc_maxpool_output(hidden_size, sequence_length)
        
        self.linear = nn.Linear(max_pool_output, hidden_size * 2)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

        self.fc_last = nn.Linear(hidden_size * 2, output_size)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.conv1(x)
        x = self.maxpool(x)
        x = torch.flatten(x, 1)
        x = self.linear(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc_last(x)
        x = self.softmax(x)
        return x
    
class LSTM(nn.Module):
    """
    https://github.com/aladdinpersson/Machine-Learning-Collection/blob/master/ML/Pytorch/Basics/pytorch_rnn_gru_lstm.py
    """

    def __init__(self, input_size, hidden_size, is_bidirectional, num_layers, output_size, sequence_length, dropout, device):
        super(LSTM, self).__init__()
        self.num_directions = 2 if is_bidirectional else 1
        self.hidden_size = hidden_size
        self.device = device
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, bidirectional=is_bidirectional, dropout=dropout)
        self.fc = nn.Linear(hidden_size * sequence_length * self.num_directions, output_size)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers * self.num_directions, x.size(0), self.hidden_size).to(self.device)
        c0 = torch.zeros(self.num_layers * self.num_directions, x.size(0), self.hidden_size).to(self.device)

        out, _ = self.lstm(x, (h0, c0))  
        out = out.reshape(out.shape[0], -1)
        out = self.fc(out)
        return out

class GRU(nn.Module):
    """
    https://github.com/aladdinpersson/Machine-Learning-Collection/blob/master/ML/Pytorch/Basics/pytorch_rnn_gru_lstm.py
    """

    def __init__(self, input_size, hidden_size, is_bidirectional, num_layers, output_size, sequence_length, dropout, device):
        super(GRU, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.device = device
        self.num_directions = 2 if is_bidirectional else 1

        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True, bidirectional=is_bidirectional, dropout=dropout)
        self.fc = nn.Linear(hidden_size * sequence_length * self.num_directions, output_size)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers * self.num_directions, x.size(0), self.hidden_size).to(self.device)

        out, _ = self.gru(x, h0)
        out = out.reshape(out.shape[0], -1)
        out = self.fc(out)
        return out

class CNN_LSTM(nn.Module):
    
    def __init__(self, input_size, hidden_size, is_bidirectional, num_layers, sequence_length, output_size, dropout, device):
        super(CNN_LSTM, self).__init__()
        self.num_directions = 2 if is_bidirectional else 1
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.device = device

        self.conv1 = nn.Conv1d(input_size, hidden_size, kernel_size=12, stride=1, padding=0, dilation=1)
        self.maxpool = nn.MaxPool1d(kernel_size=12, stride=5, padding=0, dilation=1)
        
        max_pool_output = calc_maxpool_output(hidden_size, sequence_length)
        
        self.linear = nn.Linear(max_pool_output, hidden_size * 2)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        
        self.lstm = nn.LSTM(hidden_size * 2, hidden_size, num_layers, batch_first=True, bidirectional=is_bidirectional, dropout=dropout)
        self.last_linear = nn.Linear(hidden_size * self.num_directions, output_size)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers * self.num_directions, x.size(0), self.hidden_size).to(self.device)
        c0 = torch.zeros(self.num_layers * self.num_directions, x.size(0), self.hidden_size).to(self.device)

        x = x.permute(0, 2, 1)
        x = self.conv1(x)
        x = self.maxpool(x)
        x = torch.flatten(x, 1)
        x = self.linear(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = x.reshape(x.size(0), 1, -1) 
        out, _ = self.lstm(x, (h0, c0))
        out = out.reshape(out.shape[0], -1)
        y = self.last_linear(out)
        return y
    
class CNN_GRU(nn.Module):
    """
    https://medium.com/geekculture/recap-of-how-to-implement-lstm-in-pytorch-e17ec11b061e
    """
    def __init__(self, input_size, hidden_size, is_bidirectional, num_layers, sequence_length, output_size, dropout, device):
        super(CNN_GRU, self).__init__()
        
        self.num_directions = 2 if is_bidirectional else 1
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.device = device
        
        self.conv1 = nn.Conv1d(input_size, hidden_size, kernel_size=12, stride=1, padding=0, dilation=1)
        self.maxpool = nn.MaxPool1d(kernel_size=12, stride=5, padding=0, dilation=1)
        
        max_pool_output = calc_maxpool_output(hidden_size, sequence_length)
        
        self.linear = nn.Linear(max_pool_output, hidden_size * 2)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        
        self.gru = nn.GRU(hidden_size * 2, hidden_size, num_layers, batch_first=True, bidirectional=is_bidirectional, dropout=dropout)
        self.last_linear = nn.Linear(hidden_size * self.num_directions, output_size)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers * self.num_directions, x.size(0), self.hidden_size).to(self.device)

        x = x.permute(0, 2, 1)
        x = self.conv1(x)
        x = self.maxpool(x)
        x = torch.flatten(x, 1)
        x = self.linear(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = x.reshape(x.size(0), 1, -1)     
        out, _ = self.gru(x, h0)
        out = out.reshape(out.shape[0], -1)
        y = self.last_linear(out)
        return y
    