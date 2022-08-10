import torch
from torch import nn
import math


class MLP(nn.Module):
    """
    Implementation of DeepHe's MLP model
    - Paper: https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1008229
    - Code using Keras: https://github.com/xzhang2016/DeepHE/blob/master/DNN.py
    """

    def __init__(self, input_size, hidden_size, output_size, dropout):
        super(MLP, self).__init__()

        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.fc2 = nn.Linear(hidden_size, hidden_size * 2)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.fc3 = nn.Linear(hidden_size * 2, hidden_size * 4)
        self.relu3 = nn.ReLU()
        self.dropout3 = nn.Dropout(dropout)

        self.fc4 = nn.Linear(hidden_size * 4, output_size)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.dropout1(x)
        x = self.fc2(x)
        x = self.relu2(x)
        x = self.dropout2(x)
        x = self.fc3(x)
        x = self.relu3(x)
        x = self.dropout3(x)
        x = self.fc4(x)
        x = self.sigmoid(x)
        return x


class CNN(nn.Module):
    """
    Implementation of Primer's CNN model (https://github.com/abidlabs/deep-learning-genomics-primer/blob/master/A_Primer_on_Deep_Learning_in_Genomics_Public.ipynb)
    """

    def __init__(self, sequence_length, input_size, hidden_size, output_size):
        super(CNN, self).__init__()

        # ------------------- Calculation of max pool output size -------------------
        Conv1_padding = 0
        Conv1_dilation = 1
        Conv1_kernel_size = 12
        Conv1_stride = 1

        L_out = ((sequence_length + 2*Conv1_padding - Conv1_dilation*(Conv1_kernel_size-1) - 1)/Conv1_stride + 1)
        MaxPool_padding = 0
        MaxPool_dilation = 1
        MaxPool_stride = 5
        MaxPool_kernel_size = 12
        max_pool_output = int((L_out+2*MaxPool_padding-MaxPool_dilation*(MaxPool_kernel_size-1)-1)/MaxPool_stride+1)

        max_pool_output *= hidden_size
        # ---------------------------------------------------------------------------

        self.conv1 = nn.Conv1d(input_size, hidden_size, kernel_size=Conv1_kernel_size,
                               stride=Conv1_stride, padding=Conv1_padding, dilation=Conv1_dilation)
        self.maxpool = nn.MaxPool1d(kernel_size=MaxPool_kernel_size, stride=MaxPool_stride,
                                    padding=MaxPool_padding, dilation=MaxPool_dilation)
        self.fc1 = nn.Linear(max_pool_output, hidden_size)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.conv1(x)
        x = self.maxpool(x)
        x = x.reshape(x.shape[0], -1)
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.fc2(x)
        x = self.softmax(x)
        return x

class LSTM(nn.Module):
    """
    https://github.com/aladdinpersson/Machine-Learning-Collection/blob/master/ML/Pytorch/Basics/pytorch_rnn_gru_lstm.py
    """

    def __init__(self, input_size, hidden_size, is_bidirectional, num_layers, num_classes, sequence_length, device):
        super(LSTM, self).__init__()
        self.num_directions = 2 if is_bidirectional else 1
        self.hidden_size = hidden_size
        self.device = device
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, bidirectional=is_bidirectional)
        self.fc = nn.Linear(hidden_size * sequence_length * self.num_directions, num_classes)

    def forward(self, x):
        # Set initial hidden and cell states
        h0 = torch.zeros(self.num_layers * self.num_directions, x.size(0), self.hidden_size).to(self.device)
        c0 = torch.zeros(self.num_layers * self.num_directions, x.size(0), self.hidden_size).to(self.device)

        # Forward propagate LSTM
        out, _ = self.lstm(
            x, (h0, c0)
        )  # out: tensor of shape (batch_size, seq_length, hidden_size)
        out = out.reshape(out.shape[0], -1)

        # Decode the hidden state of the last time step
        out = self.fc(out)
        return out

class GRU(nn.Module):
    """
    https://github.com/aladdinpersson/Machine-Learning-Collection/blob/master/ML/Pytorch/Basics/pytorch_rnn_gru_lstm.py
    """

    def __init__(self, input_size, hidden_size, num_layers, num_classes, sequence_length, device):
        super(GRU, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.device = device

        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size * sequence_length, num_classes)

    def forward(self, x):
        # Set initial hidden and cell states
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(self.device)

        # Forward propagate LSTM
        out, _ = self.gru(x, h0)
        out = out.reshape(out.shape[0], -1)

        # Decode the hidden state of the last time step
        out = self.fc(out)
        return out

class CNN_LSTM(nn.Module):
    """
    https://medium.com/geekculture/recap-of-how-to-implement-lstm-in-pytorch-e17ec11b061e
    """
    def __init__(self, input_size, hidden_size, is_bidirectional, num_layers, sequence_length, no_classes, device):
        super(CNN_LSTM, self).__init__()
        
        self.num_directions = 2 if is_bidirectional else 1
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.device = device
        
        self.conv1 = nn.Sequential(
            nn.Conv1d(in_channels=input_size, out_channels=16, kernel_size=5, stride=2, padding=2),
            nn.ReLU(),
        )
        self.conv2 = nn.Sequential(
            nn.Conv1d(in_channels=16, out_channels=32, kernel_size=5, stride=2, padding=2),
            nn.ReLU(),
        )
        
        self.lstm = nn.LSTM(32, hidden_size, num_layers, batch_first=True, bidirectional=is_bidirectional)
        
        linear_input = (math.ceil((sequence_length / 2) / 2) * hidden_size) * self.num_directions
        self.linear = nn.Linear(linear_input, no_classes)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers * self.num_directions, x.size(0), self.hidden_size).to(self.device)
        c0 = torch.zeros(self.num_layers * self.num_directions, x.size(0), self.hidden_size).to(self.device)

        x = x.permute(0, 2, 1)
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.permute(0, 2, 1)       
        out, _ = self.lstm(
            x, (h0, c0)
        ) 
        out = out.reshape(out.shape[0], -1)
        y  = self.linear(out)
        return y
    
class CNN_GRU(nn.Module):
    """
    https://medium.com/geekculture/recap-of-how-to-implement-lstm-in-pytorch-e17ec11b061e
    """
    def __init__(self, input_size, hidden_size, is_bidirectional, num_layers, sequence_length, no_classes, device):
        super(CNN_GRU, self).__init__()
        
        self.num_directions = 2 if is_bidirectional else 1
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.device = device
        
        self.conv1 = nn.Sequential(
            nn.Conv1d(in_channels=input_size, out_channels=16, kernel_size=5, stride=2, padding=2),
            nn.ReLU(),
        )
        self.conv2 = nn.Sequential(
            nn.Conv1d(in_channels=16, out_channels=32, kernel_size=5, stride=2, padding=2),
            nn.ReLU(),
        )
        
        self.gru = nn.GRU(32, hidden_size, num_layers, batch_first=True, bidirectional=is_bidirectional)
        
        linear_input = (math.ceil((sequence_length / 2) / 2) * hidden_size) * self.num_directions
        self.linear = nn.Linear(linear_input, no_classes)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers * self.num_directions, x.size(0), self.hidden_size).to(self.device)

        x = x.permute(0, 2, 1)
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.permute(0, 2, 1)       
        out, _ = self.gru(x, h0)
        out = out.reshape(out.shape[0], -1)
        y  = self.linear(out)
        return y