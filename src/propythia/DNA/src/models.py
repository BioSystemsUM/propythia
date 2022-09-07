import torch
from torch import nn
import math

class MLP(nn.Module):
    """
    Implementation of DeepHe's MLP model
    - Paper: https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1008229
    - Code using Keras: https://github.com/xzhang2016/DeepHE/blob/master/DNN.py
    """

    def __init__(self, input_size, hidden_size, output_size, num_layers, dropout, isHalf):
        super(MLP, self).__init__()

        original_hidden_size = hidden_size
        self.num_layers = num_layers
        
        convert = (lambda x, y: x // y) if isHalf else (lambda x,y: x * y)

        for i in range(num_layers):
            if i == 0:
                linear_input = input_size
            else:
                linear_input = hidden_size
            self.add_module(
                'fc{}'.format(i),
                nn.Linear(linear_input, convert(hidden_size,2))
            )
            self.add_module(
                'relu{}'.format(i),
                nn.ReLU()
            )
            self.add_module(
                'dropout{}'.format(i),
                nn.Dropout(dropout)
            )
            hidden_size = convert(hidden_size,2)

        self.fc_last = nn.Linear(convert(original_hidden_size,(2 ** num_layers)), output_size)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        for i in range(self.num_layers):
            x = getattr(self, 'fc{}'.format(i))(x)
            x = getattr(self, 'relu{}'.format(i))(x)
            x = getattr(self, 'dropout{}'.format(i))(x) 
        x = self.fc_last(x)
        x = self.sigmoid(x)
        return x


class CNN(nn.Module):
    """
    Implementation of Primer's CNN model (https://github.com/abidlabs/deep-learning-genomics-primer/blob/master/A_Primer_on_Deep_Learning_in_Genomics_Public.ipynb)
    """

    def __init__(self, input_size, hidden_size, output_size, sequence_length, num_layers, dropout, isHalf):
        super(CNN, self).__init__()
        self.num_layers = num_layers
        original_hidden_size = hidden_size

        # ------------------- Calculation of max pool output size -------------------
        conv1_padding = 0
        conv1_dilation = 1
        conv1_kernel_size = 12
        conv1_stride = 1

        l_out = ((sequence_length + 2*conv1_padding - conv1_dilation*(conv1_kernel_size-1) - 1)/conv1_stride + 1)
        maxpool_padding = 0
        maxpool_dilation = 1
        maxpool_stride = 5
        maxpool_kernel_size = 12
        max_pool_output = int((l_out+2*maxpool_padding-maxpool_dilation*(maxpool_kernel_size-1)-1)/maxpool_stride+1)

        max_pool_output *= hidden_size
        # ---------------------------------------------------------------------------

        self.conv1 = nn.Conv1d(input_size, hidden_size, kernel_size=conv1_kernel_size,
                               stride=conv1_stride, padding=conv1_padding, dilation=conv1_dilation)
        self.maxpool = nn.MaxPool1d(kernel_size=maxpool_kernel_size, stride=maxpool_stride,
                                    padding=maxpool_padding, dilation=maxpool_dilation)
        
        
        for i in range(num_layers):
            if i == 0:
                linear_input = max_pool_output
            else:
                linear_input = hidden_size
            self.add_module(
                'fc{}'.format(i),
                nn.Linear(linear_input, hidden_size * 2)
            )
            self.add_module(
                'relu{}'.format(i),
                nn.ReLU()
            )
            self.add_module(
                'dropout{}'.format(i),
                nn.Dropout(dropout)
            )
            hidden_size = hidden_size * 2

        self.fc_last = nn.Linear(original_hidden_size * (2 ** num_layers), output_size)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.conv1(x)
        x = self.maxpool(x)
        x = torch.flatten(x, 1)
        for i in range(self.num_layers):
            x = getattr(self, 'fc{}'.format(i))(x)
            x = getattr(self, 'relu{}'.format(i))(x)
            x = getattr(self, 'dropout{}'.format(i))(x)
        x = self.fc_last(x)
        x = self.softmax(x)
        return x
    
class LSTM(nn.Module):
    """
    https://github.com/aladdinpersson/Machine-Learning-Collection/blob/master/ML/Pytorch/Basics/pytorch_rnn_gru_lstm.py
    """

    def __init__(self, input_size, hidden_size, is_bidirectional, num_layers, output_size, sequence_length, device):
        super(LSTM, self).__init__()
        self.num_directions = 2 if is_bidirectional else 1
        self.hidden_size = hidden_size
        self.device = device
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, bidirectional=is_bidirectional)
        self.fc = nn.Linear(hidden_size * sequence_length * self.num_directions, output_size)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers * self.num_directions, x.size(0), self.hidden_size).to(self.device)
        c0 = torch.zeros(self.num_layers * self.num_directions, x.size(0), self.hidden_size).to(self.device)

        out, _ = self.lstm(
            x, (h0, c0)
        )  
        out = out.reshape(out.shape[0], -1)
        out = self.fc(out)
        return out

class GRU(nn.Module):
    """
    https://github.com/aladdinpersson/Machine-Learning-Collection/blob/master/ML/Pytorch/Basics/pytorch_rnn_gru_lstm.py
    """

    def __init__(self, input_size, hidden_size, is_bidirectional, num_layers, output_size, sequence_length, device):
        super(GRU, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.device = device
        self.num_directions = 2 if is_bidirectional else 1

        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True, bidirectional=is_bidirectional)
        self.fc = nn.Linear(hidden_size * sequence_length * self.num_directions, output_size)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers * self.num_directions, x.size(0), self.hidden_size).to(self.device)

        out, _ = self.gru(x, h0)
        out = out.reshape(out.shape[0], -1)
        out = self.fc(out)
        return out

class CNN_LSTM(nn.Module):
    """
    https://medium.com/geekculture/recap-of-how-to-implement-lstm-in-pytorch-e17ec11b061e
    """
    def __init__(self, input_size, hidden_size, is_bidirectional, num_layers, sequence_length, output_size, device):
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
        self.linear = nn.Linear(linear_input, output_size)

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
    def __init__(self, input_size, hidden_size, is_bidirectional, num_layers, sequence_length, output_size, device):
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
        self.linear = nn.Linear(linear_input, output_size)

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