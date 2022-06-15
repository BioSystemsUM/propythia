from torch import nn

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv1d(4, 20, 10, stride=1, padding=0)
        self.fc1 = nn.Linear(940, 10)
        self.dropout1 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(10, 2)

        self.max_pool = nn.MaxPool1d(10, stride=5)

        self.act1 = nn.ReLU()
        self.act2 = nn.ReLU()
        self.act3 = nn.Softmax(dim=1)

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.conv1(x)
        x = self.act1(x)
        x = self.max_pool(x)
        x = x.reshape(x.shape[0], -1)
        x = self.fc1(x)
        x = self.act2(x)
        x = self.dropout1(x)
        x = self.fc2(x)
        x = self.act3(x)

        return x

class MLP(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, dropout):
        super(MLP, self).__init__()
        # one input layer
        self.fc1 = nn.Linear(input_size, hidden_size)
        # three hidden layers
        self.fc2 = nn.Linear(hidden_size, hidden_size * 2)
        self.fc3 = nn.Linear(hidden_size * 2, hidden_size * 4)
        # one output layer
        self.fc4 = nn.Linear(hidden_size * 4, output_size)
        
        # All the hidden layers utilize the ReLU activation function. The output layer uses sigmoid activation function to perform discrete classification.  A dropout layer is used after each hidden layer.
        self.relu1 = nn.ReLU()
        self.relu2 = nn.ReLU()
        self.relu3 = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)
    
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

