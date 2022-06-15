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
        print("Break 0\n", x.shape, "\n", x, "\n", "-" * 50)
        x = self.fc1(x)
        print("Break 1\n", x.shape, "\n", x, "\n", "-" * 50)
        x = self.relu1(x)
        print("Break 2\n", x.shape, "\n", x, "\n", "-" * 50)
        x = self.dropout1(x)
        print("Break 3\n", x.shape, "\n", x, "\n", "-" * 50)
        x = self.fc2(x)
        print("Break 4\n", x.shape, "\n", x, "\n", "-" * 50)
        x = self.relu2(x)
        print("Break 5\n", x.shape, "\n", x, "\n", "-" * 50)
        x = self.dropout2(x)
        print("Break 6\n", x.shape, "\n", x, "\n", "-" * 50)
        x = self.fc3(x)
        print("Break 7\n", x.shape, "\n", x, "\n", "-" * 50)
        x = self.relu3(x)
        print("Break 8\n", x.shape, "\n", x, "\n", "-" * 50)
        x = self.dropout3(x)
        print("Break 9\n", x.shape, "\n", x, "\n", "-" * 50)
        x = self.fc4(x)
        print("Break 10\n", x.shape, "\n", x, "\n", "-" * 50)
        x = self.sigmoid(x)
        print("Break 11\n", x.shape, "\n", x, "\n", "-" * 50)
        return x
