from torch import nn
 
class Net(nn.Module):
    """
    Implementation of https://github.com/onceupon/deep_learning_DNA/blob/master/learnseq.ipynb
    """
    def __init__(self, input_size, hidden_size, output_size, dropout):
        super(Net, self).__init__()
        
        self.conv1 = nn.Conv1d(input_size, hidden_size, 10, stride=1, padding=0)
        self.act1 = nn.ReLU()
        self.max_pool = nn.MaxPool1d(10, stride=5)
        
        self.fc1 = nn.Linear(hidden_size * 47, (hidden_size // 4))
        self.act2 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)
        
        self.fc2 = nn.Linear((hidden_size // 4), output_size)
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