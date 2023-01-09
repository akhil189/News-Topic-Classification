import torch
import torch.nn as nn
import torch.nn.functional as F

class Bi_LSTM(nn.ModuleList):
    '''Bi-directional LSTM'''
    
    def __init__(self, input_size, hidden_dim, lstm_layers, num_classes):
        super(Bi_LSTM, self).__init__()

        self.input_size = input_size
        self.hidden_dim = hidden_dim
        self.LSTM_layers = lstm_layers
        
        self.dropout = nn.Dropout(0.25)
        self.embedding = nn.Embedding(self.input_size, self.hidden_dim, padding_idx=0)
        self.lstm = nn.LSTM(input_size=self.hidden_dim,
                            hidden_size=self.hidden_dim,
                            num_layers=self.LSTM_layers,
                            batch_first=True,
                            bidirectional=True)
        self.fc1 = nn.Linear(128*2, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 4)
        
    def forward(self, x):
        
        # Hidden and cell state definion
        h = torch.zeros((2*self.LSTM_layers, x.size(0), self.hidden_dim), device=torch.device('cuda'))
        c = torch.zeros((2*self.LSTM_layers, x.size(0), self.hidden_dim), device=torch.device('cuda'))
        
        # Initialization of hidden and cell states: Xvaier Initialization
        torch.nn.init.xavier_normal_(h)
        torch.nn.init.xavier_normal_(c)

        out = self.embedding(x)
        out, (hidden, cell) = self.lstm(out, (h,c))
        out = F.relu(self.fc1(out[:,-1,:]))
        out = self.dropout(out)

        out = F.relu(self.fc2(out))
        out = self.dropout(out)
        
        logits = self.fc3(out)
        return logits