import torch
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class BiLSTM(torch.nn.Module):
    def __init__(self, input_dim, hidden_size, num_layers, num_classes):
        super(BiLSTM, self).__init__()
        print("bilstm instantiated")
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = torch.nn.LSTM(input_dim, hidden_size, num_layers, batch_first=True, bidirectional=True)
        self.fc1 = torch.nn.Linear(hidden_size*2, hidden_size)  # 2 for bidirection
        self.fc2 = torch.nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        # Set initial states
        h0 = torch.zeros(self.num_layers*2, x.size(0), self.hidden_size).to(device) # 2 for bidirection
        c0 = torch.zeros(self.num_layers*2, x.size(0), self.hidden_size).to(device)

        # Forward propagate LSTM
        out, _ = self.lstm(x, (h0, c0))  # out: tensor of shape (batch_size, seq_length, hidden_size*2)

        # Decode the hidden state of the last time step
        out = self.fc1(out[:, -1, :])
        out = self.fc2(out)
        return out
