import torch
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#device = "cpu"
# Dot Product Based Attention Layer
class Attention_Layer(torch.nn.Module):
    def __init__(self, hidden_size, bidir = True):
        super(Attention_Layer, self).__init__()
        self.att_hidden_size = hidden_size*2 if bidir else hidden_size
        self.linear_layer = torch.nn.Linear(self.att_hidden_size, self.att_hidden_size)
        self.tanh_ = torch.nn.Tanh()
        self.softmax_ = torch.nn.Softmax(dim=1)

    def forward(self, x):
        out = self.tanh_(self.linear_layer(x))
        out = self.softmax_(out)
        out = torch.mul(x, out).sum(dim=1)
        return out


# Vanilla Bi-LSTM model with only response vector as input.

class BiLSTM(torch.nn.Module):
    def __init__(self, input_dim, hidden_size, num_layers, num_classes, attention = False, bidir = True, *args, **kwargs):
        super(BiLSTM, self).__init__()
        self.attention = attention
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = torch.nn.LSTM(input_dim, hidden_size, num_layers, batch_first=True, bidirectional=True, dropout = 0.2)
        if self.attention:
            print("Attention enabled")
            self.attention_layer = Attention_Layer(hidden_size, bidir).to(device)

        self.fc1 = torch.nn.Linear(hidden_size*2, hidden_size)  # 2 for bidirection
        self.fc2 = torch.nn.Linear(hidden_size, num_classes)
        print("{} instantiated".format(self.__class__))


    def forward(self, x, *args, **kwargs):
        # Set initial states
        h0 = torch.zeros(self.num_layers*2, x.size(0), self.hidden_size).to(device) # 2 for bidirection
        c0 = torch.zeros(self.num_layers*2, x.size(0), self.hidden_size).to(device)

        # Forward propagate LSTM
        out, _ = self.lstm(x, (h0, c0))  # out: tensor of shape (batch_size, seq_length, hidden_size*2)
        if self.attention:
            out = self.attention_layer.forward(out)
        else:
            # Decode the hidden state of the last time step
            out = out[:, -1, :]

        out = self.fc1(out)
        out = self.fc2(out)
        return out
