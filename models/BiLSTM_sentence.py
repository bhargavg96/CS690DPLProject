import torch
from models.BiLSTM import Attention_Layer
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# Condtional Bi-LSTM model with both Original Parent Post and Reesponse as Input
class BiLSTM_Sentence(torch.nn.Module):
    def __init__(self, input_dim,
                 hidden_size,
                 num_layers, num_classes,
                 attention = False,
                 sentence_attention = False,
                 bidir = True,
                 input_dim_sentence = None):
        super(BiLSTM_Sentence, self).__init__()
        self.attention = attention
        self.sentence_attention = sentence_attention
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm_response = torch.nn.LSTM(input_dim, hidden_size, num_layers, batch_first=True, bidirectional=True, dropout = 0.2)

        self.lstm_sentence = torch.nn.LSTM(input_dim_sentence,
                                               hidden_size, num_layers,
                                               batch_first=True, bidirectional=True, dropout = 0.2)
        if attention:
            print("response attention enabled")
            self.response_attention_layer = Attention_Layer(hidden_size, bidir).to(device)
        if sentence_attention:
            print("sentence attention enabled")
            self.sentence_attention_layer = Attention_Layer(hidden_size, bidir).to(device)
        self.fc1 = torch.nn.Linear(hidden_size*4, hidden_size*2)  # 2 for bidirection
        self.fc2 = torch.nn.Linear(hidden_size*2, num_classes)
        print("{} instantiated".format(self.__class__))

    def forward(self, x, X_post):
        h0 = torch.zeros(self.num_layers*2, x.size(0), self.hidden_size).to(device) # 2 for bidirection
        c0 = torch.zeros(self.num_layers*2, x.size(0), self.hidden_size).to(device)

        out_sentence, (hn, cn) = self.lstm_sentence(X_post, (h0, c0))
        out_response, _ = self.lstm_response(x, (hn, cn))

        if self.attention:
            out_response = self.response_attention_layer.forward(out_response)
        else:
            out_response = out_response[:, -1, :]
        if self.sentence_attention:
            out_sentence = self.sentence_attention_layer.forward(out_sentence)
        else:
            out_sentence = out_sentence[:, -1, :]
        out = torch.cat((out_sentence, out_response), dim = 1)

        out = self.fc1(out)
        out = self.fc2(out)
        return out

