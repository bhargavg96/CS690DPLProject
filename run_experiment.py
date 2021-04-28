__author__ = "bganguly@purdue.edu"
import torch
import numpy as np
import pandas as pd
import argparse
from models.BiLSTM import BiLSTM
from utils.preprocessor import *
from sklearn.metrics import f1_score
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#Tunable Hyper-params
sequence_length = 16
input_dim = 64
hidden_size = 128
num_layers = 2
num_classes = 2

def train_model(Xtr, Ytr, X_valid, Y_valid, batch_size, num_epochs, learning_rate, model):
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    min_train_epoch_loss = np.inf
    print("Xtr : {}, Ytr : {}, X_valid : {}, Y_valid : {}".format(Xtr.shape,
                                                                  Ytr.shape,
                                                                  X_valid.shape,
                                                                  Y_valid.shape))
    for epoch in range(num_epochs):
        curr_epoch_loss = 0.
        indices = np.random.permutation(Xtr.shape[0])
        Xtr = Xtr[indices]
        Ytr = Ytr[indices]
        num_batches = int(Xtr.shape[0]/batch_size)
        for i in range(num_batches):
            optimizer.zero_grad()
            model.train()
            X_batch = Xtr[i*batch_size:(i+1)*batch_size].reshape(-1,
                                                                 sequence_length,
                                                                 input_dim).to(device)
            Y_batch = Ytr[i*batch_size:(i+1)*batch_size].to(device)
            outputs = model(X_batch)

            loss = criterion(outputs, Y_batch)

            loss.backward()

            optimizer.step()

            curr_epoch_loss += loss

        if min_train_epoch_loss > curr_epoch_loss:
            best_model = model.state_dict()
            torch.save(best_model, "best_model.pth")

        with torch.no_grad():
            indices = np.random.permutation(X_valid.shape[0])
            X_valid = X_valid[indices]
            Y_valid = Y_valid[indices]
            X = X_valid.reshape(-1, sequence_length, input_dim).to(device)
            Y = Y_valid.to(device)
            outputs = model(X)
            fullbatch_loss = criterion(outputs, Y)
            _, predicted = torch.max(outputs.data, 1)
            total = X.shape[0]
            correct = (predicted == Y).sum().item()
            accuracy = (correct/total)*100.
            f1 = f1_score(Y.cpu().numpy(), predicted.cpu().numpy())

        print("Epoch Num: {}, Train Loss : {}, Valid Loss : {}, Valid Accuracy : {}, Valid F1 Score : {}".format(epoch,
                                                                                                                 curr_epoch_loss,
                                                                           fullbatch_loss,
                                                                           accuracy,
                                                                           f1))









def _main(args):
    model = globals()[args.model](input_dim, hidden_size, num_layers, num_classes).to(device)
    if args.data_flag:
        Xtr = torch.load(args.Xtr)
        train_dataset_size = min(args.dataset_size, Xtr.shape[0])
        Ytr = torch.load(args.Ytr)
        Xtr = Xtr[:train_dataset_size]
        Ytr = Ytr[:train_dataset_size]
        X_valid = torch.load(args.Xvalid)
        valid_dataset_size = min(train_dataset_size//10, X_valid.shape[0])
        Y_valid = torch.load(args.Yvalid)
        X_valid = X_valid[:valid_dataset_size]
        Y_valid = Y_valid[:valid_dataset_size]

    else:
        if args.embedder == "ELMO":
            preproc = SARC_ELMO_Preprocessor()
        else:
            preproc = SARC_GLOVE_Preprocessor()

        Xtr,Ytr = preproc.build_dataset(dataset_size = args.dataset_size, phase = 'train')
        X_valid, Y_valid = preproc.build_dataset(dataset_size = args.dataset_size//10, phase = 'test')

    train_model(Xtr = Xtr, Ytr = Ytr,
                X_valid = X_valid, Y_valid = Y_valid,
                batch_size = args.batch_size,
                num_epochs = args.num_epochs,
                learning_rate = args.learning_rate,
                model = model)




if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_flag", default = False, type = bool)
    parser.add_argument("--Xtr", default = None, type = str, help = "location of Xtr")
    parser.add_argument("--Ytr", default = None, type = str, help = "location of Ytr")
    parser.add_argument("--Xvalid", default = None, type = str, help = "location of Xvalid")
    parser.add_argument("--Yvalid", default = None, type = str, help = "location of Yvalid")
    parser.add_argument("-b", "--batch_size", default=200, type=int)
    parser.add_argument("-n", "--num_epochs", default=40, type=int)
    parser.add_argument("-l", "--learning_rate", default=0.002, type=float)
    parser.add_argument("-sz","--dataset_size", default = 5000, type = int)
    parser.add_argument("--model",default = 'BiLSTM', choices = ['BiLSTM'], type = str)
    parser.add_argument("--embedder", default = 'ELMO', choices = ['ELMO', 'GLOVE'], type = str)
    args = parser.parse_args()
    _main(args)