__author__ = "bganguly@purdue.edu"
import torch
import numpy as np
import pandas as pd
import argparse
from models.BiLSTM import BiLSTM
from models.BiLSTM_sentence import BiLSTM_Sentence
from utils.preprocessor import *
from sklearn.metrics import f1_score, precision_recall_fscore_support
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#device = 'cpu'
#Tunable Hyper-params
sequence_length = 16
input_dim = 64
hidden_size = 256
num_layers = 2
num_classes = 2
input_dim_sentence = 32

def train_model(Xtr, Ytr, X_test, Y_test, batch_size, num_epochs, learning_rate, model,
                X_tr_post = None,
                X_ts_post = None):
    """

    :param Xtr: Training Reply Feature Tensor (|Train| x D) shape
    :param Ytr: Training Label Tensor (|Train|) shape
    :param X_test: Testing Feature Tensor (|Test| x D) shape
    :param Y_test: Testing Label Tensor (|Test|)
    :param batch_size: size of each batch
    :param num_epochs: number of epochs
    :param learning_rate:
    :param model: model choice = LSTM_conditional : BiLSTM_Sentence, LSTM_reply : BiLSTM
    :param X_tr_post: Training Original Reddit Post Feature Tensor
    :param X_ts_post: Testing Original Reddit Post Feature Tensor
    :return: Dataframe with performance metrics
    """
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    min_valid_epoch_loss = np.inf
    print("Xtr : {}, Ytr : {}, X_test : {}, Y_test : {}".format(Xtr.shape,
                                                                  Ytr.shape,
                                                                  X_test.shape,
                                                                  Y_test.shape))
    curr_epoch_loss_list = []
    valid_loss_list = []
    valid_acc_list = []
    valid_f1_score_list = []
    if X_ts_post is not None:
        X_test_post = X_ts_post
    else:
        X_test_post = None
    print("Training Begins: ...")
    for epoch in range(num_epochs):
        curr_epoch_loss = 0.
        indices = np.random.permutation(Xtr.shape[0])
        Xtr = Xtr[indices]
        Ytr = Ytr[indices]
        dataset_size = Xtr.shape[0]
        X_train = Xtr[:int(0.9*dataset_size)]
        Y_train = Ytr[:int(0.9*dataset_size)]
        X_valid = Xtr[int(0.9*dataset_size):]
        Y_valid = Ytr[int(0.9*dataset_size):]
        if X_tr_post is not None:
            X_tr_post = X_tr_post[indices]
            X_train_post = X_tr_post[:int(0.9*dataset_size)]
            X_valid_post = X_tr_post[int(0.9*dataset_size):]

        else:
            X_train_post = None
            X_valid_post = None

        num_batches = int(X_train.shape[0]/batch_size)
        count_data = 0
        for i in range(num_batches):
            optimizer.zero_grad()
            model.train()
            X_batch = X_train[i*batch_size:(i+1)*batch_size].reshape(-1,
                                                                 sequence_length,
                                                                 input_dim).to(device)
            if X_tr_post is not None:
                X_post_batch = X_train_post[i*batch_size:(i+1)*batch_size].reshape(-1,
                                                                                   sequence_length,
                                                                                   input_dim_sentence).to(device)
            Y_batch = Y_train[i*batch_size:(i+1)*batch_size].to(device)
            outputs = model(X_batch, X_post = X_post_batch)

            loss = criterion(outputs, Y_batch)

            loss.backward()

            optimizer.step()

            curr_epoch_loss += loss*batch_size*1.0
            count_data += batch_size
        curr_epoch_loss = curr_epoch_loss/batch_size
        curr_epoch_loss_list.append(curr_epoch_loss.detach().to('cpu').item())

        print("Training Ends for current epoch.")

        with torch.no_grad():
            X = X_valid.reshape(-1, sequence_length, input_dim).to(device)
            if X_tr_post is not None:
                X_post = X_valid_post.reshape(-1, sequence_length,
                                              input_dim_sentence).to(device)
            Y = Y_valid.to(device)
            outputs = model(X, X_post = X_post)
            fullbatch_loss = criterion(outputs, Y)
            _, predicted = torch.max(outputs.data, 1)
            total = X.shape[0]
            correct = (predicted == Y).sum().item()
            accuracy = (correct/total)*100.
            f1 = f1_score(Y.cpu().numpy(), predicted.cpu().numpy())

            if min_valid_epoch_loss > fullbatch_loss:
                best_model = model.state_dict()
                best_epoch = epoch
                min_valid_epoch_loss = fullbatch_loss
                torch.save(best_model, "data/best_model_{}.pth".format(model.__class__))

        print("Epoch Num: {}, Train Loss : {:2f}, Valid Loss : {:2f}, Valid Accuracy : {:2f}, Valid F1 Score : {:2f}".format(epoch,
                                                                                                                 curr_epoch_loss,
                                                                                                                 fullbatch_loss,
                                                                                                                 accuracy,
                                                                                                                 f1))
        valid_loss_list.append(fullbatch_loss.to('cpu').item())
        valid_acc_list.append(accuracy)
        valid_f1_score_list.append(f1)

    test_f1_score_list = [None]*(len(valid_f1_score_list)-1)
    test_loss_list = [None]*(len(valid_f1_score_list)-1)
    test_acc_list = [None]*(len(valid_f1_score_list)-1)

    print("Best model is at Epoch : {}".format(best_epoch))
    print("Testing Begins : ...")
    with torch.no_grad():
        model.load_state_dict(best_model)
        model = model.to(device)
        Xts = X_test.reshape(-1, sequence_length, input_dim).to(device)
        if X_ts_post is not None:
            Xts_post = X_test_post.reshape(-1, sequence_length, input_dim_sentence).to(device)
        Yts = Y_test.to(device)

        outputs = model(Xts, X_post = Xts_post).to(device)
        fullbatch_loss = criterion(outputs, Yts)
        _, predicted = torch.max(outputs.data, 1)
        total = Xts.shape[0]
        correct = (predicted == Yts).sum().item()
        accuracy = (correct/total)*100.
        f1 = f1_score(Yts.cpu().numpy(), predicted.cpu().numpy())
        print("Test Loss : {}, Test Acc : {}, Test F1 Score : {}".format(fullbatch_loss,
                                                                         accuracy,
                                                                         f1))
        test_acc_list.append(accuracy)
        test_f1_score_list.append(f1)
        test_loss_list.append(fullbatch_loss.to('cpu').item())
    print("Testing Ends.")

    print("Saving Results: .....")


    df = pd.DataFrame({'Train_Loss' : curr_epoch_loss_list,
                       'Valid_Loss' : valid_loss_list,
                       'Valid_Acc' : valid_acc_list,
                       'Valid_F1_Score' : valid_f1_score_list,
                       'Test_Loss' : test_loss_list,
                       'Test_Acc' : test_acc_list,
                       'Test F1 Score' : test_loss_list})
    print("df : columns : {}, shape : {}".format(df.columns, df.shape))

    return df




def model_test(X_test, Y_test, model,model_path, X_tr_post = None, X_ts_post = None):
    with torch.no_grad():

        model.load_state_dict(torch.load(model_path))
        model = model.to(device)
        Xts = X_test.reshape(-1, sequence_length, input_dim).to(device)
        Yts = Y_test.to(device)
        if X_ts_post is not None:
            Xts_post = X_ts_post.reshape(-1, sequence_length, input_dim_sentence).to(device)
        outputs = model(Xts, X_post = Xts_post).to(device)
        fullbatch_loss = criterion(outputs, Yts)
        _, predicted = torch.max(outputs.data, 1)
        total = Xts.shape[0]
        correct = (predicted == Yts).sum().item()
        accuracy = (correct/total)*100.
        f1 = f1_score(Yts.cpu().numpy(), predicted.cpu().numpy())
        print("Test Loss : {}, Test Acc : {}, Test F1 Score : {}".format(fullbatch_loss,
                                                                         accuracy,
                                                                         f1))










def _main(args):
    model = globals()[args.model](input_dim,
                                  hidden_size,
                                  num_layers,
                                  num_classes,
                                  attention = args.attention,
                                  sentence_attention = args.sentence_attention,
                                  input_dim_sentence = input_dim_sentence).to(device)
    if args.data_flag:
        if args.phase == "train":
            Xtr = torch.load(args.Xtr, map_location = device)
            train_dataset_size = min(args.dataset_size, Xtr.shape[0])
            Ytr = torch.load(args.Ytr, map_location = device)
            Xtr = Xtr[:train_dataset_size]
            Ytr = Ytr[:train_dataset_size]
        X_test = torch.load(args.Xts, map_location = device)[:args.test_datasize]
        Y_test = torch.load(args.Yts, map_location = device)[:args.test_datasize]
        if args.post:
            X_train_post = torch.load(args.Xtr_post, map_location = device)
            X_test_post = torch.load(args.Xts_post, map_location = device)[:args.test_datasize]
        else:
            X_train_post = None
            X_test_post = None


    else:
        if args.embedder == "ELMO":
            preproc = SARC_ELMO_Preprocessor().to(device)
        else:
            preproc = SARC_GLOVE_Preprocessor().to(device)

        Xtr,Ytr = preproc.build_dataset(dataset_size = args.dataset_size, phase = 'train')
        X_test, Y_test = preproc.build_dataset(dataset_size = args.dataset_size//10, phase = 'test')


    if args.phase == 'train':
        df = train_model(Xtr = Xtr, Ytr = Ytr,
                X_test = X_test, Y_test = Y_test,
                batch_size = args.batch_size,
                num_epochs = args.num_epochs,
                learning_rate = args.learning_rate,
                model = model,
                X_tr_post = X_train_post,
                X_ts_post = X_test_post)
        df.to_csv("data/model={}_attention={}_sentence_attention={}_results_glove.csv".format(model.__class__,
                                                                                        args.attention ,
                                                                                        args.sentence_attention))
        print("Results saved ....")
    else:
        model_test(X_test= X_test,
                   Y_test = Y_test,
                   model = model,
                   model_path=args.model_path,
                   X_tr_post = X_train_post,
                   X_ts_post = X_test_post)




if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--attention", default = False, type = bool)
    parser.add_argument("--data_flag", default = False, type = bool)
    parser.add_argument("--Xtr", default = None, type = str, help = "location of Xtr")
    parser.add_argument("--Ytr", default = None, type = str, help = "location of Ytr")
    parser.add_argument("--Xts", default = None, type = str, help = "location of X test")
    parser.add_argument("--Yts", default = None, type = str, help = "location of Y test")
    parser.add_argument("-b", "--batch_size", default=500, type=int)
    parser.add_argument("-n", "--num_epochs", default=20, type=int)
    parser.add_argument("-l", "--learning_rate", default=0.002, type=float)
    parser.add_argument("-sz","--dataset_size", default = 5000, type = int)
    parser.add_argument("--model",default = 'BiLSTM', choices = ['BiLSTM', 'BiLSTM_Sentence'], type = str)
    parser.add_argument("--embedder", default = 'ELMO', choices = ['ELMO', 'GLOVE'], type = str)
    parser.add_argument("--model_path", default = "", type = str)
    parser.add_argument("--phase", default = "train", choices = ["train", "test"] , type = str)
    parser.add_argument("--test_datasize", default = 10000, type = int)
    parser.add_argument("--post", default = False, type = bool)
    parser.add_argument("--Xtr_post", default = None, type = str)
    parser.add_argument("--Xts_post", default = None, type = str)
    parser.add_argument("--sentence_attention", default = False, type = bool)
    args = parser.parse_args()
    _main(args)