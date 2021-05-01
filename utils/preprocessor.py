import torch
import numpy as np
import nltk
import os
from collections import defaultdict
import json
import csv
from allennlp.modules.elmo import Elmo, batch_to_ids
import time


filepath = os.path.abspath(__file__)
data_dir = os.path.abspath(os.path.join(filepath,'../..','data'))
elmo_1024dim_options_file = "elmo_2x4096_512_2048cnn_2xhighway_5.5B_options.json"
elmo_1024dim_weights_file = "elmo_2x4096_512_2048cnn_2xhighway_5.5B_weights.hdf5"
elmo_512dim_options_file = "elmo_2x2048_256_2048cnn_1xhighway_options.json"
elmo_512dim_weights_file = "elmo_2x2048_256_2048cnn_1xhighway_weights.hdf5"
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#device = 'cpu'
class SARC_Preprocessor():

    def __init__(self, post_flag = False):

        self.train_data_file = os.path.join(data_dir, 'main/train-balanced.csv')
        self.test_data_file = os.path.join(data_dir, 'main/test-balanced.csv')
        self.comments_file = os.path.join(data_dir,'main/comments.json')
        with open(self.comments_file, 'r') as f:
            self.comments = json.load(f)
        self.post_flag = post_flag

        print("{} __init__ is called".format(self.__class__))

    def prepare_data_lists(self, dataset_size = 1e4, phase = 'train'):
        count = 0
        self.post_list = []
        self.response_list = []
        self.label_list = []
        data_file = self.test_data_file if phase == "test" else self.train_data_file
        with open(data_file, 'r') as f:
            reader = csv.reader(f, delimiter='|')
            for row in reader:
                if count == dataset_size:
                    break
                count += 1
                post_key = [row[0].split(' ')[0]]
                response_key = row[1].split(' ')
                label = int(row[2].split(' ')[0])
                self.post_list.append([self.comments[k]['text'].lower() for k in post_key])
                self.response_list.append([self.comments[k]['text'].lower() for k in response_key])
                self.label_list.append(label)

    def prepare_vocabulary(self):
        self.vocab = defaultdict(int)
        sentence_list = self.post_list if self.post_flag else self.response_list
        for curr_response in sentence_list:
            for comment in curr_response:
                for word in nltk.word_tokenize(comment):
                    self.vocab[word] += 1

class SARC_GLOVE_Preprocessor(SARC_Preprocessor):
    def __init__(self, embedding_dim = 100, post_flag = False, *args, **kwargs):
        super(SARC_GLOVE_Preprocessor, self).__init__(post_flag)
        self.embedding_dim = embedding_dim
        self.embedding_file = os.path.join(data_dir, 'glove.6B.{}d.txt'.format(embedding_dim))

    def prepare_embedding_lookup(self):
        self.embedding_lookup_dict = {}
        with open(self.embedding_file, 'r') as f:
            while True:
                try:
                    x = next(f)
                except:
                    break
                try:
                    fields = x.strip().split()
                    word = fields[0]
                    if word not in self.vocab: continue
                    if word in self.embedding_lookup_dict:
                        print("Duplicate! ", word)
                    word_vector = np.array(fields[1:], dtype=np.float32)
                    self.embedding_lookup_dict[word] = word_vector
                except:
                    pass

    def build_dataset(self, dataset_size = 1e4, phase = 'train'):
        self.prepare_data_lists(dataset_size=dataset_size, phase = phase)
        self.prepare_vocabulary()
        self.prepare_embedding_lookup()

        X = torch.tensor([], device = device, requires_grad = False)
        unk = np.zeros(self.embedding_dim,)
        sentence_list = self.post_list if self.post_flag else self.response_list
        i = 0
        for curr_response in sentence_list:
            curr_response_feature = torch.tensor([], device = device, requires_grad = False)
            for comment in curr_response:
                words = nltk.word_tokenize(comment)
                words_embedding_avg = torch.tensor(np.average(np.array([self.embedding_lookup_dict[w] if w in self.embedding_lookup_dict else unk for w in words]),
                                                 axis = 0), device = device, requires_grad = False).reshape(self.embedding_dim, 1)

                curr_response_feature = torch.cat((curr_response_feature, words_embedding_avg), axis = 0)
                #curr_response_feature_vector.append(words_embedding_avg)

            X = torch.cat((X,torch.tensor(curr_response_feature.T, device = device)), axis = 0)
            i +=1
            if i%500 == 0:
                print(i)

        X = X.detach()
        Y = torch.tensor(self.label_list, requires_grad = False, device = device)

        return X, Y

class SARC_ELMO_Preprocessor(SARC_Preprocessor):
    def __init__(self, post_flag = False, *args, **kwargs):
        super(SARC_ELMO_Preprocessor, self).__init__(post_flag)
        self.elmo = Elmo(os.path.join(data_dir, elmo_512dim_options_file),
                         os.path.join(data_dir, elmo_512dim_weights_file),
                         num_output_representations=1,
                         dropout = 0,
                         requires_grad = False)
        self.elmo = self.elmo.to(device)

    def build_dataset(self, dataset_size = 1e4, phase = 'train'):
        self.prepare_data_lists(phase = phase, dataset_size= dataset_size)
        self.prepare_vocabulary()
        print("reach here")

        X = torch.tensor([], device = device, requires_grad = False)
        i = 0
        sentence_list = self.post_list if self.post_flag else self.response_list
        for curr_response in sentence_list:
            words_list = [nltk.word_tokenize(comment) for comment in curr_response]
            character_ids = batch_to_ids(words_list).to(device)
            #Following step calculates avg embedding of each comment sentence and concats them into 1 feature
            embedding_tensor_avg = torch.mean(self.elmo(character_ids)['elmo_representations'][0],
                                              axis = 1).reshape(-1)

            embedding_tensor_avg = embedding_tensor_avg.reshape(embedding_tensor_avg.shape[0], 1)
            X = torch.cat((X,embedding_tensor_avg), axis = 1)
            i +=1
            if i%500 == 0:
                print(i)
        X = X.T.detach()
        Y = torch.tensor(self.label_list, device = device, requires_grad = False)
        return X, Y











if __name__ == '__main__':
    timer_start = time.time()
    preproc = SARC_GLOVE_Preprocessor(embedding_dim=300, post_flag=True)
    Xtr,Ytr = preproc.build_dataset(dataset_size=100000, phase = 'test')
    print("X : {}".format(Xtr.shape))
    print("Y : {}".format(Ytr.shape))
    torch.save(Xtr,os.path.join(data_dir,"Xts_post_Glove_{}_preprocessed.pt".format(Xtr.shape[0])))
    torch.save(Ytr,os.path.join(data_dir,"Yts_post_Glove_{}_preprocessed.pt".format(Ytr.shape[0])))
    preproc_time = time.time() - timer_start
    print(f"Total time in pre-processing datapoints is {preproc_time:.1f} s")











