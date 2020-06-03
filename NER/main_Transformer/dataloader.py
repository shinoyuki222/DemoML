from const import *
import itertools
import os
import argparse
import unicodedata
import re
import json
import numpy as np
import torch
import copy
import pickle
from utils import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")




class Dictionary(object):
    def __init__(self):
        self.word2idx = {
            WORD[PAD]: PAD,
            WORD[UNK]: UNK,
            WORD[BOS]: BOS,
            WORD[EOS]: EOS
        }
        self.word2count = {}
        self.idx = len(self.word2idx)

    def addSents(self, sentences):
        for sent in sentences:
            for word in sent.split(' '):
                self.addWord(word)

    def addWord(self, word):
        if word not in self.word2count:
            self.word2count[word] = 1
        else:
            self.word2count[word] += 1

    def add2Dict(self, word):
        self.word2idx[word] = self.idx
        self.idx += 1

    def __len__(self):
        assert self.idx == len(self.word2idx)
        return self.idx

    def __call__(self, sentences, min_count=1):
        self.addSents(sentences)
        for k,v in self.word2count.items():
            if v >= min_count:
                self.add2Dict(k)

class Dict_lbl(object):
    def __init__(self):
        self.word2idx = {
            WORD[PAD]: PAD
        }
        self.word2count = {}
        self.idx = len(self.word2idx)

    def addSents(self, sentences):
        for sent in sentences:
            for word in sent.split(' '):
                self.addWord(word)

    def addWord(self, word):
        if word not in self.word2count:
            self.word2count[word] = 1
        else:
            self.word2count[word] += 1

    def add2Dict(self, word):
        self.word2idx[word] = self.idx
        self.idx += 1

    def __len__(self):
        assert self.idx == len(self.word2idx)
        return self.idx

    def __call__(self, sentences):
        self.addSents(sentences)
        for k,v in self.word2count.items():
            self.add2Dict(k)

class Dict_clsf(object):
    def __init__(self):
        self.word2idx = {}
        self.word2count = {}
        self.idx = len(self.word2idx)

    def addSents(self, sentences):
        for sent in sentences:
            self.addWord(sent)

    def addWord(self, word):
        if word not in self.word2count:
            self.word2count[word] = 1
        else:
            self.word2count[word] += 1

    def add2Dict(self, word):
        self.word2idx[word] = self.idx
        self.idx += 1

    def __len__(self):
        assert self.idx == len(self.word2idx)
        return self.idx

    def __call__(self, sentences):
        self.addSents(sentences)
        for k,v in self.word2count.items():
            self.add2Dict(k)

class Corpus(object):
    def __init__(self, corpus, w2v_file, save_dir, min_count=1, train = 1):
        self.corpus = corpus
        self.dict = Dictionary()
        self.dict_clsf = Dict_clsf()
        self.dict_lbl = Dict_lbl()
        self.sents = []
        self.clss = []
        self.lbls = []
        self.w2v_file = w2v_file
        self.min_count=1
        self.save_dir = save_dir
        self.max_len = 0
        if train:
            self.parse()
            self.load_w2v()
            self.save()


    def parse(self):
        lines = open(self.corpus, encoding='utf-8').read().strip().split('\n')

        for line in lines:
            segs = line.split('\t')
            if len(segs)<3:
                continue
            cls = segs[0]
            sent = segs[1]
            lbl = segs[2]
            sent = textprocess(sent)
            if len(sent.split(' ')) == len(lbl.split(' ')):
                self.sents.append(sent)
                self.clss.append(cls)
                self.lbls.append(lbl)
                self.max_len = max(self.max_len, len(sent.split(' ')))

        self.dict(self.sents)
        self.dict_clsf(self.clss)
        self.dict_lbl(self.lbls)

    def load_w2v(self):
        w2c_dict = {}
        lines = open(self.w2v_file, encoding='utf-8').read().strip().split('\n')

        for line in lines:
            segs = line.strip().split(" ")

            if len(segs) < 10:
                continue
            w2c_dict[segs[0]] = list(map(float, segs[1:]))
            
            if "dim" not in locals():
                dim = len(segs[1:])

        self.pre_w2v = np.random.rand(self.dict.idx, dim)

        for word, idx in sorted(self.dict.word2idx.items(), key=lambda x: x[1]):
            if word in w2c_dict:
                self.pre_w2v[idx] = np.asarray(w2c_dict[word])

    
    def save(self):

        save_obj(self.dict.word2idx, self.save_dir + "dict.json")
        save_obj(self.dict_clsf.word2idx, self.save_dir + "dict_clsf.json")
        save_obj(self.dict_lbl.word2idx, self.save_dir + "dict_lbl.json")
        torch.save(self.pre_w2v, self.save_dir + 'pre_w2v')
        
        print("There are {0} examples".format(len(self.sents)),flush=True)

        save_obj(self.sents, self.save_dir + "DataSentence.txt")
        save_obj(self.clss, self.save_dir + "DataClass.txt")
        save_obj(self.lbls, self.save_dir + "DataLabels.txt")



        print('Data saved.', flush=True)

        config = {}
        config['num_word'] = len(self.dict.word2idx)
        config['num_label'] = len(self.dict_lbl.word2idx)
        config['num_class'] = len(self.dict_clsf.word2idx)
        config['num_train'] = len(self.sents)
        config['max_len'] = self.max_len

        save_obj(config, self.save_dir + 'Config.json')





class DataLoader(object):
    def __init__(self, save_dir, batch_size=218):
        self.save_dir = save_dir

        self.dict = load_obj(self.save_dir + "dict.json")
        self.dict_clsf = load_obj(self.save_dir + "dict_clsf.json")
        self.dict_lbl = load_obj(self.save_dir + "dict_lbl.json")

        self.sents = load_obj(self.save_dir + "DataSentence.txt")
        self.clss = load_obj(self.save_dir + "DataClass.txt")
        self.lbls = load_obj(self.save_dir + "DataLabels.txt")
        self.config = load_obj(self.save_dir + 'Config.json')
        self.n_sents = len(self.sents)
        print("There are {0} examples".format(self.n_sents),flush=True)


        # self.n_sents = len(src_sents)
        self.batch_size = batch_size

        self.n_batch = self.n_sents//self.batch_size+1
        # self.n_batch = 2
        self.max_len = self.config['max_len']

        self.ds_loader = []

        # self._shuffle()
    def __call__(self):
        opt = self.gen_data()
        self.update_config()
        return self.gen_data()

    # def _shuffle(self):
    #     indices = np.arange(len(self.sents))
    #     np.random.shuffle(indices)
    #     print(type(self.sents))
    #     # print(indices.tolist())
    #     self.sents = self.sents[indices.tolist()]
    #     self.clss = self.clss[indices.tolist()]
    #     self.lbls = self.lbls[indices.tolist()]

    def tk2idx(self, dict_tks, tks):
        return [dict_tks[entity] for entity in tks.split(' ')]

    def EncoderVar(self, sents_batch):
        enc_batch = []
        indexes_batch = [self.tk2idx(self.dict, sentence) for sentence in sents_batch]
        for indexes in indexes_batch:
            enc_batch.append([BOS] + indexes)
        padList = zeroPadding(enc_batch)
        padVar = torch.LongTensor(padList)
        return padVar.transpose(0,1)

    def DecoderVar(self, lbls_batch):
        dec_batch = [self.tk2idx(self.dict_lbl, sentence) for sentence in lbls_batch]
        padList = zeroPadding(dec_batch)
        padVar = torch.LongTensor(padList)
        return padVar.transpose(0,1)

    def ClassVar(self, clss_batch):
        dec_batch = [ [self.dict_clsf[cls]]  for cls in clss_batch]
        dec_batch = torch.LongTensor(dec_batch)
        return dec_batch.squeeze(1)


    def batch2TrainData(self, sent_batch, lbls_batch, clss_batch):
        enc = self.EncoderVar(sent_batch)
        tgt = self.DecoderVar(lbls_batch)
        cls = self.ClassVar(clss_batch)
        return enc, tgt, cls

    def gen_data(self):
        sents = copy.deepcopy(self.sents)
        lbls = copy.deepcopy(self.lbls)
        clss = copy.deepcopy(self.clss)
        
        enc, tgt, cls = self.batch2TrainData(sents, lbls, clss)
        # shuffle
        indices = np.arange(len(self.sents))
        np.random.shuffle(indices)
        enc = enc[indices]
        tgt = tgt[indices]
        cls = cls[indices]

        for i in range(self.n_batch):
            idx_s = i*self.batch_size
            idx_e = (i+1)*self.batch_size

            sent_batch = enc[idx_s:idx_e,:]
            lbls_batch = tgt[idx_s:idx_e,:]
            clss_batch = cls[idx_s:idx_e]

            self.ds_loader.append((sent_batch, lbls_batch, clss_batch))
        return self.ds_loader



    def update_config(self):
        return 
        # self.config['max_len'] = self.max_len
        # save_obj(config, self.save_dir + 'Config.json')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Transformer NER')
    parser.add_argument('--corpus-data', type=str, default='../data/nav.txt',
                        help='path to corpus data')
    parser.add_argument('--save-dir', type=str, default='./data/',
                        help='path to save processed data')
    parser.add_argument('--pre-w2v', type=str, default='../data/w2v')
    args = parser.parse_args()

    corpus = Corpus(args.corpus_data, args.pre_w2v, args.save_dir)

    # # data = torch.load("data/vae_nlg.pt")
    # dl = DataLoader(args.save_dir)()

    # for sent, label,cls in dl:
    #     print(sent,label,cls)

    # a = data['dict']['src']['，']
    # print(a)
    # # print(is_chinese("。"))
    # # print(is_chinese("，"))




