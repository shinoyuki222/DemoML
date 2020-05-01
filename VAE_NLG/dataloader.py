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

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def unicodeToAscii(s):
    return ''.join(c for c in unicodedata.normalize('NFD', s) if unicodedata.category(c) != 'Mn')
    # return unicodedata.normalize('NFKD',s).encode('ascii','ignore')
def textprocess(s):
    return unicodeToAscii(s)

def word2idx(sents, word2idx):
    return [[word2idx[w] if w in word2idx else UNK for w in s] for s in sents]

def zeroPadding(l, fillvalue=PAD):
    return list(itertools.zip_longest(*l, fillvalue=fillvalue))

def binaryMatrix(l, value=PAD):
    m = []
    for i, seq in enumerate(l):
        m.append([])
        for token in seq:
            if token == value:
                m[i].append(0)
            else:
                m[i].append(1)
    return m   

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
            for word in list(sent):
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


class Corpus(object):
    def __init__(self, corpus, w2v_file, save_data, min_count=1, train = 1):
        self.corpus = corpus
        self.dict = Dictionary()
        self.sents = []
        self.w2v_file = w2v_file
        self.min_count=1
        self.save_data = save_data
        if train:
            self.parse()
            self.load_w2v()
            self.save()


    def parse(self):
        lines = open(self.corpus, encoding='utf-8').read().strip().split('\n')
        for line in lines:
            self.sents.append(textprocess(line))
        # self.sents = lines
        self.dict(self.sents)

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
        data = {
            'pre_w2v': self.pre_w2v,
            # 'max_word_len': self._max_len,
            'dict': {
                'src': self.dict.word2idx,
                'src_size': len(self.dict),
            },
            'train': word2idx(self.sents, self.dict.word2idx)
        }

        torch.save(data, self.save_data)
        print('word length - [{}]'.format(len(self.dict)))



class DataLoader(object):
    def __init__(self, src_sents, max_len=30, batch_size=16):
        self.n_sents = len(src_sents)
        self.batch_size = batch_size

        self.n_batch = self.n_sents//self.batch_size+1
        self.n_batch = 2
        self.max_len = max_len
        self.enc_sents = src_sents
        self.ds_loader = []

        # self._shuffle()
        self.gen_data()

    # def _shuffle(self):
    #     indices = np.arange(len(self.enc_sents))
    #     np.random.shuffle(indices)
    #     print(indices)
    #     # indices = list(indices)
    #     self.enc_sents = self.enc_sents[indices]

    def EncoderVar(self, sents_batch):
        lengths = torch.tensor([len(indexes) for indexes in sents_batch])
        padList = zeroPadding(sents_batch)
        padVar = torch.LongTensor(padList)
        return padVar, lengths

    def DecoderVar(self, sents_batch):
        dec_batch = []
        for indexes in sents_batch:
            dec_batch.append([BOS] + indexes)
        padList = zeroPadding(dec_batch)
        padVar = torch.LongTensor(padList)
        # bos_tag = torch.LongTensor([[BOS for _ in range(self.batch_size)]])
        # pad_0 = torch.LongTensor(padList)
        # padVar = torch.cat((bos_tag,pad_0),0)
        return padVar

    def OutputVar(self, sents_batch):
        opt_batch = []
        max_target_len = 0
        for indexes in sents_batch:
            opt_batch.append(indexes+[EOS])
            max_target_len = max(max_target_len,len(indexes)+1)
        # max_target_len = max([len(indexes)+1 for indexes in sents_batch])
        padList = zeroPadding(opt_batch)
        mask = binaryMatrix(padList)
        mask = torch.BoolTensor(mask)
        padVar = torch.LongTensor(padList)
        return padVar, mask, max_target_len


    def batch2TrainData(self,sent_batch):
        enc, lengths = self.EncoderVar(sent_batch)
        dec = self.DecoderVar(sent_batch)
        opt, mask, max_target_len= self.OutputVar(sent_batch)
        return enc, lengths, dec, opt, mask

    def gen_data(self):
        sents = copy.deepcopy(self.enc_sents)
        batched_var = []
        for i in range(self.n_batch):
            idx_s = i*self.batch_size
            idx_e = (i+1)*self.batch_size

            sent_batch =  sorted(sents[idx_s:idx_e],key = lambda i:len(i),reverse=True)
            # print(sent_batch)
            # for i in sent_batch:
            #     print(len(i))
            self.ds_loader.append(self.batch2TrainData(sent_batch))
        return self.ds_loader




if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='VAE NLG')
    parser.add_argument('--corpus-data', type=str, default='data/songci',
                        help='path to corpus data')
    parser.add_argument('--save-data', type=str, default='data/vae_nlg.pt',
                        help='path to save processed data')
    parser.add_argument('--pre-w2v', type=str, default='data/w2v')
    args = parser.parse_args()

    corpus = Corpus(args.corpus_data, args.pre_w2v, args.save_data)

    data = torch.load("data/vae_nlg.pt")
    dl = DataLoader(data['train'])

    # a = data['dict']['src']['，']
    # print(a)



