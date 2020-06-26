"""Data loader"""

import random
import numpy as np
import os
import sys

import torch

from pytorch_pretrained_bert import BertTokenizer
from sklearn.model_selection import train_test_split

from utils import *
from const import *

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
        self.word2idx = {}
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
    def __init__(self, corpus, save_dir, min_count=1, train = 1):
        self.corpus = corpus
        self.dict = Dictionary()
        self.dict_clsf = Dict_clsf()
        self.dict_lbl = Dict_lbl()
        self.sents = []
        self.clss = []
        self.lbls = []
        self.min_count=1
        self.save_dir = save_dir
        self.max_len = 0
        if train:
            self.parse()
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

    def split_data(self, test_size=0.33):
        sents_train, sents_test, lbls_train, lbls_test, clss_train, clss_test =train_test_split(self.sents, self.lbls, self.clss, test_size=test_size, random_state=42)
        for sent, label in zip(sents_train,lbls_train):
            if not len(sent.split(' ')) == len(label.split(' ')):
                print(sent, label)
        return sents_train, lbls_train, clss_train, sents_test, lbls_test, clss_test

    def save(self):

        # save_obj(self.dict.word2idx, self.save_dir + "dict.json")
        save_obj(self.dict_clsf.word2idx, self.save_dir + "dict_clsf.json")
        save_obj(self.dict_lbl.word2idx, self.save_dir + "dict_lbl.json")
        # torch.save(self.pre_w2v, self.save_dir + 'pre_w2v')
        
        print("There are {0} examples".format(len(self.sents)),flush=True)

        if not check_dir(self.save_dir):
            os.mkdir(self.save_dir)

        save_obj(self.sents, self.save_dir + "DataSentence.txt")
        save_obj(self.clss, self.save_dir + "DataClass.txt")
        save_obj(self.lbls, self.save_dir + "DataLabels.txt")

        sents_train, lbls_train, clss_train, sents_test, lbls_test, clss_test = self.split_data()

        if not check_dir(self.save_dir+'train'):
            os.mkdir(self.save_dir+'train')
        if not check_dir(self.save_dir+'dev'):
            os.mkdir(self.save_dir+'dev')

        sentences_file = os.path.join(self.save_dir, 'train','DataSentence.txt')
        tags_path = os.path.join(self.save_dir, 'train','DataLabels.txt')
        clss_path = os.path.join(self.save_dir, 'train','DataClass.txt')

        save_obj(sents_train, sentences_file)
        save_obj(lbls_train, tags_path)
        save_obj(clss_train, clss_path)

        sentences_file = os.path.join(self.save_dir, 'dev', 'DataSentence.txt')
        tags_path = os.path.join(self.save_dir, 'dev', 'DataLabels.txt')
        clss_path = os.path.join(self.save_dir, 'dev', 'DataClass.txt')

        save_obj(sents_test, sentences_file)
        save_obj(lbls_test, tags_path)
        save_obj(clss_test, clss_path)

        print('Data saved.', flush=True)



        config = {}
        config['num_word'] = len(self.dict.word2idx)
        config['num_label'] = len(self.dict_lbl.word2idx)
        config['num_class'] = len(self.dict_clsf.word2idx)
        config['num_train'] = len(self.sents)
        config['max_len'] = self.max_len

        save_obj(config, self.save_dir + 'Config.json')



class DataLoader(object):
    def __init__(self, data_dir, bert_model_dir, params, token_pad_idx=0):
        self.data_dir = data_dir
        self.batch_size = params.batch_size
        self.max_len = params.max_len
        self.device = params.device
        self.seed = params.seed
        self.token_pad_idx = 0

        # self.parse()
        tags = self.load_tags()
        self.tag2idx = {tag: idx for idx, tag in enumerate(tags)}
        self.idx2tag = {idx: tag for idx, tag in enumerate(tags)}
        params.tag2idx = self.tag2idx
        params.idx2tag = self.idx2tag
        self.tag_pad_idx = self.tag2idx["O"]

        self.tokenizer = BertTokenizer.from_pretrained(bert_model_dir, do_lower_case=True)
        # special_tokens_dict = {'unk_token': '<unk>'}

        # self.tokenizer.add_special_tokens(special_tokens_dict)

        # assert tokenizer.unk_token == '<unk>'

    def load_tags(self):
        # tags = []
        # file_path = os.path.join(self.data_dir, 'dict_lbl.json')
        # with open(file_path, 'r',encoding='utf-8') as file:
        #     for tag in file:
        #         tags.append(tag.strip())
        tags = load_obj(self.data_dir + "dict_lbl.json")
        return tags

    def load_sentences_tags(self, sentences_file, tags_file, d):
        """Loads sentences and tags from their corresponding files. 
            Maps tokens and tags to their indices and stores them in the provided dict d.
        """
        sentences = []
        tags = []

        sentences_raw = load_obj(sentences_file)
        for line in sentences_raw:
            # replace each token by its index
            tokens = line.strip().split(' ')

            vocab = self.tokenizer.vocab
            ids = []
            for token in tokens:
                if token in vocab:
                    ids.append(vocab[token])
                else:
                    ids.append(vocab[WORD[UNK]])
            sentences.append(ids)
        
        tags_raw = load_obj(tags_file)
        for line in tags_raw:
            # replace each tag by its index
            tag_seq = [self.tag2idx.get(tag) for tag in line.strip().split(' ')]
            tags.append(tag_seq)

        # checks to ensure there is a tag for each token
        assert len(sentences) == len(tags)
        for i in range(len(sentences)):
            if not len(tags[i]) == len(sentences[i]):
                print(i, len(tags[i]),len(sentences[i]))
                print(i, tags_raw[i] , sentences_raw[i])
                print(i, tags[i] , sentences[i])
            assert len(tags[i]) == len(sentences[i])

        # storing sentences and tags in dict d
        d['data'] = sentences
        d['tags'] = tags
        d['size'] = len(sentences)

    def load_data(self, data_type):
        """Loads the data for each type in types from data_dir.

        Args:
            data_type: (str) has one of 'train', 'val', 'test' depending on which data is required.
        Returns:
            data: (dict) contains the data with tags for each type in types.
        """
        data = {}
        
        if data_type in ['train', 'dev', 'test']:
            sentences_file = os.path.join(self.data_dir, data_type,'DataSentence.txt')
            tags_path = os.path.join(self.data_dir, data_type,'DataLabels.txt')
            self.load_sentences_tags(sentences_file, tags_path, data)
        else:
            raise ValueError("data type not in ['train', 'dev', 'test']")
        return data

    def data_iterator(self, data, shuffle=False):
        """Returns a generator that yields batches data with tags.

        Args:
            data: (dict) contains data which has keys 'data', 'tags' and 'size'
            shuffle: (bool) whether the data should be shuffled
            
        Yields:
            batch_data: (tensor) shape: (batch_size, max_len)
            batch_tags: (tensor) shape: (batch_size, max_len)
        """

        # make a list that decides the order in which we go over the data- this avoids explicit shuffling of data
        order = list(range(data['size']))
        if shuffle:
            random.seed(self.seed)
            random.shuffle(order)

        # one pass over data
        for i in range(data['size']//self.batch_size):
            # fetch sentences and tags
            sentences = [data['data'][idx] for idx in order[i*self.batch_size:(i+1)*self.batch_size]]
            tags = [data['tags'][idx] for idx in order[i*self.batch_size:(i+1)*self.batch_size]]

            # batch length
            batch_len = len(sentences)

            # compute length of longest sentence in batch
            batch_max_len = max([len(s) for s in sentences])
            max_len = min(batch_max_len, self.max_len)

            # prepare a numpy array with the data, initialising the data with pad_idx
            batch_data = self.token_pad_idx * np.ones((batch_len, max_len))
            batch_tags = self.tag_pad_idx * np.ones((batch_len, max_len))

            # copy the data to the numpy array
            for j in range(batch_len):
                cur_len = len(sentences[j])
                if cur_len <= max_len:
                    batch_data[j][:cur_len] = sentences[j]
                    batch_tags[j][:cur_len] = tags[j]
                else:
                    batch_data[j] = sentences[j][:max_len]
                    batch_tags[j] = tags[j][:max_len]

            # since all data are indices, we convert them to torch LongTensors
            batch_data = torch.tensor(batch_data, dtype=torch.long)
            batch_tags = torch.tensor(batch_tags, dtype=torch.long)

            # shift tensors to GPU if available
            batch_data, batch_tags = batch_data.to(self.device), batch_tags.to(self.device)
    
            yield batch_data, batch_tags

