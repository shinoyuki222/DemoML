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


def check_dir(path):
    return os.path.exists(path)

def remove_old_file(path):
    if check_dir(path):
        os.remove(path)

def save_obj(obj, filename):
    remove_old_file(filename)
    json.dump(obj, open(filename, 'w', encoding="utf8"), ensure_ascii=False)

def load_obj(filename):
    obj = json.load(open(filename, 'r', encoding="utf8"))
    return obj


def is_chinese(w):
    if (w >= '\u4e00' and w <= '\u9fa5' or w == ' '):
        return 1
    else:
        return 0


def unicodeToAscii(s):
    return ''.join(c for c in unicodedata.normalize('NFD', s) if unicodedata.category(c) != 'Mn')
    # return unicodedata.normalize('NFKD',s).encode('ascii','ignore')

def split(sent):
    return list(sent)

def filter(s):
    # if ' ' in s:
    #     print(s)
    return ''.join(c for c in s if is_chinese(c))
    # s = split(s)
    # for i,w in enumerate(s):
    #     if not is_chinese(w):
    #         s[i] = ' '
    # return ''.join(s)

def textprocess(s):
    s = unicodeToAscii(s)
    # s = filter(s)
    return s


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
