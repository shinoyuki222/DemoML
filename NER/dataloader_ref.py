"""Data loader"""

import os
import torch
import argparse
import unicodedata
import re
import json
from consts import *
import itertools
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Voc:
    def __init__(self, name):
        self.name = name
        self.trimmed = False
        self.word2index = {}
        self.word2count = {}
        self.index2word = {PAD_token: "PAD", OOV_token: "UNK"}
        self.num_words = 2  # Count UNK, PAD

    def addSentence(self, sentence):
        for word in sentence.split(' '):
            self.addWord(word)

    def addWord(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.num_words
            self.word2count[word] = 1
            self.index2word[self.num_words] = word
            self.num_words += 1
        else:
            self.word2count[word] += 1

    # Remove words below a certain count threshold
    def trim(self, min_count):
        if self.trimmed:
            return
        self.trimmed = True

        keep_words = []

        for k, v in self.word2count.items():
            if v >= min_count:
                keep_words.append(k)

        print('keep_words {} / {} = {:.4f}'.format(len(keep_words), len(self.word2index),
                                                   len(keep_words) / len(self.word2index)))

        # Reinitialize dictionaries
        self.word2index = {}
        self.word2count = {}
        self.index2word = {PAD_token: "PAD", OOV_token: "UNK"}
        self.num_words = 2  # Count default tokens

        for word in keep_words:
            self.addWord(word)


class Tag:
    def __init__(self, name):
        self.name = name
        self.trimmed = False
        self.tag2index = {}
        self.index2tag = {PAD_token: "PAD"}
        self.num_tags = 1  # Count PAD

    def addSentence(self, sentence):
        for tag in sentence.split(' '):
            self.addTag(tag)

    def addTag(self, tag):
        if tag not in self.tag2index.keys():
            self.tag2index[tag] = self.num_tags
            self.index2tag[self.num_tags] = tag
            self.num_tags += 1

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

def save_static_dict(voc,tag,save_dir):
    corpus_name = voc.name
    _ = os.path.join(save_dir, corpus_name)
    directory = os.path.join(_, "Dicts")
    if not os.path.exists(directory):
        os.makedirs(directory)
    voc_dir = os.path.join(directory, 'voc.json')
    tag_dir = os.path.join(directory, 'tag.json')
    save_obj(voc.__dict__, voc_dir)
    save_obj(tag.__dict__, tag_dir)
    print("voc and tag saved for corpus {}".format(corpus_name))

def char2int(DCT):
    return {int(k):v for k,v in DCT.items()}

def load_static_dict(save_dir,corpus_name):
    _ = os.path.join(save_dir, corpus_name)
    directory = os.path.join(_, "Dicts")
    try:
        voc_dir = os.path.join(directory, 'voc.json')
        tag_dir = os.path.join(directory, 'tag.json')
        voc = Voc(corpus_name)
        tag = Tag(corpus_name)
        voc.__dict__ = load_obj(voc_dir)
        tag.__dict__ = load_obj(tag_dir)
        voc.index2word = char2int(voc.index2word)
        tag.index2tag = char2int(tag.index2tag)
    except KeyError:
        print("Not such dictionary")
    return voc, tag



def split(sent):
    return list(sent)
    # return(" ".join(jieba.cut(sent,HMM=True)).split(" "))

def unicodeToAscii(s):
    return ''.join(c for c in unicodedata.normalize('NFD', s) if unicodedata.category(c) != 'Mn')

# Turn a Unicode string to plain ASCII, thanks to
# https://stackoverflow.com/a/518232/2809427
def unicodeToAscii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
    )
# Lowercase, trim, and remove non-letter characters
def normalizeString(s):
    s = unicodeToAscii(s.lower().strip())
    s = re.sub(r"([.!?])", r" \1", s)
    # s = re.sub(r"[^a-zA-Z.!?]+", r" ", s)
    s = re.sub(r"\s+", r" ", s).strip()
    return s


def text_process(input_text):
    RE_R_ALPHABET = r'([a-zA-Z_\-]+)'
    RE_R_A_NUM = r'([0-9\.]+)'
    RE_R_C_NUM = r'([零一幺二两三四五六七八九十百千万]{2,})'
    RE_M_ALL = r'^[a-zA-Z_\-]+$|^[0-9\.]+$|^[零一幺二两三四五六七八九十百千万]{2,}$'
    # split alphabets and numbers
    text = unicodeToAscii(input_text)
    text = normalizeString(text)
    text = re.sub(RE_R_ALPHABET, r' \1 ', text)
    text = re.sub(RE_R_A_NUM, r' \1 ', text)
    text = re.sub(RE_R_C_NUM, r' \1 ', text)
    word_lst = text.split(" ")

    char_lst = list()
    char_tag_lst = list()

    for word in word_lst:
        if (re.match(RE_M_ALL, word)):
            char_lst.append(word)
        else:
            char_in_word = split(word)
            for char in char_in_word:
                char_lst.append(char)

    for char in char_lst:
        char_tag = char.lower()
        char_tag_lst.append(char_tag)

    char_text = " ".join(char_lst)
    char_tag_text = " ".join(char_tag_lst)

    return char_tag_text


def readLines(file):
    lines = open(file, encoding='utf-8').read().strip().split('\n')
    return lines


# Read query/response pairs and return a voc object,train time
def readVocs(datafile, corpus_name, split=True):
    print("Reading lines and generate vocs and tags...")
    # Read the file and split into lines
    pairs = []
    if split:
        sentfile = os.path.join(datafile, "sentences.txt")
        tagfile = os.path.join(datafile, "tags.txt")
        line_sents = readLines(sentfile)
        line_entities = readLines(tagfile)
        for sent, entity in zip(line_sents,line_entities):
            pairs.append([text_process(sent), entity])
    voc = Voc(corpus_name)
    tag = Tag(corpus_name)
    return voc, tag, pairs

# Read sentence and tag pair for dev and test time
def readPairs(datafile, split=True):
    print("Reading lines...")
    # Read the file and split into lines
    pairs = []
    if split:
        sentfile = os.path.join(datafile, "sentences.txt")
        tagfile = os.path.join(datafile, "tags.txt")
        line_sents = readLines(sentfile)
        line_entities = readLines(tagfile)
        for sent, entity in zip(line_sents,line_entities):
            pairs.append([text_process(sent), entity])
    return pairs

# Returns True iff both sentences in a pair 'p' are under the MAX_LENGTH threshold
def length_check(p):
    # Input sequences need to preserve the last word for EOS token
    return len(p[0].split(' ')) < MAX_LENGTH and len(p[1].split(' ')) < MAX_LENGTH

# Returns True iff tag in pair 'p' are valid
def tag_check(p, tag):
    for entity in p[1].split(' '):
        if entity not in tag.tag2index.keys():
            print("tag {0} not in training data".format(tag))
            return False
    return True

# Filter pairs using filterPair condition
def filterPairs(pairs, tag=None):
    if tag == None:
        return [pair for pair in pairs if length_check(pair)]
    # check if tag in tag_dict for dev and test corpus
    else:
        return [pair for pair in pairs if (length_check(pair) and tag_check(pair,tag))]


# Using the functions defined above, return a populated voc object and pairs list
def loadTrainData(corpus_name, datafile, split =True):
    print("Start preparing training data ...")
    voc, tag, pairs = readVocs(datafile, corpus_name, split =split)
    print("Read {!s} sentence pairs".format(len(pairs)))
    pairs = filterPairs(pairs)
    print("Trimmed to {!s} sentence pairs".format(len(pairs)))
    print("Counting words...")
    for pair in pairs:
        voc.addSentence(pair[0])
        tag.addSentence(pair[1])
    print("Counted words:", voc.num_words)
    return voc, tag, pairs

# Using the functions defined above, return pairs list
def loadDevData(datafile, tag, split =True):
    print("Start preparing dev data ...")
    pairs = readPairs(datafile, split =split)
    print("Read {!s} sentence pairs".format(len(pairs)))
    pairs = filterPairs(pairs, tag)
    print("Trimmed to {!s} sentence pairs".format(len(pairs)))
    print("Counting words...")
    return pairs


def trimRareWords(voc, pairs, MIN_COUNT):
    # Trim words used under the MIN_COUNT from the voc
    voc.trim(MIN_COUNT)
    # Filter out pairs with trimmed words
    keep_pairs = []
    for pair in pairs:
        input_sentence = pair[0]
        keep= True
        # Check input sentence
        for word in input_sentence.split(' '):
            if word not in voc.word2index:
                keep = False
                break
        # Only keep pairs that do not contain trimmed word(s) in their input or output sentence
        if keep:
            keep_pairs.append(pair)
    print("Trimmed from {} pairs to {}, {:.4f} of total".format(len(pairs), len(keep_pairs),len(keep_pairs) / len(pairs)))
    return voc, keep_pairs


def indexesFromSentence(voc, sentence):
    output = []
    for word in sentence.split(' '):
        if word in voc.word2index.keys():
            output.append(voc.word2index[word])
        else:
            output.append(OOV_token)
    return output

def indexesFromTag(tag, tag_seq):
    return [tag.tag2index[entity] for entity in tag_seq.split(' ')]


def zeroPadding(l, fillvalue=PAD_token):
    return list(itertools.zip_longest(*l, fillvalue=fillvalue))


def binaryMatrix(l, value=PAD_token):
    m = []
    for i, seq in enumerate(l):
        m.append([])
        for token in seq:
            if token == value:
                m[i].append(0)
            else:
                m[i].append(1)
    return m

# Returns padded input sequence tensor and lengths
def inputVar(l, voc):
    indexes_batch = [indexesFromSentence(voc, sentence) for sentence in l]
    lengths = torch.tensor([len(indexes) for indexes in indexes_batch])
    padList = zeroPadding(indexes_batch)
    padVar = torch.LongTensor(padList)
    return padVar, lengths


# Returns padded target sequence tensor, padding mask, and max target length
def outputVar(l, tag):
    indexes_batch = [indexesFromTag(tag, sentence) for sentence in l]
    max_target_len = max([len(indexes) for indexes in indexes_batch])
    padList = zeroPadding(indexes_batch)
    mask = binaryMatrix(padList)
    mask = torch.BoolTensor(mask)
    padVar = torch.LongTensor(padList)
    return padVar, mask, max_target_len


# Returns all items for a given batch of pairs
def batch2TrainData(voc, tag, pair_batch):
    pair_batch.sort(key=lambda x: len(x[0].split(" ")), reverse=True)
    input_batch, output_batch = [], []
    for pair in pair_batch:
        input_batch.append(pair[0])
        output_batch.append(pair[1])
    inp, lengths = inputVar(input_batch, voc)
    output, mask, max_target_len = outputVar(output_batch, tag)
    return inp, lengths, output, mask, max_target_len


def Set_DataLoader(voc, tag, pairs, batch_size = 64):
    total = len(pairs)
    n_batch = total//batch_size+1
    ds_loader = [batch2TrainData(voc,tag, pairs[i*batch_size:(i+1)*batch_size]) for i in range(n_batch)]
    return ds_loader


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--corpus', action='store', dest='corpus_name', default='MSRA',help='Store corpus name')
    args = parser.parse_args()

    save_dir = os.path.join("", "save")
    corpus_name = args.corpus_name
    corpus = os.path.join('..',"NER_data", corpus_name)
    datafile_train = os.path.join(corpus, "train")
    datafile_dev = os.path.join(corpus, "val")
    print("corpus_name: {0}, corpus = {1}, datafile_train = {2}".format(corpus_name, corpus, datafile_train))

    # Load/Assemble voc and pairs
    voc, tag, pairs = loadTrainData(corpus_name, datafile_train)
    save_static_dict(voc,tag,save_dir)
    # Print some pairs to validate
    print("\npairs:")
    for pair in pairs[:10]:
        print(pair,encoding='utf-8')
    # Trim voc and pairs
    voc, pairs = trimRareWords(voc, pairs, MIN_COUNT)




