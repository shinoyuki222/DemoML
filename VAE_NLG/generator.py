import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.nn.init as init
import numpy as np
from math import log10,log,exp
from copy import copy
# from collections import queue

# from highway import Highway
from const import *

# ignore_indices = len(WORD)-1
ignore_indices = 0

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



def generator_greedy(model, args, max_len=30):

    z = torch.normal(0,1,size=(1,args.latent_dim))
    word_idx = torch.ones(1, 1, device=device, dtype=torch.long) * BOS
    poetry = ""
    length = 0
    hidden = None

    while True:
        input_sent = word_idx.expand(1,1).to(device)
        decoder_input = model.embed(input_sent)
        output, hidden = model.decode(decoder_input, z, hidden)
        prob = output.squeeze().data

        score, word_idx = torch.max(prob[ignore_indices:],dim=-1)
        word = args.idx2word[word_idx.item()+ ignore_indices]

        if word == WORD[EOS] or length == max_len:
            poetry += "。"
            break
        else:
            poetry += word
            length += 1

    return poetry[:-1] + "。"

class Node(object):
    def __init__(self, hidden, previous_toks, word_idx, log_prob, length):
        self.hidden = hidden
        self.previous_toks = previous_toks
        self.word_idx = word_idx
        self.log_prob = log_prob
        self.length = length

def score(log_prob,length):
    return exp(log_prob/2)/(length)

def generator_beam(model, args):
    num_sample_batch = args.num_sample_batch
    samples = []
    for i in range(num_sample_batch):
        samples += beam_search(model, args)
    return samples


def beam_search(model, args):
    beam_width = args.beam_width
    max_len = args.max_len
    z = torch.normal(0,1,size=(1,args.latent_dim))
    word_idx = torch.ones(1, 1, device=device, dtype=torch.long) * BOS
    poetry = []
    length = 0
    hidden = None
    log_prob = 0
    previous_toks = ""

    root = Node(hidden, previous_toks, word_idx, log_prob, length)
    q = []
    q.append(root)
    
    end_nodes = [] #最终节点的位置，用于回溯
    while q:
        candidates = []  #每一层的可能被拓展的节点，只需选取每个父节点的儿子节点中概率最大的k个即可
    
        for _ in range(len(q)):
            node = q.pop()
            # print(node.previous_toks)
            word_idx = node.word_idx
            
            # 搜索终止条件
            if word_idx == EOS or node.length >= max_len:
                end_nodes.append((score(node.log_prob,node.length), node.previous_toks))
                continue
            
            hidden = node.hidden
            input_sent = word_idx.expand(1,1).to(device)
            decoder_input = model.embed(input_sent)
            # z = torch.normal(0,1,size=(1,args.latent_dim))         
            output, hidden = model.decode(decoder_input, z, hidden)
            log_prob = output.squeeze().data

            log_prob, indices = log_prob[ignore_indices:].topk(beam_width) #选取某个父节点的儿子节点概率最大的k个
            for k in range(beam_width):
                word_index = indices[k].unsqueeze(0)+ ignore_indices
                previous_toks = node.previous_toks+args.idx2word[word_index.item()]
                log_p = log_prob[k].item()
                child = Node(hidden, previous_toks, word_index, node.log_prob + log_p, node.length + 1)
                candidates.append((score(child.log_prob,child.length), child))  #建立候选儿子节点，注意这里概率需要累计
                # print(child.previous_toks)
           
        candidates = sorted(candidates, key=lambda x:x[0], reverse=True) #候选节点排序

        # for i in range(len(candidates)):
        #     print(candidates[i][1].previous_toks,candidates[i][0])

        length = min(len(candidates), beam_width)  #取前k个，如果不足k个，则全部入选
        for i in range(length):
            q.append(candidates[i][1])

    # poetry = []
    # scores = []
    nodes_selected = []
    candidates_end = sorted(end_nodes, key=lambda x:x[0], reverse=True)
    length = min(len(candidates_end), beam_width)
    for i in range(length):
        nodes_selected.append(candidates_end[i])
        # scores.append(candidates_end[i][0])
        # poetry.append(candidates_end[i][1])
    
    return nodes_selected
