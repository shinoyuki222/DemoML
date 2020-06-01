import argparse
from const import *
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import numpy as np
import torch.optim as optim
from tqdm import tqdm

import os

from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from dataloader import DataLoader, Corpus, load_obj
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
from model import Transformer_Mix, get_attn_pad_mask

def split_data(data, test_size=0.33):
    dl_train, dl_test =train_test_split(data, test_size=test_size, random_state=42)
    return dl_train, dl_test
    # idx = len(data)*(1-test_size)
    # return data[:idx], data[idx:]

def evaluate_f1(model, dl_test, criterion_clsf = nn.CrossEntropyLoss().to(device), criterion_tgt = nn.CrossEntropyLoss(ignore_index=PAD).to(device)):
    loss_test = 0
    pred_tags = []
    true_tags = []

    pred_clss = []
    true_clss = []
    criterion_clsf = criterion_clsf
    criterion_tgt = criterion_tgt

    for enc, tgt, cls in dl_test[:]:
        model.eval()
        with torch.no_grad():
            enc = enc.to(device)
            tgt = tgt.to(device)
            cls = cls.to(device)
            enc_self_attn_mask = get_attn_pad_mask(enc, enc)
            enc_self_attn_mask.to(device)

            logits_tgt, logits_clsf = model(enc,enc_self_attn_mask)
            loss_tgt = criterion_tgt(logits_tgt.transpose(1, 2), tgt) # for masked LM
            loss_tgt = (loss_tgt.float()).mean()
            loss_clsf = criterion_clsf(logits_clsf, cls)# for sentence classification
            loss = loss_tgt + loss_clsf
            loss_test+=loss

        pad_mask = enc.data.eq(0).sum(axis = 1)

        score_tgt, tgt_idx = torch.max(logits_tgt,dim = -1)
        score_cls, cls_idx = torch.max(logits_clsf, dim = -1)

        for pre, true, pad_num in zip(tgt_idx, tgt, pad_mask):
            pred_tags += pre[1:-pad_num].data.tolist()
            true_tags += true[1:-pad_num].data.tolist()

        # print(cls_idx.size())
        pred_clss += cls_idx.tolist()
        true_clss += cls.tolist()
        # print(len(pred_tags), len(true_tags))
        # print(pred_tags)
        # print(true_tags)
        # print(len(pred_clss), len(true_clss))
        # print(pred_clss)

        # print(true_clss)
        assert len(pred_tags) == len(true_tags)
        assert len(pred_clss) == len(true_clss)
    f1_tgt = f1_score(pred_tags, true_tags, average='macro')
    f1_cls = f1_score(pred_clss, true_clss, average='macro')
    return loss_test/len(dl_test), f1_cls, f1_tgt

if __name__ == '__main__':

    print('device = ', device, flush=True)
    parser = argparse.ArgumentParser(description='Transformer NER')
    parser.add_argument('--corpus-data', type=str, default='../data/auto_only-nav-distance_BOI.txt',
                        help='path to corpus data')
    parser.add_argument('--save-dir', type=str, default='./data/',
                        help='path to save processed data')
    parser.add_argument('--pre-w2v', type=str, default='../data/w2v')
    args = parser.parse_args()

    config = load_obj(args.save_dir+'Config.json')
    cls_size = config['num_class']
    tgt_size = config['num_label']
    # corpus = Corpus(args.corpus_data, args.pre_w2v, args.save_dir)

    dl = DataLoader(args.save_dir, batch_size = 128)()
    dl_train, dl_test = train_test_split(dl)
    pre_w2v = torch.load(args.save_dir + 'pre_w2v')
    pre_w2v = torch.Tensor(pre_w2v).to(device)

    model_ckpt = torch.load(os.path.join(args.save_dir, '{}.ptn'.format("Transformer_NER")))
    model =Transformer_Mix(cls_size, tgt_size, pre_w2v).to(device)
    model.load_state_dict (model_ckpt['model'])
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    optimizer.load_state_dict(model_ckpt['model_opt'])

    loss_epoch_test = 0
    pred_tags = []
    true_tags = []

    criterion_clsf = nn.CrossEntropyLoss().to(device)
    criterion_tgt = nn.CrossEntropyLoss(ignore_index=PAD).to(device)

    loss_epoch_test, f1_cls, f1_tgt = evaluate_f1(model, dl_test)
    print('test_cost = {0:6f}, f1_intent = {1:4f}, f1_slot = {2:4f}'.format(loss_epoch_test, f1_cls, f1_tgt), flush=True)

        
