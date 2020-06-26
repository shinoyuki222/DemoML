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
from metrics import f1_score_merged
from metrics import classification_report

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

def load_mask(save_dir):
    config = load_obj(save_dir + "Config.json")
    num_label = config['num_label']
    idx2lbl = load_obj(save_dir + "idx2lbl.json")
    idx2cls  = load_obj(save_dir + "idx2cls.json")
    dict_lbl = load_obj(save_dir + "dict_lbl.json")
    dict_clsf  = load_obj(save_dir + "dict_clsf.json")
    lbl_mask = load_obj(save_dir + "lbl_mask.json")

    idx_mask = {}
    for intent in lbl_mask:
        valid_slot = lbl_mask[intent]
        mask = [0] * num_label
        for s in valid_slot:
            idx = dict_lbl[s]
            mask[idx] = 1
        idx_mask[dict_clsf[intent]] = torch.LongTensor(mask).to(device)
        # print(idx_mask)
    return idx_mask

def softmax_mask(logits_tgt, cls_idx, idx_mask):
    idx_mask_torch = torch.LongTensor(logits_tgt.size()).to(device)
    length, tgt_num = logits_tgt.size(1), logits_tgt.size(2)
    for i, pre in enumerate(cls_idx):
        idx_mask_torch[i] = idx_mask[pre.data.tolist()].expand(size=(length, tgt_num))

    scores_exp = torch.nn.Softmax(dim = -1)(logits_tgt).to(device)
    scores_exp = scores_exp * idx_mask_torch # this step masks
    return scores_exp.to(device)

def evaluate_f1_no_mask(model, dl_test, save_dir, criterion_clsf = nn.CrossEntropyLoss().to(device), criterion_tgt = nn.CrossEntropyLoss(ignore_index=PAD).to(device), verbose = False):
    loss_test = 0
    pred_tags = []
    true_tags = []

    pred_clss = []
    true_clss = []
    criterion_clsf = criterion_clsf
    criterion_tgt = criterion_tgt
    idx2lbl = load_obj(save_dir+'idx2lbl.json')
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
            loss = loss_clsf + loss_tgt
            # loss = loss_clsf
            loss_test+=loss

        pad_mask = enc.data.eq(0).sum(axis = 1)

        score_tgt, tgt_idx = torch.max(logits_tgt,dim = -1)
        score_cls, cls_idx = torch.max(logits_clsf, dim = -1)

        for pre, true, pad_num in zip(tgt_idx, tgt, pad_mask):
            pred_tags += pre[0:-pad_num].data.tolist()
            true_tags += true[0:-pad_num].data.tolist()

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
    # print(pred_clss[-20:])
    # print(true_clss[-20:])
    # print(pred_tags[-20:])
    # print(true_tags[-20:])

    # print(enc[-20:])

    f1_tgt = f1_score(pred_tags, true_tags, average='micro')
    f1_cls = f1_score(pred_clss, true_clss, average='micro')

    # logging loss, f1 and report

    metrics = {}
    true_lbls = []
    pred_lbls = []

    for t,p in zip(true_tags,pred_tags):
        true_lbls.append(idx2lbl[str(t)])
        pred_lbls.append(idx2lbl[str(p)])

    f1_tgt_merged = f1_score_merged(true_lbls, pred_lbls)

    if verbose:
        report = classification_report(true_lbls, pred_lbls)
        print("============no_mask_slot================")
        print(report, flush=True)

    return loss_test/len(dl_test), f1_cls*100, f1_tgt*100, f1_tgt_merged


def evaluate_f1(model, dl_test, save_dir, criterion_clsf = nn.CrossEntropyLoss().to(device), criterion_tgt = nn.CrossEntropyLoss(ignore_index=PAD).to(device), verbose = False):
    loss_test = 0
    pred_tags = []
    true_tags = []

    pred_clss = []
    true_clss = []
    criterion_clsf = criterion_clsf
    criterion_tgt = criterion_tgt

    idx2lbl = load_obj(save_dir+'idx2lbl.json')

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
            loss = loss_clsf + loss_tgt
            # loss = loss_clsf
            loss_test+=loss

        pad_mask = enc.data.eq(0).sum(axis = 1)

        score_cls, cls_idx = torch.max(logits_clsf, dim = -1)
        # get valid slot for a specific intent
        idx_mask = load_mask(save_dir)
        masked_logits_tgt= softmax_mask(logits_tgt, cls_idx, idx_mask)
        score_tgt, tgt_idx = torch.max(masked_logits_tgt ,dim = -1)
        

        for pre, true, pad_num in zip(tgt_idx, tgt, pad_mask):
            pred_tags += pre[0:-pad_num].data.tolist()
            true_tags += true[0:-pad_num].data.tolist()

        pred_clss += cls_idx.tolist()
        true_clss += cls.tolist()

        assert len(pred_tags) == len(true_tags)
        assert len(pred_clss) == len(true_clss)


    f1_tgt = f1_score(pred_tags, true_tags, average='micro')
    f1_cls = f1_score(pred_clss, true_clss, average='micro')

    # logging loss, f1 and report

    metrics = {}
    true_lbls = []
    pred_lbls = []

    for t,p in zip(true_tags,pred_tags):
        true_lbls.append(idx2lbl[str(t)])
        pred_lbls.append(idx2lbl[str(p)])

    f1_tgt_merged = f1_score_merged(true_lbls, pred_lbls)

    if verbose:
        report = classification_report(true_lbls, pred_lbls)
        print("============mask_slot================")
        print(report, flush=True)

    return loss_test/len(dl_test), f1_cls*100, f1_tgt*100, f1_tgt_merged



if __name__ == '__main__':

    print('device = ', device, flush=True)
    parser = argparse.ArgumentParser(description='Transformer NER')
    # parser.add_argument('--corpus-data', type=str, default='../data/auto_only-nav-distance_BOI.txt',
                        # help='path to corpus data')
    parser.add_argument('-src','--corpus-name', type=str, default='data_less.txt',
                        help='path to corpus data')
    parser.add_argument('--save-dir', type=str, default='./data/',
                        help='path to save processed data')

    parser.add_argument('--pre-w2v', type=str, default='../data/w2v')
    args = parser.parse_args()
    
    args.corpus_data = args.save_dir + args.corpus_name
    # corpus = Corpus(args.corpus_data, args.pre_w2v, args.save_dir)

    dl = DataLoader(args.save_dir, batch_size = 128)()
    dl_train, dl_test = train_test_split(dl, test_size=0.33)
    pre_w2v = torch.load(args.save_dir + 'pre_w2v')
    pre_w2v = torch.Tensor(pre_w2v).to(device)
    

    model_ckpt = torch.load(os.path.join(args.save_dir, '{}.pyt'.format("Transformer_NER_best")),map_location=torch.device(device))
    config = load_obj(args.save_dir+'Config.json')
    # cls_size = config['num_class']
    # tgt_size = config['num_label']
    model =Transformer_Mix(config, pre_w2v).to(device)
    model.load_state_dict (model_ckpt['model'])
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    optimizer.load_state_dict(model_ckpt['model_opt'])

    loss_epoch_test = 0
    pred_tags = []
    true_tags = []

    criterion_clsf = nn.CrossEntropyLoss().to(device)
    criterion_tgt = nn.CrossEntropyLoss(ignore_index=PAD).to(device)

    loss_epoch_test, f1_cls, f1_tgt, f1_tgt_merged = evaluate_f1(model, dl_test, args.save_dir, verbose=1)
    _, _, f1_tgt_no_mask, f1_tgt_merged_no_mask = evaluate_f1_no_mask(model, dl_test, args.save_dir, verbose=0)

    print('test_cost = {0:6f}, f1_intent = {1:4f}, f1_slot = {2:4f}, f1_slot_merged = {3:4f}, f1_slot_no_mask = {4:4f}, f1_slot_merged_no_mask = {5:4f}'.format(loss_epoch_test, f1_cls, f1_tgt, f1_tgt_merged, f1_tgt_no_mask, f1_tgt_merged_no_mask), flush=True)


    # generate_report_txt(model, dl_test, args.save_dir, verbose=1)

        
