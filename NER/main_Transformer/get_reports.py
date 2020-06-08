import argparse
from const import *
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import numpy as np

import os
from metrics import f1_score_merged
from metrics import classification_report

from sklearn.metrics import f1_score
from dataloader import DataLoader, Corpus, load_obj
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

from model import Transformer_Mix, get_attn_pad_mask

from evaluate import load_mask, softmax_mask, evaluate_f1
from metrics import get_entities
from utils import create_dir, remove_old_file

def generate_report_txt(model, dl_test, save_dir, criterion_clsf = nn.CrossEntropyLoss().to(device), criterion_tgt = nn.CrossEntropyLoss(ignore_index=PAD).to(device), verbose = False):
    loss_test = 0
    pred_tags = []
    true_tags = []

    pred_clss = []
    true_clss = []
    criterion_clsf = criterion_clsf
    criterion_tgt = criterion_tgt

    idx2lbl = load_obj(save_dir+'idx2lbl.json')
    idx2cls = load_obj(save_dir + "idx2cls.json")

    sents = load_obj(save_dir + "TestDataSentence.txt")
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
            pred_tags.append(pre[0:-pad_num].data.tolist())
            true_tags.append(true[0:-pad_num].data.tolist())

        pred_clss += cls_idx.tolist()
        true_clss += cls.tolist()
        
    print("Prediction completed", flush=True)

    lines_correct = []
    lines_intent_error = []
    lines_slot_error = []
    for idx in range(len(true_clss)):
        tokens = sents[idx].split(' ')
        true_lbls = []
        pred_lbls = []
        true_tags_idx = true_tags[idx]
        pred_tags_idx = pred_tags[idx]
        for t,p in zip(true_tags_idx,pred_tags_idx):
            true_lbls.append(idx2lbl[str(t)])
            pred_lbls.append(idx2lbl[str(p)])

        true_entities = get_entities(true_lbls)
        pred_entities = get_entities(pred_lbls)

        slots_true = []
        slots_pred = []

        for chunk_true, chunk_pred, cls_true, cls_pred in zip(true_entities,pred_entities, true_clss, pred_clss):
            tag, start, end = chunk_true[0], chunk_true[1], chunk_true[2]
            tok = ''.join(tokens[start:end+1])
            slot_true = '<{0}>: {1}'.format(tag, tok)
            slots_true.append(slot_true)

            tag, start, end = chunk_pred[0], chunk_pred[1], chunk_pred[2]
            tok = ''.join(tokens[start:end+1])
            slot_pred = '<{0}>: {1}'.format(tag, tok)
            slots_pred.append(slot_pred)
        
        intent_true = idx2cls[str(true_clss[idx])]
        intent_pred = idx2cls[str(pred_clss[idx])]

        line = "Sentence:{0:}\nExpect: \t{1}\t{2}\nPredict:\t{3}\t{4}\n".format(sents[idx],intent_true,slots_true,intent_pred,slots_pred)
        if intent_true != intent_pred:
            lines_intent_error.append(line)
        elif slots_true != slots_pred:
            lines_slot_error.append(line)
        else:
            lines_correct.append(line)


    correct_num = len(lines_correct)
    intent_w_num = len(lines_intent_error)
    slot_w_num = len(lines_slot_error)
    total_line = len(lines_intent_error) + len(lines_correct) + len(lines_slot_error)

    score1 = 'total line = {0}; Exact match = {1}, with intent fail = {2}, with slot fail = {3};'.format(total_line, correct_num, intent_w_num, slot_w_num)
    score2 = 'Accuracy = {0:.4f}'.format(correct_num/total_line)
    scores = [score1, score2]

    # saveing report
    print("Saving reports...", flush = True)
    report_dir = os.path.join(save_dir, 'reports','')
    create_dir(report_dir)

    remove_old_file(report_dir+'reports_correct.txt')
    remove_old_file(report_dir+'reports_intent_error.txt')
    remove_old_file(report_dir+'reports_slot_error.txt')
    remove_old_file(report_dir+'scores.txt')

    with open(report_dir+'reports_correct.txt', 'w', encoding ='utf-8') as f:
        for line in lines_correct:
            f.write("{0}".format(line+ '\n'))
    with open(report_dir+'reports_intent_error.txt', 'w', encoding ='utf-8') as f:
        for line in lines_intent_error:
            f.write("{0}".format(line+ '\n'))
    with open(report_dir+'reports_slot_error.txt', 'w', encoding ='utf-8') as f:
        for line in lines_slot_error:
            f.write("{0}".format(line+ '\n'))
    with open(report_dir+'scores.txt', 'w', encoding ='utf-8') as f:
        for line in scores:
            f.write("{0}".format(line+ '\n'))

    # return lines

if __name__ == '__main__':

    print('device = ', device, flush=True)
    parser = argparse.ArgumentParser(description='Transformer NER')
    parser.add_argument('--corpus-dir', type=str, default='../data/',
                        help='path to corpus dir')   
    parser.add_argument('-src','--corpus-name', type=str, default='data.txt',
                        help='path to corpus data')
    parser.add_argument('--save-dir', type=str, default='./data/',
                        help='path to save processed data')
    parser.add_argument('--pre-w2v', type=str, default='../data/w2v')
    args = parser.parse_args()

    args.corpus_data = args.corpus_dir + args.corpus_name
    corpus = Corpus(args.corpus_data, args.pre_w2v, args.save_dir, train_dev=0)
    dl = DataLoader(args.save_dir, batch_size = 128, train_dev=0)()
    # dl_train, dl_test = train_test_split(dl, test_size=0.33)
    pre_w2v = torch.load(args.save_dir + 'pre_w2v')
    pre_w2v = torch.Tensor(pre_w2v).to(device)
    

    model_ckpt = torch.load(os.path.join(args.save_dir, '{}.pyt'.format("Transformer_NER_best")),map_location=torch.device(device))
    config = load_obj(args.save_dir+'Config.json')
    model =Transformer_Mix(config, pre_w2v).to(device)
    model.load_state_dict(model_ckpt['model'])


    # pred_tags = []
    # true_tags = []

    # loss_epoch_test, f1_cls, f1_tgt, f1_tgt_merged = evaluate_f1(model, dl_test, args.save_dir, verbose=1)


    # print('test_cost = {0:6f}, f1_intent = {1:4f}, f1_slot = {2:4f}, f1_slot_merged = {3:4f}, f1_slot_no_mask = {4:4f}, f1_slot_merged_no_mask = {5:4f}'.format(loss_epoch_test, f1_cls, f1_tgt, f1_tgt_merged, f1_tgt_no_mask, f1_tgt_merged_no_mask), flush=True)


    generate_report_txt(model, dl, args.save_dir, verbose=1)