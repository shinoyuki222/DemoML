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
from evaluate import evaluate_f1

from dataloader import DataLoader, Corpus, load_obj
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
from model import Transformer_Mix, get_attn_pad_mask

def split_data(data, test_size=0.33):
    dl_train, dl_test =train_test_split(data, test_size=test_size, random_state=42)
    return dl_train, dl_test
    # idx = len(data)*(1-test_size)
    # return data[:idx], data[idx:]

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

    dl = DataLoader(args.save_dir, batch_size = 256)()
    dl_train, dl_test = train_test_split(dl)
    pre_w2v = torch.load(args.save_dir + 'pre_w2v')
    pre_w2v = torch.Tensor(pre_w2v).to(device)

    model =Transformer_Mix(cls_size, tgt_size,pre_w2v).to(device)
    criterion_clsf = nn.CrossEntropyLoss().to(device)
    criterion_tgt = nn.CrossEntropyLoss(ignore_index=PAD).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(20):
        loss_epoch = 0
        model.train()
        for enc, tgt, cls in tqdm(dl_train[:], mininterval=1, desc='Generator Train Processing', leave=False):
            optimizer.zero_grad()
            enc = enc.to(device)
            tgt = tgt.to(device)
            cls = cls.to(device)
            enc_self_attn_mask = get_attn_pad_mask(enc, enc)
            enc_self_attn_mask.to(device)

            logits_tgt, logits_clsf = model(enc,enc_self_attn_mask)
            loss_tgt = criterion_tgt(logits_tgt.transpose(1, 2), tgt) # for masked LM
            loss_tgt = (loss_tgt.float()).mean()
            loss_clsf = criterion_clsf(logits_clsf, cls.squeeze(1))# for sentence classification
            loss = loss_tgt + loss_clsf
            loss_epoch+=loss
            loss.backward()
            optimizer.step()
            # print('batch_cost = {:0.6f}'.format(loss), flush=True)

        loss_epoch_test, f1_clsf, f1_tgt = evaluate_f1(model, dl_test)

        torch.save({
        'model': model.state_dict(),
        'model_opt': optimizer.state_dict()}, os.path.join(args.save_dir, '{}.ptn'.format("Transformer_NER")))

        print('Epoch:{0} , train_cost = {1:4f}, test_cost = {2:4f}, f1_intent = {3:4f}, f1_slot = {4:4f}'.format(epoch + 1, loss_epoch, loss_epoch_test, f1_clsf, f1_tgt), flush=True)        
