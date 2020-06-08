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

from utils import remove_old_file

from dataloader import DataLoader, Corpus, load_obj
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
from model import Transformer_Mix, get_attn_pad_mask

def split_data(data, test_size=0.33):
    dl_train, dl_test =train_test_split(data, test_size=test_size, random_state=42)
    return dl_train, dl_test
    # idx = len(data)*(1-test_size)
    # return data[:idx], data[idx:]

def train_iter(model, dl_train, optimizer, criterion_clsf = nn.CrossEntropyLoss().to(device), criterion_tgt = nn.CrossEntropyLoss(ignore_index=PAD).to(device)):
    model.train()
    loss_epoch = 0

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
        loss_clsf = criterion_clsf(logits_clsf, cls)# for sentence classification
        loss = loss_tgt + loss_clsf
        loss_epoch+=loss
        loss.backward()
        # nn.utils.clip_grad_norm_(parameters=model.parameters(), max_norm=0.25)
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.25)
        optimizer.step()
        # flag += 1
        # if flag >2:
        #     break 
    return loss_epoch/len(dl_train)

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
    parser.add_argument('-e','--epoch', type=int, default=2)
    args = parser.parse_args()

    args.corpus_data = args.corpus_dir + args.corpus_name

    corpus = Corpus(args.corpus_data, args.pre_w2v, args.save_dir)

    dl = DataLoader(args.save_dir, batch_size = 256)()
    dl_train, dl_test = split_data(dl)
    pre_w2v = torch.load(args.save_dir + 'pre_w2v')
    pre_w2v = torch.Tensor(pre_w2v).to(device)


    config = load_obj(args.save_dir+'Config.json')
    # cls_size = config['num_class']
    # tgt_size = config['num_label']
    model =Transformer_Mix(config, pre_w2v).to(device)
    criterion_clsf = nn.CrossEntropyLoss().to(device)
    criterion_tgt = nn.CrossEntropyLoss(ignore_index=PAD).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    
    idx2lbl = load_obj(args.save_dir+'idx2lbl.json')
    best_f1 = 0

    log_file = 'train.log'
    remove_old_file(args.save_dir+log_file)
    patience = 0
    for epoch in range(args.epoch):
        loss_epoch = train_iter(model, dl_train[:], optimizer, criterion_clsf, criterion_tgt)

        loss_epoch_test, f1_clsf, f1_tgt, f1_slot_merged = evaluate_f1(model, dl_test[:], args.save_dir)
        if epoch%10 ==0:
            torch.save({
            'model': model.state_dict(),
            'model_opt': optimizer.state_dict()}, os.path.join(args.save_dir, '{0}_{1}.pyt'.format("Transformer_NER", epoch)))

        if f1_slot_merged > best_f1:
            best_f1 = f1_slot_merged
            print("f1_slot_merged {0} better than previous score {1}".format(f1_slot_merged, best_f1), flush=True)
            torch.save({
            'model': model.state_dict(),
            'model_opt': optimizer.state_dict()}, os.path.join(args.save_dir, '{0}.pyt'.format("Transformer_NER_best")))
            patience = 0
        else:
            if patience  == 5:
                exit()
            patience += 1

        log = 'Epoch:{0} , train_cost = {1:4f}, test_cost = {2:4f}, f1_intent = {3:4f}, f1_slot = {4:4f}, f1_slot_merged = {5:4f}'.format(epoch + 1, loss_epoch, loss_epoch_test, f1_clsf, f1_tgt, f1_slot_merged)
        f = open(args.save_dir+log_file, 'a')
        f.write(log + '\n')
        f.close()
        print(log, flush=True)    
