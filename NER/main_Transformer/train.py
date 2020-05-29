import argparse
from const import *
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import numpy as np
import torch.optim as optim
from tqdm import tqdm

from dataloader import DataLoader, Corpus, load_obj
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
from model import Transformer_Mix, get_attn_pad_mask


if __name__ == '__main__':
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

    dl = DataLoader(args.save_dir)()
    pre_w2v = torch.load(args.save_dir + 'pre_w2v')
    pre_w2v = torch.Tensor(pre_w2v).to(device)

    model =Transformer_Mix(cls_size, tgt_size,pre_w2v).to(device)
    criterion_clsf = nn.CrossEntropyLoss().to(device)
    criterion_tgt = nn.CrossEntropyLoss(ignore_index=PAD).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    for epoch in range(20):
        loss_epoch = 0
        for enc, tgt, cls in tqdm(dl, mininterval=1, desc='Generator Train Processing', leave=False):
            optimizer.zero_grad()
            enc_self_attn_mask = get_attn_pad_mask(enc, enc)
            logits_tgt, logits_clsf = model(enc,enc_self_attn_mask)
            loss_tgt = criterion_tgt(logits_tgt.transpose(1, 2), tgt) # for masked LM
            loss_tgt = (loss_tgt.float()).mean()
            loss_clsf = criterion_clsf(logits_clsf, cls.squeeze(1))# for sentence classification
            loss = loss_tgt + loss_clsf
            loss_epoch+=loss
            loss.backward()
            optimizer.step()
            # print('batch_cost = {:0.6f}'.format(loss), flush=True)
        print('Epoch:{0} , cost = {:1.6f}'.format(epoch + 1, loss_epoch), flush=True)
        
    # for sent, label,cls in dl:
    #     print(sent,label,cls)