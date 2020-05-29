import argparse
from const import *
import torch
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
    pre_w2v = torch.Tensor(pre_w2v)
    for enc, tgt, cls in dl:
        enc_self_attn_mask = get_attn_pad_mask(enc, enc)
        a, b = Transformer_Mix(cls_size, tgt_size,pre_w2v)(enc,enc_self_attn_mask)
        print(a.size(), b.size())
        exit()
    # for sent, label,cls in dl:
    #     print(sent,label,cls)