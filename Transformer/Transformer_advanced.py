import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import numpy as np


sentences = ['我 喜欢 机器 学习 P P P', 'S i like machine learning P P P', 'i like machine learning E P P P']

# Transformer Parameters
# Padding Should be Zero index
src_vocab = {'P' : 0, '我' : 1, '喜欢' : 2, '机器' : 3, '学习' : 4}
src_vocab_size = len(src_vocab)

tgt_vocab = {'P' : 0, 'i' : 1, 'like' : 2, 'machine' : 3, 'learning' : 4, 'S' : 5, 'E' : 6}
number_dict = {i: w for i, w in enumerate(tgt_vocab)}
tgt_vocab_size = len(tgt_vocab)

src_len = 4
tgt_len = 5

d_model = 512  # Embedding Size
d_ff = 2048 # FeedForward dimension
d_k = d_v = 64  # dimension of K(=Q), V
n_layers = 6  # number of Encoder of Decoder Layer
n_heads = 4  # number of heads in Multi-Head Attention

def make_batch(sentences):
    input_batch = [[src_vocab[n] for n in sentences[0].split()]]
    output_batch = [[tgt_vocab[n] for n in sentences[1].split()]]
    target_batch = [[tgt_vocab[n] for n in sentences[2].split()]]
    return torch.LongTensor(input_batch), torch.LongTensor(output_batch), torch.LongTensor(target_batch)


def get_attn_pad_mask(seq_q, seq_k):
    # print(seq_q)
    batch_size, len_q = seq_q.size()
    batch_size, len_k = seq_k.size()
    # eq(zero) is PAD token
    pad_attn_mask = seq_k.data.eq(0).unsqueeze(1)  # (batch_size, 1, len_k) one is masking
    return pad_attn_mask.expand(batch_size, len_q, len_k)  # (batch_size, len_q, len_k)

def get_attn_subsequent_mask(seq):
    attn_shape = [seq.size(0), seq.size(1), seq.size(1)]
    subsequent_mask = np.triu(np.ones(attn_shape), k=1)
    subsequent_mask = torch.from_numpy(subsequent_mask).byte()
    return subsequent_mask


def get_mask_encoder(enc_inputs):
    enc_self_attn_mask = get_attn_pad_mask(enc_inputs, enc_inputs)
    return enc_self_attn_mask

def get_mask_decoder(dec_inputs, enc_inputs):
    dec_self_attn_pad_mask = get_attn_pad_mask(dec_inputs, dec_inputs)
    dec_self_attn_subsequent_mask = get_attn_subsequent_mask(dec_inputs)
    dec_self_attn_mask = torch.gt((dec_self_attn_pad_mask + dec_self_attn_subsequent_mask), 0)
    dec_enc_attn_mask = get_attn_pad_mask(dec_inputs, enc_inputs)
    return dec_self_attn_mask, dec_enc_attn_mask

def get_mask(enc_inputs, dec_inputs):
    





enc_inputs, dec_inputs, target_batch = make_batch(sentences)

enc_self_attn_mask = get_attn_pad_mask(enc_inputs, enc_inputs)
dec_self_attn_pad_mask = get_attn_pad_mask(dec_inputs, dec_inputs)
dec_self_attn_subsequent_mask = get_attn_subsequent_mask(dec_inputs)
dec_self_attn_mask = torch.gt((dec_self_attn_pad_mask + dec_self_attn_subsequent_mask), 0)
dec_enc_attn_mask = get_attn_pad_mask(dec_inputs, enc_inputs)


print(dec_self_attn_pad_mask)
print(dec_self_attn_subsequent_mask)
print(dec_self_attn_mask)




