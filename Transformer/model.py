import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import numpy as np
import torch.optim as optim


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_attn_pad_mask(seq_q, seq_k):
    # print(seq_q)
    batch_size, len_q = seq_q.size()
    batch_size, len_k = seq_k.size()
    # eq(zero) is PAD token
    pad_attn_mask = seq_k.data.eq(0).unsqueeze(1)  # batch_size x 1 x len_k(=len_q), one is masking
    return pad_attn_mask.expand(batch_size, len_q, len_k)  # batch_size x len_q x len_k


def positional_encoding(n_position, d_model):
    def get_freq(position,hid_i):
        return position / np.power(10000, hid_i / d_model)
    def get_pos_vec(position):
        return [ get_freq(position,hid_i) for hid_i in range(d_model)]

    sinusoid_table = np.array([get_pos_vec(pos) for pos in range(n_position)])
    sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])
    sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])
    return torch.FloatTensor(sinusoid_table)

class ScaledDotProductAttention(nn.Module):
    def __init__(self):
        super(ScaledDotProductAttention, self).__init__()

    def forward(self, Q, K, V, attn_mask):
        # Q K V (batch_size,n_head,len,dim)
        scores = torch.matmul(Q, K.transpose(-1, -2)) / np.sqrt(d_k) # scores : [batch_size x n_heads x len_q(=len_k) x len_k(=len_q)]
        scores.masked_fill_(attn_mask, -1e9) # Fills elements of self tensor with value where mask is one.
        attn = nn.Softmax(dim=-1)(scores) # dim=-1 ensure scores match to len_k, attn with (batch_size, n_head, len_q, len_k)
        context = torch.matmul(attn, V) #(batch_size, n_head, len_q, dim)
        return context, attn



sentences = ['ich mochte ein bier P', 'S i want a beer', 'i want a beer E']

# Transformer Parameters
# Padding Should be Zero index
src_vocab = {'P' : 0, 'ich' : 1, 'mochte' : 2, 'ein' : 3, 'bier' : 4}
src_vocab_size = len(src_vocab)

tgt_vocab = {'P' : 0, 'i' : 1, 'want' : 2, 'a' : 3, 'beer' : 4, 'S' : 5, 'E' : 6}
number_dict = {i: w for i, w in enumerate(tgt_vocab)}
tgt_vocab_size = len(tgt_vocab)

src_len = 5
tgt_len = 5

d_model = 512  # Embedding Size
d_ff = 2048 # FeedForward dimension
d_k = d_v = 64  # dimension of K(=Q), V
n_layers = 6  # number of Encoder of Decoder Layer
n_heads = 8  # number of heads in Multi-Head Attention

def make_batch(sentences):
    input_batch = [[src_vocab[n] for n in sentences[0].split()]]
    output_batch = [[tgt_vocab[n] for n in sentences[1].split()]]
    target_batch = [[tgt_vocab[n] for n in sentences[2].split()]]
    return torch.LongTensor(input_batch), torch.LongTensor(output_batch), torch.LongTensor(target_batch)

enc_inputs, dec_inputs, target_batch = make_batch(sentences)
src_emb = nn.Embedding(src_vocab_size, d_model)
pos_emb = nn.Embedding.from_pretrained(positional_encoding(src_len+1, d_model),freeze=True)
enc_outputs = src_emb(enc_inputs) + pos_emb(torch.LongTensor([[1,2,3,4,0]]))
enc_self_attn_mask = get_attn_pad_mask(enc_inputs, enc_inputs)
# enc_outputs, enc_self_attn = layer(enc_outputs, enc_self_attn_mask)

context, attn = ScaledDotProductAttention()(enc_outputs, enc_outputs, enc_outputs, enc_self_attn_mask)

