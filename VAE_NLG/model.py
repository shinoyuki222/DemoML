import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.nn.init as init
import numpy as np

# from highway import Highway
from const import *

igore_idx = len(WORD)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")




class Encoder(nn.Module):
    def __init__(self, embed_dim, hidden_size, num_layers, dropout):
        super().__init__()

        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.dropout = dropout
        # self.embedding = embedding

        self.rnn = nn.LSTM(embed_dim, hidden_size, num_layers, dropout=(0 if num_layers == 1 else dropout), bidirectional=True)

    def forward(self, input_sent, input_lengths, hidden=None):
        # embedded = self.embedding(input_sent)
        packed = nn.utils.rnn.pack_padded_sequence(input_sent, input_lengths)

        rnn_outputs, hidden = self.rnn(packed, hidden)

        rnn_outputs, _ = nn.utils.rnn.pad_packed_sequence(rnn_outputs)

        outputs = rnn_outputs[-1]
        # outputs = torch.cat((hidden[0][-1], hidden[1][-1]), -1)

        out = F.dropout(outputs[-1], p=self.dropout)

        return out, hidden


class Decoder(nn.Module):
    def __init__(self, embed_dim, latent_dim, hidden_size, num_layers, dropout, vocab_size):
        super().__init__()

        self.hidden_size = hidden_size
        self.dropout = dropout
        self.num_layers = num_layers
        self.latent_dim = latent_dim

        self.rnn = nn.LSTM(embed_dim + latent_dim, hidden_size,
                           num_layers, dropout=dropout, batch_first=False)
        self.lr = nn.Linear(hidden_size, vocab_size)


    def forward(self, input_dec, z, hidden=None):
        _len, bsz, _ = input_dec.size()

        z = z.unsqueeze(0).expand(_len, bsz, self.latent_dim)


        input_dec = torch.cat((input_dec, z), -1)

        rnn_out, hidden = self.rnn(input_dec)
        rnn_out = F.dropout(rnn_out, p=self.dropout)
        out = self.lr(rnn_out.view(-1, self.hidden_size))

        return F.log_softmax(out, dim=-1), hidden


class VAE(nn.Module):
    def __init__(self, args):
        super().__init__()

        for k, v in args.__dict__.items():
            self.__setattr__(k, v)

        self.embed = nn.Embedding(self.vocab_size, self.embed_dim)
        self.embed.weight.data = torch.Tensor(self.pre_w2v)

        self.encode = Encoder(self.embed_dim,
                              self.enc_hsz, self.enc_layers, self.dropout)

        self._enc_mu = nn.Linear(self.enc_hsz * 2, self.latent_dim)
        self._enc_log_sigma = nn.Linear(self.enc_hsz * 2, self.latent_dim)

        self.decode = Decoder(self.embed_dim, self.latent_dim,
                              self.dec_hsz, self.dec_layers, self.dropout, self.vocab_size)

    def forward(self, enc_input, dec_input, input_lengths, enc_hidden=None, dec_hidden=None):
        enc_ = self.embed(enc_input)
        enc_output, enc_hidden = self.encode(enc_, input_lengths)
        z = self.post_gaussian(enc_output)

        dec_ = self.embed(dec_input)
        dec, dec_hidden = self.decode(dec_, z)

        return dec, self.latent_loss, enc_hidden, dec_hidden

    def post_gaussian(self, enc_output):
        mu = self._enc_mu(enc_output)
        sigma = torch.exp(.5 * self._enc_log_sigma(enc_output))
        self.latent_loss = self._latent_loss(mu, sigma)
        std_z = torch.normal(0,1,size=sigma.size())
        return mu + sigma * std_z

    def _latent_loss(self,mu, sigma):
        pow_mu = mu * mu
        pow_sigma = sigma * sigma
        # return 0.5 * torch.mean(pow_mu + pow_sigma - torch.log(pow_sigma) - 1)
        return 0.5 * torch.sum(pow_mu + pow_sigma - torch.log(pow_sigma) - 1, dim=-1).mean()
