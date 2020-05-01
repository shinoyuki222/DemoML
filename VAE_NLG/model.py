import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.nn.init as init
import numpy as np

# from highway import Highway
from const import BOS

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

        self._init_weight()

    def forward(self, input_dec, z, hidden=None):
        _len, bsz, _ = input_dec.size()

        z = z.unsqueeze(0).expand(_len, bsz, self.latent_dim)


        input_dec = torch.cat((input_dec, z), -1)

        rnn_out, hidden = self.rnn(input_dec)
        rnn_out = F.dropout(rnn_out, p=self.dropout)
        out = self.lr(rnn_out.contiguous().view(-1, self.hidden_size))

        return F.log_softmax(out, dim=-1), hidden


    def _init_weight(self, scope=.1):
        self.lr.weight.data.uniform_(-scope, scope)
        self.lr.bias.data.fill_(0)


class VAE(nn.Module):
    def __init__(self, args):
        super().__init__()

        for k, v in args.__dict__.items():
            self.__setattr__(k, v)

        self.lookup_table = nn.Embedding(self.vocab_size, self.embed_dim)
        self.lookup_table.weight.data.copy_(torch.from_numpy(self.pre_w2v))

        # self.hw = Highway(self.hw_layers, self.hw_hsz, F.relu)
        self.encode = Encoder(self.embed_dim,
                              self.enc_hsz, self.enc_layers, self.dropout)

        self._enc_mu = nn.Linear(self.enc_hsz * 2, self.latent_dim)
        self._enc_log_sigma = nn.Linear(self.enc_hsz * 2, self.latent_dim)

        self.decode = Decoder(self.embed_dim, self.latent_dim,
                              self.dec_hsz, self.dec_layers, self.dropout, self.vocab_size)

        self._init_weight()

    def forward(self, enc_input, dec_input, input_lengths, enc_hidden=None, dec_hidden=None):
        enc_ = self.lookup_table(enc_input)
        # enc_ = F.dropout(self.hw(enc_), p=self.dropout)

        enc_output, enc_hidden = self.encode(enc_, input_lengths)
        # mu = self._enc_mu(enc_output)
        z = self._gaussian(enc_output)

        dec_ = self.lookup_table(dec_input)
        dec, dec_hidden = self.decode(dec_, z)

        return dec, self.latent_loss, enc_hidden, dec_hidden

    def _gaussian(self, enc_output):
        def latent_loss(mu, sigma):
            pow_mu = mu * mu
            pow_sigma = sigma * sigma
            # return 0.5 * torch.mean(pow_mu + pow_sigma - torch.log(pow_sigma) - 1)
            return 0.5 * torch.sum(pow_mu + pow_sigma - torch.log(pow_sigma) - 1, dim=-1).mean()
        mu = self._enc_mu(enc_output)
        sigma = torch.exp(.5 * self._enc_log_sigma(enc_output))
        self.latent_loss = latent_loss(mu, sigma)

        weight = next(self.parameters()).data
        std_z = Variable(weight.new(*sigma.size()), requires_grad=False)
        std_z.data.copy_(torch.from_numpy(
            np.random.normal(size=sigma.size())))

        return mu + sigma * std_z

    def _init_weight(self):
        init.xavier_normal(self._enc_mu.weight)
        init.xavier_normal(self._enc_log_sigma.weight)

    def generate(self, max_len):
        size = (1, self.latent_dim)

        weight = next(self.parameters()).data
        z = Variable(weight.new(*size), volatile=True)
        z.data.copy_(torch.from_numpy(
            np.random.normal(size=size)))

        next_word = torch.ones(1, 1, device=device, dtype=torch.long) * BOS
        
        portry = ""
        # hidden = self.decode.init_hidden(1)
        for index in range(1, max_len + 1):
            input_sent = next_word.expand(1,1).to(device)
            encode = self.lookup_table(input_sent)
            output, hidden = self.decode(encode, z)

            prob = output.squeeze().data
            score, next_word = torch.max(prob,dim=-1)


            if index % 5 == 0:
                portry += self.idx2word[next_word.item()]
                portry += "，"
            else:
                portry += self.idx2word[next_word.item()]

        return portry[:-1] + "。"
