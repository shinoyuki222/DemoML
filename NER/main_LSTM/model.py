'''model'''

import torch
import torch.nn as nn
import torch.nn.functional as F

from consts import *
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def SetCriterion(tag=None, tag_ignore=None, weight = None,size_average=None, ignore_index= -100, reduce=None, reduction='mean'):
    print("Setting critetion...")
    if tag_ignore:
        if not weight:
            weight = torch.ones(tag.num_tags)
        if tag:
            try:
                for tag_i in tag_ignore:
                    weight[tag.tag2index[tag_i]] = 0.01
                print("Training with weight:\n{}\n{}".format(tag.index2tag, weight.data))
            except KeyError:
                print("Error: Encountered unknown tag.")

        else:
            print("no tag file given.")
    print("Ignore idx: {}".format(ignore_index))
    return nn.CrossEntropyLoss(weight=weight,
                               size_average=size_average,
                               ignore_index=ignore_index,
                               reduce=reduce,
                               reduction=reduction)

class EncoderRNN(nn.Module):
    def __init__(self, hidden_size, embedding, n_layers=1, dropout=0):
        super(EncoderRNN, self).__init__()
        self.n_layers = n_layers
        self.hidden_size = hidden_size
        self.embedding = embedding

        # Initialize GRU; the input_size and hidden_size params are both set to 'hidden_size'
        #   because our input size is a word embedding with number of features == hidden_size
        self.gru = nn.GRU(hidden_size, hidden_size, n_layers, dropout=(0 if n_layers == 1 else dropout), bidirectional=True)

    def forward(self, input_seq, input_lengths, hidden=None):
        # Convert word indexes to embeddings
        embedded = self.embedding(input_seq)
        # Pack padded batch of sequences for RNN module
        packed = nn.utils.rnn.pack_padded_sequence(embedded, input_lengths)
        # Forward pass through GRU
        outputs, hidden = self.gru(packed, hidden)
        # Unpack padding
        outputs, _ = nn.utils.rnn.pad_packed_sequence(outputs)
        # Sum bidirectional GRU outputs
        outputs = outputs[:, :, :self.hidden_size] + outputs[:, :, self.hidden_size:]
        # Return output and final hidden state
        return outputs, hidden


# Luong attention layer
class Attn(nn.Module):
    def __init__(self, method, hidden_size):
        super(Attn, self).__init__()
        self.method = method
        if self.method not in ['dot', 'general', 'concat']:
            raise ValueError(self.method, "is not an appropriate attention method; Must be on in dot or general or concat ")
        self.hidden_size = hidden_size
        if self.method == 'general':
            self.attn = nn.Linear(self.hidden_size, hidden_size)
        elif self.method == 'concat':
            self.attn = nn.Linear(self.hidden_size * 2, hidden_size)
            self.v = nn.Parameter(torch.FloatTensor(hidden_size))

    def dot_score(self, hidden, encoder_output):
        return torch.sum(hidden * encoder_output, dim=2)

    def general_score(self, hidden, encoder_output):
        energy = self.attn(encoder_output)
        return torch.sum(hidden * energy, dim=2)

    def concat_score(self, hidden, encoder_output):
        # print("encoder_output size = {}".format(encoder_output.shape))
        # print("hidden dim  = {}".format(hidden.shape))
        # print("reshape hidden as = {}".format(hidden.expand(encoder_output.size(0), -1, -1).shape))
        # print("concat result = {}".format(torch.cat((hidden.expand(encoder_output.size(0), -1, -1), encoder_output), 2).shape))
        energy = self.attn(torch.cat((hidden.expand(encoder_output.size(0), -1, -1), encoder_output), 2)).tanh()
        return torch.sum(self.v * energy, dim=2)

    def forward(self, hidden, encoder_outputs):
        # Calculate the attention weights (energies) based on the given method
        if self.method == 'general':
            attn_energies = self.general_score(hidden, encoder_outputs)
        elif self.method == 'concat':
            attn_energies = self.concat_score(hidden, encoder_outputs)
        elif self.method == 'dot':
            attn_energies = self.dot_score(hidden, encoder_outputs)

        # Transpose max_length and batch_size dimensions
        attn_energies = attn_energies.t()

        # Return the softmax normalized probability scores (with added dimension)
        return F.softmax(attn_energies, dim=1).unsqueeze(1)


# Bidirectional recurrent neural network (many-to-many)
class BiRNN_tagger(nn.Module):
    def __init__(self, hidden_size, output_size, embedding, n_layers=1, dropout=0):
        super(BiRNN_tagger, self).__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.hidden_size = hidden_size
        self.embedding = embedding
        self.n_layers = n_layers
        self.num_classes = output_size
        self.dropout = dropout

        self.lstm = nn.LSTM(self.hidden_size, self.hidden_size//2, self.n_layers, batch_first=False,
                            bidirectional=True)
        self.fc_out = nn.Linear(self.hidden_size, self.num_classes) # 2 for bidirection
        # self.word_embeddings = nn.Embedding(self.vocab_size, self.embedding_dim).to(self.device)
        self.dropout = nn.Dropout(dropout)

    def forward(self, input_seq, input_lengths, hidden=None):
        # Convert word indexes to embeddings
        embedded = self.embedding(input_seq)
        # Set initial states
        h0 = torch.zeros(self.n_layers * 2, input_seq.size(1), self.hidden_size//2).to(self.device)  # 2 for bidirection
        c0 = torch.zeros(self.n_layers * 2, input_seq.size(1), self.hidden_size//2).to(self.device)
        # embeds = self.word_embeddings(x.transpose(0, 1))
        packed_input = nn.utils.rnn.pack_padded_sequence(embedded, input_lengths)
        # Forward propagate LSTM
        packed_output, hidden = self.lstm(packed_input,(h0, c0))  # out: tensor of shape (batch_size, seq_length, hidden_size*2)
        output, _ = nn.utils.rnn.pad_packed_sequence(packed_output)
        # Decode the hidden state of the last time step
        output = self.dropout(output)
        output = self.fc_out(output)
        output = F.softmax(output, dim=2)
        return output


class AttnRNN(nn.Module):
    def __init__(self, attn_model, hidden_size, output_size, embedding, n_layers=1, dropout=0):
        super(AttnRNN, self).__init__()

        self.attn_model = attn_model
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.embedding = embedding
        self.n_layers = n_layers
        self.dropout = dropout

        # Define layers
        self.encoderrnn = EncoderRNN(hidden_size//2, embedding, n_layers, dropout)
        self.concat = nn.Linear(hidden_size * 2, hidden_size)
        self.out = nn.Linear(hidden_size, output_size)
        self.gru = nn.GRU(hidden_size, hidden_size//2, n_layers,
                          dropout=(0 if n_layers == 1 else dropout), bidirectional=True)

        self.attn = Attn(attn_model, hidden_size)
    def forward(self, input_seq, input_lengths, hidden=None):
        # Convert word indexes to embeddings
        embedded = self.embedding(input_seq)
        # Pack padded batch of sequences for RNN module
        packed = nn.utils.rnn.pack_padded_sequence(embedded, input_lengths)
        # Forward pass through GRU
        rnn_output, hidden = self.gru(packed, hidden)
        # Unpack padding
        rnn_output, _ = nn.utils.rnn.pad_packed_sequence(rnn_output)
        # # Sum bidirectional GRU outputs
        # rnn_output = rnn_output[:, :, :self.hidden_size] + rnn_output[:, :, self.hidden_size:]
        # Return output and final hidden state
        # Calculate attention weights from the current GRU output
        attn_weights = self.attn(rnn_output, embedded)
        # attn_weights = attn_weights
        attn_weights = attn_weights.repeat(1,attn_weights.shape[2],1)
        # Calculate weighted rnn output
        rnn_output_weighted = torch.bmm(attn_weights, rnn_output.transpose(0,1))
        # Concatenate weighted context vector and GRU output using Luong eq. 5
        # rnn_output = rnn_output.squeeze(0)
        # rnn_output_weighted = rnn_output_weighted.squeeze(1)

        concat_input = torch.cat((embedded, rnn_output_weighted.transpose(0,1)), 2)
        concat_output = torch.tanh(self.concat(concat_input))
        # Predict next word using Luong eq. 6
        output = self.out(concat_output)
        output = F.softmax(output, dim=2)
        # print("output shape", output.shape)
        # Return output and final hidden state
        return output


class LuongAttnDecoderRNN(nn.Module):
    def __init__(self, attn_model, embedding, hidden_size, output_size, n_layers=1, dropout=0.1):
        super(LuongAttnDecoderRNN, self).__init__()

        # Keep for reference
        self.attn_model = attn_model
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.n_layers = n_layers
        self.dropout = dropout

        # Define layers
        self.embedding = embedding
        self.embedding_dropout = nn.Dropout(dropout)
        self.gru = nn.GRU(hidden_size, hidden_size, n_layers, dropout=(0 if n_layers == 1 else dropout))
        self.concat = nn.Linear(hidden_size * 2, hidden_size)
        self.out = nn.Linear(hidden_size, output_size)

        self.attn = Attn(attn_model, hidden_size)

    def forward(self, input_step, last_hidden, encoder_outputs):
        # Note: we run this one step (word) at a time
        # Get embedding of current input word
        embedded = self.embedding(input_step)
        embedded = self.embedding_dropout(embedded)
        # Forward through unidirectional GRU
        rnn_output, hidden = self.gru(embedded, last_hidden)
        # Calculate attention weights from the current GRU output
        attn_weights = self.attn(rnn_output, encoder_outputs)
        # Multiply attention weights to encoder outputs to get new "weighted sum" context vector
        context = attn_weights.bmm(encoder_outputs.transpose(0, 1))
        # Concatenate weighted context vector and GRU output using Luong eq. 5
        rnn_output = rnn_output.squeeze(0)
        context = context.squeeze(1)
        concat_input = torch.cat((rnn_output, context), 1)
        concat_output = torch.tanh(self.concat(concat_input))
        # Predict next word using Luong eq. 6
        output = self.out(concat_output)
        output = F.softmax(output, dim=1)
        # Return output and final hidden state
        return output, hidden


class GreedySearchDecoder(nn.Module):
    def __init__(self, encoder, decoder):
        super(GreedySearchDecoder, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, input_seq, input_length, max_length):
        # Forward input through encoder model
        encoder_outputs, encoder_hidden = self.encoder(input_seq, input_length)
        # Prepare encoder's final hidden layer to be first hidden input to the decoder
        decoder_hidden = encoder_hidden[:decoder.n_layers]
        # Initialize decoder input with SOS_token
        decoder_input = torch.ones(1, 1, device=device, dtype=torch.long) * SOS_token
        # Initialize tensors to append decoded words to
        all_tokens = torch.zeros([0], device=device, dtype=torch.long)
        all_scores = torch.zeros([0], device=device)
        # Iteratively decode one word token at a time
        for _ in range(max_length):
            # Forward pass through decoder
            decoder_output, decoder_hidden = self.decoder(decoder_input, decoder_hidden, encoder_outputs)
            # Obtain most likely word token and its softmax score
            decoder_scores, decoder_input = torch.max(decoder_output, dim=1)
            # Record token and score
            all_tokens = torch.cat((all_tokens, decoder_input), dim=0)
            all_scores = torch.cat((all_scores, decoder_scores), dim=0)
            # Prepare current token to be next decoder input (add a dimension)
            decoder_input = torch.unsqueeze(decoder_input, 0)
        # Return collections of word tokens and scores
        return all_tokens, all_scores

if __name__ == '__main__':
    import argparse
    import os
    from dataloader import *

    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--corpus', action='store', dest='corpus_name', default='MSRA',help='Store corpus name')
    parser.add_argument('-a', '--attn_model', action='store', dest='attn_model', default='dot',
                        help='Store attention mode dot concat or general')

    parser.add_argument('-hs', '--hidden_size', action="store", dest='hidden_size', default=500, type=int,
                        help='Set hidden_size')
    parser.add_argument('-en', '--rnn_num', action="store", dest='rnn_n_layers', default=2, type=int,
                        help='Set rnn_n_layers')
    parser.add_argument('-dp', '--dropout', action="store", dest='dropout', default=0.1, type=int,
                        help='Set dropout rate')
    parser.add_argument('-b', '--batch_size', action="store", dest='batch_size', default=64, type=int,
                        help='Set batch_size')

    parser.add_argument('-n', '--n_iteration', action="store", dest='n_iteration', default=4000, type=int,
                        help='Set n_iteration')

    parser.add_argument('-s', '--save_every', action="store", dest='save_every', default=500, type=int,
                        help='Set save_every')
    parser.add_argument('-p', '--print_every', action="store", dest='print_every', default=1, type=int,
                        help='Set print_every')

    args = parser.parse_args()

    save_dir = os.path.join("", "save")
    corpus_name = args.corpus_name
    corpus = os.path.join("NER_data", corpus_name)
    datafile_train = os.path.join(corpus, "train")
    datafile_dev = os.path.join(corpus, "val")
    print("corpus_name: {0}, corpus = {1}, datafile_train = {2}".format(corpus_name, corpus, datafile_train))

    voc, tag = load_static_dict(save_dir, corpus_name)
    # Configure models
    model_name = 'NER_model'
    attn_model = args.attn_model
    hidden_size = args.hidden_size
    rnn_n_layers = args.rnn_n_layers
    dropout = args.dropout
    batch_size = args.batch_size
    output_size = tag.num_tags
    print('Building model ...')
    # Initialize word embeddings
    embedding = nn.Embedding(voc.num_words, hidden_size)
    # Initialize encoder & decoder models
    model=AttnRNN(attn_model,hidden_size, output_size, embedding, rnn_n_layers, dropout)
    # Use appropriate device
    model = model.to(device)
    print('Models built and ready to go!')
