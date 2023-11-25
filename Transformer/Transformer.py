import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import nu
as np
import torch.optim as optim


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
    subsequent_mask = torch.from_nu
(subsequent_mask).byte()
    return subsequent_mask


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
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, Q, K, V, attn_mask):
        # Q K V (batch_size, len , d_model * n_heads)
        # context = softmax(Q*K.T/sqrt(d_model)) * V
        scores = torch.matmul(Q, K.transpose(-1,-2))/np.sqrt(d_k)
        scores.masked_fill_(attn_mask, -1e9)
        attn_weights = self.softmax(scores)
        context = torch.matmul(attn_weights, V)
        return context, attn_weights

class MultiHeadAttention(nn.Module):
    def __init__(self):
        super(MultiHeadAttention, self).__init__()
        self.W_Q = nn.Linear(d_model, d_k * n_heads)
        self.W_K = nn.Linear(d_model, d_k * n_heads)
        self.W_V = nn.Linear(d_model, d_v * n_heads)

    def forward(self, Q, K, V, attn_mask):
        # batch first
        # Q: (batch_size, len_q, d_model)
        # K: (batch_size, len_k, d_model)
        # V: (batch_size, len_k, d_model)
        # attn_mask: (batch_size, len_q, len_k)
        residual, batch_size = Q, Q.size(0)
        # (B, S, dk/d_q) -proj-> (B, S, D) -split-> (B, S, H, d_k(d_q)) -trans-> (B, H, S, d_k(d_q))
        q_s = self.W_Q(Q).view(batch_size, -1, n_heads, d_k).transpose(1,2)  # q_s: (batch_size, n_heads, len_q, d_k)
        k_s = self.W_K(K).view(batch_size, -1, n_heads, d_k).transpose(1,2)  # k_s: (batch_size, n_heads, len_k, d_k)
        v_s = self.W_V(V).view(batch_size, -1, n_heads, d_v).transpose(1,2)  # v_s: (batch_size, n_heads, len_k, d_v)

        #(B, len_q, len_k) -unsqueeze(1)-> (B, 1, len_q, len_k) - repeat -> (B, n_heads, len_q, len_k)
        attn_mask = attn_mask.unsqueeze(1).repeat(1, n_heads, 1, 1) # attn_mask : (batch_size, n_heads, len_q, len_k)

        # context: (batch_size, n_heads, len_q, d_v)
        # attn: (batch_size, n_heads, len_q/len_k, len_k/len_q)
        context, attn = ScaledDotProductAttention()(q_s, k_s, v_s, attn_mask)
        # context: (batch_size, n_heads, len_q, d_v) -trans->(batch_size, len_q, n_heads, d_v) -reshape->(batch_size , len_q , n_heads * d_v)
        context = context.transpose(1, 2).reshape(batch_size, -1, n_heads * d_v) # context: (batch_size , len_q , n_heads * d_v)
        # output: (batch_size, len_q, d_model)
        output = nn.Linear(n_heads * d_v, d_model)(context)
        return nn.LayerNorm(d_model)(output + residual), attn 

class PoswiseFFN(nn.Module):
    def __init__(self):
        super(PoswiseFFN, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)

    def forward(self, inputs):
        # inputs : (batch_size, len_q, d_model)
        residual = inputs 
        output = nn.ReLU()(self.conv1(inputs.transpose(1, 2)))
        output = self.conv2(output).transpose(1, 2)
        return nn.LayerNorm(d_model)(output + residual)

class EncoderLayer(nn.Module):
    def __init__(self):
        super(EncoderLayer, self).__init__()
        self.enc_self_attn = MultiHeadAttention()
        self.pos_ffn = PoswiseFFN()

    def forward(self, enc_inputs, enc_self_attn_mask):
        # Q = K = V = enc_inputs
        enc_outputs, attn = self.enc_self_attn(enc_inputs, enc_inputs, enc_inputs, enc_self_attn_mask)
        # enc_outputs: (batch_size, len_q, d_model)
        enc_outputs = self.pos_ffn(enc_outputs) 
        return enc_outputs, attn

class DecoderLayer(nn.Module):
    def __init__(self):
        super(DecoderLayer, self).__init__()
        self.dec_self_attn = MultiHeadAttention()
        self.dec_enc_attn = MultiHeadAttention()
        self.pos_ffn = PoswiseFFN()

    def forward(self, dec_inputs, enc_outputs, dec_self_attn_mask, dec_enc_attn_mask):
        dec_outputs, dec_self_attn = self.dec_self_attn(dec_inputs, dec_inputs, dec_inputs, dec_self_attn_mask)
        dec_outputs, dec_enc_attn = self.dec_enc_attn(dec_outputs, enc_outputs, enc_outputs, dec_enc_attn_mask)
        dec_outputs = self.pos_ffn(dec_outputs)
        return dec_outputs, dec_self_attn, dec_enc_attn

class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.src_emb = nn.Embedding(src_vocab_size, d_model)
        self.pos_emb = nn.Embedding.from_pretrained(positional_encoding(src_len+1, d_model),freeze=True)
        self.layers = nn.ModuleList([EncoderLayer() for _ in range(n_layers)])

    def forward(self, enc_inputs): # enc_inputs : (batch_size, source_len)
        enc_outputs = self.src_emb(enc_inputs) + self.pos_emb(enc_inputs)
        enc_self_attn_mask = get_attn_pad_mask(enc_inputs, enc_inputs)
        enc_self_attns = []
        for layer in self.layers:
            enc_outputs, enc_self_attn = layer(enc_outputs, enc_self_attn_mask)
            enc_self_attns.append(enc_self_attn)
        # enc_self_attns (batch_size, n_layers, source_len, d_model)
        return enc_outputs, enc_self_attns

class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.tgt_emb = nn.Embedding(tgt_vocab_size, d_model)
        self.pos_emb = nn.Embedding.from_pretrained(positional_encoding(tgt_len+1, d_model),freeze=True)
        self.layers = nn.ModuleList([DecoderLayer() for _ in range(n_layers)])

    def forward(self, dec_inputs, enc_inputs, enc_outputs): # dec_inputs : (batch_size, target_len)
        dec_outputs = self.tgt_emb(dec_inputs) + self.pos_emb(dec_inputs)
        dec_self_attn_pad_mask = get_attn_pad_mask(dec_inputs, dec_inputs)
        dec_self_attn_subsequent_mask = get_attn_subsequent_mask(dec_inputs)
        dec_self_attn_mask = torch.gt((dec_self_attn_pad_mask + dec_self_attn_subsequent_mask), 0)

        dec_enc_attn_mask = get_attn_pad_mask(dec_inputs, enc_inputs)

        dec_self_attns, dec_enc_attns = [], []
        for layer in self.layers:
            dec_outputs, dec_self_attn, dec_enc_attn = layer(dec_outputs, enc_outputs, dec_self_attn_mask, dec_enc_attn_mask)
            dec_self_attns.append(dec_self_attn)
            dec_enc_attns.append(dec_enc_attn)
        return dec_outputs, dec_self_attns, dec_enc_attns

class Transformer(nn.Module):
    def __init__(self):
        super(Transformer, self).__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()
        self.projection = nn.Linear(d_model, tgt_vocab_size, bias=False)
    def forward(self, enc_inputs, dec_inputs):
        enc_outputs, enc_self_attns = self.encoder(enc_inputs)
        dec_outputs, dec_self_attns, dec_enc_attns = self.decoder(dec_inputs, enc_inputs, enc_outputs)
        dec_logits = self.projection(dec_outputs) # dec_logits : (batch_size, src_vocab_size, tgt_vocab_size)
        return dec_logits.view(-1, dec_logits.size(-1)), enc_self_attns, dec_self_attns, dec_enc_attns

class Transformer_Mix(nn.Module):
    def __init__(self):
        super(Transformer_Mix,self).__init__()
        self.transformer = Encoder()
        self.fc = nn.Linear(d_model, d_model)
        self.activ = nn.Tanh()
        self.classifier = nn.Linear(d_model, 2)
        self.norm = nn.LayerNorm(d_model)
        self.decoder = nn.Linear(d_model, tgt_size)

    def forward(self, enx_inputs, enc_self_attn_mask):
        enc_outputs, enc_self_attns = self.encoder(enc_inputs)
        h_pooled = self.activ(self.fc(enc_outputs[:, 0]))
        logits_clsf = self.classifier(h_pooled)

        masked_pos = masked_pos[:, :, None].expand(-1, -1, output.size(-1)) # [batch_size, maxlen, d_model]
        h_masked = torch.gather(output, 1, masked_pos) # masking position [batch_size, len, d_model]
        h_masked = self.norm(self.activ2(self.linear(enc_outputs)))

        

if __name__ == '__main__':

    sentences = ['我 喜欢 机器 学习', 'S i like machine learning', 'i like machine learning E']

    # Transformer Parameters
    # Padding Should be Zero index
    src_vocab = {'P' : 0, '我' : 1, '喜欢' : 2, '机器' : 3, '学习' : 4}
    src_vocab_size = len(src_vocab)

    tgt_vocab = {'P' : 0, 'i' : 1, 'like' : 2, 'machine' : 3, 'learning' : 4, 'S' : 5, 'E' : 6}
    number_dict = {i: w for i, w in enumerate(tgt_vocab)}
    tgt_vocab_size = len(tgt_vocab)

    src_len = 5
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

    enc_inputs, dec_inputs, target_batch = make_batch(sentences)
    print(enc_inputs)

    model = Transformer()

    predict, _, _, _ = model(enc_inputs, dec_inputs)



    src_emb = nn.Embedding(src_vocab_size, d_model)
    pos_emb = nn.Embedding.from_pretrained(positional_encoding(src_len+1, d_model),freeze=True)
    enc_outputs = src_emb(enc_inputs) + pos_emb(enc_inputs)
    enc_self_attn_mask = get_attn_pad_mask(enc_inputs, enc_inputs)
    # enc_outputs, enc_self_attn = layer(enc_outputs, enc_self_attn_mask)

    #unit tests

    context, attn = ScaledDotProductAttention()(enc_outputs, enc_outputs, enc_outputs, enc_self_attn_mask)

    ans, attn = MultiHeadAttention()(enc_outputs, enc_outputs, enc_outputs, enc_self_attn_mask)

    enc_o, self_attn = Encoder()(enc_inputs)

    enc_inputs, dec_inputs, target_batch = make_batch(sentences)
    # greedy_dec_input = greedy_decoder(model, enc_inputs, start_symbol=tgt_vocab["S"])
    # scripted_module = torch.jit.script(model(enc_inputs, dec_inputs))


    print(context.size())
    print(context.transpose(1,2).is_contiguous())
    # print(context.transpose(1,2).reshape(1, -1, n_heads * d_v))

    print(enc_outputs.size())
    # print(enc_outputs.view(-1, 8, 256).size())

    criterion = nn.CrossEntr
Loss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)


    dec_self_attn_pad_mask = get_attn_pad_mask(dec_inputs, dec_inputs)
    dec_self_attn_subsequent_mask = get_attn_subsequent_mask(dec_inputs)
    dec_self_attn_mask = torch.gt((dec_self_attn_pad_mask + dec_self_attn_subsequent_mask), 0)
    print(dec_self_attn_pad_mask)
    print(dec_self_attn_subsequent_mask)
    print(dec_self_attn_mask)

    exit()

    for epoch in range(20):
        optimizer.zero_grad()
        enc_inputs, dec_inputs, target_batch = make_batch(sentences)
        outputs, enc_self_attns, dec_self_attns, dec_enc_attns = model(enc_inputs, dec_inputs)
        loss = criterion(outputs, target_batch.contiguous().view(-1))
        print('Epoch:', '%04d' % (epoch + 1), 'cost =', '{:.6f}'.format(loss))
        loss.backward()
        optimizer.step()

