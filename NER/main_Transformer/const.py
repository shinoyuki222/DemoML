PAD = 0
UNK = 1
BOS = 2
EOS = 3

WORD = {
    PAD: '<pad>',
    UNK: '<unk>',
    BOS: '<s>',
    EOS: '</s>'
}


d_model = 128  # Embedding Size
d_ff = 1024 # FeedForward dimension
d_k = d_v = 64  # dimension of K(=Q), V
n_layers = 3  # number of Encoder of Decoder Layer
n_heads = 2  # number of heads in Multi-Head Attention

MIN_COUNT = 4