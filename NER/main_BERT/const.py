PAD = 0
UNK = 1
BOS = 2
EOS = 3

WORD = {
    PAD: "[PAD]",
    UNK: "[UNK]",
    BOS: "[CLS]",
    EOS: "[SEP]"
}


d_model = 512  # Embedding Size
d_ff = 2048 # FeedForward dimension
d_k = d_v = 64  # dimension of K(=Q), V
n_layers = 6  # number of Encoder of Decoder Layer
n_heads = 4  # number of heads in Multi-Head Attention