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



def generate_songci(model, args, max_len=30):

    z = torch.normal(0,1,size=(1,args.latent_dim))
    next_word = torch.ones(1, 1, device=device, dtype=torch.long) * BOS
    
    portry = ""
    # hidden = self.decode.init_hidden(1)
    length = 0

    hidden = None

    while True:
        input_sent = next_word.expand(1,1).to(device)
        encode = model.lookup_table(input_sent)
        output, hidden = model.decode(encode, z, hidden)
        prob = output.squeeze().data

        score, next_word = torch.max(prob[igore_idx-1:],dim=-1)
        word = args.idx2word[next_word.item()+ igore_idx-1]

        if word == WORD[EOS] or length == max_len-1:
            portry += "。"
            break
        else:
            portry += word
            length += 1

    return portry[:-1] + "。"
