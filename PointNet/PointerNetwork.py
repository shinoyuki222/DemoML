import random
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from typing import Tuple

def batch(batch_size, min_len=5, max_len=12):
    array_len = torch.randint(low=min_len, 
                            high=max_len + 1,
                            size=(1,))

    x = torch.randint(high=10, size=(batch_size, array_len))
    return x, x.argsort(dim=1)

HIDDEN_SIZE = 256

BATCH_SIZE = 32
STEPS_PER_EPOCH = 500
EPOCHS = 1


class Encoder(nn.Module):
    def __init__(self, hidden_size: int):
        super(Encoder, self).__init__()
        self.lstm = nn.LSTM(1, hidden_size, batch_first=True)

    def forward(self, x: torch.Tensor):
        # x: (BATCH, ARRAY_LEN, 1)
        return self.lstm(x)

# Luong attention layer
class Attn(nn.Module):
    def __init__(self, hidden_size, units, method = 'concat'):
        super(Attn, self).__init__()
        self.method = method
        if self.method not in ['dot', 'general', 'concat']:
            raise ValueError(self.method, "is not an appropriate attention method; Must be on in dot or general or concat ")
        self.hidden_size = hidden_size
        if self.method == 'general':
            self.attn = nn.Linear(self.hidden_size, hidden_size)
        elif self.method == 'concat':
            self.attn = nn.Linear(self.hidden_size * 2, units)
            self.v = nn.Linear(units,1)

    def dot_score(self, hidden, encoder_output):
        # method 1
        # dot_prod = torch.bmm(hidden,encoder_output.transpose(1,2))
        # atten_weights = dot_prod.transpose(1,2)
        
        # method 2
        mul = hidden * encoder_output
        atten_weights = torch.sum(mul,dim=2)
        atten_weights = atten_weights.unsqueeze(2)
        return atten_weights

    def general_score(self, hidden, encoder_output):
        energy = self.attn(encoder_output)
        # print(energy.size())
        # print(encoder_output.size())
        # print(hidden.size())

        mul = hidden * energy
        atten_weights = torch.sum(mul,dim=2)
        atten_weights = atten_weights.unsqueeze(2)

        return atten_weights

    def concat_score(self, hidden, encoder_output):
        # print("encoder_output size = {}".format(encoder_output.shape))
        # print("hidden dim  = {}".format(hidden.shape))
        # print("reshape hidden as = {}".format(hidden.expand(encoder_output.size(0), -1, -1).shape))
        # print("concat result = {}".format(torch.cat((hidden.expand(encoder_output.size(0), -1, -1), encoder_output), 2).shape))
        _hidden = hidden.expand(encoder_output.size())
        energy = self.attn(torch.cat((_hidden, encoder_output),axis = 2)).tanh()
        return self.v(energy)

    def forward(self, encoder_outputs, decoder_hidden):
        # Calculate the attention weights (energies) based on the given method
        decoder_hidden = decoder_hidden.unsqueeze(1)
        if self.method == 'general':
            attn_energies = self.general_score(decoder_hidden, encoder_outputs)
        elif self.method == 'concat':
            attn_energies = self.concat_score(decoder_hidden, encoder_outputs)
        elif self.method == 'dot':
            attn_energies = self.dot_score(decoder_hidden, encoder_outputs)

        # Transpose max_length and batch_size dimensions
        # attn_energies = attn_energies.t()

        # Return the softmax normalized probability scores (with added dimension)
        attn_weights = F.softmax(attn_energies, dim=1)

        di_prime = attn_weights * encoder_outputs
        di_prime = di_prime.sum(1)

        return di_prime, attn_weights.squeeze(-1)

class Decoder(nn.Module):
    def __init__(self, hidden_size: int, attention_units: int = 10):
        super(Decoder, self).__init__()
        self.lstm = nn.LSTM(hidden_size + 1, hidden_size, batch_first=True)
        self.attention = Attn(hidden_size, attention_units)
        # self.atten = Attention(hidden_size, attention_units)

    def forward(self, x: torch.Tensor, hidden: Tuple[torch.Tensor], encoder_out: torch.Tensor):
        # x: (BATCH, 1, 1) 
        # hidden: (1, BATCH, HIDDEN_SIZE)
        # encoder_out: (BATCH, ARRAY_LEN, HIDDEN_SIZE)
        # For a better understanding about hidden shapes read: https://pytorch.org/docs/stable/nn.html#lstm

        # Get hidden states (not cell states) 
        # from the first and unique LSTM layer 
        ht = hidden[0][0]  # ht: (BATCH, HIDDEN_SIZE)

        # di: Attention aware hidden state -> (BATCH, HIDDEN_SIZE)
        # att_w: Not 'softmaxed', torch will take care of it -> (BATCH, ARRAY_LEN)
        # di, att_w = self.atten(encoder_out, ht)
        di, att_w = self.attention(encoder_out, ht)


        # Append attention aware hidden state to our input
        # x: (BATCH, 1, 1 + HIDDEN_SIZE)
        x = torch.cat([di.unsqueeze(1), x], dim=2)

        # Generate the hidden state for next timestep
        _, hidden = self.lstm(x, hidden)
        return hidden, att_w


class PointerNetwork(nn.Module):
    def __init__(self, 
               encoder: nn.Module, 
               decoder: nn.Module):
        super(PointerNetwork, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self,  x: torch.Tensor, y: torch.Tensor, teacher_force_ratio=.5):
        # x: (BATCH_SIZE, ARRAY_LEN)
        # y: (BATCH_SIZE, ARRAY_LEN)

        # Array elements as features
        # encoder_in: (BATCH, ARRAY_LEN, 1)
        encoder_in = x.unsqueeze(-1).type(torch.float)

        # out: (BATCH, ARRAY_LEN, HIDDEN_SIZE)
        # hs: tuple of (NUM_LAYERS, BATCH, HIDDEN_SIZE)
        out, hs = encoder(encoder_in)

        # Accum loss throughout timesteps
        loss = 0

        # Save outputs at each timestep
        # outputs: (ARRAY_LEN, BATCH)
        outputs = torch.zeros(out.size(1), out.size(0), dtype=torch.long)
                                    
        # First decoder input is always 0
        # dec_in: (BATCH, 1, 1)
        dec_in = torch.zeros(out.size(0), 1, 1, dtype=torch.float)
        
        for t in range(out.size(1)):
            hs, att_w = decoder(dec_in, hs, out)
            predictions = F.softmax(att_w, dim=1).argmax(1)

            # Pick next index
            # If teacher force the next element will we the ground truth
            # otherwise will be the predicted value at current timestep
            teacher_force = random.random() < teacher_force_ratio
            idx = y[:, t] if teacher_force else predictions
            dec_in = torch.stack([x[b, idx[b].item()] for b in range(x.size(0))])
            dec_in = dec_in.view(out.size(0), 1, 1).type(torch.float)

            # Add cross entropy loss (F.log_softmax + nll_loss)
            loss += F.cross_entropy(att_w, y[:, t])
            outputs[t] = predictions

        # Weight losses, so every element in the batch 
        # has the same 'importance' 
        batch_loss = loss / y.size(0)

        return outputs, batch_loss


def train(model, optimizer, epoch, clip=1.):
    """Train single epoch"""
    print('Epoch [{}] -- Train'.format(epoch))
    for step in range(STEPS_PER_EPOCH):
        optimizer.zero_grad()

        # Forward
        x, y = batch(BATCH_SIZE)
        out, loss = model(x, y)

        # Backward
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()

    if (step + 1) % 100 == 0:
      print('Epoch [{}] loss: {}'.format(epoch, loss.item()))


@torch.no_grad()
def evaluate(model, epoch):
    """Evaluate after a train epoch"""
    print('Epoch [{}] -- Evaluate'.format(epoch))

    x_val, y_val = batch(4)

    out, _ = model(x_val, y_val, teacher_force_ratio=0.)
    out = out.permute(1, 0)

    for i in range(out.size(0)):
        print('{} --> {} --> {}'.format(x_val[i], x_val[i].gather(0, out[i]), x_val[i].gather(0, y_val[i])))


encoder = Encoder(HIDDEN_SIZE)
decoder = Decoder(HIDDEN_SIZE)
ptr_net = PointerNetwork(encoder, decoder)

optimizer = optim.Adam(ptr_net.parameters())

for epoch in range(EPOCHS):
    train(ptr_net, optimizer, epoch + 1)
    evaluate(ptr_net, epoch + 1)