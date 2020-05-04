import argparse
import time
import torch
import os
import model
from dataloader import DataLoader
from generator import generator_beam

parser = argparse.ArgumentParser(description='VAE-NLG')
parser.add_argument('--data', type=str, default='./data/vae_nlg.pt',
                    help='location of the data corpus')
parser.add_argument('--embed-dim', type=int, default=128)
parser.add_argument('--hw-layers', type=int, default=2)
parser.add_argument('--hw-hsz', type=int, default=128)
parser.add_argument('--latent-dim', type=int, default=128)
parser.add_argument('--dropout', type=float, default=0.5)
parser.add_argument('--enc-hsz', type=int, default=128)
parser.add_argument('--enc-layers', type=int, default=1)
parser.add_argument('--dec-hsz', type=int, default=128)
parser.add_argument('--dec-layers', type=int, default=2)

args = parser.parse_args()

data = torch.load(args.data)
args.idx2word = {v: k for k, v in data['word2idx'].items()}

args.vocab_size = data['vocab_size']
args.pre_w2v = data['pre_w2v']

print('loading model...')

save_dir = 'model'
model_name = 'vae_nlg_full_model'
directory = os.path.join(save_dir,model_name+'.tar')
ckpt = torch.load(directory)
vae_model = model.VAE(args)
vae_model.load_state_dict(ckpt['model'])
# poetry = generate_songci(vae_model,args, -1)
scores, poetries = generator_beam(vae_model,args,10,10)
for score,poetry in zip(scores,poetries):
	print("poetry generation - [{0}] with score {1}".format(poetry, score,encoding = 'utf-8',ascii=True))
print('-' * 90)
# generate_songci(vae_model,args.idx2word)



