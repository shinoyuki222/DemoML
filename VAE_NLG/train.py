import argparse
import time

import torch


parser = argparse.ArgumentParser(description='VAE-NLG')
parser.add_argument('--epochs', type=int, default=100,
                    help='number of epochs for train')
parser.add_argument('--batch-size', type=int, default=16,
                    help='batch size for training')
parser.add_argument('--seed', type=int, default=1111,
                    help='random seed')
parser.add_argument('--unuse-cuda', action='store_true',
                    help='unuse cuda')
parser.add_argument('--n-warmup-steps', type=int, default=0)

parser.add_argument('--save', type=str, default='./vae_nlg.pt',
                    help='path to save the final model')
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
parser.add_argument('--clip', type=float, default=0.25)

args = parser.parse_args()


torch.manual_seed(args.seed)
use_cuda = torch.cuda.is_available() and not args.unuse_cuda

if use_cuda:
    torch.cuda.manual_seed(args.seed)


'''
load data
'''
from dataloader import DataLoader

data = torch.load(args.data)
# args.max_len = data["max_word_len"]
args.max_len = 30
args.vocab_size = data['vocab_size']
args.pre_w2v = data['pre_w2v']
args.idx2word = {v: k for k, v in data['word2idx'].items()}

dl = DataLoader(data['train'],args.max_len, args.batch_size)
training_data = dl.ds_loader

'''
build model
'''
import model
from optim import ScheduledOptim, SetCriterion
# from metric import SetCriterion
vae = model.VAE(args)
if use_cuda:
    vae = vae.cuda()

criterion = SetCriterion(data['word2idx'],label_ignore=['。','，','、',' '])

optimizer = ScheduledOptim(
    torch.optim.Adam(vae.parameters(), betas=(0.9, 0.98), eps=1e-09),
    args.embed_dim, args.n_warmup_steps, vae.parameters(), args.clip)

'''
train model
'''
import time
from tqdm import tqdm


def train():
    vae.train()
    total_loss = 0.
    for enc_input, lengths, dec_input, opt, mask in tqdm(training_data, mininterval=1, desc='Generator Train Processing', leave=False):
        optimizer.zero_grad()
        target, latent_loss, enc_hidden, dec_hidden = vae(enc_input, dec_input, lengths)
        loss = criterion(target, opt.view(-1)) + latent_loss

        loss.backward()
        optimizer.clip_grad_norm()
        optimizer.step()

        total_loss += loss.data

    return total_loss.data / dl.n_batch


'''
save model
'''
import os
from generator import generate_songci
def save(vae_model,save_dir = 'model'):
    directory = os.path.join(save_dir)
    if not os.path.exists(directory):
        os.makedirs(directory)

    torch.save({
        'model': vae_model.state_dict(),
        'model_opt': optimizer.optimizer.state_dict()
    }, os.path.join(directory, '{}.tar'.format("vae_nlg_full_model")))

def save_inference(vae_model,save_dir = 'model'):
    directory = os.path.join(save_dir)
    if not os.path.exists(directory):
        os.makedirs(directory)

    # print(vae_model.state_dict())
    torch.save({
        'encoder':vae_model.encode.state_dict(),
        'embedding':vae_model.lookup_table.state_dict()
        # 'model_opt': optimizer.state_dict()
    }, os.path.join(directory, '{}.tar'.format("vae_nlg_inference_model")))



best_acc = None
total_start_time = time.time()

try:
    print('-' * 90)
    for epoch in range(1, args.epochs + 1):
        epoch_start_time = time.time()
        loss = train()

        save(vae)
        print('| start of epoch {:3d} | time: {:2.2f}s | loss {:5.6f}'.format(
            epoch, time.time() - epoch_start_time, loss))
        print('-' * 90)
        vae.eval()
        for _ in range(10):
            with torch.no_grad():
                # portry = vae.generate_songci(args.idx2word)
                poetry = generate_songci(vae, args)
                print("poetry generation - [{}]".format(poetry, encoding = 'utf-8',ascii=True))
                print('-' * 90)
    # save_inference(vae)

except KeyboardInterrupt:
    print("-" * 90)
    print("Exiting from training early | cost time: {:5.2f}min".format(
        (time.time() - total_start_time) / 60.0))

'''
load model

'''


save_dir = 'model'
model_name = 'vae_nlg_full_model'
directory = os.path.join(save_dir,model_name+'.tar')
ckpt = torch.load(directory)
vae_model = model.VAE(args)
vae_model.load_state_dict(ckpt['model'])
poetry = generate_songci(vae_model,args)
print('loading model^')
print("poetry generation - [{}]".format(poetry, encoding = 'utf-8',ascii=True))
print('-' * 90)
# generate_songci(vae_model,args.idx2word)



