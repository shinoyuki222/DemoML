import numpy as np
import torch
from const import *
import itertools

use_cuda = torch.cuda.is_available() and not args.unuse_cuda

PAD_token = 0
def zeroPadding(l, fillvalue=PAD_token):
    opt =list(itertools.zip_longest(*l, fillvalue=fillvalue))
    return opt
    # return list(itertools.zip_longest(*l, fillvalue=fillvalue))

class DataLoader(object):
    def __init__(self, src_sents, max_len, batch_size, cuda=use_cuda):
        self.cuda = cuda
        self.n_sents = len(src_sents)
        self._step = 0
        self._batch_size = batch_size

        self.n_batch = self.n_sents//self._batch_size+1
        self.n_batch = 2
        self._max_len = max_len
        self._enc_sents = np.asarray(src_sents)

        self._shuffle()
        self.ds_loader = self.gen_data()

    def gen_data(self):

        sents = np.copy(self._enc_sents)

        eos_tag = np.asarray([EOS] * self.n_sents).reshape((-1, 1))
        bos_tag = np.asarray([BOS] * self.n_sents).reshape((-1, 1))

        self._dec_sents = np.concatenate((bos_tag, sents), axis=-1)
        self._label = np.concatenate((sents, eos_tag), axis=-1)
        # print(self._enc_sents[0*self._batch_size:(0+1)*self._batch_size].shape)
        self.ds_loader = [self.batch2TrainData(self._enc_sents[i*self._batch_size:(i+1)*self._batch_size], self._dec_sents[i*self._batch_size:(i+1)*self._batch_size], self._label[i*self._batch_size:(i+1)*self._batch_size]) for i in range(self.n_batch)]
        return self.ds_loader


    def _shuffle(self):
        indices = np.arange(self._enc_sents.shape[0])
        np.random.shuffle(indices)
        self._enc_sents = self._enc_sents[indices]

    def inputVar(self, sents_batch):
        lengths = torch.tensor([len(indexes) for indexes in sents_batch])
        # print(lengths.size())
        padList = zeroPadding(sents_batch)
        padVar = torch.LongTensor(padList)
        return padVar, lengths

    def outputVar(self, sents_batch):
        indexes_batch = [indexesFromTag(tag, sentence) for sentence in l]
        max_target_len = max([len(indexes) for indexes in indexes_batch])
        padList = zeroPadding(indexes_batch)
        mask = binaryMatrix(padList)
        mask = torch.BoolTensor(mask)
        padVar = torch.LongTensor(padList)
        return padVar, mask, max_target_len

    # Returns all items for a given batch of pairs
    def batch2TrainData(self,enc_batch, dec_batch, opt_batch):
        enc, lengths = self.inputVar(enc_batch)
        dec, _ = self.inputVar(dec_batch)
        opt, _ = self.inputVar(opt_batch)
        # print([len(sent) for sent in end_batch])
        # print('====')
        # lengths = torch.tensor([len(sent) for sent in enc_batch])
        return enc, lengths, dec, opt


if __name__ == "__main__":
    data = torch.load("data/vae_nlg.pt")
    _data = DataLoader(
        data['train'],
        data["max_word_len"],
        4)

    for enc, lengths, dec, opt in _data.ds_loader:
        print(enc.size(), lengths.size(), dec.size(), opt.size())
        print(dec,'\n',opt)
        break

    # print(enc_input)
    # print(dec_input)
    # print(label)
