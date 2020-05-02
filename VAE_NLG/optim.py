import numpy as np
import torch
import torch.nn as nn

class ScheduledOptim(object):
    def __init__(self, optimizer, d_model, n_warmup_steps, parameters, clip):
        self.optimizer = optimizer
        self.d_model = d_model
        self.n_warmup_steps = n_warmup_steps
        self.n_current_steps = 0
        self.clip = clip
        self.parameters = parameters

    def step(self):
        self.optimizer.step()

    def zero_grad(self):
        self.optimizer.zero_grad()

    def clip_grad_norm(self):
        torch.nn.utils.clip_grad_norm_(self.parameters, self.clip)

    def update_learning_rate(self):
        self.n_current_steps += 1
        new_lr = np.power(self.d_model, -0.5) * np.min([np.power(self.n_current_steps, -0.5), np.power(self.n_warmup_steps, -1.5) * self.n_current_steps])

        for param_group in self.optimizer.param_groups:
            param_group['lr'] = new_lr

def SetCriterion(label_dict=None, label_ignore=None, weight = None,size_average=None, ignore_index= -100, reduce=None, reduction='mean'):
    print("Setting critetion...")
    if label_ignore:
        if not weight:
            weight = torch.ones(len(label_dict))
        if label_dict:
            for label_i in label_ignore:
                if label_i in label_dict:
                    weight[label_dict[label_i]] = 0.05
                    print("weight for `{}` set to {}".format(label_i,weight[label_dict[label_i]]))
                else:
                    print("Cannot set weight for unknown word `{}`".format(label_i))

            # try:
            #     for label_i in label_ignore:
            #         weight[label_dict[label_i]] = 0.01
            #         print("weight for {} set to {}".format(label_i,weight[label_dict[label_i]]))
            #     # print("Training with weight:\n{}\n{}".format(tag.index2tag, weight.data))
            # except KeyError:
            #     print("Error: Encountered unknown tag.")

        else:
            print("no tag file given.")
    print("Ignore idx: {}".format(ignore_index))
    return nn.NLLLoss(weight=weight,
                               size_average=size_average,
                               ignore_index=ignore_index,
                               reduce=reduce,
                               reduction=reduction)