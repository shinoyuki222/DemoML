import numpy as np
import torch
import torch.nn as nn

def SetCriterion(label_dict=None, label_ignore=None, weight = None,size_average=None, ignore_index= -100, reduce=None, reduction='mean'):
    print("Setting critetion...")
    if label_ignore:
        if not weight:
            weight = torch.ones(len(label_dict))
        if label_dict:
            try:
                for label_i in label_ignore:
                    weight[label_dict[label_i]] = 0.1
                # print("Training with weight:\n{}\n{}".format(tag.index2tag, weight.data))
            except KeyError:
                print("Error: Encountered unknown tag.")

        else:
            print("no tag file given.")
    print("Ignore idx: {}".format(ignore_index))
    return nn.NLLLoss(weight=weight,
                               size_average=size_average,
                               ignore_index=ignore_index,
                               reduce=reduce,
                               reduction=reduction)
                               
