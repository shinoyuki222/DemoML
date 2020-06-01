"""Evaluate the model"""

import argparse
import random
import logging
import os

import numpy as np
import torch

from pytorch_pretrained_bert import BertForTokenClassification, BertConfig
from pytorch_pretrained_bert import BertTokenizer
from metrics import f1_score
from metrics import classification_report

from data_loader import DataLoader
import utils

parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', default='..\\NER_data\\MSRA', help="Directory containing the dataset")
parser.add_argument('--bert_model_dir', default='bert-base-chinese-pytorch',
                    help="Directory containing the BERT model in PyTorch")
parser.add_argument('--model_dir', default='experiments\\base_model', help="Directory containing params.json")
parser.add_argument('--seed', type=int, default=23, help="random seed for initialization")
parser.add_argument('--restore_file', default='best', help="name of the file in `model_dir` containing weights to load")
parser.add_argument('--multi_gpu', default=False, action='store_true', help="Whether to use multiple GPUs if available")
parser.add_argument('--fp16', default=False, action='store_true',
                    help="Whether to use 16-bit float precision instead of 32-bit")


class DataLoader_test(object):
    def __init__(self, data_dir, bert_model_dir, params, token_pad_idx=0):
        self.data_dir = data_dir
        self.batch_size = params.batch_size
        self.max_len = params.max_len
        self.device = params.device
        self.seed = params.seed
        self.token_pad_idx = 0

        tags = self.load_tags()
        self.tag2idx = {tag: idx for idx, tag in enumerate(tags)}
        self.idx2tag = {idx: tag for idx, tag in enumerate(tags)}
        params.tag2idx = self.tag2idx
        params.idx2tag = self.idx2tag
        self.tag_pad_idx = self.tag2idx['O']

        self.tokenizer = BertTokenizer.from_pretrained(bert_model_dir, do_lower_case=True)

    def load_tags(self):
        tags = []
        file_path = os.path.join(self.data_dir, 'tags.txt')
        with open(file_path, 'r', encoding='utf-8') as file:
            for tag in file:
                tags.append(tag.strip())
        return tags

    def load_sentences(self, sent):
        """Loads sentences and tags from their corresponding files.
            Maps tokens and tags to their indices and stores them in the provided dict d.
        """
        sentence = []

        tokens = self.tokenizer.tokenize(sent.strip())
        sentence.append(self.tokenizer.convert_tokens_to_ids(tokens))
        return torch.tensor(sentence, dtype=torch.long)


def test(model, sentence, params, mark='Eval', verbose=False):
    """Evaluate the model on `steps` batches."""
    # set model to evaluation mode
    model.eval()

    idx2tag = params.idx2tag
    pred_tags = []

    batch_masks = sentence.gt(0)

    batch_output = model(sentence, token_type_ids=None,
                         attention_mask=batch_masks)  # shape: (batch_size, max_len, num_labels)

    batch_output = batch_output.detach().cpu().numpy()
    pred_tags.extend([idx2tag.get(idx) for indices in np.argmax(batch_output, axis=2) for idx in indices])

    return pred_tags


if __name__ == '__main__':
    args = parser.parse_args()

    # Load the parameters from json file
    json_path = os.path.join(args.model_dir, 'params.json')
    assert os.path.isfile(json_path), "No json configuration file found at {}".format(json_path)
    params = utils.Params(json_path)

    # Use GPUs if available
    params.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    params.n_gpu = torch.cuda.device_count()
    params.multi_gpu = args.multi_gpu

    # Set the random seed for reproducible experiments
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if params.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)  # set random seed for all GPUs
    params.seed = args.seed

    # Set the logger
    utils.set_logger(os.path.join(args.model_dir, 'evaluate.log'))

    # Create the input data pipeline
    logging.info("Loading the dataset...")

    # Initialize the DataLoader
    data_loader = DataLoader_test(args.data_dir, args.bert_model_dir, params, token_pad_idx=0)


    # logging.info("- done.")

    # Define the model
    config_path = os.path.join(args.bert_model_dir, 'bert_config.json')
    config = BertConfig.from_json_file(config_path)
    model = BertForTokenClassification(config, num_labels=len(params.tag2idx))

    model.to(params.device)
    # Reload weights from the saved file
    utils.load_checkpoint(os.path.join(args.model_dir, args.restore_file + '.pth.tar'), model)
    if args.fp16:
        model.half()
    if params.n_gpu > 1 and args.multi_gpu:
        model = torch.nn.DataParallel(model)

    logging.info("Starting evaluation...")
    while(1):
        try:
            # Get input sentence
            input_sentence = input('> ')
            # Check if it is quit case
            if input_sentence == 'q' or input_sentence == 'quit': break
            # Normalize sentence
            test_data = data_loader.load_sentences(input_sentence)
            print(input_sentence)
            # Evaluate sentence
            pred_tags = test(model, test_data, params, mark='Test', verbose=True)
            # Format and print response sentence
            pred_tags[:] = [x for x in pred_tags if x != 'PAD']
            print('Entity tags:', ' '.join(pred_tags))
        except KeyError:
            print("Error: Encountered unknown word.")


