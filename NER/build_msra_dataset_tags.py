"""split the MSRA dataset for our model and build tags"""
import os
import random


def load_dataset(path_dataset):
    """Load dataset into memory from text file"""
    dataset = []
    with open(path_dataset, 'r', encoding='utf-8') as f:
        words, tags = [], []
        # Each line of the file corresponds to one word and tag
        for line in f:
            if line != '\n':
                line = line.strip('\n')
                word, tag = line.split('\t')
                try:
                    if len(word) > 0 and len(tag) > 0:
                        word, tag = str(word), str(tag)
                        words.append(word)
                        tags.append(tag)
                except Exception as e:
                    print('An exception was raised, skipping a word: {}'.format(e))
            else:
                if len(words) > 0:
                    assert len(words) == len(tags)
                    dataset.append((words, tags))
                    words, tags = [], []
    return dataset


def save_dataset(dataset, save_dir):
    """Write sentences.txt and tags.txt files in save_dir from dataset

    Args:
        dataset: ([(["a", "cat"], ["O", "O"]), ...])
        save_dir: (string)
    """
    # Create directory if it doesn't exist
    print('Saving in {}...'.format(save_dir))
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    # Export the dataset
    with open(os.path.join(save_dir, 'sentences.txt'), 'w', encoding='utf-8') as file_sentences, \
        open(os.path.join(save_dir, 'tags.txt'), 'w', encoding='utf-8') as file_tags:
        for words, tags in dataset:
            file_sentences.write('{}\n'.format(' '.join(words)))
            file_tags.write('{}\n'.format(' '.join(tags)))
    print('- done.')

def build_tags(data_dir, tags_file):
    """Build tags from dataset
    """
    data_types = ['train', 'dev', 'test']
    tags = set()
    for data_type in data_types:
        tags_path = os.path.join(data_dir, data_type, 'tags.txt')
        with open(tags_path, 'r', encoding='utf-8') as file:
            for line in file:
                tag_seq = filter(len, line.strip().split(' '))
                tags.update(list(tag_seq))
    with open(tags_file, 'w', encoding='utf-8') as file:
        file.write('\n'.join(tags))
    return tags


if __name__ == '__main__':
    # Check that the dataset exist, two balnk lines at the end of the file
    task_name = 'NER'
    data_path = task_name + '_data'
    # data_path = ''
    corpus_name = 'MSRA'

    datafile_train = os.path.join(data_path, corpus_name, 'msra_train_bio')
    datafile_test =os.path.join(data_path, corpus_name, 'msra_test_bio')
    save_path = os.path.join(data_path, corpus_name)
    save_path_train = os.path.join(save_path, 'train')
    save_path_dev = os.path.join(save_path, 'dev')
    save_path_test = os.path.join(save_path, 'test')
    msg = '{} or {} file not found. Make sure you have downloaded the right dataset'.format(datafile_train, datafile_test)
    assert os.path.isfile(datafile_train) and os.path.isfile(datafile_test), msg

    # Load the dataset into memory
    print('Loading {} dataset into memory...'.format(corpus_name))
    dataset_train_val = load_dataset(datafile_train)
    dataset_test = load_dataset(datafile_test)
    print('- done.')

    # Make a list that decides the order in which we go over the data
    order = list(range(len(dataset_train_val)))
    random.seed(2020)
    random.shuffle(order)

    # Split the dataset into train, val(split with shuffle) and test
    train_dataset = [dataset_train_val[idx] for idx in order[:42000]]  # 42000 for train
    val_dataset = [dataset_train_val[idx] for idx in order[42000:]]  # 3000 for val
    test_dataset = dataset_test  # 3442 for test
    save_dataset(train_dataset, save_path_train)
    save_dataset(val_dataset, save_path_dev)
    save_dataset(test_dataset, save_path_test)

    # Build tags from dataset
    build_tags(save_path, os.path.join(save_path, 'tags.txt'))

