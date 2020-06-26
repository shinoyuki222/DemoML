from typing import Any

import torch
from torch import optim
from consts import *
from model import *
from dataloader import *
from model import BiRNN_tagger
import random
from tqdm import tqdm
# from time import sleep


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def test(model, voc, tag, sentence, max_length=MAX_LENGTH):
    ### Format input sentence as a batch
    # words -> indexes
    indexes_batch = [indexesFromSentence(voc, sentence)]
    # Create lengths tensor
    lengths = torch.tensor([len(indexes) for indexes in indexes_batch])
    # Transpose dimensions of batch to match models' expectations
    input_batch = torch.LongTensor(indexes_batch).transpose(0, 1)
    # Use appropriate device
    input_batch = input_batch.to(device)
    lengths = lengths.to(device)
    # Decode sentence with searcher
    output = model(input_batch, lengths)
    # print(output.shape)
    # find max scores and token idex
    token_scores, token_indexes = torch.max(output[:,:,1:], dim=2)
    # indexes -> words
    decoded_words = [tag.index2tag[token_index.item()+1] for token_index in token_indexes]
    return decoded_words


def testInput(model, voc, tag):
    print("==============get=================")
    print("Type q or quit to quit chat")
    print("===========================================")
    input_sentence = ''
    while (1):
        try:
            # Get input sentence
            input_sentence = input('> ')
            # Check if it is quit case
            if input_sentence == 'q' or input_sentence == 'quit': break
            # Normalize sentence
            input_sentence = text_process(input_sentence)
            print(input_sentence)
            # Evaluate sentence
            output_words = test(model, voc, tag, input_sentence)
            # Format and print response sentence
            output_words[:] = [x for x in output_words if x != 'PAD']
            print('Entity tags:', ' '.join(output_words))

        except KeyError:
            print("Error: Encountered unknown word.")

def evaluate(input_variable, lengths, target_variable, criterion, mask, model):
    with torch.no_grad():
        # Set device options
        # criterion = nn.CrossEntropyLoss()
        # criterion = nn.NLLLoss()
        input_variable = input_variable.to(device)
        lengths = lengths.to(device)
        target_variable = target_variable.to(device)
        mask = mask.to(device)
        criterion = criterion.to(device)

        # Forward pass through model
        output = model(input_variable, lengths)
        output = output.view(-1, output.shape[-1])
        loss = criterion(output, target_variable.view(-1))

    return loss

def train(input_variable, lengths, target_variable, criterion, mask, model,model_optimizer,clip):
    # Zero gradients
    model_optimizer.zero_grad()

    # Set device options
    input_variable = input_variable.to(device)
    lengths = lengths.to(device)
    target_variable = target_variable.to(device)
    mask = mask.to(device)
    criterion = criterion.to(device)

    # criterion = nn.NLLLoss()
    # Forward pass through model
    output = model(input_variable, lengths)
    output = output.view(-1, output.shape[-1])

    loss = criterion(output, target_variable.view(-1))

    # Perform backpropatation
    model_optimizer.zero_grad()
    loss.backward()

    # Clip gradients: gradients are modified in place
    _ = nn.utils.clip_grad_norm_(model.parameters(), clip)

    # Adjust model weights
    model_optimizer.step()

    return loss


def trainIters(model_name, train_loader, dev_loader, criterion, model, model_optimizer, embedding,rnn_n_layers, save_dir, n_iteration, print_every, save_every, clip,
               corpus_name, start_iteration = 0):

    # Initializations
    print('Initializing ...')
    start_iteration += 1
    print_loss = 0
    print_loss_dev = 0
    best_loss = float('Inf')

    # criterion = torch.nn.CrossEntropyLoss().to(device)
    # Training loop
    print("Training...")
    for iteration in range(start_iteration, n_iteration + 1):
        loss = 0
        print_line = 'Iteration: {0}/{1}; Average batch loss: {2:.4f}; Progress'.format(iteration,n_iteration, loss)
        print_loss = 0
        print_loss_dev = 0
        n_batch = len(train_loader)

        # show with tqdm
        try:
            with tqdm(range(n_batch)) as tqdm_t:
                for i_batch, train_batch in zip(tqdm_t,train_loader):
                    tqdm_t.set_description(print_line)
                    tqdm_t.refresh() # to show immediately the update
                    # sleep(0.01)
                    input_variable, lengths, target_variable, mask, max_target_len = train_batch
                    # Run a training iteration with batch
                    model.train()
                    loss = train(input_variable, lengths, target_variable, criterion, mask, model, model_optimizer, clip)
                    print_line = 'Iteration: {0}/{1}; Average batch loss: {2:.4f}; Progress'.format(iteration,n_iteration, loss)
                    # print("Iteration: {0}; Batch: {1}/{2}; Average batch loss: {3:.4f}".format(iteration,i_batch,n_batch,loss))
                    print_loss += loss

                    for dev_batch in dev_loader:
                        input_variable, lengths, target_variable, mask, max_target_len = dev_batch
                        model.eval()
                        loss_dev = evaluate(input_variable, lengths, target_variable, criterion, mask, model)
                        print_loss_dev += loss_dev

                # Print progress
                if iteration % print_every == 0:
                    print_loss_avg = print_loss /n_batch/print_every
                    print_loss_dev_avg = print_loss_dev /n_batch/print_every
                    print("Iteration: {}; Percent complete: {:.1f}%; Average train loss: {:.4f}, Average dev loss: {:.4f}".format(iteration,
                                                                                                  iteration / n_iteration * 100,
                                                                                                  print_loss_avg, print_loss_dev_avg))
                # Save best model
                if print_loss_dev_avg - best_loss < 0.0:
                    print("validation loss {0} is better than {1}, saving checkpoint....".format(print_loss_dev_avg,best_loss))
                    best_loss = print_loss_dev_avg
                    directory = os.path.join(save_dir, corpus_name, model_name,
                                             '{}_{}'.format(rnn_n_layers, hidden_size))
                    if not os.path.exists(directory):
                        os.makedirs(directory)
                    torch.save({
                        'iteration': iteration,
                        'model': model.state_dict(),
                        'model_opt': model_optimizer.state_dict(),
                        'loss': loss,
                        'loss_dev': loss_dev,
                        'embedding': embedding.state_dict()
                    }, os.path.join(directory, '{}.tar'.format('BestModel')))

                # Save checkpoint
                if (iteration % save_every == 0):
                    directory = os.path.join(save_dir, corpus_name, model_name,
                                             '{}_{}'.format(rnn_n_layers, hidden_size))
                    if not os.path.exists(directory):
                        os.makedirs(directory)
                    torch.save({
                        'iteration': iteration,
                        'model': model.state_dict(),
                        'model_opt': model_optimizer.state_dict(),
                        'loss': loss,
                        'loss_dev': loss_dev,
                        'embedding': embedding.state_dict()
                    }, os.path.join(directory, '{}_{}.tar'.format(iteration, 'checkpoint')))

        except KeyboardInterrupt:
            tqdm_t.close()
            raise
        tqdm_t.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--corpus', action='store', dest='corpus_name', default='MSRA',help='Store corpus name')
    parser.add_argument('-a', '--attn_model', action='store', dest='attn_model', default='concat',
                        help='Store attention mode dot concat or general')
    parser.add_argument('-lm', '--load_model', action="store_true", dest='load_model', default=False, help='Load saved model')
    parser.add_argument('-ld', '--load_dct', action="store_true", dest='load_dct', default=False,
                        help='Load saved dct including tag and voc')

    parser.add_argument('-cp', '--checkpoint', action="store", dest='checkpoint_iter', default=-1, type=int,
                        help='Set loaded checkpoint_iter')
    parser.add_argument('-xt', '--train', action="store_true", dest='skip_train', default=False,
                        help='Skip train model')
    parser.add_argument('-xe', '--evaluate', action="store_true", dest='skip_evaluate', default=False,
                        help='Skip evaluate model')

    parser.add_argument('-hs', '--hidden_size', action="store", dest='hidden_size', default=500, type=int,
                        help='Set hidden_size')
    parser.add_argument('-en', '--rnn_num', action="store", dest='rnn_n_layers', default=2, type=int,
                        help='Set rnn_n_layers')
    parser.add_argument('-dp', '--dropout', action="store", dest='dropout', default=0.1, type=int,
                        help='Set dropout rate')
    parser.add_argument('-b', '--batch_size', action="store", dest='batch_size', default=64, type=int,
                        help='Set batch_size')

    parser.add_argument('-n', '--n_iteration', action="store", dest='n_iteration', default=100, type=int,
                        help='Set n_iteration')

    parser.add_argument('-s', '--save_every', action="store", dest='save_every', default=1, type=int,
                        help='Set save_every')
    parser.add_argument('-p', '--print_every', action="store", dest='print_every', default=1, type=int,
                        help='Set print_every')

    args = parser.parse_args()

    save_dir = os.path.join("..", "save")
    corpus_name = args.corpus_name
    corpus = os.path.join("..", "NER_data", corpus_name)
    datafile_train = os.path.join(corpus, "train")
    datafile_dev = os.path.join(corpus, "dev")
    print("corpus_name: {0}, corpus = {1}, datafile_train = {2}".format(corpus_name, corpus, datafile_train))

    if args.load_model:
        print("load_model: {0}, checkpoint = {1}".format(args.load_model, args.checkpoint_iter))
    if args.load_model == False and args.skip_train:
        print("load_model: {0}, skip_train = {1}".format(args.load_model, args.skip_train))
        print("Loading Best Trained model\nPlease using -l to load a specific trained model.")



    # set dictionary and pair
    if args.skip_train:
        voc, tag = load_static_dict(save_dir, corpus_name)
    elif args.load_dct:
        voc, tag = load_static_dict(save_dir, corpus_name)
        pairs = loadDevData(datafile_train, tag)
        pairs_dev = loadDevData(datafile_dev, tag)
    else:
        # Load/Assemble voc and pairs
        voc, tag, pairs = loadTrainData(corpus_name, datafile_train)
        # Print some pairs to validate
        print("\npairs:")
        for pair in pairs[:10]:
            print(pair)
        # Trim voc and pairs
        voc, pairs = trimRareWords(voc, pairs, MIN_COUNT)
        save_static_dict(voc, tag, save_dir)
        pairs_dev = loadDevData(datafile_dev, tag)

    # Configure models
    model_name = 'NER_model'
    attn_model = args.attn_model
    hidden_size = args.hidden_size
    rnn_n_layers = args.rnn_n_layers
    dropout = args.dropout
    batch_size = args.batch_size
    output_size = tag.num_tags
    print('Building model ...')
    # Initialize word embeddings
    embedding = nn.Embedding(voc.num_words, hidden_size)

    # Initialize RNN models
    # model = AttnRNN(attn_model,hidden_size, output_size, embedding, rnn_n_layers, dropout)
    model = BiRNN_tagger(hidden_size, output_size, embedding, rnn_n_layers, dropout)



    # Load model if a loadFilename is provided
    if args.load_model:
        # Set checkpoint to load from; set to None if starting from scratch
        checkpoint_iter = args.checkpoint_iter
        if args.checkpoint_iter < 0:
            loadFilename = os.path.join(save_dir, corpus_name, model_name,
                                     '{}_{}'.format(rnn_n_layers, hidden_size),
                                    '{}.tar'.format("BestModel"))
        else:
            loadFilename = os.path.join(save_dir, corpus_name, model_name,
                                     '{}_{}'.format(rnn_n_layers, hidden_size),
                                    '{}_checkpoint.tar'.format(checkpoint_iter))
        # If loading on same machine the model was trained on
        # checkpoint = torch.load(loadFilename)
        # If loading a model trained on GPU to CPU
        # checkpoint = torch.load(loadFilename, map_location=torch.device('cpu'))

        checkpoint = torch.load(loadFilename, map_location=torch.device(device))
        iteration = checkpoint['iteration']
        print("Load interation {}".format(iteration))
        model_sd = checkpoint['model']
        model_opt_sd = checkpoint['model_opt']
        embedding_sd = checkpoint['embedding']
        embedding.load_state_dict(embedding_sd)
        model.load_state_dict(model_sd)
    # Use appropriate device
    model = model.to(device)
    print('Models built and ready to go!')


    # Train model
    if args.skip_train == False:
        clip = 50.0
        # teacher_forcing_ratio = 1.0
        learning_rate = 0.0001
        # decoder_learning_ratio = 5.0
        n_iteration = args.n_iteration
        #set start_iteration if load model
        try:
            iteration
            start_iteration = iteration
        except NameError:
            start_iteration = 0

        print_every = args.print_every
        save_every = args.save_every

        # Ensure dropout layers are in train mode
        model.train()

        # Initialize optimizers
        print('Building optimizers ...')
        model_optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        if args.load_model:
            model_optimizer.load_state_dict(model_opt_sd)

        # If you have cuda, configure cuda to call
        if device == 'cuda':
            for state in model.state.values():
                for k, v in state.items():
                    if isinstance(v, torch.Tensor):
                        state[k] = v.cuda()

        # Run training iterations
        print("Starting Training!")
        train_loader = Set_DataLoader(voc, tag, pairs, batch_size = batch_size)
        dev_loader = Set_DataLoader(voc, tag, pairs_dev, batch_size = batch_size)
        criterion = SetCriterion(tag=tag, tag_ignore=['O'],ignore_index= PAD_token)
        trainIters(model_name, train_loader, dev_loader, criterion, model, model_optimizer,embedding, rnn_n_layers, save_dir, n_iteration, print_every, save_every, clip, corpus_name, start_iteration)

    # Test model
    if args.skip_evaluate==False:
        model.eval()
        testInput(model, voc, tag)