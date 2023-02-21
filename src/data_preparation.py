import numpy as np
import nltk
from nltk.corpus import stopwords
import torch
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm
from collections import Counter

nltk.download("stopwords")
nltk.download("wordnet")
stopwords = set(stopwords.words("english"))

tqdm.pandas()


def pad_features(reviews, pad_id, seq_length=128):
    # features = np.zeros((len(reviews), seq_length), dtype=int)
    features = np.full((len(reviews), seq_length), pad_id, dtype=int)

    for i, row in enumerate(reviews):
        # if seq_length < len(row) then review will be trimmed
        features[i, : len(row)] = np.array(row)[:seq_length]

    return features


def data_processing(data):
    # get all processed reviews
    reviews = data.processed.values
    # merge into single variable, separated by whitespaces
    words = " ".join(reviews)
    # obtain list of words
    words = words.split()

    # build vocabulary
    counter = Counter(words)
    vocab = sorted(counter, key=counter.get, reverse=True)
    int2word = dict(enumerate(vocab, 1))
    int2word[0] = "<PAD>"
    word2int = {word: id for id, word in int2word.items()}

    # encode words
    reviews_enc = [
        [word2int[word] for word in review.split()] for review in tqdm(reviews)
    ]

    seq_length = 256
    features = pad_features(
        reviews_enc, pad_id=word2int["<PAD>"], seq_length=seq_length
    )

    assert len(features) == len(reviews_enc)
    assert len(features[0]) == seq_length

    # get labels as numpy
    labels = data.label.to_numpy()

    # train test split
    train_size = 0.7  # we will use 80% of whole data as train set
    val_size = 0.5  # and we will use 50% of test set as validation set

    # make train set
    split_id = int(len(features) * train_size)
    train_x, remain_x = features[:split_id], features[split_id:]
    train_y, remain_y = labels[:split_id], labels[split_id:]

    # make val and test set
    split_val_id = int(len(remain_x) * val_size)
    val_x, test_x = remain_x[:split_val_id], remain_x[split_val_id:]
    val_y, test_y = remain_y[:split_val_id], remain_y[split_val_id:]

    # print out the shape
    print("Feature Shapes:")
    print("===============")
    print("Train set: {}".format(train_x.shape))
    print("Validation set: {}".format(val_x.shape))
    print("Test set: {}".format(test_x.shape))

    # define batch size
    batch_size = 128

    # create tensor datasets
    trainset = TensorDataset(torch.from_numpy(train_x), torch.from_numpy(train_y))
    validset = TensorDataset(torch.from_numpy(val_x), torch.from_numpy(val_y))
    testset = TensorDataset(torch.from_numpy(test_x), torch.from_numpy(test_y))

    # create dataloaders
    trainloader = DataLoader(trainset, shuffle=True, batch_size=batch_size)
    valloader = DataLoader(validset, shuffle=True, batch_size=batch_size)
    testloader = DataLoader(testset, shuffle=True, batch_size=batch_size)

    # check our batches
    dataiter = iter(trainloader)
    xtrain, ytrain = next(dataiter)
    print(xtrain.shape, ytrain.shape)

    dataiterv = iter(valloader)
    xval, yval = next(dataiterv)
    print(xval.shape, yval.shape)

    dataitert = iter(testloader)
    xtest, ytest = next(dataitert)
    print(xtest.shape, ytest.shape)

    return dataiter, dataiterv, dataitert, len(word2int)
