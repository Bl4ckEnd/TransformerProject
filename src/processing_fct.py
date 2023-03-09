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


def data_processing(data, new_input, label, seq_length=256):
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

    # process single input
    new_input = " ".join(new_input)


    # encode words
    input_enc = [
        [word2int[word] for word in new_input.split()] for new_input in tqdm(new_input)
    ]

    features_dev = pad_features(
        input_enc, pad_id=word2int["<PAD>"], seq_length=seq_length
    )


    assert len(features_dev) == len(input_enc)
    assert len(features_dev[0]) == seq_length

    #Test and its label 
    test_x = features_dev
    #Create tendor Datasets
    test_set = TensorDataset(torch.tensor([test_x]), torch.tensor([label]))
    return test_set, len(word2int_dev)




  
