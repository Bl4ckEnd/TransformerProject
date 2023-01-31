from transformers import AutoTokenizer
import torchtext.datasets as datasets
from torchtext.vocab import build_vocab_from_iterator as torchbuild_vocab_from_iterator
from os.path import exists
import torch


def tokenize(text, tokenizer):
    return tokenizer.encode(text, add_special_tokens=True, max_length=512)


def yield_tokens(data_iter, tokenizer, index):
    for from_to_tuple in data_iter:
        yield tokenizer(from_to_tuple[index])


def build_vocabulary(tokenizer):
    def tokenize_en(text):
        return tokenize(text, tokenizer)

    print("Building English Vocabulary ...")
    train, val, test = datasets.SST2(split=("train", "dev", "test"))
    vocab = torchbuild_vocab_from_iterator(
        yield_tokens(train + val + test, tokenize_en, index=1),
        min_freq=2,
        specials=["<s>", "</s>", "<blank>", "<unk>"],
    )

    vocab.set_default_index(vocab["<unk>"])

    return vocab


def load_vocab(tokenizer):
    if not exists("vocab.pt"):
        vocab = build_vocabulary(tokenizer)
        torch.save((vocab), "vocab.pt")
    else:
        vocab = torch.load("vocab.pt")
    print("Finished.\nVocabulary sizes:")
    print(len(vocab))
    return vocab


if __name__ == "__main__":

    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

    vocab = load_vocab(tokenizer)
    print(vocab)
