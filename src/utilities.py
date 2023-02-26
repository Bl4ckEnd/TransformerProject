import torch
import torch.nn as nn
import math
import yaml


def subsequent_mask(size):
    "Mask out subsequent positions."
    attn_shape = (1, size, size)
    subsequent_mask = torch.triu(torch.ones(attn_shape), diagonal=1).type(torch.uint8)
    return subsequent_mask == 0


class PositionwiseFeedForward(nn.Module):
    "Implements FFN equation."

    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.w_1(x).relu()
        x = self.dropout(x)
        x = self.w_2(x)
        return x


class Embeddings(nn.Module):
    def __init__(self, d_model, vocab):
        super(Embeddings, self).__init__()
        self.lut = nn.Embedding(vocab, d_model)
        self.d_model = d_model

    def forward(self, x):
        x = self.lut(x)
        x *= math.sqrt(self.d_model)
        return x


class PositionalEncoding(nn.Module):
    "Implement the PE function."

    def __init__(self, d_model, dropout, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x):
        x = x + self.pe[:, : x.size(1)].requires_grad_(False)
        return self.dropout(x)


def set_device(name="cpu"):
    if name == "cpu":
        return torch.device("cpu")
    elif name == "gpu":
        try:
            return torch.device("cuda")
        except RuntimeError:
            print("GPU unavailable, falling back to CPU")
            return torch.device("cpu")
    elif name == "mps":
        if torch.backends.mps.is_available():
            try:
                return torch.device("mps")
            except RuntimeError:
                print("MPS unavailable, falling back to CPU")
                return torch.device("cpu")
        else:
            print("MPS unavailable, falling back to CPU")
            return torch.device("cpu")
    else:
        raise ValueError("Invalid device name")


def load_params():
    with open("params.yaml") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    data_path = config["paths"]["data"]
    batch_size = config["model"]["batch_size"]
    seq_length = config["model"]["seq_length"]
    N = config["model"]["N"]
    d_model = config["model"]["d_model"]
    d_ff = config["model"]["d_ff"]
    h = config["model"]["h"]
    amount_of_data = config["training"]["amount_of_data"]
    learning_rate = config["training"]["learning_rate"]
    epochs = config["training"]["epochs"]
    SAVE_PATH = config["paths"]["weights"]
    device = config["training"]["device"]

    return (
        data_path,
        batch_size,
        seq_length,
        N,
        d_model,
        d_ff,
        h,
        amount_of_data,
        learning_rate,
        epochs,
        SAVE_PATH,
        device,
    )
