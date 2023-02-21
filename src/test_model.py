import torch

from model import make_model
import pandas as pd
from data_preparation import dataprocessing


data_path = "../online_dataset/imdb_processed.csv"
data = pd.read_csv(data_path)

train_iter, val_iter, test_iter, vocab_len = dataprocessing(data)

model = make_model(vocab_len, 1, 64, 128, 2)
x, y = next(train_iter)
model(x)

x, y = next(train_iter)
y_pred = model(x)
print("y_pred: ", y_pred, "dim: ", y_pred.shape)

print("y: ", y, "dim: ", y.shape)
print(torch.abs(y-y_pred))


