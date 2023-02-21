from model import make_model
import pandas as pd
from data_preparation import dataprocessing


data_path = "../online_dataset/imdb_processed.csv"
data = pd.read_csv(data_path)

test_iter, val_iter, test_iter, vocab_len = dataprocessing(data)

model = make_model(256, 1, 64, 128, 2)


