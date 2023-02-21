from model import make_model
import torch
import pandas as pd
from data_preparation import data_processing
from torch import optim
from torch import nn
from tqdm import tqdm


data_path = "../online_dataset/imdb_processed.csv"
data = pd.read_csv(data_path)
# only use 20% of data
data = data.sample(frac=0.2, random_state=42)

train_iter, val_iter, test_iter, vocab_len = data_processing(data)


model = make_model(vocab_len, N=2, d_model=64, d_ff=128, h=2)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
epochs = 1

for e in tqdm(range(epochs)):
    running_loss = 0
    for x, y in tqdm(train_iter):
        y = y.float()
        optimizer.zero_grad()
        y_pred = model(x)
        loss = criterion(y_pred, y)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    else:
        print(f"Training loss: {running_loss / len(train_iter)}")

SAVE_PATH = "../saved_model_parameters/model.pth"
torch.save(model.state_dict(), SAVE_PATH)
