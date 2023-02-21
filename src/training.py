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
train_loader, val_loader, test_loader, vocab_len = data_processing(data)

model = make_model(vocab_len, N=6, d_model=128, d_ff=256, h=8, seq_length=128)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
epochs = 5

for e in tqdm(range(epochs)):
    running_loss = 0
    for x, y in tqdm(train_loader):
        y = y.float()
        optimizer.zero_grad()
        y_pred = model(x)
        loss = criterion(y_pred, y)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    else:
        print(f"Training loss: {running_loss / len(train_loader)}")

SAVE_PATH = "../saved_model_parameters/model.pth"
torch.save(model.state_dict(), SAVE_PATH)
