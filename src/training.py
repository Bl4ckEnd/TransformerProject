from model import make_model
import torch
import pandas as pd
from data_preparation import data_processing
from torch import optim
from torch import nn
from tqdm import tqdm
import yaml
from datetime import datetime as dt

with open("params.yaml") as f:
    config = yaml.load(f, Loader=yaml.FullLoader)

data_path = config["paths"]["data"]
data = pd.read_csv(data_path)

# SET HYPER-PARAMETERS
batch_size = config["model"]["batch_size"]
seq_length = config["model"]["seq_length"]
N = config["model"]["N"]
d_model = config["model"]["d_model"]
d_ff = config["model"]["d_ff"]
h = config["model"]["h"]
amount_of_data = config["training"]["amount_of_data"]

data = data.sample(frac=amount_of_data, random_state=42)
train_loader, val_loader, test_loader, vocab_len = data_processing(
    data, batch_size=batch_size, seq_length=seq_length
)

model = make_model(
    vocab_len, N=N, d_model=d_model, d_ff=d_ff, h=h, seq_length=seq_length
)

# Training
criterion = nn.CrossEntropyLoss()
learning_rate = config["training"]["learning_rate"]
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
epochs = config["training"]["epochs"]

# set to training mode
model.train()

for e in tqdm(range(epochs)):
    running_loss = 0
    for x, y in tqdm(train_loader):
        optimizer.zero_grad()
        y_pred = model(x)
        loss = criterion(y_pred, y)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    else:
        print(f"Training loss: {running_loss / len(train_loader)}")

# Save model
time = dt.now().strftime("%Y-%m-%d_%H-%M-%S")
SAVE_PATH = config["paths"]["weights"]
torch.save(model, SAVE_PATH + time + ".pth")
