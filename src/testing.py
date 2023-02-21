from model import make_model
import torch
import pandas as pd
from data_preparation import data_processing
from tqdm import tqdm


data_path = "../online_dataset/imdb_processed.csv"
data = pd.read_csv(data_path)

# only use 20% of data
data = data.sample(frac=0.2, random_state=42)
train_loader, val_loader, test_loader, vocab_len = data_processing(data)

# load model from saved parameters
model = make_model(vocab_len, N=2, d_model=64, d_ff=128, h=2)
model.load_state_dict(torch.load("../saved_model_parameters/model.pth"))

# create test loop

correct = 0
total = 0
# since we're not training, we don't need to calculate the gradients for our outputs
with torch.no_grad():
    for x, y in tqdm(test_loader):
        y_pred = model(x)

        # set y_pred to 1 if y_pred > 0.5
        predicted = torch.where(
            y_pred > 0.5, torch.ones_like(y_pred), torch.zeros_like(y_pred)
        )
        total += y.size(0)
        correct += (predicted == y).sum().item()

print(f"Accuracy of the network on {len(test_loader)}: {100 * correct // total} %")
