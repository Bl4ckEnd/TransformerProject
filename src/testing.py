import torch
import torch.nn as nn
import pandas as pd
from data_preparation import data_processing
from tqdm import tqdm
import yaml


with open("params.yaml") as f:
    config = yaml.load(f, Loader=yaml.FullLoader)


data_path = config["paths"]["data"]
data = pd.read_csv(data_path)

# SET HYPER-PARAMETERS
batch_size = config["model"]["batch_size"]
seq_length = config["model"]["seq_length"]
amount_of_data = config["training"]["amount_of_data"]

data = data.sample(frac=amount_of_data, random_state=42)
_, val_loader, test_loader, _ = data_processing(
    data, batch_size=batch_size, seq_length=seq_length
)

# Load model
PATH = "../saved_models/2023-02-23_17-29-54.pth"

model = torch.load(PATH)
# create test loop
model.eval()

correct = 0
total = 0
# since we're not training, we don't need to calculate the gradients for our outputs
with torch.no_grad():
    for x, y in tqdm(test_loader):
        y_pred = model(x)
        y_pred = nn.Softmax(dim=1)(y_pred)
        y_pred = torch.max(y_pred, dim=1)
        y_pred = y_pred.indices

        total += y.size(0)
        correct += torch.where(y_pred == y, 1, 0).sum().item()


accuracy = round(100 * correct / total, 3)
print(f"Accuracy of the network on test set: {accuracy} %")
