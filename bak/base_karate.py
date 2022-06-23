import sys
sys.path.append("../")

from arch.karate_graph import KarateGraph

from gnn_utils import *
import numpy as np
import torch
import tqdm

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

args = {
    'device': device,
    'num_layers': 5,
    'hidden_dim': 256,
    'dropout': 0.5,
    'lr': 0.001,
    'epochs': 100
}

pyg_graph = load_image_data(16)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

pyg_graph = pyg_graph.to(device)

model = KarateGraph(960, 256, 2).to(device) 

print("Training")
for epoch in tqdm.tqdm(np.arange(1, args['epochs']+1)):
    train_data(model, pyg_graph, args['lr'])

print("Evaluating")
train_acc,test_acc = test_data(model, pyg_graph)

print('#' * 70)
print('Train Accuracy: %s' %train_acc )
print('Test Accuracy: %s' % test_acc)
print('#' * 70)
