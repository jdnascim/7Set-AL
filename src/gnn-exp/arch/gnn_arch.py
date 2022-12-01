from torch_geometric.nn import GCNConv, SAGEConv, GATConv
import torch
import torch.nn.functional as F

class NSAGELin(torch.nn.Module):
    def __init__(self, input_size, output_size):
        super(NSAGELin, self).__init__()

        hidden_size = 4096

        self.conv1 = SAGEConv(input_size, hidden_size)
        self.linear = torch.nn.Linear(hidden_size, output_size)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = x.float()
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.linear(x)

        return F.log_softmax(x, dim=1)
    
    def reset_parameters(self):
        self.conv1.reset_parameters()
        self.linear.reset_parameters()

class NGCNLin(torch.nn.Module):
    def __init__(self, input_size, output_size):
        super(NGCNLin, self).__init__()

        hidden_size = 4096

        self.conv1 = GCNConv(input_size, hidden_size)
        self.linear = torch.nn.Linear(hidden_size, output_size)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = x.float()
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.linear(x)

        return F.log_softmax(x, dim=1)
    
    def reset_parameters(self):
        self.conv1.reset_parameters()
        self.linear.reset_parameters()