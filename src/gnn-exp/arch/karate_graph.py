from torch_geometric.nn import GCNConv, SAGEConv, GATConv
import torch
import torch.nn.functional as F

class NGCN(torch.nn.Module):
    def __init__(self, input_size, output_size):
        super(NGCN, self).__init__()

        hidden_size = 4096

        self.conv1 = GCNConv(input_size, hidden_size)
        self.conv2 = GCNConv(hidden_size, output_size)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = x.float()
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)

        return F.log_softmax(x, dim=1)
    
    def reset_parameters(self):
        self.conv1.reset_parameters()
        self.conv2.reset_parameters()

class NSAGE(torch.nn.Module):
    def __init__(self, input_size, output_size):
        super(NSAGE, self).__init__()

        hidden_size = 4096

        self.conv1 = SAGEConv(input_size, hidden_size)
        self.conv2 = SAGEConv(hidden_size, output_size)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = x.float()
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)

        return F.log_softmax(x, dim=1)
    
    def reset_parameters(self):
        self.conv1.reset_parameters()
        self.conv2.reset_parameters()

class NAtt(torch.nn.Module):
    def __init__(self, input_size, output_size):
        super(NAtt, self).__init__()

        hidden_size = 4096

        self.conv1 = GATConv(input_size, hidden_size)
        self.conv2 = GATConv(hidden_size, output_size)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = x.float()
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)

        return F.log_softmax(x, dim=1)
    
    def reset_parameters(self):
        self.conv1.reset_parameters()
        self.conv2.reset_parameters()


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

class N3GCN(torch.nn.Module):
    def __init__(self, input_size, output_size):
        super(N3GCN, self).__init__()

        self.conv1 = GCNConv(input_size, 4096)
        self.conv2 = GCNConv(4096, 1024)
        self.conv3 = GCNConv(1024, output_size)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = x.float()
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv3(x, edge_index)

        return F.log_softmax(x, dim=1)
    
    def reset_parameters(self):
        self.conv1.reset_parameters()
        self.conv2.reset_parameters()
        self.conv3.reset_parameters()

class N3Att(torch.nn.Module):
    def __init__(self, input_size, output_size):
        super(N3Att, self).__init__()
        self.conv1 = GATConv(input_size, 4096)
        self.conv2 = GATConv(4096, 1024)
        self.conv3 = GATConv(1024, output_size)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = x.float()
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv3(x, edge_index)

        return F.log_softmax(x, dim=1)
    
    def reset_parameters(self):
        self.conv1.reset_parameters()
        self.conv2.reset_parameters()
        self.conv3.reset_parameters()

class N3SAGE(torch.nn.Module):
    def __init__(self, input_size, output_size):
        super(N3SAGE, self).__init__()
        self.conv1 = SAGEConv(input_size, 4096)
        self.conv2 = SAGEConv(4096, 1024)
        self.conv3 = SAGEConv(1024, output_size)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = x.float()
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv3(x, edge_index)

        return F.log_softmax(x, dim=1)
    
    def reset_parameters(self):
        self.conv1.reset_parameters()
        self.conv2.reset_parameters()
        self.conv3.reset_parameters()