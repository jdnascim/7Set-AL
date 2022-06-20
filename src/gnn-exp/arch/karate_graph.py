from torch_geometric.nn import GCNConv, SAGEConv, GATConv
import torch
import torch.nn.functional as F

class KarateGraph(torch.nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(KarateGraph, self).__init__()
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

class KarateSAGE(torch.nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(KarateSAGE, self).__init__()
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

class KarateAtt(torch.nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(KarateAtt, self).__init__()
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


class KarateGraphLin1(torch.nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(KarateGraphLin1, self).__init__()
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

class KarateGraphLin2(torch.nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(KarateGraphLin2, self).__init__()
        self.conv1 = GCNConv(input_size, hidden_size)
        self.conv2 = GCNConv(hidden_size, 1024)
        self.linear = torch.nn.Linear(1024, output_size)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = x.float()
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.linear(x)

        return F.log_softmax(x, dim=1)
    
    def reset_parameters(self):
        self.conv1.reset_parameters()
        self.conv2.reset_parameters()
        self.linear.reset_parameters()

class KarateGraph3GCN(torch.nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(KarateGraph3GCN, self).__init__()
        self.conv1 = GCNConv(input_size, hidden_size)
        self.conv2 = GCNConv(hidden_size, 1024)
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

class KarateGraph3Att(torch.nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(KarateGraph3Att, self).__init__()
        self.conv1 = GATConv(input_size, hidden_size)
        self.conv2 = GATConv(hidden_size, 1024)
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

class KarateGraph3SAGE(torch.nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(KarateGraph3SAGE, self).__init__()
        self.conv1 = SAGEConv(input_size, hidden_size)
        self.conv2 = SAGEConv(hidden_size, 1024)
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

class KarateGraph4Att(torch.nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(KarateGraph4Att, self).__init__()
        self.conv1 = GATConv(input_size, hidden_size)
        self.conv2 = GATConv(hidden_size, 1024)
        self.conv3 = GATConv(1024, 512)
        self.conv4 = GATConv(512, output_size)

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
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv4(x, edge_index)

        return F.log_softmax(x, dim=1)
    
    def reset_parameters(self):
        self.conv1.reset_parameters()
        self.conv2.reset_parameters()
        self.conv3.reset_parameters()
        self.conv4.reset_parameters()

class KarateGraph4GCN(torch.nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(KarateGraph4GCN, self).__init__()
        self.conv1 = GCNConv(input_size, hidden_size)
        self.conv2 = GCNConv(hidden_size, 1024)
        self.conv3 = GCNConv(1024, 512)
        self.conv4 = GCNConv(512, output_size)

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
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv4(x, edge_index)

        return F.log_softmax(x, dim=1)
    
    def reset_parameters(self):
        self.conv1.reset_parameters()
        self.conv2.reset_parameters()
        self.conv3.reset_parameters()
        self.conv4.reset_parameters()
class KarateGraph5Att(torch.nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(KarateGraph5Att, self).__init__()
        self.conv1 = GATConv(input_size, hidden_size)
        self.conv2 = GATConv(hidden_size, 1024)
        self.conv3 = GATConv(1024, 512)
        self.conv4 = GATConv(512, 256)
        self.conv5 = GATConv(256, output_size)

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
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv4(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv5(x, edge_index)

        return F.log_softmax(x, dim=1)
    
    def reset_parameters(self):
        self.conv1.reset_parameters()
        self.conv2.reset_parameters()
        self.conv3.reset_parameters()
        self.conv4.reset_parameters()
        self.conv5.reset_parameters()