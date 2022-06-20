import sys
sys.path.append("../")

from torch_geometric.nn import global_add_pool, global_mean_pool, GCNConv
import torch

class GCN(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers,
                 dropout, device, return_embeds=False):
        # TODO: Implement a function that initializes self.convs, 
        # self.bns, and self.softmax.

        super(GCN, self).__init__()

        # A list of GCNConv layers
        self.convs = None

        # A list of 1D batch normalization layersin_channels: int, out_channels: int
        self.bns = None

        # The log softmax layer
        self.softmax = None

        ############# Your code here ############
        ## Note:
        ## 1. You should use torch.nn.ModuleList for self.convs and self.bns
        ## 2. self.convs has num_layers GCNConv layers
        ## 3. self.bns has num_layers - 1 BatchNorm1d layers
        ## 4. You should use torch.nn.LogSoftmax for self.softmax
        ## 5. The parameters you can set for GCNConv include 'in_channels' and 
        ## 'out_channels'. For more information please refer to the documentation:
        ## https://pytorch-geometric.readthedocs.io/en/latest/modules/nn.html#torch_geometric.nn.conv.GCNConv
        ## 6. The only parameter you need to set for BatchNorm1d is 'num_features'
        ## For more information please refer to the documentation: 
        ## https://pytorch.org/docs/stable/generated/torch.nn.BatchNorm1d.html
        ## (~10 lines of code)
        in_and_outs = [input_dim, hidden_dim, output_dim]

        self.convs = torch.nn.ModuleList()
        self.convs.append(GCNConv(input_dim, hidden_dim).to(device))
        for i in range(num_layers - 2):
          self.convs.append(GCNConv(hidden_dim, hidden_dim).to(device))
        self.convs.append(GCNConv(hidden_dim, output_dim).to(device))

        self.bns = [torch.nn.BatchNorm1d(hidden_dim).to(device) for i in range(num_layers - 1)]

        #########################################

        # Probability of an element getting zeroed
        self.dropout = dropout

        # Skip classification layer and return node embeddings
        self.return_embeds = return_embeds

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        for bn in self.bns:
            bn.reset_parameters()

    def forward(self, x, adj_t):
        # TODO: Implement a function that takes the feature tensor x and
        # edge_index tensor adj_t and returns the output tensor as
        # shown in the figure.

        out = None

        ############# Your code here ############
        ## Note:
        ## 1. Construct the network as shown in the figure
        ## 2. torch.nn.functional.relu and torch.nn.functional.dropout are useful
        ## For more information please refer to the documentation:
        ## https://pytorch.org/docs/stable/nn.functional.html
        ## 3. Don't forget to set F.dropout training to self.training
        ## 4. If return_embeds is True, then skip the last softmax layer
        ## (~7 lines of code)

        out = x
        for i, gcnconv in enumerate(self.convs):
          out = gcnconv(out, adj_t)

          if i < len(self.convs) - 1:
            out = self.bns[i](out)
            out = torch.nn.ReLU()(out)
            out = torch.nn.functional.dropout(out, self.dropout, training=self.training)
          elif self.return_embeds is False:
            out = torch.nn.functional.log_softmax(out, -1)
        
          out = out

        #########################################

        return out

### GCN to predict graph property
class StanfordGraph(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers, dropout, device):
        super(StanfordGraph, self).__init__()

        # Node embedding model
        # Note that the input_dim and output_dim are set to hidden_dim
        self.gnn_node = GCN(input_dim, hidden_dim,
            hidden_dim, num_layers, dropout, device, return_embeds=True)

        self.pool = None
        ############# Your code here ############
        ## Note:
        ## 1. Initialize self.pool as a global mean pooling layer
        ## For more information please refer to the documentation:
        ## https://pytorch-geometric.readthedocs.io/en/latest/modules/nn.html#global-pooling-layers
        #self.pool = global_mean_pool
        #########################################

        # Output layer
        self.linear = torch.nn.Linear(hidden_dim, output_dim)

        
    def reset_parameters(self):
        self.gnn_node.reset_parameters()
        self.linear.reset_parameters()

        
    def forward(self, pyg_graph):

        embed, edge_index = pyg_graph.x, pyg_graph.edge_index.long()

        out = None

        out = self.gnn_node(embed, edge_index)
        #out = self.pool(out)
        out = self.linear(out)

        return out