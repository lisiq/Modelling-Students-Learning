import torch
import torch.nn.functional as F
from torch.nn import Linear, Softplus
from torch_geometric.nn import GATConv, BatchNorm, SAGEConv
from tqdm import tqdm


class GNNEncoder(torch.nn.Module):
    def __init__(self, hidden_channels):
        super().__init__()
        self.hidden_channels = hidden_channels

        self.layers = torch.nn.ModuleList()
        self.batch_norm_layers = torch.nn.ModuleList()
        
        for i in range(len(hidden_channels)-1):
            self.layers.append(SAGEConv(hidden_channels[i], hidden_channels[i+1]))
            self.batch_norm_layers.append(BatchNorm(hidden_channels[i+1]))

    def forward(self, x, edge_index):

        for i in range(len(self.hidden_channels)-2):
            x = F.elu(self.layers[i](x, edge_index))
            x = F.dropout(x, training=self.training, p=0.4)

            x = self.batch_norm_layers[i](x)

        x = self.layers[-1](x, edge_index)
        x = self.batch_norm_layers[-1](x)

        return x
    

class Classifier_heterogeneous(torch.nn.Module):
    def __init__(self, input_channel, edge_dim):
        super().__init__()
        # output is bi-dimensional because and item is either passed or not
        self.linear = Linear(2*input_channel+edge_dim, 2)

    def forward(self, x_student, x_code, edge_label_index, edge_feat):
        # Convert node embeddings to edge-level representations:
        edge_feat_student = x_student[edge_label_index[0]]
        edge_feat_code = x_code[edge_label_index[1]]



        # concatenate node representations with edge features, and obtain edge feature
        if edge_feat is None: # for the synthetic dataset
            x = self.linear(torch.cat([edge_feat_student, edge_feat_code], dim=-1))
        else:
            x = self.linear(torch.cat([edge_feat_student, edge_feat, edge_feat_code], dim=-1))
        return x  