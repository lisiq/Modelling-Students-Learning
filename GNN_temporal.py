import torch
import torch.nn.functional as F
from tqdm import tqdm
from torch_geometric.nn import HeteroConv, GCNConv, SAGEConv, BatchNorm
from torch.nn import Linear, Parameter, Sequential, ELU
from torch_geometric.nn import MessagePassing

class SimpleConv(MessagePassing):
    def __init__(self, in_channels, out_channels):
        super().__init__(aggr='mean')
        if True:
            self.mlp = Sequential(
                       Linear(2 * in_channels, out_channels),
                       ELU(),
                       Linear(out_channels, out_channels))       
        else:
            self.mlp = Linear(2 * in_channels, out_channels)
        
    def forward(self, x, edge_index):
        # x has shape [N, channels]

        return self.propagate(edge_index, x=x)

    def message(self, x_i, x_j):
        
        # assumes first feature is age
        tmp = torch.cat([x_j, x_i - x_j], dim=1)  

        # transform linearly
        return self.mlp(tmp)
                           
class GNNEncoder(torch.nn.Module):
    def __init__(self, hidden_channels):
        super().__init__()
        self.hidden_channels = hidden_channels
        self.layers = torch.nn.ModuleList()
        self.batch_norm_layers = torch.nn.ModuleList()
        
        print(hidden_channels)
        if True:
            conv1 = HeteroConv({
                    ('student', 'responds', 'item'): SAGEConv(hidden_channels[0], hidden_channels[1]),
                    ('item', 'rev_responds', 'student'): SAGEConv(hidden_channels[0], hidden_channels[1]),
                    ('student', 'preceeds', 'student'): SimpleConv(hidden_channels[0], hidden_channels[1])
            }, aggr='mean')  
            #conv2 = HeteroConv({
            #        ('student', 'responds', 'item'): SAGEConv(hidden_channels[1], hidden_channels[1]),
            #        ('item', 'rev_responds', 'student'): SAGEConv(hidden_channels[1], hidden_channels[1])
            #}, aggr='mean')  
            self.layers.append(conv1)
            #self.layers.append(conv2)
            # need to be different?
            self.batch_norm_layers.append(BatchNorm(hidden_channels[1]))
            self.batch_norm_layers.append(BatchNorm(hidden_channels[1]))
            #self.batch_norm_layers.append(BatchNorm(hidden_channels[1]))
            #self.batch_norm_layers.append(BatchNorm(hidden_channels[1]))
        
        else:
            # for now using only one layer
            for i in range(len(hidden_channels)-1):
                print(i, hidden_channels[i], hidden_channels[i+1])
                conv = HeteroConv({
                    ('student', 'responds', 'item'): SAGEConv(hidden_channels[i], hidden_channels[i+1]),
                    ('item', 'rev_responds', 'student'): SAGEConv(hidden_channels[i], hidden_channels[i+1]),
                    ('student', 'preceeds', 'student'): SimpleConv(hidden_channels[i], hidden_channels[i+1])
                }, aggr='mean')  
                self.layers.append(conv)
                self.batch_norm_layers.append(BatchNorm(hidden_channels[i+1]))
            
    def forward(self, x_dict, edge_index_dict):
            
        if False:
            # for now manually
            for i in range(len(self.hidden_channels)-2):
                x_dict = self.layers[i](x_dict, edge_index_dict)
                x_dict['item'] = F.elu(x_dict['item'])
                x_dict['item'] = self.batch_norm_layers[i](x_dict['item'])

        x_dict = self.layers[0](x_dict, edge_index_dict)
        x_dict = { k: F.elu(v) for k, v in x_dict.items() } 
        x_dict['item'] = self.batch_norm_layers[0](x_dict['item'])
        x_dict['student'] = self.batch_norm_layers[1](x_dict['student'])
#        x_dict = self.layers[1](x_dict, edge_index_dict)
#        x_dict = { k: F.elu(v) for k, v in x_dict.items() } 
#        x_dict['item'] = self.batch_norm_layers[2](x_dict['item'])
#        x_dict['student'] = self.batch_norm_layers[3](x_dict['student'])
        return x_dict

class Classifier_heterogeneous(torch.nn.Module):
    def __init__(self, input_channel, edge_dim):
        super().__init__()
        # output is bi-dimensional because and item is either passed or not
        self.linear = Linear(2*input_channel, 16) # changed to 1 from 2
        self.linear_2 = Linear(16, 1)

    def forward(self, x_student, x_item, edge_label_index, edge_feat):
        #print(x_student)
        #print(x_item)
        #print(len(x_student), len(x_item))
        # Convert node embeddings to edge-level representations:
        edge_feat_student = x_student[edge_label_index[0]]
        edge_feat_item = x_item[edge_label_index[1]]

        x = F.elu(self.linear(torch.cat([edge_feat_student, edge_feat_item], dim=-1)))
        # concatenate node representations with edge features, and obtain edge feature
        #if edge_feat is None: # for the synthetic dataset
        #    x =F.elu( self.linear(torch.cat([edge_feat_student, edge_feat_item], dim=-1)))
        #else:
        #    x = self.linear(torch.cat([edge_feat_student, edge_feat, edge_feat_item], dim=-1))
        x = self.linear_2(x)
        return x 