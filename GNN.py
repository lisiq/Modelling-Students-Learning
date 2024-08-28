import torch
import torch.nn.functional as F
from torch.nn import Linear
from torch_geometric.nn import BatchNorm, SAGEConv
from tqdm import tqdm




class GNNEncoder(torch.nn.Module):
    def __init__(self, hidden_channels, batch_norm, dropout):
        super().__init__()
        self.hidden_channels = hidden_channels
        self.batch_norm = batch_norm
        self.dropout = dropout
        self.layers = torch.nn.ModuleList()

        if self.batch_norm:
            self.batch_norm_layers = torch.nn.ModuleList()
        
        for i in range(len(hidden_channels)-1):
            self.layers.append(SAGEConv(hidden_channels[i], hidden_channels[i+1], aggr='mean'))
            if self.batch_norm:
                self.batch_norm_layers.append(BatchNorm(hidden_channels[i+1]))

    def forward(self, x, edge_index):

        for i in range(len(self.hidden_channels)-2):
            x = F.elu(self.layers[i](x, edge_index))
            if self.dropout > 0:
                x = F.dropout(x, training=self.training, p=self.dropout)
            if self.batch_norm:
                x = self.batch_norm_layers[i](x)

        x = self.layers[-1](x, edge_index)
        x = F.elu(x)
        if self.dropout > 0:
            x = F.dropout(x, training=self.training, p=self.dropout)
        if self.batch_norm:
            x = self.batch_norm_layers[-1](x)

        return x
    

class Classifier_heterogeneous(torch.nn.Module):
    #def __init__(self, input_channel, edge_dim, decoder_channel):
    def __init__(self, input_channel, edge_dim, decoder_channel): ##
        super().__init__()
        # output is bi-dimensional because and item is either passed or not
        self.decoder_channel = decoder_channel
        if decoder_channel == 0:
            self.linear = Linear(2*input_channel+edge_dim, 1) # changed to 1 from 2
        else:            
            self.linear = Linear(2*input_channel+edge_dim, decoder_channel) # changed to 1 from 2
            self.linear_2 = Linear(decoder_channel, 1)

    def forward(self, x_student, x_item, edge_label_index, edge_feat, offset=None):
        # Convert node embeddings to edge-level representations:
        edge_feat_student = x_student[edge_label_index[0]]
        edge_feat_item = x_item[edge_label_index[1]]


        # concatenate node representations with edge features, and obtain edge feature
        if edge_feat is None: # for the synthetic dataset
            x = F.elu(self.linear(torch.cat([edge_feat_student, edge_feat_item], dim=-1)))
        else:
            #x = F.elu(self.linear(torch.cat([edge_feat_student, edge_feat, edge_feat_item], dim=-1)))
            x = F.elu(self.linear(torch.cat([edge_feat_student, edge_feat, edge_feat_item], dim=-1)) )
        # x = F.dropout(x, training=self.training, p=0.2)

        if self.decoder_channel > 0:
            x = self.linear_2(x) 
        
        if offset is not None:
            edge_feat_offset = offset[edge_label_index[1]] ###
            x = x + edge_feat_offset
            
        return x 

class Classifier_heterogeneous_irt(torch.nn.Module):
    def __init__(self, input_channel, edge_dim, decoder_channel):
        super().__init__()
        # output is bi-dimensional because and item is either passed or not
        self.decoder_channel = decoder_channel
        if decoder_channel == 0:
            raise NotImplementedError
            #self.linear = Linear(2*input_channel+edge_dim, 1) # changed to 1 from 2
        else:            
            self.linear = Linear(input_channel+edge_dim, decoder_channel) 
            self.linear_2 = Linear(input_channel, decoder_channel)

    def forward(self, x_student, x_item, edge_label_index, edge_feat, offset=None):
        # Convert node embeddings to edge-level representations:
        edge_feat_student = x_student[edge_label_index[0]]
        edge_feat_item = x_item[edge_label_index[1]]
        
        if self.decoder_channel > 0:
            if edge_feat is None: # for the synthetic dataset
                x = F.elu(self.linear(edge_feat_student))
            else:
                x = F.elu(self.linear((torch.cat([edge_feat_student, edge_feat], dim=-1))))
            y = F.softplus(self.linear_2(edge_feat_item))
            x = torch.sum(x*y, dim=-1, keepdim=True)
        else:             
            raise NotImplementedError
            if edge_feat is None: # for the synthetic dataset
                x = F.elu(self.linear(torch.cat([edge_feat_student, edge_feat_item], dim=-1)))
            else:
                x = F.elu(self.linear(torch.cat([edge_feat_student, edge_feat, edge_feat_item], dim=-1)))
        if offset is not None:
            edge_feat_offset = offset[edge_label_index[1]] ###
            x = x + edge_feat_offset
            
        return x 
        
