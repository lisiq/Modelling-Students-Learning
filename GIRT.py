import torch
import torch.nn.functional as F
from torch.nn import Linear, init, Softplus

from torch_geometric.nn import  to_hetero
from tqdm import tqdm

from GNN import GNNEncoder, Classifier_heterogeneous
from utils import calculate_metrics
    

class EmbedderHeterogeneous(torch.nn.Module):
    def __init__(
            self,
            n_students,
            n_items,
            student_inchannel,
            item_inchannel,
            hidden_channels,
            edge_channel,
            metadata, # data.metadata()
            degree=2,
            lambda1=0, 
            lambda2=0
            # heads
            ):
        super().__init__()
        
        print('Parameters')
        print({'n_students': n_students,
              'n_items': n_items,
              'student_inchannel': student_inchannel,
              'item_inchannel': item_inchannel,
              'hidden_channels': hidden_channels,
              'edge_channel': edge_channel
              })
        
        self.edge_channel = edge_channel
        self.student_lin = torch.nn.Linear(student_inchannel, hidden_channels[0]) if student_inchannel is not None else None
        self.item_lin = torch.nn.Linear(item_inchannel, hidden_channels[0]) if item_inchannel is not None else None
        # Since the dataset does not come with rich features, we also learn two
        # embedding matrices for student and items:
        self.student_emb = torch.nn.Embedding(n_students, hidden_channels[0])
        self.item_emb = torch.nn.Embedding(n_items, hidden_channels[0])
        self.item_offset_emb = torch.nn.Embedding(n_items, 1)
        self.lambda1 = lambda1
        self.lambda2 = lambda2
        self.degree = degree 
        
        init.normal_(self.student_emb.weight, 0, 1)
        init.normal_(self.item_emb.weight, 0, 1)
        init.normal_(self.item_offset_emb.weight, 0, 1)
        
        self.encoder = GNNEncoder(hidden_channels)
        self.encoder = to_hetero(self.encoder, metadata , aggr='mean')
        if edge_channel == None:
            self.classifier = Classifier_heterogeneous(edge_channel*self.degree, 1) 
        else:
            # self.classifier = Classifier_heterogeneous(2 * hidden_channels[-1] + edge_channel)
            self.classifier = Classifier_heterogeneous(edge_channel*self.degree, 1) 

    # Regularization penalty.
    # Not used
    def get_penalty(self):
        """
        Regularization penalty.
        """
        #x_student = self.student_emb(data['student'].node_id)
        #x_item = self.item_emb(data['item'].node_id)
        
        reg = 0
        if self.lambda1 > 0:
            reg += torch.sum(self.student_x.pow(2.0))/2
        if self.lambda2 > 0:
            reg += torch.sum(self.softplus(self.item_x).pow(2.0))/2

        return reg


    def forward(self, data):
        if self.student_lin is not None:
            self.student_x = self.student_lin(data['student'].x) +  self.student_emb(data['student'].node_id)
        else:
            self.student_x = self.student_emb(data['student'].node_id)

        if self.item_lin is not None:
            self.item_x = self.item_lin(data['item'].x) + self.item_emb(data['item'].node_id)
            self.item_offset_x = self.item_offset_emb(data['item'].node_id)
        else:
            self.item_x = self.item_emb(data['item'].node_id)
            self.item_offset_x = self.item_offset_emb(data['item'].node_id)

            
        x_dict = {
              'student': self.student_x,
              'item': self.item_x
            } 
        
        x_dict = self.encoder(x_dict, data.edge_index_dict)

        row, col = data['student', 'responds', 'item'].edge_index
        z_student = x_dict['student'][row]
        z_item = x_dict['item'][col] # discrimination
        z_offset = self.item_offset_x[col] # difficulty
        
        # demean the features        
        edge_feat0 = data['student', 'responds', 'item'].edge_attr
        edge_feat0 = edge_feat0 - torch.mean(edge_feat0, dim=0)
        
        edge_feat = edge_feat0

        if self.degree > 1:
            for i in range(1, self.degree):
                edge_feat = torch.cat([edge_feat, edge_feat0**(i+1)], dim=-1)
            

        
        pred = self.classifier(
            x_student = self.student_x,
            x_item = self.item_x,
            z_student = z_student,
            z_item = z_item,
            z_offset = z_offset,
            edge_feat = edge_feat,
            x_offset = self.item_offset_x
            )        
        
        self.x_dict = self.classifier.x_dict

        return pred    
    
    def get_embeddings(self, data):
        self.eval()
        pred = self.forward(data) # to compute abilities
        x_dict = {k:v.detach().cpu().numpy() for k, v in self.x_dict.items()}
        
        return x_dict

# Train the model function
def train_embedder_heterogeneous(model, data, optimizer, criterion):
    model.train()
    optimizer.zero_grad()
    # from torch import autograd
    # with autograd.detect_anomaly():
    pred = model(
                data=data
                )
    assert pred.isnan().sum() == 0, 'Output'
    target = data['student', 'item'].y.float()
    loss = criterion(pred.squeeze(), target)+ model.get_penalty()#, pos_weight=class_weights) # target.long() cross_entropy
    loss.backward()
    optimizer.step()
    return loss


# Test the model function
@torch.no_grad()
def test_embedder_heterogeneous(model, data, fold, type):
    model.eval()
    pred = model(
                data=data
                ).cpu()
    target = data['student', 'item'].y.long().cpu().numpy()
    
    preds = calculate_metrics(target, pred.sigmoid())

    metrics = {k+f'_{fold}_{type}':v for k,v in preds.items()}
    metrics['fold'] = fold
    # metrics[f'fold_truths_{fold}_{type}'] = target.tolist()
    # metrics[f'fold_preds_{fold}_{type}'] = pred.tolist()
    return metrics



import torch
import torch.nn.functional as F
from torch.nn import Linear, Softplus
from torch_geometric.nn import BatchNorm, SAGEConv
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
            # x = F.dropout(x, training=self.training, p=0.2)

            x = self.batch_norm_layers[i](x)

        x = self.layers[-1](x, edge_index)
        x = F.elu(x)
        # x = F.dropout(x, training=self.training, p=0.2)
        x = self.batch_norm_layers[-1](x)

        return x
    

class Classifier_heterogeneous(torch.nn.Module):
    def __init__(self, input_channel, out_channels):
        super().__init__()
        # output is bi-dimensional because and item is either passed or not
        self.input_channel = input_channel
        self.softplus = Softplus()
        self.W = Linear(input_channel, out_channels)

    def forward(self, x_student, x_item,  z_student, z_item, z_offset, edge_feat, x_offset):

        z_ability = z_student + self.W(edge_feat)

        z_edge = self.softplus(z_item) * (z_ability) 
        pred = z_edge.sum(dim=-1, keepdim=True) + z_offset

        self.z_student = z_student
        self.z_item = z_item

        self.x_dict = {
              'student': x_student,
              'item': x_item,
              'discrimination': self.softplus(x_item),
              'offset': x_offset,
              'ability': z_ability
            }

        return pred 
