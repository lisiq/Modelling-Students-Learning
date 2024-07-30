import torch
import torch.nn.functional as F
from torch.nn import Linear, init

from torch_geometric.nn import  to_hetero
from tqdm import tqdm

from GNN import GNNEncoder, Classifier_heterogeneous, Classifier_heterogeneous_irt
from utils import calculate_metrics
    

class EmbedderHeterogeneous(torch.nn.Module):
    def __init__(
            self,
            n_students,
            n_items,
            student_inchannel,
            item_inchannel,
            hidden_channels,
            decoder_channel,
            edge_channel,
            metadata, # data.metadata()
            dropout=0,
            batch_norm=False,
            irt_output=False,
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
              'decoder_channel': decoder_channel,               
              'edge_channel': edge_channel,
              'dropout': dropout,
              'batch_norm': batch_norm,
              'irt_output': irt_output
              })
        
        self.edge_channel = edge_channel
        self.decoder_channel = decoder_channel
        self.student_lin = torch.nn.Linear(student_inchannel, hidden_channels[0], bias=False) if student_inchannel is not None else None
        self.item_lin = torch.nn.Linear(item_inchannel, hidden_channels[0], bias=False) if item_inchannel is not None else None
        # Since the dataset does not come with rich features, we also learn two
        # embedding matrices for student and items:
        self.student_emb = torch.nn.Embedding(n_students, hidden_channels[0])
        self.item_emb = torch.nn.Embedding(n_items, hidden_channels[0])
        self.batch_norm = batch_norm
        self.dropout = dropout
        self.lambda1 = lambda1
        self.lambda2 = lambda2        
        
        init.normal_(self.student_emb.weight, 0, 1)
        init.normal_(self.item_emb.weight, 0, 1)
        
        self.encoder = GNNEncoder(hidden_channels, batch_norm, dropout)
        self.encoder = to_hetero(self.encoder, metadata , aggr='mean')

        if irt_output: 
            classifier = Classifier_heterogeneous
        else:
            classifier = Classifier_heterogeneous_irt
        
        if edge_channel == None:
            edge_channel = 0
            
        self.classifier = classifier(hidden_channels[-1], edge_channel, decoder_channel) 

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
        else:
            self.item_x = self.item_emb(data['item'].node_id)
            
        x_dict = {
              'student': self.student_x,
              'item': self.item_x
            } 
        
        x_dict = self.encoder(x_dict, data.edge_index_dict)#, data.edge_attr_dict)
        if self.edge_channel == None:
            pred = self.classifier(
            x_dict['student'],
            x_dict['item'],
            data['student', 'responds', 'item'].edge_index,
            None
            )   
        else:
            pred = self.classifier(
            x_dict['student'],
            x_dict['item'],
            data['student', 'responds', 'item'].edge_index,
            data['student', 'responds', 'item'].edge_attr
            )      

        return pred    
    
    def get_embeddings(self, data, encoded=True):
        self.eval()
        #pred = self.forward(data) if edge embeddings needed
        
        if self.student_lin is not None:
            student_x = self.student_lin(data['student'].x) +  self.student_emb(data['student'].node_id)
        else:
            student_x = self.student_emb(data['student'].node_id)

        if self.item_lin is not None:
            item_x = self.item_lin(data['item'].x) + self.item_emb(data['item'].node_id)
        else:
            item_x = self.item_emb(data['item'].node_id)
            
        x_dict = {
              'student': student_x,
              'item': item_x
            }
        
        if encoded:
            # embeddings after applying the encoder
            x_dict = self.encoder(x_dict, data.edge_index_dict)
            
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

