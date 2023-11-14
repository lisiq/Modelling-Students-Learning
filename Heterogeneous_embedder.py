import torch
import torch.nn.functional as F
from torch.nn import Linear

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
            # heads
            ):
        super().__init__()
        self.edge_channel = edge_channel
        self.student_lin = torch.nn.Linear(student_inchannel, hidden_channels[0])
        self.item_lin = torch.nn.Linear(item_inchannel, hidden_channels[0])
        # Since the dataset does not come with rich features, we also learn two
        # embedding matrices for student and items:
        self.student_emb = torch.nn.Embedding(n_students, hidden_channels[0])
        self.item_emb = torch.nn.Embedding(n_items, hidden_channels[0])
        self.encoder = GNNEncoder(hidden_channels)
        self.encoder = to_hetero(self.encoder, metadata , aggr='mean')
        if edge_channel == None:
            self.classifier = Classifier_heterogeneous(hidden_channels[-1],0)
        else:
            # self.classifier = Classifier_heterogeneous(2 * hidden_channels[-1] + edge_channel)
            self.classifier = Classifier_heterogeneous(hidden_channels[-1], edge_channel)


    def forward(self, data):
        x_dict = {
          'student': self.student_lin(data['student'].x) +  self.student_emb(data['student'].node_id),
          'item': self.item_lin(data['item'].x) + self.item_emb(data['item'].node_id),
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
    
    def get_embeddings(self, data):
        self.eval()
        #pred = self.forward(data) if edge embeddings needed
        
        if hasattr(data['student'], 'x'):
            student_x = self.student_lin(data['student'].x) +  self.student_emb(data['student'].node_id)
        else:
            student_x = self.student_emb(data['student'].node_id)

        if hasattr(data['item'], 'x'):
            item_x = self.item_lin(data['item'].x) + self.item_emb(data['item'].node_id)
        else:
            item_x = self.item_emb(data['item'].node_id)
            
        x_dict = {
              'student': student_x,
              'item': item_x
            } 

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
    loss = criterion(pred.squeeze(), target)#, pos_weight=class_weights) # target.long() cross_entropy
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

