import torch
from torch.nn import Linear, init, Softplus
import torch.nn.functional as F
from utils import calculate_metrics
    
class MIRT_2PL(torch.nn.Module):
    def __init__(self, ndims, edge_dim, data, degree=2, lambda1=0, lambda2=0):
        super().__init__()
        
        self.degree = degree # allow for non-linear effects   
        self.W = Linear(edge_dim*self.degree, 1)
        self.lambda1 = lambda1
        self.lambda2 = lambda2
            
        self.student_emb = torch.nn.Embedding(data['student'].num_nodes, ndims)
        self.item_emb = torch.nn.Embedding(data['item'].num_nodes, ndims) # adding discrimination parameter
        self.offset_emb = torch.nn.Embedding(data['item'].num_nodes, 1) # adding difficulty parameter
        
        init.normal_(self.student_emb.weight, 0, 1)
        init.normal_(self.item_emb.weight, 0, 1)
            
        self.softplus = Softplus()    
        #self.classifier = EdgeClassifier(hidden_channels[-1], edge_dim)
        #self.classifier = EdgeClassifier_ability(hidden_channels[-1], hidden_channels[-1], edge_dim)

    def get_penalty(self):
        """
        Regularization penalty.
        """
        #x_student = self.student_emb(data['student'].node_id)
        #x_item = self.item_emb(data['item'].node_id)
        
        reg = 0
        if self.lambda1 > 0:
            reg += torch.sum(self.z_student.pow(2.0))/2
        if self.lambda2 > 0:
            reg += torch.sum(self.softplus(self.z_item).pow(2.0))/2

        return reg
            
    def forward(self, data):
        x_student = self.student_emb(data['student'].node_id)
        x_item = self.item_emb(data['item'].node_id)
        x_offset = self.offset_emb(data['item'].node_id)
                
        row, col = data['student', 'responds', 'item'].edge_index
        z_student = x_student[row]
        z_item = x_item[col]
        z_offset = x_offset[col]
        
        # demean the features
        
        edge_feat0 = data['student', 'responds', 'item'].edge_attr
        edge_feat0 = edge_feat0 - torch.mean(edge_feat0, dim=0)
        
        edge_feat = edge_feat0

        if self.degree > 1:
            for i in range(1, self.degree):
                edge_feat = torch.cat([edge_feat, edge_feat0**(i+1)], dim=-1)
            
        z_ability = z_student + self.W(edge_feat)
        
        self.x_dict = {
              'student': x_student,
              'item': x_item,
              'offset': x_offset,
              'ability': z_ability
            }
        z_edge = self.softplus(z_item) * (z_ability) 
        pred = z_edge.sum(dim=-1, keepdim=True) + z_offset

        self.z_student = z_student
        self.z_item = z_item
        
        return pred #, z_dict, z_edge
    
    def get_embeddings(self, data):
        
        self.eval()       
        pred = self.forward(data) # to compute abilities
        x_dict = {k:v.detach().cpu().numpy() for k, v in self.x_dict.items()}
        
        return x_dict

def train_IRT(model, data, optimizer, criterion):
    model.train()
    optimizer.zero_grad()
    # from torch import autograd
    # with autograd.detect_anomaly():
    pred = model(
                data=data
                )
    assert pred.isnan().sum() == 0, 'Output'
    target = data['student', 'item'].y.float()
    #loss = F.cross_entropy(pred, target.long()) + model.get_penalty()
    loss = criterion(pred.squeeze(), target) + model.get_penalty()
    loss.backward()
    optimizer.step()
    return loss


# Test the model function
@torch.no_grad()
def test_IRT(model, data, fold, type):
    model.eval()
    pred = model(
                data=data
                ).cpu()
    target = data['student', 'item'].y.long().cpu().numpy()
    
    preds = calculate_metrics(target, pred.sigmoid())

    metrics = {k+f'_{fold}_{type}':v for k,v in preds.items()}
    metrics['fold'] = fold
    metrics[f'fold_truths_{fold}_{type}'] = target.tolist()
    metrics[f'fold_preds_{fold}_{type}'] = pred.tolist()
    return metrics

        
        
  
