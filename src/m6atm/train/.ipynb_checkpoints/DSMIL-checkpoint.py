import torch
import torch.nn.functional as F
from torch import nn, Tensor


class FCLayer(nn.Module):
    
    def __init__(self, in_size, out_size = 1, dropout = 0.5):
        super(FCLayer, self).__init__()
        self.fc = nn.Sequential(
                    nn.Linear(in_size, 128),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                    nn.Linear(128, out_size)
                  )
        
    def forward(self, feats):
        x = self.fc(feats)
        return feats, x

class BClassifier(nn.Module):
    
    def __init__(self, input_size, output_class, dropout_v = 0.0, nonlinear = True, passing_v = False): # K, L, N
        super(BClassifier, self).__init__()
        
        if nonlinear:
            self.q = nn.Sequential(nn.Linear(input_size, 128),
                                   nn.ReLU(),
                                   nn.Linear(128, 128),
                                   nn.Tanh())
        else:
            self.q = nn.Linear(input_size, 128)
        
        # if dropout
        if passing_v:
            self.v = nn.Sequential(nn.Dropout(dropout_v),
                                   nn.Linear(input_size, input_size),
                                   nn.ReLU())
        else:
            self.v = nn.Identity()
        
        ### 1D convolutional layer that can handle multiple class (including binary)
        self.fcc = nn.Conv1d(output_class, output_class, kernel_size = input_size)  
        
    def forward(self, feats, c): # N x K, N x C
        
        device = feats.device
        V = self.v(feats) # N x V, unsorted
        Q = self.q(feats).view(feats.shape[0], -1) # N x Q, unsorted
        
        # handle multiple classes without for loop
        _, m_indices = torch.sort(c, 0, descending = True) # sort class scores along the instance dimension, m_indices in shape N x C
        m_feats = torch.index_select(feats, dim = 0, index = m_indices[0,:]) # select critical instances, m_feats in shape C x K 
        q_max = self.q(m_feats) # compute queries of critical instances, q_max in shape C x Q
        A = torch.mm(Q, q_max.transpose(0, 1)) # inner product of Q to each entry of q_max, A in shape N x C, each column for unnormalized attention scores
        A = F.softmax( A / torch.sqrt(torch.tensor(Q.shape[1], dtype=torch.float32, device=device)), 0) # normalize attention scores, A in shape N x C, 
        B = torch.mm(A.transpose(0, 1), V) # compute bag representation, B in shape C x V
                
        B = B.view(1, B.shape[0], B.shape[1]) # 1 x C x V
        C = self.fcc(B) # 1 x C x 1
        C = C.view(1, -1)
        
        return C, A, B
    
class DSMIL(nn.Module):
    
    def __init__(self, i_classifier, b_classifier):
        super(DSMIL, self).__init__()
        
        self.i_classifier = i_classifier
        self.b_classifier = b_classifier
        
    def forward(self, x):
        
        feats, classes = self.i_classifier.encoder(x)
        pred_bag, A, B = self.b_classifier(feats, classes)
        
        return classes, pred_bag, A, B
    
    
class ReadClassifier(nn.Module):
    
    def __init__(self, in_size = 1024, out_size = 2, dropout = 0.5):
        super(ReadClassifier, self).__init__()
        
        self.fc = nn.Sequential(nn.Linear(in_size, 512),
                                nn.ReLU(),
                                nn.Dropout(dropout),
                                nn.Linear(512, 128),
                                nn.ReLU(),
                                nn.Dropout(dropout),
                                nn.Linear(128, out_size))
        
    def forward(self, feats):
        
        x = self.fc(feats)
        y_pred = torch.argmax(x, dim = 1)
        y_prob = F.softmax(x, dim = 1)
        
        return y_pred, y_prob