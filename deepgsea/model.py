import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import pdb

class MultiChannelLinear(nn.Module):
    
    def __init__(self, in_dim, out_dim, n_channel=1):
        super(MultiChannelLinear, self).__init__()
        
        #initialize weights
        self.w = torch.nn.Parameter(torch.zeros(n_channel, out_dim, in_dim))
        self.b = torch.nn.Parameter(torch.zeros(1, n_channel, out_dim))
        
        #change weights to kaiming
        nn.init.kaiming_uniform_(self.w, a=math.sqrt(3))
        fan_in, _ = torch.nn.init._calculate_fan_in_and_fan_out(self.w)
        bound = 1 / math.sqrt(fan_in)
        nn.init.uniform_(self.b, -bound, bound)
    
    def forward(self, x):
        '''
            args:
                x: input, whose shape can be 
                    batch_size, (channel), in_dim
            return:
                output, whose shape will be
                    batch_size, (channel), out_dim
        '''
        return (self.w * x.unsqueeze(-2)).sum(-1) + self.b


class DeepGSEA(nn.Module): # mask on importance estimation for subsampling (remember to average the scores with correct number of valid concepts)
    def __init__(self, M, n_gene, h_dim, z_dim, n_layer_enc, n_class, n_proto=1, sigma=0.01, dropout=0.0, d_min=1):
        
        '''
            M (size: [n_concept, n_gene]): concept masks, which are used to mask unwanted variations that are not directly related to the biological concept differences of interest
            n_gene: the number of genes in total, which is also the dimension of the input
            h_dim: the dimension of hidden layers
            z_dim: the dimension of concept embeddings of the cell / the dimension of concepet prototypes
            n_layer_enc: the number of hidden layers for the backbone encoder + 1 (1 for the embedding layer)
            # n_layer_imp: the number of hidden layers for the importance scorer + 1 (1 for the embedding layer)
            n_class: the number of classes of interest (e.g., different cell types, patient phenotypes)
            n_proto: the number of prototypes for each class of interest (1 for cell-type identification tasks, 'no. of cell types' for other tasks)
            d_min: threshold for prototype distance penalty
        '''
        
        super(DeepGSEA, self).__init__()

        self.M = M
        self.n_concept = len(M)
        self.n_gene = n_gene
        self.h_dim = h_dim
        self.z_dim = z_dim
        self.n_layer_enc = n_layer_enc
        self.n_class = n_class
        self.n_proto = n_proto
        self.d_min = d_min
        self.sigma = sigma
            
        assert self.n_layer_enc > 0
        # assert self.n_layer_imp > 0

        self.activate = nn.LeakyReLU()

        self.backbone = nn.Sequential(
                    nn.Linear(self.n_gene, self.h_dim), self.activate,
                    *([item for _ in range(self.n_layer_enc - 1) for item in (nn.Linear(self.h_dim, self.h_dim), self.activate)]),
        )
        
        self.c_heads = MultiChannelLinear(self.h_dim, self.z_dim, self.n_concept)

        self.importance = nn.parameter.Parameter(torch.zeros(1, self.n_concept), requires_grad = True)

        self.prototypes = nn.parameter.Parameter(torch.empty(self.n_concept, self.n_class, self.n_proto, self.z_dim), requires_grad = True)
        self.logvar = nn.parameter.Parameter(torch.empty(self.n_concept, self.n_class, self.n_proto), requires_grad = True)
        self.c_bias = nn.parameter.Parameter(torch.zeros(self.n_concept, self.n_class), requires_grad = True)
        self.bias = nn.parameter.Parameter(torch.zeros(self.n_class), requires_grad = True)

        self.dropout = nn.Dropout(p=dropout)

        for layer in self.backbone:
            if type(layer) == nn.Linear:
                nn.init.xavier_normal_(layer.weight)
        
        nn.init.xavier_normal_(self.prototypes)
        nn.init.xavier_normal_(self.logvar)
        nn.init.xavier_normal_(self.importance)

    def forward(self, X, y=None, label=None):
        '''
            Input:
                X (size: [n_cell, n_gene]): the transcriptal information of a batch of cells
            Output:
                logits (size: [n_cell, n_class]): estimated probabilities (before softmax activation) (size)
                c_logits (size: [n_cell, n_concept, n_class]): estimated probabilities based on each concept only
                losses: a dictionary of various losses
        '''

        Z = self.encode(X) # (n_cell, n_concept, z_dim)
        imp_scores = self.compute_imp(X) # (n_cell, n_concept)
        imp_scores = self.dropout(imp_scores)
        c2p_dist_sq = (torch.pow(Z[:, :, None, None, :] - self.prototypes[None, :, :, :, :], 2).sum(-1)) # (n_cell, n_concept, n_class, n_proto)

        c_logits = - (c2p_dist_sq / (2 * torch.exp(self.logvar)[None])).min(dim=-1)[0] + self.c_bias[None] # (n_cell, n_concept, n_class)
        logits = torch.log((torch.exp(c_logits) * imp_scores[:,:,None] + 1e-16).sum(1)) + self.bias[None] # (n_cell, n_class)

        if label is not None:
            p2p_loss = nn.CrossEntropyLoss(reduction="none")(- (c2p_dist_sq / (2 * torch.exp(self.logvar)[None])).min(dim=-2)[0].reshape(-1, self.n_proto), label.repeat_interleave(self.n_concept)).reshape(-1, self.n_concept).sum(1).mean() # (n_cell, n_concept, n_proto) -> 1
        else:
            p2p_dists = (torch.pow(self.prototypes[:,:,:,None,:] - self.prototypes[:,:,None,:,:], 2).sum(-1)+1e-16).sqrt() # (n_concept, n_class, n_proto, n_proto)
            p2p_loss = ((self.d_min - p2p_dists > 0) * (self.d_min - p2p_dists)).pow(2).mean()

        if y is not None:

            if label is not None:
                c2p_loss = c2p_dist_sq[torch.arange(len(X)),:,y,label].mean()
                p_idx = F.one_hot(y[:,None].repeat((1,self.n_concept)),num_classes=self.n_class)[:,:,:,None] * F.one_hot(label[:,None].repeat((1,self.n_concept)),num_classes=self.n_proto)[:,:,None,:]
                p2c_vec = ((Z[:,:,None, None,:] - self.prototypes[None,:,:,:,:]) * p_idx[:,:,:,:,None]).sum(0) / (p_idx.sum(0)[:,:,:,None] + 1e-16) # (n_concept, n_class, n_proto, z_dim)
                p2c_loss = p2c_vec.pow(2).sum(-1).mean()
            else:
                c2p_loss = c2p_dist_sq[torch.arange(len(X)),:,y].min(dim=-1)[0].mean() # mean of (n_cell, n_concept)
                # p2c_loss: minimize the distance between each prototype and the mean of cells to which it is the closest (the cells that it represent)
                p_idx = F.one_hot(y[:,None].repeat((1,self.n_concept)),num_classes=self.n_class)[:,:,:,None] * F.one_hot(c2p_dist_sq[torch.arange(len(X)),:,y].min(dim=-1)[1],num_classes=self.n_proto)[:,:,None,:] # (n_cell, n_concept, n_class, n_proto)
                p2c_vec = ((Z[:,:,None, None,:] - self.prototypes[None,:,:,:,:]) * p_idx[:,:,:,:,None]).sum(0) / (p_idx.sum(0)[:,:,:,None] + 1e-16) # (n_concept, n_class, n_proto, z_dim)
                p2c_loss = p2c_vec.pow(2).sum(-1).mean()

            losses = {
                "p2p_loss": p2p_loss,
                "c2p_loss": c2p_loss,
                "p2c_loss": p2c_loss
            }
        else:
            losses = {}
        return logits, c_logits, losses
    
    def encode(self, X):
        '''
            Input:
                X (size: [n_cell, n_gene]): the transcriptal information of a batch of cells
            Output:
                Z (size: [n_cell, n_concept, z_dim]): the encoded concept information of the cells
        '''
        res = self.backbone(X[:,None,:]*self.M[None]) # (n_cell, n_concept, h_dim)
        Z = self.c_heads(res)
        return Z
    
    def compute_imp(self, X=None):
        return torch.abs(self.importance)