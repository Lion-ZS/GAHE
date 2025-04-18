import numpy as np
import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
from hyperbolic import expmap0, project,logmap0,expmap1,logmap1, mobius_add

class MyLoss(torch.nn.Module):
    def __init__(self):
        super(MyLoss, self).__init__()
        return

    def forward(self, pred1, tar1):
        pred1 = F.softmax(pred1, dim=1)
        loss = -torch.log(pred1[tar1 == 1]).sum()
        return loss


class GAHE(nn.Module):
    def __init__(self, K, n_r, n_v, rdim, vdim, n_parts, max_ary, cu,rate,device, **kwargs):
        super(GAHE, self).__init__()
        self.loss = MyLoss()
        self.device = device
        self.n_parts = n_parts
        self.c = cu
        self.rate=rate
        self.n_ary = max_ary
        self.RolU = nn.Embedding(K, embedding_dim=rdim, padding_idx=0) 
        self.RelV = torch.nn.ParameterList([torch.nn.Parameter((torch.rand(n_r, arity, K, requires_grad=True)).to(device)) for arity in range(2, max_ary + 1)])
        self.RelV1 = torch.nn.ParameterList([torch.nn.Parameter((torch.rand(n_r, arity, rdim, arity, requires_grad=True)).to(device)) for arity in range(2, max_ary + 1)])
        self.Val = nn.Embedding(n_v, embedding_dim=vdim, padding_idx=0)  
        self.Plist = torch.nn.ParameterList([torch.nn.Parameter((torch.rand(K, arity, self.n_parts, requires_grad=True)).to(device))
             for arity in range(2, max_ary + 1)])
        self.max_ary = max_ary
        self.drop_role, self.drop_value = torch.nn.Dropout(kwargs["drop_role"]), torch.nn.Dropout(kwargs["drop_ent"])
        self.init_weight()
        embedding_dim = rdim

        self.mlp_q = torch.nn.Sequential(
            torch.nn.Linear(embedding_dim, embedding_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(embedding_dim, embedding_dim)
        ).to(device)

        self.W_I = torch.nn.Linear(embedding_dim * 3, embedding_dim).to(device)
        self.W_Spe_E = torch.nn.Linear(embedding_dim * 3, embedding_dim).to(device)
        self.W_Spe_H = torch.nn.Linear(embedding_dim * 3, embedding_dim).to(device)
        self.W_Spe_S = torch.nn.Linear(embedding_dim * 3, embedding_dim).to(device)
        self.W_U = torch.nn.Linear(embedding_dim * 4, embedding_dim).to(device)


    def compute_q(self,h):
        return self.mlp_q(h)

    def euclidean_interaction(self,h, q):
        h_E1 = (h - q) * torch.sigmoid(h - q)
        q_E1 = q + h_E1
        h_E2 = (q_E1 - h) * torch.sigmoid(q_E1 - h)
        h_final = h + h_E2
        return h_final

    def h1_interaction(self,h, q, c=1.0):
        h_H = expmap0(h, c)
        q_H = expmap0(q, c)

        h_H1 = logmap0(mobius_add(h_H, -q_H, c), c) * torch.sigmoid(logmap0(mobius_add(h_H, -q_H, c), c))
        q_H1 = mobius_add(q_H, expmap0(h_H1, c), c)

        h_H2 = logmap0(mobius_add(q_H1, -h_H, c), c) * torch.sigmoid(logmap0(mobius_add( q_H1,-h_H, c), c))
        h_final = mobius_add(h_H, expmap0(h_H2, c), c)
        h_final = logmap0(h_final, c)  
        return h_final

    def h2_interaction(self,h, q, u=-1.0):
        c = abs(u)
        h_S = expmap1(h, c)
        q_S = expmap1(q, c)

        h_S1 = logmap1(mobius_add(h_S, -q_S, c), c) * torch.sigmoid(logmap1(mobius_add(h_S, -q_S, c), c))
        q_S1 = mobius_add(q_S, expmap1(h_S1, c), c)

        h_S2 = logmap1(mobius_add(q_S1, -h_S, c), c) * torch.sigmoid(logmap1(mobius_add( q_S1,-h_S, c), c))
        h_final = mobius_add(h_S, expmap1(h_S2, c), c)
        h_final = logmap1(h_final, c)  
        return h_final

    def multi_space_fusion(self,h_e, h_h, h_s):
        h_concat = torch.cat([h_e, h_h, h_s], dim=-1)

        h_I = self.W_I(h_concat)
        hE_Spe = h_e * torch.sigmoid(self.W_Spe_E(h_concat))
        hH_Spe = h_h * torch.sigmoid(self.W_Spe_H(h_concat))
        hS_Spe = h_s * torch.sigmoid(self.W_Spe_S(h_concat))

        h_final = self.W_U(torch.cat([h_I, hE_Spe, hH_Spe, hS_Spe], dim=-1))
        return h_final

    def init_weight(self):
        nn.init.xavier_normal_(self.RolU.weight.data)
        nn.init.xavier_normal_(self.Val.weight.data)
        for i in range(2, self.max_ary + 1):
            nn.init.xavier_normal_(self.Plist[i - 2])
            nn.init.xavier_normal_(self.RelV[i-2])

    def Sinkhorn(self, X):
        S = torch.exp(X)
        S = S / S.sum(dim=[1, 2], keepdim=True).repeat(1, S.shape[1], S.shape[2])
        return S

    def givens_trans(self,r, x):
        a,b,c=r.shape[0],r.shape[1],r.shape[2]
        r1=r
        givens = r1.reshape((r.shape[0], -1, 2))
        givens = givens / torch.norm(givens, p=2, dim=-1, keepdim=True)
        x = x.reshape((r.shape[0], -1, 2))
        x_rot = givens[:, :, 0:1] * x + givens[:, :, 1:] * torch.cat((-x[:, :, 1:], x[:, :, 0:1]), dim=-1)
        return x_rot.view((r.shape[0], b,c))

    def forward(self, rel_idx, value_idx, miss_value_domain):
        n_b, n_v, arity = value_idx.shape[0], self.Val.weight.shape[0], value_idx.shape[1]+1
        RelV = self.RelV[arity-2][rel_idx] 
        RelV_bi = self.RelV1[arity-2][rel_idx]  
        RelV = F.softmax(RelV, dim=2)
        role = torch.matmul(RelV, self.RolU.weight)         
        value = self.Val(value_idx)                       
        role, value = self.drop_role(role), self.drop_value(value)
        value = value.reshape(n_b, arity-1, self.n_parts, -1)
        Plist = self.Sinkhorn(self.Plist[arity-2])
        P = torch.einsum('bak,kde->bade', RelV, Plist)
        idx = [i for i in range(arity) if i + 1 != miss_value_domain]
        V0 = torch.einsum('bijk,baij->baik', value, P[:, :, idx, :])   
        V1 = torch.prod(V0, dim=2)         
        V0_miss = torch.einsum('njk,baj->bnak', self.Val.weight.reshape(n_v, self.n_parts, -1),
                               P[:, :, miss_value_domain - 1, :])
        score=0
        for index in range(V0.shape[2]):
            h = V0[:, :, index, :]  
            q = self.compute_q(h)       
            h_E = self.euclidean_interaction(h, q)
            h_H = self.h1_interaction(h, q, c=self.c) 
            h_S = self.h2_interaction(h, q, u=-self.c) 
            h_multi_space = self.multi_space_fusion(h_E, h_H, h_S)
            h_multi_space=F.normalize(h_multi_space, p=2, dim=2)
            agg=self.rate*h+(1-self.rate)*h_multi_space
            score += torch.einsum('bak,bnak,bak->bn', self.givens_trans(RelV_bi[:,:,:,index],agg), V0_miss,role) 
        return score
