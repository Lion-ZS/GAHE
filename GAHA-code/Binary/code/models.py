from abc import ABC, abstractmethod
from typing import Tuple, List, Dict

import torch
from torch import nn
from hyperbolic import expmap0, project,logmap0,expmap1,logmap1, mobius_add
from tqdm import tqdm
import torch.nn.functional as F

class KBCModel(nn.Module, ABC):
    def get_ranking(
            self, queries: torch.Tensor,
            filters: Dict[Tuple[int, int], List[int]],
            batch_size: int = 1000, chunk_size: int = -1
    ):
        ranks = torch.ones(len(queries))
        with tqdm(total=queries.shape[0], unit='ex') as bar:
            bar.set_description(f'Evaluation')
            with torch.no_grad():
                b_begin = 0
                while b_begin < len(queries):
                    these_queries = queries[b_begin:b_begin + batch_size]
                    target_idxs = these_queries[:, 2].cpu().tolist()
                    scores, _ = self.forward(these_queries)
                    targets = torch.stack([scores[row, col] for row, col in enumerate(target_idxs)]).unsqueeze(-1)

                    for i, query in enumerate(these_queries):
                        filter_out = filters[(query[0].item(), query[1].item())]
                        filter_out += [queries[b_begin + i, 2].item()]   
                        scores[i, torch.LongTensor(filter_out)] = -1e6
                    ranks[b_begin:b_begin + batch_size] += torch.sum(
                        (scores >= targets).float(), dim=1
                    ).cpu()
                    b_begin += batch_size
                    bar.update(batch_size)
        return ranks




class GAHE(KBCModel):
    def __init__(
            self, sizes: Tuple[int, int, int], rank: int,
            init_size: float = 1e-3, rate: float = 0.5, c: float = 0.1
    ):
        super(GAHE, self).__init__()
        self.sizes = sizes
        self.rank = rank
        self.rate = rate
        self.c = c
        self.embeddings = nn.ModuleList([
            nn.Embedding(s, 2 * rank, sparse=True)
            for s in sizes[:2]
        ])
        self.embeddings[0].weight.data *= init_size
        self.embeddings[1].weight.data *= init_size

        self.mlp_q = nn.Sequential(
            nn.Linear(rank, rank),
            nn.ReLU(),
            nn.Linear(rank, rank)
        )

        self.W_I = nn.Linear(rank * 3, rank)
        self.W_Spe_E = nn.Linear(rank * 3, rank)
        self.W_Spe_H = nn.Linear(rank * 3, rank)
        self.W_Spe_S = nn.Linear(rank * 3, rank)
        self.W_U = nn.Linear(rank * 4, rank)

    def compute_q(self, h):
        return self.mlp_q(h)

    def euclidean_interaction(self, h, q):
        h_E1 = (h - q) * torch.sigmoid(h - q)
        q_E1 = q + h_E1
        h_E2 = (q_E1 - h) * torch.sigmoid(q_E1 - h)
        return h + h_E2

    def h1_interaction(self, h, q):
        h_H = expmap0(h, self.c)
        q_H = expmap0(q, self.c)
        delta = logmap0(mobius_add(h_H, -q_H, self.c), self.c)
        q_H1 = mobius_add(q_H, expmap0(delta * torch.sigmoid(delta), self.c), self.c)
        delta2 = logmap0(mobius_add(q_H1, -h_H, self.c), self.c)
        h_final = mobius_add(h_H, expmap0(delta2 * torch.sigmoid(delta2), self.c), self.c)
        return logmap0(h_final, self.c)

    def h2_interaction(self, h, q):
        c_s = abs(-self.c)
        h_S = expmap1(h, c_s)
        q_S = expmap1(q, c_s)
        delta = logmap1(mobius_add(h_S, -q_S, c_s), c_s)
        q_S1 = mobius_add(q_S, expmap1(delta * torch.sigmoid(delta), c_s), c_s)
        delta2 = logmap1(mobius_add(q_S1, -h_S, c_s), c_s)
        h_final = mobius_add(h_S, expmap1(delta2 * torch.sigmoid(delta2), c_s), c_s)
        return logmap1(h_final, c_s)

    def multi_space_fusion(self, h_e, h_h, h_s):
        h_concat = torch.cat([h_e, h_h, h_s], dim=-1)
        h_I = self.W_I(h_concat)
        hE_Spe = h_e * torch.sigmoid(self.W_Spe_E(h_concat))
        hH_Spe = h_h * torch.sigmoid(self.W_Spe_H(h_concat))
        hS_Spe = h_s * torch.sigmoid(self.W_Spe_S(h_concat))
        return self.W_U(torch.cat([h_I, hE_Spe, hH_Spe, hS_Spe], dim=-1))

    def forward(self, x):
        lhs = self.embeddings[0](x[:, 0])
        rel = self.embeddings[1](x[:, 1])
        rhs = self.embeddings[0](x[:, 2])
        lhs_real, lhs_imag = lhs[:, :self.rank], lhs[:, self.rank:]
        rel_real, rel_imag = rel[:, :self.rank], rel[:, self.rank:]
        rhs_real, rhs_imag = rhs[:, :self.rank], rhs[:, self.rank:]
        q = self.compute_q(lhs_real)
        h_e = self.euclidean_interaction(lhs_real, q)
        h_h = self.h1_interaction(lhs_real, q)
        h_s = self.h2_interaction(lhs_real, q)
        h_fused = self.multi_space_fusion(h_e, h_h, h_s)
        h_fused=F.normalize(h_fused, p=2, dim=1)
        lhs_real = self.rate * lhs_real + (1 - self.rate) * h_fused
        q_imag = self.compute_q(lhs_imag)
        hE_imag = self.euclidean_interaction(lhs_imag, q_imag)
        hH_imag = self.h1_interaction(lhs_imag, q_imag)
        hS_imag = self.h2_interaction(lhs_imag, q_imag)
        fused_imag = self.multi_space_fusion(hE_imag, hH_imag, hS_imag)
        fused_imag=F.normalize(fused_imag, p=2, dim=1)
        lhs_imag = self.rate * lhs_imag + (1 - self.rate) * fused_imag
        to_score = self.embeddings[0].weight
        to_score_real = to_score[:, :self.rank]
        to_score_imag = to_score[:, self.rank:]

        score = (
            (lhs_real * rel_real - lhs_imag * rel_imag) @ to_score_real.t() +
            (lhs_real * rel_imag + lhs_imag * rel_real) @ to_score_imag.t()
        )

        return score, [
            (torch.sqrt(lhs_real ** 2 + lhs_imag ** 2),
             torch.sqrt(rel_real ** 2 + rel_imag ** 2),
             torch.sqrt(rhs_real ** 2 + rhs_imag ** 2))
        ]
