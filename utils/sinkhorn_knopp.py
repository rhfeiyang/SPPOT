"""
Authors: Hui Ren
Licensed under the CC BY-NC 4.0 license (https://creativecommons.org/licenses/by-nc/4.0/)
"""
import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
from ot.optim import line_search_armijo

class SK_Class(torch.nn.Module):
    def set_rho(self,rho):
        if hasattr(self,"rho"):
            self.rho=rho
    def get_rho(self):
        if hasattr(self,"rho"):
            return self.rho
        else:
            return None
    def set_gamma(self,gamma):
        if hasattr(self,"gamma"):
            self.gamma=gamma
    def get_gamma(self):
        if hasattr(self,"gamma"):
            return self.gamma
        else:
            return None

class SemiCurrSinkhornKnopp(SK_Class):
    """
    naive SinkhornKnopp algorithm for semi-relaxed curriculum optimal transport, one side is equality constraint, the other side is KL divergence constraint (the algorithm is not stable)
    """
    def __init__(self, num_iters=3, epsilon=0.1, gamma=1, stoperr=1e-6, numItermax=1000, rho=0., semi_use=True, prior = None):
        super().__init__()
        self.num_iters = num_iters
        self.epsilon = epsilon
        self.gamma = gamma
        self.stoperr = stoperr
        self.numItermax = numItermax
        self.rho = rho
        self.b = None
        self.semi_use = semi_use
        self.prior = prior.reshape(-1,1) if prior is not None else None
        print(f"prior: {prior}")
        print(f"semi_use: {semi_use}")
        print(f"epsilon: {epsilon}")
        print(f"sk_numItermax: {numItermax}")

    @torch.no_grad()
    def cost_forward(self, cost, final=True, count=False, pred_order=None):
        cost=cost.double()
        n=cost.shape[0]
        k=cost.shape[1]
        mu = torch.zeros(n, 1).cuda()
        expand_cost = torch.cat([cost, mu], dim=1)
        Q = torch.exp(- expand_cost / self.epsilon)

        # prior distribution
        Pa = torch.ones(n, 1).cuda() / n  # how many samples
        Pb = self.rho * torch.ones(Q.shape[1], 1).cuda() / k # how many prototypes
        if self.prior is not None:
            if pred_order is None:
                pred_distribution = cost.sum(dim=0)
                pred_order = pred_distribution.argsort(descending=True)
            # print(f"pred_order: {pred_sort_order}")
            Pb[pred_order,:] = self.prior * self.rho
        Pb[-1] = 1 - self.rho

        # init b
        b = torch.ones(Q.shape[1], 1).double().cuda() / Q.shape[1] if self.b is None else self.b

        fi = self.gamma / (self.gamma + self.epsilon)
        err = 1
        last_b = b.clone()
        iternum = 0
        while err > self.stoperr and iternum < self.numItermax:
            a = Pa / (Q @ b)
            b =  Pb / (Q.t() @ a)
            if self.semi_use:
                b[:-1,:] = torch.pow(b[:-1,:], fi)

            err = torch.norm(b - last_b)
            last_b = b.clone()
            iternum += 1

        plan = a*Q*b.T
        if final:
            plan*=Q.shape[0]
        self.b=b # for two view speed up
        # print(f"sk_iter: {iternum}"
        # print(iternum,end=" ")
        # scale the plan
        # plan = plan / torch.sum(plan, dim=1, keepdim=True)
        plan = plan[:, :-1].float()
        # loss = (plan * cost).sum()
        # print(f"sk loss: {loss}")
        return (plan, iternum) if count else plan

    @torch.no_grad()
    def forward(self, logits):
        logits = logits.detach()
        logits = -torch.log(torch.softmax(logits, dim=1))
        return self.cost_forward(logits)

class SemiCurrSinkhornKnopp_stable(SemiCurrSinkhornKnopp):
    """
    naive SinkhornKnopp algorithm for semi-relaxed curriculum optimal transport, one side is equality constraint, the other side is KL divergence constraint (the algorithm is not stable)
    """
    def __init__(self, num_iters=3, epsilon=0.1, gamma=1, stoperr=1e-10, numItermax=1000, rho=0., semi_use=True, prior = None):
        super().__init__(num_iters, epsilon, gamma, stoperr, numItermax, rho, semi_use, prior)
        self.u = None
        self.v = None
    def reset(self):
        self.u=None
        self.v=None
        self.b=None
    @torch.no_grad()
    def cost_forward(self, cost, final=True,count=False, pred_order=None):
        cost=cost.double()
        n=cost.shape[0]
        k=cost.shape[1]
        if self.u is not None and self.u.shape[0]!=n:
            self.reset()
        mu = torch.zeros(n, 1).cuda()
        expand_cost = torch.cat([cost, mu], dim=1)

        if self.u is None:
            u = torch.zeros(n, 1).cuda()
            v= torch.zeros(k+1, 1).cuda()
            Q = torch.exp(- expand_cost / self.epsilon)
        else:
            u=self.u
            v=self.v
            Q = torch.exp((u - expand_cost + v.T) / self.epsilon)

        # prior distribution
        Pa = torch.ones(n, 1).cuda() / n  # how many samples
        Pb = self.rho * torch.ones(Q.shape[1], 1).cuda() / k # how many prototypes
        if self.prior is not None:
            if pred_order is None:
                pred_distribution = cost.sum(dim=0)
                pred_order = pred_distribution.argsort(descending=True)
            # print(f"pred_order: {pred_sort_order}")
            Pb[pred_order,:] = self.prior * self.rho
        Pb[-1] = 1 - self.rho
        fi = self.gamma / (self.gamma + self.epsilon)

        b = torch.ones(Q.shape[1], 1, dtype=Q.dtype).cuda() / Q.shape[1] if self.b is None else self.b

        w = torch.exp(v[:-1, :] * (fi - 1) / self.epsilon)
        err = 1
        last_b = b.clone()
        iternum = 0
        stabled=False
        while err > self.stoperr and iternum < self.numItermax:
            a = Pa / (Q @ b)
            b =  Pb / (Q.t() @ a)
            if self.semi_use:
                b[:-1,:] = torch.pow(b[:-1,:], fi) * w

            # print((a*Q*b.T).sum(), err)

            err = torch.norm(b - last_b)
            if max(a.max(), b.max())>1e8:
                # print(f"stabled at {iternum}")
                u+= self.epsilon * torch.log(a)
                v+= self.epsilon * torch.log(b + torch.finfo(b.dtype).eps)
                w *= torch.pow(b[:-1,:], fi-1)
                Q = torch.exp((u - expand_cost + v.T) / self.epsilon)
                b[:,:] = 1
                # a[:,:] = 1
                stabled = True
            else:
                stabled=False

            last_b = b.clone()
            iternum += 1

        plan = Q if stabled else a*Q*b.T
        if final:
            plan*=Q.shape[0]
        self.b=b # for two view speed up
        self.u=u
        self.v=v
        # print(f"sk_iter: {iternum}")
        # print(iternum,end=" ")

        plan = plan[:, :-1].float()
        # loss = (plan * cost).sum()
        # print(f"sk_stable loss: {loss}")
        return (plan, iternum) if count else plan


class MMOT(torch.nn.Module):
    def __init__(self, sk, lam1 = 0.5, numItermax = 10, stoperr=1e-6, lam_fix=False, ema=1):
        super().__init__()
        self.sk = sk
        self.lam_init = lam1
        self.lam1 = lam1
        self.numItermax = numItermax
        self.stoperr = stoperr
        self.lam_fix = lam_fix
        self.ema=ema
        print(f"MM ema factor: {self.ema}")

    def set_rho(self,rho):
        self.sk.set_rho(rho)
        if not self.lam_fix:
            self.lam1 = (1-rho) * self.lam_init

    def set_gamma(self,gamma):
        self.sk.set_gamma(gamma)
    def get_rho(self):
        print(f"MM lam1: {self.lam1}")
        return self.sk.get_rho()
    def get_gamma(self):
        return self.sk.get_gamma()
    def get_grad_omega1(self, Q):
        # get gradient for <S, Q @ Q.T>
        # return (S+S.T) @ Q
        return self.s_st @ Q

    def get_omega1(self, S, Q):
        return (S * (Q @ Q.T)).sum()

    def get_cost(self, C, Q):
        f1_grad = C
        f2_grad = - self.lam1 * self.get_grad_omega1(Q)
        # print(f"min_f1_grad: {f1_grad.min()}, min_f2_grad: {f2_grad.min()}")
        return f1_grad+f2_grad

    def objective_func(self, Q, S, C):
        return (Q * C).sum() - self.lam1 * self.get_omega1(S=S, Q=Q) + self.sk.epsilon * (Q * torch.log(Q + torch.finfo(torch.float).tiny)).sum()

    @torch.no_grad()
    def forward(self, logits: torch.Tensor, S: torch.Tensor):
        # get C
        S=S.detach()
        self.s_st = S + S.T
        logits=logits.detach()
        prob = torch.softmax(logits, dim=1)
        C = -torch.log(prob)
        pred_sort_order = None
        if self.sk.prior is not None:
            pred_distribution = C.sum(dim=0)
            pred_sort_order = pred_distribution.argsort(descending=True)
        # TODO: how to initialize Q ?
        Q = torch.ones_like(C) / (C.shape[0] * C.shape[1]) * self.sk.rho # 40 170.5756
        # Q = prob * self.sk.rho # 38 170.5758

        # last_Q = Q
        old_fval = self.objective_func(Q, S, C)
        sk_iter_count=0
        for i in range(self.numItermax):
            tmp_cost = self.get_cost(C, Q) # gradient of f(Q)
            Q_new, count = self.sk.cost_forward(tmp_cost, final=False, count=True, pred_order=pred_sort_order)
            Q = (1-self.ema) * Q + self.ema * Q_new
            sk_iter_count+=count
            new_fval = self.objective_func(Q, S, C)
            # print(f"mm_fval: {new_fval}")
            fval_delta = abs(new_fval - old_fval)
            if fval_delta < self.stoperr:
                break
            old_fval = new_fval

        print(f"{i+1},{sk_iter_count}", end=" ")
        # final_fval = self.objective_func(Q, S, C)
        return Q * Q.shape[0]


