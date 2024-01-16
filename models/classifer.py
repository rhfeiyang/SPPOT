"""
Authors: Hui Ren
Licensed under the CC BY-NC 4.0 license (https://creativecommons.org/licenses/by-nc/4.0/)
"""
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
class Cos_classifier(nn.Module):
    def __init__(self, dim, num_prototypes, logit_factor=10.0):
        super().__init__()
        self.logit_factor = logit_factor
        self.embedding = nn.utils.weight_norm(nn.Linear(dim, num_prototypes, bias=False))
        self.embedding.weight_g.data.fill_(1)
        self.embedding.weight_g.requires_grad = False

    def forward(self, x):
        x = torch.nn.functional.normalize(x, dim=1, p=2)
        logits= self.embedding(x)
        return logits * self.logit_factor
    def init_prototype(self,prototype):
        assert prototype.shape==self.embedding.weight_v.data.shape
        self.embedding.weight_v.data.copy_(prototype)

def classify(x, prototypes):
    x = torch.nn.functional.normalize(x, dim=1, p=2)
    head = torch.nn.functional.normalize(prototypes, dim=0, p=2)
    logits = x @ head
    return logits

class ETF_Classifier(nn.Module):
    def __init__(self, feat_in, num_classes, fix_bn=False, LWS=False, reg_ETF=False, logit_factor=10.0):
        super(ETF_Classifier, self).__init__()
        self.logit_factor = logit_factor
        self.expand = False
        if feat_in < num_classes:
            print("Warning: feature dimension is smaller than number of classes, ETF can not be initialized. We expand the dimension of feature.")
            self.expand = True
            expand_dim = feat_in
            while expand_dim < num_classes:
                expand_dim = expand_dim * 2
            self.fc = nn.Linear(feat_in, expand_dim)
            feat_in = expand_dim
        P = self.generate_random_orthogonal_matrix(feat_in, num_classes)
        I = torch.eye(num_classes)
        one = torch.ones(num_classes, num_classes)
        M = np.sqrt(num_classes / (num_classes-1)) * \
            torch.matmul(P, I-((1/num_classes) * one))
        try:
            self.ori_M = M.cuda()
        except RuntimeError:
            self.ori_M = M
        self.LWS = LWS
        self.reg_ETF = reg_ETF
        if LWS:
            self.learned_norm = nn.Parameter(torch.ones(1, num_classes))
            self.alpha = nn.Parameter(1e-3 * torch.randn(1, num_classes).cuda())
            self.learned_norm = (F.softmax(self.alpha, dim=-1) * num_classes)
        else:
            self.learned_norm = torch.ones(1, num_classes).cuda()

    def generate_random_orthogonal_matrix(self, feat_in, num_classes):
        # feat in has to be larger than num classes.
        a = np.random.random(size=(feat_in, num_classes))
        P, _ = np.linalg.qr(a)
        P = torch.tensor(P).float()
        assert torch.allclose(torch.matmul(P.T, P), torch.eye(
            num_classes), atol=1e-06), torch.max(torch.abs(torch.matmul(P.T, P) - torch.eye(num_classes)))
        return P

    def forward(self, x):
        if self.expand:
            x = self.fc(x)
        x = x / torch.clamp(
            torch.sqrt(torch.sum(x ** 2, dim=1, keepdims=True)), 1e-8)
        output = torch.matmul(x, self.ori_M)

        return output*self.logit_factor