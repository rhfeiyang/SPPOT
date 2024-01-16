"""
Authors: Hui Ren
Licensed under the CC BY-NC 4.0 license (https://creativecommons.org/licenses/by-nc/4.0/)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.sinkhorn_knopp import SemiCurrSinkhornKnopp, SemiCurrSinkhornKnopp_stable, MMOT
import itertools as it
from termcolor import colored
from losses.ramps import sigmoid_rampup
import os
from utils.evaluate_utils import get_feature

EPS=1e-8


class SK_loss(nn.Module):
    def __init__(self,p, total_iter=100000,start_iter=0,logits_bank=None,feature_bank=None, prior_pred=None):
        super(SK_loss, self).__init__()
        sk_type = p.sk_type
        factor = p.sk_factor
        prior_distribution=None
        if p.prior_type == "kmeans":
            assert prior_pred is not None
            prior_pred = prior_pred.int()
            prior_distribution = (prior_pred.bincount().float() / len(prior_pred)).sort(descending=True)[0].cuda()

        self.criterion=torch.nn.CrossEntropyLoss().cuda()
        self.supervised = p["supervised"]
        if self.supervised:
            print(colored("WARNING: Using supervised loss", 'red'))
            return
        sk_iter_limit = p["sk_iter_limit"]
        self.num_heads=p["num_heads"]
        if p["ot_frame"] == "mm":
            ot_frame = MMOT
        else:
            raise NotImplementedError
        if sk_type==["uot", "suot"]:
            self.sk = [SemiCurrSinkhornKnopp(gamma=p["gamma_upper"], epsilon=factor, numItermax=sk_iter_limit, prior=prior_distribution) for _ in range(self.num_heads)]
            p["rho_base"] = 1
            p["rho_upper"] = 1
            p["rho_fix"] = True
            print(colored(f"using uot, rho fixed to: {p['rho_base']}", 'red'))
            if sk_type == "suot":
                self.sk = [ot_frame(sk,lam1=p["mm_factor"], numItermax=p["mm_iter_limit"], lam_fix=p["mm_factor_fix"],ema=p["ema_mm"]) for sk in self.sk]
        elif sk_type in ["ppot", "sppot"]:
            self.sk = [SemiCurrSinkhornKnopp(gamma=p["gamma_upper"], epsilon=factor, numItermax=sk_iter_limit, prior=prior_distribution) for _ in range(self.num_heads)]
            if sk_type=="sppot":
                self.sk = [ot_frame(sk,lam1=p["mm_factor"], numItermax=p["mm_iter_limit"], lam_fix=p["mm_factor_fix"],ema=p["ema_mm"]) for sk in self.sk]
        elif sk_type == "ppot_stable":
            self.sk = [SemiCurrSinkhornKnopp_stable(gamma=p["gamma_upper"], epsilon=factor, numItermax=sk_iter_limit, prior=prior_distribution) for _ in range(self.num_heads)]
        elif sk_type == "sppot_stable":
            self.sk = [SemiCurrSinkhornKnopp_stable(gamma=p["gamma_upper"], epsilon=factor, numItermax=sk_iter_limit, prior=prior_distribution) for _ in range(self.num_heads)]
            self.sk = [ot_frame(sk,lam1=p["mm_factor"], numItermax=p["mm_iter_limit"], lam_fix=p["mm_factor_fix"],ema=p["ema_mm"]) for sk in self.sk]
        elif sk_type=="pot":
            self.sk = [SemiCurrSinkhornKnopp(gamma=p["gamma_upper"], epsilon=factor,semi_use=False, numItermax=sk_iter_limit, prior=prior_distribution) for _ in range(self.num_heads)]
        # elif sk_type=="sla":
        #     self.sk = [SinkhornLabelAllocation(p["num_examples"], p["log_upper_bounds"], allocation_param=0, reg=100, update_tol=1e-2,device="cuda") for _ in range(self.num_heads)]
        else:
            raise NotImplementedError
        self.logits_bank=logits_bank
        self.feature_bank = feature_bank
        self.sk_type=sk_type
        self.labels=[[] for _ in range(self.num_heads)] # to compute label acc
        self.target=[] # to compute label acc
        self.i = start_iter
        self.total_iter = total_iter
        self.rho_base=p["rho_base"]
        self.rho_upper = p["rho_upper"] - p["rho_base"]
        self.gamma_base = p["gamma_upper"]
        self.gamma_upper = p["gamma_base"]-p["gamma_upper"] if p["gamma_base"] is not None else None
        self.rho_fix = p["rho_fix"]
        self.rho_strategy = p["rho_strategy"]
        self.label_quality_show = p["label_quality_show"]
        for sk in self.sk:
            sk.set_rho(p["rho_base"])
    @staticmethod
    @torch.no_grad()
    def get_feature_similarity(feature):
        feature=feature.detach()
        feature_norm = F.normalize(feature, dim=1, p=2)
        similarity = feature_norm @ feature_norm.T
        return similarity

    def forward(self, logits, features=None, similarity_matrix=None,target=None, data_idxs=None):
        # For multi-view: logits[view[head]]
        if self.supervised:
            return self.supervised_loss(logits, target)

        batch_size=logits[0][0].shape[0]
        if not self.rho_fix:
            self.set_rho(self.i, self.total_iter)
        if self.gamma_upper is not None:
            self.set_gamma(self.i, self.total_iter)
        feat_sim=None
        if self.logits_bank is None:
            if self.sk_type == "sla":
                assert data_idxs is not None, "data_idxs should not be None for SLA"
                pseudo_labels=[[self.sk[head_id](head, data_idxs) for head_id ,head in enumerate(view)] for view in logits]
            elif self.sk_type in ["sppot", "suot", "sppot_stable"]:
                assert features is not None or similarity_matrix is not None
                if similarity_matrix is not None:
                    similarity = similarity_matrix(data_idxs).cuda(non_blocking=True)
                    feat_sim = [similarity for _ in range(len(logits))]
                else:
                    feat_sim = [self.get_feature_similarity(feat) for feat in features]
                pseudo_labels=[[self.sk[head_id](head, sim) for head_id ,head in enumerate(view)] for view, sim in zip(logits, feat_sim)]
            else:
                pseudo_labels=[[self.sk[head_id](head) for head_id ,head in enumerate(view)] for view in logits]
        elif self.logits_bank is not None and self.sk_type in ["sppot", "suot", "sppot_stable"]:
            assert features is not None or similarity_matrix is not None
            pseudo_labels=[]
            for view_id,view in enumerate(logits):
                pseudo_labels_view=[]
                if similarity_matrix is None:
                    feature_memory, feature_data_idx, feature_memory_idx = self.feature_bank(features[view_id],enqueue=True if view_id==0 else False, data_idxs=data_idxs)
                    feat_sim = self.get_feature_similarity(feature_memory)

                for head_id,head in enumerate(view):
                    logit_memory, logit_memory_data_idx, logit_memory_idx = self.logits_bank[head_id](head,enqueue=True if view_id==0 else False, data_idxs=data_idxs)
                    if feat_sim is None:
                        feat_sim = similarity_matrix(logit_memory_data_idx).cuda(non_blocking=True)
                    pseudo_label=self.sk[head_id](logit_memory, feat_sim)[-batch_size:,:] if logit_memory_idx==0 else self.sk[head_id](logit_memory, feat_sim)[logit_memory_idx-batch_size:logit_memory_idx,:]
                    pseudo_labels_view.append(pseudo_label)
                pseudo_labels.append(pseudo_labels_view)
        else:
            pseudo_labels=[]
            for view_id,view in enumerate(logits):
                pseudo_labels_view=[]
                for head_id,head in enumerate(view):
                    memory, memory_data_idx, memory_idx = self.logits_bank[head_id](head,enqueue=True if view_id==0 else False, data_idxs=data_idxs)
                    pseudo_label=self.sk[head_id](memory)[-batch_size:,:] if memory_idx==0 else self.sk[head_id](memory)[memory_idx-batch_size:memory_idx,:]
                    pseudo_labels_view.append(pseudo_label)
                pseudo_labels.append(pseudo_labels_view)

        ### information display
        if self.i % 100 == 0:
            if pseudo_labels[0][0].shape[-1]<=10:
                print(colored(f"The distribution of pseudo_labels: {pseudo_labels[0][0].sum(dim=0).cpu().numpy()}", 'red'))
        ###
        self.i += 1
        total_loss=[]
        for i,(logits_head, label_head) in enumerate(zip(zip(*logits), zip(*pseudo_labels))):
            loss=0
            if self.label_quality_show:
                self.labels[i].append(label_head[0].cpu())
            for a,b in it.permutations(range(len(logits_head)), 2):
                loss+=self.criterion(logits_head[a],label_head[b])

            total_loss.append(loss)
        if target is not None and self.label_quality_show:
            self.target.append(target.cpu())
        return torch.stack(total_loss)

    def single_forward(self, logits):
        pseudo_label = self.sk[0](logits)
        loss=self.criterion(logits,pseudo_label)
        return loss

    def reset(self):
        '''empty the labels and targets recorded'''
        self.labels=[[] for _ in range(self.num_heads)]
        self.target=[]

    def prediction_log(self,top_rho=False):
        assert len(self.target)>0 and len(self.target) == len(self.labels[0])
        probs = [torch.cat(head,dim=0) for head in self.labels]
        predictions = [torch.argmax(head,dim=1) for head in probs]
        targets = torch.cat(self.target,dim=0)
        combine = [{'predictions': pred, 'probabilities': prob, 'targets': targets} for pred,prob in zip(predictions,probs)]

        if top_rho:
            ### get top 10% confidence samples
            select_num = int(targets.size(0) * self.sk[0].get_rho())
            print(f"top_rho select_num: {select_num}")
            sample_w = [torch.sum(head,dim=1) for head in probs]
            sample_top = [torch.topk(head, select_num, 0, largest=True)[1] for head in sample_w]
            pred_top = [torch.index_select(pred, 0, ind) for pred,ind in zip(predictions, sample_top)]
            prob_top = [torch.index_select(prob, 0, ind) for prob,ind in zip(probs, sample_top)]
            target_top = [torch.index_select(targets, 0, sample) for sample in sample_top]
            combine_top = [{'predictions': pred, 'probabilities': prob, 'targets': target_top[i]} for i,(pred,prob) in enumerate(zip(pred_top, prob_top))]
            ###
            return combine, combine_top
        else:
            return combine

    def set_rho(self, current, total):
        for sk in self.sk:
            if self.rho_strategy == "sigmoid":
                sk.set_rho(sigmoid_rampup(current, total)* self.rho_upper + self.rho_base)
            elif self.rho_strategy == "linear":
                sk.set_rho(current / total * self.rho_upper + self.rho_base)
            else:
                raise NotImplementedError
    def set_gamma(self, current, total):
        val = current / total * self.gamma_upper + self.gamma_base
        if current % 100 == 0:
            print(f"gamma: {val}")
        for sk in self.sk:
            sk.set_gamma(val)

    def get_rho(self):
        return self.sk[0].get_rho()
    def supervised_loss(self, logits, target):
        if self.supervised:
            target=target.cuda(non_blocking=True)
            loss = [0 * len(logits[0])]
            for view in logits:
                for head_id, head in enumerate(view):
                    loss[head_id] += self.criterion(head, target)
            return torch.stack(loss)


class Feat_regulation(nn.Module):
    def __init__(self, p, dataloader, model, feature_dir):
        super(Feat_regulation, self).__init__()
        self.features, _ = get_feature(p, dataloader, model, save_dir=feature_dir)
        self.criterion = nn.MSELoss()
    def forward(self, new_feature, data_idxs):
        old_feature = self.features[data_idxs].cuda()
        loss = 0
        for feat in new_feature:
            loss += self.criterion(feat, old_feature)
        loss /= len(new_feature)
        return loss
