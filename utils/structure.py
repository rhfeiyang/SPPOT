"""
Authors: Hui Ren
Licensed under the CC BY-NC 4.0 license (https://creativecommons.org/licenses/by-nc/4.0/)
"""

import torch
import torch.nn.functional as F
import os
import sys
if __name__ == "__main__":
    sys.path[0] = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

from utils.evaluate_utils import get_feature
# from sklearn.cluster import KMeans
class Similarity_matrix:
    def __init__(self, similarity_matrix, p=None):
        self.similarity_matrix = similarity_matrix
        self.similarity_type = p["similarity_type"]
        self.clip_thres=p["threshold_similarity"]
        self.offset_similarity = p["offset_similarity"]
        self.scale_similarity = p["scale_similarity"]
        print(f"similarity operation type: {self.similarity_type}")
        if "clip" in self.similarity_type:
            if self.clip_thres > -1:
                self.similarity_matrix[self.similarity_matrix<self.clip_thres]=0
        if "upclip" in self.similarity_type:
            if self.clip_thres < 1:
                self.similarity_matrix[self.similarity_matrix>self.clip_thres]=self.clip_thres
        if "affine" in self.similarity_type:
            self.similarity_matrix = (self.similarity_matrix+self.offset_similarity) * self.scale_similarity
        if "identity" in self.similarity_type:
            self.similarity_matrix += (self.similarity_matrix.shape[0] - 2) * torch.eye(self.similarity_matrix.shape[0])
        if "rmDiag" in self.similarity_type:
            self.similarity_matrix[torch.eye(self.similarity_matrix.shape[0]).bool()] = 0
        if "topk" in self.similarity_type:
            select_num = p["topk_similarity"] if "rmDiag" in self.similarity_type else p["topk_similarity"]+1
            print(f"similarity topk select_num:{select_num}")
            indices = self.similarity_matrix.topk(select_num, dim=1)[1]
            self.similarity_matrix[torch.arange(self.similarity_matrix.shape[0]).unsqueeze(1), indices] = 1
        if "knn" in self.similarity_type:
            # set not in topk to 0
            select_num = p["topk_similarity"] if "rmDiag" in self.similarity_type else p["topk_similarity"]+1
            print(f"similarity knn select_num:{select_num}")
            indices = self.similarity_matrix.topk(select_num, dim=1)[1]
            knn_similarity = torch.zeros_like(self.similarity_matrix)
            knn_similarity[torch.arange(self.similarity_matrix.shape[0]).unsqueeze(1), indices] = self.similarity_matrix[torch.arange(self.similarity_matrix.shape[0]).unsqueeze(1), indices]
            self.similarity_matrix = knn_similarity
        if "consistency" in self.similarity_type:
            # make to symmetric
            mask = self.similarity_matrix.T.abs()<1e-5
            self.similarity_matrix[mask] = 0


    def __call__(self, index):
        ret = self.similarity_matrix[index][:, index].float()
        return ret
    def __len__(self):
        return len(self.similarity_matrix)

@torch.no_grad()
def get_dataset_feature_similarity(p, dataloader, model, save_dir=None):
    features, targets = get_feature(p, dataloader, model, save_dir=save_dir)

    if p["similarity_type"] == "oracle":
        target_group = torch.stack([(targets==i).float()*2-1 for i in range(p['num_classes'][0])])
        similarity_matrix = target_group[targets]
        return Similarity_matrix(similarity_matrix, p=p)

    if p["kernel_type"] == "cos":
        features = F.normalize(features, dim=1)
        similarity = features @ features.T
    elif p["kernel_type"] == "linear":
        similarity = features @ features.T
    elif p["kernel_type"] == "gauss":
        similarity = torch.exp(-torch.cdist(features, features, p=2)**2 / (2 * p["kernel_param"]**2 * model.module.backbone_dim))
    elif p["kernel_type"] == "laplace":
        similarity = torch.exp(-torch.cdist(features, features, p=2) / (p["kernel_param"] * model.module.backbone_dim))
    elif p["kernel_type"] == "abel":
        similarity = torch.exp(-torch.cdist(features, features, p=1) / p["kernel_param"] * model.module.backbone_dim)

    else:
        raise NotImplementedError

    return Similarity_matrix(similarity, p=p)

def similarity_metric(pred_similarity, target_similarity):
    # assume target_similarity is 0-1
    dist = (target_similarity-pred_similarity).abs()
    sum0 = (target_similarity == 0).sum()
    sum1 = (target_similarity == 1).sum()
    weight = torch.where(target_similarity==0, 0.5 * (1/sum0), 0.5 * (1/sum1))
    score = ((1-dist)*weight).sum()
    return score

import faiss
def feature_cluster(features, feature_path=None, cluster_num=100):
    assert features is not None or feature_path is not None
    if feature_path is not None:
        features = torch.load(feature_path, map_location="cpu")

    features = F.normalize(features, dim=1, p=2)
    # cluster
    kmeans = faiss.Kmeans(features.shape[1], cluster_num, niter=100, verbose=False, gpu=False, spherical=True)

    kmeans.train(features)
    # centroid = faiss.vector_float_to_array(clus.centroids).reshape(cluster_k, dim)
    D, I = kmeans.index.search(features, 1) # for each sample, find cluster distance and assignments
    I = torch.Tensor(I[:,0])
    return I


if __name__ == "__main__":
    paths = [
            "/public/home/renhui/code/Imbalanced_clustering/PPOT_structure/data/backbone/init_feature_result_iNature_im_100_0.01.pth",
            "/public/home/renhui/code/Imbalanced_clustering/PPOT_structure/data/backbone/init_feature_result_iNature_im_500_0.01.pth",
            "/public/home/renhui/code/Imbalanced_clustering/PPOT_structure/data/backbone/init_feature_result_iNature_im_1000_0.01.pth"]
    class_nums = [100,500,1000]
    for path,class_num in zip(paths, class_nums):
        feature_sim = torch.load(path)
        features = feature_sim["features"]
        targets = feature_sim["targets"]
        p = {"similarity_type":["rmDiag","knn"],"topk_similarity":20,"threshold_similarity":0,"offset_similarity":0,"scale_similarity":1}
        target_group = torch.stack([(targets==i).float()*2-1 for i in range(class_num)])
        similarity_matrix = target_group[targets]
        gt_sim = Similarity_matrix(similarity_matrix, p=p)
        gt_sim.similarity_matrix[gt_sim.similarity_matrix==-1]=0

        similarity_matrix = torch.exp(-torch.cdist(features, features, p=2)**2 / (2 * 2**2 * 768))
        knn_sim = Similarity_matrix(similarity_matrix, p=p)
        knn_sim.similarity_matrix[knn_sim.similarity_matrix>0]=1

        # gt_sim.similarity_matrix = gt_sim.similarity_matrix.cuda()
        # knn_sim.similarity_matrix= knn_sim.similarity_matrix.cuda()
        right_pos = gt_sim.similarity_matrix==1
        acc= (knn_sim.similarity_matrix[right_pos]).sum()/ (right_pos.sum())
        print(f"{class_num} acc:{acc}")
        del features, targets, feature_sim, gt_sim, knn_sim
        torch.cuda.empty_cache()


