import numpy as np
import torch
import pickle


def MMO(W, score, k_list=[1, 2, 3]):
    # W: similarity_matrix
    # score: min-max normalized
    S_list = []
    for k in k_list:
        _, topk_matrix = torch.topk(
            W.float(), W.shape[0] - k, largest=False, sorted=True
        )  # indices of dissimilar items
        W_mask = W.clone()
        for i in range(W.shape[0]):
            W_mask[i, topk_matrix[i]] = 0
        n = W.shape[-1]

        D_ = torch.zeros_like(W).float()
        for i in range(n):
            D_[i, i] = 1 / (
                W_mask[i, :].sum()
            )  # only diagonal, 1/sum_of_top_k_similarities

        P = D_ @ W_mask  # 1/sum_of_top_k_similarities * each_similarity

        S = score.clone().unsqueeze(-1)
        S = P @ S  # [n, 1]
        S_list.append(S)
    S = torch.concat(S_list, -1).mean(-1)
    return S


def RsCIN(scores_old, cls_tokens=None, k_list=[0]):
    if cls_tokens is None or 0 in k_list:
        return scores_old
    cls_tokens = np.array(cls_tokens)
    scores = (scores_old - scores_old.min()) / (scores_old.max() - scores_old.min())
    similarity_matrix = cls_tokens @ cls_tokens.T
    similarity_matrix = torch.tensor(similarity_matrix)
    scores_new = MMO(
        similarity_matrix.clone().float(),
        score=torch.tensor(scores).clone().float(),
        k_list=k_list,
    )
    scores_new = scores_new.numpy()
    return scores_new
