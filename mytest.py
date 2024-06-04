import numpy as np
import torch


N = 5
similarity_matrix = torch.randn(size=(N, N))
print(similarity_matrix.device)


# def MMO(W, score, k_list=[1, 2, 3]):
#     # W: similarity_matrix
#     # score: min-max normalized
#     S_list = []
#     for k in k_list:
#         _, topk_matrix = torch.topk(
#             W.float(), W.shape[0] - k, largest=False, sorted=True
#         )  # indices of dissimilar items
#         W_mask = W.clone()
#         for i in range(W.shape[0]):
#             W_mask[i, topk_matrix[i]] = 0
#         n = W.shape[-1]

#         D_ = torch.zeros_like(W).float()
#         for i in range(n):
#             D_[i, i] = 1 / (W_mask[i, :].sum())
#         print(f"D_: {D_}")
#         print(f"W_mask: {W_mask}")
#         P = D_ @ W_mask
#         print(f"P: {P}")
#         S = score.clone().unsqueeze(-1)
#         S = P @ S  # shape: [n, 1]
#         # print("===========")
#         S_list.append(S)
#     S = torch.concat(S_list, -1).mean(-1)
#     return S


# similarity_matrix = torch.randn(size=(N, N))
# score = torch.randn(size=(N,))
# k_list = [2]

# MMO(similarity_matrix, score, k_list)
