# -*- encoding: utf-8 -*-
'''
Copyright (c) Microsoft Corporation.
Licensed under the MIT License.

@File    :   codebook_reorder.py
@Time    :   2025/11/27 15:05:06
@Author  :   XinChen 
'''

import torch



class CodeBookReSort:
    def __init__(self, codebook: torch.Tensor, bits: int = 5):
        """
        codebook: shape (K, C)
        """
        assert codebook.dim() == 2
        self.codebook = codebook
        self.device = codebook.device
        self.K, self.C = codebook.shape
        self.bits = bits
        
        self.compute_l2_matrix()
        self.find_center_and_sort()
        self.assign_new_indices(self.bits)

    def compute_l2_matrix(self):
        x = self.codebook
        xx = (x ** 2).sum(dim=1, keepdim=True)
        yy = (x ** 2).sum(dim=1).unsqueeze(0)
        dist2 = xx + yy - 2 * (x @ x.t())
        dist2 = dist2.clamp(min=0)
        self.l2_matrix = torch.sqrt(dist2)
        return self.l2_matrix

    # ---------------------------------------------------------
    # 需求2：找中心向量 + 排序
    # ---------------------------------------------------------
    def find_center_and_sort(self):
        assert hasattr(self, "l2_matrix")
        l2_avg = self.l2_matrix.mean(dim=1)
        self.center_idx = torch.argmin(l2_avg)
        dist_row = self.l2_matrix[self.center_idx]
        sorted_dist, sorted_idx = torch.sort(dist_row)
        self.sorted_dist = sorted_dist
        self.sorted_idx = sorted_idx
        return self.center_idx, sorted_dist, sorted_idx

    # ---------------------------------------------------------
    # 需求3：重新分配索引（生成双向映射）
    # ---------------------------------------------------------
    def assign_new_indices(self, b: int):
        """
        b 比特 → 区间为 [-2^(b-1), 2^(b-1)-1]
        """
        assert hasattr(self, "sorted_idx")

        low = -(2 ** (b - 1))
        high = 2 ** (b - 1) - 1
        assert high - low + 1 >= self.K

        # 生成连续的新索引
        new_idx_range = torch.arange(low, low + self.K, device=self.device)

        # 原索引 → 新索引
        mapping_tensor = torch.empty(self.K, dtype=torch.int64, device=self.device)
        mapping_tensor[self.sorted_idx] = new_idx_range
        self.mapping_tensor = mapping_tensor

        # ----------------------------
        # ⚡ 需求5：新索引 → 原索引
        # ----------------------------
        # 新索引空间是连续区间 [low, low+K)，要构建一个查表用的字典（GPU 友好）
        # 例：new->old 关系为： inverse_mapping_tensor[new_index - low] = old_index

        inverse_mapping_tensor = torch.empty(self.K, dtype=torch.int64, device=self.device)
        # sorted_idx[i] 是排在第 i 个的新索引，对应 new_idx_range[i]
        # 因此 inverse[new_idx_range[i]-low] = sorted_idx[i]
        inverse_mapping_tensor[new_idx_range - low] = self.sorted_idx
        self.inverse_mapping_tensor = inverse_mapping_tensor
        self.inverse_low = low  # 用于偏移

        return mapping_tensor, inverse_mapping_tensor

    # ---------------------------------------------------------
    # 需求4：旧索引 → 新索引（列表输入）
    # ---------------------------------------------------------
    def remap_index(self, index):
        """
        index: Tensor(old indices)
        """
        assert hasattr(self, "mapping_tensor")
        mapping = self.mapping_tensor
        new_index = mapping[index.to(self.device)]
        return new_index

    # ---------------------------------------------------------
    # 需求5：新索引 → 原索引（列表输入）
    # ---------------------------------------------------------
    def inverse_remap_index(self, index):
        """
        index_list: Tensor(new indices)
        返回对应的原索引列表
        """
        assert hasattr(self, "inverse_mapping_tensor")

        inv_map = self.inverse_mapping_tensor  # (K,)
        low = self.inverse_low

        original_list = []
        index = index.to(self.device)
        # 新索引范围是 [low, low+K)
        offset = index - low
        original_index = inv_map[offset]
        return original_index
    
