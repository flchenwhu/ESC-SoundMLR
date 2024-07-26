"""
@author:  zhouxiao
@contact: zhouxiao17@mails.tsinghua.edu.cn
"""

import torch
import torch.nn as nn


__all__ = ["AdaSPLoss"]



class AdaSPLoss(object):
    """
    Adaptive sparse pairwise (AdaSP) loss
    """

    def __init__(self, temp=0.04, loss_type='adasp'):
        self.temp = temp
        self.loss_type = loss_type

    def __call__(self, feats, targets):
        feats_n = nn.functional.normalize(feats, dim=1)

        bs_size = feats_n.size(0)
        N_id = len(torch.unique(targets))
        # N_ins = bs_size // N_id

        scale = 1. / self.temp

        sim_qq = torch.matmul(feats_n, feats_n.T)
        sf_sim_qq = sim_qq * scale

        right_factor = torch.zeros((bs_size, N_id)).cuda()
        for i in range(N_id):
            mask_i = (targets == i).float().view(-1, 1)
            right_factor += mask_i

        pos_mask = torch.zeros((bs_size, N_id)).cuda()
        for i in range(N_id):
            mask_i = (targets == i).float().view(-1, 1)
            pos_mask += mask_i

        left_factor = right_factor.T

        ## hard-hard mining for pos
        mask_HH = torch.zeros_like(sf_sim_qq)
        for i in range(N_id):
            mask_i = (targets == i).nonzero().squeeze(1)
            mask_HH[mask_i, mask_i] = -1.
        mask_HH[mask_HH == 0] = 1.

        ID_sim_HH = torch.exp(sf_sim_qq.mul(mask_HH))

        ID_sim_HH = ID_sim_HH.mm(right_factor)
        ID_sim_HH = left_factor.mm(ID_sim_HH)

        pos_mask_id = torch.eye(N_id).cuda()
        pos_sim_HH = ID_sim_HH.mul(pos_mask_id)
        pos_sim_HH[pos_sim_HH == 0] = 1.
        pos_sim_HH = 1. / pos_sim_HH
        ID_sim_HH = ID_sim_HH.mul(1 - pos_mask_id) + pos_sim_HH.mul(pos_mask_id)

        ID_sim_HH_L1 = nn.functional.normalize(ID_sim_HH, p=1, dim=1)

        ## hard-easy mining for pos
        mask_HE = torch.zeros_like(sf_sim_qq)
        for i in range(N_id):
            mask_i = (targets == i).nonzero().squeeze(1)
            mask_HE[mask_i, mask_i] = -1.
        mask_HE[mask_HE == 0] = 1.

        ID_sim_HE = torch.exp(sf_sim_qq.mul(mask_HE))
        ID_sim_HE = ID_sim_HE.mm(right_factor)

        pos_sim_HE = ID_sim_HE.mul(pos_mask)
        neg_mask = 1 - pos_mask
        neg_sim_HE = ID_sim_HE.mul(neg_mask)
        pos_sim_HE[pos_sim_HE == 0] = 1.
        pos_sim_HE = 1. / pos_sim_HE
        ID_sim_HE = neg_sim_HE + pos_sim_HE.mul(pos_mask)

        # hard-hard for neg
        ID_sim_HE = left_factor.mm(ID_sim_HE)

        ID_sim_HE_L1 = nn.functional.normalize(ID_sim_HE, p=1, dim=1)

        l_sim = torch.log(torch.diag(ID_sim_HH))
        s_sim = torch.log(torch.diag(ID_sim_HE))

        weight_sim_HH = torch.log(torch.diag(ID_sim_HH)).detach() / scale
        weight_sim_HE = torch.log(torch.diag(ID_sim_HE)).detach() / scale
        wt_l = 2 * weight_sim_HE.mul(weight_sim_HH) / (weight_sim_HH + weight_sim_HE)
        wt_l[weight_sim_HH < 0] = 0
        both_sim = l_sim.mul(wt_l) + s_sim.mul(1 - wt_l)

        adaptive_pos = torch.diag(torch.exp(both_sim))

        adaptive_sim_mat = adaptive_pos.mul(pos_mask_id) + ID_sim_HE.mm(1 - pos_mask_id)

        adaptive_sim_mat_L1 = nn.functional.normalize(adaptive_sim_mat, p=1, dim=1)

        loss_sph = -1 * torch.log(torch.diag(ID_sim_HH_L1)).mean()
        loss_splh = -1 * torch.log(torch.diag(ID_sim_HE_L1)).mean()
        loss_adasp = -1 * torch.log(torch.diag(adaptive_sim_mat_L1)).mean()

        if self.loss_type == 'sp-h':
            loss = loss_sph.mean()
        elif self.loss_type == 'sp-lh':
            loss = loss_splh.mean()
        elif self.loss_type == 'adasp':
            loss = loss_adasp

        return loss
