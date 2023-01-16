import sys
import torch
import numpy as np
sys.path.append('..')
from loss_functions import chamfer_3DDist
from models.utils import fps_subsample


def pc_normalize(pc, radius):
    l = pc.shape[0]
    centroid = np.mean(pc, axis=0)
    pc = pc - centroid
    m = np.max(np.sqrt(np.sum(pc**2, axis=1)))
    pc = pc / m * radius
    return pc


class LossSVR:
    def __init__(self):
        self.chamfer_dist = chamfer_3DDist()

    def chamfer_l1(self, p1, p2):
        d1, d2, _, _ = self.chamfer_dist(p1, p2)
        d1 = torch.mean(torch.sqrt(d1))
        d2 = torch.mean(torch.sqrt(d2))
        return (d1 + d2) / 2

    def chamfer_l2(self, p1, p2):
        d1, d2, _, _ = self.chamfer_dist(p1, p2)
        return torch.mean(d1) + torch.mean(d2)

    def calc_cd(self, output, gt):
        cd_p = self.chamfer_l1(output, gt)
        cd_t = self.chamfer_l2(output, gt) # (dist1.mean(1) + dist2.mean(1))

        return cd_p, cd_t

    def loss_snowflake(self, outputs, gt):
        x_512 = fps_subsample(gt, 512)
        p1 = outputs[0]
        p3 = outputs[-1]
        cd1 = self.chamfer_l1(p1, x_512)
        cd3 = self.chamfer_l1(p3, gt)
        return cd1 + cd3



