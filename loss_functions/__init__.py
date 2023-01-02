import torch
from .Chamfer3D.dist_chamfer_3D import chamfer_3DDist
from .emd.emd_module import emdModule

chamfer_dist = chamfer_3DDist()
EMD = torch.nn.DataParallel(emdModule().cuda()).cuda()

def chamfer_l1(p1, p2):
    d1, d2, _, _ = chamfer_dist(p1, p2)
    d1 = torch.mean(torch.sqrt(d1))
    d2 = torch.mean(torch.sqrt(d2))
    return (d1 + d2) / 2

def chamfer_l2(p1, p2):
    d1, d2, _, _ = chamfer_dist(p1, p2)
    return torch.mean(d1) + torch.mean(d2)

def chamfer_partial_l1(pcd1, pcd2):
    d1, d2, _, _ = chamfer_dist(pcd1, pcd2)
    d1 = torch.mean(torch.sqrt(d1))
    return d1

def chamfer_partial_l2(pcd1, pcd2):
    d1, d2, _, _ = chamfer_dist(pcd1, pcd2)
    d1 = torch.mean(d1)
    return d1

def emd_loss(p1, p2):
    d1, _ = EMD(p1, p2, eps=0.005, iters=50)
    d = torch.sqrt(d1).mean(1).mean()

    return d