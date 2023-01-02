import sys
import torch
sys.path.append('..')

from loss_functions import chamfer_l2 as chamfer
from models.utils import fps_subsample


def get_loss_ae(pcds, gt):

    x_512 = fps_subsample(gt, pcds[0].shape[1])
    # cd_c = chamfer(pc, x_512)
    cd_1 = chamfer(pcds[0], x_512)
    # emd_1 = emd(p1, x_512) * 0.2
    # cd_2 = chamfer(p2, x_1024)
    cd_3 = chamfer(pcds[-1], gt)

    loss = cd_1 + cd_3  #  + emd_1 + emd_3  # + cd_2
    return loss