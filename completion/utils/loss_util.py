import sys
import torch
sys.path.append('..')

from loss_functions import chamfer_l1, chamfer_l2, chamfer_partial_l1, chamfer_partial_l2, emd_loss
from models.utils import fps_subsample

def get_loss(pcds_pred, partial, gt, loss_func='cd_l1'):
    if loss_func == 'cd_l1':
        metric = chamfer_l1
        partial_matching = chamfer_partial_l1
    elif loss_func == 'cd_l2':
        metric = chamfer_l2
        partial_matching = chamfer_partial_l2
    elif loss_func == 'emd':
        metric = emd_loss
    else:
        raise Exception('loss function {} not supported yet!'.format(loss_func))

    Pc, P1, P2, P3 = pcds_pred
    gt_2 = fps_subsample(gt, P2.shape[1])
    gt_1 = fps_subsample(gt_2, P1.shape[1])
    gt_c = fps_subsample(gt_1, Pc.shape[1])

    loss_c = metric(Pc, gt_c)
    loss_1 = metric(P1, gt_1)
    loss_2 = metric(P2, gt_2)
    loss_3 = metric(P3, gt)

    partial_matching = torch.tensor(0).cuda() if loss_func == 'emd' else partial_matching(partial, P3)

    loss_all = loss_c + loss_1 + loss_2 + loss_3 + partial_matching
    losses = [partial_matching, loss_c, loss_1, loss_2, loss_3]
    return loss_all, losses



if __name__ == '__main__':
    gt = torch.randn(10, 2048, 3).cuda()
    pc = torch.randn(10, 256, 3).cuda()
    p1 = torch.randn(10, 512, 3).cuda()
    p2 = torch.randn(10, 1024, 3).cuda()
    p3 = torch.randn(10, 2048, 3).cuda()


    loss = get_loss([pc, p1, p2, p3], gt, gt, 'emd')[0]
    print(loss.item())
