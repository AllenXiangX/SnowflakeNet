import sys
import torch
sys.path.append('..')

# from loss_functions import chamfer_l1, chamfer_l2, chamfer_partial_l1, chamfer_partial_l2, emd_loss
from loss_functions import chamfer_3DDist, emdModule
from models.utils import fps_subsample

class Completionloss:
    def __init__(self, loss_func='cd_l1'):
        self.loss_func = loss_func
        self.chamfer_dist = chamfer_3DDist()
        self.EMD = torch.nn.DataParallel(emdModule().cuda()).cuda()

        if loss_func == 'cd_l1':
            self.metric = self.chamfer_l1
            self.partial_matching = self.chamfer_partial_l1
        elif loss_func == 'cd_l2':
            self.metric = self.chamfer_l2
            self.partial_matching = self.chamfer_partial_l2
        elif loss_func == 'emd':
            self.metric = self.emd_loss
        else:
            raise Exception('loss function {} not supported yet!'.format(loss_func))

    def chamfer_l1(self, p1, p2):
        d1, d2, _, _ = self.chamfer_dist(p1, p2)
        d1 = torch.mean(torch.sqrt(d1))
        d2 = torch.mean(torch.sqrt(d2))
        return (d1 + d2) / 2

    def chamfer_l2(self, p1, p2):
        d1, d2, _, _ = self.chamfer_dist(p1, p2)
        return torch.mean(d1) + torch.mean(d2)

    def chamfer_partial_l1(self, pcd1, pcd2):
        d1, d2, _, _ = self.chamfer_dist(pcd1, pcd2)
        d1 = torch.mean(torch.sqrt(d1))
        return d1

    def chamfer_partial_l2(self, pcd1, pcd2):
        d1, d2, _, _ = self.chamfer_dist(pcd1, pcd2)
        d1 = torch.mean(d1)
        return d1

    def emd_loss(self, p1, p2):
        d1, _ = self.EMD(p1, p2, eps=0.005, iters=50)
        d = torch.sqrt(d1).mean(1).mean()

        return d

    def get_loss(self, pcds_pred, partial, gt):
        Pc, P1, P2, P3 = pcds_pred
        gt_2 = fps_subsample(gt, P2.shape[1])
        gt_1 = fps_subsample(gt_2, P1.shape[1])
        gt_c = fps_subsample(gt_1, Pc.shape[1])

        loss_c = self.metric(Pc, gt_c)
        loss_1 = self.metric(P1, gt_1)
        loss_2 = self.metric(P2, gt_2)
        loss_3 = self.metric(P3, gt)

        partial_matching = torch.tensor(0).cuda() if self.loss_func == 'emd' else self.partial_matching(partial, P3)

        loss_all = loss_c + loss_1 + loss_2 + loss_3 + partial_matching
        losses = [partial_matching, loss_c, loss_1, loss_2, loss_3]
        return loss_all, losses






if __name__ == '__main__':
    gt = torch.randn(10, 2048, 3).cuda()
    pc = torch.randn(10, 256, 3).cuda()
    p1 = torch.randn(10, 512, 3).cuda()
    p2 = torch.randn(10, 1024, 3).cuda()
    p3 = torch.randn(10, 2048, 3).cuda()


    # loss = get_loss([pc, p1, p2, p3], gt, gt, 'emd')[0]
    # print(loss.item())
