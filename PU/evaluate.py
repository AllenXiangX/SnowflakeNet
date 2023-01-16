import os
import sys
sys.path.append('..')
import torch
import numpy as np
from tqdm import tqdm
from loss_functions import chamfer_3DDist


def normalize_point_cloud(pc):
    """
    pc: tensor [N, P, 3]
    """
    centroid = torch.mean(pc, dim=1, keepdim=True)
    pc = pc-centroid
    furthest_distance = torch.max(torch.sqrt(torch.sum(pc**2, dim=-1, keepdim=True)), dim=1, keepdim=True)[0]
    pc = pc / furthest_distance
    return pc, centroid, furthest_distance

def load_lists(dir_pred, dir_gt):
    gt_files = [f for f in os.listdir(dir_gt) if f.endswith('xyz')]
    pred_list = []
    gt_list = []
    for f in gt_files:
        pred_list.append(np.loadtxt(os.path.join(dir_pred, f)))
        gt_list.append(np.loadtxt(os.path.join(dir_gt, f)))
    return pred_list, gt_list


class TensorEvaluator:
    def __init__(self):
        self.chamfer = chamfer_3DDist()

    def evaluate(self, pred_list, gt_list):
        """Evaluate batched and normalized predictions and ground truths,
        """
        n = len(pred_list)
        total_cd = 0
        total_hd = 0
        for i in tqdm(range(n)):
            pred = pred_list[i: i + 1]
            gt = gt_list[i: i + 1]

            pred = normalize_point_cloud(pred)[0]
            gt = normalize_point_cloud(gt)[0]

            d1, d2, _, _ = self.chamfer(pred, gt)
            d1 = d1.squeeze(0).cpu().numpy()
            d2 = d2.squeeze(0).cpu().numpy()

            hd_value = np.max(np.amax(d1, axis=0) + np.amax(d2, axis=0))

            total_cd += np.mean(d1) + np.mean(d2)
            total_hd += hd_value

        avg_cd = total_cd / n
        avg_hd = total_hd / n
        # print('avg_cd: ', avg_cd)
        # print('avg_hd: ', avg_hd)
        return avg_cd, avg_hd


if __name__ == '__main__':
    p_list, g_list = load_lists('/data1/xp/PUGAN/data/test/groundtruth/output', '/data1/xp/PUGAN/data/test/groundtruth')
    p_list = torch.from_numpy(np.stack(p_list, 0)).float().cuda('cuda:8')
    g_list = torch.from_numpy(np.stack(g_list, 0)).float().cuda('cuda:8')


    g_list, c, f = normalize_point_cloud(g_list)
    p_list = (p_list - c) / f

    cd, hd = 0, 0# evaluate_tensor(p_list, g_list)
    print('cd: ', cd, 'hd: ', hd)

    # evaluate(p_list, g_list, 'cuda:8')
