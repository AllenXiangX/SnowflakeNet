import sys
sys.path.append('..')
import torch
import yaml
from models.utils import fps_subsample, query_knn, grouping_operation
from easydict import EasyDict as edict
from loss_functions import chamfer_3DDist



def create_edict(pack):
    d = edict()
    for key, value in pack.items():
        if isinstance(value, dict):
            d[key] = create_edict(value)
        else:
            d[key] = value
    return d


def read_yaml(path):
    with open(path, 'r') as file:
        config = yaml.safe_load(file)

    return create_edict(config)


def knn_sample(p1, p2, k=256):
    """
    Args:
        p1: b, s, 3
        p2: b, n, 3
    """
    idx_knn = query_knn(k, p2, p1, include_self=True)
    point_groups = grouping_operation(p2.permute(0, 2, 1).contiguous(), idx_knn).permute(0, 2, 3, 1).contiguous()
    return point_groups

def patch_extraction(point_clouds, num_per_patch=256, patch_num_ratio=3):
    """
    Args:
        point_clouds: b, n, 3
    """
    b, n, _ = point_clouds.shape
    seed_num = int(n / num_per_patch * patch_num_ratio)

    seed_points = fps_subsample(point_clouds, seed_num)

    patch_points = knn_sample(seed_points, point_clouds, k=num_per_patch)
    patch_points = patch_points.reshape((b*seed_num, num_per_patch, 3)).contiguous()

    return patch_points


def random_subsample(pcd, n_points=256):
    """
    Args:
        pcd: (B, N, 3)

    returns:
        new_pcd: (B, n_points, 3)
    """
    b, n, _ = pcd.shape
    device = pcd.device
    batch_idx = torch.arange(b, dtype=torch.long, device=device).reshape((-1, 1)).repeat(1, n_points)
    idx = torch.cat([torch.randperm(n, dtype=torch.long, device=device)[:n_points].reshape((1, -1)) for i in range(b)], 0)
    return pcd[batch_idx, idx, :]


class PULoss:
    def __init__(self):
        self.chamfer_distance = chamfer_3DDist()



    def chamfer_radius(self, p1, p2, radius=1.0):
        d1, d2, _, _ = self.chamfer_distance(p1, p2)
        cd_dist = 0.5 * d1 + 0.5 * d2
        cd_dist = torch.mean(cd_dist, dim=1)
        cd_dist_norm = cd_dist / radius
        cd_loss = torch.mean(cd_dist_norm)
        return cd_loss


    def get_loss(self, pcds, gt, radius):
        """
        Args:
            pcds: list of point clouds, [256, 512, 1048, 1048]
        """
        p1, p2, p3, p4 = pcds
        gt_1 = fps_subsample(gt, p1.shape[1])
        cd_1 = self.chamfer_radius(p1, gt_1, radius)

        cd_3 = self.chamfer_radius(p3, gt, radius)
        cd_4 = self.chamfer_radius(p4, gt, radius)

        return cd_1 + cd_3 + cd_4, cd_4
