from torch import nn
from .SPD import SPD


class ModelPU(nn.Module):
    def __init__(self, up_factors=None):
        super(ModelPU, self).__init__()
        if up_factors is None:
            up_factors = [1]
        else:
            up_factors = up_factors

        uppers = []
        for i, factor in enumerate(up_factors):
            uppers.append(SPD(up_factor=factor, i=i, global_feat=False))

        self.uppers = nn.ModuleList(uppers)

    def forward(self, x):
        """
        Args:
            x: Tensor, (b, n_coarse, 3), coarse point cloud
        """
        arr_pcd = []
        pcd = x.permute(0, 2, 1).contiguous()
        feat_prev = None
        for upper in self.uppers:
            pcd, feat_prev = upper(pcd, K_prev=feat_prev)
            arr_pcd.append(pcd.permute(0, 2, 1).contiguous())
        return arr_pcd