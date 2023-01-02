import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from .SPD import SPD
from .utils import MLP_Res, fps_subsample
from loss_functions import chamfer_l2 as chamfer


class SeedGenerator(nn.Module):
    def __init__(self, dim_feat=512, num_pc=256):
        super(SeedGenerator, self).__init__()
        self.ps = nn.ConvTranspose1d(dim_feat, 128, num_pc, bias=True)
        self.mlp_1 = MLP_Res(in_dim=dim_feat + 128, hidden_dim=128, out_dim=128)
        self.mlp_2 = MLP_Res(in_dim=128, hidden_dim=64, out_dim=128)
        self.mlp_3 = MLP_Res(in_dim=dim_feat + 128, hidden_dim=128, out_dim=128)
        self.mlp_4 = nn.Sequential(
            nn.Conv1d(128, 64, 1),
            nn.ReLU(),
            nn.Conv1d(64, 3, 1)
        )

    def forward(self, feat):
        """
        Args:
            feat: Tensor (b, dim_feat, 1)
        """
        x1 = self.ps(feat)  # (b, 128, 256)
        x1 = self.mlp_1(torch.cat([x1, feat.repeat((1, 1, x1.size(2)))], 1))
        x2 = self.mlp_2(x1)
        x3 = self.mlp_3(torch.cat([x2, feat.repeat((1, 1, x2.size(2)))], 1))  # (b, 128, 256)
        out_points = self.mlp_4(x3)  # (b, 3, 256)
        return out_points

class Decoder(nn.Module):
    def __init__(self, dim_feat=512, num_p0=512,
                 radius=1, bounding=True, up_factors=None):
        super(Decoder, self).__init__()
        self.decoder_coarse = SeedGenerator(dim_feat=dim_feat, num_pc=num_p0)
        if up_factors is None:
            up_factors = [1]
        else:
            up_factors = up_factors

        uppers = []
        for i, factor in enumerate(up_factors):
            uppers.append(SPD(dim_feat=dim_feat, up_factor=factor, i=i, bounding=bounding, radius=radius))

        self.uppers = nn.ModuleList(uppers)

    def forward(self, feat):
        """
        Args:
            feat: Tensor, (b, dim_feat)
            partial_coarse: Tensor, (b, n_coarse, 3)
        """
        feat = feat.unsqueeze(-1)
        arr_pcd = []
        pcd = self.decoder_coarse(feat).permute(0, 2, 1).contiguous()  # (B, 512, 3)

        # no fps with input
        arr_pcd.append(pcd)
        feat_prev = None
        pcd = pcd.permute(0, 2, 1).contiguous()
        for upper in self.uppers:
            pcd, feat_prev = upper(pcd, feat, feat_prev)
            arr_pcd.append(pcd.permute(0, 2, 1).contiguous())

        return arr_pcd

class PointNetEncoder(nn.Module):
    def __init__(self, zdim, input_dim=3):
        super().__init__()
        self.zdim = zdim
        self.conv1 = nn.Conv1d(input_dim, 128, 1)
        self.conv2 = nn.Conv1d(128, 128, 1)
        self.conv3 = nn.Conv1d(128, 256, 1)
        self.conv4 = nn.Conv1d(256, 512, 1)
        self.bn1 = nn.BatchNorm1d(128)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(256)
        self.bn4 = nn.BatchNorm1d(512)

        # Mapping to [c], cmean
        self.fc1_m = nn.Linear(512, 256)
        self.fc2_m = nn.Linear(256, 128)
        self.fc3_m = nn.Linear(128, zdim)
        self.fc_bn1_m = nn.BatchNorm1d(256)
        self.fc_bn2_m = nn.BatchNorm1d(128)

        # Mapping to [c], cmean
        self.fc1_v = nn.Linear(512, 256)
        self.fc2_v = nn.Linear(256, 128)
        self.fc3_v = nn.Linear(128, zdim)
        self.fc_bn1_v = nn.BatchNorm1d(256)
        self.fc_bn2_v = nn.BatchNorm1d(128)

    def forward(self, x):
        x = x.transpose(1, 2)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.bn4(self.conv4(x))
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 512)

        m = F.relu(self.fc_bn1_m(self.fc1_m(x)))
        m = F.relu(self.fc_bn2_m(self.fc2_m(m)))
        m = self.fc3_m(m)
        v = F.relu(self.fc_bn1_v(self.fc1_v(x)))
        v = F.relu(self.fc_bn2_v(self.fc2_v(v)))
        v = self.fc3_v(v)

        return m, v


class ModelAE(nn.Module):

    def __init__(self, **kwargs):
        super().__init__()
        dim_feat = kwargs.get('dim_feat', 128)
        num_p0 = kwargs.get('num_p0', 512)
        radius = kwargs.get('radius', 1)
        bounding = kwargs.get('bounding', True)
        up_factors = kwargs.get('up_factors', [2, 2])
        self.encoder = PointNetEncoder(zdim=dim_feat)
        self.decoder = Decoder(dim_feat=dim_feat, num_p0=num_p0,
                               radius=radius, up_factors=up_factors, bounding=bounding)
        # self.encoder = PointNetEncoder(zdim=args.latent_dim)
        # self.decoder = Decoder(dim_feat=args.latent_dim, up_factors=[2, 2])

    def encode(self, x):
        """
        Args:
            x:  Point clouds to be encoded, (B, N, d).
        """
        code, _ = self.encoder(x)
        # print('code.shape ', code.shape)
        return code

    def decode(self, code):
        pcds = self.decoder(code)
        # print('decode.shape ', pcd.shape)
        return pcds[-1]

    def forward(self, x):
        code = self.encode(x)
        p1, p2, p3 = self.decoder(code)
        return p1, p2, p3

    def get_loss(self, x):
        code = self.encode(x)
        p1, p2, p3 = self.decoder(code)

        x_512 = fps_subsample(x, 512)

        cd_1 = chamfer(p1, x_512)

        cd_3 = chamfer(p3, x)

        loss = cd_1 + cd_3 #  + emd_1 + emd_3  # + cd_2
        return loss
