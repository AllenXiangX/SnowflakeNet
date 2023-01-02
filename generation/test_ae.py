import os
import time
import argparse
import torch
import numpy as np

import sys
sys.path.append('..')

from tqdm.auto import tqdm
from utils.dataset import ShapeNetCore
from utils.misc import seed_all, get_logger, str_list
from utils.data import DataLoader
from models.model_ae import ModelAE
method = 'SPD'
from evaluation import EMD_CD



# Arguments
parser = argparse.ArgumentParser()
parser.add_argument('--ckpt', type=str, default='/data/xp/experiments/generation/logs_ae/AE_2021_08_02__12_48_38(all_cd_emd1e-2_cd420)/ckpt_0.000420_198000_spd.pt')
parser.add_argument('--categories', type=str_list, default=['all'])
parser.add_argument('--save_dir', type=str, default='./results')
parser.add_argument('--device', type=str, default='cuda')
# Datasets and loaders
parser.add_argument('--dataset_path', type=str, default='/data/xp/code/diffusion/data/shapenet.hdf5')
parser.add_argument('--batch_size', type=int, default=128)
parser.add_argument('--num_workers', type=int, default=8)
args = parser.parse_args()

# Logging
save_dir = os.path.join(args.save_dir, 'AE_%s_%s_%d' % (method, '_'.join(args.categories), int(time.time())) )
if not os.path.exists(save_dir):
    os.makedirs(save_dir)
logger = get_logger('test', save_dir)
for k, v in vars(args).items():
    logger.info('[ARGS::%s] %s' % (k, repr(v)))

# Checkpoint
ckpt = torch.load(args.ckpt)
seed_all(ckpt['args'].seed)

# Datasets and loaders
logger.info('Loading datasets...')
test_dset = ShapeNetCore(
    path=args.dataset_path,
    cates=args.categories,
    split='test',
    scale_mode=ckpt['args'].scale_mode
)
test_loader = DataLoader(test_dset, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=False)

# Model
logger.info('Loading model...')
model = ModelAE(dim_feat=ckpt['args'].latent_dim, up_factors=[2, 2]).to(args.device)
model.load_state_dict(ckpt['state_dict'])

all_ref = []
all_recons = []
for i, batch in enumerate(tqdm(test_loader)):
    ref = batch['pointcloud'].to(args.device)
    shift = batch['shift'].to(args.device)
    scale = batch['scale'].to(args.device)

    # print(ref.shape)
    model.eval()
    with torch.no_grad():
        code = model.encode(ref)
        recons = model.decode(code).detach()



    ref = ref * scale + shift
    recons = recons * scale + shift

    all_ref.append(ref.detach().cpu())
    all_recons.append(recons.detach().cpu())

all_ref = torch.cat(all_ref, dim=0)
all_recons = torch.cat(all_recons, dim=0)

logger.info('Saving point clouds...')
np.save(os.path.join(save_dir, 'ref.npy'), all_ref.numpy())
np.save(os.path.join(save_dir, 'out.npy'), all_recons.numpy())

logger.info('Start computing metrics...')
metrics = EMD_CD(all_recons.to(args.device), all_ref.to(args.device), batch_size=args.batch_size, accelerated_cd=True)
cd, emd = metrics['MMD-CD'].item(), metrics['MMD-EMD'].item()
logger.info('CD:  %.12f' % cd)
logger.info('EMD: %.12f' % emd)
