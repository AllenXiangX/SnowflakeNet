# -*- coding: utf-8 -*-
# @Author: XP

import logging
import os
import torch
import utils.data_loaders
import utils.helpers
import utils.io
from tqdm import tqdm
from models.model import SnowflakeNet as Model


def inference_net(cfg):
    # Enable the inbuilt cudnn auto-tuner to find the best algorithm to use
    torch.backends.cudnn.benchmark = True

    # Set up data loader
    dataset_loader = utils.data_loaders.DATASET_LOADER_MAPPING[cfg.DATASET.TEST_DATASET](cfg)
    test_data_loader = torch.utils.data.DataLoader(dataset=dataset_loader.get_dataset(
        utils.data_loaders.DatasetSubset.TEST),
                                                   batch_size=1,
                                                   num_workers=cfg.CONST.NUM_WORKERS,
                                                   collate_fn=utils.data_loaders.collate_fn,
                                                   pin_memory=True,
                                                   shuffle=False)

    model = Model(dim_feat=512, up_factors=[2, 2])

    if torch.cuda.is_available():
        model = torch.nn.DataParallel(model).cuda()

    # Load the pretrained model from a checkpoint
    logging.info('Recovering from %s ...' % (cfg.CONST.WEIGHTS))
    checkpoint = torch.load(cfg.CONST.WEIGHTS)
    model.load_state_dict(checkpoint['model'])

    # Switch models to evaluation mode
    model.eval()

    # The inference loop
    n_samples = len(test_data_loader)
    t_obj = tqdm(test_data_loader)


    for model_idx, (taxonomy_id, model_id, data) in enumerate(t_obj):
        taxonomy_id = taxonomy_id[0] if isinstance(taxonomy_id[0], str) else taxonomy_id[0].item()
        model_id = model_id[0]

        with torch.no_grad():
            for k, v in data.items():
                data[k] = utils.helpers.var_or_cuda(v)

            partial = data['partial_cloud']

            pcds = model(partial, return_P0=True)
            pcdc, pcd0, pcd1, pcd2, pcd3 = pcds


            output_folder = os.path.join(cfg.DIR.OUT_PATH, 'benchmark', taxonomy_id)
            if not os.path.exists(output_folder):
                os.makedirs(output_folder)
            output_folder_pcdc = os.path.join(output_folder, 'pcdc')
            output_folder_pcd0 = os.path.join(output_folder, 'pcd0')
            output_folder_pcd1 = os.path.join(output_folder, 'pcd1')
            output_folder_pcd2 = os.path.join(output_folder, 'pcd2')
            output_folder_pcd3 = os.path.join(output_folder, 'pcd3')
            if not os.path.exists(output_folder_pcd1):
                os.makedirs(output_folder_pcdc)
                os.makedirs(output_folder_pcd0)
                os.makedirs(output_folder_pcd1)
                os.makedirs(output_folder_pcd2)
                os.makedirs(output_folder_pcd3)

            output_file_path = os.path.join(output_folder, 'pcdc', '%s.h5' % model_id)
            utils.io.IO.put(output_file_path, pcdc.squeeze().cpu().numpy())

            output_file_path = os.path.join(output_folder, 'pcd0', '%s.h5' % model_id)
            utils.io.IO.put(output_file_path, pcd0.squeeze().cpu().numpy())

            output_file_path = os.path.join(output_folder, 'pcd1', '%s.h5' % model_id)
            utils.io.IO.put(output_file_path, pcd1.squeeze().cpu().numpy())

            output_file_path = os.path.join(output_folder, 'pcd2', '%s.h5' % model_id)
            utils.io.IO.put(output_file_path, pcd2.squeeze().cpu().numpy())

            output_file_path = os.path.join(output_folder, 'pcd3', '%s.h5' % model_id)
            utils.io.IO.put(output_file_path, pcd3.squeeze().cpu().numpy())

            t_obj.set_description('Test[%d/%d] Taxonomy = %s Sample = %s File = %s' %
                         (model_idx + 1, n_samples, taxonomy_id, model_id, output_file_path))

