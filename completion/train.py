# -*- coding: utf-8 -*-
# @Author: XP

import os
import torch
import logging
import argparse
import numpy as np
from datetime import datetime
from tqdm import tqdm
from tensorboardX import SummaryWriter
from utils import helpers, average_meter, scheduler, yaml_reader, loss_util, misc
from core import builder
from test import test

def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_args_from_command_line():
    parser = argparse.ArgumentParser(description='The argument parser of SnowflakeNet')
    parser.add_argument('--config', type=str, default='./configs/pcn_cd1.yaml', help='Configuration File')
    args = parser.parse_args()
    return args

def train(config):


    # dataloaders
    train_dataloader = builder.make_dataloader(config, 'train')
    test_dataloader = builder.make_dataloader(config, config.test.split)

    model = builder.make_model(config)
    if torch.cuda.is_available():
        model = torch.nn.DataParallel(model).cuda()



    # out folders
    if not config.train.out_path:
        config.train.out_path = './exp'
    output_dir = os.path.join(config.train.out_path, '%s', datetime.now().isoformat())
    config.train.path_checkpoints = output_dir % 'checkpoints'
    config.train.path_logs = output_dir % 'logs'
    if not os.path.exists(config.train.path_checkpoints):
        os.makedirs(config.train.path_checkpoints)

    # log writers
    train_writer = SummaryWriter(os.path.join(config.train.path_logs, 'train'))
    val_writer = SummaryWriter(os.path.join(config.train.path_logs, 'test'))

    init_epoch = 1
    best_metric = float('inf')
    steps = 0

    if config.train.resume:
        if not os.path.exists(config.train.model_path):
            raise Exception('checkpoints does not exists: {}'.format(config.test.model_path))

        print('Recovering from %s ...' % (config.train.model_path), end='')
        checkpoint = torch.load(config.test.model_path)
        model.load_state_dict(checkpoint['model'])
        print('recovered!')

        init_epoch = checkpoint['epoch_index']
        best_metric = checkpoint['best_metric']

    optimizer = builder.make_optimizer(config, model)
    scheduler = builder.make_schedular(config, optimizer, last_epoch=init_epoch if config.train.resume else -1)

    multiplier = 1.0
    if config.test.loss_func == 'cd_l1':
        multiplier = 1e3
    elif config.test.loss_func == 'cd_l2':
        multiplier = 1e4
    elif config.test.loss_func == 'emd':
        multiplier = 1e2

    n_batches = len(train_dataloader)
    avg_meter_loss = average_meter.AverageMeter(['loss_partial', 'loss_pc', 'loss_p1', 'loss_p2', 'loss_p3'])
    for epoch_idx in range(init_epoch, config.train.epochs):
        avg_meter_loss.reset()
        model.train()

        with tqdm(train_dataloader) as t:
            for batch_idx, (taxonomy_ids, model_ids, data) in enumerate(t):
                if config.dataset.name in ['PCN', 'Completion3D']:
                    for k, v in data.items():
                        data[k] = helpers.var_or_cuda(v)
                    partial = data['partial_cloud']
                    gt = data['gtcloud']
                elif config.dataset.name in ['ShapeNet-34', 'ShapeNet-Unseen21']:
                    npoints = config.dataset.n_points
                    gt = data.cuda()
                    partial, _ = misc.seprate_point_cloud(gt, npoints, [int(npoints * 1 / 4), int(npoints * 3 / 4)],
                                                          fixed_points=None)
                    partial = partial.cuda()


                pcds_pred = model(partial)
                loss_total, losses = loss_util.get_loss(pcds_pred, partial, gt, loss_func=config.train.loss_func)

                optimizer.zero_grad()
                loss_total.backward()
                optimizer.step()

                losses = [ls*multiplier for ls in losses]
                avg_meter_loss.update(losses)
                n_itr =  epoch_idx * n_batches + batch_idx
                train_writer.add_scalar('Loss/Batch/partial_matching', losses[0], n_itr)
                train_writer.add_scalar('Loss/Batch/cd_pc', losses[1], n_itr)
                train_writer.add_scalar('Loss/Batch/cd_p1', losses[2], n_itr)
                train_writer.add_scalar('Loss/Batch/cd_p2', losses[3], n_itr)
                train_writer.add_scalar('Loss/Batch/cd_p3', losses[4], n_itr)

                t.set_description(
                    '[Epoch %d/%d][Batch %d/%d]' % (epoch_idx, config.train.epochs, batch_idx + 1, n_batches))
                t.set_postfix(
                    loss='%s' % ['%.4f' % l for l in losses])

        scheduler.step()
        print('epoch: ', epoch_idx, 'optimizer: ', optimizer.param_groups[0]['lr'])
        train_writer.add_scalar('Loss/Epoch/partial_matching', avg_meter_loss.avg(0), epoch_idx)
        train_writer.add_scalar('Loss/Epoch/cd_pc', avg_meter_loss.avg(1), epoch_idx)
        train_writer.add_scalar('Loss/Epoch/cd_p1', avg_meter_loss.avg(2), epoch_idx)
        train_writer.add_scalar('Loss/Epoch/cd_p2', avg_meter_loss.avg(3), epoch_idx)
        train_writer.add_scalar('Loss/Epoch/cd_p3', avg_meter_loss.avg(4), epoch_idx)


        cd_eval = test(config, model=model, test_dataloader=test_dataloader, validation=True,
                       epoch_idx=epoch_idx, test_writer=val_writer)

        # Save checkpoints
        if epoch_idx % config.train.save_freq == 0 or cd_eval < best_metric:
            file_name = 'ckpt-best.pth' if cd_eval < best_metric else 'ckpt-epoch-%03d.pth' % epoch_idx
            output_path = os.path.join(config.train.path_checkpoints, file_name)
            torch.save({
                'epoch_index': epoch_idx,
                'best_metric': best_metric,
                'model': model.state_dict()
            }, output_path)

            logging.info('Saved checkpoint to %s ...' % output_path)
            if cd_eval < best_metric:
                best_metric = cd_eval

    train_writer.close()
    val_writer.close()






if __name__ == '__main__':
    args = get_args_from_command_line()

    config = yaml_reader.read_yaml(args.config)


    set_seed(config.train.seed)

    torch.backends.cudnn.benchmark = True
    train(config)
