import os
import sys
sys.path.append('..')
import torch
import logging
import argparse
import torch.utils.data
from datetime import datetime
from tqdm import tqdm
from time import time
from tensorboardX import SummaryWriter

from dataset.dataloader import PUGANDataset, PUGANTestset, collate_fn, collate_fn_test
from models.model_pu import ModelPU
from test import test
from utils import PULoss, read_yaml
from evaluate import TensorEvaluator

def get_args_from_command_line():
    parser = argparse.ArgumentParser(description='The argument spd_pu of SPD for point cloud upsampling')
    parser.add_argument('--config', type=str, default='./configs/spd_pu.yaml', help='Configuration File')
    args = parser.parse_args()
    return args

def lr_lambda(epoch):
    if epoch >= 0 and epoch <=80:
        return 1
    elif epoch > 80 and epoch <= 120:
        return 0.5
    elif epoch > 120 and epoch <= 130:
        return 0.4
    elif epoch > 130 and epoch <= 140:
        return 0.2
    else:
        return 0.1


def train(config):
    # Enable the inbuilt cudnn auto-tuner to find the best algorithm to use

    train_dataset = PUGANDataset(config.dataset.train_path)
    test_dataset = PUGANTestset(path=config.dataset.test_gt_path, path_inp=config.dataset.test_input_path)

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=config.train.batch_size,
        num_workers=config.train.num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
        shuffle=True,
        drop_last=False
    )

    test_dataloader = torch.utils.data.DataLoader(
        test_dataset,
        collate_fn=collate_fn_test,
        batch_size=1,
        num_workers=1,
        shuffle=False,
        drop_last=False
    )

    # Set up folders for logs and checkpoints
    output_dir = os.path.join(config.train.save_path, '%s', datetime.now().isoformat())
    path_checkpoints = output_dir % 'checkpoints'
    path_logs = output_dir % 'logs'

    if not os.path.exists(path_checkpoints):
        os.makedirs(path_checkpoints)

    # Create tensorboard writers
    train_writer = SummaryWriter(os.path.join(path_logs, 'train'))
    val_writer = SummaryWriter(os.path.join(path_logs, 'test'))

    # Create the networks
    # grnet = GRNet(cfg)
    # grnet.apply(utils.helpers.init_weights)
    # logging.debug('Parameters in GRNet: %d.' % utils.helpers.count_parameters(grnet))
    model = ModelPU(up_factors=config.model.up_factors)
    # Move the network to GPU if possible

    if torch.cuda.is_available():
        model = torch.nn.DataParallel(model).cuda()



    # Load pre-trained model if exists
    init_epoch = 0
    best_metrics = float('inf')

    if config.train.weights is not None:
        logging.info('Recovering from %s ...' % (config.weights))
        checkpoint = torch.load(config.weights)
        best_metrics = checkpoint['best_cd']
        model.load_state_dict(checkpoint['model'])
        init_epoch = checkpoint['epoch_index']
        logging.info('Recover complete. Current epoch = #%d; best cd = %s.' % (init_epoch, best_metrics))

    # Create the optimizers
    optimizer = torch.optim.Adam([{'params': filter(lambda p: p.requires_grad, model.parameters()), 'initial_lr': config.train.base_lr}],
                                 lr=config.train.base_lr,
                                 betas=config.train.betas)
    lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer,
                                                     lr_lambda=lr_lambda, last_epoch=init_epoch)

    pu_loss = PULoss()
    tensor_evaluator = TensorEvaluator()
    # Training/Testing the network
    for epoch_idx in range(init_epoch + 1, config.train.n_epochs + 1):
        # metrics = test_net(cfg, epoch_idx, val_data_loader, val_writer, model)
        epoch_start_time = time()
        model.train()

        total_cd = 0

        n_batches = len(train_dataset)

        # cd, hd = test_net(model, test_dataloader, epoch_idx, best_cd=float('inf'), path=cfg.DIR.OUT_PATH)
        with tqdm(train_dataloader) as t:
            for batch_idx, (inp, gt, radius) in enumerate(t):
                inp = inp.cuda()
                gt = gt.cuda()
                radius = radius.cuda()


                pcds = model(inp)

                _loss, loss_cd = pu_loss.get_loss(pcds, gt, radius)

                optimizer.zero_grad()
                _loss.backward()
                optimizer.step()

                total_cd += loss_cd.item()*1e3



                t.set_description('[Epoch %d/%d][Batch %d/%d]' % (epoch_idx, config.train.n_epochs, batch_idx + 1, n_batches))
                t.set_postfix(loss='{}'.format(loss_cd.item()))

                # torch.cuda.empty_cache()

        lr_scheduler.step()
        epoch_end_time = time()

        avg_cd = total_cd / n_batches


        train_writer.add_scalar('Loss/Epoch/cd', avg_cd, epoch_idx)

        logging.info(
            '[Epoch %d/%d] EpochTime = %.3f (s) Losses = %s' %
            (epoch_idx, config.train.n_epochs, epoch_end_time - epoch_start_time, ['%.4f' % l for l in [avg_cd]]))

        # Validate the current model
        cd, hd = test(config, model=model, data_loader=test_dataloader, epoch=epoch_idx,
                      best_cd=best_metrics, path=config.train.save_path, tensor_evaluator=tensor_evaluator)

        val_writer.add_scalar('Loss/Epoch/cd', cd, epoch_idx)
        val_writer.add_scalar('Loss/Epoch/hd', hd, epoch_idx)

        # Save ckeckpoints
        if epoch_idx % config.train.save_freq == 0 or cd < best_metrics:
            file_name = 'ckpt-best-{:03d}-{:.4f}.pth'.format(epoch_idx, cd*1e4) if cd < best_metrics else 'ckpt-epoch-%03d.pth' % epoch_idx
            output_path = os.path.join(path_checkpoints, file_name)
            torch.save({
                'epoch_index': epoch_idx,
                'best_cd': cd,
                'model': model.state_dict()
            }, output_path)  # yapf: disable

            print('Saved checkpoint to %s ...' % output_path)
            if cd < best_metrics:
                best_metrics = cd

    train_writer.close()
    val_writer.close()


if __name__ == '__main__':
    args = get_args_from_command_line()

    config = read_yaml(args.config)
    torch.backends.cudnn.benchmark = True
    os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(str(x) for x in config.train.gpu)
    train(config)