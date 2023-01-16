import torch
import logging
import math
import random
import munch
import yaml
import os
import sys
import argparse
import warnings
import time as timetmp
import torch.optim as optim
from tqdm import tqdm
from time import time

from utils.model_utils import LossSVR # loss_snowflake, calc_cd
from utils.train_utils import save_model, AverageValueMeter
from dataset_svr.trainer_dataset import build_dataset
from models.model_svr import ModelSVR

warnings.filterwarnings("ignore")

def setFolders(args):

    LOG_DIR = args.dir_outpath
    MODEL_NAME = '%s-%s'%('SPD', timetmp.strftime("%m%d_%H%M", timetmp.localtime()))

    OUT_DIR = os.path.join(LOG_DIR, MODEL_NAME)
    args.dir_checkpoints = os.path.join(OUT_DIR, 'checkpoints')
    if not os.path.exists(OUT_DIR): os.makedirs(OUT_DIR)
    if not os.path.exists(args.dir_checkpoints):
        os.makedirs(args.dir_checkpoints)


    os.system('cp -r models %s' % (OUT_DIR))
    os.system('cp train_svr.py %s' % (OUT_DIR))

    os.system('cp -r cfgs %s' % (OUT_DIR))

    LOG_FOUT = open(os.path.join(OUT_DIR, 'log_%s.csv' %(MODEL_NAME)), 'w')
    return MODEL_NAME, OUT_DIR, LOG_FOUT

def log_string(out_str, LOG_FOUT):
    LOG_FOUT.write(out_str+'\n')
    LOG_FOUT.flush()

def train():
    # Set up folders for logs and checkpoints
    exp_name, log_dir, LOG_FOUT = setFolders(args)

    log_string('EPOCH,CD_L1,BEST CDL1,CD_L2,BEST CDL2', LOG_FOUT)

    logging.basicConfig(level=logging.INFO, handlers=[logging.FileHandler(os.path.join(log_dir, 'train.log')),
                                                      logging.StreamHandler(sys.stdout)])
                                                      
    logging.info(str(args))

    metrics = ['cd_p', 'cd_t', 'f1']
    best_epoch_losses = {m: (0, 0) if m == 'f1' else (0, math.inf) for m in metrics}
    train_loss_meter = AverageValueMeter()
    val_loss_meters = {m: AverageValueMeter() for m in metrics}

    dataloader, dataloader_test = build_dataset(args)

    if not args.manual_seed:
        seed = random.randint(1, 10000)
    else:
        seed = int(args.manual_seed)
    logging.info('Random Seed: %d' % seed)
    random.seed(seed)
    torch.manual_seed(seed)

    net = ModelSVR(
        dim_feat=args.model.dim_feat,
        num_pc=args.model.num_pc,
        num_p0=args.model.num_p0,
        radius=args.model.radius,
        bounding=args.model.bounding,
        up_factors=args.model.up_factors,
    )
    net = torch.nn.DataParallel(net)
    net.cuda()

    print('# encoder parameters:', sum(param.numel() for param in net.module.encoder.parameters()))
    print('# decoder parameters:', sum(param.numel() for param in net.module.decoder.parameters()))

    lr = args.lr

    if args.lr_decay:
        if args.lr_decay_interval and args.lr_step_decay_epochs:
            raise ValueError('lr_decay_interval and lr_step_decay_epochs are mutually exclusive!')
        if args.lr_step_decay_epochs:
            decay_epoch_list = [int(ep.strip()) for ep in args.lr_step_decay_epochs.split(',')]
            decay_rate_list = [float(rt.strip()) for rt in args.lr_step_decay_rates.split(',')]

    optimizer = getattr(optim, args.optimizer)
    betas = args.betas.split(',')
    betas = (float(betas[0].strip()), float(betas[1].strip()))
    optimizer = optimizer(filter(lambda p: p.requires_grad,net.module.parameters()), lr=lr, weight_decay=args.weight_decay, betas=betas)

    if args.varying_constant:
        varying_constant_epochs = [int(ep.strip()) for ep in args.varying_constant_epochs.split(',')]
        varying_constant = [float(c.strip()) for c in args.varying_constant.split(',')]
        assert len(varying_constant) == len(varying_constant_epochs) + 1

    best_cd_l1 = float("inf")
    best_cd_l2 = float("inf")

    loss_svr = LossSVR()

    for epoch in range(args.start_epoch, args.nepoch):
        epoch_start_time = time()
        total_cd_l1 = 0
        total_cd_l2 = 0

        train_loss_meter.reset()
        net.module.train()

        if args.lr_decay:
            if args.lr_decay_interval:
                if epoch > 0 and epoch % args.lr_decay_interval == 0:
                    lr = lr * args.lr_decay_rate
            elif args.lr_step_decay_epochs:
                if epoch in decay_epoch_list:
                    lr = lr * decay_rate_list[decay_epoch_list.index(epoch)]
            if args.lr_clip:
                lr = max(lr, args.lr_clip)
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr

        n_batches = len(dataloader)
        with tqdm(dataloader) as t:
            for batch_idx, data in enumerate(t):
                optimizer.zero_grad()

                images = data['image'].cuda()
                gt = data['points'].cuda()
                batch_size = gt.shape[0]

                pred_points = net(images)

                net_loss, loss_t = loss_svr.calc_cd(pred_points[-1], gt)

                net_loss = net_loss.mean()
                net_loss_all = loss_svr.loss_snowflake(pred_points, gt)

                train_loss_meter.update(net_loss.item())
                net_loss_all.backward(torch.squeeze(torch.ones(torch.cuda.device_count())).cuda())
                optimizer.step()

                cd_l1_item = net_loss.item() * 1e2
                total_cd_l1 += cd_l1_item
                cd_l2_item = loss_t.item() * 1e4
                total_cd_l2 += cd_l2_item


                t.set_description('[Epoch %d/%d][Batch %d/%d]' % (epoch, args.nepoch, batch_idx + 1, n_batches))
                t.set_postfix(loss='%s' % ['%.4f' % l for l in [cd_l1_item, cd_l2_item]])
        
        avg_cd_l1 = total_cd_l1 / n_batches
        avg_cd_l2 = total_cd_l2 / n_batches
        
        epoch_end_time = time()
        logging.info(' ')
        logging.info(
            exp_name + '[Epoch %d/%d] EpochTime = %.3f (s) Losses = %s' %
            (epoch, args.nepoch, epoch_end_time - epoch_start_time, ['%.4f' % l for l in [avg_cd_l1, avg_cd_l2]]))

        if epoch % args.epoch_interval_to_save == 0:
            save_model(str(log_dir) + '/checkpoints/' + str(epoch) + 'network.pth', net)
            logging.info("Saving net...")

        if epoch % args.epoch_interval_to_val == 0 or epoch == args.nepoch - 1:
            best_cd_l1, best_cd_l2 = val(net, epoch, val_loss_meters, dataloader_test, best_epoch_losses, LOG_FOUT, log_dir, best_cd_l1, best_cd_l2)


def val(net, curr_epoch_num, val_loss_meters, dataloader_test, best_epoch_losses, LOG_FOUT, log_dir, best_cd_l1, best_cd_l2, loss_svr=None):
    
    val_start_time = time()
    metrics_val = ['cd_t']
    val_loss_meters = {m: AverageValueMeter() for m in metrics_val}
    logging.info('Testing...')
    for v in val_loss_meters.values():
        v.reset()
    net.module.eval()

    total_cd_l1 = 0
    total_cd_l2 = 0
    n_batches = len(dataloader_test)

    if loss_svr is None:
        loss_svr = LossSVR()

    with torch.no_grad():
        for i, data in enumerate(dataloader_test):
            
            images = data['image'].cuda()
            gt = data['points'].cuda()

            batch_size = gt.shape[0]

            pred_points = net(images)

            loss_p, loss_t = loss_svr.calc_cd(pred_points[-1], gt)

            cd_l1_item = loss_p.item() * 1e2
            cd_l2_item = loss_t.item() * 1e4
            total_cd_l1 += cd_l1_item
            total_cd_l2 += cd_l2_item
        
        avg_cd_l1 = total_cd_l1 / n_batches
        avg_cd_l2 = total_cd_l2 / n_batches

        if avg_cd_l1 < best_cd_l1:
            best_cd_l1 = avg_cd_l1
            save_model(str(log_dir) + '/checkpoints/bestl1_network.pth', net)
            logging.info("Saving net...")

        if avg_cd_l2 < best_cd_l2:
            best_cd_l2 = avg_cd_l2
            save_model(str(log_dir) + '/checkpoints/bestl2_network.pth', net)
            logging.info("Saving net...")

        log_string('%d,%.2f,%.2f,%.2f,%.2f'%(curr_epoch_num, avg_cd_l1, best_cd_l1, avg_cd_l2, best_cd_l2), LOG_FOUT)

        val_end_time = time()

        logging.info(
            '[Epoch %d/%d] TestTime = %.3f (s) Curr_cdl1 = %s Best_cdl1 = %s Curr_cdl2 = %s Best_cdl2 = %s' %
            (curr_epoch_num, args.nepoch, val_end_time - val_start_time, avg_cd_l1, best_cd_l1, avg_cd_l2, best_cd_l2))        

    return best_cd_l1, best_cd_l2



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train config file')
    parser.add_argument('-c', '--config', help='path to config file', required=True)
    parser.add_argument('-gpu', '--gpu_id', help='gpu_id', required=True)
    arg = parser.parse_args()
    config_path = arg.config
    args = munch.munchify(yaml.safe_load(open(config_path)))

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(arg.gpu_id)
    print('Using gpu:' + str(arg.gpu_id))
    train()

