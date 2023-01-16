import os
import json
import torch
import argparse
from tqdm import tqdm
from utils import helpers, average_meter, yaml_reader, loss_util, misc
from core import builder

def get_args_from_command_line():
    parser = argparse.ArgumentParser(description='The argument parser of SnowflakeNet')
    parser.add_argument('--config', type=str, default='./configs/pcn_cd1.yaml', help='Configuration File')
    args = parser.parse_args()
    return args

crop_ratio = {
    'easy': 1/4,
    'median' :1/2,
    'hard':3/4
}

def test(config, model=None, test_dataloader=None, epoch_idx=-1, validation=False, test_writer=None, completion_loss=None):
    if test_dataloader is None:
        test_dataloader = builder.make_dataloader(config, config.test.split)

    if model is None:
        model = builder.make_model(config)
        if torch.cuda.is_available():
            model = torch.nn.DataParallel(model).cuda()

        if not os.path.exists(config.test.model_path):
            raise Exception('checkpoints does not exists: {}'.format(config.test.model_path))

        print('Recovering from %s ...' % (config.test.model_path), end='')
        checkpoint = torch.load(config.test.model_path)
        model.load_state_dict(checkpoint['model'])
        print('recovered!')

    model.eval()

    n_samples = len(test_dataloader)
    test_losses = average_meter.AverageMeter(['partial_matching', 'cdc', 'cd1', 'cd2', 'cd3'])
    test_metrics = average_meter.AverageMeter([config.test.loss_func])
    category_metrics = dict()

    multiplier = 1.0
    if config.test.loss_func == 'cd_l1':
        multiplier = 1e3
    elif config.test.loss_func == 'cd_l2':
        multiplier = 1e4
    elif config.test.loss_func == 'emd':
        multiplier = 1e2

    if completion_loss is None:
        completion_loss = loss_util.Completionloss(loss_func=config.test.loss_func)

    with tqdm(test_dataloader) as t:
        for model_idx, (taxonomy_id, model_id, data) in enumerate(t):
            taxonomy_id = taxonomy_id[0] if isinstance(taxonomy_id[0], str) else taxonomy_id[0].item()
            model_id = model_id[0]

            if config.dataset.name  in ['PCN', 'Completion3D']:
                with torch.no_grad():
                    for k, v in data.items():
                        data[k] = helpers.var_or_cuda(v)

                    partial = data['partial_cloud']
                    gt = data['gtcloud']

                    b, n, _ = partial.shape

                    pcds_pred = model(partial.contiguous())

                    # print('gt.shape', gt.shape)
                    # print('p3.shape', pcds_pred[-1].shape)
                    loss_total, losses = completion_loss.get_loss(pcds_pred, partial, gt)

                    partial_matching = losses[0].item() * multiplier
                    loss_c = losses[1].item() * multiplier
                    loss_1 = losses[2].item() * multiplier
                    loss_2 = losses[3].item() * multiplier
                    loss_3 = losses[4].item() * multiplier

                    _metrics = [loss_3]
                    test_losses.update([partial_matching, loss_c, loss_1, loss_2, loss_3])

                    test_metrics.update(_metrics)
                    if taxonomy_id not in category_metrics:
                        category_metrics[taxonomy_id] = average_meter.AverageMeter([config.test.loss_func])
                    category_metrics[taxonomy_id].update(_metrics)
            elif config.dataset.name in ['ShapeNet-34', 'ShapeNet-Unseen21']:
                gt = data.cuda()
                npoints = config.dataset.n_points
                if validation:
                    choice = [None]
                    num_crop = [int(npoints * 1/4) , int(npoints * 3/4)]
                else:
                    choice = [torch.Tensor([1, 1, 1]), torch.Tensor([1, 1, -1]), torch.Tensor([1, -1, 1]),
                              torch.Tensor([-1, 1, 1]),
                              torch.Tensor([-1, -1, 1]), torch.Tensor([-1, 1, -1]), torch.Tensor([1, -1, -1]),
                              torch.Tensor([-1, -1, -1])]
                    num_crop = int(npoints * crop_ratio[config.test.mode])
                for item in choice:
                    partial, _ = misc.seprate_point_cloud(gt, npoints, num_crop, fixed_points = item)

                    b, n, _ = partial.shape

                    pcds_pred = model(partial.contiguous())

                    # print('gt.shape', gt.shape)
                    # print('p3.shape', pcds_pred[-1].shape)
                    loss_total, losses = completion_loss.get_loss(pcds_pred, partial, gt)

                    partial_matching = losses[0].item() * multiplier
                    loss_c = losses[1].item() * multiplier
                    loss_1 = losses[2].item() * multiplier
                    loss_2 = losses[3].item() * multiplier
                    loss_3 = losses[4].item() * multiplier

                    _metrics = [loss_3]
                    test_losses.update([partial_matching, loss_c, loss_1, loss_2, loss_3])

                    test_metrics.update(_metrics)
                    if taxonomy_id not in category_metrics:
                        category_metrics[taxonomy_id] = average_meter.AverageMeter([config.test.loss_func])
                    category_metrics[taxonomy_id].update(_metrics)
            else:
                raise NotImplementedError(f'Dataset {config.dataset.name} not supported! ')

            t.set_description('Test[%d/%d] Taxonomy = %s Sample = %s Losses = %s Metrics = %s' %
                              (model_idx + 1, n_samples, taxonomy_id, model_id,
                               ['%.4f' % l for l in test_losses.val()
                                ], '%.4f' % loss_3))

    shapenet_dict = json.load(open('./category_files/shapenet_synset_dict.json', 'r'))
    print('============================ TEST RESULTS ============================')
    print('Taxonomy', end='\t')
    print('#Sample', end='\t')
    for metric in test_metrics.items:
        print(metric, end='\t')
    print('#ModelName\t')

    for taxonomy_id in category_metrics:
        print(taxonomy_id, end='\t')
        print(category_metrics[taxonomy_id].count(0), end='\t')
        for value in category_metrics[taxonomy_id].avg():
            print('%.4f' % value, end='\t')
        print(shapenet_dict[taxonomy_id]+'\t')

    print('Overall', end='\t\t\t')
    for value in test_metrics.avg():
        print('%.4f' % value, end='\t')
    print('\n')

    print('Epoch ', epoch_idx, end='\t')
    for value in test_losses.avg():
        print('%.4f' % value, end='\t')
    print('\n')

    # Add testing results to TensorBoard
    if test_writer is not None:
        test_writer.add_scalar('Loss/Epoch/partial_matching', test_losses.avg(0), epoch_idx)
        test_writer.add_scalar('Loss/Epoch/loss_c', test_losses.avg(1), epoch_idx)
        test_writer.add_scalar('Loss/Epoch/loss_1', test_losses.avg(2), epoch_idx)
        test_writer.add_scalar('Loss/Epoch/loss_2', test_losses.avg(3), epoch_idx)
        test_writer.add_scalar('Loss/Epoch/loss_3', test_losses.avg(4), epoch_idx)
        for i, metric in enumerate([config.test.loss_func]):
            test_writer.add_scalar('Metric/%s' % metric, test_metrics.avg(i), epoch_idx)

    return test_losses.avg(4)


if __name__ == '__main__':
    args = get_args_from_command_line()

    config = yaml_reader.read_yaml(args.config)
    torch.backends.cudnn.benchmark = True
    os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(str(x) for x in config.test.gpu)
    test(config)