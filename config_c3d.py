# -*- coding: utf-8 -*-
# @Author: XP

from easydict import EasyDict as edict

__C                                              = edict()
cfg                                              = __C

#
# Dataset Config
#
__C.DATASETS                                     = edict()
__C.DATASETS.COMPLETION3D                        = edict()
__C.DATASETS.COMPLETION3D.CATEGORY_FILE_PATH     = './datasets/Completion3D.json'
__C.DATASETS.COMPLETION3D.PARTIAL_POINTS_PATH    = '/path/to/datasets/Completion3D/%s/partial/%s/%s.h5'
__C.DATASETS.COMPLETION3D.COMPLETE_POINTS_PATH   = '/path/to/datasets/Completion3D/%s/gt/%s/%s.h5'
__C.DATASETS.SHAPENET                            = edict()
__C.DATASETS.SHAPENET.CATEGORY_FILE_PATH         = './datasets/ShapeNet.json'
__C.DATASETS.SHAPENET.N_RENDERINGS               = 8
__C.DATASETS.SHAPENET.N_POINTS                   = 2048
__C.DATASETS.SHAPENET.PARTIAL_POINTS_PATH        = '/path/to/datasets/PCN/%s/partial/%s/%s/%02d.pcd'
__C.DATASETS.SHAPENET.COMPLETE_POINTS_PATH       = '/path/to/datasets/PCN/%s/complete/%s/%s.pcd'

#
# Dataset
#
__C.DATASET                                      = edict()
# Dataset Options: Completion3D, ShapeNet, ShapeNetCars
__C.DATASET.TRAIN_DATASET                        = 'Completion3D'
__C.DATASET.TEST_DATASET                         = 'Completion3D'

#
# Constants
#
__C.CONST                                        = edict()

__C.CONST.NUM_WORKERS                            = 8
__C.CONST.N_INPUT_POINTS                         = 2048

#
# Directories
#

__C.DIR                                          = edict()
__C.DIR.OUT_PATH                                 = 'outpath'
__C.CONST.DEVICE                                 = '0'
# __C.CONST.WEIGHTS                                = '/data/xp/experiments/snowflake_test/c3d/checkpoints/2021-08-22T11:25:50.879135/ckpt-best.pth' # 'ckpt-best.pth'  # specify a path to run test and inference

#
# Memcached
#
__C.MEMCACHED                                    = edict()
__C.MEMCACHED.ENABLED                            = False
__C.MEMCACHED.LIBRARY_PATH                       = '/mnt/lustre/share/pymc/py3'
__C.MEMCACHED.SERVER_CONFIG                      = '/mnt/lustre/share/memcached_client/server_list.conf'
__C.MEMCACHED.CLIENT_CONFIG                      = '/mnt/lustre/share/memcached_client/client.conf'

#
# Network
#
__C.NETWORK                                      = edict()
__C.NETWORK.N_SAMPLING_POINTS                    = 2048

#
# Train
#
__C.TRAIN                                        = edict()
__C.TRAIN.BATCH_SIZE                             = 32
__C.TRAIN.N_EPOCHS                               = 150
__C.TRAIN.SAVE_FREQ                              = 25
__C.TRAIN.LEARNING_RATE                          = 0.001
__C.TRAIN.LR_MILESTONES                          = [50, 100, 150, 200, 250]
__C.TRAIN.LR_DECAY_STEP                          = 50
__C.TRAIN.WARMUP_STEPS                           = 200
__C.TRAIN.GAMMA                                  = .5
__C.TRAIN.BETAS                                  = (.9, .999)
__C.TRAIN.WEIGHT_DECAY                           = 0

#
# Test
#
__C.TEST                                         = edict()
__C.TEST.METRIC_NAME                             = 'ChamferDistance'
