# import datetime
# from os import path
# import math
# import random
# import numpy as np
# import torch
# from torch.utils.data import DataLoader, ConcatDataset
# import torch.distributed as distributed

# from model.trainer import XMemTrainer
# from dataset.static_dataset import StaticTransformDataset
# from dataset.vos_dataset import VOSDataset
# from dataset.mri_mha_dataset import MRIMHADataset


# from util.logger import TensorboardLogger
# from util.configuration import Configuration
# from util.load_subset import load_sub_davis, load_sub_yv

# # ==== Try to get Git info ====
# try:
#     import git
#     repo = git.Repo(".", search_parent_directories=True)
#     git_info = str(repo.active_branch) + ' ' + str(repo.head.commit.hexsha)
# except Exception as e:
#     print(f"[Warning] Git info not available: {e}")
#     git_info = "unknown"

# # ==== Check if running distributed ====
# if torch.distributed.is_available() and torch.distributed.is_initialized():
#     distributed_mode = True
# else:
#     distributed_mode = False

# if distributed_mode:
#     distributed.init_process_group(backend="nccl")
#     local_rank = torch.distributed.get_rank()
#     world_size = torch.distributed.get_world_size()
#     torch.cuda.set_device(local_rank)
#     print(f'Running in distributed mode. Rank {local_rank} of {world_size}')
# else:
#     local_rank = 0
#     world_size = 1
#     torch.cuda.set_device(0)
#     print("Running in non-distributed (single-GPU) mode.")
#     print(f'CUDA Device count: {torch.cuda.device_count()}')

# # ==== Parse Config ====
# raw_config = Configuration()
# raw_config.parse()

# if raw_config['benchmark']:
#     torch.backends.cudnn.benchmark = True

# network_in_memory = None
# stages = raw_config['stages']
# stages_to_perform = list(stages)

# for si, stage in enumerate(stages_to_perform):
#     torch.manual_seed(14159265)
#     np.random.seed(14159265)
#     random.seed(14159265)

#     stage_config = raw_config.get_stage_parameters(stage)
#     config = dict(**raw_config.args, **stage_config)

#     if config['exp_id'] != 'NULL':
#         config['exp_id'] = config['exp_id']+'_s%s' % stages[:si+1]

#     config['single_object'] = (stage == '0')
#     config['num_gpus'] = world_size
#     config['batch_size'] //= config['num_gpus']
#     config['num_workers'] //= config['num_gpus']
#     print(f'[INFO] Using {config["num_gpus"]} GPU(s).')

#     # ==== Logger ====
#     if local_rank == 0:
#         if config['exp_id'].lower() != 'null':
#             long_id = '%s_%s' % (datetime.datetime.now().strftime('%b%d_%H.%M.%S'), config['exp_id'])
#             print(f'I will take the role of logging as rank {local_rank}')
#         else:
#             long_id = None
#         logger = TensorboardLogger(config['exp_id'], long_id, git_info)
#         logger.log_string('hyperpara', str(config))
#         model = XMemTrainer(config, logger=logger,
#                             save_path=path.join('saves', long_id, long_id) if long_id else None,
#                             local_rank=local_rank, world_size=world_size).train()
#     else:
#         model = XMemTrainer(config, local_rank=local_rank, world_size=world_size).train()

#     # ==== Load Pretrained ====
#     if raw_config['load_checkpoint'] is not None:
#         total_iter = model.load_checkpoint(raw_config['load_checkpoint'])
#         raw_config['load_checkpoint'] = None
#     else:
#         total_iter = 0

#     if network_in_memory is not None:
#         model.load_network_in_memory(network_in_memory)
#         network_in_memory = None
#     elif raw_config['load_network'] is not None:
#         model.load_network(raw_config['load_network'])
#         raw_config['load_network'] = None

#     # ==== Dataset & Loader ====

#     def worker_init_fn(worker_id):
#         worker_seed = torch.initial_seed() % (2**31) + worker_id + local_rank * 100
#         np.random.seed(worker_seed)
#         random.seed(worker_seed)

#     def construct_loader(dataset):
#         if distributed_mode:
#             train_sampler = torch.utils.data.distributed.DistributedSampler(dataset, rank=local_rank, shuffle=True)
#         else:
#             train_sampler = torch.utils.data.RandomSampler(dataset)

#         train_loader = DataLoader(dataset, config['batch_size'], sampler=train_sampler,
#                                   num_workers=config['num_workers'], worker_init_fn=worker_init_fn, drop_last=True)
#         return train_sampler, train_loader

#     # ==== Custom dataset loading ====
#     def renew_trackrad_loader(data_root, mask_root):
#         # from dataset.vos_dataset import MRIMHADataset
#         train_dataset = MRIMHADataset(root_dir=data_root, num_frames=config['num_frames'])
#         return construct_loader(train_dataset)

#     # === Example: Replace with your dataset path ===
#     trackrad_data_root = path.expanduser(config.get("trackrad_image_root", "./TrackRAD/JPEGImages"))
#     trackrad_mask_root = path.expanduser(config.get("trackrad_mask_root", "./TrackRAD/Annotations"))
#     train_sampler, train_loader = renew_trackrad_loader(trackrad_data_root, trackrad_mask_root)

#     print(f'[DATASET] TrackRAD dataset size: {len(train_loader.dataset)}')

#     total_epoch = math.ceil(config['iterations'] / len(train_loader))
#     current_epoch = total_iter // len(train_loader)
#     print(f'[TRAIN] Training for approx {total_epoch} epochs')

#     # ==== Start Training ====
#     try:
#         while total_iter < config['iterations'] + config['finetune']:
#             train_sampler.set_epoch(current_epoch) if distributed_mode else None
#             current_epoch += 1
#             print(f'[TRAIN] Epoch: {current_epoch}')

#             model.train()
#             for data in train_loader:
#                 # ==== DEBUG: print input shape ====
#                 if isinstance(data, dict) and "rgb" in data:
#                     print(f'[DEBUG] Input RGB shape: {data["rgb"].shape}')
#                 else:
#                     print(f'[DEBUG] Raw input: {type(data)}')

#                 model.do_pass(data, total_iter)
#                 total_iter += 1

#                 if total_iter >= config['iterations'] + config['finetune']:
#                     break

#     finally:
#         if not config['debug'] and model.logger is not None and total_iter > 5000:
#             model.save_network(total_iter)
#             model.save_checkpoint(total_iter)

#     network_in_memory = model.XMem.module.state_dict()

# if distributed_mode:
#     distributed.destroy_process_group()


import datetime
from os import path
import math
import random
import numpy as np
import torch
from torch.utils.data import DataLoader
import torch.distributed as distributed

from model.trainer import XMemTrainer
from dataset.mri_mha_dataset import MRIMHADataset
from util.logger import TensorboardLogger
from util.configuration import Configuration

# ==== Try to get Git info ====
try:
    import git
    repo = git.Repo(".", search_parent_directories=True)
    git_info = str(repo.active_branch) + ' ' + str(repo.head.commit.hexsha)
except Exception as e:
    print(f"[Warning] Git info not available: {e}")
    git_info = "unknown"

# ==== Check if running distributed ====
if torch.distributed.is_available() and torch.distributed.is_initialized():
    distributed_mode = False
else:
    distributed_mode = False

if distributed_mode:
    distributed.init_process_group(backend="nccl")
    local_rank = torch.distributed.get_rank()
    world_size = torch.distributed.get_world_size()
    torch.cuda.set_device(local_rank)
    print(f'Running in distributed mode. Rank {local_rank} of {world_size}')
else:
    local_rank = 0
    world_size = 1
    torch.cuda.set_device(0)
    print("Running in non-distributed (single-GPU) mode.")
    print(f'CUDA Device count: {torch.cuda.device_count()}')

# ==== Parse Config ====
raw_config = Configuration()
raw_config.parse()

if raw_config['benchmark']:
    torch.backends.cudnn.benchmark = True

network_in_memory = None
stages = raw_config['stages']
stages_to_perform = list(stages)
# print("=======================")
# print(stages_to_perform)

for si, stage in enumerate(stages_to_perform):
    
    print(si)
    if(si==0):
        continue
    # 固定随机种子
    torch.manual_seed(14159265)
    np.random.seed(14159265)
    random.seed(14159265)

    stage_config = raw_config.get_stage_parameters(stage)
    config = dict(**raw_config.args, **stage_config)

    if config['exp_id'] != 'NULL':
        config['exp_id'] = config['exp_id'] + '_s%s' % stages[:si+1]

    config['single_object'] = (stage == '0')
    config['num_gpus'] = world_size
    config['batch_size'] //= config['num_gpus']
    config['num_workers'] //= config['num_gpus']
    config['batch_size'] = 1

    print(f'[INFO] Using {config["num_gpus"]} GPU(s).')
    

    # ==== Logger ====
    if local_rank == 0:
        if config['exp_id'].lower() != 'null':
            long_id = '%s_%s' % (datetime.datetime.now().strftime('%b%d_%H.%M.%S'), config['exp_id'])
            print(f'I will take the role of logging as rank {local_rank}')
        else:
            long_id = None
        logger = TensorboardLogger(config['exp_id'], long_id, git_info)
        logger.log_string('hyperpara', str(config))
        # print("====================")
        # print(long_id)
        # p=path.join('saves', long_id, long_id)
        # print(f"path:{p}")
        trainer = XMemTrainer(config, logger=logger,
                              # save_path=path.join('saves', long_id, long_id) if long_id else None,
                              save_path=path.join('save/','Weight'),
                              local_rank=local_rank, world_size=world_size)
    else:
        trainer = XMemTrainer(config, local_rank=local_rank, world_size=world_size)

    # ==== Load pretrained ====
    if raw_config['load_checkpoint'] is not None:
        total_iter = trainer.load_checkpoint(raw_config['load_checkpoint'])
        raw_config['load_checkpoint'] = None
    else:
        total_iter = 0

    if network_in_memory is not None:
        trainer.load_network_in_memory(network_in_memory)
        network_in_memory = None
    elif raw_config['load_network'] is not None:
        trainer.load_network(raw_config['load_network'])
        raw_config['load_network'] = None

    # ==== Dataset & Loader setup ====

    def worker_init_fn(worker_id):
        worker_seed = torch.initial_seed() % (2**31) + worker_id + local_rank * 100
        np.random.seed(worker_seed)
        random.seed(worker_seed)

    def construct_loader(dataset):
        if distributed_mode:
            train_sampler = torch.utils.data.distributed.DistributedSampler(dataset, rank=local_rank, shuffle=True)
        else:
            train_sampler = torch.utils.data.RandomSampler(dataset)

        train_loader = DataLoader(dataset, config['batch_size'], sampler=train_sampler,
                                  num_workers=config['num_workers'], worker_init_fn=worker_init_fn, drop_last=True)
        return train_sampler, train_loader

    # ==== Custom dataset loading ====
    def renew_trackrad_loader(data_root, mask_root=None):
        # mask_root参数目前不使用，兼容接口
        train_dataset = MRIMHADataset(root_dir=data_root, num_frames=config['num_frames'])
        # train_dataset = MRIMHADataset(images_dir=images_dir, targets_dir=targets_dir, num_frames=config['num_frames'])
        return construct_loader(train_dataset)

    # === 替换为你的数据集路径 ===
    trackrad_data_root = path.expanduser(config.get("trackrad_image_root", "../dataset/trackrad2025_labeled_training_data"))
    trackrad_mask_root = path.expanduser(config.get("trackrad_mask_root", "../dataset/trackrad2025_labeled_training_data"))

    train_sampler, train_loader = renew_trackrad_loader(trackrad_data_root, trackrad_mask_root)

    print(f'[DATASET] TrackRAD dataset size: {len(train_loader.dataset)}')

    total_epoch = math.ceil(config['iterations'] / len(train_loader))
    current_epoch = total_iter // len(train_loader)
    print(f'[TRAIN] Training for approx {total_epoch} epochs')

    # ==== Start Training ====
    try:
        while total_iter < config['iterations'] + config['finetune']:
            if distributed_mode:
                train_sampler.set_epoch(current_epoch)
            current_epoch += 1
            print(f'[TRAIN] Epoch: {current_epoch}')

            trainer.model.train()
            # trainer.net.train()

            for data in train_loader:
                # ==== DEBUG: print input shape ====
                # if isinstance(data, dict) and "rgb" in data:
                #     # 你根据MRIMHADataset返回数据格式调试输出，可能是list等结构需调整
                #     print(f'[DEBUG] Input RGB type: {type(data["rgb"])}')
                # else:
                #     print(f'[DEBUG] Raw input: {type(data)}')

                trainer.do_pass(data, total_iter)
                total_iter += 1

                if total_iter >= config['iterations'] + config['finetune']:
                    break

    finally:
        if not config['debug'] and trainer.logger is not None and total_iter > 1000:
            trainer.save_network(total_iter)
            trainer.save_checkpoint(total_iter)

    network_in_memory = trainer.XMem.state_dict()

if distributed_mode:
    distributed.destroy_process_group()
