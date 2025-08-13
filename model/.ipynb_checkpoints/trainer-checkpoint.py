"""
trainer.py - warpper and utility functions for network training
Compute loss, back-prop, update parameters, logging, etc.
"""


import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from model.network import XMem
from model.losses import LossComputer
from util.log_integrator import Integrator
from util.image_saver import pool_pairs

def debug_tensor(name, tensor):
    if isinstance(tensor, torch.Tensor):
        print(f"[调试] {name} shape: {tuple(tensor.shape)} device: {tensor.device}")
    else:
        print(f"[调试] {name} 类型: {type(tensor)}")

class XMemTrainer:
    def __init__(self, config, logger=None, save_path=None, local_rank=0, world_size=1):
        self.config = config
        self.num_frames = config['num_frames']
        self.num_ref_frames = config['num_ref_frames']
        self.deep_update_prob = config['deep_update_prob']
        self.local_rank = local_rank

        # self.XMem = nn.parallel.DistributedDataParallel(
        #     XMem(config).cuda(), 
        #     device_ids=[local_rank], output_device=local_rank, broadcast_buffers=False)
        self.XMem = XMem(config).cuda()
        self.model = XMem(config).cuda()

        # Set up logger when local_rank=0
        self.logger = logger
        self.save_path = save_path
        if logger is not None:
            self.last_time = time.time()
            self.logger.log_string('model_size', str(sum([param.nelement() for param in self.XMem.parameters()])))
        self.train_integrator = Integrator(self.logger, distributed=True, local_rank=local_rank, world_size=world_size)
        self.loss_computer = LossComputer(config)

        self.train()
        self.optimizer = optim.AdamW(filter(
            lambda p: p.requires_grad, self.XMem.parameters()), lr=config['lr'], weight_decay=config['weight_decay'])
        self.scheduler = optim.lr_scheduler.MultiStepLR(self.optimizer, config['steps'], config['gamma'])
        if config['amp']:
            self.scaler = torch.cuda.amp.GradScaler()

        # Logging info
        self.log_text_interval = config['log_text_interval']
        self.log_image_interval = config['log_image_interval']
        self.save_network_interval = config['save_network_interval']
        self.save_checkpoint_interval = config['save_checkpoint_interval']
        if config['debug']:
            self.log_text_interval = self.log_image_interval = 1

#     def do_pass(self, data, it=0):
#         torch.set_grad_enabled(self._is_train)

#         for k, v in data.items():
#             if type(v) != list and type(v) != dict and type(v) != int:
#                 print(f"[DEBUG] Before cuda transfer: {k} shape: {v.shape}")
#                 data[k] = v.cuda(non_blocking=True)
#                 print(f"[DEBUG] After cuda transfer: {k} shape: {data[k].shape}")
#         out = {}
#         frames = data['rgb']
#         first_frame_gt = data['first_frame_gt'].float()
#         b = frames.shape[0]
#         num_filled_objects = [o.item() for o in data['info']['num_objects']]
#         num_objects = max(num_filled_objects)

#         device = frames.device

#         print(f"[DEBUG] frames shape right after loading from data: {frames.shape}")
#         print(f"[DEBUG] first_frame_gt shape: {first_frame_gt.shape}")

#         if 'selector' in data:
#             print(f"[DEBUG] selector shape before unsqueeze: {data['selector'].shape}")
#             selector = data['selector'].unsqueeze(2).unsqueeze(2)
#             print(f"[DEBUG] selector shape after unsqueeze: {selector.shape}")
#         else:
#             print("[DEBUG] selector key not found in data")

#         # with torch.cuda.amp.autocast(enabled=self.config['amp']):
#         with torch.cuda.amp.autocast(enabled=False):
#             # image features never change, compute once
#             print(f"[DEBUG] frames shape before encode_key call: {frames.shape}")
#             key, shrinkage, selection, f16, f8, f4 = self.XMem('encode_key', frames)
#             print(f"[DEBUG] key shape: {key.shape}")

#             filler_one = torch.zeros(1, dtype=torch.int64)
#             hidden = torch.zeros((b, num_objects, self.config['hidden_dim'], *key.shape[-2:]))
            
#         if hidden is not None:
#             print(f"[DEBUG] hidden type: {type(hidden)}")

#             if isinstance(hidden, torch.Tensor):
#                 hidden = hidden.to(device)
#                 print(f"[DEBUG] hidden moved to: {hidden.device}")
#             elif isinstance(hidden, (tuple, list)) and all(isinstance(h, torch.Tensor) for h in hidden):
#                 hidden = tuple(h.to(device) for h in hidden)
#                 print(f"[DEBUG] hidden tuple moved to devices: {[h.device for h in hidden]}")
#             else:
#                 print("[ERROR] Unexpected hidden structure, cannot move to device")


#             print(f"[DEBUG] frame device: {frames[:,0].device}")
#             print(f"[DEBUG] feat device: {f16[:,0].device}")
#             print(f"[DEBUG] mask device: {first_frame_gt[:,0].device}")
            
#             v16, hidden = self.XMem('encode_value', frames[:,0], f16[:,0], hidden, first_frame_gt[:,0])
#             values = v16.unsqueeze(3) # add the time dimension

#             for ti in range(1, self.num_frames):
#                 if ti <= self.num_ref_frames:
#                     ref_values = values
#                     ref_keys = key[:,:,:ti]
#                     ref_shrinkage = shrinkage[:,:,:ti] if shrinkage is not None else None
#                 else:
#                     # pick num_ref_frames random frames
#                     # this is not very efficient but I think we would 
#                     # need broadcasting in gather which we don't have
#                     indices = [
#                         torch.cat([filler_one, torch.randperm(ti-1)[:self.num_ref_frames-1]+1])
#                     for _ in range(b)]
#                     ref_values = torch.stack([
#                         values[bi, :, :, indices[bi]] for bi in range(b)
#                     ], 0)
#                     ref_keys = torch.stack([
#                         key[bi, :, indices[bi]] for bi in range(b)
#                     ], 0)
#                     ref_shrinkage = torch.stack([
#                         shrinkage[bi, :, indices[bi]] for bi in range(b)
#                     ], 0) if shrinkage is not None else None

#                 # Segment frame ti
#                 memory_readout = self.XMem('read_memory', key[:,:,ti], selection[:,:,ti] if selection is not None else None, 
#                                         ref_keys, ref_shrinkage, ref_values)
#                 hidden, logits, masks = self.XMem('segment', (f16[:,ti], f8[:,ti], f4[:,ti]), memory_readout, 
#                         hidden, selector, h_out=(ti < (self.num_frames-1)))

#                 # No need to encode the last frame
#                 if ti < (self.num_frames-1):
#                     is_deep_update = np.random.rand() < self.deep_update_prob
#                     v16, hidden = self.XMem('encode_value', frames[:,ti], f16[:,ti], hidden, masks, is_deep_update=is_deep_update)
#                     values = torch.cat([values, v16.unsqueeze(3)], 3)

#                 out[f'masks_{ti}'] = masks
#                 out[f'logits_{ti}'] = logits

#             if self._do_log or self._is_train:
#                 losses = self.loss_computer.compute({**data, **out}, num_filled_objects, it)

#                 # Logging
#                 if self._do_log:
#                     self.integrator.add_dict(losses)
#                     if self._is_train:
#                         if it % self.log_image_interval == 0 and it != 0:
#                             if self.logger is not None:
#                                 images = {**data, **out}
#                                 size = (384, 384)
#                                 self.logger.log_cv2('train/pairs', pool_pairs(images, size, num_filled_objects), it)

#             if self._is_train:
#                 if (it) % self.log_text_interval == 0 and it != 0:
#                     if self.logger is not None:
#                         self.logger.log_scalar('train/lr', self.scheduler.get_last_lr()[0], it)
#                         self.logger.log_metrics('train', 'time', (time.time()-self.last_time)/self.log_text_interval, it)
#                     self.last_time = time.time()
#                     self.train_integrator.finalize('train', it)
#                     self.train_integrator.reset_except_hooks()

#                 if it % self.save_network_interval == 0 and it != 0:
#                     if self.logger is not None:
#                         self.save_network(it)

#                 if it % self.save_checkpoint_interval == 0 and it != 0:
#                     if self.logger is not None:
#                         self.save_checkpoint(it)

#         # Backward pass
#         self.optimizer.zero_grad(set_to_none=True)
#         if self.config['amp']:
#             self.scaler.scale(losses['total_loss']).backward()
#             self.scaler.step(self.optimizer)
#             self.scaler.update()
#         else:
#             losses['total_loss'].backward() 
#             self.optimizer.step()

#         self.scheduler.step()

    def do_pass(self, data, it=0):
        torch.set_grad_enabled(self._is_train)

        # 转cuda前后数据形状
        for k, v in data.items():
            if type(v) != list and type(v) != dict and type(v) != int:
                # print(f"[调试] 传入数据 {k}，转cuda前形状: {v.shape}")
                data[k] = v.cuda(non_blocking=True)
                # print(f"[调试] {k} 转cuda后形状: {data[k].shape}")
                
        out = {}
        frames = data['rgb']  # 输入帧
        first_frame_gt = data['first_frame_gt'].float()
        b = frames.shape[0]
        num_filled_objects = [o.item() for o in data['info']['num_objects']]
        num_objects = first_frame_gt.shape[2]
        # selector = data['selector'].unsqueeze(2).unsqueeze(2)
        device = frames.device

        # print(f"[调试] frames形状: {frames.shape}  batch={b}, 通道={frames.shape[1]}, 帧数={frames.shape[2]}, H={frames.shape[-2]}, W={frames.shape[-1]}")
        # print(f"[调试] first_frame_gt形状: {first_frame_gt.shape}")
        # print(f"[调试] num_objects: {num_objects}")
        # print(f"[调试] device: {device}")

        if 'selector' in data:
            # print(f"[调试] selector转cuda前形状: {data['selector'].shape}")
            selector = data['selector'].unsqueeze(2).unsqueeze(2)
            # print(f"[调试] selector转cuda后unsqueeze形状: {selector.shape}")
        else:
            print("[调试] selector key不存在")

        with torch.cuda.amp.autocast(enabled=False):
            # print(f"[调试] 调用 encode_key 前 frames形状: {frames.shape}")
            key, shrinkage, selection, f16, f8, f4 = self.XMem('encode_key', frames)
            # filler_one = torch.zeros(1, dtype=torch.int64)

            # print(f"[调试] encode_key 输出 key形状: {key.shape}")
            # if shrinkage is not None:
            #     print(f"[调试] encode_key 输出 shrinkage形状: {shrinkage.shape}")
            # if selection is not None:
            #     print(f"[调试] encode_key 输出 selection形状: {selection.shape}")
            # print(f"[调试] encode_key 输出 f16形状: {f16.shape}")
            # print(f"[调试] encode_key 输出 f8形状: {f8.shape}")
            # print(f"[调试] encode_key 输出 f4形状: {f4.shape}")

            hidden = torch.zeros((b, num_objects, self.config['hidden_dim'], *key.shape[-2:]))
            # print(f"[调试] hidden 初始化形状: {hidden.shape}")

        if hidden is not None:
            if isinstance(hidden, torch.Tensor):
                hidden = hidden.to(device)
                # print(f"[调试] hidden 移动到设备: {hidden.device}，形状: {hidden.shape}")
            elif isinstance(hidden, (tuple, list)) and all(isinstance(h, torch.Tensor) for h in hidden):
                hidden = tuple(h.to(device) for h in hidden)
                # print(f"[调试] hidden tuple 移动到设备")
                for idx, h in enumerate(hidden):
                    print(f"hidden[{idx}] 设备: {h.device}, 形状: {h.shape}")
            else:
                print("[错误] hidden 类型异常，无法移动到设备")

            # print(f"[调试] frames第0帧设备: {frames[:,0].device}")
            # print(f"[调试] f16第0帧设备: {f16[:,0].device}")
            # print(f"[调试] first_frame_gt第0帧设备: {first_frame_gt[:,0].device}")

            v16, hidden = self.XMem('encode_value', frames[:,0], f16[:,0], hidden, first_frame_gt[:,0])
            # print(f"[调试] encode_value 输出 v16形状 (第0帧): {v16.shape}")

            values = v16.unsqueeze(3)  # 添加时间维度
            # print(f"[调试] values 初始形状 (含时间维): {values.shape}")

#             for ti in range(1, self.num_frames):
#                 print(f"[调试] 处理第 {ti} 帧")

#                 if ti <= self.num_ref_frames:
#                     ref_values = values
#                     ref_keys = key[:,:,:ti]
#                     ref_shrinkage = shrinkage[:,:,:ti] if shrinkage is not None else None
#                 else:
#                     filler_one = torch.zeros(1, dtype=torch.int64)
#                     indices = [torch.cat([filler_one, torch.randperm(ti-1)[:self.num_ref_frames-1]+1]) for _ in range(b)]
#                     ref_values = torch.stack([values[bi, :, :, indices[bi]] for bi in range(b)], 0)
#                     ref_keys = torch.stack([key[bi, :, indices[bi]] for bi in range(b)], 0)
#                     ref_shrinkage = torch.stack([shrinkage[bi, :, indices[bi]] for bi in range(b)], 0) if shrinkage is not None else None

#                 print(f"[调试] ref_keys形状: {ref_keys.shape}")
#                 if ref_shrinkage is not None:
#                     print(f"[调试] ref_shrinkage形状: {ref_shrinkage.shape}")
#                 print(f"[调试] ref_values形状: {ref_values.shape}")

#                 memory_readout = self.XMem('read_memory', key[:,:,ti], selection[:,:,ti] if selection is not None else None, ref_keys, ref_shrinkage, ref_values)
#                 print(f"[调试] memory_readout 形状: {memory_readout.shape}")

#                 hidden, logits, masks = self.XMem('segment', (f16[:,ti], f8[:,ti], f4[:,ti]), memory_readout, hidden, selector, h_out=(ti < (self.num_frames-1)))
#                 print(f"[调试] segment 输出 logits形状: {logits.shape}, masks形状: {masks.shape}, hidden形状: {hidden.shape}")

#                 if ti < (self.num_frames-1):
#                     is_deep_update = np.random.rand() < self.deep_update_prob
#                     v16, hidden = self.XMem('encode_value', frames[:,ti], f16[:,ti], hidden, masks, is_deep_update=is_deep_update)
#                     print(f"[调试] encode_value 输出 v16形状 (第{ti}帧): {v16.shape}")

#                     # 重点检查维度是否匹配
#                     if v16.shape[1] != values.shape[1]:
#                         print(f"[警告] v16 对象维度 {v16.shape[1]} 与 values 对象维度 {values.shape[1]} 不匹配！")

#                     values = torch.cat([values, v16.unsqueeze(3)], 3)
#                     print(f"[调试] values 更新后形状: {values.shape}")
            for ti in range(1, self.num_frames):
                # print(f"[调试] 处理第 {ti} 帧")

                if ti <= self.num_ref_frames:
                    ref_values = values
                    ref_keys = key[:, :, :ti]
                    ref_shrinkage = shrinkage[:, :, :ti] if shrinkage is not None else None
                else:
                    filler_one = torch.zeros(1, dtype=torch.int64, device=frames.device)
                    indices = [torch.cat([filler_one, torch.randperm(ti - 1, device=frames.device)[:self.num_ref_frames - 1] + 1]) for _ in range(b)]
                    ref_values = torch.stack([values[bi, :, :, indices[bi]] for bi in range(b)], 0)
                    ref_keys = torch.stack([key[bi, :, indices[bi]] for bi in range(b)], 0)
                    ref_shrinkage = torch.stack([shrinkage[bi, :, indices[bi]] for bi in range(b)], 0) if shrinkage is not None else None

                # print(f"[调试] ref_keys形状: {ref_keys.shape}")
                # if ref_shrinkage is not None:
                #     print(f"[调试] ref_shrinkage形状: {ref_shrinkage.shape}")
                # print(f"[调试] ref_values形状: {ref_values.shape}")

                memory_readout = self.XMem('read_memory',
                                          key[:, :, ti],
                                          selection[:, :, ti] if selection is not None else None,
                                          ref_keys, ref_shrinkage, ref_values)
                # print(f"[调试] memory_readout 形状: {memory_readout.shape}")

                hidden, logits, masks = self.XMem('segment',
                                                 (f16[:, ti], f8[:, ti], f4[:, ti]),
                                                 memory_readout, hidden, selector,
                                                 h_out=(ti < (self.num_frames - 1)))
                # print(f"[调试] segment 输出 logits形状: {logits.shape}, masks形状: {masks.shape}, hidden形状: {hidden.shape}")

                if ti < (self.num_frames - 1):
                    is_deep_update = np.random.rand() < self.deep_update_prob

                    # 关键：确保 frames[:, ti] 有物体维度，即形状变成 [b, num_objects, c, h, w]
                    # 如果只有1个物体，增加维度即可
                    # frame_ti = frames[:, ti]
                    # if frame_ti.dim() == 4:  # 形状 [b, c, h, w]
                    #     frame_ti = frame_ti.unsqueeze(1)  # 增加物体维度，变成 [b, 1, c, h, w]
                    #     print(f"[调试] frames[:, {ti}] 增加物体维度后形状: {frame_ti.shape}")
                    # else:
                    #     print(f"[调试] frames[:, {ti}] 已有物体维度: {frame_ti.shape}")

                    v16, hidden = self.XMem('encode_value', frames[:,ti], f16[:, ti], hidden, masks, is_deep_update=is_deep_update)
                    # print(f"[调试] encode_value 输出 v16形状 (第{ti}帧): {v16.shape}")

                    # 重点检查维度是否匹配
                    # if v16.shape[1] != values.shape[1]:
                    #     print(f"[警告] v16 对象第二维度 {v16.shape[1]} 与 values 对象第二维度 {values.shape[1]} 不匹配！")

                    values = torch.cat([values, v16.unsqueeze(3)], 3)
                    # print(f"[调试] values 更新后形状: {values.shape}")

                out[f'masks_{ti}'] = masks
                out[f'logits_{ti}'] = logits

            if self._do_log or self._is_train:
                losses = self.loss_computer.compute({**data, **out}, num_filled_objects, it)
                # print(f"[Debug]:losses={losses}")

                if self._do_log:
                    self.integrator.add_dict(losses)
                    if self._is_train and it % self.log_image_interval == 0 and it != 0:
                        if self.logger is not None:
                            images = {**data, **out}
                            size = (384, 384)
                            self.logger.log_cv2('train/pairs', pool_pairs(images, size, num_filled_objects), it)

            if self._is_train:
                if (it) % self.log_text_interval == 0 and it != 0:
                    if self.logger is not None:
                        self.logger.log_scalar('train/lr', self.scheduler.get_last_lr()[0], it)
                        self.logger.log_metrics('train', 'time', (time.time()-self.last_time)/self.log_text_interval, it)
                    self.last_time = time.time()
                    self.train_integrator.finalize('train', it)
                    self.train_integrator.reset_except_hooks()

                if it % self.save_network_interval == 0 and it != 0:
                    if self.logger is not None:
                        self.save_network(it)

                if it % self.save_checkpoint_interval == 0 and it != 0:
                    if self.logger is not None:
                        self.save_checkpoint(it)

        # 反向传播
        self.optimizer.zero_grad(set_to_none=True)
        # print(f"[调试] losses: {losses}")
        # print(f"[调试] losses['total_loss'] 类型: {type(losses['total_loss'])}")

        if self.config['amp']:
            self.scaler.scale(losses['total_loss']).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            losses['total_loss'].backward()
            self.optimizer.step()

        self.scheduler.step()



#     def do_pass(self, data, it=0):
#         # No need to store the gradient outside training
#         torch.set_grad_enabled(self._is_train)

#         for k, v in data.items():
#             if type(v) != list and type(v) != dict and type(v) != int:
#                 data[k] = v.cuda(non_blocking=True)

#         out = {}
#         frames = data['rgb']
#         first_frame_gt = data['first_frame_gt'].float()
#         b = frames.shape[0]
#         num_filled_objects = [o.item() for o in data['info']['num_objects']]
#         num_objects = first_frame_gt.shape[2]
#         selector = data['selector'].unsqueeze(2).unsqueeze(2)
        
#         device = frames.device



#         with torch.cuda.amp.autocast(enabled=self.config['amp']):
#             # image features never change, compute once
#             key, shrinkage, selection, f16, f8, f4 = self.XMem('encode_key', frames)

#             filler_one = torch.zeros(1, dtype=torch.int64)
#             hidden = torch.zeros((b, num_objects, self.config['hidden_dim'], *key.shape[-2:]))
#             if hidden is not None:
#                 print(f"[DEBUG] hidden before encode_value: {[h_i.device for h_i in hidden]}")
#             hidden = tuple([h_i.to(device) for h_i in hidden])
#             v16, hidden = self.XMem('encode_value', frames[:,0], f16[:,0], hidden, first_frame_gt[:,0])
#             values = v16.unsqueeze(3) # add the time dimension

#             for ti in range(1, self.num_frames):
#                 if ti <= self.num_ref_frames:
#                     ref_values = values
#                     ref_keys = key[:,:,:ti]
#                     ref_shrinkage = shrinkage[:,:,:ti] if shrinkage is not None else None
#                 else:
#                     # pick num_ref_frames random frames
#                     # this is not very efficient but I think we would 
#                     # need broadcasting in gather which we don't have
#                     indices = [
#                         torch.cat([filler_one, torch.randperm(ti-1)[:self.num_ref_frames-1]+1])
#                     for _ in range(b)]
#                     ref_values = torch.stack([
#                         values[bi, :, :, indices[bi]] for bi in range(b)
#                     ], 0)
#                     ref_keys = torch.stack([
#                         key[bi, :, indices[bi]] for bi in range(b)
#                     ], 0)
#                     ref_shrinkage = torch.stack([
#                         shrinkage[bi, :, indices[bi]] for bi in range(b)
#                     ], 0) if shrinkage is not None else None

#                 # Segment frame ti
#                 memory_readout = self.XMem('read_memory', key[:,:,ti], selection[:,:,ti] if selection is not None else None, 
#                                         ref_keys, ref_shrinkage, ref_values)
#                 hidden, logits, masks = self.XMem('segment', (f16[:,ti], f8[:,ti], f4[:,ti]), memory_readout, 
#                         hidden, selector, h_out=(ti < (self.num_frames-1)))

#                 # No need to encode the last frame
#                 if ti < (self.num_frames-1):
#                     is_deep_update = np.random.rand() < self.deep_update_prob
#                     v16, hidden = self.XMem('encode_value', frames[:,ti], f16[:,ti], hidden, masks, is_deep_update=is_deep_update)
#                     values = torch.cat([values, v16.unsqueeze(3)], 3)

#                 out[f'masks_{ti}'] = masks
#                 out[f'logits_{ti}'] = logits

#             if self._do_log or self._is_train:
#                 losses = self.loss_computer.compute({**data, **out}, num_filled_objects, it)

#                 # Logging
#                 if self._do_log:
#                     self.integrator.add_dict(losses)
#                     if self._is_train:
#                         if it % self.log_image_interval == 0 and it != 0:
#                             if self.logger is not None:
#                                 images = {**data, **out}
#                                 size = (384, 384)
#                                 self.logger.log_cv2('train/pairs', pool_pairs(images, size, num_filled_objects), it)

#             if self._is_train:
#                 if (it) % self.log_text_interval == 0 and it != 0:
#                     if self.logger is not None:
#                         self.logger.log_scalar('train/lr', self.scheduler.get_last_lr()[0], it)
#                         self.logger.log_metrics('train', 'time', (time.time()-self.last_time)/self.log_text_interval, it)
#                     self.last_time = time.time()
#                     self.train_integrator.finalize('train', it)
#                     self.train_integrator.reset_except_hooks()

#                 if it % self.save_network_interval == 0 and it != 0:
#                     if self.logger is not None:
#                         self.save_network(it)

#                 if it % self.save_checkpoint_interval == 0 and it != 0:
#                     if self.logger is not None:
#                         self.save_checkpoint(it)

#         # Backward pass
#         self.optimizer.zero_grad(set_to_none=True)
#         if self.config['amp']:
#             self.scaler.scale(losses['total_loss']).backward()
#             self.scaler.step(self.optimizer)
#             self.scaler.update()
#         else:
#             losses['total_loss'].backward() 
#             self.optimizer.step()

#         self.scheduler.step()
    def save_network(self, it):
        if self.save_path is None:
            print('Saving has been disabled.')
            return
        print(self.save_path)
        os.makedirs(os.path.dirname(self.save_path), exist_ok=True)
        model_path = f'{self.save_path}_{it}.pth'
        torch.save(self.XMem.state_dict(), model_path)
        print(f'Network saved to {model_path}.')

    def save_checkpoint(self, it):
        if self.save_path is None:
            print('Saving has been disabled.')
            return

        os.makedirs(os.path.dirname(self.save_path), exist_ok=True)
        checkpoint_path = f'{self.save_path}_checkpoint_{it}.pth'
        checkpoint = { 
            'it': it,
            'network': self.XMem.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'scheduler': self.scheduler.state_dict()}
        torch.save(checkpoint, checkpoint_path)
        print(f'Checkpoint saved to {checkpoint_path}.')

    def load_checkpoint(self, path):
        # This method loads everything and should be used to resume training
        map_location = 'cuda:%d' % self.local_rank
        checkpoint = torch.load(path, map_location={'cuda:0': map_location})

        it = checkpoint['it']
        network = checkpoint['network']
        optimizer = checkpoint['optimizer']
        scheduler = checkpoint['scheduler']

        map_location = 'cuda:%d' % self.local_rank
        self.XMem.load_state_dict(network)
        self.optimizer.load_state_dict(optimizer)
        self.scheduler.load_state_dict(scheduler)

        print('Network weights, optimizer states, and scheduler states loaded.')

        return it

    def load_network_in_memory(self, src_dict):
        self.XMem.load_weights(src_dict)
        print('Network weight loaded from memory.')

    def load_network(self, path):
        # This method loads only the network weight and should be used to load a pretrained model
        map_location = 'cuda:%d' % self.local_rank
        src_dict = torch.load(path, map_location={'cuda:0': map_location})

        self.load_network_in_memory(src_dict)
        print(f'Network weight loaded from {path}')

    def train(self):
        self._is_train = True
        self._do_log = True
        self.integrator = self.train_integrator
        self.XMem.eval()
        return self

    def val(self):
        self._is_train = False
        self._do_log = True
        self.XMem.eval()
        return self

    def test(self):
        self._is_train = False
        self._do_log = False
        self.XMem.eval()
        return self

