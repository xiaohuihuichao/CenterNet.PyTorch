import os
import time

import numpy as np

import torch
import torch.optim as optim
import torch.distributed as dist
from torch.utils.data import DataLoader

from net import Net
from utils import loss
from dataset import dataset

from tensorboardX import SummaryWriter


def reduce_tensor(tensor: torch.Tensor):
    rt = tensor.clone()
    dist.all_reduce(rt)
    return rt / dist.get_world_size()


def save_model(net, path):
    state = {}
    for k, v in net.state_dict().items():
        state[k.replace("module.", "")] = v.cpu()
    torch.save(state, path)

def log(msg, file, args, append=True):
    if args.local_rank == 0:
        if append:
            t = ">>"
        else:
            t = ">"
        os.system(f"echo '{msg}' {t} '{file}'")
        
        
def train_batch(net, batch_data, criterion, optimizer, cuda):
    optimizer.zero_grad()
    if cuda:
        batch_data = [i.cuda() for i in batch_data]
    imgs, hm_gts, txtys, twths, xywh_masks = batch_data
    
    hm_preds, xy_preds, wh_preds = net(imgs)
    total_loss, hm_loss, txty_loss, wh_loss = criterion(hm_preds, xy_preds, wh_preds,
                                                        hm_gts, txtys, twths,
                                                        xywh_masks)
    total_loss.backward()
    # torch.nn.utils.clip_grad_norm_(net.parameters())
    optimizer.step()
    return total_loss, hm_loss, txty_loss, wh_loss


def main(args):
    dist.init_process_group(backend="nccl")
    torch.cuda.set_device(args.local_rank)
    
    batch_save_path = f"{args.model_dir}/batch.pth"
    
    if args.local_rank == 0:
        os.system(f"rm -rf {args.tensorboard_log}")
        writer = SummaryWriter(args.tensorboard_log)
        
    ds = dataset(args.data_file_path, args.num_classes, args.wh, args.stride)
    sampler = torch.utils.data.distributed.DistributedSampler(ds, shuffle=True)
    dl = DataLoader(ds, batch_size=args.batch_size, num_workers=args.num_workers, pin_memory=True, drop_last=False, sampler=sampler)
    log(f"共有训练数据: {len(ds)}", args.log_detail_path, args)
    log(f"共有训练数据: {len(ds)}", args.log_path, args)
    
    net = Net(args.fpn_out_channel, num_classes=args.num_classes)
    if os.path.isfile(batch_save_path):
        log("载入模型中...", args.log_detail_path, args)
        log("载入模型中...", args.log_path, args)
        try:
            net.load_state_dict(torch.load(batch_save_path))
            log("模型载入完成！", args.log_detail_path, args)
            log("模型载入完成！", args.log_path, args)
        except Exception as e:
            log(f"{e}\n载入模型失败: {batch_save_path}", args.log_detail_path, args)
            log(f"{e}\n载入模型失败: {batch_save_path}", args.log_path, args)
    else:
        log(f"没找到模型: {batch_save_path}", args.log_detail_path, args)
        log(f"没找到模型: {batch_save_path}", args.log_path, args)
        
    args.cuda = args.cuda and torch.cuda.is_available()
    if args.cuda:
        net = torch.nn.parallel.DistributedDataParallel(net.cuda(), device_ids=[args.local_rank])
        log("cuda", args.log_detail_path, args)
        
    criterion = loss(xy_weight=args.xy_weight, wh_weight=args.wh_weight, hm_weight=args.hm_weight)
    
    optimizer = optim.Adam(net.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=1, verbose=True, factor=args.lr_decay, threshold=1e-3)
    schedule_loss = []
    
    batch_idx = 0
    net.train()
    for epoch in range(1, args.epochs+1):
        log(f"{'='*30}\n[{epoch}|{args.epochs}]", args.log_detail_path, args)
        log(f"{'='*30}\n[{epoch}|{args.epochs}]", args.log_path, args)
        
        for num_batch, batch_data in enumerate(dl, 1):
            batch_idx += 1
            
            t = time.time()
            total_loss, hm_loss, txty_loss, wh_loss = train_batch(net, batch_data, criterion, optimizer, args.cuda)
            t = time.time() - t
            
            total_loss, hm_loss, txty_loss, wh_loss = [reduce_tensor(i).item() for i in [total_loss, hm_loss, txty_loss, wh_loss]]
            
            msg = f"\t[{epoch}|{args.epochs}] num_batch:{num_batch}" \
                + f" total_loss:{total_loss:.4f} hm_loss:{hm_loss:.4f} txty_loss:{txty_loss:.4f} wh_loss:{wh_loss:.4f} time:{t*1000:.1f}ms"
            log(msg, args.log_detail_path, args)
            
            if num_batch % args.num_show == 0:
                log(msg, args.log_path, args)
                
            if args.local_rank == 0:
                writer.add_scalar("total_loss", total_loss, batch_idx)
                writer.add_scalar("hm_loss", hm_loss, batch_idx)
                writer.add_scalar("txty_loss", txty_loss, batch_idx)
                writer.add_scalar("wh_loss", wh_loss, batch_idx)
                
                if num_batch % args.num_save == 0:
                    save_model(net, batch_save_path)
                    
                schedule_loss += [total_loss]
                if num_batch % args.num_adjuest_lr == 0:
                    l = f"{'-'*10}\nschedule mean loss: {np.mean(schedule_loss):.4f}\n{'-'*10}"
                    log(l, args.log_path, args)
                    log(l, args.log_detail_path, args)
                    
                    scheduler.step(np.mean(schedule_loss))
                    schedule_loss = []
    save_model(net, f"{args.model_dir}/final-model.pth")
    