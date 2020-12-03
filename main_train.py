import time
import argparse

from train import log, main


def get_args():
    parser = argparse.ArgumentParser(description="训练参数")
    
    parser.add_argument("--local_rank", default=-1, type=int)
    
    parser.add_argument("--cuda", default=True, type=bool)
    parser.add_argument("--fpn_out_channel", default=512, type=int)
    
    parser.add_argument("--data_file_path", default="centernet_label.txt", type=str)
    parser.add_argument("--num_classes", default=1, type=int)
    parser.add_argument("--wh", default=(640, 640), type=tuple)
    parser.add_argument("--stride", default=4, type=int)
    
    parser.add_argument("--epochs", default=1000, type=int)
    parser.add_argument("--batch_size", default=8, type=int)
    parser.add_argument("--lr", default=1e-3, type=float)
    parser.add_argument("--lr_decay", default=0.7, type=float)
    parser.add_argument("--weight_decay", default=1e-6, type=float)
    parser.add_argument("--num_workers", default=16, type=int)
    
    parser.add_argument("--xy_weight", type=float, default=1)
    parser.add_argument("--wh_weight", type=float, default=0.1)
    parser.add_argument("--hm_weight", type=float, default=1)
    
    parser.add_argument("--num_adjuest_lr", default=1000, type=int)
    parser.add_argument("--num_show", default=100, type=int)
    parser.add_argument("--num_save", default=1000, type=int)
    parser.add_argument("--model_dir", default="model", type=str)
    
    parser.add_argument("--log_detail_path", default="log/detail_train.log", type=str)
    parser.add_argument("--log_path", default="log/train.log", type=str)
    parser.add_argument("--tensorboard_log", default="log/tensorboard_log", type=str)
    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()
    
    if args.local_rank == 0:
        start_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        log(f"训练开始于 {start_time}", args.log_detail_path, args)
        log(f"训练开始于 {start_time}", args.log_path, args)
    
    main(args)
    
    if args.local_rank == 0:
        start_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        log(f"训练结束于 {start_time}", args.log_detail_path, args)
        log(f"训练结束于 {start_time}", args.log_path, args)
