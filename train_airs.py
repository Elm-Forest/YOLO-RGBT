import argparse

import numpy as np
import torch

from ultralytics import YOLO

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="AIRS RGBT Training")
    parser.add_argument('--model_cfg_path', type=str,
                        default=r'ultralytics/cfg/models/11-RGBT/yolo11n-RGBT-midfusion.yaml')
    parser.add_argument('--dataset_cfg_path', type=str, default=r'ultralytics/cfg/datasets/AIRS1.yaml')
    parser.add_argument('--project_name', type=str, default='airs1_run2')
    parser.add_argument('--pretrain_model', type=str, default='yolo11n.pt')
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--weight_decay', type=float, default=5e-4)
    parser.add_argument('--optimizer', type=str, default='AdamW')
    parser.add_argument('--img_size', type=int, default=640)
    parser.add_argument('--device', type=str, default='0')
    parser.add_argument('--resume', type=str, default=None)
    parser.add_argument('--num_worker', type=int, default=2)
    parser.add_argument('--save_period', type=int, default=5)
    parser.add_argument('--cache', type=int, default=0)
    parser.add_argument('--multi_scale', type=int, default=0)
    parser.add_argument('--cos_lr', type=int, default=0)
    parser.add_argument('--warmup_epochs', type=int, default=3)
    parser.add_argument('--wandb_key', type=str, default="")

    # normal aug
    parser.add_argument('--aug_degrees', type=float, default=0)  # 旋转变换
    parser.add_argument('--aug_fliplr', type=float, default=0.5)  # 左右翻转
    parser.add_argument('--aug_flipud', type=float, default=0)  # 垂直翻转
    parser.add_argument('--aug_scale', type=float, default=0.5)  # 尺度变换

    # mid aug
    parser.add_argument('--aug_perspective', type=float, default=0.0003)  # 透视变换
    parser.add_argument('--aug_shear', type=int, default=10)  # 剪切变换
    parser.add_argument('--aug_translate', type=float, default=0.1)  # 平移变换

    # advance aug
    parser.add_argument('--aug_mosaic', type=float, default=0.5)
    parser.add_argument('--close_mosaic', type=float, default=5)

    parser.add_argument('--aug_mixup', type=float, default=0)
    parser.add_argument('--aug_cutmix', type=float, default=0.3)
    args = parser.parse_args()
    seed = args.seed
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    if args.wandb_key != "":
        import wandb
        wandb.login(key=args.wandb_key)
        from ultralytics import settings
        settings.update({"wandb": True})
    model = YOLO(args.model_cfg_path)
    # model.info(True,True)
    model.load(args.pretrain_model)
    model.train(data=args.dataset_cfg_path,
                cache=False if args.cache == 0 else True,
                imgsz=args.img_size,
                epochs=args.epochs,
                batch=args.batch_size,
                multi_scale=False if args.multi_scale == 0 else True,
                cos_lr=False if args.cos_lr == 0 else True,
                warmup_epochs=args.warmup_epochs,
                workers=args.num_worker,
                device=args.device,
                optimizer=args.optimizer,
                lr0=args.lr,
                weight_decay=args.weight_decay,
                resume=args.resume,
                save_period=args.save_period,
                amp=True,
                deterministic=True,
                seed=args.seed,
                use_simotm="RGBT",
                channels=4,
                project=f'runs/{args.project_name}',
                name='AIRS1-RGBT-',
                # aug
                close_mosaic=args.close_mosaic,
                degrees=args.aug_degrees,
                flipud=args.aug_flipud,
                fliplr=args.aug_fliplr,
                shear=args.aug_shear,
                translate=args.aug_translate,
                perspective=args.aug_perspective,
                mosaic=args.aug_mosaic,
                # cutmix=args.aug_cutmix,
                mixup=args.aug_mixup,
                scale=args.aug_scale,
                )
