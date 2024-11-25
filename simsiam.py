import json
import os
import math
import shutil
import random
import time

import torch
import torch.nn.functional as F

from dataclasses import dataclass
from PIL import ImageFilter

from torch import nn
from torch import distributed as dist
from torchvision import models, transforms, datasets

##############################
#      Hyperparameters       #
##############################

@dataclass
class args:

    # training
    train_dir: str  = "/datasets/train/"
    save_dir: str   = "./out/"
    log_freq: int   = 120
    epochs: int     = 100
    batch_size: int = 256                       # will be divided by world_size
    resume: bool    = True
    seed: int       = None                      # optional

    # eval
    subset_dir : str= "/datasets/1percent.txt" # 10percent.txt
    eval_dir : str  = "/datasets/val/"
    knn_n : int     = 200
    knn_t: float    = 0.2

    # model
    model: str      = "resnet50"
    proj_arch: str  = "2048-2048"
    pred_arch: str  = "512"

    # optim
    base_lr: float  = 0.05
    momentum: float = 0.9
    wd: float       = 1e-4
    fix_pred_lr: bool = True
    
    # hardware
    workers: int    = 16
    device: str     = "cuda" # cpu, mps
    distributed: bool = True

#############################################
#                Distributed                #
#############################################

def is_distributed(): return False if not (dist.is_available and dist.is_initialized()) else True
def get_rank(): return dist.get_rank() if is_distributed () else 0
def get_world_size(): return dist.get_world_size() if is_distributed else 0
def init_distributed():
    ddp = int(os.environ.get('RANK', -1)) != -1

    if not (ddp and args.distributed):
        args.rank, args.world_size, args.is_master, args.gpu = 0, 1, 0, True, None
        args.device = torch.device(args.device)
        print("Not using distributed mode!")
    else:
        args.rank = int(os.environ['RANK'])
        args.world_size = int(os.environ['WORLD_SIZE'])
        args.gpu = int(os.environ['LOCAL_RANK'])
        args.is_master = args.rank == 0
        args.device = torch.device(f"cuda:{args.gpu}")
        torch.cuda.set_device(args.device)
        print(f'| distributed init (rank {args.rank}), gpu {args.gpu}', flush=True)
        dist.init_process_group(backend="nccl", world_size=args.world_size, rank=args.rank, device_id=args.device)

        # update config
        args.per_device_batch_size = int(args.batch_size / torch.cuda.device_count())
        args.workers = int((args.workers + args.world_size - 1) / args.world_size)

        # fix printing
        def fix_print(is_master):
            import builtins as __builtin__
            builtin_print = __builtin__.print
            
            def print(*args, **kwargs):
                force = kwargs.pop('force', False)
                if is_master or force:
                    builtin_print(*args, **kwargs)
            __builtin__.print = print

        dist.barrier()
        fix_print(args.is_master)

    return is_distributed()

#############################################
#                Dataloader                 #
#############################################

class gaussian_blur:

    def __init__ (self, sigma):
        self.sigma = sigma
    def __call__(self, x):
        return x.filter(ImageFilter.GaussianBlur(radius=random.uniform(*self.sigma)))

class augment:
    """MoCo v2's aug: similar to SimCLR https://arxiv.org/abs/2002.05709"""
    def __init__(self):
        self.transform = transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.2, 1.)),
        transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
        transforms.RandomGrayscale(p=0.2),
        transforms.RandomApply([gaussian_blur([.1, 2.])], p=0.5),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def __call__(self, x):
        x1, x2 = self.transform(x), self.transform(x)
        return x1, x2

#############################################
#              Model Components             #
#############################################

def get_resnet():
    backbone = getattr(models, args.model)(weights=None, zero_init_residual=True)
    h_dim = backbone.fc.in_features
    backbone.fc = nn.Identity()
    return backbone, h_dim

def mlp_mapper(h_dim, arch, bn_end=False):
    arch = f"{h_dim}-{arch}-{h_dim}"
    f = list(map(int, arch.split('-')))
    layers = []
    for i in range(len(f) - 2):
        layers.append(nn.Linear(f[i], f[i + 1], bias=False))
        layers.append(nn.BatchNorm1d(f[i + 1]))
        layers.append(nn.ReLU(True))
    layers.append(nn.Linear(f[-2], f[-1]))
    if bn_end:
        layers[-1].bias.requires_grad = False
        layers.append(nn.BatchNorm1d(f[-1], affine=False))

    return nn.Sequential(*layers)

#############################################
#              Model Definition             #
#############################################

class SimSiam(nn.Module):

    def __init__(self, backbone, h_dim):
        super(SimSiam, self).__init__()
        self.backbone = backbone
        self.backbone.fc = mlp_mapper(h_dim, args.proj_arch, bn_end=True)
        self.predictor = mlp_mapper(h_dim, args.pred_arch)
        return
    
    def forward(self, x1, x2):
        z1, z2 = self.backbone(x1), self.backbone(x2)
        p1, p2 = self.predictor(z1), self.predictor(z2)
        return p1, p2, z1.detach(), z2.detach()

def make_net():
    backbone, h_dim = get_resnet()
    model = SimSiam(backbone, h_dim)
    model.to(args.device)

    if is_distributed():
        model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
        model = nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])

    return model

##############################
#        Scheduling          #
##############################

def adjust_learning_rate(optimizer, epoch):
    """Decay the learning rate based on schedule"""
    init_lr = args.base_lr * args.batch_size / 256
    cur_lr = init_lr * 0.5 * (1. + math.cos(math.pi * epoch / args.epochs))
    for param_group in optimizer.param_groups:
        if 'fix_lr' in param_group and param_group['fix_lr']:
            param_group['lr'] = init_lr
        else:
            param_group['lr'] = cur_lr
    
    return cur_lr
    
###########################################
#          Logging & Checkpoints          #
###########################################

class AverageMeter:
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name, self.fmt = name, fmt
        self.reset()

    def reset(self):
        self.val, self.avg, self.sum, self.count = 0, 0, 0, 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
    
    def __str__(self): return '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'.format(**self.__dict__)


def resume(net, optimizer):
    ckpt_path = os.path.join(args.save_dir, "checkpoint.pt")
    if not os.path.exists(ckpt_path):
        print("No checkpoint found!")
        return 0, float("inf")

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    net.load_state_dict(ckpt["model"])
    optimizer.load_state_dict(ckpt["optimizer"])
    epoch, best_loss = ckpt["epoch"]+1, ckpt["best_loss"]
    print(f"Checkpoint found! Resuming from epoch {epoch}")
    return epoch, best_loss

def save_checkpoint(state, is_best, filename="checkpoint.pt"):
    path = os.path.join(args.save_dir, filename)
    torch.save(state, path)
    if is_best: shutil.copyfile(path, os.path.join(args.save_dir,"best_checkpoint.pt"))

    return

########################################
#           Train and Eval             #
########################################

class ImageFolderSubset(datasets.ImageFolder):
    def __init__(self, subset_txt, **kwargs):
        super().__init__(**kwargs)
        if subset_txt is not None:
            with open(subset_txt, 'r') as fid:
                subset = set([line.strip() for line in fid.readlines()])

            subset_samples = []
            for sample in self.samples:
                if os.path.basename(sample[0]) in subset:
                    subset_samples.append(sample)

            self.samples = subset_samples
            self.targets = [s[1] for s in subset_samples]

@torch.no_grad()
def eval(net):

    def _load_data(type):
        assert type in ["val", "sub_train"]

        trans=transforms.Compose([
                        transforms.Resize([256, 256]),
                        transforms.CenterCrop(224),
                        transforms.ToTensor(),
                        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                            std=[0.229, 0.224, 0.225])])

        if type == "val":
            dataset = datasets.ImageFolder(args.eval_dir, transform=trans)
        else:
            dataset = ImageFolderSubset(args.subset_dir, root=args.train_dir, transform=trans)
        
        sampler = torch.utils.data.distributed.DistributedSampler(dataset) if is_distributed() else None
        loader = torch.utils.data.DataLoader(dataset, batch_size=args.per_device_batch_size, 
                                        num_workers=args.workers, pin_memory=True, 
                                        drop_last=True, sampler=sampler)
        return loader

    def predict(features, X_train, Y_train, classes, n=200, t=0.2):
        scores = features @ X_train.t()
        weights, idx = scores.topk(k=n, dim=-1)
        labels = torch.gather(Y_train.expand(features.size(0), -1), dim=-1, index=idx)
        weights = (weights / t).exp()
        oh_labels = torch.zeros(features.size(0) * n, classes, device=args.device)
        oh_labels = oh_labels.scatter(dim=-1, index=labels.view(-1, 1), value=1.0)
        preds = torch.sum(oh_labels.view(features.size(0), -1, classes) * weights.unsqueeze(dim=-1), dim=1)
        return preds.argsort(dim=-1, descending=True)

    correct, total = 0, 0
    features_bank, labels_bank = [], []
    
    net.eval()
    backbone = net.module.backbone if hasattr(net, 'module') else net.backbone
    train_loader = _load_data("sub_train")

    # Extract features from training set
    with torch.no_grad():
        for (images, labels) in train_loader:
            images = images.to(args.device, non_blocking=True)
            labels = labels.to(args.device, non_blocking=True)
            features_bank.append(F.normalize(backbone(images), dim=-1))    
            labels_bank.append(labels)
    
    features_bank = torch.cat(features_bank, dim=0)
    labels_bank = torch.cat(labels_bank, dim=0)

    # Gather features from all GPUs
    if is_distributed():
        features_bank_list = [torch.zeros_like(features_bank) for _ in range(dist.get_world_size())]
        labels_bank_list = [torch.zeros_like(labels_bank) for _ in range(dist.get_world_size())]
        
        dist.all_gather(features_bank_list, features_bank)
        dist.all_gather(labels_bank_list, labels_bank)
        
        features_bank = torch.cat(features_bank_list, dim=0)
        labels_bank = torch.cat(labels_bank_list, dim=0)

    test_loader = _load_data("val")
    classes = len(test_loader.dataset.classes)

    # Evaluate
    with torch.no_grad():
        for (images, labels) in test_loader:
            images = images.to(args.device, non_blocking=True)
            labels = labels.to(args.device, non_blocking=True)
            features = F.normalize(backbone(images), dim=-1)
            preds = predict(features, features_bank, labels_bank, classes, args.knn_n, args.knn_t)
            total += labels.size(0)
            correct += (preds[:, 0] == labels).sum().item()

    if args.distributed:
        correct = torch.tensor(correct).to(args.device)
        total = torch.tensor(total).to(args.device)
        dist.all_reduce(correct)
        dist.all_reduce(total)
        correct = correct.item()
        total = total.item()

    net.train()

    return (correct / total) * 100

def main():

    init_distributed()

    #############
    ##   INIT  ##
    #############
    device = args.device
    start_epoch = 0
    best_loss = float("inf")
    loss_fn = nn.CosineSimilarity(dim=-1).to(device)
    net = make_net()
    print(net)
    print(f"number of params: {sum(p.numel() for p in net.parameters() if p.requires_grad)}")

    if args.is_master:
        os.makedirs(args.save_dir, exist_ok=True)
        log_file = open(os.path.join(args.save_dir,"logs.txt"), "a", buffering=1)
        print(json.dumps(vars(args())), file=log_file)

    if args.fix_pred_lr:
        model = net.module if is_distributed() else net
        optims_params = [{'params': model.backbone.parameters(), 'fix_lr': False},
                        {'params': model.predictor.parameters(), 'fix_lr': True}]
    else:
        optims_params = net.parameters()

    optimizer = torch.optim.SGD(optims_params, lr=args.base_lr, momentum=args.momentum, weight_decay=args.wd)    
    dataset = datasets.ImageFolder(args.train_dir, augment())
    sampler = torch.utils.data.distributed.DistributedSampler(dataset) if is_distributed() else None
    scaler = torch.amp.GradScaler()
    loader = torch.utils.data.DataLoader(dataset, batch_size=args.per_device_batch_size, shuffle=(sampler is None),
                num_workers=args.workers, pin_memory=True, sampler=sampler, drop_last=True)

    if args.resume:
        start_epoch, best_loss = resume(net, optimizer)
    
    last_logging = time.time()
    for epoch in range(start_epoch, args.epochs):
        losses = AverageMeter("Loss", ":.4f")
        if is_distributed(): sampler.set_epoch(epoch)
        
        net.train()
        lr = adjust_learning_rate(optimizer, epoch)
        for step, ((x1, x2), _) in enumerate(loader, start=epoch*len(loader)):

            x1 = x1.to(device, non_blocking=True)
            x2 = x2.to(device, non_blocking=True)
            
            optimizer.zero_grad() 
            with torch.autocast(device_type=device.type):
                p1, p2, z1, z2 = net(x1, x2)
                loss = - (loss_fn(p1, z2).mean() + loss_fn(p2, z1).mean()) * 0.5
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            losses.update(loss.item(), x1.size(0))
            current_time = time.time()
            if args.is_master and (current_time - last_logging > args.log_freq):
                last_logging = current_time
                progress = 100 * step / (len(loader) * args.epochs)
                print(f"loss {losses.avg:.4f} ({progress:.2f}%)")

        knn_acc = eval(net)
        if args.is_master:

            is_best = losses.avg < best_loss
            if is_best: best_loss = losses.avg

            # checkpoint
            state = dict(epoch=epoch, model=net.state_dict(), optimizer=optimizer.state_dict(), best_loss=best_loss)
            save_checkpoint(state, is_best)
            logs = dict(epoch=epoch, loss=losses.avg, best_loss=best_loss, lr=lr, knn_acc=knn_acc)
            print(json.dumps(logs), file=log_file)
            print(f"Epoch {epoch} => KNN accuracy ({knn_acc:.4f} %)")
            
    if is_distributed():
        dist.destroy_process_group()
    return

if __name__ == "__main__":
    main()