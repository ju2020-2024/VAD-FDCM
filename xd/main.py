import copy
import os
import random
import numpy as np
import torch
from matplotlib import pyplot as plt

import datasets
import option
import model
import time
from fvcore.nn import FlopCountAnalysis
from torch import optim
from torch.utils.data import DataLoader
from test import test
from train import train

import datetime
import os


def setup_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


if __name__ == '__main__':
    args = option.parser.parse_args()
    setup_seed(args.seed)
    args.device = 'cuda:' + str(args.cuda) if int(args.cuda) >= 0 else 'cpu'

    print("==> Preparing data...")
    train_data = datasets.AllDataset(args=args, test_mode=False)
    test_data = datasets.AllDataset(args=args, test_mode=True)
    gt = np.load(args.gt)

    print("==> Preparing loader...")
    train_loader = DataLoader(
        train_data,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=True,
        drop_last=True
    )
    test_loader = DataLoader(
        test_data,
        batch_size=5,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True,
    )

    print("==> Building model...")
    model = model.TemporalSpatialModel(args)
    if torch.cuda.is_available():
        model = model.to(args.device)
    test_tensor = (torch.rand(128, 200, 1152),)
    flops = FlopCountAnalysis(model, test_tensor)
    print(">>> training params: {:.3f}M".format(
        sum(p.numel() for p in model.parameters() if p.requires_grad) / 1000000.0))
    criterion = torch.nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=60, eta_min=0)
    if not os.path.exists('./ckpt'):
        os.makedirs('./ckpt')

    print('==> Start Random test...')
    random_auc, random_ap, _ = test(
        test_loader,
        model,
        gt,
        args
    )
    print('Random initialized AUC: {:.4f}   Random initialized AP:  {:.4f} \n'.format(random_auc, random_ap))
    best_model_wts = copy.deepcopy(model.state_dict())
    best_auc, best_ap = 0, 0

    print('==> Start Training...')
    st = time.time()

    for epoch in range(1, args.max_epoch + 1):
        total_loss = train(
            train_loader=train_loader,
            model=model,
            optimizer=optimizer,
            criterion=criterion,
            args=args,
        )
        scheduler.step()
        AUC, AP, _ = test(
            test_loader=test_loader,
            model=model,
            gt=gt,
            args=args
        )

        if AUC > best_auc or AP > best_ap:
            best_auc = max(best_auc, AUC)
            best_ap = max(best_ap, AP)
            best_model_wts = copy.deepcopy(model.state_dict())

        print(
            '[Epoch {0}/{1}]: total_loss: {2}  | pr_auc:{3:.4} | pr_ap:{4:.4}\n'.
            format(
                epoch,
                args.max_epoch,
                total_loss,
                AUC,
                AP
            )
        )

    print('==> Saving checkpoint...')
    model.load_state_dict(best_model_wts)
    torch.save(model.state_dict(), './ckpt/' + args.model_name + '.pkl')
    time_elapsed = time.time() - st
    print(
        'Training completes in {:.0f}m {:.0f}s | '
        'best test AP: {:.4f} | '
        'best test AUC:{:.4f}\n'.
        format(
            time_elapsed // 60,
            time_elapsed % 60,
            best_ap,
            best_auc
        )
    )
