from torch.utils.data import DataLoader
import torch
import numpy as np
import model
import datasets
from test import test
import option
import time
import os

os.environ['CUDA_VISIBLE_DEVICES'] = '1'

if __name__ == '__main__':
    print('perform testing...')
    args = option.parser.parse_args()
    args.device = 'cuda:' + str(args.cuda) if torch.cuda.is_available() else 'cpu'

    test_loader = DataLoader(datasets.AllDataset(args=args, test_mode=True),
                             batch_size=1, shuffle=False,
                             num_workers=args.workers, pin_memory=True)
    model = model.TemporalSpatialModel(args)
    model = model.to(args.device)
    # model_dict = model.load_state_dict(
    #     {k.replace('module.', ''): v for k, v in torch.load('ckpt/111.pkl', map_location=torch.device('cpu')).items()})
    model_dict = model.load_state_dict(
        {k.replace('module.', ''): v for k, v in
         torch.load('ckpt/best_model_ucf.pkl', map_location=torch.device('cpu')).items()})
    gt = np.load(args.gt)
    st = time.time()

    pr_auc, pr_ap, _ = test(test_loader, model, gt, args, False)
    time_elapsed = time.time() - st
    print('AUC: {:.4f}   AP:  {:.4f} \n'.format(pr_auc, pr_ap))
    print('Test complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
