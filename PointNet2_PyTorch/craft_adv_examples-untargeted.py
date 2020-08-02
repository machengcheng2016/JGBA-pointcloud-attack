import warnings
warnings.filterwarnings("ignore")

import numpy as np
import adversarial_untargeted

import torch
import torch.nn as nn
import utils.pytorch_utils as pytorch_utils
from scipy.io import loadmat, savemat
import random
import pickle as pkl
from tqdm import tqdm
import argparse

import os

torch.manual_seed(123)
torch.cuda.manual_seed(123)
np.random.seed(123)
random.seed(123)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', default='PointNet2-SSG', help='PointNet2-SSG PointNet2-MSG')
    parser.add_argument('--adv', type=str, required=True, help='JGBA JGBA_sw')
    parser.add_argument('--eps', type=float, required=True, help=0.3)
    parser.add_argument('--eps_iter', required=True, type=float, help='0.01')
    parser.add_argument('--n', type=int, required=True, help='40')
    opt = parser.parse_args()

    model_name = opt.model_name
    adv = opt.adv
    eps = opt.eps
    eps_iter = opt.eps_iter
    n = opt.n

    if adv == 'JGBA':
        attack = (adversarial_untargeted.JGBA, {"eps": eps, "n": n, "eps_iter":eps_iter})
    elif adv == 'JGBA_sw':
        attack = (adversarial_untargeted.JGBA_sw, {"eps": eps, "n": n, "eps_iter":eps_iter})

    attack_fn = attack[0]
    attack_param = attack[1]

    if not os.path.exists(os.path.join('save', model_name, adv+'-'+str(eps)+'-'+str(int(n))+'-'+str(eps_iter))):
        os.makedirs(os.path.join('save', model_name, adv+'-'+str(eps)+'-'+str(int(n))+'-'+str(eps_iter)))
    
    with open(os.path.join('dataset', 'random1024', 'whole_data_and_whole_label.pkl'), 'rb') as fid:
        whole_data, whole_label = pkl.load(fid)

    if args.model_name == 'PointNet2-SSG':
        from pointnet2.models.pointnet2_ssg_cls import Pointnet2SSG
        model = Pointnet2SSG(40, input_channels=0)  # , use_xyz=True)
        ckpt = torch.load('checkpoints_ssg/pointnet2_cls_best.pth.tar')['model_state']
    elif args.model_name == 'PointNet2-MSG':
        from pointnet2.models.pointnet2_msg_cls import Pointnet2MSG
        model = Pointnet2MSG(40, input_channels=0)  # , use_xyz=True)
        ckpt = torch.load('checkpoints_msg/pointnet2_cls_best.pth.tar')['model_state']
    else:
        print('No such model architecture')
        assert False

    model = model.to(device)
    model.load_state_dict(torch.load(checkpoint))
    model.eval()

    pytorch_utils.requires_grad_(model, False)
    
    print("Model name\t%s" % model_name)

    cnt = 0        # adv pointcloud successfully attacked
    CNT = 0        # clean pointcloud correctly classified
    
    for idx in tqdm(range(len(whole_data))):
        x = whole_data[idx]
        label = whole_label[idx]
        
        with torch.no_grad():
            y_pred = model(torch.from_numpy(x[np.newaxis,:,:]).float().to(device))
            y_pred_idx = np.argmax(y_pred.detach().cpu().numpy().flatten())
            
        if label != y_pred_idx:        # make sure the attack is based on the correct prediction
            continue
            
        CNT += 1

        x_adv_original = attack_fn(model, np.copy(x), label, attack_param)

        with torch.no_grad():
            y_pred_adv_original = model(torch.from_numpy(np.copy(x_adv_original)[np.newaxis,:,:]).float().to(device))
            y_pred_adv_original_idx = np.argmax(y_pred_adv_original.detach().cpu().numpy().flatten())
            
            if y_pred_adv_original_idx != label:        # make sure that original PGD fail to attack
                cnt += 1
                savemat(os.path.join('save', model_name, adv+'-'+str(eps)+'-'+str(int(n))+'-'+str(eps_iter), str(idx)+'.mat'), {'x_adv':x_adv_original, 'y_adv':y_pred_adv_original_idx, 'x':x, 'y':y_pred_idx})
                
    print("Total Sample: {}, correctly classified: {}, successfully attacked: {}".format(len(whole_data), CNT, cnt))
