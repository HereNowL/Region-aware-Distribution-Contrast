import os
import torch
import fnmatch
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import argparse
import torch.utils.data.sampler as sampler
import shutil
from dataset.nyuv2ssl import *
from torch.autograd import Variable
from utils.evaluation import ConfMatrix, DepthMeter, NormalsMeter
import numpy as np
import pdb
from progress.bar import Bar as Bar
from utils import Logger, AverageMeter, accuracy, mkdir_p, savefig
from torch.autograd import Variable
import copy
from model.models import get_MTL
from losses.loss_functions import ComputeLoss
from evaluation.metrics import ComputeMetric
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import time
import matplotlib.pyplot as plt
from scipy.io import loadmat
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

parser = argparse.ArgumentParser(description='Multi-task partially-supervised learning with cross-task consistency (SegNet)')
parser.add_argument('--type', default='standard', type=str, help='split type: standard, wide, deep')
parser.add_argument('--weight', default='uniform', type=str, help='multi-task weighting: uniform')
parser.add_argument('--train_bs', default=9, type=int, help='train data batch size')
parser.add_argument('--val_bs', default=10, type=int, help='val data batch size')
parser.add_argument('--dataroot', default='./data/nyuv2', type=str, help='dataset root')
parser.add_argument('--temp', default=1.0, type=float, help='temperature for DWA (must be positive)')
parser.add_argument('--wlr', default=0.001, type=float, help='initial learning rate')
parser.add_argument('--out', default='./results/nyuv2/Gaussian_view/onelabel', help='Directory to output the result')
parser.add_argument('--alpha', default=1.5, type=float, help='hyper params of GradNorm')
parser.add_argument('--ssl-type', default='onelabel', type=str, help='ssl type: onelabel, randomlabels, full')
parser.add_argument('--labelroot', default='./data/nyuv2_settings/', type=str, help='partially setting root')
parser.add_argument('--resume', default='', type=str, metavar='PATH', help='path to latest checkpoint (default: none)')
parser.add_argument('--eval-last20', default=0, type=int, help='1 means we evaluate models in the last 20 epochs')
parser.add_argument('--rampup', default='fixed', type=str, help='up for ramp-up loss weight of cross-task consistency loss, fixed use constant loss weight.')
parser.add_argument('--con-weight', default=2.0, type=float, help='weight for cross-task consistency loss')
parser.add_argument('--reg-weight', default=0.5, type=float, help='weight for cross-task consistency loss')
parser.add_argument('--best', default='checkpoints/RDCNet_onelabel_best_model.pth.tar', type=str, metavar='PATH', help='path to best checkpoint (default: none)')
parser.add_argument('--seed', default=0, type=int, help='random seed ID')
opt = parser.parse_args()

tasks = ['semantic', 'depth', 'normal']
input_channels = [13, 1, 3]
# torch.use_deterministic_algorithms(True)
def seed_torch(seed):
	random.seed(seed)
	os.environ['PYTHONHASHSEED'] = str(seed) # 为了禁止hash随机化，使得实验可复现
	np.random.seed(seed)
	torch.manual_seed(seed)
	torch.cuda.manual_seed(seed)
	torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.

seed_torch(opt.seed)

import sys
sys.stdout = Logger(os.path.join(opt.out, 'log_file.txt'))

# map_color = loadmat('utils/class13Mapping.mat')
model = get_MTL(tasks)
model = model.cuda()

# define dataset path
dataset_path = opt.dataroot

nyuv2_test_set = NYUv2(root=dataset_path, train=False)

nyuv2_test_loader = torch.utils.data.DataLoader(
    dataset=nyuv2_test_set,
    batch_size=opt.val_bs,
    shuffle=False, num_workers=0)

test_batch = len(nyuv2_test_loader)
avg_cost = np.zeros([12], dtype=np.float32)
# initialize compute loss, compute metric
comp_loss = ComputeLoss()
comp_metric = ComputeMetric()

if opt.best:
	checkpoint = torch.load(opt.best)
	model.load_state_dict(checkpoint, strict=True)
	print('=> best checkpoint from {} loaded!'.format(opt.best))

stl_performance = {
                    'full': {'semantic': 37.447399999999995, 'depth': 0.607902, 'normal': 25.938105}, 
                    'onelabel': {'semantic': 26.1113, 'depth': 0.771502, 'normal': 30.073763}, 
                    'randomlabels': {'semantic': 28.7153, 'depth': 0.754012, 'normal': 28.946388}
}

model.eval()
conf_mat = ConfMatrix(model.class_nb)
depth_mat = DepthMeter()
normal_mat = NormalsMeter()
cost = np.zeros(12, dtype=np.float32)
with torch.no_grad():  # operations inside don't track history
    nyuv2_test_dataset = iter(nyuv2_test_loader)
    for k in range(test_batch):
        test_data, test_label, test_depth, test_normal, test_sam, test_edge = nyuv2_test_dataset.__next__()
        test_data, test_label = test_data.cuda(),  test_label.type(torch.LongTensor).cuda()
        test_depth, test_normal = test_depth.cuda(), test_normal.cuda()

        test_pred_dict, _ = model(test_data)
        test_pred = [test_pred_dict['semantic'], test_pred_dict['depth'], test_pred_dict['normal']]
        test_loss = comp_loss.compute_supervision(test_pred[0], test_label, test_pred[1], test_depth, test_pred[2], test_normal)

        conf_mat.update(test_pred[0].argmax(1).flatten(), test_label.flatten())
        depth_mat.update(test_pred[1], test_depth)
        normal_mat.update(test_pred[2], test_normal)
        cost[0] = test_loss[0].item()
        cost[3] = test_loss[1].item()
        cost[6] = test_loss[2].item()

        avg_cost[0:] += cost[0:] / test_batch
    avg_cost[1:3] = conf_mat.get_metrics()
    depth_metric = depth_mat.get_score()
    avg_cost[4], avg_cost[5] = depth_metric['l1'], depth_metric['rmse']
    normal_metric = normal_mat.get_score()
    avg_cost[7], avg_cost[8], avg_cost[9], avg_cost[10], avg_cost[11] = normal_metric['mean'], normal_metric['rmse'], normal_metric['11.25'], normal_metric['22.5'], normal_metric['30']


mtl_performance = 0.0
mtl_performance += (avg_cost[1]* 100 - stl_performance[opt.ssl_type]['semantic']) / stl_performance[opt.ssl_type]['semantic']
mtl_performance -= (avg_cost[4] - stl_performance[opt.ssl_type]['depth']) / stl_performance[opt.ssl_type]['depth']
mtl_performance -= (avg_cost[7] - stl_performance[opt.ssl_type]['normal']) / stl_performance[opt.ssl_type]['normal']
mtl_performance = mtl_performance / len(tasks) * 100


print('TEST: {:.4f} {:.4f} {:.4f} | {:.4f} {:.4f} {:.4f} | {:.4f} {:.4f} {:.4f} {:.4f} {:.4f} {:.4f} | {:.2f}'
        .format(avg_cost[0], avg_cost[1], avg_cost[2], avg_cost[3],
            avg_cost[4], avg_cost[5], avg_cost[6], avg_cost[7], avg_cost[8], avg_cost[9],
            avg_cost[10], avg_cost[11], mtl_performance))
