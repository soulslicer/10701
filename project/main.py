from __future__ import print_function
import argparse
import os
import sys
import shutil
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.nn.init as init 
import torchvision.models as models 
from torchvision import transforms
from torch.autograd import Variable
import torch.backends.cudnn as cudnn
from triplet_image_loader import TripletImageLoader
import scipy.io as sio

################################################
# insert this to the top of your scripts (usually main.py)
# This is due to updated PyTorch
################################################
import sys, warnings, traceback, torch
def warn_with_traceback(message, category, filename, lineno, file=None, line=None):
    sys.stderr.write(warnings.formatwarning(message, category, filename, lineno, line))
    traceback.print_stack(sys._getframe(2))
warnings.showwarning = warn_with_traceback; warnings.simplefilter('always', UserWarning);
torch.utils.backcompat.broadcast_warning.enabled = True
torch.utils.backcompat.keepdim_warning.enabled = True


import numpy as np

################################################
### Training settings
### These are different parameters for model/data/hyperparameter 
### The details for each can be found in "help = ...." descriptions
################################################

# that can be set while running the script from the terminal.
parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
parser.add_argument('--train_batch_size', type=int, default=4, metavar='N',
                    help='input batch size for training (default: 64)')
parser.add_argument('--batch_size', type=int, default=1, metavar='N',
                    help='input batch size for testing (default: 64)')
parser.add_argument('--epochs', type=int, default=200, metavar='N',
                    help='number of epochs to train (default: 200)')
parser.add_argument('--start_epoch', type=int, default=1, metavar='N',
                    help='number of start epoch (default: 1)')
parser.add_argument('--lr', type=float, default=1e-3, metavar='LR',
                    help='learning rate (default: 5e-5)')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='enables CUDA training')
parser.add_argument('--log-interval', type=int, default=20, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--margin', type=float, default=0.2, metavar='M',
                    help='margin for triplet loss (default: 0.2)')
parser.add_argument('--resume', default='', type=str,
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--name', default='Conditional_Similarity_Network', type=str,
                    help='name of experiment')
parser.add_argument('--embed_loss', type=float, default=5e-3, metavar='M',
                    help='parameter for loss for embedding norm')
parser.add_argument('--vae_loss', type=float, default=1, metavar='M',
                    help='parameter for loss for embedding norm')
parser.add_argument('--triplet_loss', type=float, default=1, metavar='M',
                    help='parameter for loss for embedding norm')
parser.add_argument('--mask_loss', type=float, default=5e-4, metavar='M',
                    help='parameter for loss for mask norm')
parser.add_argument('--num_traintriplets', type=int, default=50000, metavar='N',
                    help='how many unique training triplets (default: 50000)')
parser.add_argument('--num_valtriplets', type=int, default=20000, metavar='N',
                    help='how many unique validation triplets (default: 10000)')
parser.add_argument('--num_testtriplets', type=int, default=40000, metavar='N',
                    help='how many unique test triplets (default: 20000)')

parser.add_argument('--dim_embed', type=int, default=64, metavar='N',
                    help='how many dimensions in embedding (default: 64)')
parser.add_argument('--test', dest='test', action='store_true',
                    help='To only run inference on test set')
parser.add_argument('--learned', dest='learned', action='store_true',
                    help='To learn masks from random initialization')
parser.add_argument('--prein', dest='prein', action='store_true',
                    help='To initialize masks to be disjoint')
parser.add_argument('--visdom', dest='visdom', action='store_true',
                    help='Use visdom to track and plot')

parser.add_argument('--image_size', type=int, default=224,  
                    help='height/width length of the input images, default=64')

parser.add_argument('--ndf', type=int, default=32,
                    help='number of output channels for the first decoder layer, default=32')

parser.add_argument('--nef', type=int, default=32,
                    help='number of output channels for the first encoder layer, default=32')

#same as dim_embed
parser.add_argument('--nz', type=int, default=64,
                    help='size of the latent vector z, default=64')
parser.add_argument('--ngpu', type=int, default=1,
                    help='number of GPUs to use')

parser.add_argument('--outf', default='./output',
                    help='folder to output images and model checkpoints')


parser.add_argument('--beta1', type=float, default=0.9,
                    help='beta1 for adam, default=0.1')
parser.add_argument('--beta2', type=float, default=0.999,
                    help='beta2 for adam, default=0.001')


parser.add_argument('--nc', type=int, default=3,
                    help='number of input channel in data. 3 for rgb, 1 for grayscale')
parser.set_defaults(test=False)
parser.set_defaults(learned=False)
parser.set_defaults(prein=False)
parser.set_defaults(visdom=False)

best_acc = 0

def main():
    global args, best_acc
    global  log_interval
    log_interval = 30
    args = parser.parse_args()
    print(args)
    nz = int(args.dim_embed)
    nef = int(args.nef)
    ndf = int(args.ndf)
    ngpu = int(args.ngpu)
    nc = int(args.nc)
    out_size = args.image_size // 16

    args.cuda = not args.no_cuda and torch.cuda.is_available()
    torch.manual_seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed(args.seed)
    
    normalize = transforms.Normalize(mean=[0.0, 0.0, 0.0],
                                     std=[1, 1, 1])

    out_size = args.image_size // 16  
    kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}
    train_loader = torch.utils.data.DataLoader(
        TripletImageLoader('data/FID-300/', [0,250],
                        transform=transforms.Compose([
                            transforms.Resize((224,112)),
                            transforms.CenterCrop((224,112)),
                            #transforms.RandomHorizontalFlip(),
                            #transforms.RandomRotation(20, resample=False, expand=False, center=None),
                            transforms.ToTensor(),
                            normalize,
                    ])),
        batch_size=args.train_batch_size, shuffle=True, **kwargs)


    # Test
    for batch_idx, (anchor_out, pos_out, neg_out, anchor_in, pos_in, neg_in) in enumerate(train_loader):
          print("LOAD")

    print("OK")

if __name__ == '__main__':
    main()    
