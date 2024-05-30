import torch
import torch.nn as nn
import argparse
import torch.nn.functional as F
import os
from model import PreActResNet18, WRN28_10, DeiT
from autoattack import AutoAttack
from utils import *
import argparse

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, required=True)
    parser.add_argument('--model', type=str, default='PRN', choices=['PRN', 'WRN', 'DeiT'])
    parser.add_argument('--dataset', default='cifar10', choices=['cifar10', 'cifar100', 'tiny-imagenet-200'])
    parser.add_argument('--attacker', default='PGD', choices=['PGD', 'FGSM', 'CW', 'AutoAttack'])
    parser.add_argument('--eps', default=8./255., type=float)
    parser.add_argument('--batch-size', default=1024, type=int)
    parser.add_argument('--norm', default='linf', choices=['linf', 'l2'])
    return parser.parse_args()

args = get_args()

if __name__ == '__main__':
    model_path = args.model_path
    dataset = args.dataset
    model_name = args.model
    label_dim = {'cifar10': 10, 'cifar100': 100, 'tiny-imagenet-200': 200}[dataset]
    model = {'PRN': PreActResNet18(label_dim), 'WRN': WRN28_10(label_dim), 'DeiT': DeiT(label_dim)}[model_name]
    normalizer = {'cifar10': normalize_cifar, 'cifar100': normalize_cifar, 'tiny-imagenet-200': normalize_tinyimagenet}[dataset]
    attacker = args.attacker
    
    #PGD1 = PGD(10, 0.25/255., 1./255., 'linf')
    #PGD2 = PGD(10, 0.5/255., 2./255., 'linf')
    
    #PGD16 = PGD(10, 2./255., 16./255., 'l2')
    #PGD32 = PGD(10, 4./255., 32./255., 'l2')
    #FGSM1 = PGD(1, 0.25/255., 1./255., 'linf')
    #FGSM16 = PGD(1, 2./255., 16./255., 'l2')

    pgd_iters = 10 if attacker == 'PGD' else 1
    eps = args.eps
    alpha = eps / 4
    norm = args.norm
    pgd = PGD(pgd_iters, alpha, eps, norm, False, normalizer)
    
    _, loader = load_dataset(dataset, args.batch_size)

    ckpt = torch.load(model_path, map_location='cpu')
    model.load_state_dict(ckpt)
    model.eval()
    model.cuda()
    acc = 0
    if args.attacker in ['PGD', 'FGSM']:
        for x,y in loader:
            x, y = x.cuda(), y.cuda()
            delta = pgd.perturb(model, x, y)
            pred = model((normalizer(x+delta)))
            acc += (pred.max(1)[1] == y).float().sum().item()
        acc /= 100
    elif args.attacker == 'CW':
        for x,y in loader:
            x, y = x.cuda(), y.cuda()
            x = normalizer(x)
            attacked_images = cw_l2_attack(model, x, y)
            pred = model(attacked_images)
            acc += (pred.max(1)[1] == y).float().sum().item()
        acc /= 100
    elif args.attacker == 'AutoAttack':
        norm = 'Linf' if args.norm == 'linf' else 'L2'
        adversary = AutoAttack(model, norm=norm, eps=args.eps, version='standard')
        for x,y in loader:
            x, y = x.cuda(), y.cuda()
            x = normalizer(x)
            adv_images = adversary.run_standard_evaluation(x, y, bs=64)
            pred = model(adv_images)
            acc += (pred.max(1)[1] == y).float().sum().item()
        acc /= 100
    print("Model: {}, Dataset: {}, Attack: {}, Accuracy: {}".format(model_name, dataset, args.attacker, acc))