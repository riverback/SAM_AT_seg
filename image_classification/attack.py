import torchattacks
from model import PreActResNet18, WRN28_10, DeiT
from utils import *
import recoloradv.mister_ed.config as config
from recoloradv.mister_ed.utils.pytorch_utils import DifferentiableNormalize
from recoloradv.utils import get_attack_from_name
from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument('--model_path', default='cifar10_prn_sam_0_1.pth', type=str)
args = parser.parse_args()
file_name = args.model_path

class Model(nn.Module):
    def __init__(self, model, norm):
        super(Model, self).__init__()
        self.model = model
        self.norm = norm

    def forward(self, x):
        return self.model(self.norm(x))

label_dim = 10
if 'cifar10_' in file_name:
    label_dim = 10
    normalizer = DifferentiableNormalize(
        mean=config.CIFAR10_MEANS,
        std=config.CIFAR10_STDS,
    )
    norm = normalize_cifar
    train_loader, test_loader = load_dataset('cifar10', 1000)
elif 'cifar100_' in file_name:
    label_dim = 100
    normalizer = DifferentiableNormalize(
        mean=CIFAR100_MEAN,
        std=CIFAR100_STD,
    )
    norm = normalize_cifar100
    train_loader, test_loader = load_dataset('cifar100', 1000)
elif 'tiny' in file_name:
    label_dim = 200
    normalizer = DifferentiableNormalize(
        mean=TINYIMAGENET_MEAN,
        std=TINYIMAGENET_STD,
    )
    norm = normalize_tinyimagenet
    train_loader, test_loader = load_dataset('tiny-imagenet-200', 1000)
else:
    raise ValueError('Unknown dataset')

if 'prn' in file_name and 'deit' not in file_name and 'wrn' not in file_name:
    model = PreActResNet18(label_dim)
elif 'wrn' in file_name:
    model = WRN28_10(label_dim)
elif 'deit' in file_name:
    model = DeiT(label_dim)

d = torch.load('./models/' + file_name, map_location='cuda:0')
for k in list(d.keys()):
    if k.startswith('module.'):
        d[k[7:]] = d[k]
        del d[k]

model.load_state_dict(d)
model.eval()
model.cuda()

normed_model = Model(model, norm)
normed_model.eval()
normed_model.cuda()

# test clean accuracy on the whole test set
acc = 0.
for x, y in test_loader:
    x, y = x.cuda(), y.cuda()
    with torch.no_grad():
        pred = normed_model(x)
        acc += (pred.max(1)[1] == y).float().sum().item()
acc /= len(test_loader.dataset)
print('Model: {}, Clean Accuracy: {:.4f}'.format(file_name, acc))
with open(f'./logs/{file_name}-attack_log.txt', 'a') as f:
    f.write('Model: {}, Clean Accuracy: {:.4f}\n'.format(file_name, acc))

# attackers
pgd_1 = torchattacks.PGD(normed_model, eps=1 / 255, alpha=0.25 / 255, steps=10)
pgd_2 = torchattacks.PGD(normed_model, eps=2 / 255, alpha=0.5 / 255, steps=10)
pgd_4 = torchattacks.PGD(normed_model, eps=4 / 255, alpha=1 / 255, steps=10)
pgd_8 = torchattacks.PGD(normed_model, eps=8 / 255, alpha=2 / 255, steps=10)
fgsm_1 = torchattacks.FGSM(normed_model, eps=1 / 255)
fgsm_8 = torchattacks.FGSM(normed_model, eps=8 / 255)
cw = torchattacks.CW(normed_model, c=1, kappa=0, steps=10)
autoattack = torchattacks.APGDT(normed_model, norm='Linf', eps=4/255, steps=5, n_restarts=1, seed=0
                                , eot_iter=1, rho=.75, verbose=False, n_classes=label_dim)
pgd_l2_32 = torchattacks.PGDL2(normed_model, eps=32 / 255, alpha=8 / 255, steps=10)
pgd_l2_64 = torchattacks.PGDL2(normed_model, eps=64 / 255, alpha=16 / 255, steps=10)
pixle = torchattacks.Pixle(normed_model, max_iterations=5, restarts=5)
fab = torchattacks.FAB(normed_model, eps=8 / 255, norm='L2')

recolor_attack = get_attack_from_name('recoloradv+stadv+delta', model, normalizer, verbose=True)
stadv_attack = get_attack_from_name('stadv', model, normalizer, verbose=True)

lib_attacker_list = [pgd_1, pgd_2, pgd_4, pgd_8, fgsm_1, fgsm_8, cw, autoattack,  pgd_l2_32,
                     pgd_l2_64, pixle, fab]
lib_atkname_list = ['pgd_1', 'pgd_2', 'pgd_4', 'pgd_8', 'fgsm_1', 'fgsm_8', 'cw', 'autoattack',
                    'pgd_l2_32', 'pgd_l2_64', 'pixle', 'fab']
sem_attacker_list = [recolor_attack, stadv_attack]
sem_atkname_list = ['recolor', 'stadv']
for i in range(len(lib_attacker_list)):
    try:
        lib_attacker = lib_attacker_list[i]
        lib_atkname = lib_atkname_list[i]
        acc = 0
        # get first 1000 imgs and calculate acc
        for x, y in test_loader:
            x, y = x.cuda(), y.cuda()
            adv_x = lib_attacker(x, y)
            pred = normed_model(adv_x)
            acc += (pred.max(1)[1] == y).float().sum().item()
            break
        acc /= 1000
        print('Model: {}, Attack: {}, Accuracy: {}'.format(file_name, lib_atkname, acc))
        # write to log in ./logs/attack_log.txt
        with open(f'./logs/{file_name}-attack_log.txt', 'a') as f:
            f.write('Model: {}, Attack: {}, Accuracy: {}\n'.format(file_name, lib_atkname, acc))
    except:
        print('Model: {}, Attack: {}, Accuracy: {}'.format(file_name, lib_atkname, 'failed'))
        # write to log in ./logs/attack_log.txt
        with open(f'./logs/{file_name}-attack_log.txt', 'a') as f:
            f.write('Model: {}, Attack: {}, Accuracy: {}\n'.format(file_name, lib_atkname, 'failed'))

for i in range(len(sem_attacker_list)):
    try:
        sem_attacker = sem_attacker_list[i]
        sem_atkname = sem_atkname_list[i]
        acc = 0
        # get first 1000 imgs and calculate acc
        for x, y in test_loader:
            x, y = x.cuda(), y.cuda()
            adv_x = sem_attacker.attack(x, y)[0]
            pred = normed_model(adv_x)
            acc += (pred.max(1)[1] == y).float().sum().item()
            break
        acc /= 1000
        print('Model: {}, Attack: {}, Accuracy: {}'.format(file_name, sem_atkname, acc))
        # write to log in ./logs/attack_log.txt
        with open(f'./logs/{file_name}-attack_log.txt', 'a') as f:
            f.write('Model: {}, Attack: {}, Accuracy: {}\n'.format(file_name, sem_atkname, acc))
    except:
        print('Model: {}, Attack: {}, Accuracy: {}'.format(file_name, sem_atkname, 'failed'))
        # write to log in ./logs/attack_log.txt
        with open(f'./logs/{file_name}-attack_log.txt', 'a') as f:
            f.write('Model: {}, Attack: {}, Accuracy: {}\n'.format(file_name, sem_atkname, 'failed'))
