import torchattacks
from model import PreActResNet18, WRN28_10, DeiT
from utils import *
import recoloradv.mister_ed.config as config
from recoloradv.mister_ed.utils.pytorch_utils import DifferentiableNormalize
from recoloradv.utils import get_attack_from_name
from argparse import ArgumentParser

from robustbench.data import load_cifar10c, load_cifar100c
from robustbench.utils import clean_accuracy

parser = ArgumentParser()
parser.add_argument('--model_path', default='put filename here', type=str)
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

corruption_test_types = [['brightness'], ['fog'], ['frost'], ['gaussian_blur'], ['impulse_noise'], ['jpeg_compression'], ['shot_noise'], ['snow'], ['speckle_noise']]
for corruptions in corruption_test_types:
    print(f'\n#####  corruption  type: {corruptions}\n')
    x_test, y_test = load_cifar10c(n_examples=1000, corruptions=corruptions, severity=3)
    for model_name in ['put file name here', 'put file name here']:
        model = PreActResNet18(label_dim)
        if 'awp' in model_name:
            d = torch.load('./models/' + model_name, map_location='cuda:0')
            for k in list(d.keys()):
                if k.startswith('module.'):
                    d[k[7:]] = d[k]
                    del d[k]
            model.load_state_dict(d)
            
        else:
            model.load_state_dict(torch.load('./models/' + model_name, map_location='cuda:0'))
        model.eval()
        model.cuda()
        acc = clean_accuracy(model, x_test, y_test, device=torch.device('cuda'))
        print(f'Model: {model_name}, CIFAR-10-C accuracy: {acc:.1%}')