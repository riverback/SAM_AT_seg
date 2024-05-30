import recoloradv.mister_ed.config as config
from recoloradv.mister_ed.utils.pytorch_utils import DifferentiableNormalize

# ReColorAdv
from recoloradv.utils import get_attack_from_name
from model import PreActResNet18
from utils import *


class Model(nn.Module):
    def __init__(self, model, norm):
        super(Model, self).__init__()
        self.model = model
        self.norm = norm

    def forward(self, x):
        return self.model(self.norm(x))


model = PreActResNet18(10)
model.load_state_dict(torch.load('./cifar10_models/cifar10_prn_sgd_sub.pth'))
model.eval()
model.cuda()

# PGD attack
# Mod = Model(model, normalize_cifar)
# Mod.eval()
# Mod.cuda()

# get imgs and labels
train_loader, test_loader = load_dataset('cifar10', 1024)
normalizer = DifferentiableNormalize(
    mean=config.CIFAR10_MEANS,
    std=config.CIFAR10_STDS,
)
attack = get_attack_from_name('recoloradv', model, normalizer, verbose=True)
acc = 0
for x, y in test_loader:
    x, y = x.cuda(), y.cuda()
    adv_x = attack.attack(x, y)[0]
    pred = model(normalizer(adv_x))
    acc += (pred.max(1)[1] == y).float().sum().item()
    break
acc /= 1024
print(acc)
