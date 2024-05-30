This folder contains the code for the image classification project. The main files are as follows:

1. `train.py`: train a classification model on CIFAR10(100) / TinyImageNet using SGD/Adam/SAM/AT. For AWP we use the code from the [official repository](https://github.com/csdongxian/AWP)
2. `attack.py`: test adversarial robustness of a model using torchattacks
3. `corruption.py`: test general robustness of a model using robustbench
