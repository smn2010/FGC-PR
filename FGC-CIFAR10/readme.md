# Pytorch implementation of [Feature-Gate Coupling for Dynamic Network Pruning](https://arxiv.org/abs/2111.14302)

This repository is built upon [BAS](https://openreview.net/forum?id=Bke89JBtvB), thanks very much!

We implement the train and test process in the paper.


## TODO
- [ ] Upload codes on CIFAR100 (resnet20, 32, 56)
- [ ] Upload codes on IMAGENET (resnet18, 34)
- [ ] more network structure, e.g., mobilenet
- [ ] ...

## Setup

```
pytorch=1.7.1
```

## Run
### FGC Training

The following is the default settings of training FGC on CIFAR10 dataset. Replace ${DATA_DIR} with your own path of CIFAR10. Replace ${BAS_PRETRAIN} with the [BAS](https://openreview.net/forum?id=Bke89JBtvB) initialization checkpoint in './weights' (Our implementation of BAS is rather slow, so we load it as pretrained weights for training efficiency and stability).  Replace ${LOG_DIR} with the path for log files.
```
# DATA_DIR = 
# BAS_PRETRAIN = 
# LOG_DIR = 

python train_cifar10_fgc.py \
-data ${DATA_DIR} \
-a resnet20 -b 256 \
--bas_pretrain ${BAS_PRETRAIN} \
--gpu 0 \
--rho 0.4 \
--nn_epoch 20 \
--nn_num 200 \
--contrast_index 7 8 \
--kappa 0.003 \
--log ${LOG_DIR} \
--desp default
```
### FGC Testing

Evaluate the trained model and show its accuracy and pruning ratio. 
```
python test_comp_bas.py \
-data ${DATA_DIR} \
-a resnet20 -b 256 \
--resume ${WEIGHTS} \
--log ${LOG_FILE} \
--gpu ${GPU_ID}
```
## Result
TBD
