# SP<sup>2</sup>OT: Semantic-Regularized Progressive Partial Optimal Transport for Imbalanced Clustering
By [Chuyu Zhang*](https://scholar.google.com/citations?user=V7IktkcAAAAJ), [Hui Ren](https://rhfeiyang.github.io), and [Xuming He](https://faculty.sist.shanghaitech.edu.cn/faculty/hexm/index.html) 

This repo contains the Pytorch implementation of our [paper](http://arxiv.org/abs/2404.03446).

## Installation
```shell
git clone https://github.com/rhfeiyang/SPPOT.git
cd SPPOT
conda env create -f environment.yml
```


## Training

### Setup
Follow the steps below to setup the datasets:
- Change the file paths to the datasets in `utils/mypath.py`, e.g. `/path/to/cifar100`.


Our experimental evaluation includes the following datasets: CIFAR100, imagenet-r and iNaturalist18. Our code will build the imbalanced datasets automatically.

### Train model
For training on different datasets, args `--train_db_name` and `--val_db_name` should be specified. For example:
```shell
# For cifar100(imbalance ratio 100):
python train.py --train_db_name cifar_im --val_db_name cifar_im --num_classes 100 --mm_factor 1000 --topk_similarity 20 --bank_start_epoch 0 --bank_use --batch_size 512 --output_dir experiment/SPPOT/cifar100/ckpts

# For imagenet-r:
python train.py --train_db_name imagenet-r_im --val_db_name imagenet-r_im --num_classes 200 --mm_factor 1000 --topk_similarity 20 --bank_start_epoch 0 --bank_use --output_dir experiment/SPPOT/imagenet-r/ckpts
# For iNature100, 500, 1000:
python train.py --mm_factor 1000 --topk_similarity 20 --bank_start_epoch 0 --bank_use --train_db_name iNature_im --val_db_name iNature_im --num_classes 100 --output_dir experiment/SPPOT/inature100/ckpts
python train.py --mm_factor 1000 --topk_similarity 20 --bank_start_epoch 0 --bank_use --train_db_name iNature_im --val_db_name iNature_im --num_classes 500 --output_dir experiment/SPPOT/inature500/ckpts
python train.py --mm_factor 1000 --topk_similarity 20 --bank_start_epoch 1 --batch_size 1024 --bank_use --train_db_name iNature_im --val_db_name iNature_im --num_classes 1000 --output_dir experiment/SPPOT/inature1000/ckpts
```



### Evaluation
For evaluation, just change the script file to "eval.py". Models in "output_dir" will be loaded. For example:
```shell
# For cifar100(imbalance ratio 100):
python eval.py --train_db_name cifar_im --val_db_name cifar_im --imbalance_ratio 0.01 --num_classes 100  --output_dir experiment/SPPOT/cifar100/ckpts
# For imagenet-r:
python eval.py --train_db_name imagenet-r_im --val_db_name imagenet-r_im --num_classes 200 --output_dir experiment/SPPOT/imagenet-r/ckpts
# For iNature100, 500, 1000:
python eval.py --train_db_name iNature_im --val_db_name iNature_im --num_classes 100 --output_dir experiment/SPPOT/inature100/ckpts
python eval.py --train_db_name iNature_im --val_db_name iNature_im --num_classes 500 --output_dir experiment/SPPOT/inature500/ckpts
python eval.py --train_db_name iNature_im --val_db_name iNature_im --num_classes 1000 --output_dir experiment/SPPOT/inature1000/ckpts
```

## License

This software is released under a creative commons license which allows for personal and research use only. For a commercial license please contact the authors. You can view a license summary [here](http://creativecommons.org/licenses/by-nc/4.0/).
