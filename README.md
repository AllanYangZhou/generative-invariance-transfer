# Generative Invariance Transfer [ICLR 2022]

## Env setup
This will create a Conda environment called "tail."
```bash
conda env create -f environment.yml
```

## Kuzushiji-49-LT

### Data preparation
Download the following files four files into `./data/raw_kmnist_data` (originally source is [here](https://github.com/rois-codh/kmnist)):

| File            | Examples |  Download (NumPy format)      |
|-----------------|--------------------|----------------------------|
| Training images | 232,365            | [k49-train-imgs.npz](http://codh.rois.ac.jp/kmnist/dataset/k49/k49-train-imgs.npz) (63MB)   |
| Training labels | 232,365            | [k49-train-labels.npz](http://codh.rois.ac.jp/kmnist/dataset/k49/k49-train-labels.npz) (200KB)  |
| Testing images  | 38,547             | [k49-test-imgs.npz](http://codh.rois.ac.jp/kmnist/dataset/k49/k49-test-imgs.npz) (11MB)   |
| Testing labels  | 38,547             | [k49-test-labels.npz](http://codh.rois.ac.jp/kmnist/dataset/k49/k49-test-labels.npz) (50KB) |

Now to make long-tailed versions with rotation, background variation, and dilation/erosion run:
```bash
python -m core.make_kmnist_lt
```
This should produce a folder `./data/proc_kmnist_data` with 30 long-tailed datasets for each transformation.

### Train generative model, then classifiers.
There is a bash script for automatically training the GIT generative models on each of the 30 instances of K49-DIL-LT and K49-BG-LT, then training classifiers with or without GIT.
```bash
# Train generative models (MUNIT)
./scripts/train_k49_munits.sh
# Train classifiers
./scripts/train_k49.sh
```
The MUNIT model produces checkpoints in `./outputs`, which may be used for training classifiers. Classifier checkpoints will be in `./outputs/baseline`.

## CIFAR-LT
### Train generative model
This is based on [MUNIT](https://github.com/NVlabs/MUNIT) code. For CIFAR10 and CIFAR100, respectively:
```bash
python -m core.train_munit --config core/munit/configs/cifar10_0.01.yaml
python -m core.train_munit --config core/munit/configs/cifar100_0.01.yaml
```

### Train classifier
Example training command:
```bash
# Cross entropy loss with Delayed ReSampling
python -m core.train_baselines --config core/ldram_drw/configs/cifar10_0.01.yaml --loss_type CE --train_rule DRS_Simple
# Cross entropy loss with Delayed ReSampling with GIT using MUNIT model.
python -m core.train_baselines --config core/ldram_drw/configs/cifar10_0.01.yaml --loss_type CE --train_rule DRS_Simple --use_munit
```

Here are common options for the arguments:

`--config`:

* `core/ldam_drw/configs/cifar10_0.01.yaml`: (CIFAR10-LT)
* `core/ldam_drw/configs/cifar100_0.01.yaml`: (CIFAR100-LT)

`--loss_type`:

* `CE`: Cross-entropy loss
* `LDAM`: LDAM loss

`--train_rule`:

* `None`: Standard ERM training
* `DRS_Simple`: Delayed ReSampling

`--use_munit`: Include this to use the GIT generative model as augmentation during training. Check that the paths for `munit_ckpt` and `munit_config` are correct in the `--config` file.

## Credits
The generative model code is based off of [MUNIT](https://github.com/NVlabs/MUNIT). Classifier training code is based off of [LDAM-DRW](https://github.com/kaidic/LDAM-DRW).
