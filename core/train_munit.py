"""
Copyright (C) 2018 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""
import argparse
import functools
import os
import shutil
import sys
import numpy as np
from torch.autograd import Variable
import torch.backends.cudnn as cudnn
import torch

try:
    from itertools import izip as zip
except ImportError:  # will be 3.x series
    pass
import tensorboardX
import wandb
from .munit.utils import (
    get_k49_data_loaders,
    get_cifar_or_gtsrb_data_loaders,
    prepare_sub_folder,
    write_html,
    write_loss,
    get_config,
    write_2images,
    Timer,
    DSET_2_PROJ_MAP,
)
from .munit.trainer import MUNIT_Trainer, UNIT_Trainer


def setup_wandb(config):
    if config["dataset_name"] not in DSET_2_PROJ_MAP:
        raise ValueError("Given dataset name is not supported.")

    project_name = DSET_2_PROJ_MAP[config["dataset_name"]]

    wandb_run = wandb.init(
        project=project_name, tags=["munit"], sync_tensorboard=True
    )
    wandb.config.update(opts)

    return project_name, wandb_run


def parse_script_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        default="configs/edges2handbags_folder.yaml",
        help="Path to the config file.",
    )
    parser.add_argument("--k49_train_version", type=int)
    parser.add_argument("--output_path", type=str, default=".", help="outputs path")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--trainer", type=str, default="MUNIT", help="MUNIT|UNIT")
    opts = parser.parse_args()

    return opts


if __name__ == "__main__":
    opts = parse_script_arguments()
    cudnn.benchmark = True

    # Load experiment setting
    config = get_config(opts.config)
    max_iter = config["max_iter"]
    display_size = config["display_size"]
    config["vgg_model_path"] = opts.output_path

    if "dataset_name" not in config:
        raise ValueError("Dataset name is not given in configuration file.")
    project_name, wandb_run = setup_wandb(config=config)

    # Setup model and data loader
    if opts.trainer == "MUNIT":
        trainer = MUNIT_Trainer(config)
    elif opts.trainer == "UNIT":
        trainer = UNIT_Trainer(config)
    else:
        sys.exit("Only support MUNIT|UNIT")
    trainer.cuda()

    if project_name == "k49lt":
        assert isinstance(opts.k49_train_version, int)
        get_loaders_fn = functools.partial(
            get_k49_data_loaders, train_version=opts.k49_train_version
        )
    elif project_name in ["gtsrb-lt", "cifar10-lt", "cifar100-lt"]:
        get_loaders_fn = get_cifar_or_gtsrb_data_loaders
    elif project_name == "dermnet-lt":
        get_loaders_fn = get_dermnet_data_loaders
    else:
        raise ValueError(f"Unknown project_name {project_name}!")

    train_loader_a, train_loader_b, test_loader_a, test_loader_b = get_loaders_fn(config)

    train_a_idcs, train_b_idcs = [], []
    test_a_idcs, test_b_idcs = [], []
    for i in range(display_size):
        (label_i_idcs,) = np.where(train_loader_a.dataset.target == i)
        train_a_idcs.append(label_i_idcs[0])
        (label_i_idcs,) = np.where(train_loader_b.dataset.target == i)
        train_b_idcs.append(label_i_idcs[0])
        (label_i_idcs,) = np.where(test_loader_a.dataset.target == i)
        test_a_idcs.append(label_i_idcs[0])
        (label_i_idcs,) = np.where(test_loader_b.dataset.target == i)
        test_b_idcs.append(label_i_idcs[0])
    train_display_images_a = torch.stack(
        [train_loader_a.dataset[i] for i in train_a_idcs]
    ).cuda()
    train_display_images_b = torch.stack(
        [train_loader_b.dataset[i] for i in train_b_idcs]
    ).cuda()
    test_display_images_a = torch.stack([test_loader_a.dataset[i] for i in test_a_idcs]).cuda()
    test_display_images_b = torch.stack([test_loader_b.dataset[i] for i in test_b_idcs]).cuda()

    # Setup logger and output folders
    model_name = os.path.splitext(os.path.basename(opts.config))[0]
    if project_name == "k49lt":
        model_name = model_name + f"_vers{opts.k49_train_version}"
    train_writer = tensorboardX.SummaryWriter(
        os.path.join(opts.output_path + "/logs", model_name)
    )
    output_directory = os.path.join(opts.output_path + "/outputs", model_name)
    checkpoint_directory, image_directory = prepare_sub_folder(output_directory)
    # copy config file to output folder
    shutil.copy(opts.config, os.path.join(output_directory, "config.yaml"))

    # Start training
    iterations = (
        trainer.resume(checkpoint_directory, hyperparameters=config) if opts.resume else 0
    )

    print("\nTraining starting...\n")
    done = False
    while not done:
        for it, (images_a, images_b) in enumerate(zip(train_loader_a, train_loader_b)):
            images_a, images_b = images_a.cuda().detach(), images_b.cuda().detach()

            #  Main training code

            trainer.dis_update(images_a, images_b, config)
            trainer.gen_update(images_a, images_b, config)
            torch.cuda.synchronize()
            trainer.update_learning_rate()

            # Dump training stats in log file
            if (iterations + 1) % config["log_iter"] == 0:
                write_loss(iterations, trainer, train_writer)

            # print to let us know training is ongoing
            if (iterations + 1) % config["print_iter"] == 0:
                print("Iteration: %08d/%08d" % (iterations + 1, max_iter))

            # Write images
            if (iterations + 1) % config["image_save_iter"] == 0:
                with torch.no_grad():
                    test_image_outputs = trainer.sample(
                        test_display_images_a, test_display_images_b
                    )
                    train_image_outputs = trainer.sample(
                        train_display_images_a, train_display_images_b
                    )
                write_2images(
                    test_image_outputs,
                    display_size,
                    image_directory,
                    "test_%08d" % (iterations + 1),
                )
                write_2images(
                    train_image_outputs,
                    display_size,
                    image_directory,
                    "train_%08d" % (iterations + 1),
                )
                # HTML
                write_html(
                    output_directory + "/index.html",
                    iterations + 1,
                    config["image_save_iter"],
                    "images",
                )

            if (iterations + 1) % config["image_display_iter"] == 0:
                with torch.no_grad():
                    image_outputs = trainer.sample(
                        train_display_images_a, train_display_images_b
                    )
                write_2images(image_outputs, display_size, image_directory, "train_current")

            # Save network weights
            if (iterations + 1) % config["snapshot_save_iter"] == 0:
                trainer.save(checkpoint_directory, iterations)

            iterations += 1
            if iterations >= max_iter:
                print("Finish training")
                done = True
                break
