# general imports
import os
import random
import time
import warnings
import numpy as np
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
from sklearn.metrics import confusion_matrix
import wandb

# imports from our scripts
from .ldam_drw.parse_arguments import parse_script_arguments
from .ldam_drw.utils import *
from .ldam_drw.losses import LDAMLoss, FocalLoss

from .models.cifar_resnet import create_model as create_resnet_small

from .datasets.gtsrb_dataset import load_gtsrb_datasets
from .datasets.cifar_lt_dataset import load_cifar_datasets
from .datasets.kdatasets import load_kdatasets
from .munit.utils import DSET_2_PROJ_MAP
from .munit.munit_loading_utils import load_munit_trainer
from .utils.common import subset_by_class


def integrate_wandb(args):
    project_name = DSET_2_PROJ_MAP[args.dataset]
    run = wandb.init(
        name=args.run_name if args.run_name else args.store_name,
        project=project_name,
        tags=["baselines", args.train_rule, args.loss_type],
    )
    wandb.config.update(args)


def make_store_name(args):
    store_name = [
        args.dataset,
        args.arch,
        args.loss_type,
        args.train_rule,
    ]
    if args.imb_type:
        store_name.append(args.imb_type)
    if args.imb_factor:
        store_name.append(str(args.imb_factor))
    if args.exp_str:
        store_name.append(args.exp_str)
    if args.use_munit:
        store_name.append("munit")
    if args.dataset.startswith("k49"):
        assert args.k49_train_version is not None
        store_name.append(f"vers{args.k49_train_version}")
        if args.k49_tfmd_class_rank is not None:
            store_name.append(f"tfmd{args.k49_tfmd_class_rank}")
    if args.oracle:
        store_name.append("oracle")
    return "_".join(store_name)


def main():
    args = parse_script_arguments()
    args.store_name = make_store_name(args)
    prepare_folders(args)
    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        warnings.warn(
            "You have chosen to seed training. "
            "This will turn on the CUDNN deterministic setting, "
            "which can slow down your training considerably! "
            "You may see unexpected behavior when restarting "
            "from checkpoints."
        )

    if args.gpu is not None:
        warnings.warn(
            "You have chosen a specific GPU. This will completely " "disable data parallelism."
        )

    ngpus_per_node = torch.cuda.device_count()
    integrate_wandb(args=args)

    main_worker(args.gpu, ngpus_per_node, args)


def get_dataset_attributes(dataset_name):
    if dataset_name == "CIFAR10":
        num_classes = 10
    elif dataset_name == "CIFAR100":
        num_classes = 100
    elif dataset_name == "GTSRB":
        num_classes = 43
    elif dataset_name.startswith("k49"):
        num_classes = 49
    else:
        raise ValueError("Given dataset name not supported.")

    if dataset_name in ["CIFAR10", "CIFAR100", "GTSRB"]:
        num_channels = 3
    elif dataset_name.startswith("k49"):
        num_channels = 1
    else:
        raise ValueError("Given dataset name not supported.")

    return {"num_channels": num_channels, "num_classes": num_classes}


def load_dataset_for_script(args, trainer_type="baseline"):
    if args.dataset in ["CIFAR10", "CIFAR100"]:
        dsets, output_tfm = load_cifar_datasets(
            root_dir=args.dataset_root,
            dataset_name=args.dataset,
            imb_type=args.imb_type,
            imb_factor=args.imb_factor,
            trainer_type=trainer_type,
            flip=args.flip_aug,
        )
    elif args.dataset == "GTSRB":
        dsets, output_tfm = load_gtsrb_datasets(
            root_dir=args.dataset_root,
            alpha=args.powerlaw_value,
            trainer_type=trainer_type,
            flip=args.flip_aug,
        )
    elif args.dataset.startswith("k49"):
        dsets, output_tfm = load_kdatasets(
            root_dir=args.dataset_root,
            dataset_name=args.dataset,
            trainer_type=trainer_type,
            train_version=args.k49_train_version,
            flip=args.flip_aug,
            tfmd_class_rank=args.k49_tfmd_class_rank,
            oracle=args.oracle,
        )
    else:
        raise ValueError("Given dataset name is not supported.")

    return dsets["train"], dsets["val"], dsets["test"], output_tfm


def create_classifier(args, dset_attributes, use_norm):
    if args.dataset in ["CIFAR10", "CIFAR100", "GTSRB"] or args.dataset.startswith("k49"):
        model = create_resnet_small(
            model_name=args.arch,
            num_channels=dset_attributes["num_channels"],
            num_classes=dset_attributes["num_classes"],
            use_norm=use_norm,
        )
    else:
        raise ValueError("Given dataset name is not supported.")

    print("\n", model, "\n")
    return model

def main_worker(gpu, ngpus_per_node, args):
    print(f"Step lr epoch milestones: {args.steplr_epoch1} and {args.steplr_epoch2}")
    global best_acc1
    best_acc1 = 0

    args.gpu = gpu

    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))

    # create model
    print("=> creating model '{}'".format(args.arch))
    dset_attributes = get_dataset_attributes(dataset_name=args.dataset)
    use_norm = True if args.loss_type == "LDAM" else False
    model = create_classifier(args=args, dset_attributes=dset_attributes, use_norm=use_norm)

    if args.gpu is not None:
        torch.cuda.set_device(args.gpu)
        model = model.cuda(args.gpu)
    else:
        # DataParallel will divide and allocate batch_size to all available GPUs
        model = torch.nn.DataParallel(model).cuda()

    optimizer = torch.optim.SGD(
        model.parameters(), args.lr, momentum=args.momentum, weight_decay=args.weight_decay
    )

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume, map_location="cuda:0")
            args.start_epoch = checkpoint["epoch"]
            best_acc1 = checkpoint["best_acc1"]
            if args.gpu is not None:
                # best_acc1 may be from a checkpoint from a different GPU
                best_acc1 = best_acc1.to(args.gpu)
            model.load_state_dict(checkpoint["state_dict"])
            optimizer.load_state_dict(checkpoint["optimizer"])
            print(
                "=> loaded checkpoint '{}' (epoch {})".format(args.resume, checkpoint["epoch"])
            )
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    cudnn.benchmark = True

    # Data loading code
    train_dataset, val_dataset, test_dataset, _ = load_dataset_for_script(args=args)
    if args.val_class_size:
        val_dataset = subset_by_class(val_dataset, per_class=args.val_class_size)
    print("Train/val/test set sizes:", len(train_dataset), len(val_dataset), len(test_dataset))

    cls_num_list = train_dataset.get_cls_num_list()
    print("\nClass num list:\n", cls_num_list, "\n")
    args.cls_num_list = cls_num_list

    munit_train_loader, output_tfm = None, None
    munit_trainer, munit_style_dim, munit_config = None, None, None
    if args.use_munit:
        munit_trainer, munit_style_dim, munit_config = load_munit_trainer(
            args=args, device="cuda"
        )

    if munit_trainer is not None:
        for param in munit_trainer.parameters():
            param.requires_grad = False
        munit_train_dataset, _, _, output_tfm = load_dataset_for_script(
            args, trainer_type="munit"
        )

    # init log for training
    log_training = open(os.path.join(args.root_log, args.store_name, "log_train.csv"), "w")
    log_testing = open(os.path.join(args.root_log, args.store_name, "log_test.csv"), "w")
    with open(os.path.join(args.root_log, args.store_name, "args.txt"), "w") as f:
        f.write(str(args))

    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True,
    )

    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True,
    )

    for epoch in range(args.start_epoch, args.epochs):
        print("Epoch: ", epoch + 1)
        adjust_learning_rate(optimizer, epoch, args)

        if args.train_rule == "None":
            train_sampler = None
            per_cls_weights = None
        elif args.train_rule == "Resample":
            train_sampler = ImbalancedDatasetSampler(train_dataset)
            per_cls_weights = None
        elif args.train_rule == "ResampleSimple":
            sample_weights = torch.from_numpy(train_dataset.sample_weights).double()
            train_sampler = torch.utils.data.WeightedRandomSampler(
                sample_weights, len(sample_weights)
            )
            per_cls_weights = None
        elif args.train_rule == "DRS_Simple":
            if epoch < args.steplr_epoch1:
                train_sampler = None
            else:
                sample_weights = torch.from_numpy(train_dataset.sample_weights).double()
                train_sampler = torch.utils.data.WeightedRandomSampler(
                    sample_weights, len(sample_weights)
                )
            per_cls_weights = None
        elif args.train_rule == "Reweight":
            train_sampler = None
            beta = 0.9999
            effective_num = 1.0 - np.power(beta, cls_num_list)
            per_cls_weights = (1.0 - beta) / np.array(effective_num)
            per_cls_weights = per_cls_weights / np.sum(per_cls_weights) * len(cls_num_list)
            per_cls_weights = torch.FloatTensor(per_cls_weights).cuda(args.gpu)
        elif args.train_rule == "DRW":
            train_sampler = None
            idx = epoch // args.steplr_epoch1
            betas = [0, 0.9999]
            effective_num = 1.0 - np.power(betas[idx], cls_num_list)
            per_cls_weights = (1.0 - betas[idx]) / np.array(effective_num)
            per_cls_weights = per_cls_weights / np.sum(per_cls_weights) * len(cls_num_list)
            per_cls_weights = torch.FloatTensor(per_cls_weights).cuda(args.gpu)
        else:
            warnings.warn("Sample rule is not listed")

        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            shuffle=(train_sampler is None),
            num_workers=args.workers,
            pin_memory=True,
            sampler=train_sampler,
        )

        if munit_trainer is not None:
            munit_train_loader = torch.utils.data.DataLoader(
                munit_train_dataset,
                batch_size=args.batch_size,
                shuffle=(train_sampler is None),
                num_workers=args.workers,
                pin_memory=True,
                sampler=train_sampler,
            )

        if args.loss_type == "CE":
            criterion = nn.CrossEntropyLoss(weight=per_cls_weights).cuda(args.gpu)
        elif args.loss_type == "LDAM":
            criterion = LDAMLoss(
                cls_num_list=cls_num_list, max_m=0.5, s=30, weight=per_cls_weights
            ).cuda(args.gpu)
        elif args.loss_type == "Focal":
            criterion = FocalLoss(weight=per_cls_weights, gamma=1).cuda(args.gpu)
        else:
            warnings.warn("Loss type is not listed")
            return

        # train for one epoch
        train(
            train_loader,
            model,
            criterion,
            optimizer,
            epoch,
            args,
            log_training,
            munit_train_loader,
            munit_trainer,
            munit_style_dim,
            output_tfm,
        )

        # evaluate on validation set
        acc1 = validate(val_loader, model, criterion, epoch, args, log_testing)

        # remember best acc@1 and save checkpoint
        is_best = acc1 > best_acc1
        best_acc1 = max(acc1, best_acc1)

        output_best = "Best Prec@1: %.3f\n" % (best_acc1)
        if args.verbose:
            print(output_best)
        log_testing.write(output_best + "\n")
        log_testing.flush()

        save_checkpoint(
            args,
            {
                "epoch": epoch + 1,
                "arch": args.arch,
                "state_dict": model.state_dict(),
                "best_acc1": best_acc1,
                "optimizer": optimizer.state_dict(),
            },
            is_best,
        )

    validate(test_loader, model, criterion, args.epochs + 1, args, log_testing, "test")


def train(
    train_loader,
    model,
    criterion,
    optimizer,
    epoch,
    args,
    log,
    munit_train_loader,
    munit_trainer,
    munit_style_dim,
    output_tfm,
):
    batch_time = AverageMeter("Time", ":6.3f")
    data_time = AverageMeter("Data", ":6.3f")
    losses = AverageMeter("Loss", ":.4e")
    top1 = AverageMeter("Acc@1", ":6.2f")
    top5 = AverageMeter("Acc@5", ":6.2f")

    # switch to train mode
    model.train()

    end = time.time()
    should_use_munit = (munit_trainer is not None) and (
        epoch > args.steplr_epoch1 or not args.delayed_munit
    )
    loaders = train_loader if not should_use_munit else zip(train_loader, munit_train_loader)

    train_targets = train_loader.dataset.target
    _, counts = np.unique(train_targets, return_counts=True)
    smallest_class_size = np.min(counts)
    if args.class_size_threshold:
        class_threshold_array = (args.class_size_threshold - counts) / (
            args.class_size_threshold - smallest_class_size
        )
        keepgen_probs = torch.from_numpy(np.maximum(class_threshold_array, 0)).cuda(
            non_blocking=True
        )
    else:
        keepgen_probs = torch.ones(counts.shape).cuda(non_blocking=True)

    for i, batch in enumerate(loaders):
        if should_use_munit:
            (input_main, target_main), (input_munit, target_munit) = batch
            input_main = input_main.cuda(non_blocking=True)
            input_munit = input_munit.cuda(non_blocking=True)
            target_main = target_main.cuda(non_blocking=True)
            target_munit = target_munit.cuda(non_blocking=True)
            gen_bs = int(args.gen_frac * input_main.shape[0])
            real_bs = input_main.shape[0] - gen_bs
            content, _ = munit_trainer.gen_a.encode(input_munit)
            style_rand = torch.randn(input_munit.shape[0], munit_style_dim, 1, 1).cuda()
            input_gen = (munit_trainer.gen_a.decode(content, style_rand) + 1) / 2
            input_gen = output_tfm(input_gen)

            if args.weighted_gen:
                mask = torch.bernoulli(keepgen_probs[target_munit]).bool()
                input_gen = torch.where(mask[:, None, None, None], input_gen, input_main)
                target_munit = torch.where(mask, target_munit, target_main)

            input = torch.cat((input_main[:real_bs], input_gen[real_bs:]), 0)
            target = torch.cat((target_main[:real_bs], target_munit[real_bs:]), 0)
        else:
            input, target = batch
            input = input.cuda(non_blocking=True)
            target = target.cuda(non_blocking=True)

        # measure data loading time
        data_time.update(time.time() - end)

        # compute output
        output = model(input)
        loss = criterion(output, target)

        # measure accuracy and record loss
        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        losses.update(loss.item(), input.size(0))
        top1.update(acc1[0], input.size(0))
        top5.update(acc5[0], input.size(0))

        # compute gradient and do SGD step
        normalized_loss = loss / args.accumulate_grad_batches
        normalized_loss.backward()
        if (i + 1) % args.accumulate_grad_batches == 0:
            optimizer.step()
            optimizer.zero_grad()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            output = (
                "Epoch: [{0}][{1}/{2}], lr: {lr:.5f}\t"
                "Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t"
                "Data {data_time.val:.3f} ({data_time.avg:.3f})\t"
                "Loss {loss.val:.4f} ({loss.avg:.4f})\t"
                "Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t"
                "Prec@5 {top5.val:.3f} ({top5.avg:.3f})".format(
                    epoch,
                    i,
                    len(train_loader),
                    batch_time=batch_time,
                    data_time=data_time,
                    loss=losses,
                    top1=top1,
                    top5=top5,
                    lr=optimizer.param_groups[-1]["lr"] * 0.1,
                )
            )

            if args.verbose:
                print(output)
            log.write(output + "\n")
            log.flush()

    wandb.log(
        {
            "train_loss": losses.avg,
            "train_acc": top1.avg,
            "lr": optimizer.param_groups[-1]["lr"],
        },
        step=epoch,
    )


def validate(val_loader, model, criterion, epoch, args, log=None, flag="val"):
    batch_time = AverageMeter("Time", ":6.3f")
    losses = AverageMeter("Loss", ":.4e")
    top1 = AverageMeter("Acc@1", ":6.2f")
    top5 = AverageMeter("Acc@5", ":6.2f")

    # switch to evaluate mode
    model.eval()
    all_preds = []
    all_targets = []
    with torch.no_grad():
        end = time.time()
        for i, (input, target) in enumerate(val_loader):
            if args.gpu is not None:
                input = input.cuda(args.gpu, non_blocking=True)
            target = target.cuda(args.gpu, non_blocking=True)

            # compute output
            output = model(input)
            loss = criterion(output, target)

            # measure accuracy and record loss
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            losses.update(loss.item(), input.size(0))
            top1.update(acc1[0], input.size(0))
            top5.update(acc5[0], input.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            _, pred = torch.max(output, 1)
            all_preds.extend(pred.cpu().numpy())
            all_targets.extend(target.cpu().numpy())

            if i % args.print_freq == 0:
                output = (
                    "Test: [{0}/{1}]\t"
                    "Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t"
                    "Loss {loss.val:.4f} ({loss.avg:.4f})\t"
                    "Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t"
                    "Prec@5 {top5.val:.3f} ({top5.avg:.3f})".format(
                        i,
                        len(val_loader),
                        batch_time=batch_time,
                        loss=losses,
                        top1=top1,
                        top5=top5,
                    )
                )

                if args.verbose:
                    print(output)
        cf = confusion_matrix(all_targets, all_preds).astype(float)
        cls_cnt = cf.sum(axis=1)
        cls_hit = np.diag(cf)
        cls_acc = cls_hit / cls_cnt
        output = "{flag} Results: Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f} Loss {loss.avg:.5f}".format(
            flag=flag, top1=top1, top5=top5, loss=losses
        )
        out_cls_acc = "%s Class Accuracy: %s" % (
            flag,
            (
                np.array2string(
                    cls_acc, separator=",", formatter={"float_kind": lambda x: "%.3f" % x}
                )
            ),
        )
        if args.verbose:
            print(output)
            print(out_cls_acc)
        if log is not None:
            log.write(output + "\n")
            log.write(out_cls_acc + "\n")
            log.flush()

        wandb.log(
            {
                flag + "_loss": losses.avg,
                flag + "_acc": top1.avg,
                flag + "_bal_acc": np.mean(cls_acc),
            },
            step=epoch,
        )

    return top1.avg


def adjust_learning_rate(optimizer, epoch, args):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    epoch = epoch + 1
    if epoch <= 5:
        lr = args.lr * epoch / 5
    elif epoch > args.steplr_epoch2:
        lr = args.lr * 0.0001
    elif epoch > args.steplr_epoch1:
        lr = args.lr * 0.01
    else:
        lr = args.lr
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr


if __name__ == "__main__":
    main()
