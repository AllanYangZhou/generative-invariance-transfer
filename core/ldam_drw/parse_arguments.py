# general package imports
import argparse
import configargparse


def parse_script_arguments():
    parser = argparse.ArgumentParser()
    parser = configargparse.ArgumentParser(
        config_file_parser_class=configargparse.YAMLConfigFileParser,
        parents=[parser],
        add_help=False,
    )

    # config file
    parser.add_argument("--config", is_config_file=True)

    # wandb config
    parser.add_argument("--run_name", type=str)

    # dataset information
    parser.add_argument("--dataset", type=str)
    parser.add_argument("--dataset_root", type=str)
    parser.add_argument("--imb_type", type=str)
    parser.add_argument("--imb_factor", type=float)
    parser.add_argument("--powerlaw_value", type=float)

    parser.add_argument("--k49_train_version", type=int, help="Training set to use (0-29).")
    parser.add_argument(
        "--k49_tfmd_class_rank", type=int, help="# 0 (largest) to 48 (smallest)."
    )
    parser.add_argument("--val_class_size", type=int)

    # model architecture
    parser.add_argument("--arch", type=str)

    # training rule
    parser.add_argument("--loss_type", type=str)
    parser.add_argument("--train_rule", type=str)
    parser.add_argument("--flip_aug", action="store_true")

    # training details
    parser.add_argument("--rand_number", type=int)
    parser.add_argument("--workers", type=int)

    parser.add_argument("--epochs", type=int)
    parser.add_argument("--start_epoch", type=int)
    parser.add_argument("--batch_size", type=int)
    parser.add_argument("--accumulate_grad_batches", type=int, default=1)

    parser.add_argument("--lr", type=float)
    parser.add_argument("--momentum", type=float)
    parser.add_argument("--weight_decay", type=float)
    parser.add_argument("--steplr_epoch1", type=int, default=160)
    parser.add_argument("--steplr_epoch2", type=int, default=180)

    parser.add_argument("--print_freq", type=int)
    parser.add_argument("--resume", type=str)
    parser.add_argument("--evaluate", action="store_true")
    parser.add_argument("--pretrained", action="store_true")

    parser.add_argument("--seed", type=int)
    parser.add_argument("--gpu", type=int)

    # saving logs
    parser.add_argument("--root_log", type=str)
    parser.add_argument("--root_model", type=str)
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--exp_str", type=str)

    # munit arguments
    parser.add_argument("--use_munit", action="store_true")
    parser.add_argument("--munit_ckpt", type=str)
    parser.add_argument("--munit_config", type=str)
    parser.add_argument("--munit_version", type=str)
    parser.add_argument("--gen_frac", type=float)
    parser.add_argument("--delayed_munit", action="store_true")
    parser.add_argument("--oracle", action="store_true")
    parser.add_argument("--weighted_gen", action="store_true")
    parser.add_argument("--class_size_threshold", type=int)

    args = parser.parse_args()

    print(parser.format_values())

    return args
