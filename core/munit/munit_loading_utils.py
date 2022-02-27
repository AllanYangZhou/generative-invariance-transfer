# general import
import torch

# imports from our packages
from .trainer import MUNIT_Trainer
from .utils import get_config


def load_munit_trainer(args, device):
    munit_trainer = None
    munit_style_dim = None
    munit_config = None

    if args.munit_ckpt is not None:
        print("\nCreating the MUNIT trainer and loading weights.\n")

        # create munit
        munit_config = get_config(args.munit_config)
        state_dict = torch.load(args.munit_ckpt)

        if args.munit_version == "original_munit":
            munit_trainer = MUNIT_Trainer(hyperparameters=munit_config)
            munit_trainer.gen_a.load_state_dict(state_dict["a"])
        else:
            raise ValueError("Given MUNIT version is not supported.")

        munit_trainer = munit_trainer.to(device).eval()

        # set munit style dimension
        munit_style_dim = munit_config["gen"]["style_dim"]

    return munit_trainer, munit_style_dim, munit_config
