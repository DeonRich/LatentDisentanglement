import os
import pprint
import argparse

from tqdm import tqdm
import time
import torch
from torchmetrics.image.fid import NoTrainInceptionV3

import util
from infogan import Generator, Discriminator
from eval_utils import evaluate, prepare_data_for_gan, prepare_data_for_inception


def parse_args():
    r"""
    Parses command line arguments.
    """

    root_dir = os.path.abspath(os.path.dirname(__file__))
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_dir",
        type=str,
        default=os.path.join(root_dir, "data"),
        help="Path to dataset directory.",
    )
    parser.add_argument(
        "--ckpt_path",
        type=str,
        help="Path to checkpoint used for evaluation.",
    )
    parser.add_argument(
        "--im_size",
        type=int,
        default=64,
        help=(
            "Images are resized to this resolution. "
            "Models are automatically selected based on resolution."
        ),
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=64,
        help="Minibatch size used during evaluation.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=("cuda:0" if torch.cuda.is_available() else "cpu"),
        help="Device to evaluate on.",
    )
    parser.add_argument(
        "--submit",
        default=False,
        action="store_true",
        help="Generate Inception embeddings used for leaderboard submission.",
    )

    return parser.parse_args()

def eval(args):
    r"""
    Evaluates specified checkpoint.
    """

    # Set parameters
    nz, nl, nc, eval_size, num_workers = (
        32,
        120,
        32,
        10000,
        4,
    )

    # Configure models
    net_g = Generator(nz, nl, nc, 64)
    net_d = Discriminator(64, nl, nc)

    # Loads checkpoint
    state_dict = {"net_g":torch.load("generator_0212400.weights"), "net_d":torch.load("discriminator_0212400.weights")}
    net_g.load_state_dict(state_dict["net_g"])
    net_d.load_state_dict(state_dict["net_d"])

    # Configures eval dataloader
    _, eval_dataloader = util.get_dataloaders(
        args.data_dir, args.im_size, args.batch_size, eval_size, num_workers
    )

    metrics = evaluate(net_g, net_d, eval_dataloader, nz, nl, nc, args.device)
    pprint.pprint(metrics)


if __name__ == "__main__":
    start = time.time()
    eval(parse_args())
    end = time.time()
    print(f"took {(end-start)/60:0.3f} minutes")
