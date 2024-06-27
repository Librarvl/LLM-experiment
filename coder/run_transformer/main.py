import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

import sys
sys.path.append('/ssd1/liboran/gitspace/github_Librarvl/LLM-experiment/')

import torch

from train import TransformerTrainer
from utils.parse_args_init import set_args
from utils.trian_logger import create_logger


def init_system(args, logger):
    os.environ["CUDA_VISIBLE_DEVICES"] = args.device
    args.cuda = torch.cuda.is_available() and not args.no_cuda
    device = 'cuda:0' if args.cuda else 'cpu'
    args.device = device
    logger.info('using device:{}'.format(device))


def main():
    # init
    # args = set_args()
    # logger = create_logger(args)
    # init_system(args, logger)
    
    # 创建trainer
    trainer = TransformerTrainer()
    trainer.train()

    return


if __name__ == "__main__":
    main()