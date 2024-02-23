import argparse
import logging

import torch
import time

from second_step.celebA_trainingrun import TrainingRun as CelebATrainingRun
from second_step.cifar10_trainingrun import TrainingRun as CIFAR10TrainingRun

datasets = ['cifar10', 'celebA']
models = ['simple', 'resnet', 'resnet50']
criterions = ['cross_entropy', 'mse']

def run_experiment(args=None):
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s ')

    if args is None:
        # Training settings
        parser = argparse.ArgumentParser(description='Machine Learning Exercise 3')
        parser.add_argument('--no-cuda', action='store_true', default=False, help='enables CUDA training')
        parser.add_argument('--gpuid', default='-1', type=str, help='id(s) for CUDA_VISIBLE_DEVICES')
        parser.add_argument('-w', '--workers', default=-1, type=int, help='Count of max Workers to use')
        parser.add_argument('-m', '--model', default='simple', required=True, type=str, choices=models)
        parser.add_argument('-d', '--dataset', default='breast_cancer', required=True, type=str, choices=datasets)
        parser.add_argument('-c', '--criterion', default='mse',  type=str, choices=criterions)
        parser.add_argument('-e', '--epochs', default=10, type=int, help='number of epochs to train')
        parser.add_argument('-b', '--batch_size', default=256, type=int, help='batch size')
        parser.add_argument('-lr', '--learning_rate', default=0.001, type=float, help='learning rate')
        parser.add_argument( '--log_interval', default=-1, type=int, help='log interval')
        parser.add_argument('-v', '--verbose', action='store_true', default=False, help='verbose')
        args = parser.parse_args()


    # check if cuda is available and set cuda flag
    args.cuda = not args.no_cuda and torch.cuda.is_available()

    # set device
    if args.cuda:
        torch.cuda.set_device(int(args.gpuid))
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    # set number of workers
    if args.workers == -1:
        args.workers = torch.get_num_threads()
    else:
        torch.set_num_threads(args.workers)

    # set device in args
    args.device = device
    # log args

    logging.info(f"device: {args.device}")
    logging.info(f"model: {args.model}")
    logging.info(f"epochs: {args.epochs}")
    logging.info(f"batch size: {args.batch_size}")
    logging.info(f"learning rate: {args.learning_rate}")
    logging.info(f"criterion: {args.criterion}")
    logging.info(f"dataset: {args.dataset}\n")

    # set debugging defaults
    args.verbose = True
    # args.workers = 0

    TrainingRun = None
    if args.dataset == 'cifar10':
        TrainingRun = CIFAR10TrainingRun
    elif args.dataset == 'celebA':
        TrainingRun = CelebATrainingRun
    else:
        raise ValueError('Dataset not recognized')

    # make dict from args
    args_dict = vars(args)
    training_run = TrainingRun(args_dict)
    # run training and time it
    start = time.time()
    training_run.run()
    end = time.time()
    logging.info(f'Training time: {end - start:.2f} seconds')
    # save trained model
    torch.save(training_run.trained_model, f'{args.dataset}_{args.model}.pt')


if __name__ == '__main__':
    run_experiment()






