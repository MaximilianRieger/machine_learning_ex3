# main script for doing experiments
# defines the experiment class as well as training and validation classes

import logging
import torch
from torch.utils.data import DataLoader
from second_step.dataloading.CIFAR10Dataset import CIFAR10Dataset
from datasets.load_cifar10_samples import load_cifar10_samples
from second_step.SimpleModel_cifar10 import SimpleModel
from torchvision import transforms
from torchsummary import summary

from second_step.Training import Training
from second_step.Validation import Validation

class TrainingRun:
    def __init__(self, args):
        self.args = args
        self.epochs = args['epochs']
        self.training = Training(args)
        self.validation = Validation(args)
        # create and set model optimizer and criterion from args
        model = self.model(args)
        summary(model, input_size=(3,32,32), batch_size=args['batch_size'])
        optimizer = self.optimizer(args, model)
        criterion = self.criterion(args)
        # set parameters for training and validation
        self.training.device = args['device']
        self.training.model = model
        self.training.optimizer = optimizer
        self.training.criterion = criterion
        self.validation.device = args['device']
        self.validation.model = model
        self.validation.criterion = criterion
        # make toTensor transform
        transform = transforms.Compose([
            transforms.Lambda(lambda x: TrainingRun.reshape_tensor(x)),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261)) # Normalize images
        ])
        if args['model'] == 'resnet':
            transform = transforms.Compose([
                transforms.Lambda(lambda x: TrainingRun.reshape_resnet_tensor(x)),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))  # Normalize images
            ])
        # set dataloader and device
        train_img_names, train_labels, test_img_names, test_labels = load_cifar10_samples()
        self.training.dataloader = self.dataloader(train_img_names, train_labels, args, transform=transform)
        self.validation.dataloader = self.dataloader(test_img_names, test_labels, args, transform=transform)
        self.validation_accuracy = None
        self.final_epoch_loss = None
        self.final_validation_loss = None
        self.trained_model = None

    @staticmethod
    def reshape_tensor(x):
        # reshapes the custom cifar10 tensor to normal image tensor
        # x: tensor of shape (3072,)
        # returns: tensor of shape (3, 32, 32)
        x = torch.tensor(x, dtype=torch.float32)
        return x.view(3, 32, 32)
    @staticmethod
    def reshape_resnet_tensor(x):
        # reshapes the custom cifar10 tensor to normal image tensor
        # x: tensor of shape (3072,)
        # returns: tensor of shape (3, 32, 32)
        x = torch.tensor(x, dtype=torch.float32)
        x = x.view(3, 32, 32)
        x = torch.nn.functional.interpolate(x.unsqueeze(0), size=(224, 224)).squeeze(0)
        return x

    @staticmethod
    def model(args):
        model = None
        # create model from args
        if args['model'] == 'simple':
            # simple convolutional model
            model = SimpleModel()
        elif args['model'] == 'resnet':
            # resnet model
            model = torch.hub.load('pytorch/vision:v0.6.0', 'resnet18', pretrained=False)
            model.fc = torch.nn.Linear(512, 10)
        else:
            raise ValueError('Model not recognized')
        # move model to device
        model = model.to((args['device']))
        return model

    @staticmethod
    def optimizer(args, model):
        # create optimizer from args
        optimizer = torch.optim.Adam(model.parameters(), lr=args['learning_rate'])
        return optimizer

    @staticmethod
    def criterion(args):
        # create criterion from args
        if args['criterion'] == 'mse':
            mse = torch.nn.MSELoss().to(args['device'])
            criterion = lambda x, y: mse(x, TrainingRun.one_hot_encoding(y, 10))
        elif args['criterion'] == 'cross_entropy':
            criterion = torch.nn.CrossEntropyLoss().to(args['device'])
        else:
            raise ValueError('Criterion not recognized')
        return criterion

    @staticmethod
    def dataloader(imgs, labels, args, transform=None):
        # create dataloader from args
        dataset = CIFAR10Dataset(imgs, labels, transform=transform)
        dataloader = DataLoader(dataset, batch_size=args['batch_size'], shuffle=True, num_workers=args['workers'])
        return dataloader

    @staticmethod
    def one_hot_encoding(labels, num_classes):
        return torch.eye(num_classes, device=labels.device)[labels]

    def run(self):
        for epoch in range(self.epochs):
            self.training.run()
            self.validation.run()

        # save model
        self.trained_model = self.training.model
        logging.info(f'Validation set: Average loss: {self.validation.epoch_loss:.4f}, Accuracy: {self.validation.validation_accuracy:.0f}%\n')
