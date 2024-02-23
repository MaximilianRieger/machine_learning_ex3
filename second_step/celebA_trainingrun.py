# main script for doing experiments
# defines the experiment class as well as training and validation classes

import logging
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader
from second_step.dataloading.CelebADataset import CelebADataset
from datasets.load_celebA_samples import load_celebA_samples
from second_step.SimpleModel import SimpleModel
from torchvision import transforms
from torchsummary import summary

class TrainingRun:
    def __init__(self, args):
        self.args = args
        self.epochs = args['epochs']
        self.training = Training(args)
        self.validation = Validation(args)
        # create and set model optimizer and criterion from args
        model = self.model(args)
        summary(model, input_size=(3,218, 178), batch_size=args['batch_size'])
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
        transform = transforms.Compose([transforms.ToTensor()])

        # set dataloader and device
        train_img_names, train_labels, test_img_names, test_labels = load_celebA_samples()
        self.training.dataloader = self.dataloader(train_img_names, train_labels, args, transform=transform)
        self.validation.dataloader = self.dataloader(test_img_names, test_labels, args, transform=transform)
        self.validation_accuracy = None
        self.final_epoch_loss = None
        self.final_validation_loss = None
        self.trained_model = None

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
        elif args['model'] == 'resnet50':
            # resnet model
            model = torch.hub.load('pytorch/vision:v0.6.0', 'resnet50', pretrained=False)
            model.fc = torch.nn.Linear(2048, 10)
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
        criterion = torch.nn.CrossEntropyLoss().to(args['device'])
        return criterion

    @staticmethod
    def dataloader(imgs, labels, args, transform=None):
        # create dataloader from args
        dataset = CelebADataset(imgs, labels, transform=transform)
        dataloader = DataLoader(dataset, batch_size=args['batch_size'], shuffle=True, num_workers=args['workers'])
        return dataloader

    def run(self):
        for epoch in range(self.epochs):
            self.training.run()
            self.validation.run()

        # save model
        self.trained_model = self.training.model
        logging.info(f'Validation set: Average loss: {self.validation.epoch_loss:.4f}, Accuracy: {self.validation.validation_accuracy:.0f}%\n')


class Training:
    def __init__(self, args):
        self.args = args
        self.model = None
        self.optimizer = None
        self.criterion = None
        self.dataloader = None
        self.device = None
        self.epoch = 0
        self.epoch_loss = 0

    def run(self):
        self.model.train()
        self.epoch += 1
        self.epoch_loss = 0
        data_len = len(self.dataloader)
        for batch_idx, (data, target) in tqdm(enumerate(self.dataloader), total=data_len, desc=f'Training epoch {self.epoch}'):
            data, target = data.to(self.device), target.to(self.device)
            self.optimizer.zero_grad()
            output = self.model(data)
            loss = self.criterion(output, target)
            loss.backward()
            self.optimizer.step()
            if self.args['log_interval'] > 0 and batch_idx % self.args['log_interval'] == 0:
                logging.info(f'Train Epoch: {self.epoch} [{batch_idx * self.args["batch_size"]}/{data_len} ({100. * batch_idx / data_len :.0f}%)]\tLoss: {loss.item():.6f}')
            self.epoch_loss += loss.item()
        self.epoch_loss /= data_len
        logging.info(f'Train Epoch: {self.epoch}\tLoss: {self.epoch_loss:.6f}')

class Validation:
    def __init__(self, args):
        self.validation_accuracy = None
        self.args = args
        self.model = None
        self.criterion = None
        self.dataloader = None
        self.device = None
        self.epoch = 0
        self.epoch_loss = 0

    def run(self):
        self.model.eval()
        self.epoch_loss = 0
        data_len = len(self.dataloader)
        imgs_done = 0
        correct = 0
        with torch.no_grad():
            for data, target in tqdm(self.dataloader, total=len(self.dataloader), desc=f'Validation epoch {self.epoch}'):
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                self.epoch_loss += self.criterion(output, target).item()
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()
                imgs_done += output.shape[0]

        self.epoch_loss /= data_len
        self.validation_accuracy = 100. * correct / imgs_done
        logging.info(f'Validation set: Average loss: {self.epoch_loss:.4f}, Accuracy: {correct}/{imgs_done} ({self.validation_accuracy:.0f}%)\n')