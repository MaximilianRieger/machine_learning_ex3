import logging
import torch
from tqdm import tqdm

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
                # self.epoch_loss += self.criterion(output, TrainingRun.one_hot_encoding(target, 10)).item()
                self.epoch_loss += self.criterion(output, target).item()
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()
                imgs_done += output.shape[0]

        self.epoch_loss /= data_len
        self.validation_accuracy = 100. * correct / imgs_done
        logging.info(f'Validation set: Average loss: {self.epoch_loss:.4f}, Accuracy: {correct}/{imgs_done} ({self.validation_accuracy:.0f}%)\n')