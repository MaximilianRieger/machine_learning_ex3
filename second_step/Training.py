import logging
from tqdm import tqdm

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