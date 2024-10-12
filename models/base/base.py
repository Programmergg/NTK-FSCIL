import abc
from utils import Timer
from dataloader.data_utils import *

class Trainer(object, metaclass=abc.ABCMeta):
    def __init__(self, args):
        self.args = args
        self.args = set_up_datasets(self.args)
        self.timer = Timer()

        # train statistics
        self.trlog = {}
        self.trlog['max_acc_epoch'] = 0
        self.trlog['max_acc'] = [0.0] * args.sessions

    @abc.abstractmethod
    def train(self):
        pass