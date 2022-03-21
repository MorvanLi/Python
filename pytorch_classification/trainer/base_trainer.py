# author_='bo.li';
# date: 3/21/22 10:32 AM

import torch
import abc
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from typing import List
import os
import time


class BasicTrainer(metaclass=abc.ABCMeta):
    """
    The Abstract class of trainer
    """

    def __init__(self,
                 model_save_dir: str,
                 log_dir: str,
                 *args,
                 **kwargs):
        self.model_save_dir: str = model_save_dir
        self.log_dir: str = log_dir
        self.logger: SummaryWriter() = None
        self.model: nn.Module = None
        self.training_loader: DataLoader = None
        self.validate_loader: DataLoader = None
        self.optimizer: torch.optim.Optimizer = None
        self.device: str = None

    def fit(self,
            model: nn.Module,
            train_loader: DataLoader,
            validate_loader: DataLoader,
            optimizer: torch.optim.Optimizer,
            criterions: None,
            epochs: int,
            device: str = "cuda:0",
            **kwargs) -> None:
        """
        Fit the model and save its weights by given the train & valid dataloader
        """
        # init logger
        time_stamp = time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime())
        log_file_name = time_stamp + "|"
        self.logger = SummaryWriter(os.path.join(self.log_dir, log_file_name))
        # init dataloader
        self.training_loader = train_loader
        self.validate_loader = validate_loader
        # init model and device
        self.device = device
        self.model = model.to(device)
        # init optimizer
        self.optimizer = optimizer
        # init criterions
        self.criterions = criterions

        # mkdir
        if not os.path.exists(self.model_save_dir):
            os.makedirs(os.path.join(self.model_save_dir))

        if not os.path.exists(self.log_dir):
            os.makedirs(os.path.join(self.log_dir))

        # start training
        best_acc = 0.
        for epoch in range(1, epochs + 1):
            train_logs = self._train_one_epoch(epoch=epoch)
            self.write_log(train_logs, epoch, "Train")
            eval_logs = self._eval_one_epoch(epoch=epoch)
            self.write_log(eval_logs, epoch, "Validate")
            # save validation best accuracy
            if eval_logs["Acc"] > best_acc:
                best_acc = eval_logs["Acc"]
                save_name = f"{kwargs['model_name']}.pth"
                save_path = os.path.join(self.model_save_dir, save_name)
                torch.save(self.model.state_dict(), save_path)
        print(f"{time_stamp} training finished! Save weights in: {save_path}, and the best accuracy is: {best_acc}")

    @abc.abstractmethod
    def _train_one_epoch(self, *args, **kwargs) -> dict:
        """
        Train the model one epoch
        """
        pass

    @abc.abstractmethod
    def _eval_one_epoch(self, *args, **kwargs) -> dict:
        """
        Validate
        """
        pass

    def write_log(self, logs: dict, epoch_id: int, mode: str) -> None:
        """
        write the training and validation log
        :param logs:
        :param epoch_id:
        :param mode: Train or Validate
        :return:
        """
        for key in logs.keys():
            value = logs[key]
            self.logger.add_scalar(f"{mode}/{key}", value, epoch_id)

    @staticmethod
    def eval_accuracy(pred: List[np.ndarray], label: List[np.ndarray]) -> float:
        """
        Evaluate the accuracy of the prediction of the model
        :param pred: the list of contains the prediction label
        :param label: the list of contains the true label
        :return: the average accuracy
        """
        results = list()
        for tem_p, tem_l in zip(pred, label):
            """
            each tem_p should in shape [Batch, num_classes]
            each tem_l should in shape [Batch, ]
            """
            tem_p = np.argmax(tem_p, axis=1)  # [Batch, num_classes] - > [Batch, ]
            correct_num = (tem_l == tem_p).sum()
            total_num = tem_p.shape[0]
            tem_acc = correct_num / total_num
            results.append(tem_acc)
        avg_acc = np.mean(results).item()
        return avg_acc
