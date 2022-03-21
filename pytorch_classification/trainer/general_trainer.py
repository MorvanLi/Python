# author_='bo.li';
# date: 3/21/22 11:01 AM

import numpy as np
import torch
import sys
from tqdm import tqdm
from .base_trainer import BasicTrainer


class CNNTrainer(BasicTrainer):

    def _train_one_epoch(self, *args, **kwargs) -> dict:
        self.model.train()
        p_bar = tqdm(self.training_loader, file=sys.stdout)
        pred_list, label_list, loss_cls_list = [], [], []
        for inp_ts, lb_cls_ts in p_bar:
            # move to cpu or gpu
            inp_ts = inp_ts.to(self.device)
            lb_cls_ts = lb_cls_ts.to(self.device)

            # forward
            self.optimizer.zero_grad()
            out_ts = self.model(inp_ts)
            cls_loss = self.criterions(out_ts, lb_cls_ts)
            loss = cls_loss
            loss.backward()
            self.optimizer.step()

            # record things
            cls_loss = cls_loss.item()
            p_bar.set_description(f"Train|Epoch: {kwargs['epoch']}|Classify Loss: {cls_loss}")
            pred_list.append(out_ts.detach().cpu().numpy())
            label_list.append(lb_cls_ts.detach().cpu().numpy())
            loss_cls_list.append(cls_loss)

        avg_acc = self.eval_accuracy(pred_list, label_list)
        avg_cls_loss = np.mean(loss_cls_list)
        log_dict = {"Acc": avg_acc, "Loss_cls": avg_cls_loss}
        return log_dict

    def _eval_one_epoch(self, *args, **kwargs) -> dict:
        with torch.no_grad():
            self.model.eval()
            p_bar = tqdm(self.validate_loader, file=sys.stdout)
            pred_list, label_list, loss_cls_list = [], [], []
            for inp_ts, lb_cls_ts in p_bar:
                inp_ts = inp_ts.to(self.device)
                lb_cls_ts = lb_cls_ts.to(self.device)
                # forward
                out_ts = self.model(inp_ts)
                cls_loss = self.criterions(out_ts, lb_cls_ts)

                cls_loss = cls_loss.item()

                p_bar.set_description(f"Validate|Epoch: {kwargs['epoch']}|Classify Loss: {cls_loss}")
                pred_list.append(out_ts.detach().cpu().numpy())
                label_list.append(lb_cls_ts.detach().cpu().numpy())
                loss_cls_list.append(cls_loss)

            avg_acc = self.eval_accuracy(pred_list, label_list)
            avg_cls_loss = np.mean(loss_cls_list)
            print(f"Epoch {kwargs['epoch']} accuracy is: {avg_acc}")
            log_dict = {"Acc": avg_acc, "Loss_cls": avg_cls_loss}
        return log_dict
