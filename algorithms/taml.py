
import tensorflow as tf
import numpy as np
import torch
import learn2learn as l2l
from torch import nn, optim
from models import Learner
from utils import *
from datasets import *
from .maml import MAML

def entropy(batch, learner, device):
    data, labels = batch
    data, labels = data.to(device), labels.to(device)
    predictions = learner(data)
    _entropy = nn.CrossEntropyLoss(reduction='mean')(predictions, labels)
    return _entropy


class TAML(MAML):
    def __init__(self, args):
        self.first_order = not (args.mode=="train")
        super(TAML, self).__init__(args)

    def batch_update(self,ineq_list):
        mean_loss = torch.mean(ineq_list)
        theil_index = []
        for task_loss in torch.unbind(ineq_list):
            task_theil_loss = (task_loss/mean_loss)*torch.log((task_loss/mean_loss)+1e-4)
            theil_index.append(task_theil_loss.requires_grad_(True))
        theil_index = torch.mean(torch.stack(theil_index))
        total_loss = mean_loss + self.args.lamda*theil_index
        total_loss.backward()
        self.meta_opt.step()
