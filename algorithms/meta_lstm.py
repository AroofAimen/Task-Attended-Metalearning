# 
import numpy as np
import learn2learn as l2l
import torch
import torch.nn as nn
import os
import copy
from models import Learner,MetaLearner
from utils import *
from datasets import *
from .trainer import Trainer

class MetaLSTM(Trainer):
    def __init__(self, args):
        args.meta_batch_size = 1
        super(MetaLSTM, self).__init__(args)
        self.learner_w_grad  = self.base_learner
        self.learner_wo_grad = copy.deepcopy(self.learner_w_grad)

    def adapt(self,batch):
        train_input,train_target,test_input,test_target = test_split(batch,self.shots,self.ways,self.dev,self.dataset)
        # self.learner_w_grad.reset_batch_stats()
        # self.learner_wo_grad.reset_batch_stats()
        self.learner_w_grad.train()
        self.learner_wo_grad.train()
        cI = train_learner(self.learner_w_grad, self.meta_learner, train_input, train_target, self.args)
        # Train meta-learner with validation loss
        self.learner_wo_grad.transfer_params(self.learner_w_grad, cI)
        output = self.learner_wo_grad(test_input)
        loss = self.learner_wo_grad.criterion(output, test_target)
        acc  = accuracy(output, test_target)
        return loss, acc
    
    def evaluate_learner(self,batch):
        cI = self.meta_learner.metalstm.cI.data
        self.learner_w_grad.copy_flat_params(cI)
        with torch.no_grad():
            evaluation_error, evaluation_accuracy = evaluate_learner(batch, self.learner_w_grad,self.loss,self.shots,self.ways,self.dev,self.dataset)
        return evaluation_error, evaluation_accuracy 
            
    def build_metalearner(self):
        meta_learner = MetaLearner(self.args.input_size, self.args.hidden_size, self.base_learner.get_flat_params().size(0)).to(self.dev)
        meta_learner.metalstm.init_cI(self.base_learner.get_flat_params())
        return meta_learner



