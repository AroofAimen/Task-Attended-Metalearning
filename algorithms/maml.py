
import numpy as np
from utils import *
from .trainer import Trainer
import learn2learn as l2l 
import torch

class MAML(Trainer):
    def __init__(self, args):
        # self.first_order = not (args.mode=="train")
        self.first_order = True
        super(MAML, self).__init__(args)
            
    def adapt(self,batch):
        learner = self.meta_learner.clone()
        return fast_adapt(batch,learner,self.loss,self.adaptation_steps,self.shots,self.ways,self.dev,self.dataset)

    def evaluate_learner(self,batch):
        learner = self.meta_learner.clone()
        with torch.no_grad():
            evaluation_error, evaluation_accuracy = evaluate_learner(batch,learner,self.loss,self.shots,self.ways,self.dev,self.dataset)
        return evaluation_error, evaluation_accuracy 
    
    def evaluate_attention(self,attention_learner):
        for _ in range(self.meta_batch_size):
            batch = self.tasksets.train.sample()
            evaluation_loss_attn,evaluation_acc_attn = fast_adapt(batch,attention_learner,self.loss,self.adaptation_steps,self.shots,self.ways,self.dev,self.dataset)
            evaluation_loss_attn.backward(retain_graph=True)
        
        for p in self.attention_network.parameters():
            p.grad = p.grad/self.meta_batch_size
        return evaluation_loss_attn,evaluation_acc_attn

    def build_metalearner(self):
        return l2l.algorithms.MAML(self.base_learner, lr=self.base_lr, first_order=self.first_order)


