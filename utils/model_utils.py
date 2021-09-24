
import numpy as np
import torch
import learn2learn as l2l
from torch import nn, optim
from models import Learner
from utils import *
from datasets import *


# def flat_params(vector):
#     return torch.cat([p.view(-1) for p in vector], 0)
# def flatten(t):
#     t = t.reshape(1, -1)
#     t = t.squeeze()
#     return t

def zero_grad(model):
    for p in model.parameters():
        p.grad = None
        
def norm(vector):
    total_norm = 0
    for p in vector:
        param_norm = p.norm(2)
        total_norm += param_norm.item() ** 2
    total_norm = total_norm ** (1. / 2)
    return total_norm
        
def grad_norm(model):
    total_norm = 0
    for p in model.parameters():
        param_norm = p.grad.data.norm(2)
        total_norm += param_norm.item() ** 2
    total_norm = total_norm ** (1. / 2)
    return total_norm
    
def preprocess_grad_loss(x):
    p = 10
    indicator = (x.abs() >= np.exp(-p)).to(torch.float32)
    # preproc1
    x_proc1 = indicator * torch.log(x.abs() + 1e-8) / p + (1 - indicator) * -1
    # preproc2
    x_proc2 = indicator * torch.sign(x) + (1 - indicator) * np.exp(p) * x
    return torch.stack((x_proc1, x_proc2), 1)


def evaluate_learner(batch, learner, loss, shots, ways, device, dataset):
    adaptation_data, adaptation_labels, evaluation_data, evaluation_labels = test_split(batch,shots,ways,device,dataset)
    predictions =  learner(evaluation_data)
    valid_error =  loss(predictions, evaluation_labels)
    valid_error /= len(evaluation_data)
    valid_accuracy = accuracy(predictions, evaluation_labels)
    # if self.compute_dist == TRUE:
    #     distance = distance(evaluation_data, evaluation_labels)
    return valid_error, valid_accuracy

def fast_adapt(batch, learner, loss, adaptation_steps, shots, ways, device, dataset):
    adaptation_data, adaptation_labels, evaluation_data, evaluation_labels = test_split(batch,shots,ways,device,dataset)
    for step in range(adaptation_steps):
        train_error = loss(learner(adaptation_data), adaptation_labels)
        train_error /= len(adaptation_data)
        learner.adapt(train_error)
    predictions = learner(evaluation_data)
    valid_error = loss(predictions, evaluation_labels)
    valid_error /= len(evaluation_data)
    valid_accuracy = accuracy(predictions, evaluation_labels)
    return valid_error, valid_accuracy

def train_learner(learner_w_grad, metalearner, train_input, train_target, args):
    cI = metalearner.metalstm.cI.data
    hs = [None]
    for adaptation_step in range(args.adaptation_steps):
        learner_w_grad.copy_flat_params(cI)
        output = learner_w_grad(train_input)
        loss = learner_w_grad.criterion(output, train_target)
        acc = accuracy(output, train_target)
        learner_w_grad.zero_grad()
        loss.backward()
        grad = torch.cat([p.grad.data.reshape(-1) for p in learner_w_grad.parameters()], 0)
        grad_prep = preprocess_grad_loss(grad)  # [n_learner_params, 2]
        loss_prep = preprocess_grad_loss(loss.data.unsqueeze(0)) # [1, 2]
        metalearner_input = [loss_prep, grad_prep, grad.unsqueeze(1)]
        cI, h = metalearner(metalearner_input, hs[-1])
        hs.append(h)
    return cI
