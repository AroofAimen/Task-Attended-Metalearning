import torch.nn as nn 
import learn2learn as l2l
import pprint
import torch 
from datasets import get_tasksets
from utils.generic_utils import *
from utils.model_utils import *
from models import *
import numpy as np 
import higher
from numpy import linalg as LA
from torch import autograd
from ray import tune
import os
import sys
from scipy.stats import entropy
from sklearn.metrics.pairwise import cosine_similarity

class Trainer():
    def __init__(self, args):
        super(Trainer, self).__init__()
        self.args    =   args
        self.ways    =   args.n_class
        self.shots   =   args.n_shot
        self.base_lr =   args.base_lr
        self.meta_lr =   args.meta_lr
        self.device  =   args.dev
        self.dataset =   args.dataset
        self.meta_batch_size =   args.meta_batch_size
        self.test_size       =   args.meta_batch_size if args.test_size is None  else args.test_size
        self.adaptation_steps=   args.adaptation_steps
        self.num_iterations  =   args.num_iterations
        self.meta_lr         =   args.meta_lr
        self.base_lr         =   args.base_lr
        self.data_dir        =   args.DATA_DIR
        self.log_dir         =   args.LOG_DIR
        self.ckpt_dir        =   args.CKPT_DIR
        self.dev             =   args.dev
        self.last_eps        =   0
        self.iteration       =   0
        self.val_freq        =   args.val_freq
        self.n_samples       =   15+self.shots if "mini-imagenet" in self.dataset else 2* self.shots
        self.memorization    =   args.memorization
        self.tasksets        =   get_tasksets(
                                            self.dataset,
                                            train_ways=self.ways,
                                            train_samples=self.n_samples,
                                            test_ways=self.ways,
                                            test_samples=self.n_samples,
                                            root=self.data_dir,
                                            memorization=self.memorization
                                            )
        self.base_learner = args.base_learner
        self.loss         = type(self.base_learner.criterion)()
        self.meta_learner = self.build_metalearner()
        self.meta_opt     = self.args.meta_opt(self.meta_learner.parameters(),self.meta_lr)
        self.history      = {
                        "iterations":{"train":[],"val":[],"test":[]},
                        "loss":{"train":[],"val":[],"test":[]},
                        "accuracy":{"train":[],"val":[],"test":[]},
                        "loss-std":{"train":[],"val":[],"test":[]},
                        "accuracy-std":{"train":[],"val":[],"test":[]},
                        "l_ratio":{"train":[],"val":[],"test":[]},
                        "l_ratio-std":{"train":[],"val":[],"test":[]},
                        "acc_ratio":{"train":[],"val":[],"test":[]},
                        "acc_ratio-std":{"train":[],"val":[],"test":[]},
                        "entropy":{"train":[]},
                        "attention":{"train":[]},
                        "attention-min":{"train":[]},
                        "attention-max":{"train":[]},
                        "attention-std":{"train":[]},
                        # "attention-vector":{"train":[]},
                        # "task-losses":{"train":[]},
                        # "max-loss-rank":{"train":[]}
                        # "task-gradients":{"train":[]}
                        "attention-input":{"train":[]},
                        "attention-output":{"train":[]}
                        }
   
    #    self.history      = {"iterations":{"train":[],"val":[],"test":[]},
    #                     "loss":{"train":[],"val":[],"test":[]},
    #                     "accuracy":{"train":[],"val":[],"test":[]},
    #                     "loss-std":{"train":[],"val":[],"test":[]},
    #                     "accuracy-std":{"train":[],"val":[],"test":[]},
    #                     "l_ratio":{"train":[],"val":[],"test":[]},
    #                     "l_ratio-std":{"train":[],"val":[],"test":[]},
    #                     "acc_ratio":{"train":[],"val":[],"test":[]},
    #                     "acc_ratio-std":{"train":[],"val":[],"test":[]},
    #                     "entropy":{"train":[]},
    #                     "attention":{"train":[]},
    #                     "attention-min":{"train":[]},
    #                     "attention-max":{"train":[]},
    #                     "attention-std":{"train":[]},
    #                     "attention-vector":{"train":[]},
    #                     "task-losses":{"train":[]},
    #                     "task-gradients":{"train":[]},
    #                     "batch-csd":{"train":[]}
    #                     }

        self.attention_network= None
        if(args.task_attention):
            self.attention_network = TaskAttention(self.args.attention_indim,self.meta_batch_size,self.args.attention_nhidden,self.args.attention_nlayer).to(self.dev)
            self.attention_opt     = args.meta_opt(self.attention_network.parameters(),self.args.attention_lr)
            
        if args.resume:
            self.load()
        
        with open("{}/config.txt".format(self.log_dir), "wt") as file:
            pprint.pprint(vars(self.args), stream=file)
    
    def log(self,mode,iteration,loss,acc,l_ratio=-1,acc_ratio=-1):
        self.history["iterations"][mode].append(iteration)
        self.history["loss"][mode].append(np.mean(loss))
        self.history["loss-std"][mode].append(np.std(loss))
        self.history["accuracy"][mode].append(np.mean(acc))
        self.history["accuracy-std"][mode].append(np.std(acc))
        self.history["l_ratio"][mode].append(np.mean(l_ratio))
        self.history["l_ratio-std"][mode].append(np.std(l_ratio))
        self.history["acc_ratio"][mode].append(np.mean(acc_ratio))
        self.history["acc_ratio-std"][mode].append(np.std(acc_ratio))
       
        
        if("train" not in mode):
            print("=="*50)
        print("|MODE : {:^10s}|ITER : {:^10d}|LOSS : {:^10f}|ACC : {:^10f} |".format(mode,iteration,np.mean(loss),np.mean(acc))) 
        print("--"*50)
    
    def save(self):
        save_history(self.history, type(self).__name__,self.log_dir)
        save_ckpt(self.iteration,self.meta_learner,self.attention_network,self.meta_opt,self.ckpt_dir)
        
    def load(self):
        self.history = load_history(self.log_dir)        
        self.last_eps, self.meta_learner, self.meta_opt, self.attention_network = resume_ckpt(metalearner=self.meta_learner,attention_net=self.attention_network, optim=self.meta_opt, resume=self.ckpt_dir,device=self.dev,include_top=self.args.include_top,ckpt_no=self.args.ckpt_no)
        self.iteration = self.last_eps

    def meta_train(self):
        for self.iteration in range(self.last_eps+1,self.last_eps+self.num_iterations+1):
            prior_batch_acc        = []
            prior_batch_loss       = []
            batch_accuracy         = []
            batch_loss             = []
            batch_data_labels      = []
         

            task_losses            = None
            task_gradients         = None
            task_accuracies        = None   
            
  
            
            self.meta_opt.zero_grad()
            if(self.args.task_attention):
                self.attention_opt.zero_grad()

            for task in range(self.meta_batch_size):
                batch   = self.tasksets.train.sample()
                # if (self.iteration%self.analysis_freq==0):
                if ((self.iteration-1)%600==0):
                    batch_data_labels.append(batch)
                    # print("batch_data_labels",batch_data_labels)


                prior_eval_error, prior_eval_acc = self.evaluate_learner(batch)
                prior_batch_acc.append(prior_eval_acc.item())
                prior_batch_loss.append(prior_eval_error.item())
                
                evaluation_error, evaluation_accuracy = self.adapt(batch)
                # self.distance(evaluation_data, evaluation_labels)
                
                task_losses     = evaluation_error if task_losses is None else torch.cat((task_losses.reshape(task), evaluation_error.reshape(1)), 0)
                task_accuracies = evaluation_accuracy if task_accuracies is None else torch.cat((task_accuracies.reshape(task), evaluation_accuracy.reshape(1)), 0)
               
                batch_accuracy.append(evaluation_accuracy.item())
                batch_loss.append(evaluation_error.item())

            l_ratio   = np.divide(np.array(batch_loss),np.array(prior_batch_loss)+1e-5)
            acc_ratio = np.divide(np.array(batch_accuracy),np.array(prior_batch_acc)+1e-5)

            if(self.iteration%self.val_freq==0):
                self.meta_validate()
        
            if(self.args.task_attention):
                # if(self.args.correlation_exp==True):
                #     pass
                # else:
                self.attention_batch_update(task_losses,task_gradients,l_ratio,acc_ratio,task_accuracies)
            else:
                self.batch_update(task_losses)
            self.log("train",self.iteration,batch_loss,batch_accuracy,l_ratio,acc_ratio)
               
        self.meta_test()
        self.save()

    def meta_validate(self):
        prior_batch_acc        = []
        prior_batch_loss       = []
        batch_accuracy         = []
        batch_loss             = []
        for task in range(self.test_size):
            batch   = self.tasksets.test.sample()                  
            prior_eval_error, prior_eval_acc = self.evaluate_learner(batch)
            prior_batch_acc.append(prior_eval_acc.item())
            prior_batch_loss.append(prior_eval_error.item())                
            evaluation_error, evaluation_accuracy = self.adapt(batch)
            batch_accuracy.append(evaluation_accuracy.item())
            batch_loss.append(evaluation_error.item())

        validation_loss, validation_accuracy = np.mean(batch_loss),np.mean(batch_accuracy)
        l_ratio   = np.divide(np.array(batch_loss),np.array(prior_batch_loss)+1e-5)
        acc_ratio = np.divide(np.array(batch_accuracy),np.array(prior_batch_acc)+1e-5)
                    
        if(self.args.report_ray):
            with tune.checkpoint_dir(self.iteration) as checkpoint_dir:
                path = os.path.join(self.args.CKPT_DIR,checkpoint_dir, "checkpoint")
                tune.report(loss=validation_loss, accuracy=validation_accuracy,iteration=self.iteration)
                sys.stdout.flush()

        


                  
        self.log("val",self.iteration,batch_loss,batch_accuracy,l_ratio,acc_ratio)
        self.save()


    def meta_test(self,load_best_ckpt=True):
      
        # self.load()#c
        if(self.args.ckpt_no==None):
        # if(load_best_ckpt):
            best_ckpt_no   = self.history["iterations"]["val"][np.argmax(self.history["accuracy"]["val"])]
            self.args.ckpt_no = best_ckpt_no
            self.load()
        else:
            self.load()

        
        batch_accuracy = []
        batch_loss     = []
        for task in range(self.test_size):
            batch   = self.tasksets.test.sample()
            evaluation_error, evaluation_accuracy = self.adapt(batch)
            batch_accuracy.append(evaluation_accuracy.item())
            batch_loss.append(evaluation_error.item())

        test_loss, test_accuracy = np.mean(batch_loss),np.mean(batch_accuracy)
        self.log("test",self.iteration,batch_loss,batch_accuracy)

        mean_test_acc = np.mean(batch_accuracy)
        conf_test_acc = 1.96*np.std(batch_accuracy)/np.sqrt(self.test_size)
        print(" Test acc : {} +- {}".format(mean_test_acc,conf_test_acc))
        return mean_test_acc,conf_test_acc

    
    def batch_update(self,task_losses):
        net_loss = torch.mean(task_losses)
        net_loss.backward(retain_graph=True)
        self.meta_opt.step()    

    def preprocess_attention_input(self,task_losses,l_ratio,acc_ratio,task_accuracies):   
        
        # print('acc_ratio',acc_ratio)
        # print('l_ratio',l_ratio)
        # print('task_losses',task_losses)
        # print('task_accuracies',task_accuracies)
        
        grad_norm_list = []
        grad_list      = []
        for task_loss in torch.unsqueeze(task_losses,1):
            grads = autograd.grad(task_loss, self.meta_learner.parameters(),allow_unused=True,retain_graph=True)
            # print('grads',grads)
            # grads = np.array([g.cpu().detach().numpy().flatten() for g in grads]).flatten()
            # print("Grad Dim ",grads.shape)
            grad_list.append(grads)
            grad_norm_list.append(norm(grads))

        acc_ratio                 = torch.tensor(acc_ratio).float()
        acc_ratio                 = acc_ratio.cpu().detach().numpy().reshape([1,self.meta_batch_size])
        l_ratio                   = torch.tensor(l_ratio).float()
        l_ratio                   = l_ratio.cpu().detach().numpy().reshape([1,self.meta_batch_size])
        if (self.args.normalized_inputs==True):
            grad_norm_list = [10*x for x in grad_norm_list]
            task_losses=100*task_losses
            # print("grad_norm_list",grad_norm_list)
            # print("task_losses",task_losses)
            

        grad_norm_list            = torch.tensor(grad_norm_list)    
        grad_norm_list            = grad_norm_list.cpu().detach().numpy().reshape([1,self.meta_batch_size])
        task_loss_list            = task_losses.cpu().detach().numpy().reshape([1,self.meta_batch_size])
        task_accuracy_list        = task_accuracies.cpu().detach().numpy().reshape([1,self.meta_batch_size])

        # grad_norm_task_loss_list  = np.expand_dims(np.concatenate((grad_norm_list_, task_loss_list_,l_ratio_,acc_ratio_),axis=0),axis=0)
        
        if (self.args.ablation_case == 5): 
            attention_input  = np.expand_dims(np.concatenate((grad_norm_list, task_loss_list,task_accuracy_list,l_ratio,acc_ratio),axis=0),axis=0)
        elif (self.args.ablation_case == 4):
            attention_input  = np.expand_dims(np.concatenate((grad_norm_list, task_loss_list,task_accuracy_list,l_ratio),axis=0),axis=0)
        elif (self.args.ablation_case == 3):
            attention_input  = np.expand_dims(np.concatenate((grad_norm_list, task_loss_list,l_ratio),axis=0),axis=0)
        elif (self.args.ablation_case == 2):
            attention_input  = np.expand_dims(np.concatenate((grad_norm_list, task_loss_list,task_accuracy_list),axis=0),axis=0)
        elif (self.args.ablation_case == 1):
            attention_input  = np.expand_dims(np.concatenate((grad_norm_list, l_ratio,task_accuracy_list),axis=0),axis=0)
        elif (self.args.ablation_case == 0):
            attention_input  = np.expand_dims(np.concatenate((task_loss_list,task_accuracy_list,l_ratio),axis=0),axis=0)
        
        # print(attention_input.shape)
        return torch.tensor(attention_input).to(self.dev)
        
    def attention_batch_update(self,task_losses,task_gradients,l_ratio,acc_ratio,task_accuracies): 
        
        attention_learner        = self.meta_learner.clone()

        # print('l_ratio,acc_ratio',l_ratio,acc_ratio)
        # attention_input,task_gradients   = self.preprocess_attention_input(task_losses,l_ratio,acc_ratio,task_accuracies)
        attention_input                    = self.preprocess_attention_input(task_losses,l_ratio,acc_ratio,task_accuracies)
        # attention_input                  = self.preprocess_attention_input(task_losses,l_ratio,acc_ratio)
        attention_vector                   = self.attention_network(attention_input).reshape(self.meta_batch_size)
        # print("attention_vector",attention_vector)
        # print("task_losses",task_losses)
        # print("task_accuracies",task_accuracies)
        # print('task_gradients',task_gradients)
        # print('task_gradients shape',len(task_gradients))
        
        attention_mean          = np.mean(attention_vector.cpu().detach().numpy())
        attention_std           = np.std(attention_vector.cpu().detach().numpy())
        attention_min           = np.min(attention_vector.cpu().detach().numpy())
        attention_max           = np.max(attention_vector.cpu().detach().numpy())
        
    
        
        self.history["attention"]["train"].append(attention_mean)
        self.history["attention-std"]["train"].append(attention_std)
        self.history["attention-min"]["train"].append(attention_min)
        self.history["attention-max"]["train"].append(attention_max)
        self.history["attention-input"]["train"].append(attention_input.cpu().detach().numpy())
        self.history["attention-output"]["train"].append(attention_vector.cpu().detach().numpy())
        
 
        batch_entropy = entropy(attention_vector.cpu().detach().numpy(), base=2)
        # print("batch_entropy",batch_entropy)
        self.history["entropy"]["train"].append(batch_entropy)

        for (task_loss,task_weight) in zip(torch.unsqueeze(task_losses,1),torch.unsqueeze(attention_vector,1)):
            gradient = autograd.grad(task_loss,self.meta_learner.parameters(),retain_graph=True)
            for p,g in zip(self.meta_learner.parameters(),gradient):
                if(p.grad is not None):
                    p.grad = p.grad+g.mul(torch.ones_like(g).mul(task_weight))
                else:
                    p.grad = g.mul(torch.ones_like(g).mul(task_weight))
            for p,g in zip(attention_learner.parameters(),gradient):
                if(p.grad is not None):
                    p.grad = p.grad+ g.mul(torch.ones_like(g).mul(task_weight))
                else:
                    p.grad = g.mul(torch.ones_like(g).mul(task_weight))

        if("train" in self.args.mode):
            if(self.args.correlation_exp):
                pass
            else:
                attention_learner = l2l.algorithms.maml_update(attention_learner,self.meta_lr,[g.grad for g in attention_learner.parameters()])       
                self.evaluate_attention(attention_learner)
                self.attention_opt.step()
        if(self.args.correlation_exp):
            pass
            # print("no opt update")
        else:
            self.meta_opt.step()
   


        
    def build_metalearner(self):
        raise NotImplementedError