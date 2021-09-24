import os
import torch
import numpy as np
import json,pickle
import matplotlib.pyplot as plt
import seaborn as sns 
import pprint
import glob
import random


def plot(history,name,log_dir):
    # print("inside plot")
    # plot_variables = ["loss","accuracy","l_ratio","acc_ratio","entropy","attention"]
    plot_variables = ["loss","accuracy"]

    sns.set_style("whitegrid")
    for plot_no,key_1 in enumerate(plot_variables):
        if(key_1 in history):
            plt.figure(figsize=(16,8))
            plt.title("{}".format(name),fontsize=18)
            # print('Data',key_1,history[key_1])
            if "train"  in history[key_1].keys() and len(history["iterations"]["train"])==len(history[key_1]["train"]):
                plt.plot(history["iterations"]["train"],history[key_1]["train"],label = "Training",color='b',linestyle='--',alpha=0.5)
            
            if "val"  in history[key_1].keys() and len(history["iterations"]["val"])==len(history[key_1]["val"]):
                plt.plot(history["iterations"]["val"],history[key_1]["val"],label = "Validation",color='coral',linewidth=2)

            if key_1+"-std" in history.keys():
            # if(history.has_key(key_1+"-std")):
                if "train"  in history[key_1+"-std"].keys():
                    plt.fill_between(history["iterations"]["train"],np.array(history[key_1]["train"])+ np.array(history[key_1+"-std"]["train"]),np.array(history[key_1]["train"])- np.array(history[key_1+"-std"]["train"]),color='b',alpha=0.25)
                # print("upper side",np.array(history[key_1]["train"])+ np.array(history[key_1+"-std"]["train"]))
                # print("lower side",np.array(history[key_1]["train"])- np.array(history[key_1+"-std"]["train"]))
                if "val"  in history[key_1+"-std"].keys():
                    plt.fill_between(history["iterations"]["val"],np.array(history[key_1]["val"])+np.array(history[key_1+"-std"]["val"]),np.array(history[key_1]["val"])-np.array(history[key_1+"-std"]["val"]),color='coral',alpha=0.25)
            
            if key_1+"-min" in history.keys():
                if "train"  in history[key_1+"-min"].keys():
                    plt.plot(history["iterations"]["train"],history[key_1+"-min"]["train"],color='k',linestyle=':',alpha=0.75)
            
            if key_1+"-max" in history.keys():
                if "train"  in history[key_1+"-max"].keys():
                    plt.plot(history["iterations"]["train"],history[key_1+"-max"]["train"],color='k',linestyle=':',alpha=0.75)
        
                
            if "test"  in history[key_1].keys():
                plt.axhline(np.mean(history[key_1]["test"]),color='teal',linestyle='--',label='Testing',linewidth=3)

    
            plt.legend(fontsize=15)
            plt.xlabel('Iterations',fontsize=15)
            plt.ylabel(' {} '.format(key_1) ,fontsize=15)
            plt.savefig("{}/{}.png".format(log_dir,key_1))
            plt.close()

def plot_bar(history,name,log_dir): 
    plot_variables       = ["max-loss-rank"] 
    plt.figure(figsize   = (16,8))
    plt.title("{}".format(name),fontsize=18)
    for plot_no,key_1 in enumerate(plot_variables):
        if(key_1 in history):
            plt.bar(history["iterations"]["train"],history[key_1]["train"],label = "Training")
        plt.legend(fontsize=15)
        plt.xlabel('Iterations',fontsize=15)
        plt.ylabel(' {} '.format(key_1) ,fontsize=15)
        plt.savefig("{}/{}.png".format(log_dir,key_1))
        plt.close()
    
def accuracy(predictions, targets):
    predictions = predictions.argmax(dim=1).view(targets.shape)
    return (predictions == targets).sum().float() / targets.size(0)

def load_history(log_dir):
    with open("{}/history.json".format(log_dir), "rb") as file:
        history = pickle.load(file)    
    return history

def save_history(history,name,log_dir):
    with open("{}/history.json".format(log_dir), "wb") as file:
        pickle.dump(history, file)    
    plot(history,name,log_dir)
    # if("max-loss-rank" in history):
    #     plot_bar(history,name,log_dir)

def save_ckpt(episode=-1, metalearner=None,attention_net=None, optim=None, save="./"):
    os.makedirs(save,exist_ok=True)
    attention_param = {} if attention_net is None else attention_net.state_dict()
    torch.save({
        'episode':     episode,
        'metalearner': metalearner.state_dict(),
        'attention_net': attention_param,
        'optim':       optim.state_dict()
    }, os.path.join(save, 'meta-learner-{}.pth.tar'.format(episode)))

def resume_ckpt(metalearner,attention_net, optim, resume, device,include_top=True,ckpt_no=None):
    if(include_top is None):
        include_top=True
    list_of_files = glob.glob(resume+'/*') # * means all if need specific format then *.csv
    if(ckpt_no is None):
        latest_file = max(list_of_files, key=os.path.getmtime)
    else:
        latest_file = resume+"/meta-learner-{}.pth.tar".format(ckpt_no)
    print("Resuming From : ", latest_file)
    ckpt = torch.load(latest_file, map_location=device)
    last_episode          = ckpt['episode']
    pretrained_state_dict = ckpt['metalearner']
    
    if(attention_net is not None):
        attention_state_dict  = ckpt['attention_net']
        attention_net.load_state_dict(attention_state_dict)
    
    if(not include_top):
        ignore_layers = ["metalstm.cI","module.model.cls.weight","module.model.cls.bias","lrs.16","lrs.17"]
        for ig_layer in ignore_layers:
            if(ig_layer in pretrained_state_dict):
                pretrained_state_dict[ig_layer]=metalearner.state_dict()[ig_layer]    
                
    metalearner.load_state_dict(pretrained_state_dict)
    optim.load_state_dict(ckpt['optim'])
    return last_episode, metalearner, optim, attention_net

def test_split(batch,shots,ways,device,dataset):
    data, labels = batch
    data, labels = data.to(device), labels.to(device)

    if("mini-imagenet" in dataset):
        adaptation_indices = np.zeros(data.size(0), dtype=bool)
        adaptation_indices[[i*(15+shots)+j for i in range(ways) for j in range(shots)]] = True
        evaluation_indices = torch.from_numpy(~adaptation_indices)
        adaptation_indices = torch.from_numpy(adaptation_indices)
        adaptation_data, adaptation_labels = data[adaptation_indices], labels[adaptation_indices]
        evaluation_data, evaluation_labels = data[evaluation_indices], labels[evaluation_indices]
    else:
        adaptation_indices = np.zeros(data.size(0), dtype=bool)
        adaptation_indices[np.arange(shots*ways) * 2] = True
        evaluation_indices = torch.from_numpy(~adaptation_indices)
        adaptation_indices = torch.from_numpy(adaptation_indices)
        adaptation_data, adaptation_labels = data[adaptation_indices], labels[adaptation_indices]
        evaluation_data, evaluation_labels = data[evaluation_indices], labels[evaluation_indices]
    return adaptation_data, adaptation_labels, evaluation_data, evaluation_labels

 

    # def index_value_separator(self,index_value):
    #         indexes = []
    #         values = []
    #         for idx, val in index_value:
    #             indexes.append(idx)
    #             values.append(val)
    #         # print (indexes)
    #         # print (values)
    #         return  indexes, values
    #         def get_activation(name):
 

    
    # def get_activation(name):
    #     def hook(model, input, output):
    #         activation[name] = output.detach()
    #     return hook


    
    # def distance(evaluation_data, evaluation_labels):
    #     meta_embedding_learner  = self.meta_learner.clone()
    #     cos             = nn.CosineSimilarity(dim=1, eps=1e-6)
    #     # sample 1 sample from each way
    #     # samples       = random.sample(evaluation_labels, self.ways)
    #     index_value     = random.sample(list(enumerate(evaluation_labels)), self.ways)
    #     indexes, labels = self.index_value_separator(index_value)
    #     samples         = []
    #     for i in indexes:
    #         samples.append(evaluation_data[i])

    #     # print('samples',samples)

    #     activation = {}
    #     sample_embeddings=[]
   
    #     mel = meta_embedding_learner()
    #     mel.pool4.register_forward_hook(get_activation('pool4'))
    #     output = mel(evaluation_data)
    #     print(activation['pool4'])
    #     for sample in samples:

    #         sample_embeddings.append(self.meta_learner._modules.get('pool4'))
        
    #     # sklearn.metrics.pairwise.euclidean_distances(sample_embeddings,sample_embeddings)   
    #     # E_dist = F.pairwise_distance(embed_x, embed_y, 2)
    #     Pair_Edistances=[]
    #     for embed_1, embed_2 in zip(sample_embeddings, sample_embeddings):
    #         Edist        = torch.nn.functional.pairwise_distance(embed_1, embed_2)
    #         Pair_Edistances.append(Edist)
    #     Task_min_Edist   = min(Pair_Edistances)
    #     Task_max_Edist   = max(Pair_Edistances)
    #     # cos_sim     = cos(embedding_1, embedding_2)
    #     # E_D         = torch.cdist(embedding_1, embedding_2, p=2.0) 
    #     return Task_min_Edist, Task_max_Edist