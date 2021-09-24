import argparse, sys
import json
import numpy as np
import torch
import os
import random
import tensorflow as tf
import shutil
from algorithms import *
from utils import save_history
from config import Config
from models import * 
from algorithms import algos
import os
from datasets import *

def parse_arguments():
    config = Config()
    parser      = argparse.ArgumentParser()
    parser.add_argument('--arg', type=json.loads)
    exp_config  = parser.parse_args().arg
    print("\n \n ", "--"*50 ,"\n")
    for param in exp_config:
        if(hasattr(config,param)):
            print(" Overriding Parameter : {} \t Initial Value : {} \t New Value : {}".format(param,config.__dict__[param],exp_config[param]))
            if(param=="algo" and exp_config[param] not in algos):
                print(" Invalid Algorithm : {} \n Should be one of {} ".format(exp_config[param],algos))
                exit(0)
            setattr(config,param,exp_config[param])
        else:
            print(" Unknown Parameter : {} ".format(param))

    os.environ["CUDA_VISIBLE_DEVICES"]="{}".format(config.gpu)
    config.dev  = torch.device("cuda:{}".format(0)) 
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)

    host_name = os.uname()[1]
    

    

    EXP_DIR  = "./../exp/{}/{}/{}_F/{}WAY_{}SHOT/{}/Attention-{}".format(config.user,config.dataset,config.n_filters,config.n_class,config.n_shot,config.algo,config.task_attention)
    DATA_DIR = "~/data/{}/".format(config.dataset)

    
    LOG_DIR  = os.path.join(EXP_DIR,'logs')
    CKPT_DIR = os.path.join(EXP_DIR,'ckpt')

    os.makedirs(EXP_DIR,exist_ok=True)
    os.makedirs(DATA_DIR,exist_ok=True)
    os.makedirs(LOG_DIR,exist_ok=True)
    os.makedirs(CKPT_DIR,exist_ok=True)

    config.DATA_DIR  = DATA_DIR
    config.LOG_DIR   = LOG_DIR
    config.CKPT_DIR  = CKPT_DIR
    config.EXP_DIR  = EXP_DIR

    get_tasksets(
                config.dataset,
                root=config.DATA_DIR,
                )
    return config
