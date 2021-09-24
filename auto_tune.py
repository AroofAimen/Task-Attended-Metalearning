
from models import * 
from functools import partial
from parser import parse_arguments
import torch
import pprint
from algorithms import algos
import learn2learn as l2l
import pprint
from ray import tune
from ray.tune import CLIReporter
from ray.tune.schedulers import ASHAScheduler,PopulationBasedTraining
import logging
import sys
import numpy as np
import os

def run(config,checkpoint_dir=None,args = None):
    args.meta_lr          = config["meta_lr"]
    args.base_lr          = config["base_lr"] if "base_lr" in config else args.base_lr
    args.attention_lr     = config["attention_lr"] if "attention_lr" in config else args.attention_lr
    # args.meta_batch_size  = int(config["meta_batch_size"])
    # args.adaptation_steps = int(config["adaptation_steps"])
    # args.attention_nlayer = int(config["attention_nlayer"])
    args.EXP_DIR          = config["exp_dir"]
    args.LOG_DIR          = os.path.join(args.EXP_DIR,'logs')
    args.CKPT_DIR         = os.path.join(args.EXP_DIR,'ckpt')

    os.makedirs(args.LOG_DIR,exist_ok=True)
    os.makedirs(args.CKPT_DIR,exist_ok=True)

    model = algos[args.algo](args)    
    model.meta_train()
    
def main(num_samples=10, max_num_epochs=10, gpus_per_trial=2,args=None,MODEL=None):
    config = {
        # "adaptation_steps" : tune.sample_from(lambda _: 2 ** np.random.randint(1, 4)),
        "meta_lr"          : tune.loguniform(1e-4, 1e-2),
        # "attention_lr"     : tune.loguniform(1e-4, 1e-2),
        # "meta_batch_size"  : tune.choice([4,8,16,32]),
        # "attention_nlayer" : tune.choice([1,2,4]),
        "exp_dir"       : "./"
    }
    scheduler = ASHAScheduler(
        metric="accuracy",
        mode="max",
        max_t=max_num_epochs,
        grace_period=4,
        reduction_factor=2
    )
    if(args.task_attention == 1):
        config["attention_lr"] = tune.loguniform(1e-4, 1e-2)
        
        
    if(args.algo in ["MAML","MetaSGD","TAML"]):
        config["base_lr"] = tune.loguniform(1e-2, 5e-1)
        
    if(args.algo in ["TAML"]):
        config["lamda"] = tune.loguniform(1e-2, 1)
    
    reporter = CLIReporter(metric_columns=["loss", "accuracy", "training_iteration"])
    result   = tune.run(
                    partial(MODEL, args=args),
                    resources_per_trial={"cpu": 8, "gpu": gpus_per_trial},
                    config           =config,
                    num_samples      =num_samples,
                    local_dir        =args.EXP_DIR,
                    scheduler        =scheduler,
                    progress_reporter=reporter,
                    name             ="HyperTune",
                    resume           = args.resume 
                    )

    best_trial = result.get_best_trial("accuracy", "max", "last")
    print("Best trial config: {}".format(best_trial.config))
    print("Best trial final validation loss: {}".format(best_trial.last_result["loss"]))
    print("Best trial final validation accuracy: {}".format(best_trial.last_result["accuracy"]))

if __name__ == "__main__":
    config = parse_arguments()
    print("=="*50,"\n Config : \n ","=="*50)
    pprint.pprint(vars(config))
    print("\n \n ", "--"*50 ,"\n")
    
    config.base_learner = Learner(config.image_size, config.bn_eps, config.bn_momentum, config.n_class,n_channels=config.n_channels,n_filters=config.n_filters).to(config.dev)
    config.meta_opt     = torch.optim.Adam
        
    main(num_samples=30, max_num_epochs=50, gpus_per_trial=1.0,args=config,MODEL=run)
