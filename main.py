
from models import * 
from parser import parse_arguments
import torch
import pprint
from algorithms import algos
import learn2learn as l2l

if __name__ == '__main__':
   config = parse_arguments()
   print("=="*50,"\n Config : \n ","=="*50)
   pprint.pprint(vars(config))
   print("\n \n ", "--"*50 ,"\n")

   config.base_learner = Learner(config.image_size, config.bn_eps, config.bn_momentum, config.n_class,n_channels=config.n_channels,n_filters=config.n_filters).to(config.dev)
   config.meta_opt     = torch.optim.Adam

   
   
   model                    = algos[config.algo](config) 
   print(model)
   # model.meta_train()

   if("train" in config.mode ):
      model.meta_train()

   elif(config.mode=="test"):
      model.meta_test()   

