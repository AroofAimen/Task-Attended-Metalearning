
import numpy as np
from utils import *
from .maml import MAML
import learn2learn as l2l 

class ANIL(MAML):
    def __init__(self, args):
        self.first_order = not (args.mode=="train")
        super(ANIL, self).__init__(args)
                
    def build_metalearner(self):
        return l2l.algorithms.LightningANIL(self.base_learner.model.features,self.base_learner.model.cls,adaptation_lr=self.base_lr)


