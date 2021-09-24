
import numpy as np
from utils import *
from .maml import MAML
import learn2learn as l2l 

class MetaSGD(MAML):
    def __init__(self, args):
        # self.first_order = not (args.mode=="train")
        self.first_order = True
        super(MetaSGD, self).__init__(args)
                
    def build_metalearner(self):
        return l2l.algorithms.MetaSGD(self.base_learner, lr=self.base_lr, first_order=self.first_order)


