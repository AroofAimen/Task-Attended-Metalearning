
import random 

class Config:
    def __init__(self):
        self.mode             = 'train'
        self.algo             = "MAML"
        self.dataset          = 'mini-imagenet'
        self.n_channels       = 3
        self.n_filters        = 32
        self.image_size       = 84
        self.n_shot           = 1
        self.n_class          = 5
        self.meta_batch_size  = 4#8
        self.input_size       = 4
        self.hidden_size      = 20
        self.meta_lr          = 1e-3
        self.base_lr          = 0.1
        self.num_iterations   = 2000
        self.adaptation_steps = 2
        self.weigh_task       = False
        self.grad_clip        = 0.25
        self.bn_momentum      = 0.95
        self.lamda            = 0.05
        self.bn_eps           = 1e-3
        self.pin_mem          = True
        self.val_freq         = 100#00
        self.resume           = False
        self.clear_logs       = True
        self.user             = "Noobie"
        self.gpu              = 1
        self.seed             = random.randint(0, 1e3)  
        self.test_size        = 100
        self.ckpt_no          = None
        self.include_top      = None
        self.memorization     = 0
        self.task_attention   = True
        self.attention_lr     = 3e-3
        self.attention_indim   = 4#5
        self.attention_nlayer  = 1
        self.attention_nhidden = 32
        self.report_ray        = False
        self.verbose           = False
        self.correlation_exp   = False
        self.ablation_case     = 4
        self.normalized_inputs = False
        