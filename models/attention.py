import torch.nn as nn 


class TaskAttention(nn.Module):
    def __init__(self, input_dim=4, output_dim=4, n_hidden=32, n_layer=2,activation=nn.ReLU):#nn.Tanh
        super(TaskAttention, self).__init__()
        self.input_dim  = input_dim
        self.output_dim = output_dim
        self.n_layer    = n_layer
        self.n_hidden   = n_hidden
        self.activation = activation
        self.model      = self.build_model()
    
    def build_model(self):
        model = nn.Sequential()
        model.add_module("1x1_conv",nn.Conv1d(self.input_dim,1,1))
        # model.add_module(nn.Conv1d(in_channels=1, out_channels=1, kernel_size=n_channels, stride=1))
        
        model.add_module("FC_input",nn.Linear(self.output_dim,self.n_hidden))
   
        model.add_module("act_0",self.activation())
        for l in range(self.n_layer-1):
            model.add_module("FC_{}".format(l+1),nn.Linear(self.n_hidden,self.n_hidden))
            model.add_module("act_{}".format(l+1),self.activation())
        model.add_module("FC_output",nn.Linear(self.n_hidden,self.output_dim))
        model.add_module("softmax",nn.Softmax(dim=-1))
        return model



    def forward(self, x):
        return self.model(x)

