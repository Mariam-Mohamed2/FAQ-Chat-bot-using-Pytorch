
import torch
import torch.nn as nn



## creating neural network with 2 hidden layers  
## we take  BOW as an input
## output layer with the number of classes (tags) 
class NeuraNet(nn.Module):
    def __init__(self,input_size,hidden_size,num_classes):
        super(NeuraNet,self).__init__()
        #super(NeuraNet,self).__init__()
        ## create 3 linear layers
        self.l1=nn.Linear(input_size,hidden_size) ##connects input layer and the first hidden layer
        self.l2=nn.Linear(hidden_size,hidden_size) ##connects first hidden layer and the second hidden layer
        self.l3=nn.Linear(hidden_size,num_classes)  ##connects second layer and the  output layer
        self.relu=nn.ReLU()
    def forward(self,x):
        ## now create the network by add activation fun(reLU)
        out=self.l1(x)
        out=self.relu(out)
        
        out=self.l2(out)
        out=self.relu(out)
        
        out=self.l3(out)
        #no activation function no soft max only cross entropy
        return out    
    print('neural')
    
    
'''
import torch
import torch.nn as nn


class NeuralNet(nn.Module):
    
    def __init__(self, input_size, hidden_size, num_classes):
        super(NeuralNet, self).__init__()
        self.l1 = nn.Linear(input_size, hidden_size) 
        self.l2 = nn.Linear(hidden_size, hidden_size) 
        self.l3 = nn.Linear(hidden_size, num_classes)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        out = self.l1(x)
        out = self.relu(out)
        out = self.l2(out)
        out = self.relu(out)
        out = self.l3(out)
        # no activation and no softmax at the end
        return out
    
'''