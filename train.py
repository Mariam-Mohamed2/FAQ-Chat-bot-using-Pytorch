import json
from Preprocessing import tokenization,stemming,Bag_of_words,remove_stop_words
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from model import NeuraNet

def acc(pred,label):
    pred = torch.round(pred.squeeze())
    return torch.sum(pred == label.squeeze()).item()

with open ('train_data.json ','r') as f:
    intents=json.load(f)
#print(intents)
# now for applying BOW we need to collect all of words in a list

all_words=[]
tags=[]
xy=[]
for intent in intents['intents']:
    #print(intent," intent")
    tag=intent['tag']
    tags.append(tag)
    for pattern in intent['patterns']:
        # tokenize each word in the sentence
        #s=remove_stop_words(pattern)
        w=tokenization(pattern)
        all_words.extend(w) #--> to be in one list
        xy.append((w,tag))
        
        
# remove punctiations
## punc=string.punctuation
#print('pun  ',punc)
## all_words=[stemming(w) for w in all_words if w not in punc]
ignore_words = ['?', '.', '!']
all_words = [stemming(w) for w in all_words if w not in ignore_words]
#print(sorted(set(all_words)))
all_words=sorted(set(all_words))
#print('now after stemming and remove punctuations')
#print(all_words)
tags=sorted(set(tags))

# create training data
x_train=[] 
y_train=[]
for(tokenized_Sen,tag) in xy:
    # X: bag of words for each pattern_sentence
    bag=Bag_of_words(tokenized_Sen,all_words)
    x_train.append(bag)
    # y: PyTorch CrossEntropyLoss needs only class labels, not one-hot
    label=tags.index(tag)
    y_train.append(label) #crossentropyloss
    
x_train=np.array(x_train)
y_train=np.array(y_train)

#hyper parameters
batch_size=20
input_size=len(all_words)
output_size=len(tags)
hidden_size=15
learning_rate=0.001
num_epochs=1000

## With PyTorch, creating a new neural network means creating a new class. 
## to iterate on the data and give more accurate training
class chat_dataset(Dataset):
    #there are 3 main functions
    def __init__(self):
        #for initilaize parameters and store data
        self.n_samples=len(x_train) #--> number of samples
        self.x_data=x_train
        self.y_data=y_train
        
        
    def __getitem__(self, index):
        #to get acccess on the item from data like df[0]
        return self.x_data[index],self.y_data[index]
    
    def __len__(self):
        return self.n_samples




dataset=chat_dataset()
train_loader=DataLoader(dataset= dataset,
                        batch_size=batch_size,
                        shuffle=True,
                        num_workers=0 #allows for parallel data loading
                        
                        )


## Create a model


#print(input_size,len(all_words))

device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model=NeuraNet(input_size,hidden_size,output_size).to(device)

## Loss and optimizers

critirian=nn.CrossEntropyLoss()
optimizer=torch.optim.Adam(model.parameters(),lr=learning_rate) #The model.parameters() method returns an iterable containing all the trainable parameters of the model 

# Train the model

for epoch in range(num_epochs):
    for (words, labels) in train_loader:
        words=words.to(device)
        labels=labels.to(dtype=torch.long).to(device)
        
        #forward
        outputs=model(words)
        loss=critirian(outputs,labels)    
         #backward and optimizer step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step() 
    if (epoch+1)%100==0:
        print (f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.10f}')
print('hiii')
print(f'final loss: {loss.item():.9f}')

data={
    "model_state": model.state_dict(),
    "input_size": input_size,
    "hidden_size": hidden_size,
    "output_size": output_size,
    "all_words": all_words,
    "tags": tags
}

FILE ="data.pth"
torch.save(data, FILE)
print(f'trainig completed, file save in {FILE}')





'''
1.loop to get intent 
2.save tags in tag list
3.and loop in patterns
4.import preprocessing functions
5.save tokenized patterns in all_words
6.remove punctation
7.stemming
8.save in tuples in xy
9.edit on all words and tags
10.



'''

'''

import torch
import torch. nn as nn -->build nearal netwirk
import torch. nn. functional as F --> activation function
from torch. optim import SGD --> stocastic gredient descent

'''


'''

import numpy as np
import random
import json

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from Preprocessing import bag_of_words, tokenize, stem
from model import NeuralNet

with open('intents.json', 'r') as f:
    intents = json.load(f)

all_words = []
tags = []
xy = []
# loop through each sentence in our intents patterns
for intent in intents['intents']:
    tag = intent['tag']
    # add to tag list
    tags.append(tag)
    for pattern in intent['patterns']:
        # tokenize each word in the sentence
        w = tokenize(pattern)
        # add to our words list
        all_words.extend(w)
        # add to xy pair
        xy.append((w, tag))

# stem and lower each word
ignore_words = ['?', '.', '!']
all_words = [stem(w) for w in all_words if w not in ignore_words]
# remove duplicates and sort
all_words = sorted(set(all_words))
tags = sorted(set(tags))

print(len(xy), "patterns")
print(len(tags), "tags:", tags)
print(len(all_words), "unique stemmed words:", all_words)

# create training data
X_train = []
y_train = []
for (pattern_sentence, tag) in xy:
    # X: bag of words for each pattern_sentence
    bag = bag_of_words(pattern_sentence, all_words)
    X_train.append(bag)
    # y: PyTorch CrossEntropyLoss needs only class labels, not one-hot
    label = tags.index(tag)
    y_train.append(label)

X_train = np.array(X_train)
y_train = np.array(y_train)

# Hyper-parameters 
num_epochs = 1000
batch_size = 8
learning_rate = 0.001
input_size = len(X_train[0])
hidden_size = 8
output_size = len(tags)
print(input_size, output_size)

class ChatDataset(Dataset):

    def __init__(self):
        self.n_samples = len(X_train)
        self.x_data = X_train
        self.y_data = y_train

    # support indexing such that dataset[i] can be used to get i-th sample
    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    # we can call len(dataset) to return the size
    def __len__(self):
        return self.n_samples

dataset = ChatDataset()
train_loader = DataLoader(dataset=dataset,
                          batch_size=batch_size,
                          shuffle=True,
                          num_workers=0)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = NeuralNet(input_size, hidden_size, output_size).to(device)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# Train the model
for epoch in range(num_epochs):
    for (words, labels) in train_loader:
        words = words.to(device)
        labels = labels.to(dtype=torch.long).to(device)
        
        # Forward pass
        outputs = model(words)
        # if y would be one-hot, we must apply
        # labels = torch.max(labels, 1)[1]
        loss = criterion(outputs, labels)
        
        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
    if (epoch+1) % 100 == 0:
        print (f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')


print(f'final loss: {loss.item():.4f}')

data = {
"model_state": model.state_dict(),
"input_size": input_size,
"hidden_size": hidden_size,
"output_size": output_size,
"all_words": all_words,
"tags": tags
}

FILE = "data.pth"
torch.save(data, FILE)

print(f'training complete. file saved to {FILE}')
'''