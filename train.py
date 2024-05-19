import json
import torch
import torch.nn as nn
from torch.utils.data import Dataset,DataLoader
import numpy as np

from Preprocessing import tokenization,Bag_of_words,stemming
from model import NeuraNet


with open('train_data.json','r') as f:
    intents=json.load(f)
#print(intents)

all_words=[]
tags=[] # will include all tags in json file 
patterns_and_tags=[]
# json file is a dictionary called intents include tag and patterns and responses lists 
for intent in intents['intents']:
    tag=intent['tag']
    tags.append(tag)
    for pattern in intent['patterns']:
        # pattern is questions which user asks so this what we tokenize
        w=tokenization(pattern)
        all_words.extend(w) 
        patterns_and_tags.append((w,tag))
        
# Remove punctuation
remove_punctuation=['.','?','!',',']
all_words=[stemming(word)for word in all_words if word not in remove_punctuation]

#Sort and Remove Duplicate Words 
all_words=sorted(set(all_words)) 
#Sort and Remove Duplicate tags
tags=sorted(set(tags)) 
'''print(tags)
print(all_words)'''

# Create training data

x_train=[]
y_train=[]
for (pattern,tag) in patterns_and_tags:
    bag=Bag_of_words(pattern,all_words)
    x_train.append(bag)
    label=tags.index(tag)
    y_train.append(label)
x_train=np.array(x_train)
y_train=np.array(y_train)
'''print(len(tags))
print(len(all_words))
print(x_train)
print(y_train)'''

# Manipulate dataset
class ChatDataset(Dataset):
    def __init__(self):
        self.n_samples=len(x_train)
        self.x_data=x_train
        self.y_data=y_train
    # support indexing such that dataset[i] can be used to get specific i-th sample
    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    #to return number of training examples
    def __len__(self):
        return self.n_samples
    
#Parameters in neural network:
num_epochs = 1500  # total number of epochs fed to neural network including all training examples to improve model accuracy.
batch_size=50    # the number of training examples are fed to one epoch.
learning_rate=.001  
input_size=len(all_words)
hidden_size=30
output_size=len(tags)  #represent number of classes


print(input_size," ",len(x_train[0]))
print(output_size," ",len(tags))


dataset=ChatDataset()
train_loader = DataLoader(dataset=dataset,batch_size=batch_size,shuffle=True,num_workers=0)  
#num_workers=0 ---> it means that the data will be loaded in the main process, without using additional subprocesses
#shuffle --> to take training examples randomly

# Device =CPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = NeuraNet(input_size, hidden_size, output_size).to(device)

#Loss and Optimizer 
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(),lr=learning_rate)

#to compute training accuracy
total_samples=0
total_correct=0
# Train the model 
for epoch in range(num_epochs):
    for (words, labels) in train_loader:
        words = words.to(device)
        labels = labels.to(dtype=torch.long).to(device)
        #   Forward propagation
        outputs=model(words)

        #    Compute loss 
        loss=criterion(outputs,labels)
        # Get predictions
        _, predicted = torch.max(outputs, dim=1)
        total_samples += labels.size(0)
        total_correct += (predicted == labels).sum().item()


        #    Backward and Update parameters through Optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

# Print loss foe each 100 epochs 
    if (epoch+1) % 100 == 0:
        #print (f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.8f}')
        accuracy = total_correct / total_samples
        print (f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.6f}, Accuracy: {accuracy:.6f}')
        
        
        


print(f'final loss: {loss.item():.4f}')
# Save model
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
    