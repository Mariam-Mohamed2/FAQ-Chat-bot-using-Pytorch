import json
import torch
from model import NeuraNet
from Preprocessing import Bag_of_words, tokenization,remove_stop_words

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

with open('test_data.json', 'r') as json_data:
    intents = json.load(json_data)

FILE = "data.pth"
data = torch.load(FILE)
correct_predictions=0
num_of_patterns=0
input_size = data["input_size"]
hidden_size = data["hidden_size"]
output_size = data["output_size"]
all_words = data['all_words']
tags = data['tags']
model_state = data["model_state"]

model = NeuraNet(input_size, hidden_size, output_size).to(device)
model.load_state_dict(model_state)
model.eval()

for intent in intents['intents']:
    tag=intent['tag']
    num_of_patterns+=len(intent['patterns'])
    # Preprocess the test sample (tokenization, bag-of-words, etc.)
    for sentence in intent['patterns']:
    #sentence = intent['pattern']
        print("sentence test ",sentence)
      #  s=remove_stop_words(sentence)
        sentence = tokenization(sentence)
        X = Bag_of_words(sentence, all_words)
        X = X.reshape(1, X.shape[0])
        X = torch.from_numpy(X).to(device)
    
        # Convert the preprocessed test sample to a PyTorch tensor
        # Move the tensor to the appropriate device (GPU or CPU)

        # Use the trained model to predict the output for the test sample
        output = model(X)
        _, predicted = torch.max(output, dim=1)
        c=predicted.item()
        predicted_tag = tags[c]
        
        #Compare the predicted tag with the actual tag

        print("predictet ",predicted_tag," actual ",tag)
        if predicted_tag == tag:
            correct_predictions += 1
        #else :
           # print("Sentence ",sentence," acutal tag ",tag," predicted ",predicted_tag)
        

print(num_of_patterns)
# Calculate the test accuracy
test_accuracy = (correct_predictions / num_of_patterns) * 100
#print(tags)
print(f'Test Accuracy: {test_accuracy:.2f}%')