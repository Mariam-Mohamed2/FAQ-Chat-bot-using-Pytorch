import json
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np

from nltk_preprocessing import word_tokenize, stem, bag_of_words
from model import Neural_network

# Load trained model and other necessary information
FILE = "data.pth"
data = torch.load(FILE)

input_size = data["input_size"]
hidden_size = data["hidden_size"]
output_size = data["output_size"]
all_words = data["all_words"]
tags = data["tags"]

# Load the trained model
model = Neural_network(input_size, hidden_size, output_size)
model.load_state_dict(data["model_state"])
model.eval()

# Load test JSON data
with open("test.json", "r") as f:
    test_data = json.load(f)

# Prepare testing data
testing_sentences = []
for item in test_data['intents']:
    testing_sentences.extend(item['patterns'])

# Tokenize and convert testing sentences to bag of words representation
x_test = []
for sentence in testing_sentences:
    tokens = word_tokenize(sentence)
    bow = bag_of_words(tokens, all_words)
    x_test.append(bow)

# Convert x_test to numpy array
x_test = np.array(x_test)

# Define a DataLoader for testing
class TestDataset(Dataset):
    def __init__(self, x_data):
        self.n_samples = len(x_data)
        self.x_data = x_data

    def __getitem__(self, index):
        return self.x_data[index]

    def __len__(self):
        return self.n_samples

test_dataset = TestDataset(x_test)
test_loader = DataLoader(dataset=test_dataset, batch_size=1, shuffle=False)

# Perform inference on testing data
predictions = []
with torch.no_grad():
    for inputs in test_loader:
        outputs = model(inputs)
        _, predicted = torch.max(outputs, 1)
        predictions.append(predicted.item())

# Print predictions
for i, prediction in enumerate(predictions):
    print(f"Input: {testing_sentences[i]}")
    print(f"Predicted Tag: {tags[prediction]}")
    print("=" * 50)

# You can further evaluate the model's performance using various metrics

# import json
# import torch
# import torch.nn as nn
# from torch.utils.data import Dataset, DataLoader
# import numpy as np

# from nltk_preprocessing import word_tokenize, stem, bag_of_words
# from model import Neural_network

# # Load trained model and other necessary information
# FILE = "data.pth"
# data = torch.load(FILE)

# input_size = data["input_size"]
# hidden_size = data["hidden_size"]
# output_size = data["output_size"]
# all_words = data["all_words"]
# tags = data["tags"]

# # Load the trained model
# model = Neural_network(input_size, hidden_size, output_size)
# model.load_state_dict(data["model_state"])
# model.eval()

# # Prepare testing data
# # Assuming you have a list of testing sentences in 'testing_sentences'
# testing_sentences = [
#     "What's the weather like today?",
#     "Tell me a joke",
#     "Do you play video games?","please provide sports and games information"
#     # Add more testing sentences here
# ]

# # Tokenize and convert testing sentences to bag of words representation
# x_test = []
# for sentence in testing_sentences:
#     tokens = word_tokenize(sentence)
#     bow = bag_of_words(tokens, all_words)
#     x_test.append(bow)

# # Convert x_test to numpy array
# x_test = np.array(x_test)

# # Define a DataLoader for testing
# class TestDataset(Dataset):
#     def __init__(self, x_data):
#         self.n_samples = len(x_data)
#         self.x_data = x_data

#     def __getitem__(self, index):
#         return self.x_data[index]

#     def __len__(self):
#         return self.n_samples

# test_dataset = TestDataset(x_test)
# test_loader = DataLoader(dataset=test_dataset, batch_size=1, shuffle=False)

# # Perform inference on testing data
# predictions = []
# with torch.no_grad():
#     for inputs in test_loader:
#         outputs = model(inputs)
#         _, predicted = torch.max(outputs, 1)
#         predictions.append(predicted.item())

# # Print predictions
# for i, prediction in enumerate(predictions):
#     print(f"Input: {testing_sentences[i]}")
#     print(f"Predicted Tag: {tags[prediction]}")
#     print("=" * 50)

# # You can further evaluate the model's performance using various metrics
