import json
import random

# Load the JSON data
with open('new_intents.json', 'r') as file:
    intents_data = json.load(file)

# Dictionary to store train and test data
train_data = {"intents": []}
test_data = {"intents": []}

# Define the split ratio for train and test
split_ratio = 0.8  # 80% for train, 20% for test

# Iterate through each tag and split the patterns
for intent in intents_data["intents"]:
    tag = intent["tag"]
    patterns = intent["patterns"]
    responses = intent["responses"]
    num_patterns = len(patterns)

    # Calculate the number of patterns for train and test
    num_train_patterns = int(split_ratio * num_patterns)
    num_test_patterns = num_patterns - num_train_patterns

    # Shuffle patterns to ensure randomness
    random.shuffle(patterns)

    # Split patterns into train and test data
    train_patterns = patterns[:num_train_patterns]
    test_patterns = patterns[num_train_patterns:]

    # Add all responses to train data
    train_responses = responses

    # Add the split patterns to train and test data
    train_data["intents"].append({"tag": tag, "patterns": train_patterns, "responses": train_responses})
    test_data["intents"].append({"tag": tag, "patterns": test_patterns})

# Save train and test data into separate JSON files
with open('train_data.json', 'w') as train_file:
    json.dump(train_data, train_file, indent=4)

with open('test_data.json', 'w') as test_file:
    json.dump(test_data, test_file, indent=4)
