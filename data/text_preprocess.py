import numpy as np

# Sample vocabulary and one-hot encoding
vocabulary = ['cat', 'dog', 'sun', 'house', 'flower']  # Define your vocabulary here
vocab_size = len(vocabulary)

def text_to_vector(text):
    words = text.lower().split()  # Simple tokenization by splitting on spaces
    vector = np.zeros((len(words), vocab_size))
    for i, word in enumerate(words):
        if word in vocabulary:
            vector[i][vocabulary.index(word)] = 1  # One-hot encode the word
    return vector.flatten()  # Flatten to a single vector

# Example
text_input = "cat sun"
text_vector = text_to_vector(text_input)
print("Text Vector:", text_vector)
