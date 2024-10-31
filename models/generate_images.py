import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

def generate_image_from_text(text):
    text_vector = text_to_vector(text)
    text_tensor = torch.tensor(text_vector, dtype=torch.float).unsqueeze(0)
    with torch.no_grad():
        generated_image = generator(text_tensor).view(28, 28)  # Reshape to 28x28 image
    return generated_image

# Main function to take input and generate image
def main():
    # Train generator (in practice, this would be more complex)
    train_generator()
    
    # Take text input from the user
    user_input = input("Enter a description using words like 'cat', 'dog', 'sun', 'house', or 'flower': ")
    
    # Generate and display the image
    generated_image = generate_image_from_text(user_input)
    plt.imshow(generated_image, cmap="gray")
    plt.axis("off")
    plt.title(f"Generated image for: '{user_input}'")
    plt.show()

# Run the main function
main()