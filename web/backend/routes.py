import os
from models.generator import generate_image_from_text

def generate_image(text_description):
    # Generate the image using the model (implemented in `generator.py`)
    image_path = generate_image_from_text(text_description)
    
    # Serve the image to the frontend
    image_url = f'/static/generated_images/{os.path.basename(image_path)}'
    return image_url
