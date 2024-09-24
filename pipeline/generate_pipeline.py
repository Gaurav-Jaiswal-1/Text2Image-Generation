# pipeline/generate_pipeline.py
import torch
from models.vae import VAE
from models.text_encoder import TextEncoder
from models.image_generator import ImageGenerator
from utils.helpers import load_checkpoint

def generate_images_from_text(text_input, config):
    # Load models
    text_encoder = TextEncoder(config['text_encoder'])
    vae = VAE(config['vae'])
    image_generator = ImageGenerator(config['image_generator'])
    
    # Load checkpoints
    load_checkpoint(vae, config['checkpoint_path'])
    
    # Set models to evaluation mode
    text_encoder.eval()
    vae.eval()
    image_generator.eval()
    
    # Process input text
    with torch.no_grad():
        text_embedding = text_encoder(text_input)
        vae_output, _, _ = vae(text_embedding)
        generated_image = image_generator(vae_output)
    
    return generated_image
