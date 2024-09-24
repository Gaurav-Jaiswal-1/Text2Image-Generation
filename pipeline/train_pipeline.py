# pipeline/train_pipeline.py
import torch
from torch.utils.data import DataLoader
from models.vae import VAE
from models.text_encoder import TextEncoder
from models.image_generator import ImageGenerator
from models.discriminator import Discriminator
from training.loss_functions import vae_loss_function, gan_loss_function
from training.optimizers import get_optimizer
from utils.checkpoint import save_checkpoint
from data.data_utils import load_data

def train_pipeline(config):
    # Load data
    train_dataset, val_dataset = load_data(config['data_path'])
    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config['batch_size'], shuffle=False)
    
    # Initialize models
    text_encoder = TextEncoder(config['text_encoder'])
    vae = VAE(config['vae'])
    image_generator = ImageGenerator(config['image_generator'])
    discriminator = Discriminator(config['discriminator'])
    
    # Initialize optimizer and scheduler
    optimizer = get_optimizer([vae, text_encoder, image_generator], config['learning_rate'])
    
    for epoch in range(config['epochs']):
        vae.train()
        image_generator.train()
        
        for i, batch in enumerate(train_loader):
            text, images = batch
            
            # Forward pass through the models
            text_embedding = text_encoder(text)
            vae_output, mu, logvar = vae(text_embedding)
            generated_images = image_generator(vae_output)
            
            # Calculate loss
            loss_vae = vae_loss_function(vae_output, images, mu, logvar)
            loss_gan = gan_loss_function(discriminator, generated_images, images)
            total_loss = loss_vae + loss_gan
            
            # Backpropagation
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
            
            if i % config['log_interval'] == 0:
                print(f'Epoch [{epoch}/{config["epochs"]}], Step [{i}/{len(train_loader)}], Loss: {total_loss.item()}')

        # Save checkpoint
        save_checkpoint({
            'epoch': epoch,
            'vae_state_dict': vae.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
        }, filename=config['checkpoint_path'])

        # Optionally, add validation step here
        
    print('Training complete!')
