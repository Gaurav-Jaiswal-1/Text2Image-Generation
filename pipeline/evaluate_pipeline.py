# pipeline/evaluate_pipeline.py
from torch.utils.data import DataLoader
from models.vae import VAE
from models.image_generator import ImageGenerator
from evaluation.metrics import calculate_fid
from data.data_utils import load_data

def evaluate_pipeline(config):
    # Load validation dataset
    val_dataset = load_data(config['data_path'], train=False)
    val_loader = DataLoader(val_dataset, batch_size=config['batch_size'], shuffle=False)
    
    # Load models
    vae = VAE(config['vae'])
    image_generator = ImageGenerator(config['image_generator'])
    
    # Set models to evaluation mode
    vae.eval()
    image_generator.eval()
    
    all_generated_images = []
    all_real_images = []
    
    for text, real_images in val_loader:
        with torch.no_grad():
            vae_output, _, _ = vae(text)
            generated_images = image_generator(vae_output)
        
        all_generated_images.append(generated_images)
        all_real_images.append(real_images)
    
    # Calculate FID
    fid_score = calculate_fid(all_real_images, all_generated_images)
    print(f"FID Score: {fid_score}")
