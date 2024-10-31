# data/preprocess_data.py
import os
from PIL import Image
import torch
from transformers import BertTokenizer

def preprocess_image(image_path, size=(128, 128)):
    image = Image.open(image_path).convert("RGB")
    image = image.resize(size)
    return image

def preprocess_text(text, tokenizer):
    tokens = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
    return tokens['input_ids'], tokens['attention_mask']

def main(data_dir, save_dir, image_size=(128, 128)):
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    for filename in os.listdir(data_dir):
        if filename.endswith(".jpg") or filename.endswith(".png"):
            image = preprocess_image(os.path.join(data_dir, filename), image_size)
            text_path = filename.replace(".jpg", ".txt").replace(".png", ".txt")
            with open(os.path.join(data_dir, text_path), 'r') as f:
                text = f.read().strip()
            tokens, mask = preprocess_text(text, tokenizer)
            # Save preprocessed image and tokens
            # Add code to save them in a format suitable for training (e.g., torch tensors)

if __name__ == "__main__":
    main(data_dir=r"C:\Users\Gaurav\OneDrive\Desktop\Text2Image\data\Extracted\archive (4)\flickr30k_images\flickr30k_images", save_dir="data/processed")
