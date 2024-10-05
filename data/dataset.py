import os
import requests
from tqdm import tqdm

# Constants
COCO_URL = "http://images.cocodataset.org/zips/"
ANNOTATIONS_URL = "http://images.cocodataset.org/annotations/annotations_trainval2017.zip"
IMAGES_URL = COCO_URL + "train2017.zip"
SAVE_PATH = r"C:\Users\Gaurav\OneDrive\Desktop\Text2Image\data\raw"

# Function to download and extract dataset
def download_file(url, save_dir, extract=True):
    local_filename = url.split("/")[-1]
    save_path = os.path.join(save_dir, local_filename)

    # Create directory if it doesn't exist
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # Download the file with progress bar
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))
    block_size = 1024  # 1 KB
    t = tqdm(total=total_size, unit='iB', unit_scale=True)

    with open(save_path, "wb") as file:
        for data in response.iter_content(block_size):
            t.update(len(data))
            file.write(data)
    t.close()

    # Extract if necessary
    if extract:
        import zipfile
        with zipfile.ZipFile(save_path, 'r') as zip_ref:
            zip_ref.extractall(save_dir)
        print(f"Extracted: {save_path}")

# Download annotations
print("Downloading annotations...")
download_file(ANNOTATIONS_URL, SAVE_PATH)

# Download training images
print("Downloading training images...")
download_file(IMAGES_URL, SAVE_PATH)

print("Download complete!")
