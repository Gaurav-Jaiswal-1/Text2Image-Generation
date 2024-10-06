from setuptools import setup, find_packages

# Define the version of the package
VERSION = '0.1.0'

# Read the README file for the long description
with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

# Define setup() with necessary arguments
setup(
    name='text2image-gen',
    version=VERSION,
    author='Gaurav Jaiswal',
    author_email='jaiswalgaurav863@gmail.com',
    description='A text-to-image generation project built from scratch using GANs',
    long_description=long_description,
    long_description_content_type='text/markdown',  # Assuming README.md is in markdown format
    url='https://github.com/Gaurav-Jaiswal-1/Text2Image-Generation.git',  # Replace with your repository URL
    packages=find_packages(),  # Automatically find packages within the project
    install_requires=[
        'torch>=1.9.0',          # PyTorch
        'torchvision>=0.10.0',   # TorchVision
        'transformers>=4.0.0',   # Hugging Face Transformers (for text processing)
        'nltk>=3.5',             # Natural Language Toolkit
        'Pillow>=8.0.0',         # PIL for image processing
        'tqdm>=4.50.0',          # Progress bar library (optional)
        # Add other dependencies you are using
    ],
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.6',  # Specify the minimum Python version required
)
