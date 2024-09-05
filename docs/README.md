# Text-to-Image Generation from Scratch

This project aims to build a Text-to-Image Generation model from scratch without using pre-trained models or external APIs. The model takes textual descriptions as input and generates corresponding images.

## Features

- Train your own Text-to-Image model using custom neural network architectures.
- Evaluate the model's performance using metrics like FID and BLEU scores.
- Interactive webpage where users can input text and view generated images.
- Clean and modular codebase with separate modules for data, models, training, evaluation, and web integration.

## Project Structure

```bash
text-to-image-generation/
├── data/                 # Datasets and preprocessing scripts
├── models/               # Custom neural network models (Generator, Discriminator, etc.)
├── training/             # Training scripts, loss functions, and optimizers
├── evaluation/           # Evaluation scripts and metrics (FID, BLEU, etc.)
├── web/                  # Webpage frontend and backend (Flask)
├── utils/                # Helper functions like logging and checkpointing
├── docs/                 # Project documentation
├── scripts/              # Utility scripts (data processing, image generation, etc.)
├── tests/                # Unit tests for different modules
├── requirements.txt      # Python dependencies
├── setup.py              # Installation script
├── README.md             # Project overview (you are here)
└── CONTRIBUTING.md       # Contribution guidelines
