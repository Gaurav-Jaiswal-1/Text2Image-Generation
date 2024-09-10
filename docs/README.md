# Text-to-Image Generation from Scratch

Welcome to the Text-to-Image Generation project! This repository contains a complete implementation of a model that generates images from textual descriptions. We build everything from scratch, using custom neural networks and integrating them with a web application.

## Project Overview

This project includes:

- **Model Architecture**: Custom neural network models for generating images from text descriptions.
- **Training Scripts**: Code to train the model on your own data.
- **Evaluation**: Metrics to assess the quality of generated images.
- **Web Interface**: An interactive webpage to input text and view generated images.

## Features

- Train your own model using custom architectures.
- Interactive web interface for text-to-image generation.
- Metrics for evaluating image quality, including FID and BLEU scores.
- Modular and clean codebase with separate components for data handling, model building, training, evaluation, and web integration.

## Installation

### Prerequisites

- Python 3.8 or higher
- Flask
- PyTorch
- Other Python dependencies listed in `requirements.txt`

### Setup

1. **Clone the repository**:
   ```bash
   git clone https://github.com/Gaurav-Jaiswal-1/Text2Image-Generation.git
   cd text-to-image-generation
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Training the Model

1. **Preprocess the Data**:
   Prepare your dataset by running:
   ```bash
   python scripts/preprocess_data.py
   ```

2. **Train the Model**:
   Start training the model with:
   ```bash
   python training/train.py --config config.yaml
   ```

3. **Monitor Training**:
   Use TensorBoard to visualize the training process:
   ```bash
   tensorboard --logdir logs/
   ```

### Running the Web Application

1. **Start the Backend Server**:
   ```bash
   python web/backend/server.py
   ```

2. **Access the Web Interface**:
   Open your web browser and go to `http://localhost:5000` to use the interactive text-to-image generation interface.

### Generating Images from Text

To generate images without using the web interface:
```bash
python scripts/generate_images.py --text "A sunny beach with palm trees"
```

### Evaluating the Model

Assess the quality of the generated images:
```bash
python evaluation/evaluate.py --model checkpoints/generator.pth
```

## Project Structure

```bash
text-to-image-generation/
├── data/                 # Datasets and preprocessing scripts
├── models/               # Custom neural network models
├── training/             # Training scripts and loss functions
├── evaluation/           # Evaluation metrics and visualization
├── web/                  # Web application (frontend and backend)
├── utils/                # Utility functions (logging, checkpointing, etc.)
├── scripts/              # Utility scripts (data processing, etc.)
├── tests/                # Unit tests for various components
├── docs/                 # Documentation
├── requirements.txt      # Python dependencies
├── setup.py              # Installation script
├── README.md             # Project overview (you are here)
└── CONTRIBUTING.md       # Contribution guidelines
```

## Documentation

- [README.md](README.md): Overview and instructions.
- [CONTRIBUTING.md](CONTRIBUTING.md): How to contribute to this project.
- [docs/CHANGELOG.md](docs/CHANGELOG.md): Change log for tracking updates.

## Contributing

We welcome contributions to enhance this project. Please refer to the [CONTRIBUTING.md](CONTRIBUTING.md) for details on how to get involved.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Contact

For any questions or issues, please open an issue on this repository or contact us at jaiswalgaurav863@gmail.com



