# Fashion Plates Dating Project

A deep learning project for automatically dating historical fashion plates using computer vision and machine learning techniques. This project implements multiple CNN architectures to classify or predict the year of fashion illustrations from the 19th century.

## 🎯 Project Overview

This project uses transfer learning with pre-trained CNN models to date fashion plates from 1820-1880. It supports both:
- **Classification**: Predicting discrete year classes
- **Regression**: Predicting continuous year values

### Key Features

- Multiple CNN architectures (InceptionV3, ResNet101, EfficientNetV2S, ConvNeXtTiny, etc.)
- Advanced data augmentation techniques
- Hyperparameter tuning with Bayesian optimization
- Cross-validation support
- Custom ordinal loss functions
- Statistical model comparison (5x2 cross-validation)
- Comprehensive metrics and visualization.

## 📋 Prerequisites

- Python 3.8+
- CUDA-compatible GPU (recommended)
- At least 8GB RAM
- 10GB+ disk space for datasets and models

## 🚀 Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/foal20ym/fashion_plates_dating.git
   cd fashion_plates_dating
   ```

2. **Create a virtual environment:**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **GPU Setup (Optional but recommended):**
   ```bash
   # For NVIDIA GPUs
   pip install tensorflow-gpu
   ```

## 📁 Project Structure

```
fashion_plates_dating/
├── fashion.py              # Main entry point
├── config.yaml            # Configuration file
├── models.py              # Model architectures and setup
├── training.py            # Training and evaluation logic
├── data.py                # Data loading and preprocessing
├── tuning.py              # Hyperparameter tuning
├── metrics.py             # Custom metrics and evaluation
├── plotting.py            # Visualization utilities
├── utils.py               # General utilities
├── model_comparison.py    # Statistical model comparison
├── saliency_mapping.py    # Model interpretability
├── background_remover.py  # Image preprocessing
├── data/
│   └── datasets/          # Dataset CSV files
│       ├── fold0.csv
│       ├── fold1.csv
│       └── ...
└── plots/                 # Generated visualizations
```

## ⚙️ Configuration

Edit `config.yaml` to customize your experiment:

```yaml
# Task type
task: classification  # or "regression"

# Model configuration
model:
  name: InceptionV3  # InceptionV3, ResNet101, EfficientNetV2S, ConvNeXtTiny
  normalize_years: true
  normalization_method: "minmax"
  fine_tune:
    use: true
    layers: 9
  dropout:
    use: true
    value: 0.4
  l2_regularization:
    use: true
    value: 1e-5

# Training parameters
training:
  batch_size: 8
  epochs: 300
  learning_rate: 0.0003
  early_stopping_patience: 16
  reduce_lr_patience: 8

# Experiment options
cross_validation: false
hyperparameter_tuning:
  enabled: false
  method: "bayesian"
  max_trials: 15
```

## 🏃‍♂️ How to Run

### 1. Basic Training (Single Fold)

Run a single training session:

```bash
python3 src/fashion.py
```

### 2. Cross-Validation

Enable 10-fold cross-validation in `config.yaml`:

```yaml
cross_validation: true
```

Then run:

```bash
python3 src/fashion.py
```

### 3. Hyperparameter Tuning

Enable hyperparameter tuning in `config.yaml`:

```yaml
hyperparameter_tuning:
  enabled: true
  method: "bayesian"  # or "random", "hyperband"
  max_trials: 15
```

Then run:

```bash
python3 src/fashion.py
```

### 4. Model Comparison

Enable statistical model comparison:

```yaml
null_hypothesis_testing:
  use: true
```

### 5. Change Task Type

Switch between classification and regression:

```yaml
task: regression  # or "classification"
```

## 📊 Data Format

Your dataset CSV files should have the following format:

```csv
file,year
data/datasets/public/1870/1879_1476vna.jpg,1879
data/datasets/public/1850/1855_2341abc.jpg,1855
...
```

- `file`: Path to image file
- `year`: Year label (1820-1880)

## 🎛️ Model Architectures

The project supports multiple pre-trained CNN architectures:

| Model           | Input Size | Memory Usage | Performance |
| --------------- | ---------- | ------------ | ----------- |
| InceptionV3     | 224x224    | Medium       | Excellent   |
| ResNet101       | 224x224    | Medium       | Good        |
| EfficientNetV2S | 384x384    | High         | Good        |
| ConvNeXtTiny    | 448x448    | Very High    | Excellent   |
| NASNetMobile    | 224x224    | Low          | Moderate    |

## 📈 Output and Results

The project generates:

1. **Training logs**: Displayed in console
2. **Model checkpoints**: Saved if `save_model: true`
3. **Plots**: Saved in `plots/` directory
4. **Metrics**: MAE, accuracy, MCC (depending on task)
5. **TensorBoard logs**: For training monitoring

### Example Output

```
===== Fold 0 =====
Using model: InceptionV3
Task: Classification
Training samples: 4500
Validation samples: 500

Epoch 1/300
563/563 [==============================] - 45s - loss: 2.1234 - accuracy: 0.4567
...

Final Metrics:
- Accuracy: 0.8234
- MAE (years): 2.45
- MCC: 0.7891
```

## 🔧 Troubleshooting

### Common Issues

1. **GPU Memory Error**:
   - Reduce `batch_size` in config
   - Use smaller input sizes
   - Enable mixed precision training

2. **Config Error**:
   - Check YAML syntax
   - Ensure all required fields are present
   - Verify data types (int vs string)

3. **Data Loading Error**:
   - Verify CSV file paths
   - Check image file accessibility
   - Ensure year values are in correct range

### Memory Optimization

For limited GPU memory:

```yaml
training:
  batch_size: 4  # Reduce from 8
model:
  name: InceptionV3  # Use smaller model
```

## 📚 Advanced Usage

### Custom Loss Functions

The project includes custom ordinal loss functions optimized for temporal data:

- `ordinal_categorical_cross_entropy`: For classification
- `ordinal_regression_loss`: For regression

### Data Augmentation

Automatic data augmentation includes:
- Random flips and rotations
- Contrast adjustments
- RandAugment techniques

### Model Interpretability

Generate saliency maps to understand model decisions:

```python
from saliency_mapping import generate_saliency_maps
generate_saliency_maps(model, test_images)
```

## 🔬 Research

This project implements techniques from computer vision and historical analysis research. If you use this code in your research, please cite appropriately.

---

**Happy dating those fashion plates! 👗📅**
