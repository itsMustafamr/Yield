# Yield Prediction with Tabular Foundation Models

A machine learning project for predicting crop yield from environmental and agronomic time-series data. This repository contains experiments comparing various tabular ML approaches including TabPFN, TabDPT, and traditional neural networks.

## Project Overview

This project explores different machine learning models for yield prediction.

## Dataset

- **Size**: ~23M time-series rows aggregated to ~86k sequences
- **Sequence Length**: 214 rows per time series
- **Features**: 57 total columns
  - 6 temporal features (Temperature, Humidity, Precipitation, Radiation, VPD, GDD)
  - 5 plant features (Protein, Oil, Lodging, PlantHeight, SeedSize)
  - 3 geographic features (MG, Latitude, Longitude)
  - 40 cluster indicator features
  - Target: Yield

**Note**: 
The dataset files (`train.csv`, `val.csv`, `test.csv`) are not included in this repository due to size constraints.
You should be able to try it on any agricultural tabular data.

## Installation

### Prerequisites
- Python 3.9+
- pip

### Setup

```bash
# Clone the repository
git clone https://github.com/yourusername/Yield.git
cd Yield

# Create a virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### PyTorch with CUDA (Optional)

For GPU acceleration, install PyTorch with CUDA support:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu118
```

## Repository Structure

```
Yield/
├── tabdptfull.py                 # TabDPT full training script
├── tabdpt_quant.py               # TabDPT with quantization
├── tabdptfull_with_quantization.py
├── tabPFN_allfeature.ipynb       # TabPFN experiments
├── tabPFN_quantization.ipynb     # Quantization bit-depth analysis
├── tabPFN_RealMLP.ipynb          # PyTorch MLP (best results)
├── tabPFN_RealMLP_with_quantization.ipynb
├── tabDPT_*.ipynb                # TabDPT notebook experiments
├── load.ipynb                    # Data exploration
├── Old_files/                    # Archived experiments
└── requirements.txt
```

## Usage

### Running Python Scripts

```bash
# Run TabDPT training
python tabdptfull.py

# Run TabDPT with quantization
python tabdpt_quant.py
```

## Key Experiments

### 1. TabPFN with Quantization
Explores feature quantization at different bit-depths (2, 4, 6, 8, 16, 32, 64, 128, 256 bits). Findings show quantization saturates around 4-6 bits with minimal performance loss.

### 2. PyTorch MLP
4-layer MLP with:
- StandardScaler preprocessing
- Batch normalization and dropout
- AdamW optimizer with LR scheduling
- Early stopping (patience=20)

### 3. TabDPT Experiments
Transformer-based tabular model with sequence aggregation and optional quantization preprocessing.

## Data Preprocessing

The time-series data is aggregated per sequence (TimeSeriesLabel):
- **Temporal features**: mean and std computed across 214 timesteps
- **Static features**: first value or mode taken
- **Missing values**: imputed with mode (MG), median (Longitude), or mean (plant features)

## License

This project is for research purposes.

## Citation

If you use this code in your research, please cite the associated paper.

