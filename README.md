# Stock Price Forecasting with Temporal Fusion Transformer

This repository implements a state-of-the-art stock price forecasting solution using the Temporal Fusion Transformer (TFT) architecture. It provides an end-to-end pipeline for data preparation, model training, evaluation, and prediction.

## Features

- **Advanced Data Processing**: Automatic data fetching, technical indicator calculation, and preprocessing
- **Temporal Fusion Transformer**: Cutting-edge deep learning model for time series forecasting
- **Probabilistic Forecasting**: Predictions with uncertainty quantification through quantile forecasting
- **Hyperparameter Optimization**: Optional tuning of model hyperparameters using Ray Tune
- **Interpretable Results**: Visualization and interpretation of forecasts
- **Production-Ready**: Comprehensive logging, error handling, and modular design

## Requirements

- Python 3.8+
- PyTorch 1.12+
- PyTorch Lightning 1.9+
- PyTorch Forecasting 0.10+
- Other dependencies in `requirements.txt`

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/stock-forecasting-tft.git
cd stock-forecasting-tft

# Create and activate a virtual environment (optional but recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Usage

### Basic Usage

```bash
# Train a model and make predictions
python stock_forecasting.py --symbol AAPL

# Only train a model
python stock_forecasting.py --mode train --symbol AAPL

# Only make predictions (uses existing model or trains one if not found)
python stock_forecasting.py --mode predict --symbol AAPL --days 10

# Use a custom configuration file
python stock_forecasting.py --config my_config.yaml
```

### Configuration

You can customize the behavior by creating a `config.yaml` file. See the example configuration file for available options:

```yaml
# Data parameters
stock_symbol: "MSFT"
start_date: "2015-01-02"
end_date: "2024-10-31"
data_directory: "stock_data"

# Model parameters
prediction_window: 5      # Forecast horizon (days)
encoder_length: 60        # Historical context length (days)
...
```

## Project Structure

```
stock-forecasting-tft/
├── stock_forecasting.py     # Main implementation
├── config.yaml              # Configuration file
├── requirements.txt         # Dependencies
├── README.md                # Documentation
└── stock_data/              # Data directory (created on first run)
    ├── historical/          # Historical price data
    ├── models/              # Trained models
    └── results/             # Predictions and visualizations
```

## How It Works

The pipeline consists of several components:

1. **ConfigManager**: Handles configuration parameters
2. **DataManager**: Manages data loading, processing, and feature engineering
3. **ModelManager**: Handles model creation, training, evaluation, and prediction
4. **StockForecastingPipeline**: Orchestrates the entire workflow

### Data Processing

The system downloads historical stock data and calculates technical indicators including:
- Moving averages (10, 50, 200-day)
- Volatility metrics
- RSI (Relative Strength Index)
- MACD (Moving Average Convergence Divergence)
- Bollinger Bands
- Volume indicators

### Model

The Temporal Fusion Transformer (TFT) is a state-of-the-art architecture for time series forecasting that combines:
- Variable selection networks
- Gated residual networks
- Multi-head attention
- Temporal processing layers

It excels at capturing long-term dependencies and handling mixed frequency data with both static and time-varying covariates.

### Forecasting

The model produces quantile forecasts (by default at the 10%, 50%, and 90% levels), providing both point predictions and uncertainty estimates.

## Advanced Usage

### Running Hyperparameter Optimization

To enable hyperparameter tuning:

1. Set `enable_hyperopt: true` in your configuration file
2. Adjust `hyperopt_trials` and `hyperopt_cpu` as needed
3. Run the pipeline with `--mode train`

### Using as a Module

You can import and use the components in your own Python code:

```python
from stock_forecasting import StockForecastingPipeline

# Initialize pipeline
pipeline = StockForecastingPipeline()

# Override configuration
pipeline.config_manager.set("stock_symbol", "TSLA")

# Run training
results = pipeline.run_training_pipeline()

# Make predictions
predictions = pipeline.run_prediction_pipeline(days_to_predict=10)
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- The implementation is based on the PyTorch Forecasting library
- The Temporal Fusion Transformer architecture was introduced by B. Lim et al. in "Temporal Fusion Transformers for Interpretable Multi-horizon Time Series Forecasting"
