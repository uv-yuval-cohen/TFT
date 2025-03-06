"""
Stock Price Forecasting with Temporal Fusion Transformer (TFT)
==============================================================

This module implements an end-to-end stock price forecasting solution using
Temporal Fusion Transformer (TFT) architecture. It handles data loading,
feature engineering, model training, evaluation, and prediction.

Key features:
- Automatic data fetching and preprocessing
- Advanced technical indicator calculation
- State-of-the-art TFT deep learning model
- Comprehensive evaluation metrics
- Probabilistic forecasting with prediction intervals
- Visualization tools for model interpretation
- Hyperparameter optimization
- Model serialization and deployment

Author: Yuval Cohen
Date: March 6, 2025
"""

import os
import json
import logging
import warnings
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Union, Optional, Any

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
import joblib
import yaml
import yfinance as yf
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger
import pytorch_forecasting as ptf
from pytorch_forecasting import TimeSeriesDataSet, TemporalFusionTransformer
from pytorch_forecasting.data import TorchNormalizer
from pytorch_forecasting.metrics import MAE, RMSE, SMAPE, QuantileLoss
from ray import tune
from ray.tune.schedulers import ASHAScheduler
from ray.tune.integration.pytorch_lightning import TuneReportCallback

# Suppress warnings
warnings.filterwarnings('ignore')

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("stock_forecasting.log")
    ]
)
logger = logging.getLogger(__name__)


class ConfigManager:
    """Manages configuration settings for the stock forecasting pipeline."""

    DEFAULT_CONFIG = {
        # Data parameters
        "stock_symbol": "AAPL",
        "start_date": "2015-01-02",
        "end_date": "2024-10-31",
        "data_directory": "stock_data",

        # Model parameters
        "prediction_window": 5,  # Forecast horizon (days)
        "encoder_length": 60,  # Historical context length (days)
        "volatility_window": 20,  # Window for volatility calculation

        # Training parameters
        "max_epochs": 50,
        "learning_rate": 1e-3,
        "batch_size": 64,
        "hidden_size": 64,
        "attention_head_size": 4,
        "dropout": 0.1,
        "hidden_continuous_size": 32,

        # Evaluation parameters
        "test_size": 0.2,  # Proportion of data for testing
        "validation_size": 0.15,  # Proportion of data for validation

        # Hyperparameter optimization
        "enable_hyperopt": False,
        "hyperopt_trials": 20,
        "hyperopt_cpu": 4,

        # Prediction parameters
        "quantiles": [0.1, 0.5, 0.9]  # Prediction quantiles
    }

    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize configuration with either a provided config file or defaults.

        Args:
            config_path: Path to a YAML configuration file (optional)
        """
        self.config = self.DEFAULT_CONFIG.copy()

        if config_path and os.path.exists(config_path):
            with open(config_path, 'r') as f:
                user_config = yaml.safe_load(f)
                self.config.update(user_config)
                logger.info(f"Loaded configuration from {config_path}")

        # Create data directories
        os.makedirs(self.config["data_directory"], exist_ok=True)
        os.makedirs(os.path.join(self.config["data_directory"], 'historical'), exist_ok=True)
        os.makedirs(os.path.join(self.config["data_directory"], 'models'), exist_ok=True)
        os.makedirs(os.path.join(self.config["data_directory"], 'results'), exist_ok=True)

        # Log configuration
        logger.info(f"Using configuration: {json.dumps(self.config, indent=2)}")

    def save_config(self, path: str) -> None:
        """
        Save current configuration to a YAML file.

        Args:
            path: Path where to save the configuration
        """
        with open(path, 'w') as f:
            yaml.dump(self.config, f, default_flow_style=False)
        logger.info(f"Configuration saved to {path}")

    def get(self, key: str, default: Optional[Any] = None) -> Any:
        """
        Get a configuration value.

        Args:
            key: Configuration key
            default: Default value if key is not found

        Returns:
            Configuration value
        """
        return self.config.get(key, default)

    def set(self, key: str, value: Any) -> None:
        """
        Set a configuration value.

        Args:
            key: Configuration key
            value: Value to set
        """
        self.config[key] = value


class DataManager:
    """Handles data loading, processing, and feature engineering for stock data."""

    def __init__(self, config_manager: ConfigManager):
        """
        Initialize DataManager with configuration.

        Args:
            config_manager: Configuration manager instance
        """
        self.config = config_manager
        self.symbol = self.config.get("stock_symbol")
        self.data_dir = self.config.get("data_directory")
        self.start_date = self.config.get("start_date")
        self.end_date = self.config.get("end_date")
        self.prediction_window = self.config.get("prediction_window")
        self.volatility_window = self.config.get("volatility_window")
        self.encoder_length = self.config.get("encoder_length")

        # Will store processed data
        self.static_features = None
        self.raw_data = None
        self.processed_data = None
        self.tft_data = None
        self.normalized_data = None
        self.scaler = None

        # Store dataset splits
        self.training_dataset = None
        self.validation_dataset = None
        self.test_dataset = None
        self.train_dataloader = None
        self.val_dataloader = None
        self.test_dataloader = None

        logger.info(f"DataManager initialized for {self.symbol}")

    def get_static_features(self) -> Dict[str, Any]:
        """
        Load static features for the stock (sector, market cap).

        Returns:
            Dictionary of static features
        """
        static_file = os.path.join(self.data_dir, 'static_features.csv')

        try:
            if os.path.exists(static_file):
                static_df = pd.read_csv(static_file)
                if self.symbol in static_df['stock_symbol'].values:
                    static_row = static_df[static_df['stock_symbol'] == self.symbol]
                    self.static_features = {
                        'stock_symbol': self.symbol,
                        'sector': static_row['sector'].iloc[0],
                        'market_cap': static_row['market_cap'].iloc[0]
                    }
                    logger.info(f"Loaded static features from {static_file}")
                    return self.static_features

            # Fallback to yfinance API
            logger.info(f"Fetching static features for {self.symbol} from yfinance API")
            ticker_info = yf.Ticker(self.symbol).info

            self.static_features = {
                'stock_symbol': self.symbol,
                'sector': ticker_info.get('sector', 'Unknown'),
                'market_cap': ticker_info.get('marketCap', 0)
            }

            # Save for future use
            if not os.path.exists(static_file):
                pd.DataFrame([self.static_features]).to_csv(static_file, index=False)
            else:
                static_df = pd.read_csv(static_file)
                if self.symbol not in static_df['stock_symbol'].values:
                    static_df = pd.concat([static_df, pd.DataFrame([self.static_features])])
                    static_df.to_csv(static_file, index=False)

            logger.info(f"Fetched and saved static features for {self.symbol}")
            return self.static_features

        except Exception as e:
            logger.error(f"Error fetching static features: {e}")
            self.static_features = {
                'stock_symbol': self.symbol,
                'sector': 'Unknown',
                'market_cap': 0
            }
            return self.static_features

    def load_stock_data(self) -> pd.DataFrame:
        """
        Load stock data from local file or download if not available.

        Returns:
            DataFrame with stock data
        """
        historical_dir = os.path.join(self.data_dir, 'historical')
        filename = os.path.join(historical_dir, f'{self.symbol}.csv')

        # Check if we need fresh data
        need_download = True
        if os.path.exists(filename):
            try:
                # Load existing data
                existing_data = pd.read_csv(filename)

                # Convert Date column to datetime
                if 'Date' in existing_data.columns:
                    existing_data['Date'] = pd.to_datetime(existing_data['Date'])

                    # Check if data covers requested date range
                    if (min(existing_data['Date']) <= pd.to_datetime(self.start_date) and
                            max(existing_data['Date']) >= pd.to_datetime(self.end_date)):
                        # Filter for the specified date range AND create a copy
                        data = existing_data[(existing_data['Date'] >= self.start_date) &
                                             (existing_data['Date'] <= self.end_date)].copy()
                        if not data.empty:
                            need_download = False
                            logger.info(f"Using existing data for {self.symbol}")
                            self.raw_data = data
                            return data
            except Exception as e:
                logger.warning(f"Error reading existing file: {e}")

        # Download fresh data if needed
        if need_download:
            try:
                logger.info(f"Downloading data for {self.symbol} from {self.start_date} to {self.end_date}")
                data = yf.download(self.symbol, start=self.start_date, end=self.end_date)
                if data.empty:
                    logger.error(f"No data downloaded for {self.symbol}")
                    return None

                # Check if columns have multiple levels (ticker symbols as a second level)
                if isinstance(data.columns, pd.MultiIndex):
                    logger.info("Detected multi-level columns, dropping second level")
                    data.columns = data.columns.droplevel(1)

                # Reset index to make Date a column
                data = data.reset_index()

                # Save to file for future use
                data.to_csv(filename, index=False)
                logger.info(f"Downloaded and saved {len(data)} records for {self.symbol}")

            except Exception as e:
                logger.error(f"Error downloading data: {e}")
                return None

        # Create explicit copy to avoid SettingWithCopyWarning
        data = data.copy()

        # Ensure Date is datetime
        if 'Date' in data.columns:
            data['Date'] = pd.to_datetime(data['Date'])

        # Reset index if needed and drop any explicit 'index' column
        data = data.reset_index()
        if 'index' in data.columns:
            data = data.drop(columns=['index'])

        self.raw_data = data

        # Save raw data
        raw_data_path = os.path.join(self.data_dir, f"{self.symbol}_raw.csv")
        data.to_csv(raw_data_path, index=False)
        logger.info(f"Raw data saved to {raw_data_path}")

        return data

    def calculate_technical_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate technical indicators for stock data.

        Args:
            data: DataFrame with stock data

        Returns:
            DataFrame with added technical indicators
        """
        # Create a copy to avoid modification warnings
        df = data.copy()

        # Moving averages
        df['MA10'] = df['Close'].rolling(window=10).mean()
        df['MA50'] = df['Close'].rolling(window=50).mean()
        df['MA200'] = df['Close'].rolling(window=200).mean()

        # Returns and volatility
        df['returns'] = df['Close'].pct_change()
        df['volatility_20d'] = df['returns'].rolling(window=20).std() * 100
        df['volatility_50d'] = df['returns'].rolling(window=50).std() * 100

        # Trend indicators
        df['trend_10d'] = df['Close'] / df['MA10'] - 1
        df['trend_50d'] = df['Close'] / df['MA50'] - 1

        # Volume indicators
        df['volume_ma10'] = df['Volume'].rolling(window=10).mean()
        df['volume_ratio'] = df['Volume'] / df['volume_ma10']

        # RSI
        delta = df['Close'].diff()
        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)
        avg_gain = gain.rolling(window=14).mean()
        avg_loss = loss.rolling(window=14).mean()
        rs = avg_gain / avg_loss
        df['rsi'] = 100 - (100 / (1 + rs))

        # MACD
        ema_fast = df['Close'].ewm(span=12).mean()
        ema_slow = df['Close'].ewm(span=26).mean()
        df['macd'] = ema_fast - ema_slow
        df['macd_signal'] = df['macd'].ewm(span=9).mean()
        df['macd_hist'] = df['macd'] - df['macd_signal']

        # Bollinger Bands
        rolling_mean = df['Close'].rolling(window=20).mean()
        rolling_std = df['Close'].rolling(window=20).std()
        df['upper_band'] = rolling_mean + (rolling_std * 2)
        df['lower_band'] = rolling_mean - (rolling_std * 2)
        df['bb_width'] = (df['upper_band'] - df['lower_band']) / df['Close']

        # Time features
        df['day_of_week'] = df['Date'].dt.dayofweek
        df['month'] = df['Date'].dt.month
        df['is_quarter_end'] = (df['Date'].dt.month % 3 == 0) & (df['Date'].dt.day >= 28)
        df['year'] = df['Date'].dt.year

        logger.info(f"Calculated technical indicators for {self.symbol}")
        return df

    def calculate_target(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate the risk-adjusted target variable and clean missing values.

        Args:
            data: DataFrame with stock data and technical indicators

        Returns:
            DataFrame with target variables
        """
        # Create a copy to avoid modification warnings
        df = data.copy()

        # Calculate returns first (needed for volatility)
        df['returns'] = df['Close'].pct_change()
        df['volatility'] = df['returns'].rolling(window=self.volatility_window,
                                                 min_periods=self.volatility_window).std() * 100

        # Use vectorized operations for the bulk calculations
        df['max_price_next'] = df['Close'].rolling(window=self.prediction_window,
                                                   min_periods=self.prediction_window).max().shift(
            -self.prediction_window)
        df['min_price_next'] = df['Close'].rolling(window=self.prediction_window,
                                                   min_periods=self.prediction_window).min().shift(
            -self.prediction_window)

        # Calculate potential gains and losses
        df['potential_gain'] = (df['max_price_next'] / df['Close'] - 1) * 100
        df['potential_loss'] = (df['min_price_next'] / df['Close'] - 1) * 100

        # Risk-reward ratio
        df['risk_reward'] = df['potential_gain'] / abs(df['potential_loss'].where(df['potential_loss'] < 0, -0.01))

        # Target calculation - risk-adjusted potential gain
        df['target'] = df['potential_gain'] / (
                df['volatility'] + abs(df['potential_loss'].where(df['potential_loss'] < 0, -0.01)))

        # Simple price targets
        df['target_price'] = df['Close'].shift(-self.prediction_window)
        df['target_return'] = df['target_price'] / df['Close'] - 1

        # Log initial data shape
        initial_rows = len(df)
        logger.info(f"Initial data shape: {df.shape}")

        # Simply drop all rows with any NaN values
        clean_df = df.dropna().copy()

        # Log how many rows were removed
        rows_removed = initial_rows - len(clean_df)
        logger.info(f"Removed {rows_removed} rows with missing values ({rows_removed / initial_rows:.1%} of data)")
        logger.info(f"Final data shape: {clean_df.shape}")

        return clean_df

    def prepare_data_for_tft(self, data: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
        """
        Prepare data for Temporal Fusion Transformer without normalization.

        Args:
            data: Processed DataFrame with all features and target

        Returns:
            Tuple of (DataFrame prepared for TFT, list of numerical features)
        """
        try:
            # Create a copy to avoid modifying the original
            df = data.copy()

            # Create a continuous time index (required by TFT)
            df['time_idx'] = np.arange(len(df))

            # Add series ID
            df['series_id'] = self.static_features['stock_symbol']

            # Add static features
            df['sector'] = self.static_features['sector']
            df['market_cap'] = self.static_features['market_cap']

            # Convert categorical features to string
            categorical_cols = ['day_of_week', 'month', 'is_quarter_end', 'sector']
            for col in categorical_cols:
                if col in df.columns:
                    df[col] = df[col].astype(str)

            # Define features to normalize
            numerical_features = [
                'Open', 'High', 'Low', 'Close', 'Volume',
                'MA10', 'MA50', 'MA200', 'returns',
                'volatility_20d', 'volatility_50d',
                'volume_ma10', 'volume_ratio', 'bb_width',
                'rsi', 'macd', 'macd_signal', 'macd_hist',
                'potential_gain', 'potential_loss', 'volatility'
            ]

            # Check which numerical features are actually in the dataframe
            available_features = [feat for feat in numerical_features if feat in df.columns]

            # Handle NaN values
            non_na_idx = ~df[available_features + ['target']].isna().any(axis=1)
            df = df[non_na_idx].reset_index(drop=True)

            logger.info(f"Data prepared for TFT with shape: {df.shape}")

            self.tft_data = df
            return df, available_features

        except Exception as e:
            logger.error(f"Error in data preparation: {e}")
            raise

    def normalize_data(self, df: pd.DataFrame, features_to_normalize: List[str]) -> pd.DataFrame:
        """
        Normalize data for training.

        Args:
            df: DataFrame to normalize
            features_to_normalize: List of features to normalize

        Returns:
            Normalized DataFrame
        """
        try:
            # Create a copy to avoid modifying the original
            normalized_df = df.copy()

            # Calculate validation and test sizes
            validation_size = int(len(df) * self.config.get("validation_size"))
            test_size = int(len(df) * self.config.get("test_size"))
            train_size = len(df) - validation_size - test_size

            # Get training indices
            train_indices = range(train_size)

            # Initialize StandardScaler
            scaler = StandardScaler()

            # Fit scaler using only training data
            scaler.fit(normalized_df.iloc[train_indices][features_to_normalize])

            # Transform all data with the scaler fitted on training data
            normalized_df[features_to_normalize] = scaler.transform(normalized_df[features_to_normalize])

            # Save the scaler
            scaler_path = os.path.join(self.data_dir, f'{self.symbol}_scaler.pkl')
            joblib.dump(scaler, scaler_path)
            logger.info(f"Scaler fitted on training data only and saved to {scaler_path}")

            self.scaler = scaler
            self.normalized_data = normalized_df

            return normalized_df

        except Exception as e:
            logger.error(f"Error in data normalization: {e}")
            raise

    def create_tft_datasets(self) -> Tuple[TimeSeriesDataSet, TimeSeriesDataSet, TimeSeriesDataSet]:
        """
        Create training, validation, and test datasets for TFT.

        Returns:
            Tuple of (training dataset, validation dataset, test dataset)
        """
        try:
            # Ensure normalized data exists
            if self.normalized_data is None:
                raise ValueError("Normalized data not available. Run prepare_data() first.")

            # Get the data
            data = self.normalized_data

            # Calculate dataset sizes
            validation_size = int(len(data) * self.config.get("validation_size"))
            test_size = int(len(data) * self.config.get("test_size"))
            train_size = len(data) - validation_size - test_size

            logger.info(f"Creating datasets with train: {train_size}, val: {validation_size}, test: {test_size}")

            # Split data
            train_data = data.iloc[:train_size].copy()
            val_data = data.iloc[train_size:train_size + validation_size].copy()
            test_data = data.iloc[train_size + validation_size:].copy()

            # Configure encoder length
            max_encoder_length = self.config.get("encoder_length")
            max_prediction_length = self.config.get("prediction_window")

            # Check if we have enough data for the specified encoder length
            min_required_points = max_encoder_length + max_prediction_length

            if len(val_data) < min_required_points or len(test_data) < min_required_points:
                # Reduce encoder length if data is insufficient
                original_encoder_length = max_encoder_length
                max_encoder_length = min(30, min(len(val_data), len(test_data)) // 2)
                logger.warning(
                    f"Reducing encoder length from {original_encoder_length} to {max_encoder_length} due to limited data")
                self.config.set("encoder_length", max_encoder_length)

            # Define features
            time_varying_known_reals = [
                'Open', 'High', 'Low', 'Volume',
                'MA10', 'MA50', 'MA200',
                'volatility_20d', 'volatility_50d',
                'volume_ma10', 'volume_ratio',
                'rsi', 'macd', 'macd_signal', 'macd_hist',
                'upper_band', 'lower_band', 'bb_width'
            ]

            time_varying_unknown_reals = [
                'Close', 'returns', 'potential_gain', 'potential_loss', 'volatility', 'target', 'target_return'
            ]

            # Filter for columns that actually exist in the data
            time_varying_known_reals = [col for col in time_varying_known_reals if col in data.columns]
            time_varying_unknown_reals = [col for col in time_varying_unknown_reals if col in data.columns]

            # Define training dataset
            training = TimeSeriesDataSet(
                data=train_data,
                time_idx="time_idx",
                target="target",  # Primary target is our risk-adjusted metric
                group_ids=["series_id"],
                min_encoder_length=max_encoder_length // 2,
                max_encoder_length=max_encoder_length,
                min_prediction_length=1,
                max_prediction_length=max_prediction_length,
                static_categoricals=["series_id", "sector"],
                static_reals=["market_cap"],
                time_varying_known_categoricals=["day_of_week", "month", "is_quarter_end"],
                time_varying_known_reals=time_varying_known_reals,
                time_varying_unknown_reals=time_varying_unknown_reals,
                target_normalizer=TorchNormalizer(),
                add_relative_time_idx=True,
                add_target_scales=True,
                add_encoder_length=True,
                allow_missing_timesteps=True,
            )

            # Create validation dataset
            validation = TimeSeriesDataSet.from_dataset(
                training, val_data, predict=True, stop_randomization=True
            )

            # Create test dataset
            testing = TimeSeriesDataSet.from_dataset(
                training, test_data, predict=True, stop_randomization=True
            )

            # Create data loaders
            batch_size = self.config.get("batch_size")
            train_dataloader = training.to_dataloader(train=True, batch_size=batch_size)
            val_dataloader = validation.to_dataloader(train=False, batch_size=batch_size)
            test_dataloader = testing.to_dataloader(train=False, batch_size=batch_size)

            logger.info(f"Created data loaders with batch size {batch_size}")

            # Store datasets and dataloaders
            self.training_dataset = training
            self.validation_dataset = validation
            self.test_dataset = testing
            self.train_dataloader = train_dataloader
            self.val_dataloader = val_dataloader
            self.test_dataloader = test_dataloader

            return training, validation, testing

        except Exception as e:
            logger.error(f"Error creating TFT datasets: {e}")
            import traceback
            logger.error(traceback.format_exc())
            raise

    def prepare_data(self) -> Dict[str, Any]:
        """
        Complete data preparation pipeline.

        Returns:
            Dictionary with all prepared data components
        """
        try:
            # 1. Get static features
            logger.info("Loading static features...")
            self.static_features = self.get_static_features()
            logger.info(f"Static features: {self.static_features}")

            # 2. Load stock data
            logger.info(f"Loading data for {self.symbol}...")
            self.raw_data = self.load_stock_data()
            if self.raw_data is None or self.raw_data.empty:
                logger.error("Failed to load data")
                return None

            # 3. Calculate technical indicators
            logger.info("Calculating technical indicators...")
            data_with_indicators = self.calculate_technical_indicators(self.raw_data)

            # 4. Calculate target variables
            logger.info("Calculating target variables...")
            self.processed_data = self.calculate_target(data_with_indicators)

            # Save processed data
            processed_data_path = os.path.join(self.data_dir, f"{self.symbol}_processed.csv")
            self.processed_data.to_csv(processed_data_path, index=False)
            logger.info(f"Processed data saved to {processed_data_path}")

            # 5. Prepare data for TFT
            logger.info("Preparing data for TFT...")
            self.tft_data, numerical_features = self.prepare_data_for_tft(self.processed_data)

            # 6. Normalize data
            logger.info("Normalizing data...")
            self.normalized_data = self.normalize_data(self.tft_data, numerical_features)

            # Save normalized data
            tft_data_path = os.path.join(self.data_dir, f"{self.symbol}_tft_ready.csv")
            self.normalized_data.to_csv(tft_data_path, index=False)
            logger.info(f"TFT-ready data saved to {tft_data_path}")

            # 7. Create TFT datasets
            logger.info("Creating TFT datasets...")
            self.create_tft_datasets()

            logger.info("Data preparation complete!")

            # Return objects for further processing
            return {
                'static_features': self.static_features,
                'raw_data': self.raw_data,
                'processed_data': self.processed_data,
                'tft_data': self.tft_data,
                'normalized_data': self.normalized_data,
                'training': self.training_dataset,
                'validation': self.validation_dataset,
                'testing': self.test_dataset,
                'train_dataloader': self.train_dataloader,
                'val_dataloader': self.val_dataloader,
                'test_dataloader': self.test_dataloader,
                'scaler': self.scaler
            }

        except Exception as e:
            logger.error(f"Error in data preparation: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return None


class ModelManager:
    """Handles TFT model creation, training, evaluation, and prediction."""

    def __init__(self, config_manager: ConfigManager, data_manager: DataManager):
        """
        Initialize ModelManager with configuration and data.

        Args:
            config_manager: Configuration manager instance
            data_manager: Data manager instance
        """
        self.config = config_manager
        self.data_manager = data_manager
        self.symbol = self.config.get("stock_symbol")
        self.data_dir = self.config.get("data_directory")

        # Will store model components
        self.model = None
        self.trainer = None
        self.best_model = None

        logger.info(f"ModelManager initialized for {self.symbol}")

    def create_tft_model(self) -> TemporalFusionTransformer:
        """
        Create and configure Temporal Fusion Transformer model.

        Returns:
            Configured TFT model
        """
        try:
            # Ensure training dataset exists
            if self.data_manager.training_dataset is None:
                raise ValueError("Training dataset not available. Run data_manager.prepare_data() first.")

            training = self.data_manager.training_dataset

            logger.info("Creating TFT model...")

            # Get hyperparameters from config
            hidden_size = self.config.get("hidden_size")
            attention_head_size = self.config.get("attention_head_size")
            dropout = self.config.get("dropout")
            hidden_continuous_size = self.config.get("hidden_continuous_size")
            learning_rate = self.config.get("learning_rate")

            # Define quantile loss
            quantiles = self.config.get("quantiles")
            loss = QuantileLoss(quantiles=quantiles)

            # Define model with hyperparameters
            tft = TemporalFusionTransformer.from_dataset(
                training,
                learning_rate=learning_rate,
                hidden_size=hidden_size,
                attention_head_size=attention_head_size,
                dropout=dropout,
                hidden_continuous_size=hidden_continuous_size,
                loss=loss,
                log_interval=10,
                reduce_on_plateau_patience=5,
                optimizer="ranger",
                logging_metrics=torch.nn.ModuleList([
                    SMAPE(),
                    MAE(),
                    RMSE()
                ]),
            )

            logger.info(f"Model created with parameters: hidden_size={hidden_size}, "
                        f"attention_head_size={attention_head_size}, dropout={dropout}")
            logger.info(f"Total parameters: {tft.size() / 1e3:.1f}k")

            self.model = tft
            return tft

        except Exception as e:
            logger.error(f"Error creating TFT model: {e}")
            raise

    def set_up_trainer(self) -> pl.Trainer:
        """
        Set up PyTorch Lightning trainer with callbacks.

        Returns:
            Configured PyTorch Lightning trainer
        """
        try:
            # Ensure model exists
            if self.model is None:
                raise ValueError("Model not available. Run create_tft_model() first.")

            # Define checkpoint directory
            checkpoint_dir = os.path.join(self.data_dir, 'models')
            os.makedirs(checkpoint_dir, exist_ok=True)

            logger.info(f"Setting up trainer with checkpoints in {checkpoint_dir}")

            # Define callbacks
            early_stop_callback = EarlyStopping(
                monitor="val_loss",
                min_delta=1e-4,
                patience=10,
                verbose=True,
                mode="min"
            )

            checkpoint_callback = ModelCheckpoint(
                dirpath=checkpoint_dir,
                filename=f"{self.symbol}_tft_{{epoch}}_{{val_loss:.2f}}",
                save_top_k=3,
                verbose=True,
                monitor="val_loss",
                mode="min"
            )

            lr_monitor = LearningRateMonitor(logging_interval='epoch')

            # Configure trainer logger
            logger_callback = TensorBoardLogger(
                save_dir=os.path.join(checkpoint_dir, 'logs')
            )

            # Determine if GPU is available
            if torch.cuda.is_available():
                accelerator = 'gpu'
                devices = 1
                logger.info("Using GPU for training")
            else:
                accelerator = 'cpu'
                devices = 1
                logger.info("Using CPU for training")

            # Create trainer
            trainer = pl.Trainer(
                max_epochs=self.config.get("max_epochs"),
                accelerator=accelerator,
                devices=devices,
                callbacks=[early_stop_callback, checkpoint_callback, lr_monitor],
                gradient_clip_val=0.1,
                logger=logger_callback,
                log_every_n_steps=10,
            )

            logger.info(f"Trainer configured with max_epochs={self.config.get('max_epochs')}, "
                        f"gradient_clip_val=0.1")

            self.trainer = trainer
            return trainer

        except Exception as e:
            logger.error(f"Error setting up trainer: {e}")
            raise

    def train_model(self) -> TemporalFusionTransformer:
        """
        Train the Temporal Fusion Transformer model.

        Returns:
            Best trained model
        """
        try:
            # Ensure model and trainer exist
            if self.model is None or self.trainer is None:
                raise ValueError("Model or trainer not available. Run create_tft_model() and set_up_trainer() first.")

            # Ensure data loaders exist
            if self.data_manager.train_dataloader is None or self.data_manager.val_dataloader is None:
                raise ValueError("Data loaders not available. Run data_manager.prepare_data() first.")

            train_dataloader = self.data_manager.train_dataloader
            val_dataloader = self.data_manager.val_dataloader

            # Log start of training
            start_time = datetime.now()
            logger.info(f"Starting model training at {start_time}")

            # Fit the model
            self.trainer.fit(
                self.model,
                train_dataloaders=train_dataloader,
                val_dataloaders=val_dataloader
            )

            # Log completion and duration
            end_time = datetime.now()
            duration = end_time - start_time
            logger.info(f"Training completed at {end_time}")
            logger.info(f"Total training time: {duration}")

            # Get best model path
            best_model_path = self.trainer.checkpoint_callback.best_model_path
            if best_model_path:
                logger.info(f"Best model saved at: {best_model_path}")

                # Load the best model
                best_model = TemporalFusionTransformer.load_from_checkpoint(best_model_path)
                logger.info("Best model loaded from checkpoint")

                # Save model configuration
                model_config_path = os.path.join(self.data_dir, 'models', f"{self.symbol}_model_config.json")
                with open(model_config_path, 'w') as f:
                    json.dump({
                        'checkpoint_path': best_model_path,
                        'hidden_size': best_model.hparams.hidden_size,
                        'attention_head_size': best_model.hparams.attention_head_size,
                        'dropout': best_model.hparams.dropout,
                        'hidden_continuous_size': best_model.hparams.hidden_continuous_size,
                        'learning_rate': best_model.hparams.learning_rate,
                        'prediction_window': self.config.get("prediction_window"),
                        'encoder_length': self.config.get("encoder_length"),
                        'volatility_window': self.config.get("volatility_window"),
                        'train_date': str(datetime.now())
                    }, f, indent=2)
                logger.info(f"Model configuration saved to {model_config_path}")

                self.best_model = best_model
                return best_model
            else:
                logger.warning("No best model checkpoint found, returning last model state")
                self.best_model = self.model
                return self.model

        except Exception as e:
            logger.error(f"Error during model training: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return None

    def evaluate_model(self, model: Optional[TemporalFusionTransformer] = None) -> Dict[str, float]:
        """
        Evaluate model performance on test dataset.

        Args:
            model: Model to evaluate (uses self.best_model if None)

        Returns:
            Dictionary of evaluation metrics
        """
        try:
            # Use provided model or fall back to best model
            if model is None:
                if self.best_model is None:
                    raise ValueError("No model available for evaluation. Train a model first.")
                model = self.best_model

            # Ensure test dataloader exists
            if self.data_manager.test_dataloader is None:
                raise ValueError("Test dataloader not available. Run data_manager.prepare_data() first.")

            test_dataloader = self.data_manager.test_dataloader

            logger.info("Evaluating model on test dataset...")

            # Create a new trainer for evaluation
            if torch.cuda.is_available():
                accelerator = 'gpu'
                devices = 1
            else:
                accelerator = 'cpu'
                devices = 1

            eval_trainer = pl.Trainer(
                accelerator=accelerator,
                devices=devices,
                enable_progress_bar=True,
                enable_model_summary=False,
            )

            # Run evaluation
            results = eval_trainer.test(model, dataloaders=test_dataloader)

            if results and len(results) > 0:
                metrics = results[0]
                logger.info(f"Evaluation metrics: {metrics}")

                # Save metrics
                metrics_path = os.path.join(self.data_dir, 'results', f"{self.symbol}_evaluation_metrics.json")
                os.makedirs(os.path.dirname(metrics_path), exist_ok=True)
                with open(metrics_path, 'w') as f:
                    json.dump(metrics, f, indent=2)
                logger.info(f"Evaluation metrics saved to {metrics_path}")

                return metrics
            else:
                logger.warning("No evaluation results returned")
                return {}

        except Exception as e:
            logger.error(f"Error during model evaluation: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return {}

    def predict(self, days_to_predict: int = 5) -> pd.DataFrame:
        """
        Generate predictions for future days.

        Args:
            days_to_predict: Number of days to predict ahead

        Returns:
            DataFrame with predictions
        """
        try:
            # Ensure best model exists
            if self.best_model is None:
                raise ValueError("No trained model available. Train a model first.")

            # Ensure data is available
            if self.data_manager.normalized_data is None:
                raise ValueError("Normalized data not available. Run data_manager.prepare_data() first.")

            logger.info(f"Generating predictions for next {days_to_predict} days...")

            # Get the most recent data for prediction
            prediction_data = self.data_manager.normalized_data.copy()

            # Prepare prediction data
            encoder_length = self.config.get("encoder_length")

            # Use the most recent data points as context
            context_length = min(encoder_length, len(prediction_data))
            context_data = prediction_data.iloc[-context_length:].copy()

            # Set up prediction dataloader
            context_dataset = TimeSeriesDataSet.from_dataset(
                self.data_manager.training_dataset,
                context_data,
                predict=True,
                stop_randomization=True
            )
            context_dataloader = context_dataset.to_dataloader(train=False, batch_size=1)

            # Generate predictions
            raw_predictions = self.best_model.predict(context_dataloader)

            # Extract quantile predictions
            predictions_dict = {}
            quantiles = self.config.get("quantiles")

            for idx, q in enumerate(quantiles):
                q_name = f"q{int(q * 100)}"
                predictions_dict[q_name] = raw_predictions[idx].numpy()

            # Create prediction dataframe
            last_date = self.data_manager.processed_data['Date'].iloc[-1]
            future_dates = [last_date + timedelta(days=i + 1) for i in range(days_to_predict)]

            # Filter out weekend days (if desired)
            # future_dates = [date for date in future_dates if date.weekday() < 5]

            pred_df = pd.DataFrame({
                'Date': future_dates[:days_to_predict],
                'Symbol': self.symbol
            })

            # Add predictions for each quantile
            for q_name, values in predictions_dict.items():
                pred_df[f'target_{q_name}'] = values[0, :days_to_predict]

            # Calculate actual price predictions
            last_close = self.data_manager.processed_data['Close'].iloc[-1]

            # Add additional metrics based on the primary target
            pred_df['last_close'] = last_close

            # Save predictions
            predictions_path = os.path.join(self.data_dir, 'results', f"{self.symbol}_predictions.csv")
            os.makedirs(os.path.dirname(predictions_path), exist_ok=True)
            pred_df.to_csv(predictions_path, index=False)
            logger.info(f"Predictions saved to {predictions_path}")

            return pred_df

        except Exception as e:
            logger.error(f"Error during prediction: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return pd.DataFrame()

    def hyperparameter_optimization(self, num_trials: int = 20, cpus_per_trial: int = 1) -> Dict[str, Any]:
        """
        Perform hyperparameter optimization for the TFT model.

        Args:
            num_trials: Number of trials to run
            cpus_per_trial: CPUs to allocate per trial

        Returns:
            Dictionary with best hyperparameters
        """
        if not self.config.get("enable_hyperopt"):
            logger.info("Hyperparameter optimization disabled in config")
            return {}

        try:
            # Ensure data is prepared
            if self.data_manager.training_dataset is None:
                raise ValueError("Training dataset not available. Run data_manager.prepare_data() first.")

            logger.info(f"Starting hyperparameter optimization with {num_trials} trials...")

            # Define hyperparameter search space
            config = {
                "hidden_size": tune.choice([32, 64, 128, 256]),
                "attention_head_size": tune.choice([1, 2, 4, 8]),
                "dropout": tune.uniform(0.1, 0.3),
                "hidden_continuous_size": tune.choice([16, 32, 64]),
                "learning_rate": tune.loguniform(1e-4, 1e-2),
            }

            def train_tft(config, checkpoint_dir=None):
                # Create model with the given hyperparameters
                model = TemporalFusionTransformer.from_dataset(
                    self.data_manager.training_dataset,
                    learning_rate=config["learning_rate"],
                    hidden_size=config["hidden_size"],
                    attention_head_size=config["attention_head_size"],
                    dropout=config["dropout"],
                    hidden_continuous_size=config["hidden_continuous_size"],
                    loss=QuantileLoss(quantiles=self.config.get("quantiles")),
                    log_interval=10,
                    reduce_on_plateau_patience=5,
                    optimizer="ranger",
                )

                # Set up callbacks
                callbacks = [
                    EarlyStopping(
                        monitor="val_loss",
                        min_delta=1e-4,
                        patience=5,
                        verbose=False,
                        mode="min"
                    ),
                    TuneReportCallback(
                        {"val_loss": "val_loss"},
                        on="validation_end"
                    )
                ]

                # Set up trainer
                if torch.cuda.is_available():
                    accelerator = 'gpu'
                    devices = 1
                else:
                    accelerator = 'cpu'
                    devices = 1

                trainer = pl.Trainer(
                    max_epochs=20,
                    accelerator=accelerator,
                    devices=devices,
                    callbacks=callbacks,
                    gradient_clip_val=0.1,
                    enable_progress_bar=False,
                    enable_model_summary=False,
                    logger=False,
                )

                # Train model
                trainer.fit(
                    model,
                    train_dataloaders=self.data_manager.train_dataloader,
                    val_dataloaders=self.data_manager.val_dataloader
                )

            # Set up scheduler
            scheduler = ASHAScheduler(
                metric="val_loss",
                mode="min",
                max_t=20,
                grace_period=5,
                reduction_factor=2
            )

            # Run hyperparameter search
            analysis = tune.run(
                train_tft,
                config=config,
                scheduler=scheduler,
                num_samples=num_trials,
                resources_per_trial={"cpu": cpus_per_trial, "gpu": 0.5 if torch.cuda.is_available() else 0},
                local_dir=os.path.join(self.data_dir, 'hyperopt'),
                progress_reporter=tune.CLIReporter(
                    parameter_columns=["hidden_size", "attention_head_size", "dropout", "learning_rate"],
                    metric_columns=["val_loss", "training_iteration"]
                )
            )

            # Get best config
            best_config = analysis.get_best_config(metric="val_loss", mode="min")
            logger.info(f"Best hyperparameters: {best_config}")

            # Save best config
            best_config_path = os.path.join(self.data_dir, 'models', f"{self.symbol}_best_hyperparameters.json")
            with open(best_config_path, 'w') as f:
                json.dump(best_config, f, indent=2)
            logger.info(f"Best hyperparameters saved to {best_config_path}")

            # Update config with best hyperparameters
            for key, value in best_config.items():
                self.config.set(key, value)

            return best_config

        except Exception as e:
            logger.error(f"Error during hyperparameter optimization: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return {}

    def save_model(self, model_path: Optional[str] = None) -> str:
        """
        Save the model to disk.

        Args:
            model_path: Path where to save the model (optional)

        Returns:
            Path where the model was saved
        """
        try:
            # Ensure model exists
            if self.best_model is None:
                raise ValueError("No model available to save. Train a model first.")

            # Define model path if not provided
            if model_path is None:
                model_path = os.path.join(self.data_dir, 'models', f"{self.symbol}_best_model.ckpt")

            # Ensure directory exists
            os.makedirs(os.path.dirname(model_path), exist_ok=True)

            # Save model
            torch.save(self.best_model.state_dict(), model_path)
            logger.info(f"Model saved to {model_path}")

            return model_path

        except Exception as e:
            logger.error(f"Error saving model: {e}")
            raise

    def load_model(self, model_path: str) -> TemporalFusionTransformer:
        """
        Load a saved model from disk.

        Args:
            model_path: Path to the saved model

        Returns:
            Loaded model
        """
        try:
            # Ensure model path exists
            if not os.path.exists(model_path):
                raise ValueError(f"Model path {model_path} does not exist")

            # Load model
            model = TemporalFusionTransformer.load_from_checkpoint(model_path)
            logger.info(f"Model loaded from {model_path}")

            self.best_model = model
            return model

        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise

    def interpret_prediction(self, prediction_df: pd.DataFrame) -> Dict[str, Any]:
        """
        Interpret the model predictions.

        Args:
            prediction_df: DataFrame with predictions

        Returns:
            Dictionary with interpretation
        """
        try:
            # Ensure prediction dataframe exists and has expected columns
            if prediction_df.empty:
                raise ValueError("Prediction dataframe is empty")

            # Extract main prediction (median quantile)
            median_col = [col for col in prediction_df.columns if 'q50' in col]
            if not median_col:
                raise ValueError("No median prediction column found")

            median_prediction = prediction_df[median_col[0]].mean()

            # Extract confidence interval
            lower_col = [col for col in prediction_df.columns if 'q10' in col]
            upper_col = [col for col in prediction_df.columns if 'q90' in col]

            if lower_col and upper_col:
                lower_bound = prediction_df[lower_col[0]].mean()
                upper_bound = prediction_df[upper_col[0]].mean()
                confidence_interval = (upper_bound - lower_bound)
            else:
                lower_bound = None
                upper_bound = None
                confidence_interval = None

            # Determine trend direction
            last_close = prediction_df['last_close'].iloc[0] if 'last_close' in prediction_df.columns else None

            if last_close is not None and median_prediction is not None:
                if median_prediction > 0:
                    trend = "upward"
                    intensity = abs(median_prediction)
                elif median_prediction < 0:
                    trend = "downward"
                    intensity = abs(median_prediction)
                else:
                    trend = "neutral"
                    intensity = 0
            else:
                trend = "unknown"
                intensity = None

            # Formulate interpretation
            interpretation = {
                "trend": trend,
                "intensity": intensity,
                "median_prediction": median_prediction,
                "lower_bound": lower_bound,
                "upper_bound": upper_bound,
                "confidence_interval": confidence_interval,
                "prediction_dates": prediction_df['Date'].tolist(),
            }

            logger.info(f"Prediction interpretation: {interpretation}")
            return interpretation

        except Exception as e:
            logger.error(f"Error interpreting prediction: {e}")
            return {"trend": "unknown", "error": str(e)}

    def visualize_prediction(self, prediction_df: pd.DataFrame,
                             historical_df: Optional[pd.DataFrame] = None,
                             save_path: Optional[str] = None) -> plt.Figure:
        """
        Visualize the model predictions.

        Args:
            prediction_df: DataFrame with predictions
            historical_df: DataFrame with historical data (optional)
            save_path: Path where to save the visualization (optional)

        Returns:
            Matplotlib figure
        """
        try:
            # Set up plotting style
            plt.style.use('seaborn-v0_8-darkgrid')

            # Create figure and axis
            fig, ax = plt.subplots(figsize=(12, 6))

            # If historical data is provided, plot it
            if historical_df is not None:
                # Ensure Date column is datetime
                historical_df['Date'] = pd.to_datetime(historical_df['Date'])

                # Plot historical prices
                ax.plot(historical_df['Date'][-30:], historical_df['Close'][-30:],
                        label='Historical Close Price', color='#1f77b4', linewidth=2)

            # Ensure prediction dates are datetime
            prediction_df['Date'] = pd.to_datetime(prediction_df['Date'])

            # Get prediction columns
            q50_col = [col for col in prediction_df.columns if 'q50' in col][0]
            q10_col = [col for col in prediction_df.columns if 'q10' in col][0] if any(
                'q10' in col for col in prediction_df.columns) else None
            q90_col = [col for col in prediction_df.columns if 'q90' in col][0] if any(
                'q90' in col for col in prediction_df.columns) else None

            # Calculate prediction values
            last_close = prediction_df['last_close'].iloc[0] if 'last_close' in prediction_df.columns else 0

            # Prepare prediction series to plot
            prediction_df['predicted_close'] = last_close * (1 + prediction_df[q50_col] / 100)

            if q10_col and q90_col:
                prediction_df['lower_bound'] = last_close * (1 + prediction_df[q10_col] / 100)
                prediction_df['upper_bound'] = last_close * (1 + prediction_df[q90_col] / 100)

                # Plot confidence interval
                ax.fill_between(prediction_df['Date'],
                                prediction_df['lower_bound'],
                                prediction_df['upper_bound'],
                                alpha=0.3, color='#ff7f0e',
                                label='90% Confidence Interval')

            # Plot median prediction
            ax.plot(prediction_df['Date'], prediction_df['predicted_close'],
                    label='Predicted Close Price', color='#ff7f0e',
                    linewidth=2, marker='o')

            # Add last known price point to connect historical and prediction
            if historical_df is not None:
                last_historical_date = historical_df['Date'].iloc[-1]
                last_historical_close = historical_df['Close'].iloc[-1]

                # Plot connection point
                ax.scatter(last_historical_date, last_historical_close,
                           color='#ff7f0e', s=80, zorder=5)

            # Customize plot
            ax.set_title(f'{self.symbol} Stock Price Forecast', fontsize=16)
            ax.set_xlabel('Date', fontsize=12)
            ax.set_ylabel('Price', fontsize=12)
            ax.grid(True, alpha=0.3)
            ax.legend(loc='best', fontsize=10)

            # Format x-axis dates
            fig.autofmt_xdate()

            # Adjust layout
            plt.tight_layout()

            # Save figure if path provided
            if save_path:
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                logger.info(f"Prediction visualization saved to {save_path}")

            return fig

        except Exception as e:
            logger.error(f"Error visualizing prediction: {e}")
            import traceback
            logger.error(traceback.format_exc())

            # Return a simple figure with error message
            fig, ax = plt.subplots(figsize=(8, 6))
            ax.text(0.5, 0.5, f"Error visualizing prediction: {e}",
                    horizontalalignment='center', verticalalignment='center',
                    transform=ax.transAxes, fontsize=12)
            plt.tight_layout()
            return fig


class StockForecastingPipeline:
    """End-to-end pipeline for stock forecasting."""

    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize the pipeline with configuration.

        Args:
            config_path: Path to a YAML configuration file (optional)
        """
        # Initialize configuration
        self.config_manager = ConfigManager(config_path)

        # Initialize data manager
        self.data_manager = DataManager(self.config_manager)

        # Initialize model manager
        self.model_manager = ModelManager(self.config_manager, self.data_manager)

        logger.info("StockForecastingPipeline initialized")

    def run_training_pipeline(self) -> Dict[str, Any]:
        """
        Run the complete training pipeline.

        Returns:
            Dictionary with pipeline results
        """
        try:
            # 1. Prepare data
            logger.info("Starting data preparation...")
            data_results = self.data_manager.prepare_data()
            if data_results is None:
                logger.error("Data preparation failed")
                return {"status": "error", "message": "Data preparation failed"}

            # 2. Optionally run hyperparameter optimization
            if self.config_manager.get("enable_hyperopt"):
                logger.info("Running hyperparameter optimization...")
                hyperopt_results = self.model_manager.hyperparameter_optimization(
                    num_trials=self.config_manager.get("hyperopt_trials"),
                    cpus_per_trial=self.config_manager.get("hyperopt_cpu")
                )

                if not hyperopt_results:
                    logger.warning("Hyperparameter optimization failed or was skipped")

            # 3. Create TFT model
            logger.info("Creating TFT model...")
            self.model_manager.create_tft_model()

            # 4. Set up trainer
            logger.info("Setting up trainer...")
            self.model_manager.set_up_trainer()

            # 5. Train model
            logger.info("Training model...")
            best_model = self.model_manager.train_model()
            if best_model is None:
                logger.error("Model training failed")
                return {"status": "error", "message": "Model training failed"}

            # 6. Evaluate model
            logger.info("Evaluating model...")
            evaluation_metrics = self.model_manager.evaluate_model()

            # 7. Save model
            logger.info("Saving model...")
            model_path = self.model_manager.save_model()

            # 8. Return results
            return {
                "status": "success",
                "model_path": model_path,
                "evaluation_metrics": evaluation_metrics,
                "config": self.config_manager.config
            }

        except Exception as e:
            logger.error(f"Error in training pipeline: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return {"status": "error", "message": str(e)}

    def run_prediction_pipeline(self, days_to_predict: int = 5) -> Dict[str, Any]:
        """
        Run the prediction pipeline.

        Args:
            days_to_predict: Number of days to predict ahead

        Returns:
            Dictionary with prediction results
        """
        try:
            # If model not loaded, try to load from default path
            if self.model_manager.best_model is None:
                model_path = os.path.join(
                    self.config_manager.get("data_directory"),
                    'models',
                    f"{self.config_manager.get('stock_symbol')}_best_model.ckpt"
                )

                # Check if there's a saved model configuration
                model_config_path = os.path.join(
                    self.config_manager.get("data_directory"),
                    'models',
                    f"{self.config_manager.get('stock_symbol')}_model_config.json"
                )

                if os.path.exists(model_config_path):
                    with open(model_config_path, 'r') as f:
                        model_config = json.load(f)
                        if 'checkpoint_path' in model_config and os.path.exists(model_config['checkpoint_path']):
                            model_path = model_config['checkpoint_path']

                if os.path.exists(model_path):
                    logger.info(f"Loading model from {model_path}")
                    self.model_manager.load_model(model_path)
                else:
                    logger.warning(f"No model found at {model_path}, will need to train a new model")

                    # If no saved model, run training pipeline
                    training_results = self.run_training_pipeline()
                    if training_results["status"] != "success":
                        return {"status": "error", "message": "Failed to train model for prediction"}

            # Ensure data is prepared
            if self.data_manager.processed_data is None:
                logger.info("Data not prepared, preparing now...")
                self.data_manager.prepare_data()

            # Generate predictions
            logger.info(f"Generating predictions for next {days_to_predict} days...")
            prediction_df = self.model_manager.predict(days_to_predict)

            if prediction_df.empty:
                return {"status": "error", "message": "Failed to generate predictions"}

            # Interpret predictions
            interpretation = self.model_manager.interpret_prediction(prediction_df)

            # Visualize predictions
            viz_path = os.path.join(
                self.config_manager.get("data_directory"),
                'results',
                f"{self.config_manager.get('stock_symbol')}_prediction_viz.png"
            )

            fig = self.model_manager.visualize_prediction(
                prediction_df,
                self.data_manager.processed_data,
                viz_path
            )

            # Return results
            return {
                "status": "success",
                "predictions": prediction_df.to_dict(orient='records'),
                "interpretation": interpretation,
                "visualization_path": viz_path
            }

        except Exception as e:
            logger.error(f"Error in prediction pipeline: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return {"status": "error", "message": str(e)}


def main():
    """Main execution function."""
    import argparse

    # Set up argument parser
    parser = argparse.ArgumentParser(description='Stock Price Forecasting with TFT')
    parser.add_argument('--config', type=str, help='Path to configuration file')
    parser.add_argument('--mode', type=str, choices=['train', 'predict', 'both'], default='both',
                        help='Pipeline mode: train, predict, or both')
    parser.add_argument('--symbol', type=str, help='Stock symbol to forecast')
    parser.add_argument('--days', type=int, default=5, help='Number of days to predict ahead')

    # Parse arguments
    args = parser.parse_args()

    # Create pipeline
    pipeline = StockForecastingPipeline(args.config)

    # Override stock symbol if provided
    if args.symbol:
        pipeline.config_manager.set("stock_symbol", args.symbol)
        logger.info(f"Stock symbol overridden to {args.symbol}")

    # Run pipeline based on mode
    if args.mode in ['train', 'both']:
        logger.info("Running training pipeline...")
        training_results = pipeline.run_training_pipeline()
        logger.info(f"Training pipeline completed with status: {training_results['status']}")

        if training_results['status'] == 'success':
            logger.info(f"Evaluation metrics: {training_results['evaluation_metrics']}")

    if args.mode in ['predict', 'both']:
        logger.info("Running prediction pipeline...")
        prediction_results = pipeline.run_prediction_pipeline(args.days)
        logger.info(f"Prediction pipeline completed with status: {prediction_results['status']}")

        if prediction_results['status'] == 'success':
            logger.info(f"Prediction interpretation: {prediction_results['interpretation']}")
            logger.info(f"Visualization saved to: {prediction_results['visualization_path']}")

    logger.info("Stock forecasting pipeline completed")
    return 0


if __name__ == "__main__":
    # Set random seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)

    # Run main function
    main()
