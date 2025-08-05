# Import libraries
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split
import keras_tuner as kt
import time

# Check for GPU availability
def check_gpu():
    devices = tf.config.list_physical_devices('GPU')
    if devices:
        print("GPU is available:")
        for device in devices:
            print(f"  - {device.name}")
    else:
        print("No GPU available.")

check_gpu()

# Data Preprocessing Function
def preprocess_data(features: pd.DataFrame, target: pd.Series):
    # Scale features to [0, 1]
    feature_scaler = MinMaxScaler()
    X_scaled = feature_scaler.fit_transform(features)
    
    # Scale target to [0, 1]
    target = target.values.reshape(-1, 1)
    target_scaler = MinMaxScaler()
    y_scaled = target_scaler.fit_transform(target)

    # Split into train/test
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_scaled, test_size=0.2, random_state=42)
    
    # Standardize features after scaling
    std_scaler = StandardScaler()
    X_train = std_scaler.fit_transform(X_train)
    X_test = std_scaler.transform(X_test)

    return X_train, y_train, X_test, y_test

# Load data
data_path = '/uufs/chpc.utah.edu/common/home/akaundal-group3/Vishal/NAM_dat.csv'
data = pd.read_csv(data_path)
features = data.iloc[:, 19:]
target = data.iloc[:, 4]

# Preprocess
X_train, y_train, X_test, y_test = preprocess_data(features, target)

# Build Model Function for Tuning
def build_model(hp):
    model = Sequential()
    
    # Tune number of layers and units per layer
    for i in range(hp.Int("num_layers", 2, 5)):
        units = hp.Int(f'units_{i}', min_value=32, max_value=512, step=32)
        model.add(Dense(units=units, activation='relu'))
        if hp.Boolean(f'dropout_{i}'):
            model.add(Dropout(rate=hp.Float(f'dropout_rate_{i}', 0.1, 0.5, step=0.1)))

    # Output layer
    model.add(Dense(1, activation='linear'))

    # Compile model with tunable learning rate
    lr = hp.Float("lr", 1e-4, 1e-2, sampling="log")
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=lr),
                  loss="mean_absolute_error",
                  metrics=["mean_absolute_error"])
    return model

# Choose tuner type (RandomSearch or BayesianOptimization)
tuner_type = 'bayesian'  # Options: 'random', 'bayesian'

if tuner_type == 'random':
    tuner = kt.RandomSearch(
        build_model,
        objective="val_mean_absolute_error",
        max_trials=50,
        executions_per_trial=4,
        directory='project2',
        project_name='GS_KSdata2')
elif tuner_type == 'bayesian':
    tuner = kt.BayesianOptimization(
        build_model,
        objective="val_mean_absolute_error",
        max_trials=50,
        executions_per_trial=4,
        directory='project2',
        project_name='GS_KSdata2')
else:
    raise ValueError("Invalid tuner_type. Choose 'random' or 'bayesian'.")

# Summarize search space
tuner.search_space_summary()

# Run hyperparameter search
tuner.search(X_train, y_train, epochs=15, validation_data=(X_test, y_test), verbose=1)

# Summarize results
tuner.results_summary()
