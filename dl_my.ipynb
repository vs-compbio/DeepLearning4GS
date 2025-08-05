# --- Environment Setup ---
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# --- Imports ---
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Dropout, Flatten, Conv1D, MaxPooling1D
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.neural_network import MLPRegressor
from keras.wrappers.scikit_learn import KerasRegressor
import time

# --- GPU Check ---
def check_gpu():
    devices = tf.config.list_physical_devices('GPU')
    if devices:
        print("GPU is available")
        for device in devices:
            print(f"Device name: {device.name}")
    else:
        print("No GPU is available")

# --- Data Preprocessing ---
def data_preprocess(dataset, target):
    # Min-Max Scaling
    X_scaled = MinMaxScaler().fit_transform(dataset)
    Y_scaled = MinMaxScaler().fit_transform(target.values.reshape(-1, 1))

    # Train-Test Split
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, Y_scaled, test_size=0.2, random_state=42)

    # Standard Scaling (on features only)
    scaler = StandardScaler().fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)

    return pd.DataFrame(X_train), pd.DataFrame(y_train), pd.DataFrame(X_test), pd.DataFrame(y_test)

# --- Correlation Calculation ---
def cal_correlation(pred, y_test, target):
    pred = pred.reshape(-1, 1)
    target_scaled = MinMaxScaler().fit_transform(target.values.reshape(-1, 1))
    pred_orig = MinMaxScaler().fit(target_scaled).inverse_transform(pred)
    y_test_orig = MinMaxScaler().fit(target_scaled).inverse_transform(y_test)

    pred_series = pd.Series(pred_orig.flatten())
    y_test_series = pd.Series(y_test_orig.flatten())

    correlation = y_test_series.corr(pred_series, method='pearson')
    return correlation

# --- MLP Model Training ---
def multi_layer_perceptron(X_train, y_train, X_test, y_test):
    mlp = MLPRegressor(max_iter=200, early_stopping=True)

    param_grid = {
        'hidden_layer_sizes': [(19, 19, 19), (19, 38, 19), (19, 38, 38, 19), (20, 20, 40, 40, 20),
                               (38, 38, 38, 19), (50, 50, 38), (90, 90, 90), (120, 90, 90)],
        'activation': ['tanh', 'relu', 'identity', 'logistic'],
        'solver': ['sgd', 'adam', 'lbfgs'],
        'alpha': [0.001, 0.05, 0.4],
        'learning_rate': ['constant', 'adaptive']
    }

    clf = GridSearchCV(mlp, param_grid, cv=5, n_jobs=-1)
    clf.fit(X_train, y_train.values.ravel())

    print('Best parameters found:', clf.best_params_)

    pred = clf.predict(X_test)
    score = clf.score(X_test, y_test)
    return score, pred

# --- CNN Model Definition ---
def create_conv_NN(nSNP):
    model = Sequential([
        Conv1D(64, kernel_size=3, strides=3, input_shape=(nSNP, 1), kernel_regularizer='l1_l2'),
        Conv1D(64, kernel_size=3, activation='relu'),
        Dropout(0.2),
        MaxPooling1D(pool_size=2),
        Flatten(),
        Dense(64, activation='relu'),
        Dense(32, activation='linear'),
        Dense(1)
    ])
    model.compile(loss='mean_squared_error', optimizer='adam')
    return model

# --- CNN Model Training ---
def cnn(X_train, y_train, X_test, y_test):
    X_train = np.expand_dims(X_train, axis=2)
    X_test = np.expand_dims(X_test, axis=2)

    nSNP = X_train.shape[1]
    cnn_model = KerasRegressor(build_fn=lambda: create_conv_NN(nSNP), verbose=1)

    param_grid = {'batch_size': [64, 128], 'epochs': [150, 200]}
    grid = GridSearchCV(cnn_model, param_grid, cv=5)

    grid_result = grid.fit(X_train, y_train, validation_split=0.2, callbacks=[EarlyStopping()])
    print('Best parameters:', grid_result.best_params_)

    pred = grid.predict(X_test)
    score = grid.score(X_test, y_test)
    return score, pred

# --- Utility Functions ---
def Average(lst):
    return sum(lst) / len(lst)

# --- Main Execution ---
def main():
    check_gpu()
    start_time = time.time()

    # Load Data
    data = pd.read_csv('/uufs/chpc.utah.edu/common/home/akaundal-group3/Vishal/NAM_dat.csv')
    dataset = data.iloc[:, 19:]
    target = data.iloc[:, 4]

    # Preprocess Data
    X_train, y_train, X_test, y_test = data_preprocess(dataset, target)

    # Initialize Lists to Store Results
    cor_mlp, acc_mlp = [], []
    cor_cnn, acc_cnn = [], []

    # Run for 200 Iterations
    for i in range(200):
        print(f'Iteration {i+1} - MLP:')
        scores_mlp, pred_mlp = multi_layer_perceptron(X_train, y_train, X_test, y_test)
        cor_m = cal_correlation(pred_mlp, y_test, target)
        print('Correlation for MLP:', cor_m)
        cor_mlp.append(cor_m)
        acc_mlp.append(scores_mlp)

        # Uncomment for CNN runs
        # print(f'Iteration {i+1} - CNN:')
        # scores_cnn, pred_cnn = cnn(X_train, y_train, X_test, y_test)
        # cor_c = cal_correlation(pred_cnn, y_test, target)
        # print('Correlation for CNN:', cor_c)
        # cor_cnn.append(cor_c)
        # acc_cnn.append(scores_cnn)

    # Print Average Metrics
    print("Average MLP Accuracy:", Average(acc_mlp))
    print("Average MLP Correlation:", Average(cor_mlp))
    # print("Average CNN Accuracy:", Average(acc_cnn))
    # print("Average CNN Correlation:", Average(cor_cnn))

    print("Total Time Elapsed:", time.time() - start_time)

if __name__ == "__main__":
    main()
