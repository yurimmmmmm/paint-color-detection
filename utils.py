import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error

def get_data(master_data: str):
    selected_columns = [
        "true_R", "true_G", "true_B", 
        "observed_R", "observed_G", "observed_B", 
        "red_R", "red_G", "red_B", 
        "green_R", "green_G", "green_B", 
        "blue_R", "blue_G", "blue_B"
    ]

    train_df = pd.read_csv(os.path.join(master_data, "train.csv"))[selected_columns]
    val_df = pd.read_csv(os.path.join(master_data, "val.csv"))[selected_columns]
    test_df = pd.read_csv(os.path.join(master_data, "test.csv"))[selected_columns]

    train = train_df.to_numpy(dtype=np.float32)
    val = val_df.to_numpy(dtype=np.float32)
    test = test_df.to_numpy(dtype=np.float32)

    return train, val, test
    
def get_model_linear(train, val, test):

    train, val, test = map(lambda x: x / 255, (train, val, test)) 
    class LinearModel(nn.Module):
        def __init__(self):
            super(LinearModel, self).__init__()
            self.linear = nn.Linear(12, 3)  # Adjusted input features to 12

        def forward(self, x):
            return self.linear(x)

    model = LinearModel()

    # Define loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01)

    # Train the model
    model.train()
    for epoch in range(100):  # Number of epochs
        inputs = torch.tensor(train[:, :-3], dtype=torch.float32)
        targets = torch.tensor(train[:, -3:], dtype=torch.float32)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch + 1}/100], Loss: {loss.item():.4f}')

    # Test the model
    model.eval()
    with torch.no_grad():
        test_inputs = torch.tensor(test[:, :-3], dtype=torch.float32)
        test_targets = torch.tensor(test[:, -3:], dtype=torch.float32)
        test_outputs = model(test_inputs)
        test_loss = criterion(test_outputs, test_targets)
        mae = torch.mean(torch.abs(test_outputs - test_targets))
        print(f'Test Loss (MSE): {test_loss.item():.4f}')
        print(f'Test MAE: {mae.item():.4f} {mae.item() * 255:.4f}')

    return model

def get_model_randomforest(train, val, test):
    train, val, test = map(lambda x: x / 255, (train, val, test)) 

    model = RandomForestRegressor(n_estimators=100, random_state=42)

    # Train the model
    model.fit(train[:, :-3], train[:, -3:])

    # Test the model
    test_predictions = model.predict(test[:, :-3])
    test_mse = mean_squared_error(test[:, -3:], test_predictions)
    test_mae = mean_absolute_error(test[:, -3:], test_predictions)
    print(f'Test Loss (MSE): {test_mse:.4f}')
    print(f'Test MAE: {test_mae:.4f} {test_mae * 255:.4f}')

    return model

def get_model_svm(train, val, test):
    train, val, test = map(lambda x: x / 255, (train, val, test)) 

    model = MultiOutputRegressor(SVR(kernel='linear'))

    # Train the model
    model.fit(train[:, :-3], train[:, -3:])

    # Test the model
    test_predictions = model.predict(test[:, :-3])
    test_mse = mean_squared_error(test[:, -3:], test_predictions)
    test_mae = mean_absolute_error(test[:, -3:], test_predictions)
    print(f'Test Loss (MSE): {test_mse:.4f}')
    print(f'Test MAE: {test_mae:.4f} {test_mae * 255:.4f}')

    return model

def get_model(model_name: str, train, val, test):
    model = None
    if model_name == 'linear':
        model = get_model_linear(train, val, test)
    elif model_name == 'randomforest':
        model = get_model_randomforest(train, val, test)
    elif model_name == 'svm':
        model = get_model_svm(train, val, test)
    return model

def get_colors(filename:str):
    ret = None
    return ret # np.array #(4, 3)

def infer(model: torch.nn.Module, filename: str):
    ret = None
    return ret # np.array # (3)


