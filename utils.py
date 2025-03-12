import torch
import numpy as np
import os
import pandas as pd

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
    
def get_model(model_name: str):
    model = None
    if model_name == 'linear':
         pass
         #return get_model_linear()
    return model

def get_colors(filename:str):
    ret = None
    return ret # np.array #(4, 3)

def infer(model: torch.nn.Module, filename: str):
    ret = None
    return ret # np.array # (3)


