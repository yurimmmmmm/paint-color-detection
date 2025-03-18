from utils import * 
import torch
import torch.nn as nn
import torch.optim as optim

def main():
    master_data_path = "./master_data"

    # Call get_data to load the datasets
    train, val, test, test_df = get_data(master_data_path)

    # Print basic information for verification
    print(f"Train Set Shape: {train.shape}")
    print(f"Validation Set Shape: {val.shape}")
    print(f"Test Set Shape: {test.shape}")

    # Check a few samples
    print("\nFirst 3 rows of Train Set:")
    print(train[:3])

    print("\nFirst 3 rows of Test Set:")
    print(test[:3])

    print("\nFirst 3 rows of Val Set:")
    print(val[:3])

    # Get the model and train/test it
    model = get_model('linear', train, val, test, test_df)
    model = get_model('randomforest', train, val, test, test_df)
    model = get_model('svm', train, val, test, test_df)

if __name__ == "__main__":
    main()
