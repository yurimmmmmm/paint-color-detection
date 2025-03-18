import numpy as np
import pandas as pd
from sklearn.metrics import r2_score, mean_squared_error

def delta_e(rgb_true, rgb_pred):
    """
    Compute the Delta E (color difference) between two RGB colors.
    Converts RGB to Lab space and computes the color difference.
    """
    from skimage.color import rgb2lab

    # Normalize RGB values to [0,1] before conversion
    rgb_true = np.array(rgb_true) / 255.0
    rgb_pred = np.array(rgb_pred) / 255.0

    lab_true = rgb2lab(rgb_true.reshape(1, 1, 3)).reshape(3)
    lab_pred = rgb2lab(rgb_pred.reshape(1, 1, 3)).reshape(3)

    return np.linalg.norm(lab_true - lab_pred)

def evaluate_metrics(true_rgb, predicted_rgb):
    """
    Compute evaluation metrics given true and predicted RGB values.
    Returns a dictionary with R² Score, RMSE, MAPE, and ΔE.
    """
    # R² Score
    r2 = r2_score(true_rgb, predicted_rgb)

    # RMSE
    rmse = np.sqrt(mean_squared_error(true_rgb, predicted_rgb))

    # MAPE (avoid division by zero)
    mape = np.mean(np.abs((true_rgb - predicted_rgb) / np.where(true_rgb == 0, 1, true_rgb))) * 100

    # ΔE for each color and take mean & median
    delta_e_values = [delta_e(true_rgb[i], predicted_rgb[i]) for i in range(len(true_rgb))]
    mean_delta_e = np.mean(delta_e_values)
    median_delta_e = np.median(delta_e_values)

    return {
        "R2_Score": r2,
        "RMSE": rmse,
        "MAPE": mape,
        "Mean_Delta_E": mean_delta_e,
        "Median_Delta_E": median_delta_e
    }

def evaluate_predictions(csv_file):
    """
    Evaluate predictions from a CSV file containing true and predicted RGB values.
    """
    df = pd.read_csv(csv_file)
    
    true_rgb = df[['true_R', 'true_G', 'true_B']].values
    predicted_rgb = df[['pred_R', 'pred_G', 'pred_B']].values 

    metrics = evaluate_metrics(true_rgb, predicted_rgb)

    for metric, value in metrics.items():
        print(f"{metric}: {value:.4f}")

    results_df = pd.DataFrame([metrics])
    results_df.to_csv("evaluation_results.csv", index=False)
    print("Evaluation results saved to evaluation_results.csv")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Evaluate color prediction model")
    parser.add_argument("csv_file", type=str, help="Path to the CSV file with true and predicted RGB values")
    args = parser.parse_args()

    evaluate_predictions(args.csv_file)
