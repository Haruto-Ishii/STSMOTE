import numpy as np
import pandas as pd
import yaml
import warnings
from pathlib import Path
from ctgan import CTGAN

warnings.filterwarnings("ignore", category=UserWarning)

def main(X_minority, y_minority, X_majority, y_majority, num_to_generate, info, random_state):
    config_path = Path(__file__).parent / "config.yaml"
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
        
    batch_size = min(config.get("batch_size", 10), len(y_minority))
    if batch_size % 2 == 1:
        batch_size -= 1
    config["batch_size"] = batch_size

    num_numerical_features = info["num_numerical_features"]
    num_cols = [f'num_{i}' for i in range(num_numerical_features)]
    cat_cols = [f'cat_{i}' for i in range(num_numerical_features, X_minority.shape[1])]
    
    df_minority = pd.DataFrame(X_minority, columns=num_cols + cat_cols)
    for col in cat_cols:
        df_minority[col] = df_minority[col].astype(object)
    
    # Unlike distance-based methods (e.g., SMOTE), TVAE and CTGAN employ 
    # an internal preprocessing mechanism (Mode-Specific Normalization using GMMs)
    # to handle complex distributions. Therefore, explicit external preprocessing
    # like QuantileTransformer is not required, and raw data is passed directly.
    ctgan = CTGAN(**config)
    ctgan.fit(df_minority, discrete_columns=cat_cols)
    X_synthetic_df = ctgan.sample(num_to_generate)
    
    X_synthetic_df_ordered = X_synthetic_df[num_cols + cat_cols]
    
    X_synthetic_raw = X_synthetic_df_ordered.to_numpy()
    y_synthetic = np.full(num_to_generate, y_minority[0])
    
    return X_synthetic_raw, y_synthetic