import numpy as np

def main(X_minority, y_minority, X_majority, y_majority, num_to_generate, info, random_state):
    rng = np.random.default_rng(random_state)
    original_indices = np.arange(len(y_minority))
    synthetic_indices = rng.choice(
        original_indices,
        size=num_to_generate,
        replace=True
    )
    
    X_synthetic = X_minority[synthetic_indices]
    y_synthetic = y_minority[synthetic_indices]
    
    return X_synthetic, y_synthetic