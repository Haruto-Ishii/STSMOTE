import numpy as np
from sklearn.preprocessing import QuantileTransformer
from sklearn.neighbors import NearestNeighbors
from imblearn.over_sampling import ADASYN
import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)

def main(X_minority, y_minority, X_majority, y_majority, num_to_generate, info, random_state):
    num_numerical_features = info['num_numerical_features']
    num_indices = list(range(num_numerical_features))
    cat_indices = list(range(num_numerical_features, X_minority.shape[1]))
    
    X_train_raw = np.vstack([X_minority, X_majority])
    y_train_raw = np.hstack([y_minority, y_majority])
    
    X_train_num_raw = X_train_raw[:, num_indices]
    X_minority_num_raw = X_minority[:, num_indices]
    X_minority_cat_raw = X_minority[:, cat_indices]
    
    # Preprocessing is essential for distance-based methods like SMOTE.
    # We use QuantileTransformer here to align the experimental setup with 
    # our proposed method (STSMOTE) and ensure a fair comparison, rather than 
    # using other scalers like StandardScaler.
    qt = QuantileTransformer(output_distribution='normal', random_state=random_state)
    qt.fit(X_train_num_raw)
    
    X_train_num_transformed = qt.transform(X_train_num_raw)
    X_minority_num_transformed = qt.transform(X_minority_num_raw)
    
    minority_class_int = y_minority[0]
    target_sample_count = len(y_minority) + num_to_generate
    k_neighbors = max(1, min(len(y_minority)-1, 5))
    
    ad = ADASYN(
        sampling_strategy={minority_class_int: target_sample_count},
        random_state=random_state,
        n_neighbors=k_neighbors
    )
    
    try:
        X_res_num, y_res = ad.fit_resample(X_train_num_transformed, y_train_raw)
    except:
        return np.empty((0, X_minority.shape[1])), np.empty(0)
    
    original_minority_count = len(y_minority)
    res_minority_indices = np.where(y_res == minority_class_int)[0]
    
    synthetic_indices_in_res = res_minority_indices[original_minority_count:]
    
    X_synthetic_num_transformed = X_res_num[synthetic_indices_in_res]
    y_synthetic = y_res[synthetic_indices_in_res]
    
    if len(y_synthetic) > num_to_generate:
        X_synthetic_num_transformed = X_synthetic_num_transformed[:num_to_generate]
        y_synthetic = y_synthetic[:num_to_generate]
        
    nn_model = NearestNeighbors(n_neighbors=1, algorithm='auto')
    nn_model.fit(X_minority_num_transformed)
    
    _, nearest_indices = nn_model.kneighbors(X_synthetic_num_transformed)
    X_synthetic_cat_raw = X_minority_cat_raw[nearest_indices.ravel()]
    
    X_synthetic_num_raw = qt.inverse_transform(X_synthetic_num_transformed)
    
    X_synthetic_raw = np.hstack([X_synthetic_num_raw, X_synthetic_cat_raw])
    return X_synthetic_raw, y_synthetic