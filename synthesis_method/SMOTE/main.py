from imblearn.over_sampling import SMOTENC
import numpy as np
from sklearn.preprocessing import QuantileTransformer, OrdinalEncoder
import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)

def main(X_minority, y_minority, X_majority, y_majority, num_to_generate, info, random_state):
    X_train_raw = np.vstack([X_minority, X_majority])
    
    num_numerical_features = info['num_numerical_features']
    num_indices = list(range(num_numerical_features))
    cat_indices = list(range(num_numerical_features, X_train_raw.shape[1]))
    
    # Preprocessing is essential for distance-based methods like SMOTE.
    # We use QuantileTransformer here to align the experimental setup with 
    # our proposed method (STSMOTE) and ensure a fair comparison, rather than 
    # using other scalers like StandardScaler.
    qt = QuantileTransformer(output_distribution='normal', random_state=random_state)
    qt.fit(X_train_raw[:, num_indices])
    
    oe = OrdinalEncoder(handle_unknown="error", dtype=int)
    oe.fit(X_minority[:, cat_indices])
    
    X_minority_num_transformed = qt.transform(X_minority[:, num_indices])
    X_minority_cat_transformed = oe.transform(X_minority[:, cat_indices])
    X_minority_transformed = np.hstack([X_minority_num_transformed, X_minority_cat_transformed])
    
    smotenc_cat_indices = list(range(len(num_indices), X_minority_transformed.shape[1]))
    minority_class_int = y_minority[0]
    target_sample_count = len(y_minority) + num_to_generate
    
    k_neighbors = max(1, min(len(y_minority)-1, 5))
    random_X = np.zeros((target_sample_count, X_minority.shape[1]))
    random_y = np.full(target_sample_count, -1)
    
    X_src = np.vstack([X_minority_transformed, random_X])
    y_src = np.hstack([y_minority, random_y])
    
    sm = SMOTENC(
        categorical_features=smotenc_cat_indices,
        random_state=random_state,
        k_neighbors=k_neighbors
    )
    
    X_res, y_res = sm.fit_resample(X_src, y_src)
    
    gen_indices = np.where(y_res == minority_class_int)[0][-num_to_generate:]
    X_syn_transformed = X_res[gen_indices]
    y_syn = y_res[gen_indices]
    
    X_syn_num_raw = qt.inverse_transform(X_syn_transformed[:, :num_numerical_features])
    X_syn_cat_raw = oe.inverse_transform(X_syn_transformed[:, num_numerical_features:])
    
    X_syn_raw = np.hstack([X_syn_num_raw, X_syn_cat_raw])
    return X_syn_raw, y_syn