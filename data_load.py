import polars as pl
import numpy as np
import json
from pathlib import Path
from sklearn.preprocessing import LabelEncoder

def load_data_from_config(config: dict) -> tuple:
    
    TEST_SAMPLE_RATIO = 0.25
    
    try:
        dataset_name = config["dataset"]
        d_config = config["dataset_configs"][dataset_name]
        data_dir = Path(d_config["data_dir"])
        task = d_config["task"]
        sampling_map = d_config["sampling_map"]
        minority_class_name = d_config.get("minority_class_name")
        random_state = config["random_state"]
        
        print(f"--- Loading data for: {dataset_name} ---")
        print(f"Task: {task}, Minority Class: {minority_class_name or 'N/A'}")

    except KeyError as e:
        print(f"Error: Missing required key in config file: {e}")
        raise

    info_path = data_dir / "info.json"
    if not info_path.exists():
        raise FileNotFoundError(f"{info_path} not found.")
        
    with open(info_path, 'r') as f:
        info = json.load(f)
    
    try:
        target_col_idx = str(info["target_col_idx"][0])
        label_mapping = info["column_info"][target_col_idx]["categories"]
        label_name_to_int = {name: int(idx) for idx, name in label_mapping.items()}
        
    except KeyError as e:
        print(f"Error: Invalid format in info.json. 'categories' key missing. {e}")
        raise

    try:
        X_num_train_full = pl.read_parquet(data_dir / "X_num_train.parquet").to_numpy()
        X_cat_train_full = pl.read_parquet(data_dir / "X_cat_train.parquet").to_numpy()
        y_train_full = pl.read_parquet(data_dir / "y_train.parquet").to_numpy().ravel()
        
        X_num_test_full = pl.read_parquet(data_dir / "X_num_test.parquet").to_numpy()
        X_cat_test_full = pl.read_parquet(data_dir / "X_cat_test.parquet").to_numpy()
        y_test_full = pl.read_parquet(data_dir / "y_test.parquet").to_numpy().ravel()
        
    except Exception as e:
        print(f"Error loading Parquet files with polars: {e}")
        raise

    X_train_full = np.hstack([X_num_train_full, X_cat_train_full])
    X_test_full = np.hstack([X_num_test_full, X_cat_test_full])
    num_numerical_features = X_num_train_full.shape[1]

    rng = np.random.default_rng(random_state)
    
    train_indices_to_keep = []
    for class_name, target_count in sampling_map.items():
        if target_count == 0:
            continue
            
        class_int = label_name_to_int.get(class_name)
        if class_int is None:
            raise ValueError(f"Class '{class_name}' not found in label mapping.")
            
        all_class_indices = np.where(y_train_full == class_int)[0]
        if len(all_class_indices) == 0:
            raise ValueError(f"No samples found for class '{class_name}' in training data.")
            
        if len(all_class_indices) > target_count:
            sampled_indices = rng.choice(all_class_indices, size=target_count, replace=False)
        else:
            sampled_indices = all_class_indices
            
        train_indices_to_keep.append(sampled_indices)

    if not train_indices_to_keep:
        raise ValueError("Training data resulted in 0 samples after sampling. Please check sampling_map.")
        
    final_train_indices = np.hstack(train_indices_to_keep)
    rng.shuffle(final_train_indices)
    
    X_train = X_train_full[final_train_indices]
    y_train_raw = y_train_full[final_train_indices]

    test_indices_to_keep = []
    for class_name, train_count in sampling_map.items():
        if train_count == 0:
            continue
            
        target_count_test = max(1, int(train_count * TEST_SAMPLE_RATIO))
        class_int = label_name_to_int.get(class_name)
        if class_int is None:
            continue
            
        all_class_indices = np.where(y_test_full == class_int)[0]

        if len(all_class_indices) == 0:
            continue
            
        if len(all_class_indices) > target_count_test:
            sampled_indices = rng.choice(all_class_indices, size=target_count_test, replace=False)
        else:
            sampled_indices = all_class_indices
            
        test_indices_to_keep.append(sampled_indices)

    if not test_indices_to_keep:
        raise ValueError("Test data resulted in 0 samples after sampling.")

    final_test_indices = np.hstack(test_indices_to_keep)
    rng.shuffle(final_test_indices)
    
    X_test = X_test_full[final_test_indices]
    y_test_raw = y_test_full[final_test_indices]


    unique_labels_raw = np.unique(np.hstack([y_train_raw, y_test_raw]))
    
    le = LabelEncoder()
    le.fit(unique_labels_raw)
    
    y_train = le.transform(y_train_raw)
    y_test = le.transform(y_test_raw)
    
    new_categories = {
        int(le.transform([raw_label])[0]): label_mapping[str(raw_label)]
        for raw_label in le.classes_
    }
    info["column_info"][target_col_idx]["categories"] = new_categories
    print(f"Multiclass labels re-mapped to: {new_categories}")
    minority_class_int = [k for k, v in new_categories.items() if v == minority_class_name][0]

    print("--- Data loading complete. ---")
    
    return X_train, y_train, X_test, y_test, info, minority_class_int, num_numerical_features