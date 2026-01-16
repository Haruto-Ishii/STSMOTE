import os
os.environ["OMP_NUM_THREADS"] = "32"
os.environ["MKL_NUM_THREADS"] = "32"
os.environ["OPENBLAS_NUM_THREADS"] = "32"

import yaml
import time
import numpy as np
import gc
import torch
from data_load import load_data_from_config
from learning_model.learn_and_test import run_training_evaluation
from synthesis_method.synthesis_main import run_augmentation
from utils.cache_manager import ExperimentCacheManager
from utils.analysis import run_comparative_analysis

#!
PROPOSED_METHOD = "STSMOTE"
METHODS_TO_COMPARE = ["Baseline", "SMOTE", "RO", "ADASYN", "CTGAN", "TVAE", "BorderlineSMOTE", "TabSyn"]
METHODS_TO_RUN = METHODS_TO_COMPARE + [PROPOSED_METHOD]
SKIP_EXISTING = True
PROPOSED_REMOVE_OLD_CACHE = False
#!

old_proposed_filename = os.path.join("experiment_results_cache", f"results_{PROPOSED_METHOD}.csv")
if os.path.exists(old_proposed_filename) and PROPOSED_REMOVE_OLD_CACHE:
    os.remove(old_proposed_filename)

with open("config.yaml", 'r') as f:
    config = yaml.safe_load(f)

dataset_name = config['dataset']
dataset_config = config['dataset_configs'][dataset_name]
minority_class_name = dataset_config.get('minority_class_name')
seed_sizes = config["seed_sizes"]
random_state = config["random_state"]
seed_undersample_states = config['seed_undersample_states']
oversample_num = dataset_config["oversample_num"]
task = dataset_config.get('task')

print(f"Experiment using {dataset_name} starts. Selected minority class is {minority_class_name}")

X_train_full, y_train_full, X_test, y_test, info, minority_class_int, num_numerical_features = load_data_from_config(config)
    
minority_indices = np.where(y_train_full == minority_class_int)[0]
majority_indices = np.where(y_train_full != minority_class_int)[0]

X_minority_full = X_train_full[minority_indices]
y_minority_full = y_train_full[minority_indices]
X_majority = X_train_full[majority_indices]
y_majority = y_train_full[majority_indices]

minority_size = len(minority_indices)

print(f"\nData separated: Minority (Full): {len(y_minority_full)} samples, Majority (Sampled): {len(y_majority)} samples")

cache_manager = ExperimentCacheManager()

for seed_undersample_state in seed_undersample_states:

    rng = np.random.default_rng(seed_undersample_state)
    print(f"\n--- 6. Starting Undersampling Loop for seed_undersample_state: {seed_undersample_state} ---")

    for seed_size in seed_sizes:
        num_to_generate = max(0, oversample_num - seed_size)
        
        if len(X_minority_full) < seed_size:
            print(f"Skipping seed_size {seed_size}: Not enough minority samples (Available: {len(X_minority_full)})")
            continue

        print(f"\n--- Running loop for seed_size = {seed_size} ---")
        
        all_indices = np.arange(len(X_minority_full))
        sampled_indices = rng.choice(len(X_minority_full), size=seed_size, replace=False)
        X_seed = X_minority_full[sampled_indices]
        y_seed = y_minority_full[sampled_indices]
        
        remaining_indices = np.setdiff1d(all_indices, sampled_indices)
        X_remaining = X_minority_full[remaining_indices]
        y_remaining = y_minority_full[remaining_indices]
        
        X_test_current = np.vstack([X_test, X_remaining])
        y_test_current = np.hstack([y_test, y_remaining])

        exp_name = "Baseline_Undersampled"
        run_name_4 = f"{exp_name}_seed_{seed_size}"
        
        if not SKIP_EXISTING or not cache_manager.check_exists("Baseline", seed_size, seed_undersample_state, minority_class_name, random_state):

            X_train_exp = np.vstack([X_seed, X_majority])
            y_train_exp = np.hstack([y_seed, y_majority])
            
            shuffle_idx = rng.permutation(len(y_train_exp))
            X_train_exp = X_train_exp[shuffle_idx]
            y_train_exp = y_train_exp[shuffle_idx]

            baseline_results = run_training_evaluation(
                X_train_exp, y_train_exp, 
                X_test_current, y_test_current, 
                num_numerical_features,
                minority_class_int,
                random_state=seed_undersample_state
            )
            cache_manager.save_result("Baseline", seed_size, seed_undersample_state, random_state, minority_class_name, baseline_results)
        else:
            df_baseline = cache_manager.load_results_for_comparison(["Baseline"], seed_size, seed_undersample_state, minority_class_name)
            df_baseline.drop(columns=["method", "seed_size", "seed_undersample_state", "random_state", "minority_class_name"], inplace=True, errors="ignore")
            baseline_results = df_baseline.iloc[0].to_dict()

        for method_name in METHODS_TO_RUN:
            is_proposed = (method_name == PROPOSED_METHOD) and PROPOSED_REMOVE_OLD_CACHE
            
            if not is_proposed and SKIP_EXISTING:
                if cache_manager.check_exists(method_name, seed_size, seed_undersample_state, minority_class_name, random_state):
                    print(f"Skipping {method_name} (Cached)")
                    continue
            
            print(f"Data Augmentation with {method_name} Start", flush=True)                
            aug_start_time = time.time()
            X_synthetic, y_synthetic = run_augmentation(method_name, X_seed, y_seed, X_majority, y_majority, num_to_generate, info, random_state)
            aug_end_time = time.time()

            if len(y_synthetic) == 0:
                print(f"> Generation Failed. Using baseline results.")
                current_results = baseline_results
            else:
                print(f"Data Augmentating {len(y_synthetic)} samples using {method_name}  took {aug_end_time - aug_start_time:.4f} seconds", flush=True)
                X_train_aug = np.vstack([X_seed, X_synthetic, X_majority])
                y_train_aug = np.hstack([y_seed, y_synthetic, y_majority])
                
                shuffle_idx = rng.permutation(len(y_train_aug))
                X_train_aug = X_train_aug[shuffle_idx]
                y_train_aug = y_train_aug[shuffle_idx]
        
                current_results = run_training_evaluation(
                    X_train_aug, y_train_aug,
                    X_test_current, y_test_current, 
                    num_numerical_features,
                    minority_class_int,
                    random_state=seed_undersample_state
                )
            cache_manager.save_result(method_name, seed_size, seed_undersample_state, random_state, minority_class_name, current_results)
                
            try:
                del X_synthetic, y_synthetic
                if 'X_train_aug' in locals(): del X_train_aug
                if 'y_train_aug' in locals(): del y_train_aug
                if 'current_results' in locals(): del current_results
            except NameError:
                pass
            
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

run_comparative_analysis(cache_manager, METHODS_TO_RUN, seed_sizes, seed_undersample_states, minority_class_name)

print("\n--- Experiment finished. ---")
