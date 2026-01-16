import os
import pandas as pd

class ExperimentCacheManager:
    def __init__(self, cache_dir="experiment_results_cache"):
        self.cache_dir = cache_dir
        os.makedirs(self.cache_dir, exist_ok=True)
    
    def _get_file_path(self, method_name):
        return os.path.join(self.cache_dir, f"results_{method_name}.csv")

    def check_exists(self, method_name, seed_size, seed_undersample, minority_class_name, random_state):
        file_path = self._get_file_path(method_name)
        if not os.path.exists(file_path):
            return False
        
        try:
            df = pd.read_csv(file_path)
            match = df[
                (df["seed_size"] == seed_size) &
                (df["seed_undersample_state"] == seed_undersample) &
                (df["random_state"] == random_state) &
                (df["minority_class_name"] == str(minority_class_name))
            ]
            return not match.empty
        except Exception as e:
            print(f"Warning: Failed to read cache for {method_name}: {e}")
            return False

    def save_result(self, method_name, seed_size, seed_undersample, random_state, minority_class_name, metrics_dict):
        file_path = self._get_file_path(method_name)
        
        flattened_metrics = {}
        for model_name, scores in metrics_dict.items():
            if isinstance(scores, dict):
                for metric_name, value in scores.items():
                    col_name = f"{model_name}_{metric_name}"
                    flattened_metrics[col_name] = value
            else:
                flattened_metrics[model_name] = scores

        data = {
            "method": method_name,
            "seed_size": seed_size,
            "seed_undersample_state": seed_undersample,
            "random_state": random_state,
            "minority_class_name": minority_class_name,
            **flattened_metrics
        }
        new_row = pd.DataFrame([data])
        
        if os.path.exists(file_path):
            try:
                existing_df = pd.read_csv(file_path)
                updated_df = pd.concat([existing_df, new_row], ignore_index=True)
                updated_df.to_csv(file_path, index=False)
            except:
                new_row.to_csv(file_path, mode='a', header=False, index=False)
        else:
            new_row.to_csv(file_path, mode='w', header=True, index=False)

    def load_results_for_comparison(self, target_methods, current_seed_sizes, current_seed_undersamples, current_minority_class_name):
        all_results = []
        
        if not isinstance(current_seed_sizes, list): current_seed_sizes = [current_seed_sizes]
        if not isinstance(current_seed_undersamples, list): current_seed_undersamples = [current_seed_undersamples]
        if not isinstance(current_minority_class_name, list): current_minority_class_name = [current_minority_class_name]
        
        for method in target_methods:
            file_path = self._get_file_path(method)
            if os.path.exists(file_path):
                df = pd.read_csv(file_path)
                df_filtered = df[
                    df["seed_size"].isin(current_seed_sizes) &
                    df["seed_undersample_state"].isin(current_seed_undersamples) &
                    df["minority_class_name"].isin(current_minority_class_name)
                ]
                all_results.append(df_filtered)
            else:
                if method != "Proposed":
                     print(f"Warning: Cache file for comparison target '{method}' not found.")

        if not all_results:
            return pd.DataFrame()
        
        return pd.concat(all_results, ignore_index=True)