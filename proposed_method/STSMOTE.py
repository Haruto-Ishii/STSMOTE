import numpy as np
import pandas as pd
import itertools
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import QuantileTransformer

# --- Helper Function ---
def generate_noise_orthogonal_vectorized(vec_axis, dim, magnitudes):
    """
    Generates noise vectors orthogonal to the axis vector (vec_axis).
    Used for creating the tube width.
    """
    n_samples = len(magnitudes)
    rand_vecs = np.random.standard_normal((n_samples, dim))
    
    norm_axis = np.linalg.norm(vec_axis)
    if norm_axis == 0:
        return np.zeros((n_samples, dim))
    
    unit_axis = vec_axis / norm_axis
    
    projections = np.outer(np.dot(rand_vecs, unit_axis), unit_axis)
    vec_perps = rand_vecs - projections
    
    perp_norms = np.linalg.norm(vec_perps, axis=1, keepdims=True) + 1e-9
    perp_units = vec_perps / perp_norms
    
    return perp_units * magnitudes[:, np.newaxis]

def main(X_minority, y_minority, X_majority, y_majority, num_to_generate, info, random_state):
    rng = np.random.default_rng(random_state)
    
    num_numerical_features = info['num_numerical_features']
    num_indices = list(range(num_numerical_features))
    cat_indices = list(range(num_numerical_features, X_minority.shape[1]))
    
    n_dim_num = len(num_indices)
    
    if num_to_generate <= 0:
        return np.empty((0, X_minority.shape[1])), np.empty(0)
    
    # --- 1. Preprocessing (Section III-A) ---
    # As described in the paper, we use QuantileTransformer for distance-based stability.
    scaler = QuantileTransformer(output_distribution='normal', random_state=random_state)
    X_num_all = np.vstack([X_minority[:, num_indices], X_majority[:, num_indices]])
    scaler.fit(X_num_all)
    
    X_min_num = scaler.transform(X_minority[:, num_indices])
    X_maj_num = scaler.transform(X_majority[:, num_indices])
    
    X_min_cat = X_minority[:, cat_indices]
    
    n_seed = len(X_min_num)
    n_bins = 1000
    
    # --- 2. Safety Radius Calculation (Section III-A) ---
    # r = d / 2, where d is the distance to the nearest majority point.
    nn = NearestNeighbors(n_neighbors=1, metric='euclidean', n_jobs=-1)
    nn.fit(X_maj_num)
    dists, _ = nn.kneighbors(X_min_num)
    safe_radii = dists.flatten() / 2.0
    
    # Weight distribution between Sphere and Tube (default: 1:1)
    weight_sphere = 1
    weight_tube = 1
    total_weight = weight_sphere + weight_tube
    
    if total_weight == 0:
        num_sphere = num_to_generate
        num_tube = 0
    else:
        num_sphere = int(num_to_generate * (weight_sphere / total_weight))
        num_tube = num_to_generate - num_sphere
    
    generated_num_data = []
    generated_cat_data = []
    
    # --- 3. Sphere Generation Strategy (Section III-A) ---
    def generate_sphere(center_idx, n_gen):
        if n_gen <= 0: return
        
        center_num = X_min_num[center_idx]
        radius = safe_radii[center_idx]
        
        directions = rng.standard_normal((n_gen, n_dim_num))
        norms = np.linalg.norm(directions, axis=1, keepdims=True) + 1e-9
        directions /= norms
        
        rs = rng.random(n_gen) * radius
        new_pts_num = center_num + directions * rs[:, np.newaxis]
        
        center_cat = X_min_cat[center_idx]
        new_pts_cat = np.tile(center_cat, (n_gen, 1))
        
        generated_num_data.append(new_pts_num)
        generated_cat_data.append(new_pts_cat)

    if num_sphere > 0:
        samples_per_point = num_sphere // n_seed
        remainder = num_sphere % n_seed
        
        for i in range(n_seed):
            n_gen = samples_per_point + (1 if i < remainder else 0)
            generate_sphere(i, n_gen)
    
    # --- 4. Tube Generation Strategy (Section III-B) ---
    pairs = list(itertools.combinations(range(n_seed), 2))
    n_pairs = len(pairs)
    
    if num_tube > 0 and n_pairs > 0:
        samples_per_pair = num_tube // n_pairs
        remainder_pair = num_tube % n_pairs
        
        for p_idx, (idx_a, idx_b) in enumerate(pairs):
            n_gen_target = samples_per_pair + (1 if p_idx < remainder_pair else 0)
            if n_gen_target == 0: continue
            
            A_num = X_min_num[idx_a]
            B_num = X_min_num[idx_b]
            vec_AB = B_num - A_num
            len_AB_sq = np.dot(vec_AB, vec_AB)
            len_AB = np.sqrt(len_AB_sq)
            
            if len_AB < 1e-6:
                # Fallback to Sphere if points are too close
                generate_sphere(idx_a, n_gen_target // 2)
                generate_sphere(idx_b, n_gen_target - (n_gen_target // 2))
                continue
                
            unit_AB = vec_AB / len_AB
            
            # --- Identify Local Obstacles (Section III-B, Step 2) ---
            vec_AP = X_maj_num - A_num
            r_proj = np.dot(vec_AP, unit_AB)
            
            vec_perp = vec_AP - np.outer(r_proj, unit_AB)
            h1 = np.linalg.norm(vec_perp, axis=1) + 1e-9
            
            dist_sq_MA = np.sum(vec_AP**2, axis=1)
            dist_sq_MB = np.sum((X_maj_num - B_num)**2, axis=1)
            
            # Map obstacles to bins
            t_norm = r_proj / len_AB
            bin_indices = np.floor(t_norm * n_bins).astype(int)
            valid_mask = (bin_indices >= 0) & (bin_indices < n_bins)
            
            if np.any(valid_mask):
                df = pd.DataFrame({
                    'bin': bin_indices[valid_mask],
                    'h1': h1[valid_mask],
                    'r_proj': r_proj[valid_mask],
                    'dist_sq': dist_sq_MA[valid_mask],
                    'idx': np.where(valid_mask)[0]
                })
                # Keep the most restrictive obstacle per bin
                min_h1_df = df.loc[df.groupby('bin')['h1'].idxmin()]
                bin_constraint_map = min_h1_df.set_index('bin').to_dict('index')
            else:
                bin_constraint_map = {}
            
            # --- Step 1: Cone Constraint ---
            # Linearly interpolate radii from endpoints A and B
            check_bins = np.arange(n_bins)
            check_r = (check_bins + 0.5) / n_bins * len_AB
            
            radius_A = safe_radii[idx_a]
            radius_B = safe_radii[idx_b]
            
            t_steps = (check_bins + 0.5) / n_bins
            interpolated_radii = (1 - t_steps) * radius_A + t_steps * radius_B
            
            final_h_limits = interpolated_radii
            
            # --- Step 2 & 3: Local Obstacle & Monotonicity Constraints ---
            mid_idx = n_bins // 2
            
            # Process Left Half (A -> Midpoint)
            left_bins = check_bins[:mid_idx]
            left_r = check_r[:mid_idx]
            
            limits_left = final_h_limits[:mid_idx]
            
            for b_idx in left_bins:
                if b_idx in bin_constraint_map:
                    info = bin_constraint_map[b_idx]
                    idx_maj = int(info['idx'])
                    
                    # Apply Local Obstacle Equation (Eq. 2)
                    h1_local = info['h1']
                    r1_local = info['r_proj']
                    dsq_local = info['dist_sq']
                    
                    val = (dsq_local - 2 * r1_local * left_r[b_idx]) / (2 * h1_local)
                    
                    limits_left[b_idx] = min(limits_left[b_idx], val)
            
            # Apply Monotonicity (Step 3)
            limits_left = np.maximum(limits_left, 0)
            limits_left = np.minimum.accumulate(limits_left)
            
            final_h_limits[:mid_idx] = limits_left
            
            # Process Right Half (B -> Midpoint)
            right_bins = check_bins[mid_idx:]
            right_r_from_A = check_r[mid_idx:]
            
            right_r_from_B = len_AB - right_r_from_A
            
            limits_right = final_h_limits[mid_idx:]
            
            for i, b_idx in enumerate(right_bins):
                if b_idx in bin_constraint_map:
                    info = bin_constraint_map[b_idx]
                    idx_maj = int(info['idx'])
                    
                    # Same logic, but from perspective of B for symmetry
                    h1_local = info['h1']
                    r1_local = info['r_proj']
                    
                    r1_local_from_B = len_AB - r1_local
                    dsq_local_from_B = dist_sq_MB[idx_maj]
                    
                    r_current_from_B = right_r_from_B[i]
                    val = (dsq_local_from_B - 2 * r1_local_from_B * r_current_from_B) / (2 * h1_local)
                    
                    limits_right[i] = min(limits_right[i], val)
            
            # Apply Monotonicity (Step 3) - from B towards Midpoint
            limits_right = np.maximum(limits_right, 0)
            limits_right = np.minimum.accumulate(limits_right[::-1])[::-1]
            
            final_h_limits[mid_idx:] = limits_right
            
            # --- Data Generation ---
            valid_bin_indices = check_bins[final_h_limits > 1e-9]
            
            if len(valid_bin_indices) == 0:
                generate_sphere(idx_a, n_gen_target // 2)
                generate_sphere(idx_b, n_gen_target - (n_gen_target // 2))
                continue

            selected_bins = rng.choice(valid_bin_indices, size=n_gen_target)
            bin_offsets = rng.random(n_gen_target)
            
            r_selected = (selected_bins + bin_offsets) / n_bins * len_AB
            
            final_h_max = final_h_limits[selected_bins]
            
            # Generate orthogonal noise bounded by the calculated safe height
            noise_mags = rng.random(n_gen_target) * final_h_max * 1.0
            vec_noises = generate_noise_orthogonal_vectorized(vec_AB, n_dim_num, noise_mags)
            
            P_new_num = A_num + np.outer(r_selected, unit_AB) + vec_noises
            
            # Inherit categorical features from the nearest endpoint
            P_new_cat = np.empty((n_gen_target, len(cat_indices)), dtype=X_min_cat.dtype)
            
            mask_near_A = (r_selected <= len_AB / 2)
            if np.any(mask_near_A):
                P_new_cat[mask_near_A] = X_min_cat[idx_a]
            
            mask_near_B = ~mask_near_A
            if np.any(mask_near_B):
                P_new_cat[mask_near_B] = X_min_cat[idx_b]
            
            generated_num_data.append(P_new_num)
            generated_cat_data.append(P_new_cat)

    if len(generated_num_data) > 0:
        X_gen_num_scaled = np.vstack(generated_num_data)
        X_gen_num = scaler.inverse_transform(X_gen_num_scaled)
        
        X_gen_cat = np.vstack(generated_cat_data)
        
        X_synthetic = np.hstack([X_gen_num, X_gen_cat])
        
        y_synthetic = np.full(len(X_synthetic), y_minority[0])
        
        return X_synthetic, y_synthetic
    else:
        return np.empty((0, X_minority.shape[1])), np.empty(0)