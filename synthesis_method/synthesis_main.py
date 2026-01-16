import numpy as np
from synthesis_method.SMOTE.main import main as smote_main
from synthesis_method.BorderlineSMOTE.main import main as bsmote_main
from synthesis_method.RO.main import main as ro_main
from synthesis_method.ADASYN.main import main as adasyn_main
from synthesis_method.CTGAN.main import main as ctgan_main
from synthesis_method.TVAE.main import main as tvae_main
from synthesis_method.TabSyn.tabsyn.main import main as tabsyn_main
from proposed_method.STSMOTE import main as stsmote_main

def run_augmentation(method_name, X_minority, y_minority, X_majority, y_majority, num_to_generate, info, random_state):
    if method_name == 'SMOTE':
        X_synthetic, y_synthetic = smote_main(X_minority, y_minority, X_majority, y_majority, num_to_generate, info, random_state)
    elif method_name == 'BorderlineSMOTE':
        X_synthetic, y_synthetic = bsmote_main(X_minority, y_minority, X_majority, y_majority, num_to_generate, info, random_state)
    elif method_name == 'RO':
        X_synthetic, y_synthetic = ro_main(X_minority, y_minority, X_majority, y_majority, num_to_generate, info, random_state)
    elif method_name == 'ADASYN':
        X_synthetic, y_synthetic = adasyn_main(X_minority, y_minority, X_majority, y_majority, num_to_generate, info, random_state)
    elif method_name == 'CTGAN':
        X_synthetic, y_synthetic = ctgan_main(X_minority, y_minority, X_majority, y_majority, num_to_generate, info, random_state)
    elif method_name == 'TVAE':
        X_synthetic, y_synthetic = tvae_main(X_minority, y_minority, X_majority, y_majority, num_to_generate, info, random_state)
    elif method_name == 'TabSyn':
        X_synthetic, y_synthetic = tabsyn_main(X_minority, y_minority, X_majority, y_majority, num_to_generate, info, random_state)
    elif method_name == 'STSMOTE':
        X_synthetic, y_synthetic = stsmote_main(X_minority, y_minority, X_majority, y_majority, num_to_generate, info, random_state)
    else:
        raise ValueError(f"Unknown method: {method_name}")
    
    
    if len(X_synthetic) == 0:
        print(f"Warning: {method_name} failed to generate any data (returned empty).")
            
        return X_synthetic, y_synthetic

    original_minority_class = y_minority[0]
    synth_minority_class = np.unique(y_synthetic)
    
    if len(synth_minority_class) > 1 or original_minority_class != synth_minority_class[0]:
        raise ValueError("Class of synthetic data doesn't match the original one.")

    return X_synthetic, y_synthetic