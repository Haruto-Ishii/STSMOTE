# STSMOTE: Official Implementation

This repository contains the implementation of **STSMOTE**, a controllable data augmentation method for zero-day attack detection in NIDS.

## 📢 Note to Reviewers
**Please check the `results` folder.**
This folder contains additional experimental data and results that could not be included in the main paper due to page limits.

---

## 1. Installation
Install the required Python modules using `requirements.txt`.

```bash
pip install -r requirements.txt
```

## 2. Dataset Setup
Due to copyright and licensing restrictions, this repository only includes the preprocessing scripts and instructions for the **CIC-IDS2018** dataset.

1. Download the preprocessed data from the following link:
   - [**Download Link (Google Drive)**](https://drive.google.com/drive/folders/1uPSIM3fwae94EdP6yxzApU-kiKncJ4JB?usp=sharing)
2. Place the downloaded files directly into the `parquet/` directory.

### Dataset Attribution
The CIC-IDS2018 dataset used in this work is sourced from:
- **Source:** [CSE-CIC-IDS2018 on AWS Registry of Open Data](https://registry.opendata.aws/cse-cic-ids2018/)
- **Reference:**
  > Iman Sharafaldin, Arash Habibi Lashkari, and Ali A. Ghorbani, “Toward Generating a New Intrusion Detection Dataset and Intrusion Traffic Characterization”, 4th International Conference on Information Systems Security and Privacy (ICISSP), Portugal, January 2018

---

## 3. Usage & Reproducibility

### Viewing Analysis Reports (Using Cached Results)
The experimental results are already stored in the `experiment_results_cache` directory. To generate the final analysis reports (e.g., Average F1-Scores, Statistical Tests) without re-running the training process:

1. Open `config.yaml`.
2. Update the following parameters to match the target experiment:
   - `dataset`
   - `seed_sizes`
   - `minority_class_name`
   - `sampling_map`
   *(Refer to `data_description` for the appropriate values regarding data counts and class names.)*
3. Run the script:
   ```bash
   python augmentation.py
   ```

### Running New Experiments
If you wish to run the experiments from scratch:

1. **Delete** the relevant files (or the entire folder) in `experiment_results_cache`.
2. Run `augmentation.py` as described above.

**⚠️ Note on Reproducibility:**
While we strive for reproducibility, please note that **TabM** results may not perfectly match the values reported in the paper due to the inherent non-determinism of the model's training process on different environments.